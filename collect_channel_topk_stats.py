import torch
import torch.nn as nn
import functools
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
from pathlib import Path
from utils.logger_setup import create_logger
import logging

from models.models import DiT_models
from utils.download import find_model
from qdit.datautils import get_loader

def default_factory():
    return defaultdict(dict)

@torch.no_grad()
def get_channel_topk_stats(model, dataloader, device, save_path, sample_steps=50, topk_percent=0.01):
    """
    Collects statistics on how many times each channel appears in the top-k% of absolute values for each token.

    Args:
        model (nn.Module): The model to analyze.
        dataloader (DataLoader): DataLoader for calibration data.
        device (str): The device to run the model on.
        save_path (str): Path to save the collected statistics.
        sample_steps (int): The number of discrete timesteps to sample.
        topk_percent (float): The percentage of channels to consider as top-k.
    """
    # Structure: layer_name -> timestep (int) -> channel_counts (Tensor)
    channel_topk_counts = defaultdict(default_factory)

    current_timesteps = None  # To communicate with the hook

    def stat_input_hook(module, x, y, name):
        nonlocal current_timesteps
        if isinstance(x, tuple):
            x = x[0]
        assert isinstance(x, torch.Tensor)

        batch_size = x.shape[0]
        
        for i in range(batch_size):
            t = int(current_timesteps[i].item())
            t = t // (1000 // sample_steps)  # Map timestep to the sample_steps range

            x_i = x[i].detach()  # Shape: [num_tokens, channels]

            if x_i.ndim != 2:
                # Assuming the last dimension is the channel dimension
                channel_dim = x_i.shape[-1]
                x_i = x_i.reshape(-1, channel_dim)
            
            num_tokens, channel_dim = x_i.shape
            
            # Initialize counter tensor if not present
            if t not in channel_topk_counts[name]:
                channel_topk_counts[name][t] = torch.zeros(channel_dim, dtype=torch.long, device='cpu')

            # Calculate k for top-k
            k = max(1, int(channel_dim * topk_percent))

            # Find top-k channels for each token
            abs_vals = x_i.abs()
            _, topk_indices = torch.topk(abs_vals, k, dim=1)  # Shape: [num_tokens, k]

            # Update counts
            # Flatten indices and use bincount for efficient counting
            topk_indices_flat = topk_indices.flatten()
            counts = torch.bincount(topk_indices_flat, minlength=channel_dim)
            
            channel_topk_counts[name][t] += counts.to('cpu')

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    model.to(device)
    model.eval()

    for calib_x, calib_t, calib_y in tqdm(dataloader, desc="Collecting channel top-k stats"):
        current_timesteps = calib_t.to("cpu")
        model(calib_x.to(device), calib_t.to(device), calib_y.to(device))

    for h in hooks:
        h.remove()

    # Save the statistics
    torch.save(channel_topk_counts, save_path)
    logging.info(f"Saved channel top-k statistics to {save_path}")

    return channel_topk_counts

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default='pretrained_models/DiT-XL-2-256x256.pt',
                        help="Path to a DiT checkpoint.")
    parser.add_argument("--calib-data-path", type=str, default='cali_data/cali_data_256_100.pth',
                        help="Path to calibration data.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nsamples", type=int, default=1280, help="Number of samples for calibration.")
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--topk-percent", type=float, default=0.01, help="Percentage of channels to be considered as top-k.")
    parser.add_argument("--save-path", type=str, default='channel_topk_stats.pth', help="Path to save the statistics.")
    parser.add_argument("--results-dir", type=str, default="../results_topk")
    return parser

def main():
    args = create_argparser().parse_args()
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(os.listdir(args.results_dir))
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}"
    os.makedirs(experiment_dir, exist_ok=True)
    create_logger(experiment_dir)
    
    logging.info(f"Using device: {device}")
    logging.info(f"Experiment directory: {experiment_dir}")
    logging.info(args)

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    # Load data
    dataloader = get_loader(args.calib_data_path, nsamples=args.nsamples, batch_size=args.batch_size)

    # Set save path
    save_path = os.path.join(experiment_dir, args.save_path)

    # Collect statistics
    get_channel_topk_stats(
        model=model,
        dataloader=dataloader,
        device=device,
        save_path=save_path,
        sample_steps=args.num_sampling_steps,
        topk_percent=args.topk_percent
    )

if __name__ == "__main__":
    main()
