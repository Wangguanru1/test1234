# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT for specific timesteps and classes.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
import argparse
import math
from tqdm import tqdm
import numpy as np
import os

cali_data_dir = "../cali_data"

def sample_cali_data_per_batch(input_list_per_batch, batch_size=32, specific_timesteps=None):
    """
    x: (N, C, H, W), t: (N,), y: (N,).
    """
    cali_data = []
    
    for samples_t in input_list_per_batch:
        timestep_val = samples_t[1][0].item()
        if specific_timesteps is not None and timestep_val not in specific_timesteps:
            continue

        for idx in range(len(samples_t[0])):
            cali_data.append([samples_t[0][idx], samples_t[1][idx], samples_t[2][idx]])
            
    if not cali_data:
        return None

    return [torch.stack([sample[i] for sample in cali_data]) for i in range(len(cali_data[0]))]

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        record_inputs=True
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    using_cfg = args.cfg_scale > 1.0

    
    iterations = int(math.ceil(args.num_cali_data / args.batch_size))
    pbar = range(iterations)
    pbar = tqdm(pbar)
    cali_data = []
    
    specific_classes = args.specific_classes
    if specific_classes:
        print(f"Using specific classes: {specific_classes}")
    
    for batch_idx in pbar:
        # Sample inputs:
        z = torch.randn(args.batch_size, model.in_channels, latent_size, latent_size, device=device)
        
        if specific_classes:
            y = torch.tensor(np.random.choice(specific_classes, args.batch_size), device=device)
        else:
            y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * args.batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)
        
        samples = diffusion.ddim_sample_loop(
            model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        # samples = vae.decode(samples / 0.18215).sample
        # samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        input_list = model.get_input_list()

        cali_data_per_batch = sample_cali_data_per_batch(input_list, batch_size=args.batch_size, specific_timesteps=args.specific_timesteps)
        if cali_data_per_batch:
            cali_data.append(cali_data_per_batch)
        model.reset_input_list()
    
    if not cali_data:
        print("No calibration data collected. Check your specific_timesteps.")
        return

    cali_data = [torch.cat([batch[i] for batch in cali_data]) for i in range(len(cali_data[0]))]
    
    num_collected = len(cali_data[0])
    print(f"Collected {num_collected} samples.")

    if not os.path.exists(cali_data_dir):
        os.mkdir(cali_data_dir)

    filename = f"cali_data_{args.image_size}_{args.num_sampling_steps}"
    if args.specific_timesteps:
        filename += f"_ts{''.join(map(str, args.specific_timesteps))}"
    if args.specific_classes:
        filename += f"_cls{''.join(map(str, args.specific_classes))}"
    filename += ".pth"
    
    torch.save(cali_data, os.path.join(cali_data_dir, filename))
    print(f"Saved calibration data to {os.path.join(cali_data_dir, filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-cali-data", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--specific-timesteps", type=int, nargs='+', default=None,
                        help="A list of specific timesteps to collect calibration data for.")
    parser.add_argument("--specific-classes", type=int, nargs='+', default=None,
                        help="A list of specific class labels to generate images for.")
    args = parser.parse_args()
    main(args)
