import torch
import torch.nn as nn
import functools
import math
from tqdm import tqdm
from qdit.qBlock import QLinearLayer
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from custom_dataloader import SyntheticLatentDataset
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import logging
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm
import math

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from qdit.learnable_quant import update_LSQplus_activation_Scalebeta
from utils.download import find_model
from models.models import DiT_models
from utils.logger_setup import create_logger
from glob import glob
from copy import deepcopy
from qdit.quant import *
from qdit.outlier import *
from qdit.datautils import *
from collections import defaultdict
from qdit.modelutils import quantize_model, quantize_model_gptq,  add_act_quant_wrapper
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

@torch.no_grad()
def get_linear_act_min_max_by_timestep(model, dataloader, device, save_path=None, sample_steps=50):
    # layer_name -> timestep (int) -> channel -> tensor
    act_max_scales = defaultdict(dict)
    act_min_scales = defaultdict(dict)

    current_timesteps = None  # 与 hook 通信

    def stat_input_hook(module, x, y, name):
        nonlocal current_timesteps
        if isinstance(x, tuple):
            x = x[0]
        assert isinstance(x, torch.Tensor)

        batch_size = x.shape[0]
        for i in range(batch_size):
            t = int(current_timesteps[i].item())
            
            t = t // (1000 // sample_steps)  # 将timestep映射到sample_steps范围内
            x_i = x[i].detach()  # 保持原始形状，不要展平
            
            # 获取最后一个维度作为通道维度
            if x_i.ndim > 1:
                # 对所有维度除了最后一个维度进行折叠，保留通道维度
                # 例如 [H, W, C] -> [H*W, C]
                original_shape = x_i.shape
                channel_dim = original_shape[-1]
                x_i = x_i.reshape(-1, channel_dim)
                
                # 对每个通道计算最大最小值
                # 在第一个维度上取最大最小值 [H*W, C] -> [C]
                min_vals = x_i.min(dim=0).values  # shape: [C]
                max_vals = x_i.max(dim=0).values  # shape: [C]
            else:
                # 如果只有一个维度，则就是通道本身
                min_vals = x_i.min()
                max_vals = x_i.max()

            if t in act_max_scales[name]:
                act_max_scales[name][t] = torch.maximum(act_max_scales[name][t], max_vals)
                act_min_scales[name][t] = torch.minimum(act_min_scales[name][t], min_vals)
            else:
                act_max_scales[name][t] = max_vals
                act_min_scales[name][t] = min_vals

    # 注册所有 nn.Linear 的 forward hook
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

    # 遍历 dataloader
    for calib_x, calib_t, calib_y in tqdm(dataloader, desc="Collecting min/max per timestep"):
        current_timesteps = calib_t.to("cpu")
        model(calib_x.to(device), calib_t.to(device), calib_y.to(device))

    # 移除 hook
    for h in hooks:
        h.remove()

    # 组装结果
    minmax_stats = {
        'max_values': act_max_scales,
        'min_values': act_min_scales
    }

    # 可选保存为 .pth
    if save_path:
        torch.save(minmax_stats, save_path)

    return minmax_stats

def main():
    args = create_argparser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    quant_method = "qdit"
    quant_string_name = f"{quant_method}_w{args.wbits}a{args.abits}"
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{quant_string_name}"  # Create an experiment folder
    args.experiment_dir = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    create_logger(experiment_dir)
    logging.info(f"Experiment directory created at {experiment_dir}")
    logging.info(f"""wbits: {args.wbits}, abits: {args.abits}, w_sym: {args.w_sym}, a_sym: {args.a_sym},
                 weight_group_size: {args.weight_group_size}, act_group_size: {args.act_group_size},
                 quant_method: {args.quant_method}, use_gptq: {args.use_gptq}, static: {args.static},
                 image_size: {args.image_size}, cfg_scale: {args.cfg_scale}""")
    
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    dataloader = get_loader(args.calib_data_path, nsamples=12800, batch_size=32)
    get_linear_act_min_max_by_timestep(model=model,dataloader=dataloader,device=device,save_path=f'minmax_timestep_{args.num_sampling_steps}.pth',sample_steps=args.num_sampling_steps)
   

def create_argparser():
    parser = argparse.ArgumentParser()

    # quantization parameters
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 5, 6, 8, 16],
        help='#bits to use for quantizing weight; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 5, 6, 8, 16],
        help='#bits to use for quantizing activation; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--exponential', action='store_true',
        help='Whether to use exponent-only for weight quantization.'
    )
    parser.add_argument(
        '--quantize_bmm_input', action='store_true',
        help='Whether to perform bmm input activation quantization. Default is not.'
    )
    parser.add_argument(
        '--a_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--w_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Whether to perform static quantization (For activtions). Default is dynamic. (Deprecated in Atom)'
    )
    parser.add_argument(
        '--weight_group_size', type=str,
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=str,
        help='Group size when quantizing activations. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity (Deprecated in Atom).'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--use_gptq', action='store_true',
        help='Whether to use GPTQ for weight quantization.'
    )
    parser.add_argument(
        '--quant_method', type=str, default='max', choices=['max', 'mse'],
        help='The method to quantize weight.'
    )
    parser.add_argument(
        '--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--w_clip_ratio', type=float, default=1.0,
        help='Clip ratio for weight quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--save_dir', type=str, default='../saved',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--quant_type', type=str, default='int', choices=['int', 'fp'],
        help='Determine the mapped data format by quant_type + n_bits. e.g. int8, fp4.'
    )
    parser.add_argument(
        '--calib_data_path', type=str, default='cali_data/cali_data_512_50.pth',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--num_groups', type=int, default=1,
        help='激活值量化分组数'
    )
    parser.add_argument(
        '--act_quant_type', type=str, default='min-max',
        help='量化方式'
    )

    # Inherited from DiT
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='pretrained_models/DiT-XL-2-512x512.pt',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument(
        "--save_ckpt", action="store_true", help="choose to save the qnn checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    # sample_ddp.py
    parser.add_argument("--tf32", action="store_true",
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    return parser


if __name__ == "__main__": 
    main()



