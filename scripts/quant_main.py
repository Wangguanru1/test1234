"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import logging
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
from utils.logger_setup import create_logger
from glob import glob
from copy import deepcopy

from qdit.quant import *
from qdit.outlier import *
from qdit.datautils import *
from collections import defaultdict
from qdit.modelutils import quantize_model, quantize_model_gptq,  add_act_quant_wrapper,add_permutation_matrix,add_timestep_permutation_matrix
from flops import count_ops_and_params


from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import re

def rename_module_names(module_name):
    """
    修改模块名称的函数，用于将量化相关的模块名称进行标准化
    
    Args:
        module_name (str): 原始模块名称
        
    Returns:
        str: 修改后的模块名称
        
    Examples:
        'blocks.0.attn.input_quant' -> 'blocks.0.attn.qkv'
        'blocks.5.attn.act_quant' -> 'blocks.5.attn.proj'
        'blocks.10.mlp.input_quant' -> 'blocks.10.mlp.fc1'
        'blocks.3.mlp.act_quant' -> 'blocks.3.mlp.fc2'
    """
    # 使用正则表达式匹配并替换
    # 匹配 blocks.数字.attn.input_quant 模式
    module_name = re.sub(r'(blocks\.\d+\.attn\.)input_quant', r'\1qkv', module_name)
    
    # 匹配 blocks.数字.attn.act_quant 模式
    module_name = re.sub(r'(blocks\.\d+\.attn\.)act_quant', r'\1proj', module_name)
    
    # 匹配 blocks.数字.mlp.input_quant 模式
    module_name = re.sub(r'(blocks\.\d+\.mlp\.)input_quant', r'\1fc1', module_name)
    
    # 匹配 blocks.数字.mlp.act_quant 模式
    module_name = re.sub(r'(blocks\.\d+\.mlp\.)act_quant', r'\1fc2', module_name)
    
    return module_name

def validate_model(args, model, diffusion, vae):
    seed_everything(args.seed)
    device = next(model.parameters()).device
    using_cfg = args.cfg_scale > 1.0
    # Labels to condition the model with (feel free to change):

    class_labels = [266, 360, 366, 974, 88, 979, 488, 888]
    if args.image_size == 512:
        class_labels = [388,291,417,972, 250,980,33,812]
    #class_labels = [547, 586, 812, 966]
    #class_labels = [547]
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, model.input_size, model.input_size, device=device)
    y = torch.tensor(class_labels, device=device)
    # Setup classifier-free guidance:
    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        # sample_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        # sample_fn = model.forward
    #z = z.half()
    t = torch.randint(0, 1000, (z.shape[0],), device=device, dtype=torch.int)
    flops, params = count_ops_and_params(model, (z, t, y))
    logging.info(f"FLOPs: {flops / 1e9:.2f} G, Parameters: {params / 1e6:.2f} M")
    
    
    samples = diffusion.p_sample_loop(
        model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    # Save and display images:
    save_image(samples, f'{args.experiment_dir}/sample.png', nrow=8, normalize=True, value_range=(-1, 1))
    print("Finish validating samples!")
    
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        #sample_pil = Image.open(f"{sample_dir}/{i}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def sample_fid(args, model, diffusion, vae):
    # Create folder to save samples:
    seed_everything(args.seed)
    device = next(model.parameters()).device
    using_cfg = args.cfg_scale > 1.0
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.seed}"
    sample_folder_dir = f"{args.experiment_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    folder_path = Path(sample_folder_dir)
    existing_samples = sorted(
        int(p.stem) for p in folder_path.glob("*.png") if p.stem.isdigit()
    )
    total = len(existing_samples)
    if existing_samples and existing_samples[-1] != total - 1:
        logging.warning(
            "Existing sample files have non-contiguous indices; resume may overwrite or fail."
        )

    if total >= args.num_fid_samples:
        print(
            f"Detected {total} existing samples, which meets or exceeds the requested {args.num_fid_samples}."
        )
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
        return

    if total > 0:
        print(f"Resuming from sample index {total} (existing samples detected).")

    pbar = tqdm(total=args.num_fid_samples, initial=total, desc="Sampling FID images", unit="img")

    while total < args.num_fid_samples:
        current_batch = min(args.batch_size, args.num_fid_samples - total)

        z = torch.randn(
            current_batch, model.in_channels, model.input_size, model.input_size, device=device
        )
        y = torch.randint(0, args.num_classes, (current_batch,), device=device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * current_batch, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)

        z = z.half()
        with autocast():
            samples = diffusion.p_sample_loop(
                model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples / 0.18215).sample
        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        generated = samples.shape[0]
        for i, sample in enumerate(samples):
            index = total + i
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += generated
        pbar.update(generated)

    pbar.close()

    create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
    print("Done.")


def main():
    args = create_argparser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    quant_method = "qdit"
    quant_string_name = f"{quant_method}_w{args.wbits}a{args.abits}"

    if args.experiment_dir:
        experiment_dir = os.path.abspath(args.experiment_dir)
        os.makedirs(experiment_dir, exist_ok=True)
    else:
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{quant_string_name}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)

    args.experiment_dir = experiment_dir
    create_logger(experiment_dir)
    logging.info(f"Experiment directory ready at {experiment_dir}")
    logging.info(f"""wbits: {args.wbits}, abits: {args.abits}, w_sym: {args.w_sym}, a_sym: {args.a_sym},
                 weight_group_size: {args.weight_group_size}, act_group_size: {args.act_group_size},
                 quant_method: {args.quant_method}, use_gptq: {args.use_gptq}, static: {args.static},
                 image_size: {args.image_size}, cfg_scale: {args.cfg_scale},sample_steps: {args.num_sampling_steps},token_group_size:{args.token_group_size},
                 act_quant_type: {args.act_quant_type}, use_diagonal_block_matrix: {args.use_diagonal_block_matrix},fix_group_num: {args.fix_group_num}""")
    
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
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    args.weight_group_size = eval(args.weight_group_size)
    args.act_group_size = eval(args.act_group_size)
    args.weight_group_size = [args.weight_group_size] * len(model.blocks) if isinstance(args.weight_group_size, int) else args.weight_group_size
    args.act_group_size = [args.act_group_size] * len(model.blocks) if isinstance(args.act_group_size, int) else args.act_group_size
    
    print("Inserting activations quantizers ...")
    if args.static:
        dataloader = get_loader(args.calib_data_path, nsamples=1024, batch_size=16)
        print("Getting activation stats...")
        scales = get_act_scales(
            model, diffusion, dataloader, device, args
        )
    else: 
        scales = defaultdict(lambda: None)
    model = add_act_quant_wrapper(model, device=device, args=args, scales=scales)
    
    #置换变换量化
    if args.act_quant_type == 'permutation':
        if args.permutation_matrix_path != None and args.permutation_matrix_path != '':
            permutation_data = torch.load(args.permutation_matrix_path)
            for name, module in model.named_modules():
                if isinstance(module, PermutationQuantizer) and name in permutation_data:
                    logging.info(f'{name} permutation indices loaded')
                    module.register_buffer('permutation_indices', permutation_data[name]['permutation_indices'])
                    module.register_buffer('inverse_permutation_indices', permutation_data[name]['inverse_permutation_indices'])
        else:
            add_permutation_matrix(model=model,device=device,args=args)
    #时间步置换变换
    if args.act_quant_type == 'timestep_permutation':
        if args.permutation_matrix_path != None and args.permutation_matrix_path != '':
            permutation_data = torch.load(args.permutation_matrix_path)
            top1pct_data =  torch.load(args.variance_path) 
            summary = top1pct_data['summary']
            

            # 为每个层加载对应的置换矩阵
            for name, module in model.named_modules():
                if isinstance(module, TimestepPermutationQuantizer):

                    module.register_buffer('permutation_indices', permutation_data[name]['permutation_indices'])
                    module.register_buffer('inverse_permutation_indices', permutation_data[name]['inverse_permutation_indices'])

                    avg = summary[rename_module_names(name)]['avg']
                    if not rename_module_names(name).endswith('fc2'):
                        num_rotate_groups = avg // module.group_size
                        module.num_rotate_groups = int(num_rotate_groups) + 1
                        logging.info(f'{name} permutation indices loaded, num_rotate_groups: {module.num_rotate_groups}')
                    
                    if name == 'blocks.27.mlp.act_quant':
                        #module.num_rotate_groups = 36
                        module.quant_mode = False
                    if name == args.save_act:
                        module.save_act = True
        else:
            add_timestep_permutation_matrix(model=model,device=device,args=args)

    print("Quantizing ...")
    if args.use_gptq:
        dataloader = get_loader(args.calib_data_path, nsamples=256)
        model = quantize_model_gptq(model, device=device, args=args, dataloader=dataloader)
    else:
        model = quantize_model(model, device=device, args=args)


    print("Finish quant!")
    logging.info(model)

    if args.act_quant_type == 'permutation' and not os.path.exists(f'../permutation_matrix_{args.act_group_size[0]}_{args.num_sampling_steps}.pth'):
        perm_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, PermutationQuantizer):
                if module.permutation_indices is not None:
                    perm_dict[name] = {
                        'permutation_indices': module.permutation_indices.clone().cpu(),
                        'inverse_permutation_indices': module.inverse_permutation_indices.clone().cpu()
                    }
        torch.save(perm_dict, f'../permutation_matrix_{args.image_size}_{args.act_group_size[0]}_{args.num_sampling_steps}.pth')

    if args.act_quant_type == 'timestep_permutation' and not os.path.exists(f'../timestep_permutation_matrix_{args.image_size}_{args.act_group_size[0]}_{args.num_sampling_steps}.pth'):
        perm_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, TimestepPermutationQuantizer):
                if module.permutation_indices is not None:
                    perm_dict[name] = {
                        'permutation_indices': module.permutation_indices.clone().cpu(),
                        'inverse_permutation_indices': module.inverse_permutation_indices.clone().cpu()
                    }
        torch.save(perm_dict, f'../timestep_permutation_matrix_{args.image_size}_{args.act_group_size[0]}_{args.num_sampling_steps}.pth')
        logging.info(f'timestep_permutation_matrix saved to ../timestep_permutation_matrix_{args.image_size}_{args.act_group_size[0]}_{args.num_sampling_steps}.pth')

    # generate some sample images
    model.to(device)

    
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)
    validate_model(args, model, diffusion, vae)
    if args.sample_fid:
        sample_fid(args, model, diffusion, vae)

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
        '--calib_data_path', type=str, default='../cali_data/cali_data_256.pth',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--num_groups', type=int, default=1,
        help='激活值量化分组数'
    )
    parser.add_argument(
        '--act_quant_type', type=str, choices=['min-max','learnable','permutation','timestep_permutation','rotation','partial','progressive'],default='min-max',
        help='量化方式'
    )
    parser.add_argument(
        '--minmax_path', type=str,default='../act_minmax_stats.pth',
        help='量化方式'
    )
    parser.add_argument(
        '--permutation_matrix_path', type=str,default='',
        help='置换矩阵路径'
    )
    parser.add_argument(
        '--variance_path', type=str,default='/data1/clinic_rag/Q-DiT/channel_top1pct_analysis/top1.0pct_channels_timestep_stats_DiT-XL-2_256_100.pth',
        help='方差路径'
    )
    parser.add_argument(
        '--use_diagonal_block_matrix', action='store_true',
        help='是否使用对角块Hadamard矩阵'
    )
    parser.add_argument(
        '--not_compress', action='store_true',
        help='不使用压缩的置换矩阵'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Whether to save the quantized model.'
    )
    parser.add_argument(
        '--sample_fid', action='store_true',
        help='Whether to sample fid.'
    )
    parser.add_argument(
        '--fix_group_num', action='store_true',
        help='Whether to fix group number.'
    )
    parser.add_argument('--precision_timestep_range', nargs=2, type=int, default=[0, -1], 
                        help='全精度区间 [start end]')
    parser.add_argument('--save_act', type=str, default='', 
                        help='保存激活值的模块')
    parser.add_argument('--token_group_size', type=int, default=16, 
                        help='Token分组大小')

    # Inherited from DiT
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--results-dir", type=str, default="../results4")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Reuse or specify an explicit experiment directory (useful for resuming sampling).")
    parser.add_argument(
        "--save_ckpt", action="store_true", help="choose to save the qnn checkpoint"
    )
    # sample_ddp.py
    parser.add_argument("--tf32", action="store_true",
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    return parser


if __name__ == "__main__": 
    main()
