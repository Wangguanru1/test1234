#!/usr/bin/env python3
"""Analyze symmetric KL divergence between timesteps for a given DiT layer."""
import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for headless environments.
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.models import DiT_models
from utils.download import find_model
from diffusion import create_diffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute KL divergence matrix across timesteps for one or more layers.")
    parser.add_argument("--layer-name", type=str, default=None, help="Target layer name, e.g. 'blocks.0.attn.input_quant'")
    parser.add_argument(
        "--layer-suffix",
        type=str,
        default='mlp.fc1',
        help="Match every layer whose qualified name ends with this suffix (e.g. 'attn.qkv' or 'mlp.fc1')",
    )
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2", help="DiT model config key")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Image size used when exporting the checkpoint")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of class labels")
    parser.add_argument("--ckpt", type=str, default=None, help="Override path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used during diffusion sampling")
    parser.add_argument("--samples-per-timestep", type=int, default=None, help="Subsample activations per timestep to this many values")
    parser.add_argument("--bins", type=int, default=256, help="Number of histogram bins when estimating distributions")
    parser.add_argument("--kl-eps", type=float, default=1e-6, help="Additive epsilon to keep histogram probabilities positive")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    parser.add_argument("--output-dir", type=str, default="kl_analysis", help="Directory to store outputs")
    parser.add_argument("--figure-name", type=str, default=None, help="Optional custom name for the heatmap figure")
    parser.add_argument("--matrix-name", type=str, default=None, help="Optional custom name for the saved KL matrix tensor")
    parser.add_argument("--max-ticks", type=int, default=12, help="Maximum number of axis ticks to display")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--num-sampling-steps", type=int, default=50, help="Number of diffusion steps (e.g., 50 for DDPM-50)")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="Classifier-free guidance scale when sampling activations")
    parser.add_argument("--diffusion-iters", type=int, default=1, help="Number of diffusion batches to run during sampling")
    return parser.parse_args()

def collect_activations_with_diffusion(
    model: torch.nn.Module,
    diffusion,
    layer_names: List[str],
    device: torch.device,
    batch_size: int,
    num_batches: int,
    num_classes: int,
    latent_size: int,
    cfg_scale: float,
    num_steps: int,
) -> Dict[str, Dict[int, List[torch.Tensor]]]:
    modules = dict(model.named_modules())
    missing_layers = [name for name in layer_names if name not in modules]
    if missing_layers:
        available = list(modules.keys())
        raise ValueError(
            f"Layers {missing_layers} not found. Available names include: {available[:10]} ..."
        )

    activations: Dict[str, Dict[int, List[torch.Tensor]]] = {
        name: defaultdict(list) for name in layer_names
    }
    current_steps: List[int] = []

    def make_capture_hook(target_name: str):
        def capture_hook(module, inputs, _output):
            tensor = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            if not current_steps or tensor is None:
                return
            if not torch.is_tensor(tensor):
                return
            tensor_cpu = tensor.detach().cpu()
            batch_dim = tensor_cpu.shape[0]
            if batch_dim != len(current_steps):
                # Skip mismatched records rather than crashing, avoids stale timestep state.
                return
            for idx, step_idx in enumerate(current_steps):
                activations[target_name][step_idx].append(tensor_cpu[idx])

        return capture_hook

    def update_current_steps(t_tensor: Optional[torch.Tensor], batch_dim: int) -> None:
        current_steps.clear()
        if torch.is_tensor(t_tensor):
            flat = t_tensor.detach().view(-1)
            if flat.numel() == batch_dim and batch_dim > 0:
                gap = 1000 // num_steps
                current_steps.extend(int(v // gap) for v in flat.tolist())
                return
        if batch_dim <= 0:
            return
        

    def _wrap_forward(fn):
        def _wrapped(x, t, *args, **kwargs):
            batch_dim = int(x.shape[0]) if torch.is_tensor(x) else 0
            update_current_steps(t if torch.is_tensor(t) else None, batch_dim)
            return fn(x, t, *args, **kwargs)
        return _wrapped

    handles = [modules[name].register_forward_hook(make_capture_hook(name)) for name in layer_names]

    original_forward = model.forward
    model.forward = _wrap_forward(model.forward)
    original_forward_with_cfg = getattr(model, "forward_with_cfg", None)

    using_cfg = cfg_scale > 1.0

    try:
        with torch.no_grad():
            for _ in range(num_batches):
                z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
                y = torch.randint(0, num_classes, (batch_size,), device=device)

                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.full((batch_size,), num_classes, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=cfg_scale)
                else:
                    model_kwargs = dict(y=y)

                diffusion.p_sample_loop(
                    model,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                )
    finally:
        current_steps.clear()
        model.forward = original_forward
        if original_forward_with_cfg is not None:
            model.forward_with_cfg = original_forward_with_cfg

    for handle in handles:
        handle.remove()
    return activations


def prepare_histograms(
    activations: Dict[int, List[torch.Tensor]],
    bins: int,
    eps: float,
    samples_per_timestep: Optional[int],
    seed: int,
) -> Tuple[List[int], Dict[int, torch.Tensor], Tuple[float, float]]:
    if not activations:
        raise RuntimeError("No activations collected; check layer name and calibration data consistency.")

    rng = torch.Generator().manual_seed(seed)
    flattened: Dict[int, torch.Tensor] = {}
    global_min: Optional[float] = None
    global_max: Optional[float] = None

    for timestep, tensors in activations.items():
        if not tensors:
            continue
        values = torch.cat([t.reshape(-1) for t in tensors], dim=0).to(torch.float64)
        if samples_per_timestep is not None and values.numel() > samples_per_timestep:
            idx = torch.randperm(values.numel(), generator=rng)[:samples_per_timestep]
            values = values[idx]
        flattened[timestep] = values
        t_min = values.min().item()
        t_max = values.max().item()
        global_min = t_min if global_min is None else min(global_min, t_min)
        global_max = t_max if global_max is None else max(global_max, t_max)

    if not flattened:
        raise RuntimeError("No activations remained after preprocessing.")

    assert global_min is not None and global_max is not None
    if global_min == global_max:
        delta = abs(global_min) if global_min != 0 else 1.0
        global_min -= 0.5 * delta
        global_max += 0.5 * delta

    histograms: Dict[int, torch.Tensor] = {}
    for timestep, values in flattened.items():
        hist = torch.histogram(values, bins=bins, range=(global_min, global_max))[0].double()
        hist += eps
        total = hist.sum()
        if total <= 0:
            raise RuntimeError(f"Histogram for timestep {timestep} has zero mass.")
        hist /= total
        histograms[timestep] = hist

    timesteps = sorted(histograms.keys())
    return timesteps, histograms, (global_min, global_max)


def compute_symmetric_kl(timesteps: Iterable[int], histograms: Dict[int, torch.Tensor]) -> torch.Tensor:
    timesteps = list(timesteps)
    n = len(timesteps)
    matrix = torch.zeros((n, n), dtype=torch.float64)

    for i in range(n):
        pi = histograms[timesteps[i]]
        for j in range(i + 1, n):
            pj = histograms[timesteps[j]]
            kl_ij = torch.sum(pi * torch.log(pi / pj))
            kl_ji = torch.sum(pj * torch.log(pj / pi))
            sym_kl = 0.5 * (kl_ij + kl_ji)
            matrix[i, j] = sym_kl
            matrix[j, i] = sym_kl

    return matrix


def plot_matrix(
    timesteps: List[int],
    matrix: torch.Tensor,
    layer_name: str,
    output_path: str,
    max_ticks: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.numpy(), cmap="viridis")
    ax.set_title(f"{layer_name}\nSymmetric KL Divergence Matrix")
    ax.set_xlabel("Timestep Index")
    ax.set_ylabel("Timestep Index")

    tick_count = min(max_ticks, len(timesteps))
    tick_positions = np.linspace(0, len(timesteps) - 1, tick_count, dtype=int)
    tick_labels = [str(timesteps[i]) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Symmetric KL Divergence")
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device_type = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    if args.layer_name is None and args.layer_suffix is None:
        args.layer_name = "blocks.0.mlp.fc1"

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion_schedule = create_diffusion(str(args.num_sampling_steps))
    num_steps = diffusion_schedule.num_timesteps
    if num_steps <= 0:
        raise ValueError("Sampler returned no timesteps; check num-sampling-steps argument.")

    modules = dict(model.named_modules())
    target_layers: List[str] = []
    if args.layer_suffix:
        suffix_matches = sorted(name for name in modules if name.endswith(args.layer_suffix))
        if not suffix_matches:
            raise ValueError(
                f"No layers end with suffix '{args.layer_suffix}'."
            )
        target_layers.extend(suffix_matches)

    if args.layer_name:
        if args.layer_name not in modules:
            available = list(modules.keys())
            raise ValueError(
                f"Layer '{args.layer_name}' not found. Available names include: {available[:10]} ..."
            )
        if args.layer_name not in target_layers:
            target_layers.append(args.layer_name)

    if not target_layers:
        raise ValueError("No target layers selected; provide --layer-name or --layer-suffix.")

    activations_by_layer = collect_activations_with_diffusion(
        model=model,
        diffusion=diffusion_schedule,
        layer_names=target_layers,
        device=device,
        batch_size=args.batch_size,
        num_batches=args.diffusion_iters,
        num_classes=args.num_classes,
        latent_size=latent_size,
        cfg_scale=args.cfg_scale,
        num_steps=num_steps,
    )

    multi_layer = len(target_layers) > 1

    for layer_name in target_layers:
        layer_activations: Dict[int, List[torch.Tensor]] = activations_by_layer.get(layer_name, defaultdict(list))
        missing_timesteps = [
            step for step in range(num_steps) if step not in layer_activations or not layer_activations[step]
        ]
        if missing_timesteps:
            preview = ", ".join(str(t) for t in missing_timesteps[:10])
            if len(missing_timesteps) > 10:
                preview += ", ..."
            print(
                f"Warning: layer '{layer_name}' missing activations for {len(missing_timesteps)} diffusion steps: {preview}"
            )

        try:
            timesteps, histograms, value_range = prepare_histograms(
                layer_activations,
                bins=args.bins,
                eps=args.kl_eps,
                samples_per_timestep=args.samples_per_timestep,
                seed=args.seed,
            )
        except RuntimeError as exc:
            print(f"Skipping layer '{layer_name}' due to error: {exc}")
            continue

        kl_matrix = compute_symmetric_kl(timesteps, histograms)
        layer_safe = layer_name.replace(".", "_").replace("/", "_")

        if args.figure_name and not multi_layer:
            figure_name = args.figure_name
        elif args.figure_name and multi_layer:
            root, ext = os.path.splitext(args.figure_name)
            ext = ext or ".png"
            figure_name = f"{root}_{layer_safe}{ext}"
        else:
            figure_name = f"kl_matrix_{layer_safe}.png"

        figure_path = os.path.join(args.output_dir, figure_name)
        plot_matrix(timesteps, kl_matrix, layer_name, figure_path, args.max_ticks)
        print(f"Saved KL divergence heatmap for '{layer_name}' to {figure_path}")

        if args.matrix_name and not multi_layer:
            matrix_name = args.matrix_name
        elif args.matrix_name and multi_layer:
            root, ext = os.path.splitext(args.matrix_name)
            ext = ext or ".pth"
            matrix_name = f"{root}_{layer_safe}{ext}"
        else:
            matrix_name = f"kl_matrix_{layer_safe}.pth"

        matrix_path = os.path.join(args.output_dir, matrix_name)
        torch.save(
            {
                "layer": layer_name,
                "timesteps": timesteps,
                "kl_matrix": kl_matrix,
                "value_range": value_range,
                "bins": args.bins,
            },
            matrix_path,
        )
        print(f"Saved KL matrix tensor for '{layer_name}' to {matrix_path}")


if __name__ == "__main__":
    main()
