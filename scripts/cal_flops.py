
import torch
import argparse
import sys
import os

# Add the parent directory of 'scripts' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import DiT_models
from flops import count_ops_and_params

def main(args):
    """
    Calculates and prints the parameters and FLOPs of a DiT model.
    """
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    
    model.eval()

    # Create dummy inputs
    x = torch.randn(args.batch_size, 4, latent_size, latent_size, device=device)
    t = torch.randint(0, 1000, (args.batch_size,), device=device)
    y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
    example_inputs = (x, t, y)

    # Calculate FLOPs and parameters
    flops, params = count_ops_and_params(model, example_inputs, layer_wise=False)

    print(f"\nModel: {args.model}")
    print(f"Image Size: {args.image_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Total Parameters: {params / 1e6:.2f} M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(args)
