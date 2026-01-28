import torch
import os

def compress_permutation_matrix(input_path, output_path):
    """
    Loads a permutation matrix file, compresses every 5 adjacent timesteps into one,
    and saves the result to a new file.

    Args:
        input_path (str): Path to the input .pth file.
        output_path (str): Path to save the compressed .pth file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    data = torch.load(input_path)
    compressed_data = {}

    print("Compressing permutation matrices...")

    for key, value in data.items():
        compressed_value = {}
        if 'permutation_indices' in value:
            perm_indices = value['permutation_indices']
            # Select every 5th timestep and make it contiguous
            compressed_perm_indices = perm_indices[::5].contiguous().to(torch.int16)
            compressed_value['permutation_indices'] = compressed_perm_indices
            print(f"  - Compressing 'permutation_indices' for '{key}': {perm_indices.shape} -> {compressed_perm_indices.shape}, dtype -> {compressed_perm_indices.dtype}")

        if 'inverse_permutation_indices' in value:
            inv_perm_indices = value['inverse_permutation_indices']
            # Select every 5th timestep and make it contiguous
            compressed_inv_perm_indices = inv_perm_indices[::5].contiguous().to(torch.int16)
            compressed_value['inverse_permutation_indices'] = compressed_inv_perm_indices
            print(f"  - Compressing 'inverse_permutation_indices' for '{key}': {inv_perm_indices.shape} -> {compressed_inv_perm_indices.shape}, dtype -> {compressed_inv_perm_indices.dtype}")
        
        if compressed_value:
            compressed_data[key] = compressed_value

    torch.save(compressed_data, output_path)
    print(f"\nCompressed data saved to {output_path}")

def main():
    input_file = '/data1/clinic_rag/Q-DiT/timestep_permutation_matrix_128_50.pth'
    output_file = '/data1/clinic_rag/Q-DiT/timestep_permutation_matrix_128_50_compressed.pth'
    compress_permutation_matrix(input_file, output_file)

if __name__ == "__main__":
    main()
