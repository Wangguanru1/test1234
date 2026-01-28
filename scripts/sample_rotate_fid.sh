#!/bin/bash

QUANT_FLAGS="--wbits 4 --abits 4 \
            --act_group_size 128 --weight_group_size 128 \
            --quant_method max \
            --calib_data_path /data1/clinic_rag/Q-DiT/cali_data/cali_data_256_100.pth \
            --act_quant_type rotation \
            --minmax_path /data1/clinic_rag/Q-DiT/minmax_timestep_100.pth \
            --permutation_matrix_path /data1/clinic_rag/Q-DiT/timestep_permutation_matrix_128_100_compressed.pth \
            --results-dir ../results3 \
            --use_gptq \
            --a_sym \
            --sample_fid \
            --use_diagonal_block_matrix
            "

SAMPLE_FLAGS="--batch-size 16 --num-fid-samples 10000 --num-sampling-steps 100 --cfg-scale 1.5 --image-size 256 --seed 0"

export PYTHONUNBUFFERED=1

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=2 python -u quant_main.py $QUANT_FLAGS $SAMPLE_FLAGS