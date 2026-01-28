#!/bin/bash
#/data1/clinic_rag/Q-DiT/timestep_permutation_matrix_128_100_compressed.pth

QUANT_FLAGS="--wbits 4 --abits 4 \
            --act_group_size 128 --weight_group_size 1152 \
            --quant_method max \
            --calib_data_path /data1/clinic_rag/Q-DiT/cali_data/cali_data_256.pth \
            --act_quant_type timestep_permutation \
            --minmax_path /data1/clinic_rag/Q-DiT/minmax_timestep_50.pth \
            --results-dir ../results_256 \
            --sample_fid \
            --permutation_matrix_path /data1/clinic_rag/Q-DiT/timestep_permutation_matrix_g128.pth \
            --variance_path /data1/clinic_rag/Q-DiT/channel_top1pct_analysis/top1.0pct_channels_timestep_stats_DiT-XL-2_256_50.pth \
            --use_diagonal_block_matrix \
            --use_gptq \
            --not_compress \
            --fix_group_num
            "

SAMPLE_FLAGS="--batch-size 25 --num-fid-samples 10000 --num-sampling-steps 50 --cfg-scale 1.5 --image-size 256 --seed 0"

export PYTHONUNBUFFERED=1

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0  python -u quant_main.py $QUANT_FLAGS $SAMPLE_FLAGS