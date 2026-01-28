#!/bin/bash
#/data1/clinic_rag/Q-DiT/timestep_permutation_matrix_128_100_compressed.pth
            # --sample_fid \
            # --use_gptq \
            #--permutation_matrix_path /data1/clinic_rag/Q-DiT/timestep_permutation_matrix_512_128_50.pth \
QUANT_FLAGS="--wbits 4 --abits 4 \
            --act_group_size 128 --weight_group_size 1152 \
            --quant_method max \
            --calib_data_path /data1/clinic_rag/Q-DiT/cali_data/cali_data_512_50.pth \
            --act_quant_type timestep_permutation \
            --minmax_path /data1/clinic_rag/Q-DiT/minmax_timestep_50.pth \
            --results-dir ../results_512 \
            --permutation_matrix_path /data1/clinic_rag/Q-DiT/timestep_permutation_matrix_512_128_50.pth \
            --token_group_size 0 \
            --use_diagonal_block_matrix \
            --variance_path /data1/clinic_rag/Q-DiT/channel_top1pct_analysis/top1.0pct_channels_timestep_stats_DiT-XL-2_512_50.pth
            --use_gptq \
            --not_compress
            "

SAMPLE_FLAGS="--batch-size 8 --num-fid-samples 100 --num-sampling-steps 50 --cfg-scale 4.0 --image-size 512 --seed 0"

export PYTHONUNBUFFERED=1

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=6 python -u quant_main.py $QUANT_FLAGS $SAMPLE_FLAGS