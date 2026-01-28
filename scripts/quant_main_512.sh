#!/bin/bash
#            --use_diagonal_block_matrix
#            --save_act blocks.15.attn.input_quant \
#            --save_act blocks.2.attn.input_quant \
#            --use_diagonal_block_matrix \
#            --save_act blocks.25.mlp.input_quant \
QUANT_FLAGS="--wbits 4 --abits 4 \
            --act_group_size 128 --weight_group_size 1152 \
            --quant_method max \
            --calib_data_path /data1/clinic_rag/Q-DiT/cali_data/cali_data_512_50.pth \
            --act_quant_type timestep_permutation \
            --minmax_path /data1/clinic_rag/Q-DiT/minmax_timestep_50.pth \
            --permutation_matrix_path /data1/clinic_rag/Q-DiT/timestep_permutation_matrix_512_128_50_compressed.pth \
            --results-dir ../results3 \
            --variance_path /data1/clinic_rag/Q-DiT/channel_top1pct_analysis/top1.0pct_channels_timestep_stats_DiT-XL-2_512_50.pth \
            --use_diagonal_block_matrix \
            --use_gptq \
            "

#SAMPLE_FLAGS="--batch-size 16 --num-fid-samples 10000 --num-sampling-steps 50 --cfg-scale 1.5 --image-size 512 --seed 0"
SAMPLE_FLAGS="--batch-size 4 --num-fid-samples 10000 --num-sampling-steps 50 --cfg-scale 4.0 --image-size 512 --seed 0"

export PYTHONUNBUFFERED=1

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=6 python -u quant_main.py $QUANT_FLAGS $SAMPLE_FLAGS