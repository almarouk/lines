#!/bin/bash
 
#set conda env
export PATH=/s/apps/users/conda/install/23.11.0/bin/:$PATH
source /s/apps/users/almarouka/miniconda3/etc/profile.d/conda.sh
conda activate /s/apps/users/almarouka/conda/envs/line-detection
 
#launch command
python /s/apps/users/almarouka/line-detection/repo/train.py \
    \
    --data-path "/s/prods/mvg/_source_global/users/almarouka/datasets/topology_dataset/output/textured/;/s/prods/mvg/_source_global/users/almarouka/datasets/topology_dataset/output/" \
    --to-sdr "scale_max" \
    --batch-size 4 \
    --num-workers 8 \
    --max-distance 10 \
    \
    --backbone attention_unet \
    --clamp-output true \
    --size 32 \
    \
    --optimizer adam \
    --learning-rate 0.0001 \
    \
    --scheduler none \
    \
    --loss-type l1 \
    --weight-valid -1 \
    --output-path "/s/prods/mvg/_source_global/users/almarouka/training/line-detection/" \
    --tag "attention_s32_l1_none_b4_scalemax_clamp_both" \
    --epochs 50 \
    --clip-grad-norm 1.0 \
    \
    --val-every-epochs 1 \
    --ckpt-best-val true \
    --ckpt-every-epochs 1 \
    --keep-last-ckpts 1 \
    --log-every-iters 10 \
    \
    --reproducible false \
    --seed 42 \
    --debug true \
    --detect-anomaly false \
    --profiling true \
    --suppress-exit true \


    # --clip-grad-value 1.0 \
