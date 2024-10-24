#!/bin/bash
 
#set conda env
export PATH=/s/apps/users/conda/install/23.11.0/bin/:$PATH
source /s/apps/users/conda/install/23.11.0/etc/profile.d/conda.sh
conda activate /s/apps/users/almarouka/conda/envs/line-detection
 
#launch command
python /s/apps/users/almarouka/line-detection/repo/train.py \
    \
    --data-path "/s/prods/mvg/_source_global/users/almarouka/datasets/topology_dataset/output/" \
    --to-sdr "scale_max" \
    --batch-size 4 \
    --num-workers 4 \
    --max-distance 10 \
    \
    --clamp-output false \
    --size 32 \
    \
    --optimizer adam \
    --learning-rate 0.0001 \
    \
    --scheduler none \
    \
    --loss l1 \
    --output-path "/s/prods/mvg/_source_global/users/almarouka/training/line-detection/" \
    --tag "debug_cuda" \
    --epochs 100 \
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
