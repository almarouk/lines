#!/bin/bash
 
#set conda env
export PATH=/s/apps/users/conda/install/23.11.0/bin/:$PATH
source /s/apps/users/conda/install/23.11.0/etc/profile.d/conda.sh
conda activate /s/apps/users/almarouka/conda/envs/line-detection
 
#launch command
python /s/apps/users/almarouka/line-detection/repo/train.py \
    --data-path "/s/prods/mvg/_source_global/users/almarouka/datasets/topology_dataset/output/" \
    --batch-size 16 \
    --num-workers 4 \
    --max-distance 10 \
    --optimizer adam \
    --learning-rate 0.001 \
    --scheduler none \
    --loss l1 \
    --output-path "/s/prods/mvg/_source_global/users/almarouka/training/line-detection/experiment2/" \
    --epochs 40 \
    --val-every-epochs 2 \
    --ckpt-best-val true \
    --ckpt-every-epochs 2 \
    --keep-last-ckpts 1 \
    --log-every-iters 10 \
    --seed 0 \
    --clamp-output true
