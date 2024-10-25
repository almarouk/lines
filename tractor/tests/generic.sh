#!/bin/bash
 
#set conda env
export PATH=/s/apps/users/conda/install/23.11.0/bin/:$PATH
source /s/apps/users/almarouka/miniconda3/etc/profile.d/conda.sh
conda activate /s/apps/users/almarouka/conda/envs/line-detection
 
#launch command
python /s/apps/users/almarouka/line-detection/repo/test.py \
    --tag "$1" \
    --data-path "$2" \
    --ckpt-path "$3" \
    --batch-size 8 \
    --num-workers 8 \
    --output-path "/s/prods/mvg/_source_global/users/almarouka/training/line-detection/" \
    --seed 42\
    --suppress-exit true \

