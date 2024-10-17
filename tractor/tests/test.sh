#!/bin/bash
 
#set conda env
export PATH=/s/apps/users/conda/install/23.11.0/bin/:$PATH
source /s/apps/users/conda/install/23.11.0/etc/profile.d/conda.sh
conda activate /s/apps/users/almarouka/conda/envs/line-detection
 
#launch command
python /s/apps/users/almarouka/line-detection/repo/test.py \
    --data-path "/s/prods/mvg/_source_global/users/almarouka/datasets/topology_dataset/output/" \
    --ckpt-path "/s/prods/mvg/_source_global/users/almarouka/training/line-detection/experiment3/ckpt_0092_best.tar" \
    --batch-size 8 \
    --num-workers 4 \
    --output-path "/s/prods/mvg/_source_global/users/almarouka/training/line-detection/experiment3_test/"
