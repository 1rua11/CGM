#!/bin/bash

port=29519
crop_size=512

file=scripts/dist_train_coco.py
config=configs/coco_attn_reg.yaml

echo python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_attn_coco
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_attn_coco
