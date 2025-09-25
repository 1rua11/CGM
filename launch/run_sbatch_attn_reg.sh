#!/bin/bash

port=29501
crop_size=512

rm -rf .job*
rm -rf .sba*

file=scripts/dist_train_voc_re.py
config=configs/voc_attn_reg.yaml

echo python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir work_dir_final

