#!/bin/bash
export NCCL_CACHE_DISABLE=1


python tools/train.py configs/mobilenetv3/mobilenetv3_onecycle_b32x2_esfair.py
# python tools/train.py configs/resnet/resnet50_b16x8_hw.py


#  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  \
# 	--nproc_per_node 2 \
# 	--master_port 10553 \
# 	--use_env \
# 	tools/train.py \
#  	configs/mobilenetv3/mobilenetv3_onecycle_b32x2_esfair.py 

