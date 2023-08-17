#!/usr/bin/sh
echo "Launching job $OAR_JOBID on `oarprint gpunb` gpus on host `oarprint host`"
module load conda/2021.11-python3.9
module load cuda/11.0
cd /home/aaniraj/code/mmsegmentation/;
source activate ananthu_venv;
core="python -m torch.distributed.launch \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--nproc_per_node=4 \
--master_port=29500 \
--use_env \
/home/aaniraj/code/mmsegmentation/tools/train.py \
/home/aaniraj/code/mmsegmentation/configs/segformer/segformer_mit-b1_8xb2-160k_cub_binary_seg-512x512.py \
--work-dir work_dirs/segformer_m1_train_init \
--launcher pytorch"
echo "$core";
$core;
