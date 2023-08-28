# MMSEGMENTATION

This is a custom fork of the 'main' branch of the MMSegmenation repository with added files for training the binary segmentation model. 

The codebase can be found [here](https://github.com/open-mmlab/mmsegmentation).

Documentation can found [here](https://mmsegmentation.readthedocs.io/en/latest/)

## Installation
Use the official installation instructions [here - Case a](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation) to install the library.

## Training
Here is an example command to train the binary segmentation model on CUB-200-2011 dataset:
```
python -m torch.distributed.launch \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--nproc_per_node=4 \
--master_port=29500 \
--use_env \
/home/aaniraj/code/mmsegmentation/tools/train.py \
/home/aaniraj/code/mmsegmentation/configs/mask2former/mask2former_swin-t_8xb2-160k_cub_binary_seg-512x512.py \
--work-dir work_dirs/mask2former_swint_train_init \
--launcher pytorch
```
## Generating Masks
Use the "gen_masked_image_dataset.py" script to generate the masks for the CUB-200-2011 dataset. Detailed usage instructions are given below:
```
usage: gen_masked_image_dataset.py [-h] [--data_path DATA_PATH]
                                   [--config_file CONFIG_FILE]
                                   [--checkpoint_file CHECKPOINT_FILE]
                                   [--save_path SAVE_PATH]

Mask out images using a mmseg model

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        directory that contains cub files, must contain folder
                        "./images"
  --config_file CONFIG_FILE
                        path to config file
  --checkpoint_file CHECKPOINT_FILE
                        path to checkpoint file
  --save_path SAVE_PATH
                        path to save directory
```