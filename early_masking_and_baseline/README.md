# Early Masking And Baseline

This folder contains the code for the early masking and baseline experiments.

# Training

To train the baseline models (frozen backbone), here is an example command to train the dinov2 pretrained "vit_large" model on CUB-200-2011 dataset:

```
torchrun --nnodes=1 --nproc_per_node=4 /home/aaniraj/code/part_detection/fine_tune_dinov2.py --model_to_finetune vit_large --data_path /home/aaniraj/data/cub200/CUB_200_2011 --wandb --batch_size 48 --epochs 90 --dataset cub --save_every_n_epochs 20 --num_workers 2 --num_layers_linear_head 1 --image_sub_path_train images --image_sub_path_test images --train_split 1 --eval_mode test --wandb_project cub_vit_cnn --job_type fine_tune_dinov2 --group vit_large --snapshot_dir exps/vit_large/vit_large_cub_90_epoch_timm_repeat --lr 1e-3 --optimizer_type adamw --weight_decay 1e-2 --scheduler_type cosine --scratch_lr_factor 1 --smoothing 0.1 --augmentations_to_use timm --freeze_backbone --image_size 518
```

Here is an example command to train the early masking models (frozen backbone), where the "vit_large" model is trained with early masking:

```
torchrun --nnodes=1 --nproc_per_node=4 /home/aaniraj/code/part_detection/fine_tune_dinov2.py --model_to_finetune vit_large --data_path /home/aaniraj/data/cub200/CUB_200_2011 --wandb --batch_size 64 --epochs 90 --dataset cub --save_every_n_epochs 20 --num_workers 2 --num_layers_linear_head 1 --image_sub_path_train images --image_sub_path_test images --train_split 1 --eval_mode test --wandb_project 2-stage-decoupled-classifier --job_type fine_tune_dinov2 --group vit_large --snapshot_dir exps/vit_large/vit_large_2stage_90_epoch_timm_patch_only --lr 1e-3 --optimizer_type adamw --weight_decay 1e-2 --scheduler_type cosine --scratch_lr_factor 1 --smoothing 0.1 --augmentations_to_use timm --freeze_backbone --image_size 518
```

Here is an example command to train the baseline models (fine-tuned backbone), where the "vit_large" model is fine-tuned on CUB-200-2011 dataset:

```
torchrun --nnodes=1 --nproc_per_node=4  /home/ananthu/Projects/code/part_detection/fine_tune_dinov2_timm.py --model_to_finetune vit_large --data_path /home/ananthu/DATA/data_ananthu/cub200/CUB_200_2011 --batch_size 6 --wandb --epochs 300 --dataset cub --save_every_n_epochs 80 --num_workers 4 --num_layers_linear_head 1 --image_sub_path_train images --image_sub_path_test images --train_split 1 --eval_mode test --wandb_project 2-stage-decoupled-classifier --job_type fine_tune_dinov2 --group vit_large --snapshot_dir /home/ananthu/DATA/classifier_masked_baseline/exps/vit_large/2stage_300_epoch_timm_unfrozen_patch --lr 4e-6 --min_lr 1e-8 --optimizer_type adamw --weight_decay 5e-2 --scheduler_type cosine_with_warmup --scheduler_warmup_epochs 20 --scratch_lr_factor 1000 --drop_path 0.1 --smoothing 0.1 --augmentations_to_use timm --image_size 518 --turn_on_mixup_or_cutmix
```
NOTES:
- Feel free to change the parameters for "nnodes" (number of machines) and "nproc_per_node" (number of GPUs per machine) as per available resources. 
- Please note that the batch size is per GPU, so the total batch size will be "batch_size * nproc_per_node * nnodes".
- The "wandb" parameter will log the training progress to Weights and Biases. Feel free to remove it if you don't want to use it. The "wandb_project" parameter specifies the name of the project in Weights and Biases.
- The "snapshot_dir" parameter specifies the directory where the model checkpoints will be saved.
- The "group" parameter specifies the name of the experiment group in Weights and Biases.
- The "job_type" parameter specifies the name of the experiment in Weights and Biases.
- The "image_size" parameter specifies the image size to use for training and evaluation. 
- The "freeze_backbone" parameter freezes the backbone weights during training. Feel free to remove it if you want to fine-tune the backbone as well.
- The "image_sub_path_train" and "image_sub_path_test" parameters specify the sub-path to the images in the train and test splits of the dataset. This is used to control if you are training on the original images or the masked images. 
- The "train_split" parameter specifies the train split to use. We use the "1" split for all our experiments.
- The "eval_mode" parameter specifies the evaluation mode. We use "test" for all our experiments.
- The "augmentations_to_use" parameter specifies the augmentations to use during training. We use "timm" for all our experiments.
- The "turn_on_mixup_or_cutmix" parameter turns on mixup or cutmix during training. In our experiments, we use such augmentations only in the case of fine-tuning.
- Use the "eval_only" flag if you just want to evaluate a trained model

