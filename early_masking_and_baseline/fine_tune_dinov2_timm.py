import argparse
import copy
import os
from pathlib import Path
from timeit import default_timer as timer

import torch
import torchvision
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from datasets.cub_dataset import CUBDataset
from engine.distributed_training import distributed_trainer
from engine.transforms_cub import build_transform_timm, make_train_transforms, make_test_transforms
from engine.utils import build_scheduler, build_optimizer
from layers.linear_head_vit_timm import linear_head_vit_dino_timm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Finetune DinoV2 on CUB'
    )
    parser.add_argument('--model_arch', default='vit_small_patch14_dinov2.lvd142m', type=str,
                        help='pick model architecture',
                        choices=['vit_small', 'vit_base',
                                 'vit_large', 'vit_giant'])
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=28, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--num_layers_linear_head', choices=[1, 4], default=1, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='fine-tune-dinov2', type=str)
    parser.add_argument('--dataset', default='CUB', type=str)
    parser.add_argument('--job_type', default='fine_tune_dino_v2', type=str)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--group', default='fine_tune_dino_v2', type=str)
    parser.add_argument('--grad_norm_clip', default=0.0, type=float)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--snapshot_dir', type=str)
    parser.add_argument('--save_every_n_epochs', default=10, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--image_sub_path_train', default='images', type=str)
    parser.add_argument('--image_sub_path_test', default='images', type=str)
    parser.add_argument('--train_split', default=0.9, type=float, help='fraction of training data to use')
    parser.add_argument('--eval_mode', default='val', choices=['train', 'val', 'test'], type=str,
                        help='which split to use for evaluation')
    parser.add_argument('--seed', default=42, type=int)
    # * Resume training params
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--wandb_resume_id', default=None, type=str)
    # * Optimizer params
    parser.add_argument('--optimizer_type', default='adamw', choices=['adam', 'adamw', 'sgd'], type=str)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--betas1', default=0.9, type=float)
    parser.add_argument('--betas2', default=0.999, type=float)
    # * Scheduler params
    parser.add_argument('--scheduler_type', default='cosine',
                        choices=['cosine', 'linearlr', 'cosine_warmup_restart', 'steplr', 'cosine_with_warmup'],
                        type=str)
    parser.add_argument('--scheduler_warmup_epochs', default=10, type=int)
    parser.add_argument('--scheduler_start_factor', default=0.333, type=float)
    parser.add_argument('--scheduler_end_factor', default=1.0, type=float)
    parser.add_argument('--scheduler_restart_factor', default=2, type=int)
    parser.add_argument('--scheduler_gamma', default=0.1, type=float)
    parser.add_argument('--scheduler_step_size', default=10, type=int)
    parser.add_argument('--scratch_lr_factor', default=100.0, type=float)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    # Evaluation params
    parser.add_argument('--crop_pct', type=float, default=None)

    # Model params
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--patch_tokens_only', action='store_true', default=False)
    parser.add_argument('--class_token_only', action='store_true', default=False)

    # * Mixup params
    parser.add_argument('--turn_on_mixup_or_cutmix', action='store_true')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Augmentation parameters
    parser.add_argument('--augmentations_to_use', type=str, default='timm',
                        choices=['timm', 'torchvision', 'cub_original'])
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--imagenet_default_mean_and_std', action='store_false', default=True)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_loggers = []
    if args.wandb:
        wandb_logger_settings = copy.deepcopy(vars(args))
        train_loggers.append(wandb_logger_settings)

    if os.path.exists(args.snapshot_dir):
        pass
    elif ".pt" not in args.snapshot_dir.split('/')[-1] or ".pth" not in args.snapshot_dir.split('/')[-1]:
        save_dir = Path(args.snapshot_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError('Snapshot checkpoint does not exist.')

    if args.class_token_only and args.patch_tokens_only:
        raise ValueError('Cannot have both class token only and patch tokens only.')

    use_pretrained = True
    pretrained_model_string_extra = "_patch14_dinov2.lvd142m"
    args.model_arch = args.model_arch + pretrained_model_string_extra

    if not args.eval_only:
        model_backbone = create_model(
            args.model_arch,
            pretrained=use_pretrained,
            drop_path_rate=args.drop_path,
        )
    else:
        model_backbone = create_model(
            args.model_arch,
            pretrained=use_pretrained,
        )

    # Get the transforms and load the dataset
    if args.augmentations_to_use == 'torchvision':
        train_transforms = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()
    elif args.augmentations_to_use == 'timm':
        train_transforms = build_transform_timm(args, is_train=True)
    elif args.augmentations_to_use == 'cub_original':
        train_transforms = make_train_transforms(args)

    test_transforms = make_test_transforms(args)

    dataset_train = CUBDataset(args.data_path, split=args.train_split, mode='train',
                               height=args.image_size, transform=train_transforms,
                               image_sub_path=args.image_sub_path_train)
    dataset_val = CUBDataset(args.data_path, train_samples=dataset_train.trainsamples,
                             mode=args.eval_mode, transform=test_transforms, image_sub_path=args.image_sub_path_test)

    num_cls = dataset_train.get_num_classes()

    # Convert BatchNorm to SyncBatchNorm if there is more than 1 GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_backbone)

    # Add extra layers
    model = linear_head_vit_dino_timm(model_backbone, model_backbone.embed_dim, num_classes=num_cls,
                                      patch_tokens_only=args.patch_tokens_only,
                                      class_token_only=args.class_token_only)

    finetune_parameters = []
    scratch_parameters = []

    # Freeze backbone if args.freeze_backbone is set
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if 'linear_head' in name:
            scratch_parameters.append(param)
        else:
            finetune_parameters.append(param)

    param_groups = [{'params': finetune_parameters, 'lr': args.lr},
                    {'params': scratch_parameters, 'lr': args.lr * args.scratch_lr_factor}]

    # Mixup/Cutmix
    mixup_fn = None
    mixup_active = args.turn_on_mixup_or_cutmix
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_cls)

    # Define loss and optimizer
    if mixup_fn is not None:
        # smoothing is handled with mix-up label transform
        loss_fn_train = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        loss_fn_train = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        loss_fn_train = torch.nn.CrossEntropyLoss()

    loss_fn_eval = torch.nn.CrossEntropyLoss()
    loss_fn = [loss_fn_train, loss_fn_eval]
    print("loss function = %s" % str(loss_fn_train))

    optimizer = build_optimizer(args, param_groups)
    scheduler = build_scheduler(args, optimizer)

    # Start the timer

    start_time = timer()

    # Setup training and save the results
    distributed_trainer(model=model,
                        train_dataset=dataset_train,
                        test_dataset=dataset_val,
                        batch_size=args.batch_size,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        epochs=args.epochs,
                        save_every=args.save_every_n_epochs,
                        loggers=train_loggers,
                        log_freq=args.log_interval,
                        use_amp=args.use_amp,
                        snapshot_path=args.snapshot_dir,
                        grad_norm_clip=args.grad_norm_clip,
                        num_workers=args.num_workers,
                        mixup_fn=mixup_fn,
                        seed=args.seed,
                        eval_only=args.eval_only,
                        )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()
