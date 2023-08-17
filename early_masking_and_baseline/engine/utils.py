from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List
from .custom_lr_scheduler import LinearWarmupCosineAnnealingLR
import matplotlib.pyplot as plt
import torch
import wandb
from torch.distributed import init_process_group
import os
import copy


@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int
    epoch_test_accuracies: List[float] = None


def init_wandb(args):
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)
    if args["resume_training"]:
        if args_dict["wandb_resume_id"] is not None:
            run = wandb.init(project=args_dict["wandb_project"], entity='ananthu-phd', job_type=args["job_type"],
                             group=args["group"],
                             config=args_dict, id=args_dict["wandb_resume_id"], resume="must")
        else:
            raise ValueError("wandb_resume_id is None")
    else:
        run = wandb.init(project=args_dict["wandb_project"], entity='ananthu-phd', job_type=args["job_type"],
                         group=args["group"],
                         config=args_dict)
    return run


def build_optimizer(args, params_groups):
    type_optim = args.optimizer_type
    weight_decay = args.weight_decay
    if type_optim == 'adamw':
        return torch.optim.AdamW(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                 weight_decay=weight_decay)
    elif type_optim == 'sgd':
        return torch.optim.SGD(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                               nesterov=True)
    elif type_optim == 'adam':
        return torch.optim.Adam(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError


def build_scheduler(args, optimizer):
    # initialize scheduler hyperparameters
    total_steps = args.epochs
    type_lr_schedule = args.scheduler_type
    warmup_steps = args.scheduler_warmup_epochs
    start_factor = args.scheduler_start_factor
    end_factor = args.scheduler_end_factor
    restart_factor = args.scheduler_restart_factor
    gamma = args.scheduler_gamma
    step_size = args.scheduler_step_size
    min_lr = args.min_lr

    if type_lr_schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    elif type_lr_schedule == 'cosine_warmup_restart':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=restart_factor,
                                                                    eta_min=min_lr)
    elif type_lr_schedule == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif type_lr_schedule == 'linearlr':
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor,
                                                 total_iters=total_steps)
    elif type_lr_schedule == 'cosine_with_warmup':
        return LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_steps, max_epochs=total_steps,
                                             eta_min=min_lr)
    else:
        raise NotImplementedError


def ddp_setup():
    init_process_group(backend="nccl")


def set_seeds(seed_value: int = 42):
    # Set the manual seeds
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)


def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("accuracy.png", bbox_inches="tight")
