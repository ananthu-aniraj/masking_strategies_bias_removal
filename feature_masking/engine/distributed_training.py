import fsspec
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import copy
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timm.data import Mixup

from .utils import ddp_setup, Snapshot, set_seeds, init_wandb


class DistributedTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            batch_size: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            loss_fn: List[torch.nn.Module],
            save_every: int,
            snapshot_path: str,
            loggers: List,
            log_freq: int = 10,
            use_amp: bool = False,
            grad_norm_clip: float = 1.0,
            max_epochs: int = 100,
            num_workers: int = 4,
            mixup_fn: Optional[Mixup] = None,
            eval_only: bool = False,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model
        self.model.base_model = self.model.base_model.to(self.local_rank)
        self.model.seg_model = self.model.seg_model.to(self.local_rank)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.eval_only = eval_only
        self.train_loader = self._prepare_dataloader(train_dataset, num_workers=num_workers)
        self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers)
        if len(loss_fn) == 1:
            self.loss_fn_train = self.loss_fn_eval = loss_fn[0]
        else:
            self.loss_fn_train = loss_fn[0]
            self.loss_fn_eval = loss_fn[1]
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.isdir(snapshot_path):
            self.is_snapshot_dir = True
        else:
            self.is_snapshot_dir = False
        if loggers:
            if self.local_rank == 0 and self.global_rank == 0:
                loggers[0] = init_wandb(loggers[0])
        self.loggers = loggers
        self.log_freq = log_freq
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.grad_norm_clip = grad_norm_clip
        self.max_epochs = max_epochs
        self.mixup_fn = mixup_fn
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        if os.path.isfile(os.path.join(snapshot_path, f"snapshot_best.pt")):
            print("Loading snapshot")
            self._load_snapshot()
        elif os.path.isfile(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()

        self.model.seg_model = DDP(self.model.seg_model, device_ids=[self.local_rank])
        self.model.base_model = DDP(self.model.base_model, device_ids=[self.local_rank])
        self.epoch_test_accuracies = []

        if self.local_rank == 0 and self.global_rank == 0:
            for logger in self.loggers:
                logger.watch(model, log="all", log_freq=self.log_freq)

    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, num_workers: int = 4):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            sampler=DistributedSampler(dataset)
        )

    def _load_snapshot(self) -> None:
        try:
            if self.is_snapshot_dir:
                snapshot = fsspec.open(os.path.join(self.snapshot_path, f"snapshot_best.pt"))
            else:
                snapshot = fsspec.open(self.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        relevant_keys = [key for key in snapshot.model_state.keys() if "base_model." in key]
        relevant_keys_2 = [key for key in snapshot.model_state.keys() if "backbone." in key]
        relevant_keys_3 = [key for key in snapshot.model_state.keys() if "head." in key]
        relevant_keys_4 = [key for key in snapshot.model_state.keys() if "linear_head" in key]
        state_dict_base = {}

        state_dict = {key.replace('base_model.module.', ''): snapshot.model_state[key] for key in
                      relevant_keys}
        state_dict_2 = {key.replace('backbone.', ''): snapshot.model_state[key] for key in
                        relevant_keys_2}
        state_dict_base.update(state_dict)
        state_dict_base.update(state_dict_2)
        if not relevant_keys_3:
            state_dict_3 = {key.replace('linear_head.', 'head.linear_head.'): snapshot.model_state[key] for
                        key in relevant_keys_4}
            state_dict_base.update(state_dict_3)
        if not state_dict_base:
            state_dict_base = snapshot.model_state

        self.model.base_model.load_state_dict(state_dict_base)
        if self.eval_only:
            return
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        if snapshot.epoch_test_accuracies is not None:
            self.epoch_test_accuracies = copy.deepcopy(snapshot.epoch_test_accuracies)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, train: bool = True) -> Tuple[Any, Any]:

        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                               enabled=self.use_amp):
            outputs = self.model.forward_feat_ddp(source, train)

        if train:
            loss = self.loss_fn_train(outputs, targets)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_norm_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                if self.model.backward_hook is not None:
                    self.model.backward_hook.remove()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_norm_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                if self.model.backward_hook is not None:
                    self.model.backward_hook.remove()
                self.optimizer.step()
        else:
            loss = self.loss_fn_eval(outputs, targets)

        return outputs, loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        loss, acc = 0, 0
        if train and self.mixup_fn is not None:
            acc = None

        for it, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank, non_blocking=True)
            targets = targets.to(self.local_rank, non_blocking=True)
            if train and self.mixup_fn is not None:
                source, targets = self.mixup_fn(source, targets)
            batch_preds, batch_loss = self._run_batch(source, targets, train)
            loss += batch_loss
            if train and self.mixup_fn is not None:
                acc = None
            else:
                # Calculate and accumulate accuracy metric across all batches
                targets_pred = torch.argmax(torch.softmax(batch_preds, dim=1), dim=1)
                acc += (targets_pred == targets).sum().item() / len(batch_preds)

            if it % self.log_freq == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} Loss {batch_loss:.5f}")
        self.scheduler.step()
        loss /= len(dataloader)
        if acc is not None:
            acc /= len(dataloader)
        return loss, acc

    def _save_snapshot(self, epoch, save_best: bool = False):
        # capture snapshot
        model = self.model.base_model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            epoch_test_accuracies=self.epoch_test_accuracies,
        )
        # save snapshot
        snapshot = asdict(snapshot)
        if self.is_snapshot_dir:
            save_path_base = self.snapshot_path
        else:
            save_path_base = os.path.dirname(self.snapshot_path)
        if epoch == self.max_epochs:
            save_path = os.path.join(save_path_base, f"snapshot_final.pt")
        elif save_best:
            save_path = os.path.join(save_path_base, f"snapshot_best.pt")
        else:
            save_path = os.path.join(save_path_base, f"snapshot_{epoch}.pt")

        torch.save(snapshot, save_path)
        print(f"Snapshot saved at epoch {epoch}")

    def finish_logging(self):
        for logger in self.loggers:
            logger.finish()

    def train(self):
        for epoch in range(self.epochs_run, self.max_epochs):
            epoch += 1
            self.model.base_model.train()
            train_loss, train_acc = self._run_epoch(epoch, self.train_loader, train=True)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            elif self.local_rank == 0 and epoch == self.max_epochs:
                self._save_snapshot(epoch)
            # eval run
            if self.test_loader:
                self.model.base_model.eval()
                test_loss, test_acc = self._run_epoch(epoch, self.test_loader, train=False)
            if self.local_rank == 0 and self.global_rank == 0:
                self.epoch_test_accuracies.append(test_acc)
                max_acc = max(self.epoch_test_accuracies)
                max_acc_index = self.epoch_test_accuracies.index(max_acc)
                if max_acc_index == len(self.epoch_test_accuracies) - 1:
                    self._save_snapshot(epoch, save_best=True)
                for logger in self.loggers:
                    logger.log(
                        {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss,
                         "test_acc": test_acc})
        if self.local_rank == 0 and self.global_rank == 0:
            self.finish_logging()

    def test_only(self):
        self.model.base_model.eval()
        with torch.inference_mode():
            if self.test_loader:
                test_loss, test_acc = self._run_epoch(0, self.test_loader, train=False)
            print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.5f}")
        if self.local_rank == 0 and self.global_rank == 0:
            for logger in self.loggers:
                logger.log({"epoch": 0, "test_loss": test_loss, "test_acc": test_acc})
        self.finish_logging()


def distributed_trainer(model: torch.nn.Module,
                        train_dataset: torch.utils.data.Dataset,
                        test_dataset: torch.utils.data.Dataset,
                        batch_size: int,
                        optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler.LRScheduler,
                        loss_fn: List[torch.nn.Module],
                        epochs: int,
                        save_every: int,
                        loggers: List,
                        log_freq: int,
                        use_amp: bool = False,
                        snapshot_path: str = "snapshot.pt",
                        grad_norm_clip: float = 1.0,
                        num_workers: int = 0,
                        mixup_fn: Optional[Mixup] = None,
                        seed: int = 42,
                        eval_only: bool = False,
                        ) -> None:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through DistributedTrainer class
     for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_datasetloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    save_every: An integer indicating how often to save the model.
    snapshot_path: A string indicating where to save the model.
    loggers: A list of loggers to log metrics to.
    log_freq: An integer indicating how often to log metrics.
    grad_norm_clip: A float indicating the maximum gradient norm to clip to.
    enable_gradient_clipping: A boolean indicating whether to enable gradient clipping.
    mixup_fn: A Mixup instance to apply mixup to the training data.
    seed: An integer indicating the random seed to use.
    eval_only: A boolean indicating whether to only run evaluation.
    @rtype: None
    """

    set_seeds(seed)
    # Loop through training and testing steps for a number of epochs
    ddp_setup()
    trainer = DistributedTrainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset,
                                 batch_size=batch_size, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn,
                                 save_every=save_every, snapshot_path=snapshot_path, loggers=loggers,
                                 log_freq=log_freq,
                                 use_amp=use_amp,
                                 grad_norm_clip=grad_norm_clip, max_epochs=epochs, num_workers=num_workers,
                                 mixup_fn=mixup_fn, eval_only=eval_only)
    if eval_only:
        trainer.test_only()
    else:
        trainer.train()

    destroy_process_group()
