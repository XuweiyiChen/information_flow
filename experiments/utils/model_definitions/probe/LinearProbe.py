# adapted from https://github.com/vturrisi/solo-learn/blob/main/solo/methods/linear.py


import logging
from typing import Any, Callable, Dict, List, Tuple, Union, Sequence

import lightning as pl
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from .accuracy_metrics import accuracy_at_k, weighted_mean
from .helpers import omegaconf_select, remove_bias_and_norm_from_weight_decay

class AverageLayers(nn.Module):
    # adapted from https://github.com/apple/ml-aim/
    def __init__(self, layers: Sequence[int], reduce: bool = False):
        super().__init__()
        self.layers = layers  # List of layer indices to average
        self.reduce = reduce  # Whether to reduce across sequence dimension

    def forward(
        self, layer_features: List[torch.Tensor]
    ) -> torch.Tensor:
        # layer_features: List[Tensor] where each tensor has shape (batch_size, seq_len, hidden_dim)
        # Select specified layers
        layer_features = [layer_features[layer_id] for layer_id in self.layers]
        # Stack along new dimension: (batch_size, seq_len, hidden_dim, num_layers)
        feats = torch.stack(layer_features, dim=-1)
        # Average across layers: (batch_size, seq_len, hidden_dim)
        feats = feats.mean(dim=-1)
        # If reduce=True, average across sequence dimension: (batch_size, hidden_dim)
        # If reduce=False, keep sequence dimension: (batch_size, seq_len, hidden_dim)
        return feats.mean(dim=1) if self.reduce else feats

    @property
    def max_block_id(self) -> int:
        return max(self.layers)  # Returns highest layer index used


class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        backbone: nn.Module,
    ):
        """Implements linear and finetune evaluation.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        backbone (nn.Module): backbone architecture for feature extraction.
        Cfg basic structure:
            data:
                num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.

            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): steps to decay the learning rate
                    if scheduler is step. Defaults to None.
                interval (str): interval to update the lr scheduler. Defaults to 'step'.

            finetune (bool): whether or not to finetune the backbone. Defaults to False.

            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        loss_func (Callable): loss function to use (for mixup, label smoothing or default).
        Defaults to None mixup_func (Callable, optional). function to convert data and targets
        with mixup/cutmix. Defaults to None.
        """

        super().__init__()

        cfg = self.add_and_assert_specific_cfg(cfg)

        # attention pooling
        # attention pooling to convert [B, T, D] -> [B, D]
        self.attention_pooling = nn.Sequential(
            nn.Linear(backbone.num_features, backbone.num_features // 4),
            nn.ReLU(),
            nn.Linear(backbone.num_features // 4, 1)
        )
        self.softmax = nn.Softmax(dim=1)  # softmax over token dimension

        # classifier
        self.classifier = nn.Linear(backbone.num_features, cfg.data.num_classes)  # type: ignore

        self.loss_func = nn.CrossEntropyLoss()

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: Union[int, None] = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd
        self.layer_decay: float = cfg.optimizer.layer_decay

        # scheduler related
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last

        # keep track of validation metrics
        self.validation_step_outputs = []

        self.eval_layer = cfg.eval_layer
        self.layer_window = cfg.layer_window
        
        # Create layer averaging module if using multiple layers
        if self.layer_window > 0:
            layers = range(max(0, self.eval_layer - self.layer_window), self.eval_layer)
            print(f'layers: {layers}')
            self.layer_averager = AverageLayers(
                layers=layers,
                reduce=False
            )

        self.backbone = backbone

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # default parameters for optimizer
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        # default for extra optimizer kwargs (use pytorch's default if not available)
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        cfg.optimizer.layer_decay = omegaconf_select(cfg, "optimizer.layer_decay", 0.0)

        # default for acc grad batches
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", 1)

        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        # default parameters for layer evaluation
        cfg.eval_layer = omegaconf_select(cfg, "eval_layer", -1)
        cfg.layer_window = omegaconf_select(cfg, "layer_window", 0)  # Default to 0 for original behavior

        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """


        learnable_params = (
            list(self.classifier.parameters()) +
            list(self.attention_pooling.parameters())
        )

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)

        return [optimizer], [scheduler]

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            pixel_values (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        with torch.no_grad():
            outputs = self.backbone(**kwargs)
            
            if self.layer_window > 0:
                # Get features from multiple layers and average them
                hidden_states = self.layer_averager(outputs['hidden_states'])  # [B, T, D]
            else:
                # Original behavior - use single layer
                hidden_states = outputs['hidden_states'][self.eval_layer-1]  # [B, T, D]

        attention_weights = self.attention_pooling(hidden_states)  # [B, T, 1]
        attention_weights = self.softmax(attention_weights)  # [B, T, 1]
        pooled_features = torch.sum(hidden_states * attention_weights, dim=1)  # [B, D]
        logits = self.classifier(pooled_features)
        return {"logits": logits}

    def shared_step(
        self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """
        inputs, labels = self.backbone.prepare_inputs(batch, return_labels=True)
        target = labels.to(self.device)
        metrics = {"batch_size": target.size(0)}

        out = self.forward(**inputs)["logits"]
        loss = F.cross_entropy(out, target)
        acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
        metrics.update({"loss": loss, "acc1": acc1, "acc5": acc5})

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """
        out = self.shared_step(batch, batch_idx)

        log = {"train_loss": out["loss"]}

        log.update({"train_acc1": out["acc1"], "train_acc5": out["acc5"]})
        # print every 100 steps
        if batch_idx % 10 == 0:
            print(log)

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """

        out = self.shared_step(batch, batch_idx)

        metrics = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        """

        val_loss = weighted_mean(self.validation_step_outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(self.validation_step_outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(self.validation_step_outputs, "val_acc5", "batch_size")
        self.validation_step_outputs.clear()

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log, sync_dist=True)
        print(f"Validation loss: {val_loss}, Validation accuracy @1: {val_acc1}, Validation accuracy @5: {val_acc5}")