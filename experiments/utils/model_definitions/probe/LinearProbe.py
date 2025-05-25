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


class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        num_queries: int = 1,
        use_batch_norm: bool = True,
        qkv_bias: bool = False,
        linear_bias: bool = True,
        average_pool: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features, bias=linear_bias)
        self.bn = (
            nn.BatchNorm1d(dim, affine=False, eps=1e-6)
            if use_batch_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x_cls = F.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls

        out = self.linear(x_cls)
        return out

class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
 
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
        self.classifier = AttentionPoolingClassifier(
            dim=backbone.num_features,
            out_features=cfg.data.num_classes,
            num_heads=cfg.probe.num_heads,
            num_queries=cfg.probe.num_queries,
            use_batch_norm=cfg.probe.use_batch_norm,
        )

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

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last

        # keep track of validation metrics
        self.validation_step_outputs = []

        self.eval_layer = cfg.eval_layer
        self.layer_window = cfg.layer_window
        
        # Create layer averaging module if using multiple layers
        if self.layer_window > 0:
            layers = range(max(0, self.eval_layer - self.layer_window), self.eval_layer)
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


        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        # default parameters for layer evaluation
        cfg.probe = omegaconf_select(cfg, "probe", {})
        cfg.probe.eval_layer = omegaconf_select(cfg, "eval_layer", -1)
        cfg.probe.layer_window = omegaconf_select(cfg, "layer_window", 0)  # Default to 0 for original behavior
        cfg.probe.num_heads = omegaconf_select(cfg, "num_heads", 8)
        cfg.probe.num_queries = omegaconf_select(cfg, "num_queries", 1)
        cfg.probe.use_batch_norm = omegaconf_select(cfg, "use_batch_norm", True)

        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """


        learnable_params = (
            list(self.classifier.parameters())
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

        return optimizer

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

        logits = self.classifier(hidden_states)
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
        print(f"Validation loss: {val_loss.item()}, Validation accuracy @1: {val_acc1.item()}, Validation accuracy @5: {val_acc5.item()}")