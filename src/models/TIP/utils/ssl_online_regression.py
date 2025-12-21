'''
Online Regression Evaluator for Self-Supervised Learning
Evaluates pretrained representations by training a regression head on frozen features.
'''
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
import torchmetrics
from omegaconf import DictConfig

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

# Import head classes from ConstructionCostHead module
from models.ConstructionCostHead import get_head_class, create_head
from omegaconf import DictConfig, OmegaConf


class SSLOnlineEvaluatorRegression(Callback):
    """
    Online regression evaluator for self-supervised learning.
    
    Attaches a regression MLP to frozen pretrained features and trains it
    during pretraining to monitor representation quality.
    
    Example::
        online_eval = SSLOnlineEvaluatorRegression(
            z_dim=512,
            hidden_dim=256,
            regression_loss='huber',
            target_mean=0.0,
            target_std=1.0,
            log_transform_target=True
        )
    """
    
    def __init__(
        self,
        z_dim: int,
        regression_head: Dict,  # Dict containing all head configuration (from hparams.regression_head)
        swav: bool = False,
        multimodal: bool = False,
        strategy: str = None,
    ):
        """
        Args:
            z_dim: Representation dimension (embedding size)
            regression_head: Dict containing all head configuration (from hparams.regression_head)
                Must contain 'type' and all required parameters for the head class
            swav: Whether using SwAV (affects batch format)
            multimodal: Whether using multimodal data
            strategy: Strategy type ('tip', 'comparison', etc.)
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.swav = swav
        self.multimodal = multimodal
        self.strategy = strategy
        
        # Convert DictConfig to dict if needed
        if isinstance(regression_head, DictConfig):
            regression_head = OmegaConf.to_container(regression_head, resolve=True)
        
        if not isinstance(regression_head, dict):
            raise ValueError(f"regression_head must be dict or DictConfig, got {type(regression_head)}")
        
        if 'type' not in regression_head:
            raise ValueError("regression_head must contain 'type' key specifying head class name")
        
        self.regression_head_config = regression_head.copy()  # Store full config for checkpoint
        self.regression_head_class = regression_head['type']  # Store head class name for checkpoint (backward compat)
        
        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[nn.Module] = None  # Can be any head class
        self._recovered_callback_state: Optional[Dict[str, Any]] = None
        
        # Store validation losses from head for epoch-end aggregation
        # Training and validation losses come from head, so we don't need torchmetrics
        self.val_losses = []  # List of loss_dicts from each validation batch
        
        # Track ALL validation samples by data_id (unique and deterministic)
        self.tracked_val_preds = {}  # Dict: {data_id: prediction_value}
        self.tracked_val_targets = {}  # Dict: {data_id: target_value}
        self.debug_printed_this_epoch = False  # Track if we've already printed debug info this epoch
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize the regression head and optimizer."""
        # Use regression_head_config directly (all parameters should be in it)
        head_config = self.regression_head_config.copy()
        
        # Preprocess loss_type: convert DictConfig to dict if needed
        if 'loss_type' in head_config and isinstance(head_config['loss_type'], DictConfig):
            head_config['loss_type'] = dict(head_config['loss_type'])
        
        self.online_evaluator = create_head(
            head_config=head_config,
            n_input=self.z_dim
        ).to(pl_module.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.online_evaluator.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Initialize storage for validation losses (head already calculates them)
        self.val_losses = []
        
        # Tracking is done by data_id (unique and deterministic), no need for indices
        
        # Load checkpoint state if available
        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            if "optimizer_state" in self._recovered_callback_state:
                self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])
    
    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[list]]:
        """Extract inputs and targets from batch, handling different batch formats.
        
        Returns:
            x: Image input
            y: Target (if available) - normalized log-transformed
            x_t: Tabular input (if available)
            y_original: Original target in original scale (if available) - for RMSLE loss
            data_ids: List of data_ids for this batch (if available)
        """
        data_ids = None  # Initialize data_ids
        y_original = None  # Initialize y_original
        if self.swav:
            x, y = batch
            x = x[0]
            x_t = None
            data_ids = None
            y_original = None
        elif self.multimodal and self.strategy == 'tip':
            # TIP multimodal batch: (imaging_views, tabular_views, labels, unaugmented_image, unaugmented_tabular, target, target_original, data_id)
            # Must have exactly 8 elements
            if len(batch) == 8:
                x_i, _, contrastive_labels, x_orig, x_t_orig, y, y_original, data_ids = batch
            else:
                raise ValueError(f"Expected batch size 8, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id)")
            x = x_orig
            x_t = x_t_orig
        elif self.multimodal and self.strategy == 'comparison':
            x_i, _, y, x_orig = batch
            x = x_orig
            x_t = None
            data_ids = None
            y_original = None
        else:
            _, x, y = batch
            x_t = None
            data_ids = None
            y_original = None
        
        x = x.to(device)
        if y is not None:
            y = y.to(device)
        if x_t is not None:
            x_t = x_t.to(device)
        if y_original is not None:
            y_original = y_original.to(device)
        # data_ids is a list of strings, no need to move to device
        
        return x, y, x_t, y_original, data_ids
    
    
    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
        dataset=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Optional[list]]:
        """
        Shared step for training and validation.
        Returns: (predictions_normalized, targets_normalized, predictions_denorm, targets_denorm, loss, loss_dict, data_ids)
        """
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y, x_t, y_original, data_ids = self.to_device(batch, pl_module.device)
                
                # Get representations from frozen encoder
                if x_t is not None:
                    # Multimodal: get multimodal embeddings
                    # Extract image and tabular embeddings
                    _, image_embeddings = pl_module.forward_imaging(x)  # Returns (projection, embeddings)
                    # Keep full sequence (B, N, C) for multimodal encoder, not just CLS token
                    tabular_embeddings = pl_module.encoder_tabular(x_t)  # (B, N, C) where N includes CLS token
                    
                    # Get multimodal representation
                    representations = pl_module.forward_multimodal_feature(
                        tabular_features=tabular_embeddings,
                        image_features=image_embeddings
                    )
                else:
                    # Image-only: get image embeddings
                    _, representations = pl_module.forward_imaging(x)  # Returns (projection, embeddings)
        
        # Get targets from batch
        if y_original is None:
            raise ValueError("target_original must be in batch. Ensure dataset returns target_original in __getitem__.")
        
        # Forward pass through regression head with y_original (head handles everything internally)
        # We only pass y_original - the head will handle decoding and loss calculation
        head_result = self.online_evaluator(
            representations,
            target_original=y_original.float()  # Pass y_original directly, head handles everything
        )
        
        # Extract results from head
        predictions_normalized = head_result['prediction_log']  # (B,) - normalized log space
        predictions_original_scale = head_result['prediction_original']  # (B,) - original scale
        loss = head_result['loss']  # scalar - total weighted loss
        loss_dict = head_result['loss_dict']  # dict with all individual losses (for monitoring)
        
        # Use y_original directly (already in original scale from dataloader)
        targets_original_scale = y_original.float() if y_original is not None else None
        
        # For backward compatibility, also compute targets_normalized if needed
        # (but we don't use it for loss calculation anymore)
        if y is not None:
            targets_normalized = y.float()
        else:
            targets_normalized = None
        
        return predictions_normalized, targets_normalized, predictions_original_scale, targets_original_scale, loss, loss_dict, data_ids
    
    def _rmsle_loss(self, y_pred: torch.Tensor, y_true_original: torch.Tensor) -> torch.Tensor:
        """
        RMSLE Loss: sqrt(mean((log1p(y_true_orig) - log1p(y_pred_orig))^2))
        
        This directly optimizes the competition metric (RMSLE).
        
        IMPORTANT: 
        - y_pred: Predicted values in normalized log space → convert to original scale
        - y_true_original: Ground truth in ORIGINAL scale (USD/m²) - use directly, don't convert
        
        Args:
            y_pred: Predicted values (in log-normalized space)
            y_true_original: True values in ORIGINAL scale (USD/m²) - not normalized, not log-transformed
        
        Returns:
            RMSLE loss (scalar tensor)
        """
        # Convert y_pred from normalized log space to original scale
        # Step 1: Denormalize (reverse normalization)
        y_pred_log = y_pred * self.target_std + self.target_mean
        
        # Step 2: Reverse log-transform to get original scale
        if self.log_transform_target:
            y_pred_original = torch.expm1(y_pred_log)  # exp(x) - 1
        else:
            y_pred_original = y_pred_log
        
        # Ensure non-negative
        y_pred_original = torch.clamp(y_pred_original, min=0.0)
        y_true_original = torch.clamp(y_true_original, min=0.0)
        
        # Compute RMSLE: sqrt(mean((log1p(y_true_orig) - log1p(y_pred_orig))^2))
        log_pred = torch.log1p(y_pred_original)  # log(1 + y_pred)
        log_true = torch.log1p(y_true_original)  # log(1 + y_true) - using original GT directly
        
        squared_log_error = (log_true - log_pred) ** 2
        rmsle = torch.sqrt(torch.mean(squared_log_error))
        
        return rmsle
    
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int
    ) -> None:
        """Update regression head on training batch."""
        # Get dataset for target extraction if needed
        dataset = trainer.train_dataloader.dataset if hasattr(trainer.train_dataloader, 'dataset') else None
        
        preds_norm, targets_norm, preds_denorm, targets_denorm, loss, loss_dict, _ = self.shared_step(
            pl_module, batch, dataset
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Log losses from head (already calculated internally) - no need for manual metric calculation
        pl_module.log("regression_online.train.loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log individual losses from head (for monitoring)
        if 'rmsle' in loss_dict:
            pl_module.log("regression_online.train.rmsle", loss_dict['rmsle'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'mae' in loss_dict:
            pl_module.log("regression_online.train.mae", loss_dict['mae'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'rmse' in loss_dict:
            pl_module.log("regression_online.train.rmse", loss_dict['rmse'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'huber' in loss_dict:
            pl_module.log("regression_online.train.huber", loss_dict['huber'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'mse' in loss_dict:
            pl_module.log("regression_online.train.mse", loss_dict['mse'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize tracking at the start of validation epoch."""
        # Clear stored predictions for this epoch (tracked by data_id)
        self.tracked_val_preds = {}
        self.tracked_val_targets = {}
        # Reset debug print flag for this epoch
        self.debug_printed_this_epoch = False
    
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Evaluate regression head on validation batch."""
        dataset = trainer.val_dataloaders[dataloader_idx].dataset if hasattr(trainer.val_dataloaders[dataloader_idx], 'dataset') else None
        
        preds_norm, targets_norm, preds_denorm, targets_denorm, loss, loss_dict, data_ids = self.shared_step(
            pl_module, batch, dataset
        )
        
        # Track predictions for ALL samples in this batch using data_id from batch
        # data_ids comes directly from the batch (7th element), ensuring correct matching
        if data_ids is not None:
            # data_ids is a list/tuple of data_id strings for each sample in the batch
            for local_idx, data_id in enumerate(data_ids):
                if local_idx < len(preds_denorm):
                    # Store prediction (denormalized and inverse log-transformed) by data_id
                    self.tracked_val_preds[data_id] = float(preds_denorm[local_idx].cpu().detach())
                    # Store target (already in original scale from shared_step)
                    self.tracked_val_targets[data_id] = float(targets_denorm[local_idx].cpu().detach())
        else:
            # Fallback: if data_ids not in batch, use dataset lookup (less reliable)
            val_dataset = trainer.val_dataloaders[dataloader_idx].dataset if hasattr(trainer.val_dataloaders[dataloader_idx], 'dataset') else None
            val_dataloader = trainer.val_dataloaders[dataloader_idx]
            if hasattr(val_dataloader, 'batch_size') and val_dataloader.batch_size is not None:
                batch_size = val_dataloader.batch_size
            else:
                batch_size = len(preds_denorm)
            start_idx = batch_idx * batch_size
            
            for local_idx in range(len(preds_denorm)):
                global_idx = start_idx + local_idx
                if val_dataset is not None and hasattr(val_dataset, 'data_ids') and val_dataset.data_ids is not None:
                    data_id = str(val_dataset.data_ids[global_idx])
                else:
                    data_id = f"sample_{global_idx}"
                
                self.tracked_val_preds[data_id] = float(preds_denorm[local_idx].cpu().detach())
                self.tracked_val_targets[data_id] = float(targets_denorm[local_idx].cpu().detach())
        
        # Store loss_dict for epoch-end aggregation (head already calculated all losses)
        self.val_losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})
        
        # Log losses from head (already calculated) - PyTorch Lightning will aggregate automatically
        pl_module.log("regression_online.val.loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log individual losses from head (for monitoring)
        if 'rmsle' in loss_dict:
            pl_module.log("regression_online.val.rmsle", loss_dict['rmsle'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'mae' in loss_dict:
            pl_module.log("regression_online.val.mae", loss_dict['mae'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'rmse' in loss_dict:
            pl_module.log("regression_online.val.rmse", loss_dict['rmse'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'huber' in loss_dict:
            pl_module.log("regression_online.val.huber", loss_dict['huber'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if 'mse' in loss_dict:
            pl_module.log("regression_online.val.mse", loss_dict['mse'], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log training metrics at epoch end."""
        # Training losses are already logged during training_step from head
        # No need to compute metrics manually - head handles all loss calculation
        # This method is kept for compatibility but does nothing
        pass
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log validation metrics at epoch end."""
        # Aggregate losses from all validation batches (head already calculated them)
        if len(self.val_losses) == 0:
            return
        
        # Compute mean of each loss across all batches
        mae_value = torch.stack([loss_dict['mae'] for loss_dict in self.val_losses if 'mae' in loss_dict]).mean()
        rmse_value = torch.stack([loss_dict['rmse'] for loss_dict in self.val_losses if 'rmse' in loss_dict]).mean()
        rmsle_value = torch.stack([loss_dict['rmsle'] for loss_dict in self.val_losses if 'rmsle' in loss_dict]).mean()
        
        # Convert to float for logging
        mae_value = float(mae_value.item())
        rmse_value = float(rmse_value.item())
        rmsle_value = float(rmsle_value.item())
        
        # Log aggregated metrics (already logged per-batch, but log here for clarity)
        pl_module.log("regression_online.val.mae", mae_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.val.rmse", rmse_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.val.rmsle", rmsle_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log with short names for progress bar (only show RMSLE - the primary metric)
        pl_module.log("val_rmsle", rmsle_value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        
        # Log ALL tracked sample predictions with data_id
        # Group them under "regression" chart group in WandB
        if len(self.tracked_val_preds) > 0:
            logged_count = 0
            for data_id in self.tracked_val_preds.keys():
                if data_id in self.tracked_val_targets:
                    pred_val = self.tracked_val_preds[data_id]
                    target_val = self.tracked_val_targets[data_id]
                    
                    # IMPORTANT: pred_val is already in original scale (USD/m²)
                    # It has been:
                    # 1. Denormalized: (pred_norm * target_std) + target_mean
                    # 2. Inverse log-transformed: expm1(pred_denorm) if log_transform_target=True
                    # This is the FINAL prediction value in USD/m² (same unit as ground truth)
                    final_prediction = pred_val  # Already in original scale, ready for submission
                    
                    # Format title: "data_id: 1UTOX, 3434.35 USD/m²"
                    # Where 3434.35 is the ground truth value
                    title = f"data_id: {data_id}, {target_val:.2f} USD/m²"
                    
                    # Log prediction with data_id and ground truth in the title
                    # Group under "regression" chart group in WandB
                    # The logged value is the FINAL prediction in USD/m² (original scale, not log-transformed, not normalized)
                    pl_module.log(f"regression/{title}", final_prediction, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                    logged_count += 1
                else:
                    print(f"⚠️ Warning: data_id {data_id} not found in tracked targets. Available targets: {list(self.tracked_val_targets.keys())}")
            
            if logged_count == 0:
                print(f"⚠️ Warning: No validation samples were logged. Available predictions: {list(self.tracked_val_preds.keys())}")
        
        # Reset losses storage for next epoch
        self.val_losses = []
        
        # Note: tracked_val_preds and tracked_val_targets are cleared in on_validation_epoch_start
    
    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> dict:
        """Save regression head state."""
        return {
            "state_dict": self.online_evaluator.state_dict(),
            "regression_head_class": self.regression_head_class,  # Save head class name for loading
            "optimizer_state": self.optimizer.state_dict()
        }
    
    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]
    ) -> None:
        """Load regression head state."""
        self._recovered_callback_state = callback_state


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to temporarily set module training mode."""
    original_mode = module.training
    module.train(mode)
    try:
        yield
    finally:
        module.train(original_mode)

