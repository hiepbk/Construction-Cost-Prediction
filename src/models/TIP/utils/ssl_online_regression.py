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

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator


class RegressionMLP(nn.Module):
    """
    Simple MLP for regression evaluation.
    Takes frozen embeddings and predicts a scalar target.
    """
    def __init__(
        self,
        n_input: int,
        n_hidden: Optional[int] = None,
        p: float = 0.2,
    ):
        super().__init__()
        n_hidden = n_hidden or n_input
        
        self.mlp = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.BatchNorm1d(n_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(n_hidden // 2, 1)  # Single output for regression
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x).squeeze(-1)  # (B, 1) -> (B,)


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
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        regression_loss: str = 'huber',  # 'huber', 'mse', 'mae'
        huber_delta: float = 1.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        log_transform_target: bool = True,
        swav: bool = False,
        multimodal: bool = False,
        strategy: str = None,
    ):
        """
        Args:
            z_dim: Representation dimension (embedding size)
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the regression MLP
            regression_loss: Loss type ('huber', 'mse', 'mae')
            huber_delta: Delta parameter for Huber loss
            target_mean: Mean of normalized targets (for denormalization)
            target_std: Std of normalized targets (for denormalization)
            log_transform_target: Whether targets were log-transformed
            swav: Whether using SwAV (affects batch format)
            multimodal: Whether using multimodal data
            strategy: Strategy type ('tip', 'comparison', etc.)
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim or z_dim
        self.drop_p = drop_p
        self.regression_loss = regression_loss
        self.huber_delta = huber_delta
        self.target_mean = target_mean
        self.target_std = target_std
        self.log_transform_target = log_transform_target
        self.swav = swav
        self.multimodal = multimodal
        self.strategy = strategy
        
        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[RegressionMLP] = None
        self._recovered_callback_state: Optional[Dict[str, Any]] = None
        
        # Regression metrics
        self.mae_train = None
        self.rmse_train = None
        self.rmsle_train = None  # PRIMARY METRIC: Root Mean Squared Logarithmic Error
        self.r2_train = None
        self.mae_val = None
        self.rmse_val = None
        self.rmsle_val = None  # PRIMARY METRIC: Root Mean Squared Logarithmic Error
        self.r2_val = None
        
        # Track ALL validation samples by data_id (unique and deterministic)
        self.tracked_val_preds = {}  # Dict: {data_id: prediction_value}
        self.tracked_val_targets = {}  # Dict: {data_id: target_value}
        self.debug_printed_this_epoch = False  # Track if we've already printed debug info this epoch
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize the regression MLP and optimizer."""
        # Create regression MLP
        self.online_evaluator = RegressionMLP(
            n_input=self.z_dim,
            n_hidden=self.hidden_dim,
            p=self.drop_p,
        ).to(pl_module.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.online_evaluator.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Initialize metrics
        self.mae_train = torchmetrics.MeanAbsoluteError().to(pl_module.device)
        self.rmse_train = torchmetrics.MeanSquaredError(squared=False).to(pl_module.device)
        # RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
        # Use MSE on log1p values, then take sqrt
        self.rmsle_train = torchmetrics.MeanSquaredError().to(pl_module.device)
        self.r2_train = torchmetrics.R2Score().to(pl_module.device)
        
        self.mae_val = torchmetrics.MeanAbsoluteError().to(pl_module.device)
        self.rmse_val = torchmetrics.MeanSquaredError(squared=False).to(pl_module.device)
        self.rmsle_val = torchmetrics.MeanSquaredError().to(pl_module.device)
        self.r2_val = torchmetrics.R2Score().to(pl_module.device)
        
        # Tracking is done by data_id (unique and deterministic), no need for indices
        
        # Load checkpoint state if available
        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            if "optimizer_state" in self._recovered_callback_state:
                self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])
    
    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[list]]:
        """Extract inputs and targets from batch, handling different batch formats.
        
        Returns:
            x: Image input
            y: Target (if available)
            x_t: Tabular input (if available)
            data_ids: List of data_ids for this batch (if available)
        """
        data_ids = None  # Initialize data_ids
        if self.swav:
            x, y = batch
            x = x[0]
            x_t = None
            data_ids = None
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
        else:
            _, x, y = batch
            x_t = None
            data_ids = None
        
        x = x.to(device)
        if y is not None:
            y = y.to(device)
        if x_t is not None:
            x_t = x_t.to(device)
        # data_ids is a list of strings, no need to move to device
        
        return x, y, x_t, data_ids
    
    
    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
        dataset=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[list]]:
        """
        Shared step for training and validation.
        Returns: (predictions_normalized, targets_normalized, predictions_denorm, targets_denorm, loss)
        """
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y, x_t, data_ids = self.to_device(batch, pl_module.device)
                
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
        
        # Forward pass through regression MLP
        predictions_normalized = self.online_evaluator(representations)  # (B,)
        
        # Get targets from batch
        if y is None:
            raise ValueError("Targets must be in batch. Ensure dataset returns targets in __getitem__.")
        
        targets_normalized = y.float()  # Ensure float
        # Note: targets from dataset are log-transformed but may not be normalized
        # We normalize them here for loss calculation
        if self.target_std != 1.0 or self.target_mean != 0.0:
            # Normalize targets for loss calculation (if not already normalized)
            targets_for_loss = (targets_normalized - self.target_mean) / self.target_std
        else:
            targets_for_loss = targets_normalized
        
        # Calculate loss on normalized predictions
        if self.regression_loss == 'huber':
            loss = F.huber_loss(predictions_normalized, targets_for_loss, delta=self.huber_delta)
        elif self.regression_loss == 'mse':
            loss = F.mse_loss(predictions_normalized, targets_for_loss)
        elif self.regression_loss == 'mae':
            loss = F.l1_loss(predictions_normalized, targets_for_loss)
        else:
            raise ValueError(f"Unknown regression loss: {self.regression_loss}")
        
        # IMPORTANT: Convert predictions to original scale (USD/m²)
        # Step 1: Denormalize (if targets were normalized)
        # predictions_normalized is in normalized log space
        predictions_log_space = (predictions_normalized * self.target_std) + self.target_mean
        
        # Step 2: Inverse log-transform to get original scale (USD/m²)
        if self.log_transform_target:
            predictions_original_scale = torch.expm1(predictions_log_space)  # exp(x) - 1
        else:
            predictions_original_scale = predictions_log_space
        
        # Step 3: Also convert targets to original scale for comparison
        if self.log_transform_target:
            targets_original_scale = torch.expm1(targets_normalized)  # Targets are log-transformed but not normalized
        else:
            targets_original_scale = targets_normalized
        
        # Ensure non-negative (construction cost can't be negative)
        predictions_original_scale = torch.clamp(predictions_original_scale, min=0.0)
        targets_original_scale = torch.clamp(targets_original_scale, min=0.0)
        
        # Extract data_ids from batch (8th element for TIP multimodal)
        # data_ids is already extracted in to_device method, so it's already available
        # This is just for consistency - data_ids should already be set from to_device
        
        return predictions_normalized, targets_normalized, predictions_original_scale, targets_original_scale, loss, data_ids
    
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
        
        preds_norm, targets_norm, preds_denorm, targets_denorm, loss, _ = self.shared_step(
            pl_module, batch, dataset
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update metrics (on original scale values in USD/m²) - accumulate across batches
        # preds_denorm and targets_denorm are already in original scale (USD/m²)
        self.mae_train.update(preds_denorm, targets_denorm)
        self.rmse_train.update(preds_denorm, targets_denorm)
        self.r2_train.update(preds_denorm, targets_denorm)
        
        # RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
        # Compute log1p difference and update MSE metric, then take sqrt
        # Both preds_denorm and targets_denorm are in original scale (USD/m²)
        log_pred = torch.log1p(preds_denorm)
        log_target = torch.log1p(targets_denorm)
        self.rmsle_train.update(log_pred, log_target)
        
        # Log loss only (metrics will be logged at epoch end)
        # Use shorter name for progress bar to avoid truncation
        pl_module.log("regression_online.train.loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
    
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
        
        preds_norm, targets_norm, preds_denorm, targets_denorm, loss, data_ids = self.shared_step(
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
        
        # Update metrics (on original scale values in USD/m²) - accumulate across batches
        # preds_denorm and targets_denorm are already in original scale (USD/m²)
        self.mae_val.update(preds_denorm, targets_denorm)
        self.rmse_val.update(preds_denorm, targets_denorm)
        self.r2_val.update(preds_denorm, targets_denorm)
        
        # RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
        # Compute log1p difference and update MSE metric, then take sqrt
        # Both preds_denorm and targets_denorm are in original scale (USD/m²)
        log_pred = torch.log1p(preds_denorm)
        log_target = torch.log1p(targets_denorm)
        self.rmsle_val.update(log_pred, log_target)
        
        # Log loss only (metrics will be logged at epoch end)
        pl_module.log("regression_online.val.loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log training metrics at epoch end."""
        # Compute and log all metrics at epoch end
        mae_value = self.mae_train.compute()
        rmse_value = self.rmse_train.compute()
        rmsle_value = torch.sqrt(self.rmsle_train.compute())
        r2_value = self.r2_train.compute()
        
        # Log with full names (for WandB/logging)
        pl_module.log("regression_online.train.mae", mae_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.train.rmse", rmse_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.train.rmsle", rmsle_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.train.r2", r2_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log with short names for progress bar (only show RMSLE - the primary metric)
        pl_module.log("train_rmsle", rmsle_value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        
        # Reset metrics for next epoch
        self.mae_train.reset()
        self.rmse_train.reset()
        self.rmsle_train.reset()
        self.r2_train.reset()
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log validation metrics at epoch end."""
        # Compute and log all metrics at epoch end
        mae_value = self.mae_val.compute()
        rmse_value = self.rmse_val.compute()
        rmsle_value = torch.sqrt(self.rmsle_val.compute())
        r2_value = self.r2_val.compute()
        
        # Log with full names (for WandB/logging)
        pl_module.log("regression_online.val.mae", mae_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.val.rmse", rmse_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.val.rmsle", rmsle_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        pl_module.log("regression_online.val.r2", r2_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
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
        
        # Reset metrics for next epoch
        self.mae_val.reset()
        self.rmse_val.reset()
        self.rmsle_val.reset()
        self.r2_val.reset()
        
        # Note: tracked_val_preds and tracked_val_targets are cleared in on_validation_epoch_start
    
    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> dict:
        """Save regression head state."""
        return {
            "state_dict": self.online_evaluator.state_dict(),
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

