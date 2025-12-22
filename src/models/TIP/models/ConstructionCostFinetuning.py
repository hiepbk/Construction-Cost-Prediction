'''
Fine-tuning module for Construction Cost Prediction
Adapts TIP's pretrained backbone for regression task with log-transformed targets.
'''
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from models.ConstructionCostPrediction import ConstructionCostPrediction


class ConstructionCostFinetuning(pl.LightningModule):
    """
    Fine-tuning module for construction cost regression using TIP-pretrained backbone.
    
    Features:
    - Log-transform target (log(1 + cost))
    - RMSLE, Huber, MAE, or MSE loss
    - MAE, RMSE, RMSLE metrics
    - Denormalization for evaluation
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Set task to regression
        if not hasattr(hparams, 'task'):
            hparams.task = 'regression'
        if not hasattr(hparams, 'num_classes'):
            hparams.num_classes = 1  # Regression: single output
        

        
        # Pass hparams directly (it already contains checkpoint and field_lengths_tabular)
        self.model = ConstructionCostPrediction(hparams=hparams)
        
        # Get head type for logging
        if hasattr(hparams, 'regression_head') or 'regression_head' in hparams:
            head_config = getattr(hparams, 'regression_head', hparams.get('regression_head', {}))
            if hasattr(head_config, 'type') or (isinstance(head_config, dict) and 'type' in head_config):
                head_type = getattr(head_config, 'type', head_config.get('type', 'RegressionMLP'))
            else:
                head_type = 'RegressionMLP'
        else:
            head_type = getattr(hparams, 'regression_head_class', 'RegressionMLP')
        print(f"✅ ConstructionCostPrediction model created with head: {head_type}")
        
        # Loss configuration is now handled by the head (dict format)
        # No need to set up criterion here - head handles all loss calculation
        
        # Metrics for test only (evaluation metrics, not losses)
        # Training and validation losses come from head, so we don't need torchmetrics
        self.mae_test = torchmetrics.MeanAbsoluteError()
        self.rmse_test = torchmetrics.MeanSquaredError()
        self.rmsle_test = torchmetrics.MeanSquaredError()
        
        # Store validation losses from head for epoch-end aggregation
        self.val_losses = []  # List of loss_dicts from each validation batch
        
        # Track predictions for WandB logging (like pretraining)
        self.tracked_val_preds = {}  # Dict: {data_id: prediction_value}
        self.tracked_val_targets = {}  # Dict: {data_id: target_value}
        
        # Target normalization parameters (for denormalization)
        self.target_mean = getattr(hparams, 'target_mean', 0.0)
        self.target_std = getattr(hparams, 'target_std', 1.0)
        self.target_log_transform = getattr(hparams, 'target_log_transform', True)
        
        self.best_val_mae = float('inf')
        self.best_val_rmse = float('inf')
        self.best_val_rmsle = float('inf')
        self.best_val_score = float('inf')  # For compatibility with evaluate.py (will be set based on eval_metric)
        
        # Get loss config for printing
        loss_config = getattr(hparams, 'regression_loss', {'rmsle': 1.0})
        
        print(f"Initialized ConstructionCostFinetuning")
        print(f"  Loss config: {loss_config}")
        print(f"  Target log-transform: {self.target_log_transform}")
        print(f"  Target normalization: mean={self.target_mean:.2f}, std={self.target_std:.2f}")
    
    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:
        """
        Denormalize target from log space to original space.
        
        Args:
            y: Normalized target (in log space if log_transform=True)
        
        Returns:
            Original target values (USD/m²)
        """
        # Denormalize: y_orig = y_norm * std + mean
        y_denorm = y * self.target_std + self.target_mean
        
        # Reverse log-transform: exp(y) - 1
        if self.target_log_transform:
            y_denorm = torch.expm1(y_denorm)  # exp(y) - 1
        
        return y_denorm
    
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
        if self.target_log_transform:
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
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Tuple of (image, tabular) or (image, tabular, mask)
        
        Returns:
            dict with keys:
                - 'prediction_log': (B,) - Prediction in normalized log space
                - 'prediction_original': (B,) - Prediction in original scale (USD/m²)
        """
        return self.model(x)  # Returns dict with 'prediction_log' and 'prediction_original'
    
    def training_step(self, batch: Tuple, _) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: From ConstructionCostTIPDataset:
                  (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id)
                  For fine-tuning, we use unaugmented views and target
        """
        # Unpack batch from ConstructionCostTIPDataset (8 elements)
        if len(batch) == 8:
            imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id = batch
            # Use unaugmented views for fine-tuning
            x = (unaugmented_image, unaugmented_tabular)
            y = target  # Target is already normalized and log-transformed
            y_original = target_original  # Ground truth in original scale (USD/m²)
        elif len(batch) == 2:
            # Fallback for other datasets
            x, y = batch
            y_original = None  # Not available for other datasets
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}. Expected 8 (ConstructionCostTIPDataset) or 2 (other datasets)")
        
        # Forward pass through model (returns x_m from backbone)
        x_m = self.model.backbone(x, visualize=False)  # (B, 20, 512)
        
        # Forward through regression head with y_original (head handles everything internally)
        # We only pass y_original - the head will handle decoding and loss calculation
        head_result = self.model.regression(
            x_m, 
            target_original=y_original.float()  # Pass y_original directly, head handles everything
        )
        
        # Extract results from head
        y_hat = head_result['prediction_log']  # (B,) - normalized log space
        y_hat_original = head_result['prediction_original']  # (B,) - original scale
        loss = head_result['loss']  # scalar - total weighted loss (for backprop)
        loss_dict = head_result['loss_dict']  # dict with all individual losses
        
        # Get ground truth in original scale for metrics
        if y_original is not None:
            y_true_original = y_original.squeeze()
        else:
            # Fallback: convert from log space
            y_true_original = self.denormalize_target(y.detach().squeeze())
        
        # Ensure non-negative
        y_true_original = torch.clamp(y_true_original, min=0.0)
        
        # Log losses from head (already calculated internally) - no need for manual metric calculation
        batch_size = y_hat_original.shape[0] if len(y_hat_original.shape) > 0 else 1
        self.log('eval.train.loss', loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        # Log individual losses from head (already calculated, no need to recompute)
        if 'rmsle' in loss_dict:
            self.log('eval.train.rmsle', loss_dict['rmsle'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
            self.log('train_rmsle', loss_dict['rmsle'], on_epoch=True, on_step=False, prog_bar=True, logger=False, sync_dist=True, batch_size=batch_size)
        if 'mae' in loss_dict:
            self.log('eval.train.mae', loss_dict['mae'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        if 'rmse' in loss_dict:
            self.log('eval.train.rmse', loss_dict['rmse'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        if 'huber' in loss_dict:
            self.log('eval.train.huber', loss_dict['huber'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        if 'mse' in loss_dict:
            self.log('eval.train.mse', loss_dict['mse'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def training_epoch_end(self, _) -> None:
        """Reset metrics after training epoch (not used for training, but kept for compatibility)"""
        # Training metrics are not calculated manually anymore (head handles losses)
        # But we keep this method for compatibility
        pass
    
    def validation_step(self, batch: Tuple, _) -> None:
        """Validation step"""
        # Unpack batch from ConstructionCostTIPDataset (8 elements)
        if len(batch) == 8:
            imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id = batch
            # Use unaugmented views for validation
            x = (unaugmented_image, unaugmented_tabular)
            y = target  # Target is already normalized and log-transformed
            y_original = target_original  # Ground truth in original scale (USD/m²)
        elif len(batch) == 2:
            # Fallback for other datasets
            x, y = batch
            y_original = None  # Not available for other datasets
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}. Expected 8 (ConstructionCostTIPDataset) or 2 (other datasets)")
        
        # Forward pass through model (returns x_m from backbone)
        x_m = self.model.backbone(x, visualize=False)  # (B, 20, 512)
        
        # Forward through regression head with y_original (head handles everything internally)
        head_result = self.model.regression(
            x_m,
            target_original=y_original.float()  # Pass y_original directly, head handles everything
        )
        
        # Extract results from head
        y_hat = head_result['prediction_log']  # (B,) - normalized log space
        y_hat_original = head_result['prediction_original']  # (B,) - original scale
        loss = head_result['loss']  # scalar - total weighted loss
        loss_dict = head_result['loss_dict']  # dict with all individual losses
        
        # Use y_original directly (already in original scale from dataloader)
        y_true_original = y_original.squeeze() if y_original is not None else None
        
        # Track predictions for WandB logging (like pretraining)
        if data_id is not None:
            for idx, did in enumerate(data_id):
                self.tracked_val_preds[str(did)] = float(y_hat_original[idx].cpu().detach())
                self.tracked_val_targets[str(did)] = float(y_true_original[idx].cpu().detach())
        
        # Store loss_dict for epoch-end aggregation (head already calculated all losses)
        self.val_losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})
        
        # Log losses from head (already calculated) - PyTorch Lightning will aggregate automatically
        batch_size = y_hat_original.shape[0] if len(y_hat_original.shape) > 0 else 1
        self.log('eval.val.loss', loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        # Log individual losses from head (already calculated)
        if 'rmsle' in loss_dict:
            self.log('eval.val.rmsle', loss_dict['rmsle'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        if 'mae' in loss_dict:
            self.log('eval.val.mae', loss_dict['mae'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        if 'rmse' in loss_dict:
            self.log('eval.val.rmse', loss_dict['rmse'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        if 'huber' in loss_dict:
            self.log('eval.val.huber', loss_dict['huber'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        if 'mse' in loss_dict:
            self.log('eval.val.mse', loss_dict['mse'], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
    
    def on_validation_epoch_start(self) -> None:
        """Clear tracked predictions and losses at start of validation epoch"""
        self.tracked_val_preds = {}
        self.tracked_val_targets = {}
        self.val_losses = []  # Reset for new epoch
    
    def validation_epoch_end(self, _) -> None:
        """Compute validation metrics and log predictions to WandB (like pretraining)"""
        if self.trainer.sanity_checking:
            return
        
        # Aggregate losses from all validation batches (head already calculated them)
        if len(self.val_losses) == 0:
            return
        
        # Compute mean of each loss across all batches
        mae_val = torch.stack([loss_dict['mae'] for loss_dict in self.val_losses if 'mae' in loss_dict]).mean()
        rmse_val = torch.stack([loss_dict['rmse'] for loss_dict in self.val_losses if 'rmse' in loss_dict]).mean()
        rmsle_val = torch.stack([loss_dict['rmsle'] for loss_dict in self.val_losses if 'rmsle' in loss_dict]).mean()
        
        # Convert to float for logging
        mae_val = float(mae_val.item())
        rmse_val = float(rmse_val.item())
        rmsle_val = float(rmsle_val.item())
        
        # Log aggregated metrics (already logged per-batch, but log here for clarity and best checkpoint tracking)
        batch_size = 1  # Not needed for epoch-level logging
        self.log('eval.val.mae', mae_val, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('eval.val.rmse', rmse_val, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('eval.val.rmsle', rmsle_val, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)  # Primary metric
        # Primary metric for progress bar
        self.log('val_rmsle', rmsle_val, on_epoch=True, on_step=False, prog_bar=True, logger=False, sync_dist=True, batch_size=batch_size)
        
        # Log ALL tracked sample predictions with data_id to WandB (like pretraining)
        if len(self.tracked_val_preds) > 0:
            for data_id in self.tracked_val_preds.keys():
                if data_id in self.tracked_val_targets:
                    pred_val = self.tracked_val_preds[data_id]
                    target_val = self.tracked_val_targets[data_id]
                    
                    # Format: "data_id: {data_id}, {ground_truth_value:.2f} USD/m²"
                    title = f"data_id: {data_id}, {target_val:.2f} USD/m²"
                    
                    # Log prediction with data_id and ground truth in the title
                    # Group under "regression" chart group in WandB
                    # The logged value is the FINAL prediction in USD/m² (original scale)
                    self.log(f"regression/{title}", pred_val, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Track best metrics
        if mae_val < self.best_val_mae:
            self.best_val_mae = mae_val
        if rmse_val < self.best_val_rmse:
            self.best_val_rmse = rmse_val
        if rmsle_val < self.best_val_rmsle:
            self.best_val_rmsle = rmsle_val
        
        # Set best_val_score based on eval_metric (for compatibility with evaluate.py)
        eval_metric = getattr(self.hparams, 'eval_metric', 'mae')
        if eval_metric == 'mae':
            self.best_val_score = self.best_val_mae
        elif eval_metric == 'rmse':
            self.best_val_score = self.best_val_rmse
        elif eval_metric == 'rmsle':
            self.best_val_score = self.best_val_rmsle
        else:
            # Default to MAE
            self.best_val_score = self.best_val_mae
        
        # Reset tracking for next epoch
        self.tracked_val_preds = {}
        self.tracked_val_targets = {}
        self.val_losses = []
    
    def test_step(self, batch: Tuple, _) -> None:
        """Test step"""
        # Unpack batch from ConstructionCostTIPDataset (8 elements)
        if len(batch) == 8:
            imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id = batch
            # Use unaugmented views for testing
            x = (unaugmented_image, unaugmented_tabular)
            y = target  # Target is already normalized and log-transformed (may be dummy 0.0 for test set)
        elif len(batch) == 2:
            # Fallback for other datasets
            x, y = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}. Expected 8 (ConstructionCostTIPDataset) or 2 (other datasets)")
        
        # Forward pass through model (returns x_m from backbone)
        x_m = self.model.backbone(x, visualize=False)  # (B, 20, 512)
        
        # Forward through regression head (no targets for test, just predictions)
        head_result = self.model.regression(x_m)
        
        # Extract results from head
        y_hat = head_result['prediction_log']  # (B,) - normalized log space
        y_hat_original = head_result['prediction_original']  # (B,) - original scale
        
        # Metrics (in log space) - only if target is valid (not dummy)
        if y is not None and not (isinstance(y, torch.Tensor) and (y == 0.0).all()):
            y_hat_detached = y_hat.detach().squeeze()
            y_detached = y.squeeze()
            
            self.mae_test(y_hat_detached, y_detached)
            self.rmse_test(y_hat_detached, y_detached)
    
    def test_epoch_end(self, _) -> None:
        """Compute test metrics"""
        mae_test = self.mae_test.compute()
        rmse_test = torch.sqrt(self.rmse_test.compute())
        
        self.log('eval.test.mae', mae_test, metric_attribute=self.mae_test)
        self.log('eval.test.rmse', rmse_test)
    
    def configure_optimizers(self):
        """
        Configure optimizer and scheduler for fine-tuning.
        If finetune_strategy='frozen', only train the regression head (classifier).
        """
        finetune_strategy = getattr(self.hparams, 'finetune_strategy', 'trainable')
        
        # Layer-wise learning rates
        param_groups = []
        
        # Regression head: always trainable
        regressor_params = list(self.model.regression.parameters())
        if regressor_params:
            regressor_lr = self.hparams.lr_finetune * 10.0  # 10x for regression head
            param_groups.append({
                'params': regressor_params,
                'lr': regressor_lr
            })
            print(f"✅ Regression head: {len(regressor_params)} parameter groups, LR={regressor_lr:.2e}")
        
        # Only add encoder parameters if finetune_strategy is 'trainable'
        if finetune_strategy == 'trainable':
            # Multimodal encoder: base LR
            multimodal_params = list(self.model.backbone.encoder_multimodal.parameters())
            if multimodal_params:
                param_groups.append({
                    'params': multimodal_params,
                    'lr': self.hparams.lr_finetune
                })
                print(f"✅ Multimodal encoder: trainable, LR={self.hparams.lr_finetune:.2e}")
            
            # Tabular encoder: base LR (or lower if freeze_tabular=True)
            tabular_params = list(self.model.backbone.encoder_tabular.parameters())
            if tabular_params and not getattr(self.hparams, 'freeze_tabular', False):
                tabular_lr = self.hparams.lr_finetune * 0.1 if getattr(self.hparams, 'freeze_tabular', False) else self.hparams.lr_finetune
                param_groups.append({
                    'params': tabular_params,
                    'lr': tabular_lr
                })
                print(f"✅ Tabular encoder: trainable, LR={tabular_lr:.2e}")
            
            # Image encoder: lower LR (or frozen)
            image_params = list(self.model.backbone.encoder_imaging.parameters())
            if image_params and not getattr(self.hparams, 'freeze_image', True):
                image_lr = self.hparams.lr_finetune * 0.01 if getattr(self.hparams, 'freeze_image', True) else self.hparams.lr_finetune * 0.1
                param_groups.append({
                    'params': image_params,
                    'lr': image_lr
                })
                print(f"✅ Image encoder: trainable, LR={image_lr:.2e}")
        else:
            # finetune_strategy == 'frozen': only train classifier
            print("🔒 Backbone frozen: Only training regression head (classifier)")
            # Verify encoders are frozen
            total_frozen = sum(1 for p in self.model.backbone.encoder_imaging.parameters() if not p.requires_grad)
            total_frozen += sum(1 for p in self.model.backbone.encoder_tabular.parameters() if not p.requires_grad)
            total_frozen += sum(1 for p in self.model.backbone.encoder_multimodal.parameters() if not p.requires_grad)
            total_params = sum(1 for p in self.model.backbone.encoder_imaging.parameters())
            total_params += sum(1 for p in self.model.backbone.encoder_tabular.parameters())
            total_params += sum(1 for p in self.model.backbone.encoder_multimodal.parameters())
            print(f"   Frozen parameters: {total_frozen}/{total_params} in encoders")
        
        if not param_groups:
            # Fallback: all parameters (should not happen)
            param_groups = [{'params': self.model.parameters()}]
            print("⚠️  WARNING: No parameter groups found, using all parameters")
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.lr_finetune,
            weight_decay=getattr(self.hparams, 'weight_decay_finetune', 1e-5)
        )
        
        # Scheduler: Use same configurable scheduler as pretraining
        scheduler_type = getattr(self.hparams, 'scheduler', 'anneal')  # Default to 'anneal'
        
        if scheduler_type == 'anneal':
            # LinearWarmupCosineAnnealingLR: Best for training from scratch
            warmup_epochs = getattr(self.hparams, 'warmup_epochs', 10)
            max_epochs = getattr(self.hparams, 'max_epochs', 200)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs
            )
            print(f"✅ Using LinearWarmupCosineAnnealingLR: warmup={warmup_epochs} epochs, max={max_epochs} epochs")
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau: Adaptive reduction (fallback option)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=getattr(self.hparams, 'plateau_factor', 0.5),
                patience=getattr(self.hparams, 'plateau_patience', 10),
                min_lr=getattr(self.hparams, 'plateau_min_lr', 1e-6)
            )
            print(f"✅ Using ReduceLROnPlateau: factor={getattr(self.hparams, 'plateau_factor', 0.5)}, patience={getattr(self.hparams, 'plateau_patience', 10)}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'eval.val.loss',
                    "strict": False
                }
            }
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported: 'anneal', 'plateau'")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

