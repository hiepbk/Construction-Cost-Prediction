'''
Fine-tuning module for Construction Cost Prediction
Adapts TIP's pretrained backbone for regression task with log-transformed targets.
'''
from typing import Tuple
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

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
        
        regression_head_class = getattr(hparams, 'regression_head_class', 'RegressionMLP')
        print(f"✅ ConstructionCostPrediction model created with head: {regression_head_class}")
        
        # Loss function: RMSLE (competition metric), Huber (robust to outliers), MAE, or MSE
        loss_type = getattr(hparams, 'regression_loss', 'huber')
        self.loss_type = loss_type  # Store for later checking
        if loss_type == 'rmsle':
            # RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
            # This directly optimizes the competition metric
            self.criterion = self._rmsle_loss
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(delta=getattr(hparams, 'huber_delta', 1.0))
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Options: 'rmsle', 'huber', 'mae', 'mse'")
        
        # Metrics (on original scale, like pretraining)
        self.mae_train = torchmetrics.MeanAbsoluteError()
        self.mae_val = torchmetrics.MeanAbsoluteError()
        self.mae_test = torchmetrics.MeanAbsoluteError()
        
        self.rmse_train = torchmetrics.MeanSquaredError()  # Will take sqrt later
        self.rmse_val = torchmetrics.MeanSquaredError()
        self.rmse_test = torchmetrics.MeanSquaredError()
        
        # RMSLE: Use MSE on log1p values, then take sqrt
        self.rmsle_train = torchmetrics.MeanSquaredError()
        self.rmsle_val = torchmetrics.MeanSquaredError()
        self.rmsle_test = torchmetrics.MeanSquaredError()
        
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
        
        print(f"Initialized ConstructionCostFinetuning")
        print(f"  Loss: {loss_type}")
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tuple of (image, tabular) or (image, tabular, mask)
        
        Returns:
            prediction: (B,) - Predicted cost in normalized log space
        """
        y_hat = self.model(x)
        return y_hat
    
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
        
        y_hat = self.forward(x)
        
        # For RMSLE loss, use original ground truth directly (not normalized/log-transformed)
        if self.loss_type == 'rmsle' and y_original is not None:
            loss = self.criterion(y_hat.squeeze(), y_original.squeeze())
        else:
            # For other losses (Huber, MAE, MSE), use normalized log-transformed targets
            loss = self.criterion(y_hat.squeeze(), y.squeeze())
        
        # Convert predictions and targets to original scale for metrics (like pretraining)
        y_hat_original = self.denormalize_target(y_hat.detach().squeeze())
        if y_original is not None:
            y_true_original = y_original.squeeze()
        else:
            # Fallback: convert from log space
            y_true_original = self.denormalize_target(y.detach().squeeze())
        
        # Ensure non-negative
        y_hat_original = torch.clamp(y_hat_original, min=0.0)
        y_true_original = torch.clamp(y_true_original, min=0.0)
        
        # Update metrics on original scale (USD/m²)
        self.mae_train(y_hat_original, y_true_original)
        self.rmse_train(y_hat_original, y_true_original)
        
        # RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
        log_pred = torch.log1p(y_hat_original)
        log_true = torch.log1p(y_true_original)
        self.rmsle_train(log_pred, log_true)
        
        # Log metrics (only compute when needed, avoid warnings)
        batch_size = y_hat_original.shape[0] if len(y_hat_original.shape) > 0 else 1
        self.log('eval.train.loss', loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('eval.train.mae', self.mae_train, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('eval.train.rmse', torch.sqrt(self.rmse_train.compute()), on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('eval.train.rmsle', torch.sqrt(self.rmsle_train.compute()), on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        # Primary metric for progress bar
        self.log('train_rmsle', torch.sqrt(self.rmsle_train.compute()), on_epoch=True, on_step=False, prog_bar=True, logger=False, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def training_epoch_end(self, _) -> None:
        """Reset metrics after training epoch"""
        self.mae_train.reset()
        self.rmse_train.reset()
        self.rmsle_train.reset()
    
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
        
        y_hat = self.forward(x)
        
        # For RMSLE loss, use original ground truth directly (not normalized/log-transformed)
        if self.loss_type == 'rmsle' and y_original is not None:
            loss = self.criterion(y_hat.squeeze(), y_original.squeeze())
        else:
            # For other losses (Huber, MAE, MSE), use normalized log-transformed targets
            loss = self.criterion(y_hat.squeeze(), y.squeeze())
        
        # Convert predictions and targets to original scale for metrics (like pretraining)
        y_hat_original = self.denormalize_target(y_hat.detach().squeeze())
        if y_original is not None:
            y_true_original = y_original.squeeze()
        else:
            # Fallback: convert from log space
            y_true_original = self.denormalize_target(y.detach().squeeze())
        
        # Ensure non-negative
        y_hat_original = torch.clamp(y_hat_original, min=0.0)
        y_true_original = torch.clamp(y_true_original, min=0.0)
        
        # Track predictions for WandB logging (like pretraining)
        if data_id is not None:
            for idx, did in enumerate(data_id):
                self.tracked_val_preds[str(did)] = float(y_hat_original[idx].cpu().detach())
                self.tracked_val_targets[str(did)] = float(y_true_original[idx].cpu().detach())
        
        # Update metrics on original scale (USD/m²)
        self.mae_val(y_hat_original, y_true_original)
        self.rmse_val(y_hat_original, y_true_original)
        
        # RMSLE: sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
        log_pred = torch.log1p(y_hat_original)
        log_true = torch.log1p(y_true_original)
        self.rmsle_val(log_pred, log_true)
        
        self.log('eval.val.loss', loss, on_epoch=True, on_step=False, sync_dist=True)
    
    def on_validation_epoch_start(self) -> None:
        """Clear tracked predictions at start of validation epoch"""
        self.tracked_val_preds = {}
        self.tracked_val_targets = {}
    
    def validation_epoch_end(self, _) -> None:
        """Compute validation metrics and log predictions to WandB (like pretraining)"""
        if self.trainer.sanity_checking:
            return
        
        mae_val = self.mae_val.compute()
        rmse_val = torch.sqrt(self.rmse_val.compute())
        rmsle_val = torch.sqrt(self.rmsle_val.compute())  # Primary metric
        
        # Get batch size from metrics (approximate) - use update count if available
        try:
            batch_size = self.mae_val._update_count if hasattr(self.mae_val, '_update_count') else 1
        except:
            batch_size = 1
        
        # Log metrics (like pretraining)
        self.log('eval.val.mae', mae_val, on_epoch=True, on_step=False, sync_dist=True, metric_attribute=self.mae_val, batch_size=batch_size)
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
        
        # Reset metrics and tracking for next epoch
        self.mae_val.reset()
        self.rmse_val.reset()
        self.rmsle_val.reset()
        self.tracked_val_preds = {}
        self.tracked_val_targets = {}
    
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
        
        y_hat = self.forward(x)
        
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
            regressor_lr = self.hparams.lr_eval * 10.0  # 10x for regression head
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
                    'lr': self.hparams.lr_eval
                })
                print(f"✅ Multimodal encoder: trainable, LR={self.hparams.lr_eval:.2e}")
            
            # Tabular encoder: base LR (or lower if freeze_tabular=True)
            tabular_params = list(self.model.backbone.encoder_tabular.parameters())
            if tabular_params and not getattr(self.hparams, 'freeze_tabular', False):
                tabular_lr = self.hparams.lr_eval * 0.1 if getattr(self.hparams, 'freeze_tabular', False) else self.hparams.lr_eval
                param_groups.append({
                    'params': tabular_params,
                    'lr': tabular_lr
                })
                print(f"✅ Tabular encoder: trainable, LR={tabular_lr:.2e}")
            
            # Image encoder: lower LR (or frozen)
            image_params = list(self.model.backbone.encoder_imaging.parameters())
            if image_params and not getattr(self.hparams, 'freeze_image', True):
                image_lr = self.hparams.lr_eval * 0.01 if getattr(self.hparams, 'freeze_image', True) else self.hparams.lr_eval * 0.1
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
            lr=self.hparams.lr_eval,
            weight_decay=getattr(self.hparams, 'weight_decay_eval', 1e-5)
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'eval.val.loss',
                "strict": False
            }
        }

