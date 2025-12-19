'''
Regression Evaluator for Construction Cost Prediction
Adapts TIP's Evaluator for regression task with log-transformed targets.
'''
from typing import Tuple
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import numpy as np

from models.Tip_utils.Tip_downstream import TIPBackbone


class Evaluator_ConstructionCost(pl.LightningModule):
    """
    Evaluator for construction cost regression using TIP-pretrained backbone.
    
    Features:
    - Log-transform target (log(1 + cost))
    - Huber loss (robust to outliers)
    - MAE, RMSE, R² metrics
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
        
        # Create TIP backbone with regression head
        self.model = TIPBackbone(hparams)
        
        # Loss function: Huber (robust to outliers) or MAE
        loss_type = getattr(hparams, 'regression_loss', 'huber')
        if loss_type == 'huber':
            self.criterion = nn.HuberLoss(delta=getattr(hparams, 'huber_delta', 1.0))
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Metrics (in log space)
        self.mae_train = torchmetrics.MeanAbsoluteError()
        self.mae_val = torchmetrics.MeanAbsoluteError()
        self.mae_test = torchmetrics.MeanAbsoluteError()
        
        self.rmse_train = torchmetrics.MeanSquaredError()
        self.rmse_val = torchmetrics.MeanSquaredError()
        self.rmse_test = torchmetrics.MeanSquaredError()
        
        self.r2_train = torchmetrics.R2Score()
        self.r2_val = torchmetrics.R2Score()
        self.r2_test = torchmetrics.R2Score()
        
        # Target normalization parameters (for denormalization)
        self.target_mean = getattr(hparams, 'target_mean', 0.0)
        self.target_std = getattr(hparams, 'target_std', 1.0)
        self.target_log_transform = getattr(hparams, 'target_log_transform', True)
        
        self.best_val_mae = float('inf')
        self.best_val_rmse = float('inf')
        
        print(f"Initialized Evaluator_ConstructionCost")
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tuple of (image, tabular) or (image, tabular, mask)
        
        Returns:
            prediction: (B, 1) - Predicted cost in log space
        """
        y_hat = self.model(x)
        return y_hat
    
    def training_step(self, batch: Tuple, _) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular)
                  For regression, we use unaugmented views and target
        """
        # For regression, we need targets
        # Batch format from ConstructionCostTIPDataset:
        # (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular)
        # We'll need to modify dataset to also return targets
        
        # For now, assume batch is (x, y) where x = (image, tabular)
        if len(batch) == 2:
            x, y = batch
        else:
            # Extended format: extract target if available
            # This will be handled by custom dataset
            raise NotImplementedError("Need to handle extended batch format")
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        
        # Metrics (in log space)
        y_hat_detached = y_hat.detach().squeeze()
        y_detached = y.squeeze()
        
        self.mae_train(y_hat_detached, y_detached)
        self.rmse_train(y_hat_detached, y_detached)
        self.r2_train(y_hat_detached, y_detached)
        
        self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
        self.log('eval.train.mae', self.mae_train, on_epoch=True, on_step=False)
        self.log('eval.train.rmse', torch.sqrt(self.rmse_train.compute()), on_epoch=True, on_step=False)
        self.log('eval.train.r2', self.r2_train, on_epoch=True, on_step=False)
        
        return loss
    
    def training_epoch_end(self, _) -> None:
        """Reset metrics after training epoch"""
        self.rmse_train.reset()
        self.r2_train.reset()
    
    def validation_step(self, batch: Tuple, _) -> None:
        """Validation step"""
        if len(batch) == 2:
            x, y = batch
        else:
            raise NotImplementedError("Need to handle extended batch format")
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        
        # Metrics (in log space)
        y_hat_detached = y_hat.detach().squeeze()
        y_detached = y.squeeze()
        
        self.mae_val(y_hat_detached, y_detached)
        self.rmse_val(y_hat_detached, y_detached)
        self.r2_val(y_hat_detached, y_detached)
        
        self.log('eval.val.loss', loss, on_epoch=True, on_step=False)
    
    def validation_epoch_end(self, _) -> None:
        """Compute validation metrics"""
        if self.trainer.sanity_checking:
            return
        
        mae_val = self.mae_val.compute()
        rmse_val = torch.sqrt(self.rmse_val.compute())
        r2_val = self.r2_val.compute()
        
        self.log('eval.val.mae', mae_val, on_epoch=True, on_step=False, metric_attribute=self.mae_val)
        self.log('eval.val.rmse', rmse_val, on_epoch=True, on_step=False)
        self.log('eval.val.r2', r2_val, on_epoch=True, on_step=False, metric_attribute=self.r2_val)
        
        # Track best metrics
        if mae_val < self.best_val_mae:
            self.best_val_mae = mae_val
        if rmse_val < self.best_val_rmse:
            self.best_val_rmse = rmse_val
        
        self.mae_val.reset()
        self.rmse_val.reset()
        self.r2_val.reset()
    
    def test_step(self, batch: Tuple, _) -> None:
        """Test step"""
        if len(batch) == 2:
            x, y = batch
        else:
            raise NotImplementedError("Need to handle extended batch format")
        
        y_hat = self.forward(x)
        
        # Metrics (in log space)
        y_hat_detached = y_hat.detach().squeeze()
        y_detached = y.squeeze()
        
        self.mae_test(y_hat_detached, y_detached)
        self.rmse_test(y_hat_detached, y_detached)
        self.r2_test(y_hat_detached, y_detached)
    
    def test_epoch_end(self, _) -> None:
        """Compute test metrics"""
        mae_test = self.mae_test.compute()
        rmse_test = torch.sqrt(self.rmse_test.compute())
        r2_test = self.r2_test.compute()
        
        self.log('eval.test.mae', mae_test, metric_attribute=self.mae_test)
        self.log('eval.test.rmse', rmse_test)
        self.log('eval.test.r2', r2_test, metric_attribute=self.r2_test)
    
    def configure_optimizers(self):
        """
        Configure optimizer and scheduler for fine-tuning.
        """
        # Layer-wise learning rates
        param_groups = []
        
        # Regression head: higher learning rate
        regressor_params = list(self.model.classifier.parameters())
        if regressor_params:
            regressor_lr = self.hparams.lr_eval * 10.0  # 10x for regression head
            param_groups.append({
                'params': regressor_params,
                'lr': regressor_lr
            })
        
        # Multimodal encoder: base LR
        multimodal_params = list(self.model.encoder_multimodal.parameters())
        if multimodal_params:
            param_groups.append({
                'params': multimodal_params,
                'lr': self.hparams.lr_eval
            })
        
        # Tabular encoder: base LR (or lower if frozen)
        tabular_params = list(self.model.encoder_tabular.parameters())
        if tabular_params and getattr(self.hparams, 'finetune_strategy', 'trainable') == 'trainable':
            tabular_lr = self.hparams.lr_eval * 0.1 if getattr(self.hparams, 'freeze_tabular', False) else self.hparams.lr_eval
            param_groups.append({
                'params': tabular_params,
                'lr': tabular_lr
            })
        
        # Image encoder: lower LR (or frozen)
        image_params = list(self.model.encoder_imaging.parameters())
        if image_params and getattr(self.hparams, 'finetune_strategy', 'trainable') == 'trainable':
            image_lr = self.hparams.lr_eval * 0.01 if getattr(self.hparams, 'freeze_image', True) else self.hparams.lr_eval * 0.1
            param_groups.append({
                'params': image_params,
                'lr': image_lr
            })
        
        if not param_groups:
            # Fallback: all parameters
            param_groups = [{'params': self.model.parameters()}]
        
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

