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
        # Track classification predictions for WandB logging
        self.tracked_val_classification_preds = {}  # Dict: {data_id: predicted_class_id}
        self.tracked_val_classification_targets = {}  # Dict: {data_id: ground_truth_class_id}
        
        # Target normalization parameters are stored in the head, not here
        # Get them from head for logging purposes only
        if hasattr(self.model.regression, 'target_mean'):
            target_mean = self.model.regression.target_mean
            target_std = self.model.regression.target_std
        else:
            target_mean = None
            target_std = None
        
        self.best_val_mae = float('inf')
        self.best_val_rmse = float('inf')
        self.best_val_rmsle = float('inf')
        self.best_val_score = float('inf')  # For compatibility with evaluate.py (will be set based on eval_metric)
        
        # Get loss config for printing (from head)
        loss_config = getattr(self.model.regression, 'loss_config', {'rmsle': 1.0})
        
        print(f"Initialized ConstructionCostFinetuning")
        print(f"  Loss config: {loss_config}")
        if target_mean is not None:
            print(f"  Target normalization: mean={target_mean:.2f}, std={target_std:.2f} (from head)")
        else:
            print(f"  ⚠️  Could not get target normalization from head")
    
    # Removed denormalize_target() and _rmsle_loss() methods
    # The head already handles all target normalization/denormalization and loss calculation internally
    
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
        # Unpack batch from ConstructionCostTIPDataset (always 9 elements with country)
        if len(batch) != 9:
            raise ValueError(f"Expected batch size 9, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country)")
        imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country = batch
        # Use unaugmented views for fine-tuning
        x = (unaugmented_image, unaugmented_tabular)
        target_log = target_log  # Target in log space (log1p(cost), NOT normalized, from dataloader)
        target_original = target_original  # Ground truth in original scale (USD/m², from dataloader)
        country_gt = country  # Country labels (0 or 1) for multi-task head
        
        # Forward pass through model (returns x_m from backbone)
        x_m = self.model.backbone(x, visualize=False)  # (B, 20, 512)
        
        # Forward through regression head with target_log, target_original and country (head handles everything internally)
        # Pass country_gt for multi-task head (if available)
        head_result = self.model.regression(
            x_m, 
            target_log=target_log if target_log is not None else None,  # Target in log space (log1p(cost), NOT normalized)
            target_original=target_original.float() if target_original is not None else None,  # Ground truth in original scale
            country_gt=country_gt  # Country labels for multi-task head
        )
        
        # Extract results from head
        y_hat = head_result['prediction_log']  # (B,) - normalized log space
        y_hat_original = head_result['prediction_original']  # (B,) - original scale
        loss = head_result['loss']  # scalar - total weighted loss (for backprop)
        loss_dict = head_result['loss_dict']  # dict with all individual losses
        
        # Get ground truth in original scale (already provided by dataloader)
        # No need to denormalize - head handles everything internally
        if target_original is not None:
            y_true_original = target_original.squeeze()
        else:
            # Fallback: target_original should always be provided by ConstructionCostTIPDataset
            y_true_original = None
        
        # Log losses from head (already calculated internally) - no need for manual metric calculation
        batch_size = y_hat_original.shape[0] if len(y_hat_original.shape) > 0 else 1
        self.log('finetune.train.loss', loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        # Log individual losses from head (automatically loop through all losses)
        for loss_name, loss_value in loss_dict.items():
            self.log(f'finetune.train.{loss_name}', loss_value, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
            # Also log RMSLE to prog_bar for quick monitoring
            if loss_name == 'rmsle':
                self.log('train_rmsle', loss_value, on_epoch=True, on_step=False, prog_bar=True, logger=False, sync_dist=True, batch_size=batch_size)
        
        # Log country classification accuracy (if multi-task head)
        if 'classification_ce' in loss_dict and country_gt is not None:
            # Get predicted country from classification logits
            classification_logits = head_result.get('classification_logits', None)
            if classification_logits is not None:
                country_pred = classification_logits.argmax(dim=1)  # (B,)
                country_acc = (country_pred == country_gt).float().mean()
                self.log('finetune.train.country_acc', country_acc, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def training_epoch_end(self, _) -> None:
        """Reset metrics after training epoch (not used for training, but kept for compatibility)"""
        # Training metrics are not calculated manually anymore (head handles losses)
        # But we keep this method for compatibility
        pass
    
    def validation_step(self, batch: Tuple, _) -> None:
        """Validation step"""
        # Unpack batch from ConstructionCostTIPDataset (always 9 elements with country)
        if len(batch) != 9:
            raise ValueError(f"Expected batch size 9, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country)")
        imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country = batch
        # Use unaugmented views for validation
        x = (unaugmented_image, unaugmented_tabular)
        target_log = target_log  # Target in log space (log1p(cost), NOT normalized, from dataloader)
        target_original = target_original  # Ground truth in original scale (USD/m², from dataloader)
        country_gt = country  # Country labels (0 or 1) for multi-task head
        
        # Forward pass through model (returns x_m from backbone)
        x_m = self.model.backbone(x, visualize=False)  # (B, 20, 512)
        
        # Forward through regression head with target_log, target_original and country (head handles everything internally)
        head_result = self.model.regression(
            x_m, 
            target_log=target_log if target_log is not None else None,  # Target in log space (log1p(cost), NOT normalized)
            target_original=target_original.float() if target_original is not None else None,  # Ground truth in original scale
            country_gt=country_gt  # Country labels for multi-task head
        )
        
        # Extract results from head
        y_hat = head_result['prediction_log']  # (B,) - normalized log space
        y_hat_original = head_result['prediction_original']  # (B,) - original scale
        loss = head_result['loss']  # scalar - total weighted loss
        loss_dict = head_result['loss_dict']  # dict with all individual losses
        
        # Safety check: Detect NaN/Inf and replace with safe values to prevent DDP hang
        if torch.any(torch.isnan(loss)) or torch.isinf(loss):
            # Replace with a large but finite value
            loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype)
            # Also fix loss_dict
            for k, v in loss_dict.items():
                if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                    loss_dict[k] = torch.tensor(1e6, device=v.device, dtype=v.dtype)
        
        # Check predictions for NaN/Inf
        if torch.any(torch.isnan(y_hat_original)) or torch.any(torch.isinf(y_hat_original)):
            # Replace with median of valid predictions or safe default
            valid_mask = torch.isfinite(y_hat_original)
            if torch.any(valid_mask):
                safe_value = torch.median(y_hat_original[valid_mask])
            else:
                safe_value = torch.tensor(1000.0, device=y_hat_original.device)  # Default: 1000 USD/m²
            y_hat_original = torch.where(torch.isfinite(y_hat_original), y_hat_original, safe_value)
        
        # Use target_original directly (already in original scale from dataloader)
        y_true_original = target_original.squeeze() if target_original is not None else None
        
        # Track predictions for WandB logging (like pretraining)
        if data_id is not None:
            for idx, did in enumerate(data_id):
                self.tracked_val_preds[str(did)] = float(y_hat_original[idx].cpu().detach())
                self.tracked_val_targets[str(did)] = float(y_true_original[idx].cpu().detach())
                
                # Track classification predictions (if multi-task head)
                # Use selected_country (from argmax) which is already binary (0 or 1)
                if 'classification_ce' in loss_dict and country_gt is not None:
                    # IMPORTANT: Get selected_country (argmax indices, 0 or 1), NOT classification_probs (probabilities, 0.0-1.0)
                    selected_country = head_result.get('selected_country', None)
                    if selected_country is not None and idx < len(selected_country):
                        # selected_country is from argmax, so it's already binary (0 or 1)
                        # Extract the value and ensure it's an integer
                        pred_class_raw = selected_country[idx].cpu().detach()
                        # Convert to Python int (0 or 1)
                        if pred_class_raw.dim() == 0:  # Scalar tensor
                            pred_class = int(pred_class_raw.item())
                        else:
                            pred_class = int(pred_class_raw[0].item())
                        
                        # CRITICAL: Ensure it's exactly 0 or 1 (not a probability like 0.5)
                        if pred_class not in [0, 1]:
                            # This should never happen if selected_country is from argmax
                            # But if it does, try to fix by recomputing argmax from logits
                            classification_logits = head_result.get('classification_logits', None)
                            if classification_logits is not None:
                                pred_class = int(classification_logits[idx].argmax().cpu().detach().item())
                            else:
                                raise ValueError(f"pred_class={pred_class} is not 0 or 1, and cannot fix without classification_logits")
                        
                        # Store the binary prediction (0 or 1)
                        self.tracked_val_classification_preds[str(did)] = pred_class
                        self.tracked_val_classification_targets[str(did)] = int(country_gt[idx].cpu().detach().item())
        
        # Store loss_dict for epoch-end aggregation (head already calculated all losses)
        self.val_losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})
        
        # Log losses from head (already calculated) - PyTorch Lightning will aggregate automatically
        batch_size = y_hat_original.shape[0] if len(y_hat_original.shape) > 0 else 1
        self.log('finetune.val.loss', loss, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        # Log individual losses from head (automatically loop through all losses)
        for loss_name, loss_value in loss_dict.items():
            self.log(f'finetune.val.{loss_name}', loss_value, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        # Log country classification accuracy (if multi-task head)
        # Log country classification accuracy (if multi-task head)
        if 'classification_ce' in loss_dict and country_gt is not None:
            # Get predicted country from classification logits
            classification_logits = head_result.get('classification_logits', None)
            if classification_logits is not None:
                country_pred = classification_logits.argmax(dim=1)  # (B,)
                country_acc = (country_pred == country_gt).float().mean()
                self.log('finetune.val.country_acc', country_acc, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
    
    def on_validation_epoch_start(self) -> None:
        """Clear tracked predictions and losses at start of validation epoch"""
        self.tracked_val_preds = {}
        self.tracked_val_targets = {}
        self.tracked_val_classification_preds = {}
        self.tracked_val_classification_targets = {}
        self.val_losses = []  # Reset for new epoch
    
    def validation_epoch_end(self, _) -> None:
        """Compute validation metrics and log predictions to WandB (like pretraining)"""
        if self.trainer.sanity_checking:
            return
        
        # Aggregate losses from all validation batches (head already calculated them)
        if len(self.val_losses) == 0:
            return
        
        # Compute mean of each loss across all batches
        # Use raw losses (without self_weight) for evaluation/metrics
        # Raw losses are stored as '{loss_name}_raw' in loss_dict
        mae_val = torch.stack([loss_dict.get('mae_raw', loss_dict.get('mae', torch.tensor(0.0))) for loss_dict in self.val_losses if 'mae' in loss_dict or 'mae_raw' in loss_dict]).mean()
        rmse_val = torch.stack([loss_dict.get('rmse_raw', loss_dict.get('rmse', torch.tensor(0.0))) for loss_dict in self.val_losses if 'rmse' in loss_dict or 'rmse_raw' in loss_dict]).mean()
        rmsle_val = torch.stack([loss_dict.get('rmsle_raw', loss_dict.get('rmsle', torch.tensor(0.0))) for loss_dict in self.val_losses if 'rmsle' in loss_dict or 'rmsle_raw' in loss_dict]).mean()
        
        # Safety check: Replace NaN/Inf with large but finite values
        if torch.isnan(mae_val) or torch.isinf(mae_val):
            mae_val = torch.tensor(1e6, device=mae_val.device)
        if torch.isnan(rmse_val) or torch.isinf(rmse_val):
            rmse_val = torch.tensor(1e6, device=rmse_val.device)
        if torch.isnan(rmsle_val) or torch.isinf(rmsle_val):
            rmsle_val = torch.tensor(1e6, device=rmsle_val.device)
        
        # Convert to float for logging
        mae_val = float(mae_val.item())
        rmse_val = float(rmse_val.item())
        rmsle_val = float(rmsle_val.item())
        
        # Log aggregated metrics (already logged per-batch, but log here for clarity and best checkpoint tracking)
        batch_size = 1  # Not needed for epoch-level logging
        self.log('finetune.val.mae', mae_val, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('finetune.val.rmse', rmse_val, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('finetune.val.rmsle', rmsle_val, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)  # Primary metric
        # Primary metric for progress bar
        self.log('val_rmsle', rmsle_val, on_epoch=True, on_step=False, prog_bar=True, logger=False, sync_dist=True, batch_size=batch_size)
        
        # Log ALL tracked sample predictions with data_id to WandB (like pretraining)
        # Log ALL tracked regression predictions with data_id to WandB
        # CRITICAL: Only log from rank 0 to avoid duplicate/conflicting entries when same data_id appears on multiple GPUs
        # With sync_dist=True, averaging across GPUs might be okay for regression, but to avoid confusion, only rank 0 logs
        if self.trainer.is_global_zero and len(self.tracked_val_preds) > 0:
            for data_id in self.tracked_val_preds.keys():
                if data_id in self.tracked_val_targets:
                    pred_val = self.tracked_val_preds[data_id]
                    target_val = self.tracked_val_targets[data_id]
                    
                    # Format: "data_id: {data_id}, {ground_truth_value:.2f} USD/m²"
                    title = f"data_id: {data_id}, {target_val:.2f} USD/m²"
                    
                    # Log prediction with data_id and ground truth in the title
                    # Group under "regression" chart group in WandB
                    # The logged value is the FINAL prediction in USD/m² (original scale)
                    # Use sync_dist=False since we're only logging from rank 0 (no need to sync)
                    self.log(f"regression/{title}", pred_val, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        
        # Log ALL tracked classification predictions with data_id to WandB (similar to regression)
        # CRITICAL: Only log from rank 0 to avoid duplicate/conflicting entries when same data_id appears on multiple GPUs
        # With sync_dist=False, each GPU would log independently, causing conflicts (e.g., GPU0 logs 0, GPU1 logs 1 for same data_id)
        # Solution: Only rank 0 logs, ensuring each data_id is logged exactly once per epoch
        if self.trainer.is_global_zero and len(self.tracked_val_classification_preds) > 0:
            for data_id in self.tracked_val_classification_preds.keys():
                if data_id in self.tracked_val_classification_targets:
                    pred_class = self.tracked_val_classification_preds[data_id]
                    target_class = self.tracked_val_classification_targets[data_id]
                    
                    # Format title: "data_id: {data_id}, class_id: {class_id}"
                    # Similar to regression format: "data_id: {data_id}, {ground_truth_value:.2f} USD/m²"
                    title = f"data_id: {data_id}, class_id: {target_class}"
                    
                    # Log prediction (0 or 1) with data_id and ground truth class_id in the title
                    # Group under "classification" chart group in WandB
                    # This creates line charts showing predictions over epochs (0 or 1, not probabilities)
                    # Ensure pred_class is binary (0 or 1)
                    pred_class_binary = int(pred_class)  # Ensure it's an integer
                    assert pred_class_binary in [0, 1], f"pred_class should be 0 or 1, got {pred_class_binary}"
                    # Use sync_dist=False since we're only logging from rank 0 (no need to sync)
                    self.log(f"classification/{title}", float(pred_class_binary), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        
        # Track best metrics (only update if value is finite and better)
        if not (mae_val == float('inf') or mae_val != mae_val) and mae_val < self.best_val_mae:
            self.best_val_mae = mae_val
        if not (rmse_val == float('inf') or rmse_val != rmse_val) and rmse_val < self.best_val_rmse:
            self.best_val_rmse = rmse_val
        if not (rmsle_val == float('inf') or rmsle_val != rmsle_val) and rmsle_val < self.best_val_rmsle:
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
        # Unpack batch from ConstructionCostTIPDataset (always 9 elements with country)
        if len(batch) != 9:
            raise ValueError(f"Expected batch size 9, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country)")
        imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country = batch
        # Use unaugmented views for testing
        x = (unaugmented_image, unaugmented_tabular)
        target_log = target_log  # Target in log space (log1p(cost), NOT normalized, may be dummy 0.0 for test set)
        country_gt = country  # Country labels (for multi-task head)
        
        # Forward pass through model (returns x_m from backbone)
        x_m = self.model.backbone(x, visualize=False)  # (B, 20, 512)
        
        # Forward through regression head (no targets for test, just predictions)
        head_result = self.model.regression(x_m, country_gt=country_gt)
        
        # Extract results from head
        y_hat = head_result['prediction_log']  # (B,) - normalized log space
        y_hat_original = head_result['prediction_original']  # (B,) - original scale
        
        # Metrics (in log space) - only if target_log is valid (not dummy)
        if target_log is not None and not (isinstance(target_log, torch.Tensor) and (target_log == 0.0).all()):
            y_hat_detached = y_hat.detach().squeeze()
            target_log_detached = target_log.squeeze()
            
            self.mae_test(y_hat_detached, target_log_detached)
            self.rmse_test(y_hat_detached, target_log_detached)
    
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
            # Reduced multiplier for better convergence (5x instead of 10x)
            # When training from scratch, lower LR is often better
            regressor_lr_multiplier = getattr(self.hparams, 'regressor_lr_multiplier', 5.0)
            regressor_lr = self.hparams.lr_finetune * regressor_lr_multiplier
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
                    "monitor": 'finetune.val.loss',
                    "strict": False
                }
            }
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported: 'anneal', 'plateau'")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

