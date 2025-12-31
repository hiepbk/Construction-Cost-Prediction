'''
Online Regression Evaluator for Self-Supervised Learning
Evaluates pretrained representations by training a regression head on frozen features.
'''
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import inspect

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




class SSLOnlineEvaluatorClassification(Callback):
    """
    Online classification evaluator for self-supervised learning.
    
    Attaches a classification head to frozen pretrained features and trains it
    during pretraining to monitor representation quality.
    Used for traintest dataset where test samples have fake ground truth (0.0),
    so regression is not meaningful - only classification can be monitored.
    
    Example::
        online_eval = SSLOnlineEvaluatorClassification(
            z_dim=512,
            classification_head={
                'type': 'CountryAwareClassification',
                'loss_type': {'classification_ce': {'self_weight': 1.0, 'global_weight': 1.0}},
                'num_countries': 2,
                'n_hidden': 2048,
                'p': 0.2,
                'num_heads': 8
            }
        )
    """
    
    def __init__(
        self,
        z_dim: int,
        classification_head: Dict,  # Dict containing all head configuration (from hparams.classification_head)
        swav: bool = False,
        multimodal: bool = False,
        strategy: str = None,
    ):
        """
        Args:
            z_dim: Representation dimension (embedding size)
            classification_head: Dict containing all head configuration (from hparams.classification_head)
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
        if isinstance(classification_head, DictConfig):
            classification_head = OmegaConf.to_container(classification_head, resolve=True)
        
        if not isinstance(classification_head, dict):
            raise ValueError(f"classification_head must be dict or DictConfig, got {type(classification_head)}")
        
        if 'type' not in classification_head:
            raise ValueError("classification_head must contain 'type' key specifying head class name")
        
        self.classification_head_config = classification_head.copy()  # Store full config for checkpoint
        self.classification_head_class = classification_head['type']  # Store head class name for checkpoint
        
        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[nn.Module] = None  # Can be any head class
        self._recovered_callback_state: Optional[Dict[str, Any]] = None
        
        # Store validation losses from head for epoch-end aggregation
        self.val_losses = []  # List of loss_dicts from each validation batch
        
        # Track ALL validation samples by data_id (unique and deterministic)
        self.tracked_val_classification_preds = {}  # Dict: {data_id: predicted_class_id}
        self.tracked_val_classification_targets = {}  # Dict: {data_id: ground_truth_class_id}
        self.debug_printed_this_epoch = False  # Track if we've already printed debug info this epoch
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize the classification head and optimizer."""
        # Use classification_head_config directly (all parameters should be in it)
        head_config = self.classification_head_config.copy()
        
        # Preprocess loss_type: convert DictConfig to dict if needed
        if 'loss_type' in head_config and isinstance(head_config['loss_type'], DictConfig):
            head_config['loss_type'] = dict(head_config['loss_type'])
        
        self.online_evaluator = create_head(
            head_config=head_config,
            n_input=self.z_dim
        ).to(pl_module.device)
        
        # Print head architecture for debugging
        if trainer.global_rank == 0:  # Only print on main process
            print("\n" + "="*80)
            print("ONLINE CLASSIFICATION HEAD ARCHITECTURE:")
            print("="*80)
            print(self.online_evaluator)
            print("="*80 + "\n")
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.online_evaluator.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        
        # Initialize storage for validation losses (head already calculates them)
        self.val_losses = []
        
        # Load checkpoint state if available
        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            if "optimizer_state" in self._recovered_callback_state:
                self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])
    
    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Optional[Tensor], Optional[list], Optional[Tensor]]:
        """Extract inputs and targets from batch, handling different batch formats.
        
        Returns:
            x: Image input
            x_t: Tabular input (if available)
            data_ids: List of data_ids for this batch (if available)
            country: Country labels (if available) - for classification
        """
        data_ids = None  # Initialize data_ids
        country = None  # Initialize country
        if self.swav:
            x, y = batch
            x = x[0]
            x_t = None
            data_ids = None
            country = None
        elif self.multimodal and self.strategy == 'tip':
            # TIP multimodal batch: (imaging_views, tabular_views, labels, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country)
            # Must have exactly 9 elements
            if len(batch) != 9:
                raise ValueError(f"Expected batch size 9, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country)")
            x_i, _, contrastive_labels, x_orig, x_t_orig, target_log, y_original, data_ids, country = batch
            x = x_orig
            x_t = x_t_orig
        elif self.multimodal and self.strategy == 'comparison':
            x_i, _, y, x_orig = batch
            x = x_orig
            x_t = None
            data_ids = None
            country = None
        else:
            _, x, y = batch
            x_t = None
            data_ids = None
            country = None
        
        x = x.to(device)
        if x_t is not None:
            x_t = x_t.to(device)
        if country is not None:
            country = country.to(device)
        # data_ids is a list of strings, no need to move to device
        
        return x, x_t, data_ids, country
    
    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
        dataset=None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[list], Optional[Dict], Optional[Tensor]]:
        """
        Shared step for training and validation.
        Returns: (loss, loss_dict, data_ids, head_result, country)
        """
        with torch.no_grad():
            with set_training(pl_module, False):
                x, x_t, data_ids, country = self.to_device(batch, pl_module.device)
                
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
        
        # Get country labels from batch
        if country is None:
            raise ValueError("country must be in batch. Ensure dataset returns country in __getitem__.")
        
        # Forward pass through classification head with country (head handles everything internally)
        head_result = self.online_evaluator(
            representations,
            country_gt=country
        )
        
        # Extract results from head
        loss = head_result['loss']  # scalar - total weighted loss
        loss_dict = head_result['loss_dict']  # dict with all individual losses (for monitoring)
        
        return loss, loss_dict, data_ids, head_result, country
    
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int
    ) -> None:
        """Update classification head on training batch."""
        # Get dataset for target extraction if needed
        dataset = trainer.train_dataloader.dataset if hasattr(trainer.train_dataloader, 'dataset') else None
        
        loss, loss_dict, _, head_result, country = self.shared_step(
            pl_module, batch, dataset
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Log losses from head (already calculated internally) - no need for manual metric calculation
        pl_module.log("classification_online.train.loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log individual losses from head (automatically loop through all losses)
        for loss_name, loss_value in loss_dict.items():
            pl_module.log(f"classification_online.train.{loss_name}", loss_value, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize tracking at the start of validation epoch."""
        # Clear stored predictions for this epoch (tracked by data_id)
        self.tracked_val_classification_preds = {}
        self.tracked_val_classification_targets = {}
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
        """Evaluate classification head on validation batch."""
        dataset = trainer.val_dataloaders[dataloader_idx].dataset if hasattr(trainer.val_dataloaders[dataloader_idx], 'dataset') else None
        
        loss, loss_dict, data_ids, head_result, country = self.shared_step(
            pl_module, batch, dataset
        )
        
        # Track predictions for ALL samples in this batch using data_id from batch
        # data_ids comes directly from the batch, ensuring correct matching
        if data_ids is not None:
            # data_ids is a list/tuple of data_id strings for each sample in the batch
            for local_idx, data_id in enumerate(data_ids):
                if local_idx < len(country):
                    # Get selected_country (argmax indices, 0 or 1), which is already binary (0 or 1)
                    selected_country = head_result.get('selected_country', None)
                    if selected_country is not None and local_idx < len(selected_country):
                        # selected_country is from argmax, so it's already binary (0 or 1)
                        # Extract the value and ensure it's an integer
                        pred_class_raw = selected_country[local_idx].cpu().detach()
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
                                pred_class = int(classification_logits[local_idx].argmax().cpu().detach().item())
                            else:
                                raise ValueError(f"pred_class={pred_class} is not 0 or 1, and cannot fix without classification_logits")
                        
                        # Store the binary prediction (0 or 1)
                        self.tracked_val_classification_preds[data_id] = pred_class
                        self.tracked_val_classification_targets[data_id] = int(country[local_idx].cpu().detach().item())
        
        # Store loss_dict for epoch-end aggregation (head already calculated all losses)
        self.val_losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})
        
        # Log losses from head (already calculated) - PyTorch Lightning will aggregate automatically
        pl_module.log("classification_online.val.loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log individual losses from head (automatically loop through all losses)
        for loss_name, loss_value in loss_dict.items():
            pl_module.log(f"classification_online.val.{loss_name}", loss_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
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
        # Use raw losses (without self_weight) for evaluation/metrics
        classification_ce_value = torch.stack([loss_dict.get('classification_ce_raw', loss_dict.get('classification_ce', torch.tensor(0.0))) for loss_dict in self.val_losses if 'classification_ce' in loss_dict or 'classification_ce_raw' in loss_dict]).mean()
        
        # Convert to float for logging
        classification_ce_value = float(classification_ce_value.item())
        
        # Log aggregated metrics (already logged per-batch, but log here for clarity)
        pl_module.log("classification_online.val.classification_ce", classification_ce_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log with short name for progress bar (only show classification_ce - the primary metric)
        pl_module.log("val_classification_ce", classification_ce_value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        
        # Log ALL tracked classification predictions with data_id to WandB
        # CRITICAL: Only log from rank 0 to avoid duplicate/conflicting entries when same data_id appears on multiple GPUs
        # With sync_dist=False, each GPU would log independently, causing conflicts (e.g., GPU0 logs 0, GPU1 logs 1 for same data_id)
        # Solution: Only rank 0 logs, ensuring each data_id is logged exactly once per epoch
        if trainer.is_global_zero and len(self.tracked_val_classification_preds) > 0:
            logged_count = 0
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
                    pl_module.log(f"classification/{title}", float(pred_class_binary), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                    logged_count += 1
                else:
                    print(f"⚠️ Warning: data_id {data_id} not found in tracked targets. Available targets: {list(self.tracked_val_classification_targets.keys())}")
            
            if logged_count == 0:
                print(f"⚠️ Warning: No validation samples were logged. Available predictions: {list(self.tracked_val_classification_preds.keys())}")
        
        # Reset losses storage for next epoch
        self.val_losses = []
        
        # Note: tracked_val_classification_preds and tracked_val_classification_targets are cleared in on_validation_epoch_start
    
    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> dict:
        """Save classification head state."""
        return {
            "state_dict": self.online_evaluator.state_dict(),
            "classification_head_class": self.classification_head_class,  # Save head class name for loading
            "optimizer_state": self.optimizer.state_dict()
        }
    
    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]
    ) -> None:
        """Load classification head state."""
        self._recovered_callback_state = callback_state




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
        # Track classification predictions for WandB logging
        self.tracked_val_classification_preds = {}  # Dict: {data_id: predicted_class_id}
        self.tracked_val_classification_targets = {}  # Dict: {data_id: ground_truth_class_id}
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
        
        # Print head architecture for debugging
        if trainer.global_rank == 0:  # Only print on main process
            print("\n" + "="*80)
            print("ONLINE REGRESSION HEAD ARCHITECTURE:")
            print("="*80)
            print(self.online_evaluator)
            print("="*80 + "\n")
        
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
    
    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[list], Optional[Tensor]]:
        """Extract inputs and targets from batch, handling different batch formats.
        
        Returns:
            x: Image input
            target_log: Target in log space (log1p(cost), NOT normalized, from dataloader)
            x_t: Tabular input (if available)
            y_original: Original target in original scale (if available) - for RMSLE loss
            data_ids: List of data_ids for this batch (if available)
            country: Country labels (if available) - for multi-task head
        """
        data_ids = None  # Initialize data_ids
        y_original = None  # Initialize y_original
        country = None  # Initialize country
        target_log = None  # Initialize target_log
        if self.swav:
            x, y = batch
            x = x[0]
            x_t = None
            data_ids = None
            y_original = None
            country = None
            target_log = None
        elif self.multimodal and self.strategy == 'tip':
            # TIP multimodal batch: (imaging_views, tabular_views, labels, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country)
            # Must have exactly 9 elements
            if len(batch) != 9:
                raise ValueError(f"Expected batch size 9, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target_log, target_original, data_id, country)")
            x_i, _, contrastive_labels, x_orig, x_t_orig, target_log, y_original, data_ids, country = batch
            x = x_orig
            x_t = x_t_orig
        elif self.multimodal and self.strategy == 'comparison':
            x_i, _, y, x_orig = batch
            x = x_orig
            x_t = None
            data_ids = None
            y_original = None
            country = None
            target_log = None
        else:
            _, x, y = batch
            x_t = None
            data_ids = None
            y_original = None
            country = None
            target_log = None
        
        x = x.to(device)
        if target_log is not None:
            target_log = target_log.to(device)
        if x_t is not None:
            x_t = x_t.to(device)
        if y_original is not None:
            y_original = y_original.to(device)
        if country is not None:
            country = country.to(device)
        # data_ids is a list of strings, no need to move to device
        
        return x, target_log, x_t, y_original, data_ids, country
    
    
    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
        dataset=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Optional[list], Optional[Dict], Optional[Tensor]]:
        """
        Shared step for training and validation.
        Returns: (predictions_normalized, targets_normalized, predictions_denorm, targets_denorm, loss, loss_dict, data_ids, head_result, country)
        """
        with torch.no_grad():
            with set_training(pl_module, False):
                x, target_log, x_t, y_original, data_ids, country = self.to_device(batch, pl_module.device)
                
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
        
        # Forward pass through regression head with target_log, target_original, and country (head handles everything internally)
        # Dynamically build kwargs - only include parameters that have values (not None)
        # This allows the head to handle optional parameters correctly
        head_kwargs = {
            'target_original': y_original.float()  # Original scale target (required)
        }
        if target_log is not None:
            head_kwargs['target_log'] = target_log.float()  # Target in log space (log1p(cost), NOT normalized)
        if country is not None:
            head_kwargs['country_gt'] = country  # Country labels for multi-task head
        
        head_result = self.online_evaluator(
            representations,
            **head_kwargs
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
        if target_log is not None:
            targets_normalized = target_log.float()
        else:
            targets_normalized = None
        
        return predictions_normalized, targets_normalized, predictions_original_scale, targets_original_scale, loss, loss_dict, data_ids, head_result, country
    
    
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
        
        preds_norm, targets_norm, preds_denorm, targets_denorm, loss, loss_dict, _, head_result, country = self.shared_step(
            pl_module, batch, dataset
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Log losses from head (already calculated internally) - no need for manual metric calculation
        pl_module.log("regression_online.train.loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log individual losses from head (automatically loop through all losses)
        for loss_name, loss_value in loss_dict.items():
            pl_module.log(f"regression_online.train.{loss_name}", loss_value, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize tracking at the start of validation epoch."""
        # Clear stored predictions for this epoch (tracked by data_id)
        self.tracked_val_preds = {}
        self.tracked_val_targets = {}
        self.tracked_val_classification_preds = {}
        self.tracked_val_classification_targets = {}
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
        
        preds_norm, targets_norm, preds_denorm, targets_denorm, loss, loss_dict, data_ids, head_result, country = self.shared_step(
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
                    
                    # Track classification predictions (if multi-task head)
                    # Use selected_country (from argmax) which is already binary (0 or 1)
                    if 'classification_ce' in loss_dict and country is not None:
                        # IMPORTANT: Get selected_country (argmax indices, 0 or 1), NOT classification_probs (probabilities, 0.0-1.0)
                        selected_country = head_result.get('selected_country', None)
                        if selected_country is not None and local_idx < len(selected_country):
                            # selected_country is from argmax, so it's already binary (0 or 1)
                            # Extract the value and ensure it's an integer
                            pred_class_raw = selected_country[local_idx].cpu().detach()
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
                                    pred_class = int(classification_logits[local_idx].argmax().cpu().detach().item())
                                else:
                                    raise ValueError(f"pred_class={pred_class} is not 0 or 1, and cannot fix without classification_logits")
                            
                            # Store the binary prediction (0 or 1)
                            self.tracked_val_classification_preds[data_id] = pred_class
                            self.tracked_val_classification_targets[data_id] = int(country[local_idx].cpu().detach().item())
        
        # Store loss_dict for epoch-end aggregation (head already calculated all losses)
        self.val_losses.append({k: v.detach().cpu() for k, v in loss_dict.items()})
        
        # Log losses from head (already calculated) - PyTorch Lightning will aggregate automatically
        pl_module.log("regression_online.val.loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Log individual losses from head (automatically loop through all losses)
        for loss_name, loss_value in loss_dict.items():
            pl_module.log(f"regression_online.val.{loss_name}", loss_value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    
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
        # Use raw losses (without self_weight) for evaluation/metrics
        # Raw losses are stored as '{loss_name}_raw' in loss_dict
        mae_value = torch.stack([loss_dict.get('mae_raw', loss_dict.get('mae', torch.tensor(0.0))) for loss_dict in self.val_losses if 'mae' in loss_dict or 'mae_raw' in loss_dict]).mean()
        rmse_value = torch.stack([loss_dict.get('rmse_raw', loss_dict.get('rmse', torch.tensor(0.0))) for loss_dict in self.val_losses if 'rmse' in loss_dict or 'rmse_raw' in loss_dict]).mean()
        rmsle_value = torch.stack([loss_dict.get('rmsle_raw', loss_dict.get('rmsle', torch.tensor(0.0))) for loss_dict in self.val_losses if 'rmsle' in loss_dict or 'rmsle_raw' in loss_dict]).mean()
        
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
        # Log ALL tracked regression predictions with data_id to WandB
        # CRITICAL: Only log from rank 0 to avoid duplicate/conflicting entries when same data_id appears on multiple GPUs
        # With sync_dist=True, averaging across GPUs might be okay for regression, but to avoid confusion, only rank 0 logs
        if trainer.is_global_zero and len(self.tracked_val_preds) > 0:
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
                    # Use sync_dist=False since we're only logging from rank 0 (no need to sync)
                    pl_module.log(f"regression/{title}", final_prediction, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
                    logged_count += 1
                else:
                    print(f"⚠️ Warning: data_id {data_id} not found in tracked targets. Available targets: {list(self.tracked_val_targets.keys())}")
            
            if logged_count == 0:
                print(f"⚠️ Warning: No validation samples were logged. Available predictions: {list(self.tracked_val_preds.keys())}")
        
        # Log ALL tracked classification predictions with data_id to WandB (similar to regression)
        # CRITICAL: Only log from rank 0 to avoid duplicate/conflicting entries when same data_id appears on multiple GPUs
        # With sync_dist=False, each GPU would log independently, causing conflicts (e.g., GPU0 logs 0, GPU1 logs 1 for same data_id)
        # Solution: Only rank 0 logs, ensuring each data_id is logged exactly once per epoch
        if trainer.is_global_zero and len(self.tracked_val_classification_preds) > 0:
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
                    pl_module.log(f"classification/{title}", float(pred_class_binary), on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        
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

