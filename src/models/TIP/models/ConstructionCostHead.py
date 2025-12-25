"""
Construction Cost Regression Head Classes

This module defines regression head architectures for construction cost prediction.
All phases (pretrain, finetune, evaluation) should use heads from this module
to ensure consistency.

Available heads:
- RegressionMLP: Simple MLP with BatchNorm and Dropout
- AttentionAggregationRegression: Attention-based aggregation with learned token importance
# - MixtureOfExpertsRegression: Sparse MoE with Top-K routing (commented out - kept for future reference)
"""
from typing import Optional, Dict, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from omegaconf import DictConfig




class RegressionMLP(nn.Module):
    """
    Simple MLP for regression evaluation.
    Takes multimodal features (sequence of tokens) and predicts a scalar target.
    
    Architecture:
    - Input: (B, N, n_input) where N is sequence length (e.g., 20 tokens)
    - Aggregation: Mean pooling over sequence dimension -> (B, n_input)
    - Hidden 1: Linear(n_input, n_hidden) -> BatchNorm -> ReLU -> Dropout
    - Hidden 2: Linear(n_hidden, n_hidden//2) -> BatchNorm -> ReLU -> Dropout
    - Output: Linear(n_hidden//2, 1)
    
    Args:
        n_input: Input dimension (embedding size)
        n_hidden: Hidden dimension (default: n_input)
        p: Dropout probability (default: 0.2)
        target_mean: Mean for target normalization (default: 0.0)
        target_std: Std for target normalization (default: 1.0)
        target_log_transform: Whether target is log-transformed (default: True)
        loss_type: Loss function type ('rmsle', 'huber', 'mae', 'mse') (default: 'rmsle')
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_log_transform: bool = True,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        # IMPORTANT: Default to 2048 (matching old checkpoint architecture) for best results
        # Using n_input (512) as default gives poor results (~0.9 RMSLE)
        # Using 2048 gives much better results (can reach 0.18 RMSLE in pretrain, 0.12 in finetune)
        n_hidden = n_hidden if n_hidden is not None else 2048
        
        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_log_transform = target_log_transform
        self.huber_delta = huber_delta
        
        # Parse loss_type: REQUIRES new format with self_weight and global_weight
        # Format: {"rmsle": {"self_weight": 1.0, "global_weight": 0.2}, ...}
        # Convert DictConfig to dict if needed
        if isinstance(loss_type, DictConfig):
            loss_type = dict(loss_type)
        
        if not isinstance(loss_type, dict):
            raise ValueError(f"loss_type must be dict or DictConfig, got {type(loss_type)}")
        
        self.loss_config = {}
        self.loss_self_weights = {}  # self_weight for normalization
        self.loss_global_weights = {}  # global_weight for contribution to total
        
        for loss_name, loss_config in loss_type.items():
            if not isinstance(loss_config, dict):
                raise ValueError(
                    f"Loss '{loss_name}' config must be a dict with 'self_weight' and 'global_weight'. "
                    f"Got: {type(loss_config)}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            if 'self_weight' not in loss_config or 'global_weight' not in loss_config:
                raise ValueError(
                    f"Loss '{loss_name}' config must have both 'self_weight' and 'global_weight'. "
                    f"Got: {loss_config}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            self.loss_self_weights[loss_name] = float(loss_config['self_weight'])
            self.loss_global_weights[loss_name] = float(loss_config['global_weight'])
            self.loss_config[loss_name] = loss_config  # Store full config for reference
        
        # Normalize global weights to sum to 1.0 (optional, but good practice)
        total_global_weight = sum(self.loss_global_weights.values())
        if total_global_weight > 0:
            self.loss_global_weights = {k: v / total_global_weight for k, v in self.loss_global_weights.items()}
        
        # Store primary loss name for backward compatibility
        self.loss_type = list(self.loss_config.keys())[0] if self.loss_config else 'rmsle'
        
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
        
        # DISABLED: Custom weight initialization causes poor convergence
        # With _initialize_weights(): pretrain reaches ~0.6 RMSLE, finetune reaches ~0.3 RMSLE
        # Without (PyTorch defaults): pretrain reaches 0.18 RMSLE, finetune reaches 0.12 RMSLE
        # PyTorch's default initialization works much better for this task
        # self._initialize_weights()
        
        # Initialize loss functions for ALL possible losses (for monitoring)
        # We calculate all losses, but only weight those in loss_config
        self.loss_fns = {
            'huber': nn.HuberLoss(delta=huber_delta),
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'rmsle': None,  # RMSLE is computed manually
            'rmse': None,   # RMSE is computed from MSE (sqrt of MSE)
        }
        
        # Validate that all losses in config are supported
        for loss_name in self.loss_config.keys():
            if loss_name not in self.loss_fns:
                raise ValueError(f"Unknown loss type: {loss_name}. Supported: 'rmsle', 'huber', 'mae', 'mse', 'rmse'")
    
    # def _initialize_weights(self):
    #     """Initialize weights using Kaiming/He initialization for ReLU layers."""
    #     for module in self.mlp:
    #         if isinstance(module, nn.Linear):
    #             # Kaiming/He initialization for ReLU (recommended for ReLU activations)
    #             nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    #             if module.bias is not None:
    #                 # Initialize bias to small positive value to avoid dead ReLU
    #                 nn.init.constant_(module.bias, 0.01)
    #             # For output layer, initialize to predict near zero (will be normalized)
    #             if module == self.mlp[-1]:  # Last layer (output)
    #                 nn.init.normal_(module.weight, mean=0.0, std=0.01)
    #                 if module.bias is not None:
    #                     # Initialize output bias to predict near normalized mean (target_mean)
    #                     nn.init.constant_(module.bias, self.target_mean)
    #         elif isinstance(module, nn.BatchNorm1d):
    #             # BatchNorm: weight=1, bias=0 (standard)
    #             nn.init.constant_(module.weight, 1.0)
    #             nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        x: Tensor, 
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, N, n_input) - Multimodal features (sequence of tokens)
            target: Optional (B,) - Target in normalized log space (for loss calculation)
            target_original: Optional (B,) - Target in original scale USD/m² (for RMSLE loss)
        
        Returns:
            dict with keys:
                - 'prediction_log': (B,) - Prediction in normalized log space
                - 'prediction_original': (B,) - Prediction in original scale (USD/m²)
                - 'loss': (scalar) - Loss value (if targets provided)
                - 'loss_dict': dict - Dictionary of loss components (if targets provided)
        """
        # (B, 20, 512) -> (B, 512)
        # Aggregate all tokens using mean pooling
        if x.dim() == 3:
            # x is (B, N, n_input) - aggregate over sequence dimension
            x = x.mean(dim=1)  # (B, N, n_input) -> (B, n_input)
        # # If x is already (B, n_input), pass through directly
        
        # x = x[:, 0, :]
        
        # Pass through MLP to get prediction in normalized log space
        prediction_log = self.mlp(x).squeeze(-1)  # (B, 1) -> (B,)
        
        # Decode to original scale
        prediction_original = self.construction_cost_decode(prediction_log)
        
        # Prepare result dict
        result = {
            'prediction_log': prediction_log,
            'prediction_original': prediction_original,
        }
        
        # Calculate loss if targets are provided
        if target is not None or target_original is not None:
            loss_dict = self.calculate_loss(
                prediction_log=prediction_log,
                prediction_original=prediction_original,
                target=target,
                target_original=target_original
            )
            result['loss'] = loss_dict['total']
            result['loss_dict'] = loss_dict
        
        return result
    
    def construction_cost_decode(self, prediction_log: Tensor) -> Tensor:
        """
        Decode prediction from normalized log space to original scale (USD/m²).
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
        
        Returns:
            (B,) - Prediction in original scale (USD/m²), clamped to >= 0
        """
        # Step 1: Denormalize (reverse normalization)
        pred_log = prediction_log * self.target_std + self.target_mean
        
        # Step 2: Reverse log-transform to get original scale
        if self.target_log_transform:
            pred_original = torch.expm1(pred_log)  # exp(x) - 1
        else:
            pred_original = pred_log
        
        # Ensure non-negative (construction cost can't be negative)
        pred_original = torch.clamp(pred_original, min=0.0)
        
        return pred_original
    
    def calculate_loss(
        self,
        prediction_log: Tensor,
        prediction_original: Tensor,
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Calculate ALL losses internally, but only weight those in loss_config.
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
            prediction_original: (B,) - Prediction in original scale (USD/m²)
            target: Optional (B,) - Target in normalized log space (not used, kept for compatibility)
            target_original: Optional (B,) - Target in original scale (USD/m²)
        
        Returns:
            dict with keys:
                - 'total': (scalar) - Total weighted loss (for backprop, only losses in loss_config)
                - 'rmsle': (scalar) - RMSLE loss (always calculated)
                - 'huber': (scalar) - Huber loss (always calculated)
                - 'mae': (scalar) - MAE loss (always calculated)
                - 'mse': (scalar) - MSE loss (always calculated)
                - 'rmse': (scalar) - RMSE loss (always calculated, sqrt of MSE)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Ensure target_original is available (required for all losses)
        if target_original is None:
            raise ValueError("target_original is required for loss calculation")
        
        # Ensure non-negative
        pred_original = torch.clamp(prediction_original, min=0.0)
        tgt_original = torch.clamp(target_original, min=0.0)
        
        # Calculate ALL losses (for monitoring)
        
        # 1. RMSLE: sqrt(mean((log1p(y_true_orig) - log1p(y_pred_orig))^2))
        log_pred = torch.log1p(pred_original)  # log(1 + y_pred)
        log_true = torch.log1p(tgt_original)   # log(1 + y_true)
        squared_log_error = (log_true - log_pred) ** 2
        rmsle = torch.sqrt(torch.mean(squared_log_error))
        loss_dict['rmsle'] = rmsle
        
        # 2. Huber loss on original scale
        huber = self.loss_fns['huber'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['huber'] = huber
        
        # 3. MAE (L1) loss on original scale
        mae = self.loss_fns['mae'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mae'] = mae
        
        # 4. MSE (L2) loss on original scale
        mse = self.loss_fns['mse'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mse'] = mse
        
        # Calculate RMSE from MSE (for monitoring)
        rmse = torch.sqrt(mse)
        loss_dict['rmse'] = rmse
        
        # Calculate total weighted loss (only losses in loss_config contribute)
        # Formula: normalized_loss = self_weight * raw_loss, then contribution = global_weight * normalized_loss
        raw_losses = {}  # Store raw losses for evaluation/metrics
        
        for loss_name in self.loss_config.keys():
            if loss_name in loss_dict:
                raw_loss = loss_dict[loss_name]
                self_weight = self.loss_self_weights.get(loss_name, 1.0)
                global_weight = self.loss_global_weights.get(loss_name, 1.0)
                normalized_loss = self_weight * raw_loss
                contribution = global_weight * normalized_loss
                
                # Store raw loss for evaluation/metrics (original value without self_weight)
                raw_losses[f'{loss_name}_raw'] = raw_loss
                
                # Update loss_dict to store normalized loss (after self_weight) for wandb logging
                loss_dict[loss_name] = normalized_loss
                
                total_loss = total_loss + contribution
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
        # Add raw losses to loss_dict for evaluation/metrics
        loss_dict.update(raw_losses)
        
        # Store total weighted loss
        loss_dict['total'] = total_loss
        
        return loss_dict




class RegressionMLPTest(nn.Module):
    """
    Simple MLP for regression evaluation.
    Takes multimodal features (sequence of tokens) and predicts a scalar target.
    
    Architecture:
    - Input: (B, N, n_input) where N is sequence length (e.g., 20 tokens)
    - Aggregation: Mean pooling over sequence dimension -> (B, n_input)
    - Hidden 1: Linear(n_input, n_hidden) -> BatchNorm -> ReLU -> Dropout
    - Hidden 2: Linear(n_hidden, n_hidden//2) -> BatchNorm -> ReLU -> Dropout
    - Output: Linear(n_hidden//2, 1)
    
    Args:
        n_input: Input dimension (embedding size)
        n_hidden: Hidden dimension (default: n_input)
        p: Dropout probability (default: 0.2)
        target_mean: Mean for target normalization (default: 0.0)
        target_std: Std for target normalization (default: 1.0)
        target_log_transform: Whether target is log-transformed (default: True)
        loss_type: Loss function type ('rmsle', 'huber', 'mae', 'mse') (default: 'rmsle')
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_log_transform: bool = True,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        # IMPORTANT: Default to 2048 (matching old checkpoint architecture) for best results
        # Using n_input (512) as default gives poor results (~0.9 RMSLE)
        # Using 2048 gives much better results (can reach 0.18 RMSLE in pretrain, 0.12 in finetune)
        n_hidden = n_hidden if n_hidden is not None else 2048
        
        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_log_transform = target_log_transform
        self.huber_delta = huber_delta
        
        # Parse loss_type: REQUIRES new format with self_weight and global_weight
        # Format: {"rmsle": {"self_weight": 1.0, "global_weight": 0.2}, ...}
        # Convert DictConfig to dict if needed
        if isinstance(loss_type, DictConfig):
            loss_type = dict(loss_type)
        
        if not isinstance(loss_type, dict):
            raise ValueError(f"loss_type must be dict or DictConfig, got {type(loss_type)}")
        
        self.loss_config = {}
        self.loss_self_weights = {}  # self_weight for normalization
        self.loss_global_weights = {}  # global_weight for contribution to total
        
        for loss_name, loss_config in loss_type.items():
            if not isinstance(loss_config, dict):
                raise ValueError(
                    f"Loss '{loss_name}' config must be a dict with 'self_weight' and 'global_weight'. "
                    f"Got: {type(loss_config)}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            if 'self_weight' not in loss_config or 'global_weight' not in loss_config:
                raise ValueError(
                    f"Loss '{loss_name}' config must have both 'self_weight' and 'global_weight'. "
                    f"Got: {loss_config}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            self.loss_self_weights[loss_name] = float(loss_config['self_weight'])
            self.loss_global_weights[loss_name] = float(loss_config['global_weight'])
            self.loss_config[loss_name] = loss_config  # Store full config for reference
        
        # Normalize global weights to sum to 1.0 (optional, but good practice)
        total_global_weight = sum(self.loss_global_weights.values())
        if total_global_weight > 0:
            self.loss_global_weights = {k: v / total_global_weight for k, v in self.loss_global_weights.items()}
        
        # Store primary loss name for backward compatibility
        self.loss_type = list(self.loss_config.keys())[0] if self.loss_config else 'rmsle'
        
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
        
        # DISABLED: Custom weight initialization causes poor convergence
        # With _initialize_weights(): pretrain reaches ~0.6 RMSLE, finetune reaches ~0.3 RMSLE
        # Without (PyTorch defaults): pretrain reaches 0.18 RMSLE, finetune reaches 0.12 RMSLE
        # PyTorch's default initialization works much better for this task
        # self._initialize_weights()
        
        # Initialize loss functions for ALL possible losses (for monitoring)
        # We calculate all losses, but only weight those in loss_config
        self.loss_fns = {
            'huber': nn.HuberLoss(delta=huber_delta),
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'rmsle': None,  # RMSLE is computed manually
            'rmse': None,   # RMSE is computed from MSE (sqrt of MSE)
        }
        
        # Validate that all losses in config are supported
        for loss_name in self.loss_config.keys():
            if loss_name not in self.loss_fns:
                raise ValueError(f"Unknown loss type: {loss_name}. Supported: 'rmsle', 'huber', 'mae', 'mse', 'rmse'")
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming/He initialization for ReLU layers."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # Kaiming/He initialization for ReLU (recommended for ReLU activations)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    # Initialize bias to small positive value to avoid dead ReLU
                    nn.init.constant_(module.bias, 0.01)
                # For output layer, initialize to predict near zero (will be normalized)
                if module == self.mlp[-1]:  # Last layer (output)
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        # Initialize output bias to predict near normalized mean (target_mean)
                        nn.init.constant_(module.bias, self.target_mean)
            elif isinstance(module, nn.BatchNorm1d):
                # BatchNorm: weight=1, bias=0 (standard)
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        x: Tensor, 
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, N, n_input) - Multimodal features (sequence of tokens)
            target: Optional (B,) - Target in normalized log space (for loss calculation)
            target_original: Optional (B,) - Target in original scale USD/m² (for RMSLE loss)
        
        Returns:
            dict with keys:
                - 'prediction_log': (B,) - Prediction in normalized log space
                - 'prediction_original': (B,) - Prediction in original scale (USD/m²)
                - 'loss': (scalar) - Loss value (if targets provided)
                - 'loss_dict': dict - Dictionary of loss components (if targets provided)
        """
        # (B, 20, 512) -> (B, 512)
        # Aggregate all tokens using mean pooling
        if x.dim() == 3:
            # x is (B, N, n_input) - aggregate over sequence dimension
            x = x.mean(dim=1)  # (B, N, n_input) -> (B, n_input)
        # If x is already (B, n_input), pass through directly
        
        # Pass through MLP to get prediction in normalized log space
        prediction_log = self.mlp(x).squeeze(-1)  # (B, 1) -> (B,)
        
        # Decode to original scale
        prediction_original = self.construction_cost_decode(prediction_log)
        
        # Prepare result dict
        result = {
            'prediction_log': prediction_log,
            'prediction_original': prediction_original,
        }
        
        # Calculate loss if targets are provided
        if target is not None or target_original is not None:
            loss_dict = self.calculate_loss(
                prediction_log=prediction_log,
                prediction_original=prediction_original,
                target=target,
                target_original=target_original
            )
            result['loss'] = loss_dict['total']
            result['loss_dict'] = loss_dict
        
        return result
    
    def construction_cost_decode(self, prediction_log: Tensor) -> Tensor:
        """
        Decode prediction from normalized log space to original scale (USD/m²).
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
        
        Returns:
            (B,) - Prediction in original scale (USD/m²), clamped to >= 0
        """
        # Step 1: Denormalize (reverse normalization)
        pred_log = prediction_log * self.target_std + self.target_mean
        
        # Step 2: Reverse log-transform to get original scale
        if self.target_log_transform:
            pred_original = torch.expm1(pred_log)  # exp(x) - 1
        else:
            pred_original = pred_log
        
        # Ensure non-negative (construction cost can't be negative)
        pred_original = torch.clamp(pred_original, min=0.0)
        
        return pred_original
    
    def calculate_loss(
        self,
        prediction_log: Tensor,
        prediction_original: Tensor,
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Calculate ALL losses internally, but only weight those in loss_config.
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
            prediction_original: (B,) - Prediction in original scale (USD/m²)
            target: Optional (B,) - Target in normalized log space (not used, kept for compatibility)
            target_original: Optional (B,) - Target in original scale (USD/m²)
        
        Returns:
            dict with keys:
                - 'total': (scalar) - Total weighted loss (for backprop, only losses in loss_config)
                - 'rmsle': (scalar) - RMSLE loss (always calculated)
                - 'huber': (scalar) - Huber loss (always calculated)
                - 'mae': (scalar) - MAE loss (always calculated)
                - 'mse': (scalar) - MSE loss (always calculated)
                - 'rmse': (scalar) - RMSE loss (always calculated, sqrt of MSE)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Ensure target_original is available (required for all losses)
        if target_original is None:
            raise ValueError("target_original is required for loss calculation")
        
        # Ensure non-negative
        pred_original = torch.clamp(prediction_original, min=0.0)
        tgt_original = torch.clamp(target_original, min=0.0)
        
        # Calculate ALL losses (for monitoring)
        
        # 1. RMSLE: sqrt(mean((log1p(y_true_orig) - log1p(y_pred_orig))^2))
        log_pred = torch.log1p(pred_original)  # log(1 + y_pred)
        log_true = torch.log1p(tgt_original)   # log(1 + y_true)
        squared_log_error = (log_true - log_pred) ** 2
        rmsle = torch.sqrt(torch.mean(squared_log_error))
        loss_dict['rmsle'] = rmsle
        
        # 2. Huber loss on original scale
        huber = self.loss_fns['huber'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['huber'] = huber
        
        # 3. MAE (L1) loss on original scale
        mae = self.loss_fns['mae'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mae'] = mae
        
        # 4. MSE (L2) loss on original scale
        mse = self.loss_fns['mse'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mse'] = mse
        
        # Calculate RMSE from MSE (for monitoring)
        rmse = torch.sqrt(mse)
        loss_dict['rmse'] = rmse
        
        # Calculate total weighted loss (only losses in loss_config contribute)
        # Formula: normalized_loss = self_weight * raw_loss, then contribution = global_weight * normalized_loss
        raw_losses = {}  # Store raw losses for evaluation/metrics
        
        for loss_name in self.loss_config.keys():
            if loss_name in loss_dict:
                raw_loss = loss_dict[loss_name]
                self_weight = self.loss_self_weights.get(loss_name, 1.0)
                global_weight = self.loss_global_weights.get(loss_name, 1.0)
                normalized_loss = self_weight * raw_loss
                contribution = global_weight * normalized_loss
                
                # Store raw loss for evaluation/metrics (original value without self_weight)
                raw_losses[f'{loss_name}_raw'] = raw_loss
                
                # Update loss_dict to store normalized loss (after self_weight) for wandb logging
                loss_dict[loss_name] = normalized_loss
                
                total_loss = total_loss + contribution
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
        # Add raw losses to loss_dict for evaluation/metrics
        loss_dict.update(raw_losses)
        
        # Store total weighted loss
        loss_dict['total'] = total_loss
        
        return loss_dict


class MixtureOfExpertsRegression(nn.Module):
    """
    Mixture of Experts (MoE) regression head with Top-K routing.
    Automatically discovers multiple patterns in the data by routing samples
    to specialized expert networks.
    
    Architecture:
    - Input: (B, N, n_input) where N is sequence length (e.g., 20 tokens)
    - Aggregation: Mean pooling over sequence dimension -> (B, n_input)
    - Gating Network: Linear(n_input -> n_input//2) -> ReLU -> Linear(n_input//2 -> num_experts) -> Softmax
    - Top-K Routing: Select top K experts, zero others, renormalize
    - K Expert MLPs: Each MLP(n_input -> n_hidden -> n_hidden//2 -> 1)
    - Weighted Sum: Σ(weight_i * expert_i)
    
    Args:
        n_input: Input dimension (embedding size)
        loss_type: Dict[str, float] - Loss names and weights (e.g., {"rmsle": 0.7, "mae": 0.2, "rmse": 0.1})
        num_experts: Number of expert networks (default: 8)
        top_k: Number of experts to route to per sample (default: 2)
        n_hidden: Hidden dimension for experts (default: n_input)
        p: Dropout probability (default: 0.2)
        target_mean: Mean for target normalization (default: 0.0)
        target_std: Std for target normalization (default: 1.0)
        target_log_transform: Whether target is log-transformed (default: True)
        huber_delta: Delta parameter for Huber loss (default: 1.0)
        use_load_balancing: Whether to add load balancing loss (default: False)
        load_balancing_weight: Weight for load balancing loss (default: 0.01)
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        num_experts: int = 8,
        top_k: int = 2,
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_log_transform: bool = True,
        huber_delta: float = 1.0,
        use_load_balancing: bool = False,
        load_balancing_weight: float = 0.01,
    ):
        super().__init__()
        # IMPORTANT: Default to 2048 (matching old checkpoint architecture) for best results
        # Using n_input (512) as default gives poor results (~0.9 RMSLE)
        # Using 2048 gives much better results (can reach 0.18 RMSLE in pretrain, 0.12 in finetune)
        n_hidden = n_hidden if n_hidden is not None else 2048
        
        # Validate top_k
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
        if top_k < 1:
            raise ValueError(f"top_k ({top_k}) must be >= 1")
        
        # Store MoE-specific parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = load_balancing_weight
        
        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_log_transform = target_log_transform
        self.huber_delta = huber_delta
        
        # loss_type must be dict (multi-loss with weights) - no string support
        # Convert DictConfig to regular dict if needed
        if isinstance(loss_type, DictConfig):
            loss_type = dict(loss_type)
        
        # Now loss_type must be a regular dict
        if not isinstance(loss_type, dict):
            raise ValueError(f"loss_type must be dict or DictConfig, got {type(loss_type)}. Example: {{'rmsle': 0.7, 'mae': 0.2, 'rmse': 0.1}}")
        
        # Store loss config (which losses to use and their weights)
        self.loss_config = loss_type.copy()
        
        # Normalize weights to sum to 1.0 (optional, but good practice)
        total_weight = sum(self.loss_config.values())
        if total_weight > 0:
            self.loss_config = {k: v / total_weight for k, v in self.loss_config.items()}
        
        # Store primary loss name for backward compatibility
        self.loss_type = list(self.loss_config.keys())[0] if self.loss_config else 'rmsle'
        
        # Gating network: learns to route samples to experts
        self.gating = nn.Sequential(
            nn.Linear(n_input, n_input // 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_input // 2, num_experts),
            # Softmax will be applied in forward pass
        )
        
        # Expert networks: each specializes in different patterns
        self.experts = nn.ModuleList([
            nn.Sequential(
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
            for _ in range(num_experts)
        ])
        
        # DISABLED: Custom weight initialization causes poor convergence
        # With _initialize_weights(): pretrain reaches ~0.6 RMSLE, finetune reaches ~0.3 RMSLE
        # Without (PyTorch defaults): pretrain reaches 0.18 RMSLE, finetune reaches 0.12 RMSLE
        # PyTorch's default initialization works much better for this task
        # self._initialize_weights()
        
        # Initialize loss functions for ALL possible losses (for monitoring)
        # We calculate all losses, but only weight those in loss_config
        self.loss_fns = {
            'huber': nn.HuberLoss(delta=huber_delta),
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'rmsle': None,  # RMSLE is computed manually
            'rmse': None,   # RMSE is computed from MSE (sqrt of MSE)
        }
        
        # Validate that all losses in config are supported
        for loss_name in self.loss_config.keys():
            if loss_name not in self.loss_fns:
                raise ValueError(f"Unknown loss type: {loss_name}. Supported: 'rmsle', 'huber', 'mae', 'mse', 'rmse'")
    
    # def _initialize_weights(self):
    #     """Initialize weights using Kaiming/He initialization for ReLU layers."""
    #     # Initialize gating network
    #     for module in self.gating:
    #         if isinstance(module, nn.Linear):
    #             # Kaiming/He initialization for ReLU
    #             nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0.01)
        
    #     # Initialize expert networks
    #     for expert in self.experts:
    #         for module in expert:
    #             if isinstance(module, nn.Linear):
    #                 # Kaiming/He initialization for ReLU
    #                 nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    #                 if module.bias is not None:
    #                     nn.init.constant_(module.bias, 0.01)
    #                 # For output layer, initialize to predict near zero
    #                 if module == expert[-1]:  # Last layer (output)
    #                     nn.init.normal_(module.weight, mean=0.0, std=0.01)
    #                     if module.bias is not None:
    #                         # Initialize output bias to predict near normalized mean
    #                         nn.init.constant_(module.bias, self.target_mean)
    #             elif isinstance(module, nn.BatchNorm1d):
    #                 # BatchNorm: weight=1, bias=0 (standard)
    #                 nn.init.constant_(module.weight, 1.0)
    #                 nn.init.constant_(module.bias, 0.0)
    
    def _top_k_routing(self, weights: Tensor) -> Tensor:
        """
        Top-K routing: select top K experts, zero others, renormalize.
        Uses straight-through estimator for differentiability.
        
        Args:
            weights: (B, num_experts) - Gating network output (before softmax)
        
        Returns:
            (B, num_experts) - Sparse weights (only top K non-zero, normalized)
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(weights, dim=-1)  # (B, num_experts)
        
        # Get top K indices and values
        topk_values, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)  # (B, K)
        
        # Create sparse weights: zero out non-top-K, keep top-K
        sparse_weights = torch.zeros_like(probs)
        sparse_weights.scatter_(-1, topk_indices, topk_values)
        
        # Renormalize to ensure sum = 1
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return sparse_weights
    
    def _load_balancing_loss(self, weights: Tensor) -> Tensor:
        """
        Calculate load balancing loss to encourage uniform expert usage.
        
        Args:
            weights: (B, num_experts) - Routing weights (after top-K)
        
        Returns:
            (scalar) - Load balancing loss (variance of expert usage)
        """
        # Average usage per expert across batch
        expert_usage = weights.mean(dim=0)  # (num_experts,)
        
        # Target: uniform distribution (1/num_experts per expert)
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # Loss: encourage uniform usage (minimize variance)
        # Use KL divergence or MSE between actual and uniform
        load_balance_loss = torch.mean((expert_usage - target_usage) ** 2)
        
        return load_balance_loss
    
    def forward(
        self, 
        x: Tensor, 
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, N, n_input) - Multimodal features (sequence of tokens)
            target: Optional (B,) - Target in normalized log space (for loss calculation)
            target_original: Optional (B,) - Target in original scale USD/m² (for RMSLE loss)
        
        Returns:
            dict with keys:
                - 'prediction_log': (B,) - Prediction in normalized log space
                - 'prediction_original': (B,) - Prediction in original scale (USD/m²)
                - 'loss': (scalar) - Loss value (if targets provided)
                - 'loss_dict': dict - Dictionary of loss components (if targets provided)
                - 'expert_weights': (B, num_experts) - Routing weights (if targets provided, for monitoring)
        """
        # (B, 20, 512) -> (B, 512)
        # Aggregate all tokens using mean pooling
        if x.dim() == 3:
            # x is (B, N, n_input) - aggregate over sequence dimension
            x_pooled = x.mean(dim=1)  # (B, N, n_input) -> (B, n_input)
        else:
            # If x is already (B, n_input), pass through directly
            x_pooled = x
        
        # Gating network: compute routing weights
        gating_logits = self.gating(x_pooled)  # (B, num_experts)
        
        # Top-K routing: select top K experts
        expert_weights = self._top_k_routing(gating_logits)  # (B, num_experts)
        
        # Process through all experts (in parallel)
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x_pooled)  # (B, 1)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: (B, num_experts, 1)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, 1)
        
        # Weighted sum: Σ(weight_i * expert_i)
        # expert_weights: (B, num_experts), expert_outputs: (B, num_experts, 1)
        prediction_log = (expert_weights.unsqueeze(-1) * expert_outputs).sum(dim=1).squeeze(-1)  # (B,)
        
        # Decode to original scale
        prediction_original = self.construction_cost_decode(prediction_log)
        
        # Prepare result dict
        result = {
            'prediction_log': prediction_log,
            'prediction_original': prediction_original,
        }
        
        # Calculate loss if targets are provided
        if target is not None or target_original is not None:
            loss_dict = self.calculate_loss(
                prediction_log=prediction_log,
                prediction_original=prediction_original,
                target=target,
                target_original=target_original
            )
            
            # Add load balancing loss if enabled
            if self.use_load_balancing:
                load_balance_loss = self._load_balancing_loss(expert_weights)
                loss_dict['load_balancing'] = load_balance_loss
                # Add to total loss (with weight)
                loss_dict['total'] = loss_dict['total'] + self.load_balancing_weight * load_balance_loss
            
            result['loss'] = loss_dict['total']
            result['loss_dict'] = loss_dict
            result['expert_weights'] = expert_weights  # For monitoring
        
        return result
    
    def construction_cost_decode(self, prediction_log: Tensor) -> Tensor:
        """
        Decode prediction from normalized log space to original scale (USD/m²).
        Same as RegressionMLP.
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
        
        Returns:
            (B,) - Prediction in original scale (USD/m²), clamped to >= 0
        """
        # Step 1: Denormalize (reverse normalization)
        pred_log = prediction_log * self.target_std + self.target_mean
        
        # Step 2: Reverse log-transform to get original scale
        if self.target_log_transform:
            pred_original = torch.expm1(pred_log)  # exp(x) - 1
        else:
            pred_original = pred_log
        
        # Ensure non-negative (construction cost can't be negative)
        pred_original = torch.clamp(pred_original, min=0.0)
        
        return pred_original
    
    def calculate_loss(
        self,
        prediction_log: Tensor,
        prediction_original: Tensor,
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Calculate ALL losses internally, but only weight those in loss_config.
        Same as RegressionMLP.
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
            prediction_original: (B,) - Prediction in original scale (USD/m²)
            target: Optional (B,) - Target in normalized log space (not used, kept for compatibility)
            target_original: Optional (B,) - Target in original scale (USD/m²)
        
        Returns:
            dict with keys:
                - 'total': (scalar) - Total weighted loss (for backprop, only losses in loss_config)
                - 'rmsle': (scalar) - RMSLE loss (always calculated)
                - 'huber': (scalar) - Huber loss (always calculated)
                - 'mae': (scalar) - MAE loss (always calculated)
                - 'mse': (scalar) - MSE loss (always calculated)
                - 'rmse': (scalar) - RMSE loss (always calculated, sqrt of MSE)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Ensure target_original is available (required for all losses)
        if target_original is None:
            raise ValueError("target_original is required for loss calculation")
        
        # Ensure non-negative
        pred_original = torch.clamp(prediction_original, min=0.0)
        tgt_original = torch.clamp(target_original, min=0.0)
        
        # Calculate ALL losses (for monitoring)
        
        # 1. RMSLE: sqrt(mean((log1p(y_true_orig) - log1p(y_pred_orig))^2))
        log_pred = torch.log1p(pred_original)  # log(1 + y_pred)
        log_true = torch.log1p(tgt_original)   # log(1 + y_true)
        squared_log_error = (log_true - log_pred) ** 2
        rmsle = torch.sqrt(torch.mean(squared_log_error))
        loss_dict['rmsle'] = rmsle
        
        # 2. Huber loss on original scale
        huber = self.loss_fns['huber'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['huber'] = huber
        
        # 3. MAE (L1) loss on original scale
        mae = self.loss_fns['mae'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mae'] = mae
        
        # 4. MSE (L2) loss on original scale
        mse = self.loss_fns['mse'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mse'] = mse
        
        # Calculate RMSE from MSE (for monitoring)
        rmse = torch.sqrt(mse)
        loss_dict['rmse'] = rmse
        
        # Calculate total weighted loss (only losses in loss_config contribute)
        # Formula: normalized_loss = self_weight * raw_loss, then contribution = global_weight * normalized_loss
        raw_losses = {}  # Store raw losses for evaluation/metrics
        
        for loss_name in self.loss_config.keys():
            if loss_name in loss_dict:
                raw_loss = loss_dict[loss_name]
                self_weight = self.loss_self_weights.get(loss_name, 1.0)
                global_weight = self.loss_global_weights.get(loss_name, 1.0)
                normalized_loss = self_weight * raw_loss
                contribution = global_weight * normalized_loss
                
                # Store raw loss for evaluation/metrics (original value without self_weight)
                raw_losses[f'{loss_name}_raw'] = raw_loss
                
                # Update loss_dict to store normalized loss (after self_weight) for wandb logging
                loss_dict[loss_name] = normalized_loss
                
                total_loss = total_loss + contribution
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
        # Add raw losses to loss_dict for evaluation/metrics
        loss_dict.update(raw_losses)
        
        # Store total weighted loss
        loss_dict['total'] = total_loss
        
        return loss_dict


class AttentionAggregationRegression(nn.Module):
    """
    Attention-based aggregation regression head.
    Uses multi-head attention to learn which tokens/features to emphasize,
    then aggregates using attention-weighted sum before feeding to MLP.
    
    Architecture:
    - Input: (B, N, n_input) where N is sequence length (e.g., 20 tokens)
    - Multi-Head Attention: Learn attention weights over tokens -> (B, N, n_input)
    - Attention Aggregation: Weighted sum over sequence -> (B, n_input)
    - Hidden 1: Linear(n_input, n_hidden) -> BatchNorm -> ReLU -> Dropout
    - Hidden 2: Linear(n_hidden, n_hidden//2) -> BatchNorm -> ReLU -> Dropout
    - Output: Linear(n_hidden//2, 1)
    
    Args:
        n_input: Input dimension (embedding size)
        loss_type: Dict[str, float] - Loss names and weights (e.g., {"rmsle": 0.7, "mae": 0.2})
        n_hidden: Hidden dimension for MLP (default: n_input)
        p: Dropout probability (default: 0.2)
        num_heads: Number of attention heads (default: 8)
        target_mean: Mean for target normalization (default: 0.0)
        target_std: Std for target normalization (default: 1.0)
        target_log_transform: Whether target is log-transformed (default: True)
        huber_delta: Delta parameter for Huber loss (default: 1.0)
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        num_heads: int = 8,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_log_transform: bool = True,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        # IMPORTANT: Default to 2048 (matching old checkpoint architecture) for best results
        # Using n_input (512) as default gives poor results (~0.9 RMSLE)
        # Using 2048 gives much better results (can reach 0.18 RMSLE in pretrain, 0.12 in finetune)
        n_hidden = n_hidden if n_hidden is not None else 2048
        
        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_log_transform = target_log_transform
        self.huber_delta = huber_delta
        
        # Parse loss_type: REQUIRES new format with self_weight and global_weight
        # Format: {"rmsle": {"self_weight": 1.0, "global_weight": 0.2}, ...}
        # Convert DictConfig to dict if needed
        if isinstance(loss_type, DictConfig):
            loss_type = dict(loss_type)
        
        if not isinstance(loss_type, dict):
            raise ValueError(f"loss_type must be dict or DictConfig, got {type(loss_type)}")
        
        self.loss_config = {}
        self.loss_self_weights = {}  # self_weight for normalization
        self.loss_global_weights = {}  # global_weight for contribution to total
        
        for loss_name, loss_config in loss_type.items():
            if not isinstance(loss_config, dict):
                raise ValueError(
                    f"Loss '{loss_name}' config must be a dict with 'self_weight' and 'global_weight'. "
                    f"Got: {type(loss_config)}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            if 'self_weight' not in loss_config or 'global_weight' not in loss_config:
                raise ValueError(
                    f"Loss '{loss_name}' config must have both 'self_weight' and 'global_weight'. "
                    f"Got: {loss_config}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            self.loss_self_weights[loss_name] = float(loss_config['self_weight'])
            self.loss_global_weights[loss_name] = float(loss_config['global_weight'])
            self.loss_config[loss_name] = loss_config  # Store full config for reference
        
        # Normalize global weights to sum to 1.0 (optional, but good practice)
        total_global_weight = sum(self.loss_global_weights.values())
        if total_global_weight > 0:
            self.loss_global_weights = {k: v / total_global_weight for k, v in self.loss_global_weights.items()}
        
        # Store primary loss name for backward compatibility
        self.loss_type = list(self.loss_config.keys())[0] if self.loss_config else 'rmsle'
        
        # Ensure num_heads divides n_input
        if n_input % num_heads != 0:
            # Adjust num_heads to be compatible
            num_heads = max(1, n_input // (n_input // num_heads))
            print(f"⚠️  Adjusted num_heads to {num_heads} to be compatible with n_input={n_input}")
        
        # Multi-head self-attention for learning token importance
        # We use self-attention where query, key, value all come from input
        self.attention = nn.MultiheadAttention(
            embed_dim=n_input,
            num_heads=num_heads,
            dropout=p,
            batch_first=True  # (B, N, n_input) format
        )
        
        # Layer norm after attention
        self.attention_norm = nn.LayerNorm(n_input)
        
        # MLP for final prediction (same structure as RegressionMLP)
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
        
        # DISABLED: Custom weight initialization causes poor convergence
        # With _initialize_weights(): pretrain reaches ~0.6 RMSLE, finetune reaches ~0.3 RMSLE
        # Without (PyTorch defaults): pretrain reaches 0.18 RMSLE, finetune reaches 0.12 RMSLE
        # PyTorch's default initialization works much better for this task
        # self._initialize_weights()
        
        # Initialize loss functions for ALL possible losses (for monitoring)
        # We calculate all losses, but only weight those in loss_config
        self.loss_fns = {
            'huber': nn.HuberLoss(delta=huber_delta),
            'mae': nn.L1Loss(),
            'mse': nn.MSELoss(),
            'rmsle': None,  # RMSLE is computed manually
            'rmse': None,   # RMSE is computed from MSE (sqrt of MSE)
        }
        
        # Validate that all losses in config are supported
        for loss_name in self.loss_config.keys():
            if loss_name not in self.loss_fns:
                raise ValueError(f"Unknown loss type: {loss_name}. Supported: 'rmsle', 'huber', 'mae', 'mse', 'rmse'")
    
    # DISABLED: Custom weight initialization causes poor convergence
    # With _initialize_weights(): pretrain reaches ~0.6 RMSLE, finetune reaches ~0.3 RMSLE
    # Without (PyTorch defaults): pretrain reaches 0.18 RMSLE, finetune reaches 0.12 RMSLE
    # PyTorch's default initialization works much better for this task
    # def _initialize_weights(self):
    #     """Initialize weights using Kaiming/He initialization for ReLU layers."""
    #     # Attention weights are initialized by PyTorch (Xavier uniform by default)
    #     # We can leave them as default or customize if needed
    #     
    #     # Initialize MLP weights
    #     for module in self.mlp:
    #         if isinstance(module, nn.Linear):
    #             # Kaiming/He initialization for ReLU (recommended for ReLU activations)
    #             nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
    #             if module.bias is not None:
    #                 # Initialize bias to small positive value to avoid dead ReLU
    #                 nn.init.constant_(module.bias, 0.01)
    #             # For output layer, initialize to predict near zero (will be normalized)
    #             if module == self.mlp[-1]:  # Last layer (output)
    #                 nn.init.normal_(module.weight, mean=0.0, std=0.01)
    #                 if module.bias is not None:
    #                     # Initialize output bias to predict near normalized mean (target_mean)
    #                     nn.init.constant_(module.bias, self.target_mean)
    #         elif isinstance(module, nn.BatchNorm1d):
    #             # BatchNorm: weight=1, bias=0 (standard)
    #             nn.init.constant_(module.weight, 1.0)
    #             nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        x: Tensor, 
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, N, n_input) - Multimodal features (sequence of tokens)
            target: Optional (B,) - Target in normalized log space (for loss calculation)
            target_original: Optional (B,) - Target in original scale USD/m² (for RMSLE loss)
        
        Returns:
            dict with keys:
                - 'prediction_log': (B,) - Prediction in normalized log space
                - 'prediction_original': (B,) - Prediction in original scale (USD/m²)
                - 'loss': (scalar) - Loss value (if targets provided)
                - 'loss_dict': dict - Dictionary of loss components (if targets provided)
        """
        # x: (B, N, n_input) - e.g., (B, 20, 512)
        # Expected to receive full sequence from forward_multimodal_feature
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (B, N, n_input), got shape {x.shape}")
        if x.shape[2] != self.mlp[0].in_features:
            raise ValueError(f"Input x has wrong feature dimension: {x.shape[2]}, expected {self.mlp[0].in_features}. Shape: {x.shape}")
        
        # Step 1: Multi-head self-attention to learn token importance
        # Self-attention: query, key, value all from x
        attn_output, attn_weights = self.attention(x, x, x)  # (B, N, n_input)
        
        # Residual connection + layer norm
        x_attn = self.attention_norm(x + attn_output)  # (B, N, n_input)
        
        # Step 2: Attention-weighted aggregation (mean pooling over sequence)
        # We can also use the attention weights for weighted aggregation, but mean is simpler
        # and the attention already learned to emphasize important tokens
        x_agg = torch.mean(x_attn, dim=1)  # (B, n_input)
        
        # Ensure x_agg has correct shape (B, n_input)
        if x_agg.dim() != 2 or x_agg.shape[1] != self.mlp[0].in_features:
            raise ValueError(f"x_agg has wrong shape: {x_agg.shape}, expected (B, {self.mlp[0].in_features})")
        
        # Step 3: MLP for final prediction
        prediction_log = self.mlp(x_agg).squeeze(-1)  # (B,)
        
        # Decode to original scale
        prediction_original = self.construction_cost_decode(prediction_log)
        
        # Prepare return dict
        result = {
            'prediction_log': prediction_log,
            'prediction_original': prediction_original,
        }
        
        # Calculate loss if targets provided
        if target_original is not None:
            loss_dict = self.calculate_loss(
                prediction_log=prediction_log,
                prediction_original=prediction_original,
                target=target,
                target_original=target_original
            )
            result['loss'] = loss_dict['total']
            result['loss_dict'] = loss_dict
        
        return result
    
    def construction_cost_decode(self, prediction_log: Tensor) -> Tensor:
        """
        Decode prediction from normalized log space to original scale (USD/m²).
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
        
        Returns:
            (B,) - Prediction in original scale (USD/m²)
        """
        # Step 1: Denormalize from z-score normalization
        # prediction_log is in normalized space: (x - mean) / std
        # To get back: x = prediction_log * std + mean
        pred_log = prediction_log * self.target_std + self.target_mean
        
        # Clamp pred_log to prevent exp overflow (log space: reasonable range is ~[-10, 20])
        # This prevents exp(pred_log) from becoming inf
        pred_log = torch.clamp(pred_log, min=-10.0, max=20.0)
        
        # Step 2: Inverse log transform (if log-transformed)
        if self.target_log_transform:
            # Inverse of log1p: exp(x) - 1
            pred_original = torch.expm1(pred_log)  # exp(x) - 1
        else:
            pred_original = pred_log
        
        # Ensure non-negative (construction cost can't be negative)
        pred_original = torch.clamp(pred_original, min=0.0)
        
        # Additional safety: clamp to reasonable maximum (e.g., 1e6 USD/m²) to prevent inf
        pred_original = torch.clamp(pred_original, min=0.0, max=1e6)
        
        # Check for NaN/Inf and replace with a safe value
        if torch.any(torch.isnan(pred_original)) or torch.any(torch.isinf(pred_original)):
            # Replace NaN/Inf with median of valid predictions or a safe default
            valid_mask = torch.isfinite(pred_original)
            if torch.any(valid_mask):
                safe_value = torch.median(pred_original[valid_mask])
            else:
                # Fallback: use exp(target_mean) as safe value
                safe_value = torch.exp(torch.tensor(self.target_mean, device=pred_original.device))
            pred_original = torch.where(torch.isfinite(pred_original), pred_original, safe_value)
        
        return pred_original
    
    def calculate_loss(
        self,
        prediction_log: Tensor,
        prediction_original: Tensor,
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Calculate ALL losses internally, but only weight those in loss_config.
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
            prediction_original: (B,) - Prediction in original scale (USD/m²)
            target: Optional (B,) - Target in normalized log space (not used, kept for compatibility)
            target_original: Optional (B,) - Target in original scale (USD/m²)
        
        Returns:
            dict with keys:
                - 'total': (scalar) - Total weighted loss (for backprop, only losses in loss_config)
                - 'rmsle': (scalar) - RMSLE loss (always calculated)
                - 'huber': (scalar) - Huber loss (always calculated)
                - 'mae': (scalar) - MAE loss (always calculated)
                - 'mse': (scalar) - MSE loss (always calculated)
                - 'rmse': (scalar) - RMSE loss (always calculated, sqrt of MSE)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Ensure target_original is available (required for all losses)
        if target_original is None:
            raise ValueError("target_original is required for loss calculation")
        
        # Ensure non-negative
        pred_original = torch.clamp(prediction_original, min=0.0)
        tgt_original = torch.clamp(target_original, min=0.0)
        
        # Calculate ALL losses (for monitoring)
        
        # 1. RMSLE: sqrt(mean((log1p(y_true_orig) - log1p(y_pred_orig))^2))
        log_pred = torch.log1p(pred_original)  # log(1 + y_pred)
        log_true = torch.log1p(tgt_original)   # log(1 + y_true)
        squared_log_error = (log_true - log_pred) ** 2
        rmsle = torch.sqrt(torch.mean(squared_log_error))
        loss_dict['rmsle'] = rmsle
        
        # 2. Huber loss on original scale
        huber = self.loss_fns['huber'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['huber'] = huber
        
        # 3. MAE (L1) loss on original scale
        mae = self.loss_fns['mae'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mae'] = mae
        
        # 4. MSE (L2) loss on original scale
        mse = self.loss_fns['mse'](pred_original.squeeze(), tgt_original.squeeze())
        loss_dict['mse'] = mse
        
        # Calculate RMSE from MSE (for monitoring)
        rmse = torch.sqrt(mse)
        loss_dict['rmse'] = rmse
        
        # Calculate total weighted loss (only losses in loss_config contribute)
        # Formula: normalized_loss = self_weight * raw_loss, then contribution = global_weight * normalized_loss
        raw_losses = {}  # Store raw losses for evaluation/metrics
        
        for loss_name in self.loss_config.keys():
            if loss_name in loss_dict:
                raw_loss = loss_dict[loss_name]
                self_weight = self.loss_self_weights.get(loss_name, 1.0)
                global_weight = self.loss_global_weights.get(loss_name, 1.0)
                normalized_loss = self_weight * raw_loss
                contribution = global_weight * normalized_loss
                
                # Store raw loss for evaluation/metrics (original value without self_weight)
                raw_losses[f'{loss_name}_raw'] = raw_loss
                
                # Update loss_dict to store normalized loss (after self_weight) for wandb logging
                loss_dict[loss_name] = normalized_loss
                
                total_loss = total_loss + contribution
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
        # Add raw losses to loss_dict for evaluation/metrics
        loss_dict.update(raw_losses)
        
        # Store total weighted loss
        loss_dict['total'] = total_loss
        
        return loss_dict


class QueryAttentionRegression(RegressionMLP):
    """
    Learnable query attention pooling head.
    Uses one (or a few) learnable queries to attend over token sequence, then MLP to scalar.
    
    Config keys to add when using this head:
      type: QueryAttentionRegression
      num_queries: int (default 1)
      num_heads: int (default 8)
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_log_transform: bool = True,
        huber_delta: float = 1.0,
        num_queries: int = 1,
        num_heads: int = 8,
    ):
        super().__init__(
            n_input=n_input,
            loss_type=loss_type,
            n_hidden=n_hidden,
            p=p,
            target_mean=target_mean,
            target_std=target_std,
            target_log_transform=target_log_transform,
            huber_delta=huber_delta,
        )
        self.num_queries = num_queries
        self.attention = nn.MultiheadAttention(
            embed_dim=n_input,
            num_heads=num_heads,
            dropout=p,
            batch_first=True,
        )
        # Learnable queries (num_queries, n_input)
        self.query = nn.Parameter(torch.randn(num_queries, n_input))
        self.attn_norm = nn.LayerNorm(n_input)

    def forward(
        self,
        x: Tensor,
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (B, N, n_input), got shape {x.shape}")
        B = x.shape[0]
        # Expand learnable queries for batch: (B, num_queries, n_input)
        queries = self.query.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.attention(queries, x, x)  # (B, num_queries, n_input)
        attn_out = self.attn_norm(attn_out)
        # Aggregate queries (mean if multiple)
        if attn_out.shape[1] == 1:
            agg = attn_out.squeeze(1)
        else:
            agg = attn_out.mean(dim=1)
        prediction_log = self.mlp(agg).squeeze(-1)
        prediction_original = self.construction_cost_decode(prediction_log)
        result = {
            'prediction_log': prediction_log,
            'prediction_original': prediction_original,
        }
        if target is not None or target_original is not None:
            loss_dict = self.calculate_loss(
                prediction_log=prediction_log,
                prediction_original=prediction_original,
                target=target,
                target_original=target_original,
            )
            result['loss'] = loss_dict['total']
            result['loss_dict'] = loss_dict
        return result


class GatedAttentionPoolingRegression(RegressionMLP):
    """
    Gated attention pooling head (MIL-style).
    Learns token importance via a small gating MLP, aggregates tokens with soft weights, then MLP to scalar.
    
    Config keys to add when using this head:
      type: GatedAttentionPoolingRegression
      gate_hidden: int (default n_input // 2)
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_log_transform: bool = True,
        huber_delta: float = 1.0,
        gate_hidden: Optional[int] = None,
    ):
        super().__init__(
            n_input=n_input,
            loss_type=loss_type,
            n_hidden=n_hidden,
            p=p,
            target_mean=target_mean,
            target_std=target_std,
            target_log_transform=target_log_transform,
            huber_delta=huber_delta,
        )
        gate_hidden = gate_hidden if gate_hidden is not None else n_input // 2
        self.pre_ln = nn.LayerNorm(n_input)
        self.gate = nn.Sequential(
            nn.Linear(n_input, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        x: Tensor,
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (B, N, n_input), got shape {x.shape}")
        x_ln = self.pre_ln(x)
        gate_logits = self.gate(x_ln).squeeze(-1)  # (B, N)
        attn_weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)  # (B, N, 1)
        agg = torch.sum(attn_weights * x_ln, dim=1)  # (B, n_input)
        prediction_log = self.mlp(agg).squeeze(-1)
        prediction_original = self.construction_cost_decode(prediction_log)
        result = {
            'prediction_log': prediction_log,
            'prediction_original': prediction_original,
        }
        if target is not None or target_original is not None:
            loss_dict = self.calculate_loss(
                prediction_log=prediction_log,
                prediction_original=prediction_original,
                target=target,
                target_original=target_original,
            )
            result['loss'] = loss_dict['total']
            result['loss_dict'] = loss_dict
        return result


class DualPoolRegression(RegressionMLP):
    """
    Dual pooling head (mean + max) with MLP.
    Concatenates mean and max pooled tokens, then MLP to scalar.
    
    Config keys to add when using this head:
      type: DualPoolRegression
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_log_transform: bool = True,
        huber_delta: float = 1.0,
    ):
        # Store chosen hidden dim for overriding MLP
        chosen_hidden = n_hidden if n_hidden is not None else 2048
        super().__init__(
            n_input=n_input,
            loss_type=loss_type,
            n_hidden=chosen_hidden,
            p=p,
            target_mean=target_mean,
            target_std=target_std,
            target_log_transform=target_log_transform,
            huber_delta=huber_delta,
        )
        # Override MLP to accept concatenated mean+max (2 * n_input)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_input, chosen_hidden),
            nn.BatchNorm1d(chosen_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(chosen_hidden, chosen_hidden // 2),
            nn.BatchNorm1d(chosen_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(chosen_hidden // 2, 1),
        )

    def forward(
        self,
        x: Tensor,
        target: Optional[Tensor] = None,
        target_original: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3D (B, N, n_input), got shape {x.shape}")
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        agg = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*n_input)
        prediction_log = self.mlp(agg).squeeze(-1)
        prediction_original = self.construction_cost_decode(prediction_log)
        result = {
            'prediction_log': prediction_log,
            'prediction_original': prediction_original,
        }
        if target is not None or target_original is not None:
            loss_dict = self.calculate_loss(
                prediction_log=prediction_log,
                prediction_original=prediction_original,
                target=target,
                target_original=target_original,
            )
            result['loss'] = loss_dict['total']
            result['loss_dict'] = loss_dict
        return result


class MultiTaskCountryAwareRegression(nn.Module):
    """
    Multi-task head with parallel branches: classification + per-class regression.
    Uses attention-based feature aggregation (like AttentionAggregationRegression) 
    combined with multi-task learning (like YOLO/DETR object detection heads).
    
    Architecture:
    - Input: (B, N, n_input) multimodal features
    - Attention-based aggregation:
      - Multi-head self-attention: Learn token importance -> (B, N, n_input)
      - Residual connection + LayerNorm
      - Mean pooling over sequence -> (B, n_input)
    - Classification Branch: MLP -> (B, num_countries) class logits
      - Class probabilities (softmax) serve as confidence scores
    - Regression Branch: MLP -> (B, num_countries) regression values
      - One regression value per class (like bounding box per class in object detection)
    - Final prediction: Weighted sum using classification probabilities (standard YOLO approach)
    
    Loss Calculation (like object detection):
    - Classification loss: Cross-entropy on class logits
    - Regression loss: Only computed for ground truth class (standard object detection)
    
    Normalization:
    - REQUIRES target_mean_by_country and target_std_by_country (country-specific normalization)
    
    Args:
        n_input: int - Input dimension (embedding size)
        loss_type: Dict[str, float] - Loss names and weights for regression and classification
            - Regression losses: 'rmsle', 'mse', 'rmse', 'mae', 'huber'
            - Classification loss: 'country_classification' (weight for country classification loss)
            - If 'country_classification' not specified, defaults to primary regression loss weight
        n_hidden: Optional[int] - Hidden dimension (default: 2048)
        p: float - Dropout probability (default: 0.2)
        num_heads: int - Number of attention heads (default: 8)
        num_countries: int - Number of country classes (default: 2)
        confidence_threshold: float - Threshold for confidence score (default: 0.5) - unused, kept for compatibility
        target_mean: float - Overall mean (used when country-specific stats not provided)
        target_std: float - Overall std (used when country-specific stats not provided)
        target_mean_by_country: Optional[Dict[int, float]] - Country-specific means {0: mean0, 1: mean1}
        target_std_by_country: Optional[Dict[int, float]] - Country-specific stds {0: std0, 1: std1}
        huber_delta: float - Delta parameter for Huber loss (default: 1.0)
    """
    def __init__(
        self,
        n_input: int,
        loss_type: Dict[str, float],
        n_hidden: Optional[int] = None,
        p: float = 0.2,
        num_heads: int = 8,
        num_countries: int = 2,
        confidence_threshold: float = 0.5,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_mean_by_country: Optional[Dict[int, float]] = None,
        target_std_by_country: Optional[Dict[int, float]] = None,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        
        # IMPORTANT: Default to 2048 (matching old checkpoint architecture) for best results
        n_hidden = n_hidden if n_hidden is not None else 2048
        
        # Store configuration
        self.num_countries = num_countries
        self.confidence_threshold = confidence_threshold
        self.huber_delta = huber_delta
        
        # Ensure num_heads divides n_input (like AttentionAggregationRegression)
        if n_input % num_heads != 0:
            # Adjust num_heads to be compatible
            num_heads = max(1, n_input // (n_input // num_heads))
            print(f"⚠️  Adjusted num_heads to {num_heads} to be compatible with n_input={n_input}")
        
        # Multi-head self-attention for learning token importance (like AttentionAggregationRegression)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_input,
            num_heads=num_heads,
            dropout=p,
            batch_first=True  # (B, N, n_input) format
        )
        
        # Layer norm after attention (like AttentionAggregationRegression)
        self.attention_norm = nn.LayerNorm(n_input)
        
        # Store normalization parameters
        # MultiTaskCountryAwareRegression REQUIRES country-specific stats
        if not target_mean_by_country or not target_std_by_country:
            raise ValueError(
                "MultiTaskCountryAwareRegression requires target_mean_by_country and target_std_by_country to be provided. "
                f"Got target_mean_by_country={target_mean_by_country}, target_std_by_country={target_std_by_country}"
            )
        
        self.target_mean = target_mean  # Kept for backward compatibility, but not used
        self.target_std = target_std  # Kept for backward compatibility, but not used
        self.target_mean_by_country = target_mean_by_country
        self.target_std_by_country = target_std_by_country
        
        # Validate that both dicts have matching keys
        if set(self.target_mean_by_country.keys()) != set(self.target_std_by_country.keys()):
            raise ValueError(
                f"target_mean_by_country and target_std_by_country must have the same class IDs. "
                f"Got mean keys: {set(self.target_mean_by_country.keys())}, "
                f"std keys: {set(self.target_std_by_country.keys())}"
            )
        
        # Store class IDs from config (keys of target_mean_by_country dict)
        # This tells us how many classes and what their IDs are
        self.class_ids = sorted(list(self.target_mean_by_country.keys()))
        
        # Convert DictConfig to dict if needed
        if isinstance(loss_type, DictConfig):
            loss_type = dict(loss_type)
        
        if not isinstance(loss_type, dict):
            raise ValueError(f"loss_type must be dict or DictConfig, got {type(loss_type)}")
        
        # Parse loss_type: REQUIRES new format with self_weight and global_weight
        # Format: {"rmsle": {"self_weight": 1.0, "global_weight": 0.2}, ...}
        self.loss_config = {}
        self.loss_self_weights = {}  # self_weight for normalization
        self.loss_global_weights = {}  # global_weight for contribution to total
        
        for loss_name, loss_config in loss_type.items():
            if not isinstance(loss_config, dict):
                raise ValueError(
                    f"Loss '{loss_name}' config must be a dict with 'self_weight' and 'global_weight'. "
                    f"Got: {type(loss_config)}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            if 'self_weight' not in loss_config or 'global_weight' not in loss_config:
                raise ValueError(
                    f"Loss '{loss_name}' config must have both 'self_weight' and 'global_weight'. "
                    f"Got: {loss_config}. Example: {{'self_weight': 1.0, 'global_weight': 0.2}}"
                )
            
            self.loss_self_weights[loss_name] = float(loss_config['self_weight'])
            self.loss_global_weights[loss_name] = float(loss_config['global_weight'])
            self.loss_config[loss_name] = loss_config  # Store full config for reference
        
        # All losses (including classification) are treated equally in loss_config
        # No special treatment - classification_ce will be looked up in loss_config just like rmsle, mse, etc.
        
        # Normalize global weights to sum to 1.0 (optional, but good practice)
        # This applies to ALL losses including classification
        total_global_weight = sum(self.loss_global_weights.values())
        if total_global_weight > 0:
            self.loss_global_weights = {k: v / total_global_weight for k, v in self.loss_global_weights.items()}
        
        self.loss_type = list(self.loss_config.keys())[0] if self.loss_config else 'rmsle'
        
        # Classification branch: predicts class probabilities (like YOLO/DETR)
        # These probabilities serve as confidence scores
        # Uses attention-aggregated features instead of simple mean pooling
        self.classification_head = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(n_hidden, num_countries)  # (B, num_countries) class logits
        )
        
        # Regression branch: predicts regression value for each class (like YOLO/DETR)
        # Each class gets its own regression prediction
        # Uses attention-aggregated features instead of simple mean pooling
        self.regression_head = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.BatchNorm1d(n_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(n_hidden // 2, num_countries)  # (B, num_countries) - one value per class
        )
        
        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss(reduction='mean', delta=huber_delta)
    
    def construction_cost_decode(self, prediction_log: Tensor, country: Optional[Tensor] = None) -> Tensor:
        """
        Decode prediction from normalized log space to original scale (USD/m²).
        Uses country-specific normalization if country-specific stats are provided and country is given.
        
        Args:
            prediction_log: (B,) - Prediction in normalized log space
            country: Optional (B,) - Country labels for country-specific decoding
        
        Returns:
            (B,) - Prediction in original scale (USD/m²), clamped to >= 0
        """
        # Country-specific decoding (vectorized)
        # MultiTaskCountryAwareRegression always uses country-specific normalization
        if country is None:
            raise ValueError(
                "MultiTaskCountryAwareRegression requires country labels for decoding. "
                "Got country=None"
            )
        
        # Get country-specific stats for all samples at once
        country_mean = torch.zeros_like(prediction_log)
        country_std = torch.ones_like(prediction_log)
        
        for country_id in self.class_ids:
            mask = (country == country_id)
            if mask.any():
                if country_id not in self.target_mean_by_country:
                    raise ValueError(
                        f"Country ID {country_id} found in data but not in target_mean_by_country. "
                        f"Available class IDs: {self.class_ids}"
                    )
                country_mean[mask] = self.target_mean_by_country[country_id]
                country_std[mask] = self.target_std_by_country[country_id]
        
        # Denormalize (vectorized)
        pred_log = prediction_log * country_std + country_mean
        
        # Reverse log-transform to get original scale (target_log is always log-transformed from dataloader)
        pred_original = torch.expm1(torch.clamp(pred_log, min=-10.0, max=10.0))
        
        # Ensure non-negative
        pred_original = torch.clamp(pred_original, min=0.0)
        return pred_original
    
    def calculate_loss(
        self,
        prediction_log: Tensor,
        prediction_original: Tensor,
        target_log: Tensor,
        target_original: Tensor,
        country_gt: Tensor,
        classification_logits: Tensor,
        regression_values: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Calculate ALL losses internally, but only weight those in loss_config.
        Vectorized calculation (no loops) - similar to object detection heads.
        
        Args:
            prediction_log: (B,) - Final prediction in normalized log space
            prediction_original: (B,) - Final prediction in original scale (USD/m²)
            target_log: (B,) - Target in log space (log1p(cost), NOT normalized, from dataloader)
            target_original: (B,) - Target in original scale (USD/m², from dataloader)
            country_gt: (B,) - Ground truth country labels (0 or 1)
            classification_logits: (B, num_countries) - Classification logits
            regression_values: (B, num_countries) - Regression values for each class
        
        Returns:
            dict with keys:
                - 'total': (scalar) - Total weighted loss (for backprop)
                - 'classification_ce': (scalar) - Classification cross-entropy loss
                - 'rmsle': (scalar) - RMSLE loss
                - 'huber': (scalar) - Huber loss
                - 'mae': (scalar) - MAE loss
                - 'mse': (scalar) - MSE loss
                - 'rmse': (scalar) - RMSE loss
                - 'regression': (scalar) - Regression total loss
        """
        loss_dict = {}
        
        # Classification loss (always computed for multi-task head)
        classification_loss = self.cross_entropy(classification_logits, country_gt)
        loss_dict['classification_ce'] = classification_loss
        
        # Regression loss: use regression value for ground truth country (like object detection)
        # Select regression value for ground truth country (standard object detection)
        gt_regression = regression_values.gather(1, country_gt.unsqueeze(1)).squeeze(1)  # (B,)
        
        # Decode using country-specific normalization
        # MultiTaskCountryAwareRegression always uses country-specific normalization
        country_mean = torch.zeros_like(gt_regression)
        country_std = torch.ones_like(gt_regression)
        
        # Set country-specific mean and std for each sample based on country_gt
        for country_id in self.class_ids:
            mask = (country_gt == country_id)
            if mask.any():
                if country_id not in self.target_mean_by_country:
                    raise ValueError(
                        f"Country ID {country_id} found in data but not in target_mean_by_country. "
                        f"Available class IDs: {self.class_ids}"
                    )
                country_mean[mask] = self.target_mean_by_country[country_id]
                country_std[mask] = self.target_std_by_country[country_id]
        
        # Denormalize prediction (gt_regression is in normalized log space)
        pred_log_denorm = gt_regression * country_std + country_mean  # (B,) - denormalized log space
        
        # target_log is already in log space (log1p(cost), NOT normalized) - use directly!
        # No conversion needed - dataset already prepared it
        
        # Calculate losses in log space (consistent with RMSLE)
        # Use pred_log_denorm and target_log directly (both in log space)
        # 1. RMSLE: sqrt(mean((log1p(y_true_orig) - log1p(y_pred_orig))^2))
        # Since target_log = log1p(target_original) and pred_log_denorm = log1p(pred_original) after denormalize,
        # we can use them directly
        squared_log_error = (target_log - pred_log_denorm) ** 2
        rmsle = torch.sqrt(torch.mean(squared_log_error))
        loss_dict['rmsle'] = rmsle
        
        # 2. Huber loss in log space
        huber = self.huber_loss(pred_log_denorm, target_log)
        loss_dict['huber'] = huber
        
        # 3. MAE (L1) loss in log space
        mae = torch.mean(torch.abs(target_log - pred_log_denorm))
        loss_dict['mae'] = mae
        
        # 4. MSE (L2) loss in log space
        mse = torch.mean((target_log - pred_log_denorm) ** 2)
        loss_dict['mse'] = mse
        
        # Also decode to original scale for prediction_original (needed for return value)
        pred_original = torch.expm1(torch.clamp(pred_log_denorm, min=-10.0, max=10.0))
        pred_original = torch.clamp(pred_original, min=0.0)
        
        # Calculate RMSE from MSE (for monitoring)
        rmse = torch.sqrt(mse)
        loss_dict['rmse'] = rmse
        
        # Calculate total loss: ALL losses (including classification) are treated equally
        # Formula for each loss: normalized_loss = self_weight * raw_loss, then contribution = global_weight * normalized_loss
        total_loss = 0.0
        regression_total = 0.0
        
        # Store raw losses for evaluation/metrics (before normalization)
        raw_losses = {}
        
        # Process ALL losses in the same loop (classification_ce is just another loss like rmsle, mse, etc.)
        for loss_name in self.loss_config.keys():
            if loss_name in loss_dict:
                raw_loss = loss_dict[loss_name]
                self_weight = self.loss_self_weights.get(loss_name, 1.0)
                global_weight = self.loss_global_weights.get(loss_name, 1.0)
                normalized_loss = self_weight * raw_loss
                contribution = global_weight * normalized_loss
                
                # Store raw loss for evaluation/metrics (original value without self_weight)
                raw_losses[f'{loss_name}_raw'] = raw_loss
                
                # Update loss_dict to store normalized loss (after self_weight) for wandb logging
                loss_dict[loss_name] = normalized_loss
                
                total_loss = total_loss + contribution  # ALL losses (including classification_ce) go to total_loss for backprop
                
                # Track regression losses separately for monitoring (exclude classification_ce from regression_total only)
                if loss_name != 'classification_ce':
                    regression_total = regression_total + contribution
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
        # Add raw losses to loss_dict for evaluation/metrics
        loss_dict.update(raw_losses)
        
        loss_dict['regression'] = regression_total
        loss_dict['total'] = total_loss
        
        return loss_dict
    
    def forward(
        self,
        x: Tensor,  # (B, N, n_input)
        target_log: Optional[Tensor] = None,  # (B,) - Target in log space (log1p(cost), NOT normalized, from dataloader)
        target_original: Optional[Tensor] = None,  # (B,) - Target in original scale (USD/m², from dataloader)
        country_gt: Optional[Tensor] = None,  # (B,) - Ground truth country labels (0 or 1)
    ) -> Dict[str, Tensor]:
        """
        Forward pass with multi-task learning (object detection style).
        
        Args:
            x: (B, N, n_input) - Multimodal features
            target_log: Optional (B,) - Target in log space (log1p(cost), NOT normalized, from dataloader)
            target_original: Optional (B,) - Target in original scale (USD/m², from dataloader)
            country_gt: Optional (B,) - Ground truth country labels (0 or 1)
        
        Returns:
            dict with keys:
                - 'prediction_log': (B,) - Final prediction in normalized log space (selected by confidence)
                - 'prediction_original': (B,) - Final prediction in original scale
                - 'classification_logits': (B, num_countries) - Class logits
                - 'classification_probs': (B, num_countries) - Class probabilities
                - 'regression_values': (B, num_countries) - Regression value for each country
                - 'confidence_scores': (B, num_countries) - Confidence score for each country
                - 'selected_country': (B,) - Selected country based on confidence threshold
                - 'loss_dict': dict - Dictionary of loss components
        """
        # Attention-based feature aggregation (like AttentionAggregationRegression)
        if x.dim() == 3:
            # Step 1: Multi-head self-attention to learn token importance
            # Self-attention: query, key, value all from x
            attn_output, attn_weights = self.attention(x, x, x)  # (B, N, n_input)
            
            # Step 2: Residual connection + layer norm
            x_attn = self.attention_norm(x + attn_output)  # (B, N, n_input)
            
            # Step 3: Mean pooling over sequence (after attention has learned importance)
            x_agg = torch.mean(x_attn, dim=1)  # (B, n_input)
        else:
            x_agg = x  # Already (B, n_input)
        
        # Classification branch: predicts class probabilities (like YOLO/DETR)
        # These probabilities ARE the confidence scores
        classification_logits = self.classification_head(x_agg)  # (B, num_countries)
        classification_probs = F.softmax(classification_logits, dim=1)  # (B, num_countries)
        # Classification probabilities serve as confidence scores
        confidence_scores = classification_probs  # (B, num_countries)
        
        # Regression branch: predicts regression value for each class (like YOLO/DETR)
        # Each class gets its own regression prediction
        regression_values = self.regression_head(x_agg)  # (B, num_countries) - one value per class
        
        # Select final prediction (standard object detection approach):
        # Option 1: Weighted sum using classification probabilities (like YOLO)
        # Commented out - will try later if selected_country approach is bad
        # weighted_regression = (classification_probs * regression_values).sum(dim=1)  # (B,)
        
        # Option 2: Select by highest confidence (classification probability)
        selected_country = classification_probs.argmax(dim=1)  # (B,)
        selected_regression = regression_values.gather(1, selected_country.unsqueeze(1)).squeeze(1)  # (B,)
        
        # Use selected regression as final prediction (Option 2)
        pred_log = selected_regression  # (B,)
        
        # Decode prediction (use country_gt during training, selected_country during inference)
        # MultiTaskCountryAwareRegression always uses country-specific normalization
        country_for_decode = country_gt if country_gt is not None else selected_country
        pred_original = self.construction_cost_decode(pred_log, country_for_decode)
        
        # Prepare result dict
        result = {
            'prediction_log': pred_log,
            'prediction_original': pred_original,
            'classification_logits': classification_logits,
            'classification_probs': classification_probs,
            'confidence_scores': confidence_scores,  # Same as classification_probs
            'regression_values': regression_values,  # (B, num_countries) - one value per class
            'selected_country': selected_country,
        }
        
        # Calculate losses if targets are provided (all losses calculated in calculate_loss)
        if target_log is not None and target_original is not None and country_gt is not None:
            loss_dict = self.calculate_loss(
                prediction_log=pred_log,
                prediction_original=pred_original,
                target_log=target_log,
                target_original=target_original,
                country_gt=country_gt,
                classification_logits=classification_logits,
                regression_values=regression_values,
            )
            
            result['loss_dict'] = loss_dict
            result['loss'] = loss_dict['total']
        
        return result


# Registry of available head classes
HEAD_REGISTRY = {
    'RegressionMLP': RegressionMLP,
    'RegressionMLP2': RegressionMLPTest,
    'MixtureOfExpertsRegression': MixtureOfExpertsRegression,
    'AttentionAggregationRegression': AttentionAggregationRegression,
    'QueryAttentionRegression': QueryAttentionRegression,
    'GatedAttentionPoolingRegression': GatedAttentionPoolingRegression,
    'DualPoolRegression': DualPoolRegression,
    'MultiTaskCountryAwareRegression': MultiTaskCountryAwareRegression,
    # Add more head classes here in the future
    # 'RegressionTransformer': RegressionTransformer,
    # 'RegressionResNet': RegressionResNet,
}


def get_head_class(head_name: str):
    """
    Get a head class by name.
    
    Args:
        head_name: Name of the head class (e.g., 'RegressionMLP')
    
    Returns:
        Head class
    
    Raises:
        ValueError: If head_name is not in registry
    """
    if head_name not in HEAD_REGISTRY:
        available = ', '.join(HEAD_REGISTRY.keys())
        raise ValueError(
            f"Unknown head class: '{head_name}'. "
            f"Available heads: {available}"
        )
    return HEAD_REGISTRY[head_name]


def create_head(
    head_config: Dict,
    n_input: int,
):
    """
    Create a head instance from configuration dict.
    
    Args:
        head_config: Dict containing all head configuration:
            - type: str - Head class name (e.g., 'RegressionMLP', 'MixtureOfExpertsRegression')
            - loss_type: Dict[str, float] - Loss names and weights (e.g., {"rmsle": 0.7, "mae": 0.2})
            - target_mean: float - Mean for target normalization (default: 0.0)
            - target_std: float - Std for target normalization (default: 1.0)
            - target_log_transform: bool - Whether target is log-transformed (default: True)
            - huber_delta: float - Delta parameter for Huber loss (default: 1.0)
            - n_hidden: Optional[int] - Hidden dimension (default: n_input)
            - p: float - Dropout probability (default: 0.2)
            - ... (any head-specific parameters, e.g., num_experts, top_k for MoE)
        n_input: int - Input dimension (embedding size)
    
    Returns:
        Head instance
    """
    # Convert DictConfig to dict if needed
    if isinstance(head_config, DictConfig):
        head_config = dict(head_config)
    
    if not isinstance(head_config, dict):
        raise ValueError(f"head_config must be dict or DictConfig, got {type(head_config)}")
    
    # Extract head type (required)
    head_name = head_config.get('type')
    if not head_name:
        raise ValueError("head_config must contain 'type' key specifying head class name")
    
    # Get head class
    head_class = get_head_class(head_name)
    
    # Preprocess loss_type: convert DictConfig to dict if needed
    if 'loss_type' in head_config:
        if isinstance(head_config['loss_type'], DictConfig):
            head_config['loss_type'] = dict(head_config['loss_type'])
    
    # Filter kwargs to only include valid arguments for the head class
    import inspect
    sig = inspect.signature(head_class.__init__)
    # Pass all parameters from head_config via **kwargs (except 'type')
    valid_kwargs = {k: v for k, v in head_config.items() if k != 'type' and k in sig.parameters}
    
    
    # Create head instance - pass all parameters via **kwargs directly
    return head_class(
        n_input=n_input,
        **valid_kwargs
    )

