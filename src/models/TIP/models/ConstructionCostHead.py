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
        n_hidden = n_hidden or n_input
        
        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_log_transform = target_log_transform
        self.huber_delta = huber_delta
        
        # loss_type must be dict (multi-loss with weights) - no string support
        # Convert DictConfig to regular dict if needed (preprocess before using)
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
        
        # Initialize weights properly for regression with ReLU
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
        for loss_name, weight in self.loss_config.items():
            if loss_name in loss_dict:
                total_loss = total_loss + weight * loss_dict[loss_name]
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
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
        n_hidden = n_hidden or n_input
        
        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_log_transform = target_log_transform
        self.huber_delta = huber_delta
        
        # loss_type must be dict (multi-loss with weights) - no string support
        # Convert DictConfig to regular dict if needed (preprocess before using)
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
        
        # Initialize weights properly for regression with ReLU
        self._initialize_weights()
        
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
        for loss_name, weight in self.loss_config.items():
            if loss_name in loss_dict:
                total_loss = total_loss + weight * loss_dict[loss_name]
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
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
        n_hidden = n_hidden or n_input
        
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
        
        # Initialize weights properly for regression with ReLU
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
        for loss_name, weight in self.loss_config.items():
            if loss_name in loss_dict:
                total_loss = total_loss + weight * loss_dict[loss_name]
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
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
        n_hidden = n_hidden or n_input
        
        # Store target normalization parameters
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_log_transform = target_log_transform
        self.huber_delta = huber_delta
        
        # loss_type must be dict (multi-loss with weights) - no string support
        # Convert DictConfig to regular dict if needed (preprocess before using)
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
        
        # Initialize weights properly
        # when disable init, the pretrain performance is better
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
        # Attention weights are initialized by PyTorch (Xavier uniform by default)
        # We can leave them as default or customize if needed
        
        # Initialize MLP weights
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
        for loss_name, weight in self.loss_config.items():
            if loss_name in loss_dict:
                total_loss = total_loss + weight * loss_dict[loss_name]
            else:
                raise ValueError(f"Loss '{loss_name}' in loss_config but not calculated. Available: {list(loss_dict.keys())}")
        
        # Store total weighted loss
        loss_dict['total'] = total_loss
        
        return loss_dict


# Registry of available head classes
HEAD_REGISTRY = {
    'RegressionMLP': RegressionMLP,
    'RegressionMLP2': RegressionMLPTest,
    'MixtureOfExpertsRegression': MixtureOfExpertsRegression,
    'AttentionAggregationRegression': AttentionAggregationRegression,
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

