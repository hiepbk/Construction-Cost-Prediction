"""
Construction Cost Regression Head Classes

This module defines regression head architectures for construction cost prediction.
All phases (pretrain, finetune, evaluation) should use heads from this module
to ensure consistency.

Available heads:
- RegressionMLP: Simple MLP with BatchNorm and Dropout
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


# Registry of available head classes
HEAD_REGISTRY = {
    'RegressionMLP': RegressionMLP,
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
    head_name: str, 
    n_input: int,
    loss_type: Dict[str, float],  # Dict with loss names and weights
    n_hidden: Optional[int] = None, 
    p: float = 0.2,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    target_log_transform: bool = True,
    huber_delta: float = 1.0,
    **kwargs
):
    """
    Create a head instance by name.
    
    Args:
        head_name: Name of the head class (e.g., 'RegressionMLP')
        n_input: Input dimension
        n_hidden: Hidden dimension (optional, depends on head)
        p: Dropout probability (optional, depends on head)
        target_mean: Mean for target normalization (default: 0.0)
        target_std: Std for target normalization (default: 1.0)
        target_log_transform: Whether target is log-transformed (default: True)
        loss_type: Loss function type - must be dict (multi-loss with weights), e.g., {"rmsle": 0.7, "mae": 0.2, "rmse": 0.1}
                   Can also be OmegaConf DictConfig (will be converted to dict automatically)
        huber_delta: Delta parameter for Huber loss (default: 1.0)
        **kwargs: Additional arguments for head initialization
    
    Returns:
        Head instance
    """
    # Preprocess loss_type: convert DictConfig to dict before passing to head
    if isinstance(loss_type, DictConfig):
        loss_type = dict(loss_type)
    
    # Ensure loss_type is a dict
    if not isinstance(loss_type, dict):
        raise ValueError(f"loss_type must be dict or DictConfig, got {type(loss_type)}")
    
    head_class = get_head_class(head_name)
    
    # Filter kwargs to only include valid arguments for the head class
    import inspect
    sig = inspect.signature(head_class.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    return head_class(
        n_input=n_input,
        loss_type=loss_type,
        n_hidden=n_hidden, 
        p=p,
        target_mean=target_mean,
        target_std=target_std,
        target_log_transform=target_log_transform,
        huber_delta=huber_delta,
        **valid_kwargs
    )

