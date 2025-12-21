"""
Construction Cost Regression Head Classes

This module defines regression head architectures for construction cost prediction.
All phases (pretrain, finetune, evaluation) should use heads from this module
to ensure consistency.

Available heads:
- RegressionMLP: Simple MLP with BatchNorm and Dropout
"""
from typing import Optional
import torch
from torch import Tensor, nn


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
    
    Returns:
        (B,) - Scalar predictions
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
        """
        Forward pass.
        
        Args:
            x: (B, N, n_input) - Multimodal features (sequence of tokens)
        
        Returns:
            (B,) - Scalar predictions
        """
        # Aggregate all tokens using mean pooling
        # This uses information from all tokens (CLS + feature tokens)
        if x.dim() == 3:
            # x is (B, N, n_input) - aggregate over sequence dimension
            x = x.mean(dim=1)  # (B, N, n_input) -> (B, n_input)
        # If x is already (B, n_input), pass through directly
        
        # Pass through MLP
        return self.mlp(x).squeeze(-1)  # (B, 1) -> (B,)


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


def create_head(head_name: str, n_input: int, n_hidden: Optional[int] = None, p: float = 0.2, **kwargs):
    """
    Create a head instance by name.
    
    Args:
        head_name: Name of the head class (e.g., 'RegressionMLP')
        n_input: Input dimension
        n_hidden: Hidden dimension (optional, depends on head)
        p: Dropout probability (optional, depends on head)
        **kwargs: Additional arguments for head initialization
    
    Returns:
        Head instance
    """
    head_class = get_head_class(head_name)
    
    # Filter kwargs to only include valid arguments for the head class
    import inspect
    sig = inspect.signature(head_class.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    return head_class(n_input=n_input, n_hidden=n_hidden, p=p, **valid_kwargs)

