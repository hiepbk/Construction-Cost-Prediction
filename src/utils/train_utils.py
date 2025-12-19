"""Training utilities"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
import pandas as pd
from typing import Tuple, Optional


def split_data(
    df: pd.DataFrame,
    target_col: str = 'construction_cost_per_m2_usd',
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and validation sets
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Proportion of test set
        random_state: Random seed
        group_col: Column to use for group-based splitting (e.g., 'country')
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if group_col and group_col in X.columns:
        # Group-based split
        groups = X[group_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    return X_train, X_val, y_train, y_val


def extract_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and prepare tabular features"""
    # Exclude non-feature columns
    exclude_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name']
    if 'construction_cost_per_m2_usd' in df.columns:
        exclude_cols.append('construction_cost_per_m2_usd')
    
    feature_df = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    
    # Handle categorical columns
    categorical_cols = feature_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        feature_df[col] = pd.Categorical(feature_df[col]).codes
    
    # Fill missing values
    feature_df = feature_df.fillna(feature_df.median())
    
    return feature_df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics (RMSLE is the primary competition metric)"""
    # Use solafune_utils if available, otherwise use local implementation
    try:
        from src.utils.solafune_utils import calculate_regression_metrics
        return calculate_regression_metrics(y_true, y_pred)
    except ImportError:
        # Fallback to local implementation
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Ensure non-negative predictions
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        y_true = np.clip(y_true, a_min=0.0, a_max=None)
        
        # PRIMARY METRIC: RMSLE (competition metric)
        rmsle = np.sqrt(mean_squared_error(
            np.log1p(y_true),
            np.log1p(y_pred)
        ))
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'rmsle': rmsle,  # PRIMARY METRIC
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

