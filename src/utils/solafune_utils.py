"""
Utilities for integrating with Solafune competition tools
Note: Most solafune-tools are for segmentation/bbox tasks, but we can use some utilities
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os
import sys

# Add solafune-tools to path
SOLAFUNE_TOOLS_PATH = os.path.join(os.path.dirname(__file__), 'solafune-tools')
if os.path.exists(SOLAFUNE_TOOLS_PATH):
    sys.path.insert(0, SOLAFUNE_TOOLS_PATH)
    try:
        import solafune_tools
        SOLAFUNE_AVAILABLE = True
    except ImportError:
        SOLAFUNE_AVAILABLE = False
        print("Warning: solafune-tools not available. Some features may be disabled.")
else:
    SOLAFUNE_AVAILABLE = False


def validate_submission_csv(
    submission_file: str,
    sample_submission_file: str,
    check_data_ids: bool = True
) -> Tuple[bool, str]:
    """
    Validate submission CSV format for regression task
    
    Args:
        submission_file: Path to submission CSV file
        sample_submission_file: Path to sample submission CSV
        check_data_ids: Whether to check if data_ids match sample
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Load submission
        submission = pd.read_csv(submission_file)
        
        # Check required columns
        required_cols = ['data_id', 'construction_cost_per_m2_usd']
        missing_cols = [col for col in required_cols if col not in submission.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for NaN values
        if submission['construction_cost_per_m2_usd'].isna().any():
            return False, "Submission contains NaN values in construction_cost_per_m2_usd"
        
        # Check for negative values (construction cost should be >= 0)
        if (submission['construction_cost_per_m2_usd'] < 0).any():
            return False, "Submission contains negative values. Construction cost must be >= 0"
        
        # Check data_ids match sample if provided
        if check_data_ids and os.path.exists(sample_submission_file):
            sample = pd.read_csv(sample_submission_file)
            submission_ids = set(submission['data_id'].values)
            sample_ids = set(sample['data_id'].values)
            
            missing_ids = sample_ids - submission_ids
            extra_ids = submission_ids - sample_ids
            
            if missing_ids:
                return False, f"Missing data_ids: {len(missing_ids)} missing"
            if extra_ids:
                return False, f"Extra data_ids: {len(extra_ids)} extra"
        
        return True, "Valid submission format"
    
    except Exception as e:
        return False, f"Error validating submission: {str(e)}"


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calculate regression metrics for construction cost prediction
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Ensure predictions are non-negative (construction cost can't be negative)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    y_true = np.clip(y_true, a_min=0.0, a_max=None)
    
    # PRIMARY METRIC: RMSLE (Root Mean Squared Logarithmic Error)
    # This is the official competition metric
    rmsle = np.sqrt(mean_squared_error(
        np.log1p(y_true),  # log(y + 1)
        np.log1p(y_pred)   # log(ŷ + 1)
    ))
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Mean Squared Log Error
    msle = mean_squared_error(np.log1p(y_true), np.log1p(y_pred))
    
    # Symmetric Mean Absolute Percentage Error
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    return {
        'rmsle': rmsle,  # PRIMARY METRIC (competition metric)
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'msle': msle,
        'smape': smape
    }


def prepare_submission(
    predictions: np.ndarray,
    data_ids: np.ndarray,
    output_file: str,
    clip_negative: bool = True
) -> str:
    """
    Prepare submission CSV file
    
    Args:
        predictions: Array of predictions
        data_ids: Array of data_ids (must match predictions length)
        output_file: Path to save submission CSV
        clip_negative: Whether to clip negative predictions to 0
    
    Returns:
        Path to saved submission file
    """
    if len(predictions) != len(data_ids):
        raise ValueError(f"Predictions length ({len(predictions)}) != data_ids length ({len(data_ids)})")
    
    # Clip negative values
    if clip_negative:
        predictions = np.clip(predictions, a_min=0, a_max=None)
    
    # Create DataFrame
    submission = pd.DataFrame({
        'data_id': data_ids,
        'construction_cost_per_m2_usd': predictions
    })
    
    # Save
    submission.to_csv(output_file, index=False)
    
    return output_file


def load_tabular_data(
    csv_path: str,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load tabular data from CSV
    
    Args:
        csv_path: Path to CSV file
        target_col: Name of target column (if None, returns None for target)
    
    Returns:
        Tuple of (features_df, target_series)
    """
    df = pd.read_csv(csv_path)
    
    if target_col and target_col in df.columns:
        target = df[target_col]
        features = df.drop(columns=[target_col])
    else:
        target = None
        features = df
    
    return features, target

