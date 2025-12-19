'''
Utility script to generate field_lengths.pt file for TIP tabular encoder.

TIP requires a field_lengths tensor that specifies:
- For categorical features: cardinality (number of unique values)
- For continuous features: 1

This script processes the construction cost dataset and generates the field_lengths file.
'''
import os
import sys
import pandas as pd
import numpy as np
import torch

# Add path to import dataset utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


def generate_field_lengths(
    csv_path: str,
    output_path: str,
    exclude_cols: list = None,
    categorical_threshold: int = 50
):
    """
    Generate field_lengths.pt file for TIP tabular encoder.
    
    Args:
        csv_path: Path to CSV file with tabular data
        output_path: Path to save field_lengths.pt
        exclude_cols: Columns to exclude from features
        categorical_threshold: Maximum cardinality to treat as categorical
    
    Returns:
        field_lengths: List of field lengths (cardinalities for categorical, 1 for continuous)
        num_cat: Number of categorical features
        num_con: Number of continuous features
    """
    if exclude_cols is None:
        exclude_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'construction_cost_per_m2_usd']
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    field_lengths = []
    categorical_cols = []
    continuous_cols = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'int32', 'object', 'category']:
            # Check cardinality
            if df[col].dtype in ['object', 'category']:
                # Label encode to get cardinality
                encoded = pd.Categorical(df[col]).codes
                n_unique = len(pd.Categorical(df[col]).categories)
            else:
                n_unique = df[col].nunique()
            
            if n_unique <= categorical_threshold:
                # Categorical feature: cardinality
                field_lengths.append(n_unique)
                categorical_cols.append(col)
            else:
                # High cardinality: treat as continuous
                field_lengths.append(1)
                continuous_cols.append(col)
        else:
            # Float type: continuous
            field_lengths.append(1)
            continuous_cols.append(col)
    
    # Convert to tensor
    field_lengths_tensor = torch.tensor(field_lengths, dtype=torch.long)
    
    # Save
    torch.save(field_lengths_tensor, output_path)
    
    num_cat = len(categorical_cols)
    num_con = len(continuous_cols)
    
    print(f"Generated field_lengths.pt:")
    print(f"  Total features: {len(field_lengths)}")
    print(f"  Categorical: {num_cat}")
    print(f"  Continuous: {num_con}")
    print(f"  Saved to: {output_path}")
    
    return field_lengths, num_cat, num_con


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate field_lengths.pt for TIP')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save field_lengths.pt')
    parser.add_argument('--categorical_threshold', type=int, default=50, help='Max cardinality for categorical')
    
    args = parser.parse_args()
    
    generate_field_lengths(
        csv_path=args.csv_path,
        output_path=args.output_path,
        categorical_threshold=args.categorical_threshold
    )

