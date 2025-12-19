"""
Data analysis utility for tabular features
Analyzes features to determine normalization strategies and feature types
"""
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
from pathlib import Path


class TabularDataAnalyzer:
    """Analyze tabular data to determine preprocessing strategies"""
    
    def __init__(self, csv_path: str, exclude_cols: List[str] = None):
        """
        Args:
            csv_path: Path to CSV file
            exclude_cols: Columns to exclude from analysis (IDs, file names, targets, etc.)
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        if exclude_cols is None:
            exclude_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 
                           'construction_cost_per_m2_usd']
        self.exclude_cols = exclude_cols
        
        # Get feature columns
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.feature_df = self.df[self.feature_cols].copy()
        
        self.analysis = {}
    
    def analyze(self) -> Dict:
        """Perform comprehensive analysis of tabular features"""
        print("=" * 80)
        print("TABULAR DATA ANALYSIS")
        print("=" * 80)
        print(f"\nDataset: {self.csv_path}")
        print(f"Total samples: {len(self.df)}")
        print(f"Total features: {len(self.feature_cols)}")
        print(f"Excluded columns: {self.exclude_cols}")
        
        # 1. Feature type analysis
        print("\n" + "=" * 80)
        print("1. FEATURE TYPE ANALYSIS")
        print("=" * 80)
        self._analyze_feature_types()
        
        # 2. Numerical feature statistics
        print("\n" + "=" * 80)
        print("2. NUMERICAL FEATURE STATISTICS")
        print("=" * 80)
        self._analyze_numerical_features()
        
        # 3. Categorical feature analysis
        print("\n" + "=" * 80)
        print("3. CATEGORICAL FEATURE ANALYSIS")
        print("=" * 80)
        self._analyze_categorical_features()
        
        # 4. Missing value analysis
        print("\n" + "=" * 80)
        print("4. MISSING VALUE ANALYSIS")
        print("=" * 80)
        self._analyze_missing_values()
        
        # 5. Normalization recommendations
        print("\n" + "=" * 80)
        print("5. NORMALIZATION RECOMMENDATIONS")
        print("=" * 80)
        self._recommend_normalization()
        
        # 6. Summary
        print("\n" + "=" * 80)
        print("6. SUMMARY")
        print("=" * 80)
        self._print_summary()
        
        return self.analysis
    
    def _analyze_feature_types(self):
        """Analyze feature types (numerical vs categorical)"""
        numerical_cols = []
        categorical_cols = []
        
        for col in self.feature_cols:
            dtype = self.feature_df[col].dtype
            n_unique = self.feature_df[col].nunique()
            n_samples = len(self.feature_df)
            
            # Determine if categorical
            is_categorical = (
                dtype in ['object', 'category', 'string'] or
                (dtype in ['int64', 'int32', 'int16', 'int8'] and n_unique < 50 and n_unique < n_samples * 0.1)
            )
            
            if is_categorical:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        self.analysis['numerical_cols'] = numerical_cols
        self.analysis['categorical_cols'] = categorical_cols
        
        print(f"Numerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        if categorical_cols:
            print(f"\nCategorical columns: {', '.join(categorical_cols)}")
    
    def _analyze_numerical_features(self):
        """Analyze numerical features"""
        numerical_cols = self.analysis['numerical_cols']
        
        if not numerical_cols:
            print("No numerical features found.")
            self.analysis['numerical_stats'] = {}
            return
        
        stats_list = []
        for col in numerical_cols:
            series = pd.to_numeric(self.feature_df[col], errors='coerce')
            
            stats = {
                'column': col,
                'dtype': str(series.dtype),
                'count': int(series.count()),
                'missing': int(series.isna().sum()),
                'min': float(series.min()) if not series.isna().all() else np.nan,
                'max': float(series.max()) if not series.isna().all() else np.nan,
                'mean': float(series.mean()) if not series.isna().all() else np.nan,
                'median': float(series.median()) if not series.isna().all() else np.nan,
                'std': float(series.std()) if not series.isna().all() else np.nan,
                'q25': float(series.quantile(0.25)) if not series.isna().all() else np.nan,
                'q75': float(series.quantile(0.75)) if not series.isna().all() else np.nan,
            }
            
            # Detect outliers (values beyond 3 standard deviations)
            if not np.isnan(stats['std']) and stats['std'] > 0:
                z_scores = np.abs((series - stats['mean']) / stats['std'])
                stats['outliers_3std'] = int((z_scores > 3).sum())
            else:
                stats['outliers_3std'] = 0
            
            # Detect extreme values
            if not np.isnan(stats['min']) and not np.isnan(stats['max']):
                value_range = stats['max'] - stats['min']
                stats['range'] = value_range
                stats['has_extreme_values'] = value_range > 1e9  # Values in billions/trillions
            else:
                stats['range'] = np.nan
                stats['has_extreme_values'] = False
            
            stats_list.append(stats)
            
            # Print formatted statistics
            print(f"\n{col}:")
            print(f"  Type: {stats['dtype']}")
            print(f"  Range: [{stats['min']:.2e}, {stats['max']:.2e}]")
            print(f"  Mean: {stats['mean']:.2e}, Std: {stats['std']:.2e}")
            print(f"  Median: {stats['median']:.2e}")
            print(f"  IQR: [{stats['q25']:.2e}, {stats['q75']:.2e}]")
            print(f"  Missing: {stats['missing']} ({100*stats['missing']/len(series):.1f}%)")
            if stats['outliers_3std'] > 0:
                print(f"  Outliers (>3σ): {stats['outliers_3std']}")
            if stats['has_extreme_values']:
                print(f"  ⚠️  WARNING: Extreme values detected (range > 1e9)")
        
        self.analysis['numerical_stats'] = {s['column']: s for s in stats_list}
    
    def _analyze_categorical_features(self):
        """Analyze categorical features"""
        categorical_cols = self.analysis['categorical_cols']
        
        if not categorical_cols:
            print("No categorical features found.")
            self.analysis['categorical_stats'] = {}
            return
        
        stats_list = []
        for col in categorical_cols:
            series = self.feature_df[col]
            
            stats = {
                'column': col,
                'dtype': str(series.dtype),
                'n_unique': int(series.nunique()),
                'n_samples': int(len(series)),
                'missing': int(series.isna().sum()),
                'cardinality': int(series.nunique()),
            }
            
            # Get value counts
            value_counts = series.value_counts()
            stats['top_values'] = value_counts.head(10).to_dict()
            stats['value_distribution'] = {
                'most_common': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_common_pct': (value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0
            }
            
            stats_list.append(stats)
            
            # Print formatted statistics
            print(f"\n{col}:")
            print(f"  Type: {stats['dtype']}")
            print(f"  Cardinality: {stats['cardinality']} unique values")
            print(f"  Missing: {stats['missing']} ({100*stats['missing']/len(series):.1f}%)")
            print(f"  Most common: '{value_counts.index[0] if len(value_counts) > 0 else 'N/A'}' "
                  f"({stats['value_distribution']['most_common_pct']:.1f}%)")
            if stats['cardinality'] <= 20:
                print(f"  All values: {list(value_counts.index[:10])}")
        
        self.analysis['categorical_stats'] = {s['column']: s for s in stats_list}
    
    def _analyze_missing_values(self):
        """Analyze missing values"""
        missing_counts = self.feature_df.isna().sum()
        missing_pct = (missing_counts / len(self.feature_df) * 100).round(2)
        
        total_missing = missing_counts.sum()
        features_with_missing = (missing_counts > 0).sum()
        
        print(f"Total missing values: {total_missing}")
        print(f"Features with missing values: {features_with_missing}/{len(self.feature_cols)}")
        
        if features_with_missing > 0:
            print("\nMissing value breakdown:")
            for col in self.feature_cols:
                if missing_counts[col] > 0:
                    print(f"  {col}: {missing_counts[col]} ({missing_pct[col]}%)")
        
        self.analysis['missing_values'] = {
            'total': int(total_missing),
            'features_with_missing': int(features_with_missing),
            'per_feature': missing_counts.to_dict()
        }
    
    def _recommend_normalization(self):
        """Recommend normalization strategies"""
        numerical_stats = self.analysis.get('numerical_stats', {})
        
        recommendations = []
        
        for col, stats in numerical_stats.items():
            rec = {
                'column': col,
                'strategy': None,
                'reason': None
            }
            
            if stats['has_extreme_values']:
                rec['strategy'] = 'z_score'  # (x - mean) / std
                rec['reason'] = 'Extreme values detected - z-score normalization required'
            elif not np.isnan(stats['std']) and stats['std'] > 0:
                # Check if distribution is roughly normal
                if abs(stats['mean']) < 10 * stats['std']:
                    rec['strategy'] = 'z_score'
                    rec['reason'] = 'Normal distribution - z-score recommended'
                else:
                    rec['strategy'] = 'robust'  # (x - median) / IQR
                    rec['reason'] = 'Skewed distribution - robust scaling recommended'
            else:
                rec['strategy'] = 'min_max'  # (x - min) / (max - min)
                rec['reason'] = 'Low variance - min-max scaling'
            
            recommendations.append(rec)
            
            print(f"\n{col}:")
            print(f"  Recommended: {rec['strategy']}")
            print(f"  Reason: {rec['reason']}")
        
        self.analysis['normalization_recommendations'] = recommendations
    
    def _print_summary(self):
        """Print analysis summary"""
        print(f"\nTotal features to process: {len(self.feature_cols)}")
        print(f"  - Numerical: {len(self.analysis['numerical_cols'])}")
        print(f"  - Categorical: {len(self.analysis['categorical_cols'])}")
        
        numerical_stats = self.analysis.get('numerical_stats', {})
        extreme_value_features = [col for col, stats in numerical_stats.items() 
                                 if stats.get('has_extreme_values', False)]
        
        if extreme_value_features:
            print(f"\n⚠️  Features with extreme values (require normalization):")
            for col in extreme_value_features:
                stats = numerical_stats[col]
                print(f"  - {col}: range [{stats['min']:.2e}, {stats['max']:.2e}]")
        
        missing_info = self.analysis.get('missing_values', {})
        if missing_info.get('features_with_missing', 0) > 0:
            print(f"\n⚠️  Features with missing values: {missing_info['features_with_missing']}")
    
    def save_analysis(self, output_path: str):
        """Save analysis results to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        serializable_analysis = convert_to_json_serializable(self.analysis)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"\nAnalysis saved to: {output_path}")
    
    def get_feature_info(self) -> Dict:
        """Get feature information for dataset initialization"""
        return {
            'numerical_cols': self.analysis['numerical_cols'],
            'categorical_cols': self.analysis['categorical_cols'],
            'normalization_strategy': 'z_score',  # Default to z-score for all
            'normalization_recommendations': self.analysis.get('normalization_recommendations', [])
        }


def analyze_tabular_data(csv_path: str, output_dir: str = None, exclude_cols: List[str] = None) -> TabularDataAnalyzer:
    """
    Analyze tabular data and generate report
    
    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save analysis report (optional)
        exclude_cols: Columns to exclude from analysis
    
    Returns:
        TabularDataAnalyzer instance with analysis results
    """
    analyzer = TabularDataAnalyzer(csv_path, exclude_cols)
    analyzer.analyze()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tabular_analysis.json')
        analyzer.save_analysis(output_path)
    
    return analyzer


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_analysis.py <csv_path> [output_dir]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_tabular_data(csv_path, output_dir)

