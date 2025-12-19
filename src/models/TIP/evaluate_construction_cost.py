"""
Evaluation script for Construction Cost Prediction using TIP pretrained model.

This script:
1. Loads a pretrained TIP checkpoint
2. Evaluates on validation set and computes RMSLE
3. Runs inference on test set and generates submission CSV
"""
import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
import torchmetrics

# Add src/models/TIP to path for relative imports (like other TIP scripts)
tip_root = os.path.abspath(os.path.dirname(__file__))
if tip_root not in sys.path:
    sys.path.insert(0, tip_root)

# Add project root to path (for accessing src.data.augmentations if needed)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.ConstructionCostTIPDataset import ConstructionCostTIPDataset
from models.Tip_utils.Tip_downstream import TIPBackbone
from omegaconf import OmegaConf, DictConfig, open_dict


class TIPRegressionModel(nn.Module):
    """TIP model with regression head for construction cost prediction."""
    def __init__(self, checkpoint_path: str, field_lengths_path: str, freeze_backbone: bool = True):
        super().__init__()
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hparams = OmegaConf.create(checkpoint['hyper_parameters'])
        
        # Use open_dict to allow adding new keys (OmegaConf struct mode)
        with open_dict(hparams):
            # Create TIP backbone with regression head
            hparams.checkpoint = checkpoint_path
            hparams.field_lengths_tabular = field_lengths_path
            
            # Set required attributes
            if not hasattr(hparams, 'algorithm_name'):
                hparams.algorithm_name = 'tip'
            if not hasattr(hparams, 'missing_tabular'):
                hparams.missing_tabular = False
            
            # Set task to regression to use regression head
            hparams.task = 'regression'
            hparams.num_classes = 1  # Regression: single output
        
        self.backbone = TIPBackbone(hparams)
        
        # Try to load online regression head weights from checkpoint callback state
        # The online regression head was trained during pretraining and saved in callback state
        # This is critical - without loading these weights, we use a random regression head!
        online_regression_loaded = False
        if 'callbacks' in checkpoint:
            # Find the SSLOnlineEvaluatorRegression callback state
            # PyTorch Lightning saves callbacks with keys like: "SSLOnlineEvaluatorRegression{hash}"
            for key in checkpoint['callbacks'].keys():
                if 'SSLOnlineEvaluatorRegression' in key or 'ssl_online_regression' in key.lower():
                    callback_state = checkpoint['callbacks'][key]
                    if 'state_dict' in callback_state:
                        try:
                            # The online regression head has a different architecture than TIPBackbone's classifier
                            # We need to create a matching regression head and load the weights
                            from utils.ssl_online_regression import RegressionMLP
                            z_dim = hparams.multimodal_embedding_dim
                            hidden_dim = getattr(hparams, 'embedding_dim', z_dim)
                            
                            # Create the same architecture as the online evaluator
                            online_regression_head = RegressionMLP(
                                n_input=z_dim,
                                n_hidden=hidden_dim,
                                p=0.2  # Default dropout
                            )
                            
                            # Load the trained weights
                            online_regression_head.load_state_dict(callback_state['state_dict'])
                            
                            # Replace the random classifier with the trained one
                            self.backbone.classifier = online_regression_head
                            print("✅ Loaded trained online regression head weights from checkpoint")
                            online_regression_loaded = True
                            break
                        except Exception as e:
                            print(f"⚠️  Warning: Could not load online regression head weights: {e}")
                            import traceback
                            traceback.print_exc()
        
        if not online_regression_loaded:
            print("⚠️  WARNING: No online regression head weights found in checkpoint!")
            print("   Using randomly initialized regression head from TIPBackbone")
            print("   This will result in poor performance (random weights)")
            print("   Expected RMSLE: ~1.2 (random) vs ~0.3 (trained)")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen")
        else:
            print("Backbone trainable")
        
        # Load target normalization stats from config file only
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'configs', 'config_construction_cost_pretrain.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = OmegaConf.load(config_path)
        self.target_mean = float(config.get('target_mean', 0.0))
        self.target_std = float(config.get('target_std', 1.0))
        self.target_log_transform = bool(config.get('target_log_transform', True))
        
        print(f"Model initialized (from checkpoint):")
        print(f"  Embedding dim: {hparams.multimodal_embedding_dim}")
        print(f"  Target mean: {self.target_mean:.4f} (from config)")
        print(f"  Target std: {self.target_std:.4f} (from config)")
        print(f"  Log transform: {self.target_log_transform}")
    
    def forward(self, x: Tuple) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tuple of (image, tabular) or (image, tabular, mask)
        
        Returns:
            prediction: (B, 1) - Predicted cost in normalized log space
        """
        # TIPBackbone already includes regression head
        prediction = self.backbone(x)  # (B, 1)
        
        return prediction
    
    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Denormalize predictions to original scale (USD/m²).
        
        Args:
            y: Normalized predictions (in log space if log_transform=True)
        
        Returns:
            Predictions in original scale (USD/m²)
        """
        # Denormalize: y_orig = y_norm * std + mean
        y_denorm = y * self.target_std + self.target_mean
        
        # Reverse log-transform: exp(y) - 1
        if self.target_log_transform:
            y_denorm = torch.expm1(y_denorm)  # exp(y) - 1
        
        return y_denorm


def compute_rmsle(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Logarithmic Error.
    
    RMSLE = sqrt(mean((log(y_true + 1) - log(y_pred + 1))^2))
    """
    # Ensure both are in original scale (USD/m²)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Clip to avoid log(0)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    
    # Compute RMSLE
    log_true = np.log1p(y_true)  # log(1 + y)
    log_pred = np.log1p(y_pred)  # log(1 + y)
    
    rmsle = np.sqrt(np.mean((log_true - log_pred) ** 2))
    
    return rmsle


def evaluate_validation(
    model: TIPRegressionModel,
    val_loader: DataLoader,
    device: torch.device,
    save_predictions: bool = True,
    output_dir: str = None,
    checkpoint_path: Optional[str] = None
) -> dict:
    """
    Evaluate model on validation set.
    
    Returns:
        Dictionary with metrics: {'rmsle': float, 'mae': float, 'rmse': float, 'r2': float}
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_data_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id)
            if len(batch) != 8:
                raise ValueError(f"Expected batch size 8, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id)")
            imaging_views = batch[0]
            tabular_views = batch[1]
            label = batch[2]
            unaugmented_image = batch[3]
            unaugmented_tabular = batch[4]
            target = batch[5]  # Log-transformed target (for reference)
            target_original = batch[6]  # Original scale target (for metrics)
            data_id = batch[7]
            
            # Use unaugmented views for evaluation
            # Format: (image, tabular, mask)
            # image: (B, 15, H, W) - concatenated Sentinel-2 + VIIRS
            # tabular: (B, N_features) - tabular features
            # mask: (B, N_features) - mask for missing values (not needed for inference)
            
            # Prepare input
            x_image = unaugmented_image.to(device)  # (B, 15, H, W)
            x_tabular = unaugmented_tabular.to(device)  # (B, N_features)
            
            # Create mask (all ones for no missing values)
            mask = torch.ones_like(x_tabular, dtype=torch.bool).to(device)
            
            # Forward pass - model predicts in normalized log space
            pred_normalized = model((x_image, x_tabular, mask))  # (B, 1)
            
            # Convert prediction from normalized log space to original scale (USD/m²)
            # Step 1: Denormalize (reverse normalization)
            pred_log = (pred_normalized.squeeze() * model.target_std) + model.target_mean  # (B,)
            
            # Step 2: Reverse log-transform to get original scale
            if model.target_log_transform:
                pred_original = torch.expm1(pred_log)  # exp(x) - 1
            else:
                pred_original = pred_log
            
            # Ensure non-negative
            pred_original = torch.clamp(pred_original, min=0.0)
            
            # Store predictions in original scale
            all_predictions.append(pred_original.cpu())
            
            # Use target_original (ground truth in original USD/m² scale) for metrics
            if target_original is not None and target_original.numel() > 0:
                # target_original is already in original scale, no denormalization needed
                all_targets.append(target_original.cpu())
            
            # Store data IDs (data_id is a list from the batch)
            if data_id is not None:
                all_data_ids.extend(data_id)
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Compute metrics if targets are available
    metrics = {}
    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute RMSLE
        rmsle = compute_rmsle(all_targets, all_predictions)
        metrics['rmsle'] = rmsle
        
        # Compute other metrics
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
        
        # R²
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot).item()
        
        metrics['mae'] = mae
        metrics['rmse'] = rmse
        metrics['r2'] = r2
        
        print("\n" + "="*60)
        print("VALIDATION METRICS")
        print("="*60)
        print(f"RMSLE: {rmsle:.6f}")
        print(f"MAE:   {mae:.2f} USD/m²")
        print(f"RMSE:  {rmse:.2f} USD/m²")
        print(f"R²:    {r2:.6f}")
        print("="*60 + "\n")
        
        # Save predictions if requested
        if save_predictions and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pred_df = pd.DataFrame({
                'data_id': all_data_ids if all_data_ids else [f'sample_{i}' for i in range(len(all_predictions))],
                'construction_cost_per_m2_usd': all_predictions.numpy(),  # Prediction (for submission format)
                'ground_truth': all_targets.numpy(),  # Ground truth (additional column)
                'error': (all_predictions - all_targets).numpy(),
                'abs_error': torch.abs(all_predictions - all_targets).numpy()
            })
            # Generate filename with checkpoint name and timestamp
            if checkpoint_path:
                checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            else:
                checkpoint_name = 'unknown'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pred_path = os.path.join(output_dir, f'val_predictions_{checkpoint_name}_{timestamp}.csv')
            pred_df.to_csv(pred_path, index=False)
            print(f"Saved validation predictions to: {pred_path}")
    
    return metrics, all_predictions, all_data_ids


def run_inference(
    model: TIPRegressionModel,
    test_loader: DataLoader,
    device: torch.device,
    output_path: str,
    checkpoint_path: Optional[str] = None,
    data_ids: list = None
):
    """
    Run inference on test set and generate submission CSV.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run inference on
        output_path: Path to save submission CSV
        data_ids: List of data IDs (if available from dataset)
    """
    model.eval()
    
    all_predictions = []
    all_data_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id)
            if len(batch) != 8:
                raise ValueError(f"Expected batch size 8, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id)")
            imaging_views = batch[0]
            tabular_views = batch[1]
            label = batch[2]
            unaugmented_image = batch[3]
            unaugmented_tabular = batch[4]
            target = batch[5]  # Not used for test set
            target_original = batch[6]  # Not used for test set
            data_id = batch[7]
            
            # Use unaugmented views for inference
            x_image = unaugmented_image.to(device)  # (B, 15, H, W)
            x_tabular = unaugmented_tabular.to(device)  # (B, N_features)
            
            # Create mask (all ones for no missing values)
            mask = torch.ones_like(x_tabular, dtype=torch.bool).to(device)
            
            # Forward pass - model predicts in normalized log space
            pred_normalized = model((x_image, x_tabular, mask))  # (B, 1)
            
            # Convert prediction from normalized log space to original scale (USD/m²)
            # Step 1: Denormalize (reverse normalization)
            pred_log = (pred_normalized.squeeze() * model.target_std) + model.target_mean  # (B,)
            
            # Step 2: Reverse log-transform to get original scale
            if model.target_log_transform:
                pred_original = torch.expm1(pred_log)  # exp(x) - 1
            else:
                pred_original = pred_log
            
            # Ensure non-negative
            pred_original = torch.clamp(pred_original, min=0.0)
            
            # Store predictions
            all_predictions.append(pred_original.cpu())
            
            # Store data IDs (data_id is a list from the batch)
            if data_id is not None:
                all_data_ids.extend(data_id)
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    
    # Create submission DataFrame
    if all_data_ids:
        submission_df = pd.DataFrame({
            'data_id': all_data_ids,
            'construction_cost_per_m2_usd': all_predictions
        })
    else:
        # If no data IDs, use sequential indices
        submission_df = pd.DataFrame({
            'data_id': [f'test_{i}' for i in range(len(all_predictions))],
            'construction_cost_per_m2_usd': all_predictions
        })
    
    # Generate filename with checkpoint name and timestamp
    if checkpoint_path:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # If output_path is a directory or ends with '/', create filename inside it
        if os.path.isdir(output_path) or output_path.endswith('/'):
            output_path = os.path.join(output_path, f'submission_{checkpoint_name}_{timestamp}.csv')
        else:
            # If output_path is a file path, insert checkpoint name and timestamp before extension
            base_dir = os.path.dirname(output_path) or '.'
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            ext = os.path.splitext(output_path)[1] or '.csv'
            output_path = os.path.join(base_dir, f'{base_name}_{checkpoint_name}_{timestamp}{ext}')
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved to: {output_path}")
    print(f"   Number of predictions: {len(submission_df)}")
    print(f"   Prediction range: [{all_predictions.min():.2f}, {all_predictions.max():.2f}] USD/m²")
    print(f"   Prediction mean: {all_predictions.mean():.2f} USD/m²")
    print(f"   Prediction std: {all_predictions.std():.2f} USD/m²")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TIP model for construction cost prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained checkpoint (.ckpt file)')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='Path to validation CSV (with targets)')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV (without targets)')
    parser.add_argument('--composite_dir_trainval', type=str, required=True,
                        help='Directory containing satellite TIFF files for train/val sets')
    parser.add_argument('--composite_dir_test', type=str, required=True,
                        help='Directory containing satellite TIFF files for test set')
    parser.add_argument('--field_lengths', type=str, required=True,
                        help='Path to field_lengths.pt file')
    parser.add_argument('--val_metadata', type=str, required=True,
                        help='Path to validation metadata.pkl file')
    parser.add_argument('--test_metadata', type=str, required=True,
                        help='Path to test metadata.pkl file')
    parser.add_argument('--output_dir', type=str, default='work_dir/evaluation',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone during evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    model = TIPRegressionModel(
        checkpoint_path=args.checkpoint,
        field_lengths_path=args.field_lengths,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Load validation dataset
    print("\n" + "="*60)
    print("LOADING VALIDATION DATASET")
    print("="*60)
    print(f"Using composite directory: {args.composite_dir_trainval}")
    val_dataset = ConstructionCostTIPDataset(
        csv_path=args.val_csv,
        composite_dir=args.composite_dir_trainval,
        field_lengths_tabular=args.field_lengths,
        labels_path=None,  # Not needed for evaluation
        img_size=224,
        is_train=False,  # No augmentation for validation
        corruption_rate=0.0,  # No corruption for evaluation
        augmentation_rate=0.0,  # No augmentation
        metadata_path=args.val_metadata,
        target_log_transform=True
    )
    
    # Target normalization stats: only from config/checkpoint (never from pickle/metadata)
    if (model.target_mean, model.target_std) != (0.0, 1.0):
        print(f"\n✅ Using target normalization from config/checkpoint (log space):")
        print(f"   Target mean: {model.target_mean:.4f}")
        print(f"   Target std: {model.target_std:.4f}")
    else:
        print(f"\n⚠️  WARNING: target_mean and target_std are still defaults (0.0, 1.0)")
        print(f"   This will cause incorrect denormalization!")
        print(f"   Please set them in config file after running preprocessing.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("EVALUATING ON VALIDATION SET")
    print("="*60)
    metrics, val_predictions, val_data_ids = evaluate_validation(
        model=model,
        val_loader=val_loader,
        device=device,
        save_predictions=True,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint
    )
    
    # Load test dataset
    print("\n" + "="*60)
    print("LOADING TEST DATASET")
    print("="*60)
    print(f"Using composite directory: {args.composite_dir_test}")
    test_dataset = ConstructionCostTIPDataset(
        csv_path=args.test_csv,
        composite_dir=args.composite_dir_test,
        field_lengths_tabular=args.field_lengths,
        labels_path=None,  # Not needed for inference
        img_size=224,
        is_train=False,  # No augmentation for test
        corruption_rate=0.0,  # No corruption for inference
        augmentation_rate=0.0,  # No augmentation
        metadata_path=args.test_metadata,
        target_log_transform=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Run inference on test set
    print("\n" + "="*60)
    print("RUNNING INFERENCE ON TEST SET")
    print("="*60)
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        output_path=submission_path,
        checkpoint_path=args.checkpoint,
        data_ids=None  # Will extract from dataset
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"  - Validation metrics: RMSLE={metrics.get('rmsle', 'N/A'):.6f}")
    print(f"  - Submission file: {submission_path}")


if __name__ == '__main__':
    main()

