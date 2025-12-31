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
import torch.distributed as dist
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
import torchmetrics

# Add src/models/TIP to path for relative imports (like other TIP scripts)
tip_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if tip_root not in sys.path:
    sys.path.insert(0, tip_root)

# Add project root to path (for accessing src.data.augmentations if needed)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.ConstructionCostTIPDataset import ConstructionCostTIPDataset
from models.ConstructionCostPrediction import ConstructionCostPrediction
from omegaconf import OmegaConf, DictConfig, open_dict


# TIPRegressionModel removed - ConstructionCostPrediction is used directly


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


def _gather_tensors(tensor_list, world_size):
    """
    Gather tensors from all GPUs and concatenate them.
    
    Args:
        tensor_list: List of tensors from current GPU
        world_size: Number of GPUs
    
    Returns:
        Concatenated tensor from all GPUs (on rank 0), or original tensor if single GPU
    """
    if world_size == 1:
        return tensor_list
    
    # Gather all tensors from all GPUs
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, tensor_list)
    
    # Concatenate all gathered tensors
    all_tensors = []
    for gpu_tensors in gathered_list:
        all_tensors.extend(gpu_tensors)
    
    return all_tensors


def _gather_lists(list_data, world_size):
    """
    Gather lists (like data_ids) from all GPUs and concatenate them.
    
    Args:
        list_data: List from current GPU
        world_size: Number of GPUs
    
    Returns:
        Concatenated list from all GPUs (on rank 0), or original list if single GPU
    """
    if world_size == 1:
        return list_data
    
    # Gather all lists from all GPUs
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, list_data)
    
    # Concatenate all gathered lists
    all_lists = []
    for gpu_list in gathered_list:
        all_lists.extend(gpu_list)
    
    return all_lists


def evaluate_validation(
    model: ConstructionCostPrediction,
    val_loader: DataLoader,
    device: torch.device,
    save_predictions: bool = True,
    output_dir: str = None,
    checkpoint_path: Optional[str] = None
) -> dict:
    """
    Evaluate model on validation set.
    
    Supports multi-GPU evaluation by gathering predictions from all GPUs.
    Only rank 0 saves results to avoid duplicate files.
    
    Returns:
        Dictionary with metrics: {'rmsle': float, 'mae': float, 'rmse': float, 'r2': float}
    """
    model.eval()
    
    # Check if running in distributed mode
    is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0
    
    all_predictions = []
    all_targets = []
    all_data_ids = []
    
    # Calculate total batches for progress (only print from rank 0)
    total_batches = len(val_loader)
    if rank == 0:
        print(f"Total batches: {total_batches}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Print progress for every batch (single line, updates in place) - only from rank 0
            if rank == 0:
                progress_pct = 100 * (batch_idx + 1) / total_batches
                print(f"\r  Validation: {batch_idx + 1}/{total_batches} batches ({progress_pct:.1f}%)", end='', flush=True)
            
            # Unpack batch: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id, country)
            if len(batch) != 9:
                raise ValueError(f"Expected batch size 9, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id, country)")
            imaging_views = batch[0]
            tabular_views = batch[1]
            label = batch[2]
            unaugmented_image = batch[3]
            unaugmented_tabular = batch[4]
            target_log = batch[5]  # Target in log space (log1p(cost), NOT normalized, from dataloader)
            target_original = batch[6]  # Original scale target (for metrics)
            data_id = batch[7]
            country = batch[8]  # Country labels (not used in evaluation, but present in batch)
            
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
            
            # Forward pass - model returns dict with predictions
            result = model((x_image, x_tabular, mask))  # Returns dict
            
            # Extract prediction in original scale (already decoded by head)
            pred_original = result['prediction_original']  # (B,) - original scale (USD/m²)
            
            # Store predictions in original scale
            all_predictions.append(pred_original.cpu())
            
            # Use target_original (ground truth in original USD/m² scale) for metrics
            if target_original is not None and target_original.numel() > 0:
                # target_original is already in original scale, no denormalization needed
                all_targets.append(target_original.cpu())
            
            # Store data IDs (data_id is a list from the batch)
            if data_id is not None:
                all_data_ids.extend(data_id)
    
    # Gather predictions from all GPUs if running in distributed mode
    if is_distributed:
        all_predictions = _gather_tensors(all_predictions, world_size)
        all_targets = _gather_tensors(all_targets, world_size) if all_targets else []
        all_data_ids = _gather_lists(all_data_ids, world_size) if all_data_ids else []
    
    # Concatenate all predictions (now from all GPUs if distributed)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Compute metrics if targets are available (only on rank 0 to avoid duplicate computation)
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
        
        # Only print and save from rank 0 to avoid duplicate output/files
        if rank == 0:
            print("\n" + "="*60)
            print("VALIDATION METRICS")
            print("="*60)
            print(f"RMSLE: {rmsle:.6f}")
            print(f"MAE:   {mae:.2f} USD/m²")
            print(f"RMSE:  {rmse:.2f} USD/m²")
            print(f"R²:    {r2:.6f}")
            print("="*60 + "\n")
            
            # Save predictions if requested (only from rank 0)
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
    model: ConstructionCostPrediction,
    test_loader: DataLoader,
    device: torch.device,
    output_path: str,
    checkpoint_path: Optional[str] = None,
    data_ids: list = None
):
    """
    Run inference on test set and generate submission CSV.
    
    Supports multi-GPU inference by gathering predictions from all GPUs.
    Only rank 0 saves results to avoid duplicate files.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run inference on
        output_path: Path to save submission CSV
        checkpoint_path: Path to checkpoint (for filename generation)
        data_ids: List of data IDs (if available from dataset)
    """
    model.eval()
    
    # Check if running in distributed mode
    is_distributed = dist.is_initialized() if hasattr(dist, 'is_initialized') else False
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0
    
    all_predictions = []
    all_data_ids = []
    
    # Calculate total batches for progress (only print from rank 0)
    total_batches = len(test_loader)
    if rank == 0:
        print(f"Total batches: {total_batches}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Print progress for every batch (single line, updates in place) - only from rank 0
            if rank == 0:
                progress_pct = 100 * (batch_idx + 1) / total_batches
                print(f"\r  Test inference: {batch_idx + 1}/{total_batches} batches ({progress_pct:.1f}%)", end='', flush=True)
            
            # Unpack batch: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id, country)
            if len(batch) != 9:
                raise ValueError(f"Expected batch size 9, got {len(batch)}. Batch format: (imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id, country)")
            imaging_views = batch[0]
            tabular_views = batch[1]
            label = batch[2]
            unaugmented_image = batch[3]
            unaugmented_tabular = batch[4]
            target_log = batch[5]  # Not used for test set (may be dummy 0.0)
            target_original = batch[6]  # Not used for test set
            data_id = batch[7]
            country = batch[8]  # Country labels (not used in inference, but present in batch)
            
            # Use unaugmented views for inference
            x_image = unaugmented_image.to(device)  # (B, 15, H, W)
            x_tabular = unaugmented_tabular.to(device)  # (B, N_features)
            
            # Create mask (all ones for no missing values)
            mask = torch.ones_like(x_tabular, dtype=torch.bool).to(device)
            
            # Forward pass - model returns dict with predictions
            result = model((x_image, x_tabular, mask))  # Returns dict
            
            # Extract prediction in original scale (already decoded by head)
            pred_original = result['prediction_original']  # (B,) - original scale (USD/m²)
            
            # Store predictions
            all_predictions.append(pred_original.cpu())
            
            # Store data IDs (data_id is a list from the batch)
            if data_id is not None:
                all_data_ids.extend(data_id)
        
        if rank == 0:
            print()  # New line after progress
        
    # Gather predictions from all GPUs if running in distributed mode
    if is_distributed:
        all_predictions = _gather_tensors(all_predictions, world_size)
        all_data_ids = _gather_lists(all_data_ids, world_size) if all_data_ids else []
    
    # Concatenate all predictions (now from all GPUs if distributed)
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    
    # Only save from rank 0 to avoid duplicate files
    if rank == 0:
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
        print(f"\n✅ Inference complete! Processed {len(all_predictions)} test samples")
        print(f"✅ Submission saved to: {output_path}")
        print(f"   Number of predictions: {len(submission_df)}")
        print(f"   Prediction range: [{all_predictions.min():.2f}, {all_predictions.max():.2f}] USD/m²")
        print("="*60)
        print(f"   Prediction mean: {all_predictions.mean():.2f} USD/m²")
        print(f"   Prediction std: {all_predictions.std():.2f} USD/m²")


def run_evaluation(args):
    """
    Run evaluation programmatically (can be called from other scripts).
    
    Args can be either argparse.Namespace or a dict-like object with the required attributes.
    """
    
    # Always use checkpoint directory for evaluation (ignore --output_dir if provided)
    # Extract checkpoint directory and create evaluation folder structure
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    args.output_dir = os.path.join(checkpoint_dir, 'evaluation')
    print(f"Using checkpoint directory for evaluation: {checkpoint_dir}")
    
    # Create evaluation folder structure
    val_eval_dir = os.path.join(args.output_dir, 'val_evaluation')
    test_submission_dir = os.path.join(args.output_dir, 'test_submission')
    os.makedirs(val_eval_dir, exist_ok=True)
    os.makedirs(test_submission_dir, exist_ok=True)
    print(f"Evaluation folders created:")
    print(f"  - Validation: {val_eval_dir}")
    print(f"  - Test submission: {test_submission_dir}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint to get hyperparameters
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = OmegaConf.create(checkpoint['hyper_parameters'])
    
    # Use open_dict to allow adding new keys (OmegaConf struct mode)
    with open_dict(hparams):
        # Set required attributes
        if not hasattr(hparams, 'algorithm_name'):
            hparams.algorithm_name = 'tip'
        if not hasattr(hparams, 'missing_tabular'):
            hparams.missing_tabular = False
        # Set checkpoint and field_lengths paths from command line args
        hparams.checkpoint = args.checkpoint
        hparams.field_lengths_tabular = args.field_lengths
    
    # Create ConstructionCostPrediction model (handles all head loading logic)
    model = ConstructionCostPrediction(hparams=hparams)
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    else:
        print("Backbone trainable")
    
    model = model.to(device)
    
    # Extract target normalization stats from checkpoint hyperparameters (for display only)
    # NOTE: The head already handles decoding internally, so these are just for logging
    # The head's forward() returns prediction_original which is already in USD/m² scale
    target_mean = None
    target_std = None
    target_mean_by_country = None
    target_std_by_country = None
    head_type = None
    
    # Check if regression_head config exists (new nested format)
    if hasattr(hparams, 'regression_head') and isinstance(hparams.regression_head, (dict, DictConfig)):
        regression_head = hparams.regression_head
        if isinstance(regression_head, DictConfig):
            regression_head = OmegaConf.to_container(regression_head, resolve=True)
        head_type = regression_head.get('type', None)
        
        # Check for country-specific stats (MultiTaskCountryAwareRegression)
        if 'target_mean_by_country' in regression_head and 'target_std_by_country' in regression_head:
            target_mean_by_country = regression_head.get('target_mean_by_country', {})
            target_std_by_country = regression_head.get('target_std_by_country', {})
            # Convert DictConfig to dict if needed
            if isinstance(target_mean_by_country, DictConfig):
                target_mean_by_country = OmegaConf.to_container(target_mean_by_country, resolve=True)
            if isinstance(target_std_by_country, DictConfig):
                target_std_by_country = OmegaConf.to_container(target_std_by_country, resolve=True)
        else:
            # Fallback to global stats
            target_mean = float(regression_head.get('target_mean', 0.0))
            target_std = float(regression_head.get('target_std', 1.0))
    # Fallback to flat hparams (old format)
    elif hasattr(hparams, 'target_mean') and hasattr(hparams, 'target_std'):
        target_mean = float(hparams.target_mean)
        target_std = float(hparams.target_std)
    
    # Display model info
    print(f"Model initialized (from checkpoint):")
    print(f"  Embedding dim: {hparams.multimodal_embedding_dim}")
    if target_mean_by_country is not None and target_std_by_country is not None:
        print(f"  Head type: {head_type} (using country-specific normalization)")
        print(f"  Country-specific target statistics:")
        for country_id in sorted(target_mean_by_country.keys()):
            mean_val = target_mean_by_country[country_id]
            std_val = target_std_by_country[country_id]
            print(f"    Country {country_id}: mean={mean_val:.4f}, std={std_val:.4f}")
    elif target_mean is not None and target_std is not None:
        print(f"  Target mean: {target_mean:.4f} (from checkpoint - head uses this internally)")
        print(f"  Target std: {target_std:.4f} (from checkpoint - head uses this internally)")
    else:
        print(f"  ⚠️  Could not extract target_mean/std from checkpoint (head should still work if loaded correctly)")
    
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
        metadata_path=args.val_metadata
    )
    
    # NOTE: Target normalization is handled internally by the head
    # The head's forward() method returns prediction_original which is already decoded to USD/m²
    # No additional processing needed in evaluation pipeline
    if target_mean_by_country is not None and target_std_by_country is not None:
        print(f"\n✅ Head loaded with country-specific target normalization from checkpoint:")
        for country_id in sorted(target_mean_by_country.keys()):
            mean_val = target_mean_by_country[country_id]
            std_val = target_std_by_country[country_id]
            print(f"   Country {country_id}: mean={mean_val:.4f}, std={std_val:.4f} (used internally by head for decoding)")
        print(f"   Predictions are already in original scale (USD/m²) - no additional decoding needed")
    elif target_mean is not None and target_std is not None:
        print(f"\n✅ Head loaded with target normalization from checkpoint:")
        print(f"   Target mean: {target_mean:.4f} (used internally by head for decoding)")
        print(f"   Target std: {target_std:.4f} (used internally by head for decoding)")
        print(f"   Predictions are already in original scale (USD/m²) - no additional decoding needed")
    
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
    # Use val_evaluation directory for validation results (already created above)
    metrics, val_predictions, val_data_ids = evaluate_validation(
        model=model,
        val_loader=val_loader,
        device=device,
        save_predictions=True,
        output_dir=val_eval_dir,
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
        metadata_path=args.test_metadata
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
    # Use test_submission directory for test results (already created above)
    submission_path = os.path.join(test_submission_dir, 'submission.csv')
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
    print(f"  - Validation predictions: {val_eval_dir}")
    print(f"  - Test submission: {submission_path}")


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
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (if None, will use checkpoint directory/evaluation/)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone during evaluation')
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == '__main__':
    main()

