"""
Fine-tuning script for Construction Cost Prediction.

This script:
1. Loads pretraining checkpoint (has all architecture hyperparameters)
2. Creates ConstructionCostFinetuning with regression head
3. Fine-tunes only the regression head (backbone frozen)
4. Saves checkpoint with all necessary hyperparameters (architecture from pretrain + fine-tuning config)
"""
import os
import glob
import re
from os.path import join
from types import SimpleNamespace
from omegaconf import open_dict, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch import cuda
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import pandas as pd
import numpy as np

from datasets.ConstructionCostTIPDataset import ConstructionCostTIPDataset
from models.ConstructionCostFinetuning import ConstructionCostFinetuning
from utils.utils import grab_arg_from_checkpoint, create_logdir
from trainers.evaluate_construction_cost import run_evaluation


def _run_evaluation_for_fold(logdir, hparams, evaluator_config):
    """
    Run evaluation for a single fold (or fixed split if fold_index is None).
    
    Args:
        logdir: Directory containing the checkpoint files
        hparams: Hyperparameters (for eval_metric)
        evaluator_config: Evaluator configuration from hparams
    
    Returns:
        bool: True if evaluation succeeded, False otherwise
    """
    checkpoint_type = getattr(evaluator_config, 'checkpoint_type', 'best')
    checkpoint_path = None
    
    if checkpoint_type == 'best':
        # Find best checkpoint (format: checkpoint_best_{metric}_{epoch}_{value}.ckpt)
        best_pattern = join(logdir, f'checkpoint_best_{hparams.eval_metric}_*.ckpt')
        best_files = glob.glob(best_pattern)
        if best_files:
            # Sort by epoch number in filename (highest epoch = most recent)
            def get_epoch(filename):
                match = re.search(rf'checkpoint_best_{hparams.eval_metric}_(\d+)_', filename)
                return int(match.group(1)) if match else 0
            checkpoint_path = sorted(best_files, key=get_epoch, reverse=True)[0]
        else:
            print(f"⚠️  No best checkpoint found matching pattern: {best_pattern}")
            print(f"   Falling back to last checkpoint")
            checkpoint_type = 'last'
    
    if checkpoint_path is None or checkpoint_type == 'last':
        # Find last checkpoint (format: checkpoint_last_epoch_{epoch}.ckpt)
        last_pattern = join(logdir, 'checkpoint_last_epoch_*.ckpt')
        last_files = glob.glob(last_pattern)
        if last_files:
            # Sort by epoch number in filename
            def get_epoch(filename):
                match = re.search(r'checkpoint_last_epoch_(\d+)\.ckpt', filename)
                return int(match.group(1)) if match else 0
            checkpoint_path = sorted(last_files, key=get_epoch, reverse=True)[0]
        else:
            print(f"⚠️  No checkpoint found in {logdir}")
            return False
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Checkpoint type: {checkpoint_type}")
    
    # Create args object for evaluation
    eval_args = SimpleNamespace(
        checkpoint=checkpoint_path,
        val_csv=getattr(evaluator_config, 'val_csv'),
        test_csv=getattr(evaluator_config, 'test_csv'),
        composite_dir_trainval=getattr(evaluator_config, 'composite_dir_trainval'),
        composite_dir_test=getattr(evaluator_config, 'composite_dir_test'),
        field_lengths=getattr(evaluator_config, 'field_lengths'),
        val_metadata=getattr(evaluator_config, 'val_metadata'),
        test_metadata=getattr(evaluator_config, 'test_metadata'),
        batch_size=getattr(evaluator_config, 'batch_size', 32),
        num_workers=getattr(evaluator_config, 'num_workers', 2),
        device=getattr(evaluator_config, 'device', 'cuda'),
        freeze_backbone=True,
        output_dir=None  # Will be set to checkpoint_dir/evaluation/
    )
    
    try:
        run_evaluation(eval_args)
        return True
    except Exception as e:
        print(f"⚠️  Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_datasets(hparams, fold_index=None):
    """Load ConstructionCostTIPDataset for train and validation.
    
    Args:
        hparams: Hyperparameters
        fold_index: Current fold index (0 to k_fold-1). If None and use_kfold=True, 
                    will use k_fold_current from hparams. If -1, will raise error.
    """
    from sklearn.model_selection import StratifiedKFold
    
    use_kfold = getattr(hparams, 'use_kfold', False)
    
    if use_kfold:
        # K-Fold Cross-Validation: Read from unified trainval.csv and split
        print("="*60)
        print("USING STRATIFIED K-FOLD CROSS-VALIDATION (FINE-TUNING)")
        print("="*60)
        
        # Read unified trainval CSV
        trainval_csv = getattr(hparams, 'data_trainval_tabular', None)
        if trainval_csv is None:
            raise ValueError("data_trainval_tabular must be specified when use_kfold=true")
        
        print(f"Reading unified trainval CSV: {trainval_csv}")
        df_trainval = pd.read_csv(trainval_csv)
        print(f"Total samples in trainval: {len(df_trainval)}")
        
        # Get k-fold parameters
        k_fold = getattr(hparams, 'k_fold', 5)
        k_fold_seed = getattr(hparams, 'k_fold_seed', 42)
        
        # Use fold_index if provided, otherwise use k_fold_current from hparams
        if fold_index is None:
            fold_index = getattr(hparams, 'k_fold_current', 0)
        
        if fold_index < 0 or fold_index >= k_fold:
            raise ValueError(f"fold_index must be between 0 and {k_fold-1}, got {fold_index}")
        
        print(f"K-Fold parameters: k={k_fold}, seed={k_fold_seed}, current_fold={fold_index}")
        
        # Stratify by country for balanced distribution
        stratify_col = 'country'
        if stratify_col not in df_trainval.columns:
            raise ValueError(f"Stratify column '{stratify_col}' not found in CSV. Cannot perform balanced split.")
        
        stratify_labels = df_trainval[stratify_col].values
        print(f"✅ Stratifying by: {stratify_col}")
        print(f"   Original distribution:")
        orig_dist = df_trainval[stratify_col].value_counts().sort_index()
        for val, count in orig_dist.items():
            pct = count / len(df_trainval) * 100
            print(f"     {stratify_col}={val}: {count} samples ({pct:.2f}%)")
        
        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=k_fold_seed)
        
        # Get indices for current fold (stratified by country)
        indices = np.arange(len(df_trainval))
        train_indices, val_indices = list(skf.split(indices, stratify_labels))[fold_index]
        
        # Split dataframe
        df_train = df_trainval.iloc[train_indices].reset_index(drop=True)
        df_val = df_trainval.iloc[val_indices].reset_index(drop=True)
        
        print(f"\nFold {fold_index}: Train={len(df_train)} samples, Val={len(df_val)} samples")
        
        # Verify balanced distribution
        print(f"\nDistribution verification:")
        print(f"Train - {stratify_col} distribution:")
        train_dist = df_train[stratify_col].value_counts().sort_index()
        for val, count in train_dist.items():
            pct = count / len(df_train) * 100
            print(f"  {stratify_col}={val}: {count} samples ({pct:.2f}%)")
        
        print(f"Val - {stratify_col} distribution:")
        val_dist = df_val[stratify_col].value_counts().sort_index()
        for val, count in val_dist.items():
            pct = count / len(df_val) * 100
            print(f"  {stratify_col}={val}: {count} samples ({pct:.2f}%)")
        
        # Use trainval metadata (same for both train and val since they come from same source)
        trainval_metadata = getattr(hparams, 'trainval_metadata_path', None)
        
        # Pass DataFrames directly to dataset (no need to create CSV files)
        train_csv_path = df_train  # Pass DataFrame directly
        val_csv_path = df_val  # Pass DataFrame directly
        train_metadata_path = trainval_metadata
        val_metadata_path = trainval_metadata
        print("="*60)
    else:
        # Fixed Split: Use separate train/val CSV files (existing behavior)
        train_csv_path = hparams.data_train_eval_tabular
        val_csv_path = hparams.data_val_eval_tabular
        train_metadata_path = getattr(hparams, 'train_metadata_path', None)
        val_metadata_path = getattr(hparams, 'val_metadata_path', None)
    
    train_dataset = ConstructionCostTIPDataset(
        csv_path=train_csv_path,
        composite_dir=hparams.composite_dir_trainval,
        field_lengths_tabular=hparams.field_lengths_tabular,
        labels_path=hparams.labels_train_eval_imaging,
        img_size=grab_arg_from_checkpoint(hparams, 'img_size'),
        is_train=True,
        corruption_rate=0.0,  # No corruption during fine-tuning
        replace_random_rate=0.0,
        replace_special_rate=0.0,
        augmentation_rate=hparams.eval_train_augment_rate,
        one_hot_tabular=hparams.eval_one_hot,
        use_sentinel2=hparams.use_sentinel2,
        use_viirs=hparams.use_viirs,
        live_loading=hparams.live_loading,
        augmentation_speedup=hparams.augmentation_speedup,
        metadata_path=train_metadata_path
    )
    val_dataset = ConstructionCostTIPDataset(
        csv_path=val_csv_path,
        composite_dir=hparams.composite_dir_trainval,
        field_lengths_tabular=hparams.field_lengths_tabular,
        labels_path=hparams.labels_val_eval_imaging,
        img_size=grab_arg_from_checkpoint(hparams, 'img_size'),
        is_train=False,
        corruption_rate=0.0,
        replace_random_rate=0.0,
        replace_special_rate=0.0,
        augmentation_rate=0.0,  # No augmentation for validation
        one_hot_tabular=hparams.eval_one_hot,
        use_sentinel2=hparams.use_sentinel2,
        use_viirs=hparams.use_viirs,
        live_loading=hparams.live_loading,
        augmentation_speedup=hparams.augmentation_speedup,
        metadata_path=val_metadata_path
    )
    with open_dict(hparams):
        hparams.input_size = train_dataset.get_input_size()
    return train_dataset, val_dataset


def finetune(hparams, wandb_logger):
    """
    Fine-tune TIP model for construction cost regression.
    
    This function:
    1. Loads pretraining checkpoint (has all architecture hyperparameters)
    2. Creates ConstructionCostFinetuning with regression head
    3. Fine-tunes only the regression head (backbone frozen)
    4. Saves checkpoint with all necessary hyperparameters
    
    Args:
        hparams: All hyperparameters (from fine-tuning config + pretraining checkpoint)
        wandb_logger: Instantiated weights and biases logger
    """
    pl.seed_everything(hparams.seed)
    
    # CRITICAL: Load pretraining checkpoint to get architecture hyperparameters
    if not hparams.checkpoint:
        raise ValueError("checkpoint must be specified for fine-tuning (path to pretraining checkpoint)")
    
    print("="*60)
    print("LOADING PRETRAINING CHECKPOINT FOR ARCHITECTURE")
    print("="*60)
    print(f"Pretraining checkpoint: {hparams.checkpoint}")
    
    pretrain_checkpoint = torch.load(hparams.checkpoint, map_location='cpu')
    pretrain_hparams = OmegaConf.create(pretrain_checkpoint['hyper_parameters'])
    
    # Merge pretraining hyperparameters (architecture) with fine-tuning hyperparameters (training config)
    # Strategy: Use fine-tuning config as base, only copy MISSING keys from pretrain checkpoint
    # Fine-tuning config values take precedence - only fill in missing architecture parameters
    with open_dict(hparams):
        # Convert pretrain to dict for iteration
        pretrain_dict = OmegaConf.to_container(pretrain_hparams, resolve=True)
        
        # Copy only missing keys from pretrain checkpoint (fine-tuning config takes precedence)
        for key, value in pretrain_dict.items():
            # Skip checkpoint path - we'll set it separately
            if key == 'checkpoint':
                continue
            # Only copy if key is missing in fine-tuning config
            if not hasattr(hparams, key) or key not in hparams:
                setattr(hparams, key, value)
        
        # Store original pretraining checkpoint path for reference (optional, for logging/debugging)
        # hparams.checkpoint already points to the pretraining checkpoint, so TIPBackbone can use it directly
        # No need to set hparams.checkpoint again - it's already correct
    
    print("✅ Merged architecture hyperparameters from pretraining checkpoint")
    print(f"   multimodal_embedding_dim: {hparams.multimodal_embedding_dim}")
    print(f"   embedding_dim: {hparams.embedding_dim}")
    print(f"   model: {hparams.model}")
    
    use_kfold = getattr(hparams, 'use_kfold', False)
    
    if use_kfold:
        # Automatic k-fold: run all folds sequentially
        k_fold = getattr(hparams, 'k_fold', 5)
        k_fold_current = getattr(hparams, 'k_fold_current', -1)
        
        # If k_fold_current is -1 or not specified, run all folds
        if k_fold_current == -1 or k_fold_current is None:
            print("="*60)
            print(f"AUTOMATIC K-FOLD: Running all {k_fold} folds sequentially")
            print("="*60)
            
            # Create base logdir once for all folds
            # create_logdir will strip tip_finetune_ prefix from wandb name
            base_logdir = create_logdir('finetune', False, wandb_logger, allow_existing=False)
            
            for fold_idx in range(k_fold):
                print("\n" + "="*60)
                print(f"FOLD {fold_idx + 1}/{k_fold}")
                print("="*60)
                
                # Update hparams with current fold for logging
                with open_dict(hparams):
                    hparams.k_fold_current = fold_idx
                
                # Run fine-tuning for this fold (pass base_logdir to avoid recreating it)
                _finetune_single_fold(hparams, wandb_logger, fold_idx, base_logdir=base_logdir)
            
            print("\n" + "="*60)
            print(f"✅ ALL {k_fold} FOLDS COMPLETE!")
            print("="*60)
        else:
            # Run single fold
            _finetune_single_fold(hparams, wandb_logger, k_fold_current)
    else:
        # Fixed split: run normally
        _finetune_single_fold(hparams, wandb_logger, None)


def _finetune_single_fold(hparams, wandb_logger, fold_index, base_logdir=None):
    """
    Fine-tune a single fold (or fixed split if fold_index is None).
    
    Args:
        hparams: All hyperparameters
        wandb_logger: Instantiated weights and biases logger
        fold_index: Current fold index (0 to k_fold-1) or None for fixed split
        base_logdir: Optional base logdir (if provided, will be reused instead of creating new one)
    """
    # Load datasets
    train_dataset, val_dataset = load_datasets(hparams, fold_index=fold_index)
    
    drop = ((len(train_dataset) % hparams.batch_size) == 1)
    
    sampler = None
    if getattr(hparams, 'weights', None):
        print('Using weighted random sampler')
        weights_list = [hparams.weights[int(l)] for l in train_dataset.labels]
        sampler = WeightedRandomSampler(weights=weights_list, num_samples=len(weights_list), replacement=True)
    
    num_gpus = cuda.device_count()
    train_loader = DataLoader(
        train_dataset,
        num_workers=hparams.num_workers, batch_size=hparams.batch_size, sampler=sampler,
        pin_memory=True, shuffle=(sampler is None), drop_last=drop, persistent_workers=True)
    
    print(f'Train shuffle is: {sampler is None}')
    
    val_loader = DataLoader(
        val_dataset,
        num_workers=hparams.num_workers, batch_size=hparams.batch_size,
        pin_memory=True, shuffle=False, persistent_workers=True)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f'Valid batch size: {hparams.batch_size * cuda.device_count()}')
    
    # Derive logdir run name using head type + timestamp from existing wandb name (do not change wandb name)
    # Create logdir (create_logdir will strip tip_finetune_ prefix from wandb name)
    # Add fold information to logdir if using k-fold
    if base_logdir is None:
        base_logdir = create_logdir('finetune', False, wandb_logger)
    else:
        # Base logdir already exists (from k-fold loop), just ensure it exists
        os.makedirs(base_logdir, exist_ok=True)
    
    if fold_index is not None:
        logdir = os.path.join(base_logdir, f'fold_{fold_index}')
        os.makedirs(logdir, exist_ok=True)
        print(f"Fold {fold_index} checkpoint directory: {logdir}")
    else:
        logdir = base_logdir
    
    # Create model: ConstructionCostFinetuning
    # This will create TIPBackbone with pretraining checkpoint (has all architecture params)
    # and add regression head
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = ConstructionCostFinetuning(hparams)
    
    # Determine mode based on metric (lower is better for rmsle, mae, rmse)
    if hparams.eval_metric in ['rmsle', 'mae', 'rmse']:
        mode = 'min'  # Lower is better
    else:
        mode = 'min'  # Default to min (lower is better)
    
    callbacks = []
    # Save best checkpoint based on validation metric (include metric value in filename)
    metric_key = f'finetune.val.{hparams.eval_metric}'
    # Format: checkpoint_best_{metric}_{epoch:02d}_{metric_value:.4f}
    # Use double braces {{ }} so PyTorch Lightning can format them
    checkpoint_callback = ModelCheckpoint(
        monitor=metric_key,
        mode=mode,
        filename=f'checkpoint_best_{hparams.eval_metric}_{{epoch:02d}}_{{{metric_key}:.4f}}',
        dirpath=logdir,
        save_top_k=1,  # Only keep the best checkpoint
        auto_insert_metric_name=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    # Also save last epoch checkpoint (similar to pretraining)
    callbacks.append(ModelCheckpoint(
        filename='checkpoint_last_epoch_{epoch:02d}',
        dirpath=logdir,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False
    ))
    # Early stopping (optional, can be disabled via config)
    use_early_stopping = getattr(hparams, 'use_early_stopping', True)
    if use_early_stopping:
        early_stopping_patience = getattr(hparams, 'early_stopping_patience', 10)
        early_stopping_min_delta = getattr(hparams, 'early_stopping_min_delta', 0.0002)
        # Patience is multiplied by (1 / val_check_interval) to account for validation frequency
        patience = int(early_stopping_patience * (1 / hparams.val_check_interval))
        callbacks.append(EarlyStopping(
            monitor=f'finetune.val.{hparams.eval_metric}',
            min_delta=early_stopping_min_delta,
            patience=patience,
            verbose=True,  # Set to True to see when early stopping triggers
            mode=mode
        ))
        print(f"✅ Early stopping enabled: patience={patience} epochs, min_delta={early_stopping_min_delta}")
    else:
        print("⚠️  Early stopping disabled - training will run for full max_epochs")
    if hparams.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
    # Gradient clipping for stability (especially important when unfreezing backbone)
    gradient_clip_val = getattr(hparams, 'gradient_clip_val', 1.0)  # Default: clip at 1.0
    
    trainer = Trainer.from_argparse_args(
        hparams,
        accelerator="gpu",
        devices=cuda.device_count(),
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        val_check_interval=hparams.val_check_interval,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        limit_test_batches=hparams.limit_test_batches,
        log_every_n_steps=getattr(hparams, 'log_every_n_steps', 1),
        gradient_clip_val=gradient_clip_val  # Clip gradients for stability
    )
    
    print("\n" + "="*60)
    print("STARTING FINE-TUNING")
    print("="*60)
    trainer.fit(model, train_loader, val_loader)
    
    # Save evaluation results (convert tensors to scalars)
    eval_results = {}
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            # Convert tensor to scalar
            if value.numel() == 1:
                eval_results[key] = value.item()
            else:
                # If tensor has multiple elements, convert to list
                eval_results[key] = value.cpu().numpy().tolist()
        else:
            eval_results[key] = value
    
    eval_df = pd.DataFrame(eval_results, index=[0])
    eval_df.to_csv(join(logdir, 'eval_results.csv'), index=False)
    
    # Get best validation score from ModelCheckpoint callback (it already knows the best score)
    metric_key = f'finetune.val.{hparams.eval_metric}'
    best_score = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.monitor == metric_key:
            best_score = callback.best_model_score
            if best_score is not None and isinstance(best_score, torch.Tensor):
                best_score = best_score.item()
            break
    
    # Check if best_score is valid (not inf or nan)
    if best_score is not None and best_score != float('inf') and best_score == best_score:  # best_score == best_score checks for NaN
        print(f"\n✅ Best validation {hparams.eval_metric}: {best_score:.6f}")
    elif best_score == float('inf'):
        print(f"\n⚠️  Best validation {hparams.eval_metric}: inf (no valid validation metrics recorded - check for NaN/Inf issues)")
    elif best_score is not None and best_score != best_score:  # NaN check
        print(f"\n⚠️  Best validation {hparams.eval_metric}: NaN (validation metrics contained NaN values)")
    else:
        print(f"\n⚠️  Could not determine best validation score")
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Checkpoint saved to: {logdir}")
    print(f"   Best checkpoint: {join(logdir, f'checkpoint_best_{hparams.eval_metric}.ckpt')}")
    
    # Professional cleanup: Use PyTorch's distributed cleanup mechanisms
    # This properly handles DDP process destruction and resource cleanup
    # IMPORTANT: Check rank before destroying DDP, so we know which process to run evaluation on
    is_main_process = True
    try:
        if dist.is_initialized():
            is_main_process = (dist.get_rank() == 0)
            # Properly destroy DDP process group to prevent semaphore leaks
            dist.destroy_process_group()
    except Exception as e:
        # Log but don't fail on cleanup errors (cleanup should be best-effort)
        print(f"⚠️  Warning during DDP cleanup: {e}")
    
    # Release GPU memory (PyTorch Lightning doesn't always do this automatically)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Run automatic evaluation if enabled
    # CRITICAL: Only run evaluation on main process (rank 0) to avoid duplicate evaluation
    # After DDP cleanup, all processes are still running, but we only want one to evaluate
    evaluator_config = getattr(hparams, 'evaluator', None)
    if evaluator_config and getattr(evaluator_config, 'enabled', False):
        if is_main_process:
            print("\n" + "="*60)
            print("RUNNING AUTOMATIC EVALUATION")
            print("="*60)
            
            success = _run_evaluation_for_fold(logdir, hparams, evaluator_config)
            if success:
                print("\n✅ Automatic evaluation complete!")
            else:
                print("\n⚠️  Automatic evaluation failed or skipped")
        else:
            # Other processes wait/skip evaluation (they would have been destroyed in DDP cleanup anyway)
            print(f"\n⏭️  Skipping evaluation on non-main process (rank != 0)")
    
    return model, logdir

