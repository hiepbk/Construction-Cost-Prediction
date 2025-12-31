import torch
import torch.distributed as dist
from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import open_dict, DictConfig, OmegaConf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils.utils import create_logdir
from utils.ssl_online_construction_cost import SSLOnlineEvaluatorRegression, SSLOnlineEvaluatorClassification

from datasets.ConstructionCostTIPDataset import ConstructionCostTIPDataset
from models.MultimodalSimCLR import MultimodalSimCLR
from models.SimCLR import SimCLR
from models.SwAV_Bolt import SwAV
from models.BYOL_Bolt import BYOL
from models.SimSiam_Bolt import SimSiam
from models.BarlowTwins import BarlowTwins
from models.SCARF import SCARF
from models.VIME import VIME
from models.Tips.TipModel3Loss import TIP3Loss


def load_datasets(hparams, fold_index=None):
  """Load ConstructionCostTIPDataset for train and validation.
  
  Args:
    hparams: Hyperparameters
    fold_index: Current fold index (0 to k_fold-1). If None and use_kfold=True, 
                will use k_fold_current from hparams. If -1, will raise error.
  """
  
  use_kfold = getattr(hparams, 'use_kfold', False)
  
  if use_kfold:
    # K-Fold Cross-Validation: Read from unified trainval.csv and split
    print("="*60)
    print("USING K-FOLD CROSS-VALIDATION (PRETRAINING)")
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
    train_csv_path = hparams.data_train_tabular
    val_csv_path = hparams.data_val_tabular
    train_metadata_path = getattr(hparams, 'train_metadata_path', None)
    val_metadata_path = getattr(hparams, 'val_metadata_path', None)
  
  train_dataset = ConstructionCostTIPDataset(
    csv_path=train_csv_path,
    composite_dir=hparams.composite_dir_traintest,
    field_lengths_tabular=hparams.field_lengths_tabular,
    labels_path=hparams.labels_train,
    img_size=hparams.img_size,
    is_train=True,
    corruption_rate=hparams.corruption_rate,
    replace_random_rate=hparams.replace_random_rate,
    replace_special_rate=hparams.replace_special_rate,
    augmentation_rate=hparams.augmentation_rate,
    one_hot_tabular=hparams.one_hot,
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
    labels_path=hparams.labels_val,
    img_size=hparams.img_size,
    is_train=False,
    corruption_rate=hparams.corruption_rate,
    replace_random_rate=hparams.replace_random_rate,
    replace_special_rate=hparams.replace_special_rate,
    augmentation_rate=hparams.augmentation_rate,
    one_hot_tabular=hparams.one_hot,
    use_sentinel2=hparams.use_sentinel2,
    use_viirs=hparams.use_viirs,
    live_loading=hparams.live_loading,
    augmentation_speedup=hparams.augmentation_speedup,
    metadata_path=val_metadata_path
  )
  with open_dict(hparams):
    hparams.input_size = train_dataset.get_input_size()
  return train_dataset, val_dataset


def select_model(hparams, train_dataset):
  if hparams.datatype == 'multimodal':
    if hparams.strategy == 'tip':
      # TIP
      model = TIP3Loss(hparams)
      print('Using TIP3Loss')
    else:
      # MMCL
      model = MultimodalSimCLR(hparams)
  elif hparams.datatype == 'imaging':
    if hparams.loss.lower() == 'byol':
      model = BYOL(**hparams)
    elif hparams.loss.lower() == 'simsiam':
      model = SimSiam(**hparams)
    elif hparams.loss.lower() == 'swav':
      if not hparams.resume_training:
        model = SwAV(gpus=1, nmb_crops=(2,0), num_samples=len(train_dataset),  **hparams)
      else:
        model = SwAV(**hparams)
    elif hparams.loss.lower() == 'barlowtwins':
      model = BarlowTwins(**hparams)
    else:
      model = SimCLR(hparams)
    print('Imaging model: ', hparams.loss.lower())
  elif hparams.datatype == 'tabular':
    # model = TransformerSCARF(hparams)
    if hparams.algorithm_name == 'SCARF':
      model = SCARF(hparams)
    elif hparams.algorithm_name == 'VIME':
      model = VIME(hparams)
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return model


def pretrain(hparams, wandb_logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  
  If use_kfold=True and k_fold_current is not specified (or -1), 
  automatically runs all k-folds sequentially.
  """
  pl.seed_everything(hparams.seed)

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
      
      # Create base logdir once for all folds (allow existing for subsequent folds)
      # create_logdir will strip tip_pretrain_ prefix from wandb name
      base_logdir = create_logdir('pretrain', hparams.resume_training, wandb_logger, allow_existing=False)
      
      for fold_idx in range(k_fold):
        print("\n" + "="*60)
        print(f"FOLD {fold_idx + 1}/{k_fold}")
        print("="*60)
        
        # Update hparams with current fold for logging
        from omegaconf import open_dict
        with open_dict(hparams):
          hparams.k_fold_current = fold_idx
        
        # Run training for this fold (pass base_logdir to avoid recreating it)
        _pretrain_single_fold(hparams, wandb_logger, fold_idx, base_logdir=base_logdir)
      
      print("\n" + "="*60)
      print(f"✅ ALL {k_fold} FOLDS COMPLETE!")
      print("="*60)
    else:
      # Run single fold
      _pretrain_single_fold(hparams, wandb_logger, k_fold_current)
  else:
    # Fixed split: run normally
    _pretrain_single_fold(hparams, wandb_logger, None)


def _pretrain_single_fold(hparams, wandb_logger, fold_index, base_logdir=None):
  """
  Train a single fold (or fixed split if fold_index is None).
  
  Args:
    hparams: All hyperparameters
    wandb_logger: Instantiated weights and biases logger
    fold_index: Current fold index (0 to k_fold-1) or None for fixed split
    base_logdir: Optional base logdir (if provided, will be reused instead of creating new one)
  """
  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams, fold_index=fold_index)

  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)


  print(f"Number of training batches: {len(train_loader)}")
  print(f"Number of validation batches: {len(val_loader)}")
  print(f'Valid batch size: {hparams.batch_size*cuda.device_count()}')

  # Derive logdir run name using head type + timestamp from existing wandb name (do not change wandb name)
  # Create logdir (create_logdir will strip tip_pretrain_ prefix from wandb name)
  # Add fold information to logdir if using k-fold
  if base_logdir is None:
    base_logdir = create_logdir('pretrain', hparams.resume_training, wandb_logger)
  else:
    # Base logdir already exists (from k-fold loop), just ensure it exists
    os.makedirs(base_logdir, exist_ok=True)
  
  if fold_index is not None:
    logdir = os.path.join(base_logdir, f'fold_{fold_index}')
    os.makedirs(logdir, exist_ok=True)
    print(f"Fold {fold_index} checkpoint directory: {logdir}")
  else:
    logdir = base_logdir
  
  # Ensure target normalization stats are in hparams (will be saved in checkpoint via save_hyperparameters)
  # This ensures they are saved in the checkpoint for later use
  # Only relevant for regression tasks (when construction_cost_head type ends with 'Regression')
  has_construction_cost_head = hasattr(hparams, 'construction_cost_head') or 'construction_cost_head' in hparams
  if has_construction_cost_head:
    # Check if head type is regression
    construction_cost_head = getattr(hparams, 'construction_cost_head', hparams.get('construction_cost_head', {}))
    if isinstance(construction_cost_head, DictConfig):
      construction_cost_head = OmegaConf.to_container(construction_cost_head, resolve=True)
    head_type = construction_cost_head.get('type', '').strip() if isinstance(construction_cost_head, dict) else ''
    is_regression = head_type.endswith('Regression')
    
    if is_regression:
      from omegaconf import open_dict
      with open_dict(hparams):
        # Ensure these are set (they should already be from config, but make sure)
        if not hasattr(hparams, 'target_mean'):
          hparams.target_mean = 0.0
        if not hasattr(hparams, 'target_std'):
          hparams.target_std = 1.0
  
  model = select_model(hparams, train_dataset)
  
  callbacks = []

  if hparams.online_mlp:
    model.hparams.classifier_freq = float('Inf')
    z_dim = hparams.multimodal_embedding_dim if hparams.strategy=='tip' else model.pooled_dim
    
    # Get construction_cost_head config (required for online evaluation)
    if not (hasattr(hparams, 'construction_cost_head') or 'construction_cost_head' in hparams):
      raise ValueError("Must define 'construction_cost_head' in config for online evaluation.")
    
    construction_cost_head = getattr(hparams, 'construction_cost_head', hparams.get('construction_cost_head', {}))
    if isinstance(construction_cost_head, DictConfig):
      construction_cost_head = OmegaConf.to_container(construction_cost_head, resolve=True)
    
    if not isinstance(construction_cost_head, dict) or 'type' not in construction_cost_head:
      raise ValueError("construction_cost_head must be a dict containing 'type' and all head parameters")
    
    # Determine head type from the 'type' field
    head_type = construction_cost_head.get('type', '').strip()
    
    # Check if it's a regression or classification head based on type name
    is_regression = head_type.endswith('Regression')
    is_classification = head_type.endswith('Classification')
    
    if not is_regression and not is_classification:
      raise ValueError(
        f"Unknown head type '{head_type}'. "
        f"Head type must end with 'Regression' or 'Classification'. "
        f"Available types: see HEAD_REGISTRY in ConstructionCostHead.py"
      )
    
    # Use appropriate callback based on head type
    if is_regression:
      # Regression online evaluation
      callbacks.append(SSLOnlineEvaluatorRegression(
        z_dim=z_dim,
        regression_head=construction_cost_head,
        multimodal=(hparams.datatype=='multimodal'),
        strategy=hparams.strategy,
      ))
    elif is_classification:
      # Classification online evaluation
      callbacks.append(SSLOnlineEvaluatorClassification(
        z_dim=z_dim,
        classification_head=construction_cost_head,
        multimodal=(hparams.datatype=='multimodal'),
        strategy=hparams.strategy,
      ))
  # Save best checkpoint based on validation metric (lower is better)
  # Save checkpoints for both regression and classification heads
  has_construction_cost_head = hasattr(hparams, 'construction_cost_head') or 'construction_cost_head' in hparams
  if hparams.online_mlp and has_construction_cost_head:
    # Check if head type is regression or classification
    construction_cost_head = getattr(hparams, 'construction_cost_head', hparams.get('construction_cost_head', {}))
    if isinstance(construction_cost_head, DictConfig):
      construction_cost_head = OmegaConf.to_container(construction_cost_head, resolve=True)
    head_type = construction_cost_head.get('type', '').strip() if isinstance(construction_cost_head, dict) else ''
    is_regression = head_type.endswith('Regression')
    is_classification = head_type.endswith('Classification')
    
    if is_regression:
      # Get eval_metric from config (default to rmsle)
      eval_metric = getattr(hparams, 'eval_metric', 'rmsle')
      # Determine mode based on metric (lower is better for rmsle, mae, rmse)
      if eval_metric in ['rmsle', 'mae', 'rmse']:
        mode = 'min'  # Lower is better
      else:
        mode = 'min'  # Default to min (lower is better)
      
      # Monitor validation metric from online regression evaluator
      metric_key = f'regression_online.val.{eval_metric}'
      callbacks.append(ModelCheckpoint(
        monitor=metric_key,
        mode=mode,
        filename=f'checkpoint_best_{eval_metric}_{{epoch:02d}}_{{{metric_key}:.4f}}',
        dirpath=logdir,
        save_top_k=1,  # Only keep the best checkpoint
        auto_insert_metric_name=False,
        verbose=True
      ))
    elif is_classification:
      # Monitor validation classification cross-entropy loss (lower is better)
      metric_key = 'classification_online.val.classification_ce'
      callbacks.append(ModelCheckpoint(
        monitor=metric_key,
        mode='min',  # Lower loss is better
        filename=f'checkpoint_best_classification_{{epoch:02d}}_{{{metric_key}:.6e}}',
        dirpath=logdir,
        save_top_k=1,  # Only keep the best checkpoint
        auto_insert_metric_name=False,
        verbose=True
      ))
    
    # Also save last epoch checkpoint (for both regression and classification)
    callbacks.append(ModelCheckpoint(
      filename='checkpoint_last_epoch_{epoch:02d}',
      dirpath=logdir,
      save_on_train_epoch_end=True,
      auto_insert_metric_name=False
    ))
  else:
    # For non-regression or when online evaluation is disabled, just save last epoch
    callbacks.append(ModelCheckpoint(
      filename='checkpoint_last_epoch_{epoch:02d}',
      dirpath=logdir,
      save_on_train_epoch_end=True,
      auto_insert_metric_name=False
    ))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=cuda.device_count(), 
                                       callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, 
                                       limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, enable_progress_bar=hparams.enable_progress_bar,
                                       log_every_n_steps=getattr(hparams, 'log_every_n_steps', 1))

  try:
    if hparams.resume_training:
      trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
    else:
      trainer.fit(model, train_loader, val_loader)
  finally:
    # Professional cleanup: Use PyTorch Lightning's built-in cleanup mechanisms
    # This properly handles DDP process destruction and resource cleanup
    # Properly destroy DDP process group to prevent semaphore leaks
    if dist.is_initialized():
      dist.destroy_process_group()
    
    # Release GPU memory (PyTorch Lightning doesn't always do this automatically)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.synchronize()