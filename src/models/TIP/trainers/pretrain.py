from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import open_dict, DictConfig, OmegaConf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from utils.utils import create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator
from utils.ssl_online_regression import SSLOnlineEvaluatorRegression

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
    
    # Create k-fold splitter
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=k_fold_seed)
    
    # Get indices for current fold
    indices = np.arange(len(df_trainval))
    train_indices, val_indices = list(kf.split(indices))[fold_index]
    
    # Split dataframe
    df_train = df_trainval.iloc[train_indices].reset_index(drop=True)
    df_val = df_trainval.iloc[val_indices].reset_index(drop=True)
    
    print(f"Fold {fold_index}: Train={len(df_train)} samples, Val={len(df_val)} samples")
    
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
    composite_dir=hparams.composite_dir_trainval,
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
    target_log_transform=getattr(hparams, 'target_log_transform', True),
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
    target_log_transform=getattr(hparams, 'target_log_transform', True),
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

  # Create logdir based on WandB run name
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
  if hparams.num_classes == 1:
    from omegaconf import open_dict
    with open_dict(hparams):
      # Ensure these are set (they should already be from config, but make sure)
      if not hasattr(hparams, 'target_mean'):
        hparams.target_mean = 0.0
      if not hasattr(hparams, 'target_std'):
        hparams.target_std = 1.0
      if not hasattr(hparams, 'target_log_transform'):
        hparams.target_log_transform = True
  
  model = select_model(hparams, train_dataset)
  
  callbacks = []

  if hparams.online_mlp:
    model.hparams.classifier_freq = float('Inf')
    z_dim = hparams.multimodal_embedding_dim if hparams.strategy=='tip' else model.pooled_dim
    
    # Use regression evaluator for regression tasks (num_classes=1)
    if hparams.num_classes == 1:
      # Get regression_head config directly from hparams (no fallback)
      if not (hasattr(hparams, 'regression_head') or 'regression_head' in hparams):
        raise ValueError("hparams must contain 'regression_head' dict with all head configuration")
      
      regression_head = getattr(hparams, 'regression_head', hparams.get('regression_head', {}))
      if isinstance(regression_head, DictConfig):
        regression_head = OmegaConf.to_container(regression_head, resolve=True)
      
      if not isinstance(regression_head, dict) or 'type' not in regression_head:
        raise ValueError("regression_head must be a dict containing 'type' and all head parameters")
      
      # Regression online evaluation
      callbacks.append(SSLOnlineEvaluatorRegression(
        z_dim=z_dim,
        regression_head=regression_head,
        multimodal=(hparams.datatype=='multimodal'),
        strategy=hparams.strategy,
      ))
    else:
      # Classification online evaluation
      callbacks.append(SSLOnlineEvaluator(
        z_dim=z_dim,
        hidden_dim=hparams.embedding_dim,
        num_classes=hparams.num_classes,
        swav=False,
        multimodal=(hparams.datatype=='multimodal'),
        strategy=hparams.strategy
      ))
  # Save best checkpoint based on validation metric (lower is better)
  # Only save best checkpoint if we have regression online evaluation enabled
  if hparams.online_mlp and hparams.num_classes == 1:
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
    # Also save last epoch checkpoint
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

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)
  
  # Cleanup: Release GPU memory and properly destroy DDP processes
  # This prevents VRAM from staying occupied and semaphore leaks
  import torch
  if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
  
  # Properly destroy trainer to clean up DDP processes
  # This helps prevent semaphore leaks
  del trainer
  del model
  del train_loader
  del val_loader
  
  # Force garbage collection
  import gc
  gc.collect()
  
  if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Final cleanup