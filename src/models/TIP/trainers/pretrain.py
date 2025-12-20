from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import open_dict
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


def load_datasets(hparams):
  """Load ConstructionCostTIPDataset for train and validation."""
  
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
    k_fold_current = getattr(hparams, 'k_fold_current', 0)
    
    if k_fold_current < 0 or k_fold_current >= k_fold:
      raise ValueError(f"k_fold_current must be between 0 and {k_fold-1}, got {k_fold_current}")
    
    print(f"K-Fold parameters: k={k_fold}, seed={k_fold_seed}, current_fold={k_fold_current}")
    
    # Create k-fold splitter
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=k_fold_seed)
    
    # Get indices for current fold
    indices = np.arange(len(df_trainval))
    train_indices, val_indices = list(kf.split(indices))[k_fold_current]
    
    # Split dataframe
    df_train = df_trainval.iloc[train_indices].reset_index(drop=True)
    df_val = df_trainval.iloc[val_indices].reset_index(drop=True)
    
    print(f"Fold {k_fold_current}: Train={len(df_train)} samples, Val={len(df_val)} samples")
    
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
  """
  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)

  
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
  logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
  
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
      # Use target normalization stats from config only (user must set these after preprocessing)
      target_mean = getattr(hparams, 'target_mean', 0.0)
      target_std = getattr(hparams, 'target_std', 1.0)
      if target_mean == 0.0 and target_std == 1.0:
        print(f"⚠️  WARNING: target_mean and target_std are still defaults (0.0, 1.0)")
        print(f"   Please set them in config file after running preprocessing!")
      else:
        print(f"Using target stats from config: mean={target_mean:.4f}, std={target_std:.4f}")
      
      # Regression online evaluation
      callbacks.append(SSLOnlineEvaluatorRegression(
        z_dim=z_dim,
        hidden_dim=hparams.embedding_dim,
        regression_loss=getattr(hparams, 'regression_loss', 'huber'),
        huber_delta=getattr(hparams, 'huber_delta', 1.0),
        target_mean=target_mean,
        target_std=target_std,
        log_transform_target=getattr(hparams, 'target_log_transform', True),
        multimodal=(hparams.datatype=='multimodal'),
        strategy=hparams.strategy
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
  # Save best checkpoint based on validation RMSLE (lower is better)
  # Only save best checkpoint if we have regression online evaluation enabled
  if hparams.online_mlp and hparams.num_classes == 1:
    # Monitor validation RMSLE from online regression evaluator
    callbacks.append(ModelCheckpoint(
      monitor='regression_online.val.rmsle',
      mode='min',  # Lower RMSLE is better
      filename='checkpoint_best_rmsle_{epoch:02d}_{regression_online.val.rmsle:.4f}',
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