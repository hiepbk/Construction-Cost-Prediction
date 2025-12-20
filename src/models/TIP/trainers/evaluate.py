import os 
from os.path import join

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data.sampler import WeightedRandomSampler
from torch import cuda
import pandas as pd
import numpy as np

from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from datasets.ConstructionCostTIPDataset import ConstructionCostTIPDataset
from datasets.TabularDataset import TabularDataset
from models.Evaluator import Evaluator
from models.Evaluator_regression import Evaluator_Regression
from utils.utils import grab_arg_from_checkpoint, grab_hard_eval_image_augmentations, grab_wids, create_logdir
from omegaconf import open_dict
from sklearn.model_selection import KFold

def load_datasets(hparams):
  # Check if using ConstructionCostTIPDataset
  use_construction_cost_dataset = getattr(hparams, 'use_construction_cost_dataset', False)
  
  if use_construction_cost_dataset and hparams.eval_datatype == 'multimodal':
    # Use ConstructionCostTIPDataset for construction cost fine-tuning
    from datasets.ConstructionCostTIPDataset import ConstructionCostTIPDataset
    
    use_kfold = getattr(hparams, 'use_kfold', False)
    
    if use_kfold:
      # K-Fold Cross-Validation: Read from unified trainval.csv and split
      print("="*60)
      print("USING K-FOLD CROSS-VALIDATION")
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
      train_dataset = ConstructionCostTIPDataset(
        csv_path=df_train,  # Pass DataFrame directly
        composite_dir=hparams.composite_dir_trainval,
        field_lengths_tabular=hparams.field_lengths_tabular,
        labels_path=hparams.labels_train_eval_imaging,
        img_size=grab_arg_from_checkpoint(hparams, 'img_size'),
        is_train=True,
        corruption_rate=0.0,
        replace_random_rate=0.0,
        replace_special_rate=0.0,
        augmentation_rate=hparams.eval_train_augment_rate,
        one_hot_tabular=hparams.eval_one_hot,
        use_sentinel2=hparams.use_sentinel2,
        use_viirs=hparams.use_viirs,
        live_loading=hparams.live_loading,
        augmentation_speedup=hparams.augmentation_speedup,
        target_log_transform=getattr(hparams, 'target_log_transform', True),
        metadata_path=trainval_metadata  # Use trainval metadata for both
      )
      val_dataset = ConstructionCostTIPDataset(
        csv_path=df_val,  # Pass DataFrame directly
        composite_dir=hparams.composite_dir_trainval,
        field_lengths_tabular=hparams.field_lengths_tabular,
        labels_path=hparams.labels_val_eval_imaging,
        img_size=grab_arg_from_checkpoint(hparams, 'img_size'),
        is_train=False,
        corruption_rate=0.0,
        replace_random_rate=0.0,
        replace_special_rate=0.0,
        augmentation_rate=0.0,
        one_hot_tabular=hparams.eval_one_hot,
        use_sentinel2=hparams.use_sentinel2,
        use_viirs=hparams.use_viirs,
        live_loading=hparams.live_loading,
        augmentation_speedup=hparams.augmentation_speedup,
        target_log_transform=getattr(hparams, 'target_log_transform', True),
        metadata_path=trainval_metadata  # Use trainval metadata for both
      )
      print("="*60)
    else:
      # Fixed Split: Use separate train/val CSV files (existing behavior)
      train_dataset = ConstructionCostTIPDataset(
        csv_path=hparams.data_train_eval_tabular,
        composite_dir=hparams.composite_dir_trainval,
        field_lengths_tabular=hparams.field_lengths_tabular,
        labels_path=hparams.labels_train_eval_imaging,  # Contains targets
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
        target_log_transform=getattr(hparams, 'target_log_transform', True),
        metadata_path=getattr(hparams, 'train_metadata_path', None)
      )
      val_dataset = ConstructionCostTIPDataset(
        csv_path=hparams.data_val_eval_tabular,
        composite_dir=hparams.composite_dir_trainval,  # Val split uses trainval_composite (same as train)
        field_lengths_tabular=hparams.field_lengths_tabular,
        labels_path=hparams.labels_val_eval_imaging,  # Contains targets
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
        target_log_transform=getattr(hparams, 'target_log_transform', True),
        metadata_path=getattr(hparams, 'val_metadata_path', None)
      )
    
    with open_dict(hparams):
      hparams.input_size = train_dataset.get_input_size()
  elif hparams.eval_datatype=='imaging':
      train_dataset = ImageDataset(hparams.data_train_eval_imaging, hparams.labels_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, live_loading=hparams.live_loading, task=hparams.task,
                                   dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
      val_dataset = ImageDataset(hparams.data_val_eval_imaging, hparams.labels_val_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task,
                                 dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
  elif hparams.eval_datatype=='multimodal':
    assert hparams.strategy == 'tip'
    train_dataset = ImagingAndTabularDataset(
      hparams.data_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, 
      hparams.data_train_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_train_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=True, target=hparams.target, corruption_rate=hparams.corruption_rate,
      data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate, 
      augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name
    )
    val_dataset = ImagingAndTabularDataset(
      hparams.data_val_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, 
      hparams.data_val_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_val_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=hparams.corruption_rate,
      data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
      augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name
    )
    with open_dict(hparams):
      hparams.input_size = train_dataset.get_input_size()
  elif hparams.eval_datatype == 'tabular':
    train_dataset = TabularDataset(hparams.data_train_eval_tabular, hparams.labels_train_eval_tabular, hparams.eval_train_augment_rate, hparams.corruption_rate, train=True, 
                                  eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
                                  data_base=hparams.data_base, strategy=hparams.strategy, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,target=hparams.target)
    val_dataset = TabularDataset(hparams.data_val_eval_tabular, hparams.labels_val_eval_tabular, hparams.eval_train_augment_rate, hparams.corruption_rate, train=False, 
                                eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
                                data_base=hparams.data_base, strategy=hparams.strategy, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,target=hparams.target)
    with open_dict(hparams):
      hparams.input_size = train_dataset.get_input_size()
  elif hparams.eval_datatype == 'imaging_and_tabular':
    train_dataset = ImagingAndTabularDataset(
      hparams.data_train_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, 
      hparams.data_train_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_train_eval_imaging, hparams.img_size, hparams.live_loading, train=True, target=hparams.target,
      corruption_rate=hparams.corruption_rate, data_base=hparams.data_base, augmentation_speedup=hparams.augmentation_speedup,
       missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,algorithm_name=hparams.algorithm_name
    )
    val_dataset = ImagingAndTabularDataset(
      hparams.data_val_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, 
      hparams.data_val_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_val_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target,
      corruption_rate=hparams.corruption_rate, data_base=hparams.data_base, augmentation_speedup=hparams.augmentation_speedup,
       missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,algorithm_name=hparams.algorithm_name
    )
    with open_dict(hparams):
      hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception('argument dataset must be set to imaging, tabular, multimodal or imaging_and_tabular')
  return train_dataset, val_dataset


def evaluate(hparams, wandb_logger):
  """
  Evaluates trained contrastive models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)

  if hparams.missing_tabular == True and hparams.missing_strategy != 'value':
      check_list = []
      for data_path in [hparams.data_train_eval_tabular, hparams.data_val_eval_tabular, hparams.data_test_eval_tabular]:
        tabular_name = data_path.split('/')[-1].split('.')[0]
        missing_mask_path = join(hparams.data_base, 'missing_mask', f'{tabular_name}_{hparams.target}_{hparams.missing_strategy}_{hparams.missing_rate}.npy')
        missing_mask = np.load(missing_mask_path)
        check_list.append(missing_mask[0])
      assert np.all(check_list[0] == check_list[1]) and np.all(check_list[0] == check_list[2]), 'Missing mask is not the same for train, val and test'
      print(f'Num of missing: {np.sum(check_list[0])}, Current missing mask: {np.where(check_list[0])}')
  
  train_dataset, val_dataset = load_datasets(hparams)
  
  drop = ((len(train_dataset)%hparams.batch_size)==1)

  sampler = None
  if getattr(hparams, 'weights', None):
    print('Using weighted random sampler(')
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
  print(f'Valid batch size: {hparams.batch_size*cuda.device_count()}')

  logdir = create_logdir('eval', hparams.resume_training, wandb_logger)

  # Use ConstructionCostFinetuning if use_construction_cost_dataset is True
  use_construction_cost_dataset = getattr(hparams, 'use_construction_cost_dataset', False)
  if use_construction_cost_dataset and hparams.task == 'regression':
    from models.ConstructionCostFinetuning import ConstructionCostFinetuning
    model = ConstructionCostFinetuning(hparams)
  elif hparams.task == 'regression':
    model = Evaluator_Regression(hparams)
  else:
    model = Evaluator(hparams)
  
  # Determine mode based on metric (lower is better for rmsle, mae, rmse)
  if hparams.eval_metric in ['rmsle', 'mae', 'rmse']:
    mode = 'min'  # Lower is better
  else:
    mode = 'min'  # Default to min (lower is better)
  
  callbacks = []
  callbacks.append(ModelCheckpoint(monitor=f'eval.val.{hparams.eval_metric}', mode=mode, filename=f'checkpoint_best_{hparams.eval_metric}', dirpath=logdir))
  callbacks.append(EarlyStopping(monitor=f'eval.val.{hparams.eval_metric}', min_delta=0.0002, patience=int(10*(1/hparams.val_check_interval)), verbose=False, mode=mode))
  if hparams.use_wandb:
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))


  trainer = Trainer.from_argparse_args(hparams, accelerator="gpu", devices=cuda.device_count(), callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, val_check_interval=hparams.val_check_interval, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, limit_test_batches=hparams.limit_test_batches, log_every_n_steps=getattr(hparams, 'log_every_n_steps', 1))
  trainer.fit(model, train_loader, val_loader)
  eval_df = pd.DataFrame(trainer.callback_metrics, index=[0])
  eval_df.to_csv(join(logdir, 'eval_results.csv'), index=False)
  
  # Log best validation score (handle case where it might not be set)
  best_score = getattr(model, 'best_val_score', None)
  if best_score is None:
    # Fallback: try to get the metric-specific best value
    if hparams.eval_metric == 'mae':
      best_score = getattr(model, 'best_val_mae', float('inf'))
    elif hparams.eval_metric == 'rmse':
      best_score = getattr(model, 'best_val_rmse', float('inf'))
    elif hparams.eval_metric == 'rmsle':
      best_score = getattr(model, 'best_val_rmsle', float('inf'))
    else:
      best_score = getattr(model, 'best_val_mae', float('inf'))
  
  wandb_logger.log_metrics({f'best.val.{hparams.eval_metric}': best_score})


  if hparams.test_and_eval:
    
    if hparams.eval_datatype == 'imaging':
        test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task,
                                    dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
        hparams.transform_test = test_dataset.transform_val.__repr__()
    elif hparams.eval_datatype == 'multimodal':
      assert hparams.strategy == 'tip'
      test_dataset = ImagingAndTabularDataset(
      hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
      hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_test_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=0,
      data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
      augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name
    )
      with open_dict(hparams):
        hparams.input_size = test_dataset.get_input_size()
    elif hparams.eval_datatype == 'tabular':
      test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, 0, 0, train=False, 
                                  eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
                                  data_base=hparams.data_base, 
                                  strategy=hparams.strategy, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,target=hparams.target)
      with open_dict(hparams):
        hparams.input_size = test_dataset.get_input_size()
    elif hparams.eval_datatype == 'imaging_and_tabular':
      test_dataset = ImagingAndTabularDataset(
        hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
        hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
        hparams.labels_test_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target,
        corruption_rate=0.0, data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
        augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name)
      with open_dict(hparams):
        hparams.input_size = test_dataset.get_input_size()
    else:
      raise Exception('argument dataset must be set to imaging, tabular or multimodal')
    
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=512,  
      pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)
  
    print(f"Number of testing batches: {len(test_loader)}")

    model.freeze()

    trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
    test_results = trainer.test(model, test_loader, ckpt_path=os.path.join(logdir,f'checkpoint_best_{hparams.eval_metric}.ckpt'))
    df = pd.DataFrame(test_results)
    df.to_csv(join(logdir, 'test_results.csv'), index=False)