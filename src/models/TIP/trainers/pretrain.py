import os 
import sys

from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import open_dict

from utils.utils import grab_image_augmentations, grab_wids, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator
from utils.ssl_online_regression import SSLOnlineEvaluatorRegression

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from datasets.ContrastiveReconstructImagingAndTabularDataset import ContrastiveReconstructImagingAndTabularDataset
from datasets.ConstructionCostTIPDataset import ConstructionCostTIPDataset
from datasets.ContrastiveImageDataset import ContrastiveImageDataset
from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset
from datasets.MaskTabularDataset import MaskTabularDataset

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
  if hparams.datatype == 'multimodal':
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.augmentation_speedup)
    with open_dict(hparams):
      hparams.transform = transform.__repr__()
    if hparams.strategy == 'tip':
      # for TIP
      if hasattr(hparams, 'use_construction_cost_dataset') and hparams.use_construction_cost_dataset:
        # Use our custom Construction Cost dataset
        train_dataset = ConstructionCostTIPDataset(
          csv_path=hparams.data_train_tabular,
          composite_dir=hparams.composite_dir,
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
          metadata_path=getattr(hparams, 'train_metadata_path', None)
        )
        val_dataset = ConstructionCostTIPDataset(
          csv_path=hparams.data_val_tabular,
          composite_dir=hparams.composite_dir,
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
          metadata_path=getattr(hparams, 'val_metadata_path', None)
        )
      else:
        # Original TIP dataset
        train_dataset = ContrastiveReconstructImagingAndTabularDataset(
          data_path_imaging=hparams.data_train_imaging, delete_segmentation=hparams.delete_segmentation, augmentation=transform, augmentation_rate=hparams.augmentation_rate, 
          data_path_tabular=hparams.data_train_tabular, corruption_rate=hparams.corruption_rate, replace_random_rate=hparams.replace_random_rate, replace_special_rate=hparams.replace_special_rate, 
          field_lengths_tabular=hparams.field_lengths_tabular, one_hot_tabular=hparams.one_hot,
          labels_path=hparams.labels_train, img_size=hparams.img_size, live_loading=hparams.live_loading, augmentation_speedup=hparams.augmentation_speedup)
        val_dataset = ContrastiveReconstructImagingAndTabularDataset(
          data_path_imaging=hparams.data_val_imaging, delete_segmentation=hparams.delete_segmentation, augmentation=transform, augmentation_rate=hparams.augmentation_rate, 
          data_path_tabular=hparams.data_val_tabular, corruption_rate=hparams.corruption_rate, replace_random_rate=hparams.replace_random_rate, replace_special_rate=hparams.replace_special_rate, 
          field_lengths_tabular=hparams.field_lengths_tabular, one_hot_tabular=hparams.one_hot,
          labels_path=hparams.labels_val, img_size=hparams.img_size, live_loading=hparams.live_loading, augmentation_speedup=hparams.augmentation_speedup)
    else:
      # for MMCL
      train_dataset = ContrastiveImagingAndTabularDataset(
        data_path_imaging=hparams.data_train_imaging, delete_segmentation=hparams.delete_segmentation, augmentation=transform, augmentation_rate=hparams.augmentation_rate, 
        data_path_tabular=hparams.data_train_tabular, corruption_rate=hparams.corruption_rate, field_lengths_tabular=hparams.field_lengths_tabular, one_hot_tabular=hparams.one_hot,
        labels_path=hparams.labels_train, img_size=hparams.img_size, live_loading=hparams.live_loading, augmentation_speedup=hparams.augmentation_speedup)
      val_dataset = ContrastiveImagingAndTabularDataset(
        data_path_imaging=hparams.data_val_imaging, delete_segmentation=hparams.delete_segmentation, augmentation=transform, augmentation_rate=hparams.augmentation_rate, 
        data_path_tabular=hparams.data_val_tabular, corruption_rate=hparams.corruption_rate, field_lengths_tabular=hparams.field_lengths_tabular, one_hot_tabular=hparams.one_hot,
        labels_path=hparams.labels_val, img_size=hparams.img_size, live_loading=hparams.live_loading, augmentation_speedup=hparams.augmentation_speedup)
    with open_dict(hparams):
      hparams.input_size = train_dataset.get_input_size()
  elif hparams.datatype == 'tabular':
    # for SSL tabular models
    if hparams.algorithm_name == 'SCARF':
      train_dataset = ContrastiveTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
      val_dataset = ContrastiveTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    elif hparams.algorithm_name == 'VIME':
      train_dataset = MaskTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
      val_dataset = MaskTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    with open_dict(hparams):
      hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
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
    z_dim =  hparams.multimodal_embedding_dim if hparams.strategy=='tip' else model.pooled_dim
    
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