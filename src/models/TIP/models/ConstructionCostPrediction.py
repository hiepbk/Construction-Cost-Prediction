"""
Construction Cost Prediction Model

This is the final model architecture for the Construction Cost Prediction task.
It wraps TIPBackbone and handles:
1. Loading backbone weights from checkpoint
2. Detecting head class type from checkpoint
3. Replacing default classifier with correct head class
4. Loading head weights

Used by:
- Fine-tuning process (finetune.py)
- Evaluation script (evaluate_construction_cost.py)
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict, DictConfig
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.ConstructionCostHead import create_head


class ConstructionCostPrediction(nn.Module):
    """
    Final model for Construction Cost Prediction task.
    
    This model:
    1. Initializes TIPBackbone (with default Linear classifier)
    2. Loads backbone weights from checkpoint
    3. Detects head class type from checkpoint hyperparameters
    4. Replaces default classifier with correct head class from ConstructionCostHead
    5. Loads head weights from checkpoint
    
    Args:
        hparams: Hyperparameters (from checkpoint or config)
                   - hparams.checkpoint: Path to checkpoint file (for loading weights)
                   - hparams.field_lengths_tabular: Path to field_lengths.pt file
    """
    def __init__(self, hparams):
        super().__init__()
        
        # Extract paths from hparams
        checkpoint_path = getattr(hparams, 'checkpoint', None)
        field_lengths_path = getattr(hparams, 'field_lengths_tabular', None)
        
        # Store checkpoint path
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint if provided
        checkpoint = None
        preprocessed_checkpoint = None
        is_finetune_checkpoint = False
        
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_hparams = OmegaConf.create(checkpoint['hyper_parameters'])
            
            # Merge architecture params from checkpoint into hparams (if missing)
            # This is needed when loading checkpoint directly (e.g., in evaluate_construction_cost.py)
            # In finetune.py, this merge is already done, so this is a no-op in that case
            with open_dict(hparams):
                checkpoint_dict = OmegaConf.to_container(checkpoint_hparams, resolve=True)
                finetune_dict = OmegaConf.to_container(hparams, resolve=True)
                
                # Copy only missing keys from checkpoint (hparams takes precedence)
                for key, value in checkpoint_dict.items():
                    if key == 'checkpoint':
                        continue  # Skip checkpoint path
                    if key not in finetune_dict:
                        setattr(hparams, key, value)
            
            # Detect checkpoint type by checking state_dict structure:
            # - Fine-tuning checkpoints: have 'model.backbone.' or 'model.classifier.' prefixes (saved from LightningModule)
            # - Pretraining checkpoints: have direct keys like 'encoder_imaging.xxx' (saved from regular nn.Module)
            # Also check if checkpoint was saved during fine-tuning (checkpoint_hparams.finetune=True)
            state_dict = checkpoint['state_dict']
            has_model_prefix = any(key.startswith('model.') for key in state_dict.keys())
            checkpoint_is_finetune = getattr(checkpoint_hparams, 'finetune', False)
            is_finetune_checkpoint = has_model_prefix or checkpoint_is_finetune
            
            if is_finetune_checkpoint:
                # Fine-tuning checkpoint: Preprocess state_dict to strip 'model.backbone.' prefix
                print("✅ Detected fine-tuning checkpoint (saved from previous fine-tuning)")
                print("   Preprocessing state_dict keys to match TIPBackbone structure...")
                
                preprocessed_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('model.backbone.'):
                        new_key = key[15:]  # Remove 'model.backbone.'
                        # Skip regression head keys - they will be loaded separately
                        if not new_key.startswith('regression.'):
                            preprocessed_state_dict[new_key] = value
                    # Skip 'model.regression.' keys (will be loaded separately)
                
                preprocessed_checkpoint = {
                    'hyper_parameters': checkpoint['hyper_parameters'],
                    'state_dict': preprocessed_state_dict
                }
            else:
                print("✅ Detected pretraining checkpoint (loading weights and architecture)")
        
        # Set required attributes for TIPBackbone
        with open_dict(hparams):
            if field_lengths_path:
                hparams.field_lengths_tabular = field_lengths_path
            if not hasattr(hparams, 'algorithm_name'):
                hparams.algorithm_name = 'tip'
            if not hasattr(hparams, 'missing_tabular'):
                hparams.missing_tabular = False
            if not hasattr(hparams, 'task'):
                hparams.task = 'regression'
            if not hasattr(hparams, 'num_classes'):
                hparams.num_classes = 1
            
            # Set checkpoint path for TIPBackbone (always set so it uses if args.checkpoint: block)
            hparams.checkpoint = checkpoint_path if checkpoint_path else None
        
        # Create TIPBackbone without classifier (it will return x_m features)
        # Set create_classifier=False so TIPBackbone doesn't create a classifier
        with open_dict(hparams):
            hparams.create_classifier = False
        
        # Create TIPBackbone (always uses if args.checkpoint: block)
        # If fine-tuning checkpoint, pass preprocessed_checkpoint dict
        self.backbone = TIPBackbone(hparams, checkpoint_dict=preprocessed_checkpoint)
        
        # Create regression head (separate from backbone)
        # Load head from checkpoint if available
        if checkpoint:
            self._load_head_from_checkpoint(checkpoint, hparams, is_finetune_checkpoint)
        else:
            # No checkpoint: create head from hparams
            self._create_head_from_hparams(hparams)
    
    def _load_head_from_checkpoint(self, checkpoint, hparams, is_finetune_checkpoint):
        """Load head from checkpoint (backbone already loaded by TIPBackbone)."""
        if is_finetune_checkpoint:
            # Fine-tuning checkpoint: Extract head weights from state_dict
            print("Loading regression head weights from fine-tuning checkpoint state_dict...")
            state_dict = checkpoint['state_dict']
            
            # Extract regression head weights (with 'model.regression.' prefix)
            regression_state_dict = {}
            prefix = 'model.regression.'
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    regression_key = key[len(prefix):]  # Remove 'model.regression.'
                    regression_state_dict[regression_key] = value
            
            if regression_state_dict:
                # Get head class from checkpoint hyperparameters
                regression_head_class = getattr(hparams, 'regression_head_class', 'RegressionMLP')
                print(f"✅ Using regression head class from checkpoint: {regression_head_class}")
                
                # Create head with correct architecture
                z_dim = hparams.multimodal_embedding_dim
                hidden_dim = getattr(hparams, 'embedding_dim', z_dim)
                drop_p = getattr(hparams, 'regression_head_dropout', 0.2)
                
                # Get target normalization parameters from hparams
                target_mean = getattr(hparams, 'target_mean', 0.0)
                target_std = getattr(hparams, 'target_std', 1.0)
                target_log_transform = getattr(hparams, 'target_log_transform', True)
                loss_type = getattr(hparams, 'regression_loss', {'rmsle': 1.0})
                huber_delta = getattr(hparams, 'huber_delta', 1.0)
                
                # Preprocess loss_type: convert DictConfig to dict before passing to head
                if isinstance(loss_type, DictConfig):
                    loss_type = dict(loss_type)
                
                # Create regression head (always named self.regression)
                self.regression = create_head(
                    head_name=regression_head_class,
                    n_input=z_dim,
                    loss_type=loss_type,
                    n_hidden=hidden_dim,
                    p=drop_p,
                    target_mean=target_mean,
                    target_std=target_std,
                    target_log_transform=target_log_transform,
                    huber_delta=huber_delta
                )
                
                # Load head weights
                self.regression.load_state_dict(regression_state_dict)
                print("✅ Loaded regression head weights from fine-tuning checkpoint")
            else:
                print("⚠️  Warning: No regression head weights found in fine-tuning checkpoint state_dict")
                print("   Creating new regression head from hparams")
                self._create_head_from_hparams(hparams)
        else:
            # Pretraining checkpoint: Head is in callback state
            print("Loading regression head from pretraining checkpoint callback state...")
            
            if 'callbacks' in checkpoint:
                # Find the SSLOnlineEvaluatorRegression callback state
                for key in checkpoint['callbacks'].keys():
                    if 'SSLOnlineEvaluatorRegression' in key or 'ssl_online_regression' in key.lower():
                        callback_state = checkpoint['callbacks'][key]
                        if 'state_dict' in callback_state:
                            try:
                                # Get head class name from callback state
                                callback_head_class = callback_state.get('regression_head_class', 'RegressionMLP')
                                print(f"✅ Found online regression head class: {callback_head_class}")
                                
                                # Create head using the class from callback state
                                z_dim = hparams.multimodal_embedding_dim
                                hidden_dim = getattr(hparams, 'embedding_dim', z_dim)
                                drop_p = getattr(hparams, 'regression_head_dropout', 0.2)
                                
                                # Get target normalization parameters from hparams
                                target_mean = getattr(hparams, 'target_mean', 0.0)
                                target_std = getattr(hparams, 'target_std', 1.0)
                                target_log_transform = getattr(hparams, 'target_log_transform', True)
                                loss_type = getattr(hparams, 'regression_loss', {'rmsle': 1.0})
                                huber_delta = getattr(hparams, 'huber_delta', 1.0)
                                
                                # Preprocess loss_type: convert DictConfig to dict before passing to head
                                if isinstance(loss_type, DictConfig):
                                    loss_type = dict(loss_type)
                                
                                # Create regression head (always named self.regression)
                                self.regression = create_head(
                                    head_name=callback_head_class,
                                    n_input=z_dim,
                                    loss_type=loss_type,
                                    n_hidden=hidden_dim,
                                    p=drop_p,
                                    target_mean=target_mean,
                                    target_std=target_std,
                                    target_log_transform=target_log_transform,
                                    huber_delta=huber_delta
                                )
                                
                                # Load the trained weights from pretraining
                                self.regression.load_state_dict(callback_state['state_dict'])
                                print("✅ Loaded trained regression head from pretraining checkpoint")
                                return  # Successfully loaded, exit
                            except Exception as e:
                                print(f"⚠️  Warning: Could not load regression head: {e}")
                                import traceback
                                traceback.print_exc()
            
            # If we get here, no head was loaded from callback
            print("⚠️  WARNING: No regression head found in pretraining checkpoint callback state!")
            print("   Creating new regression head from hparams (random weights)")
            print("   This will result in poor performance")
            self._create_head_from_hparams(hparams)
    
    def _create_head_from_hparams(self, hparams):
        """Create head from hyperparameters (when no checkpoint available)."""
        regression_head_class = getattr(hparams, 'regression_head_class', 'RegressionMLP')
        print(f"Creating regression head from hparams: {regression_head_class}")
        
        z_dim = hparams.multimodal_embedding_dim
        hidden_dim = getattr(hparams, 'embedding_dim', z_dim)
        drop_p = getattr(hparams, 'regression_head_dropout', 0.2)
        
        # Get target normalization parameters from hparams
        target_mean = getattr(hparams, 'target_mean', 0.0)
        target_std = getattr(hparams, 'target_std', 1.0)
        target_log_transform = getattr(hparams, 'target_log_transform', True)
        loss_type = getattr(hparams, 'regression_loss', {'rmsle': 1.0})
        huber_delta = getattr(hparams, 'huber_delta', 1.0)
        
        # Preprocess loss_type: convert DictConfig to dict before passing to head
        if isinstance(loss_type, DictConfig):
            loss_type = dict(loss_type)
        
        # Create regression head (separate from backbone)
        self.regression = create_head(
            head_name=regression_head_class,
            n_input=z_dim,
            loss_type=loss_type,
            n_hidden=hidden_dim,
            p=drop_p,
            target_mean=target_mean,
            target_std=target_std,
            target_log_transform=target_log_transform,
            huber_delta=huber_delta
        )
    
    def forward(self, x, visualize=False):
        """
        Forward pass.
        
        Args:
            x: Tuple of (image, tabular) or (image, tabular, mask)
            visualize: Whether to return attention maps
        
        Returns:
            If visualize=False:
                dict with keys:
                    - 'prediction_log': (B,) - Prediction in normalized log space
                    - 'prediction_original': (B,) - Prediction in original scale (USD/m²)
            If visualize=True:
                dict with keys:
                    - 'prediction_log': (B,) - Prediction in normalized log space
                    - 'prediction_original': (B,) - Prediction in original scale (USD/m²)
                    - 'attn': Attention maps for visualization
        """
        # Get multimodal features from backbone (x_m shape: (B, 20, 512))
        if visualize:
            x_m, attn = self.backbone(x, visualize=True)
        else:
            x_m = self.backbone(x, visualize=False)
        
        # Pass x_m directly to regression head
        # The regression head will handle aggregation internally
        result = self.regression(x_m)  # Returns dict with 'prediction_log' and 'prediction_original'
        
        if visualize:
            result['attn'] = attn
        
        return result

