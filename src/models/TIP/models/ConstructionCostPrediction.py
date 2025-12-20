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
from omegaconf import OmegaConf, open_dict
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
        checkpoint_path: Path to checkpoint file (for loading weights)
        field_lengths_path: Path to field_lengths.pt file
    """
    def __init__(self, hparams, checkpoint_path: str = None, field_lengths_path: str = None):
        super().__init__()
        
        # Store checkpoint path for loading weights
        self.checkpoint_path = checkpoint_path
        
        # Load checkpoint if provided
        checkpoint = None
        if checkpoint_path:
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_hparams = OmegaConf.create(checkpoint['hyper_parameters'])
            
            # Merge checkpoint hyperparameters with provided hparams
            with open_dict(hparams):
                # Copy architecture params from checkpoint if not in hparams
                if hasattr(checkpoint_hparams, 'multimodal_embedding_dim') and not hasattr(hparams, 'multimodal_embedding_dim'):
                    hparams.multimodal_embedding_dim = checkpoint_hparams.multimodal_embedding_dim
                if hasattr(checkpoint_hparams, 'embedding_dim') and not hasattr(hparams, 'embedding_dim'):
                    hparams.embedding_dim = checkpoint_hparams.embedding_dim
                if hasattr(checkpoint_hparams, 'model') and not hasattr(hparams, 'model'):
                    hparams.model = checkpoint_hparams.model
                # ... add more if needed
        
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
                hparams.num_classes = 1  # Regression: single output
        
        # Create TIPBackbone (will have default Linear classifier)
        # Detection logic:
        # - If checkpoint_path is provided AND hparams has 'pretrain_checkpoint_path', 
        #   it means we're fine-tuning FROM a pretraining checkpoint (current case).
        #   We should pass the checkpoint to TIPBackbone so it loads the weights.
        # - If checkpoint_path is provided but NO 'pretrain_checkpoint_path',
        #   it's a pretraining checkpoint being loaded directly.
        # - If checkpoint_path is a fine-tuning checkpoint (saved during fine-tuning),
        #   it would have all architecture params in hparams already.
        
        if checkpoint_path:
            # Check if this is a fine-tuning checkpoint (saved during fine-tuning)
            # by checking if checkpoint_path points to a fine-tuning run directory
            is_finetune_checkpoint = 'finetune' in str(checkpoint_path) and not hasattr(hparams, 'pretrain_checkpoint_path')
            
            if is_finetune_checkpoint:
                # Fine-tuning checkpoint: has all architecture params, create from args
                print("✅ Detected fine-tuning checkpoint (saved from previous fine-tuning)")
                with open_dict(hparams):
                    hparams.checkpoint = None  # Don't load from checkpoint, create from args
            else:
                # Pretraining checkpoint: needs to load architecture and weights from checkpoint
                print("✅ Detected pretraining checkpoint (loading weights and architecture)")
                with open_dict(hparams):
                    hparams.checkpoint = checkpoint_path
                    print(f"✅ Pretraining checkpoint: {hparams.checkpoint}")
        else:
            # No checkpoint: create new model from args
            print("✅ No checkpoint provided, creating new model from args")
            with open_dict(hparams):
                hparams.checkpoint = None
        
        self.backbone = TIPBackbone(hparams)
        
        # Load weights and head from checkpoint
        if checkpoint:
            self._load_weights_and_head_from_checkpoint(checkpoint, hparams, is_finetune_checkpoint)
        else:
            # No checkpoint: create head from hparams
            self._create_head_from_hparams(hparams)
    
    def _load_weights_and_head_from_checkpoint(self, checkpoint, hparams, is_finetune_checkpoint):
        """Load backbone weights and head from checkpoint (either from state_dict or callback state)."""
        if is_finetune_checkpoint:
            # Fine-tuning checkpoint: Load all weights from state_dict (with 'model.' prefix)
            print("Loading model weights from fine-tuning checkpoint state_dict...")
            state_dict = checkpoint['state_dict']
            
            # Strip 'model.' prefix to match TIPBackbone structure
            backbone_state_dict = {}
            classifier_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.'
                    if new_key.startswith('classifier.'):
                        # Extract classifier weights separately
                        classifier_key = new_key[11:]  # Remove 'classifier.'
                        classifier_state_dict[classifier_key] = value
                    else:
                        # Backbone weights (encoders, etc.)
                        backbone_state_dict[new_key] = value
            
            # Load backbone weights (encoders)
            if backbone_state_dict:
                missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
                if missing_keys:
                    print(f"⚠️  Warning: Missing keys when loading backbone: {missing_keys}")
                if unexpected_keys:
                    print(f"⚠️  Warning: Unexpected keys when loading backbone: {unexpected_keys}")
                print("✅ Loaded backbone weights from fine-tuning checkpoint")
            
            # Load head
            if classifier_state_dict:
                # Get head class from checkpoint hyperparameters
                regression_head_class = getattr(hparams, 'regression_head_class', 'RegressionMLP')
                print(f"✅ Using regression head class from checkpoint: {regression_head_class}")
                
                # Create head with correct architecture
                z_dim = hparams.multimodal_embedding_dim
                hidden_dim = getattr(hparams, 'embedding_dim', z_dim)
                drop_p = getattr(hparams, 'regression_head_dropout', 0.2)
                
                self.backbone.classifier = create_head(
                    head_name=regression_head_class,
                    n_input=z_dim,
                    n_hidden=hidden_dim,
                    p=drop_p
                )
                
                # Load head weights
                self.backbone.classifier.load_state_dict(classifier_state_dict)
                print("✅ Loaded head weights from fine-tuning checkpoint")
            else:
                print("⚠️  Warning: No classifier weights found in fine-tuning checkpoint state_dict")
                self._create_head_from_hparams(hparams)
        else:
            # Pretraining checkpoint: Head is in callback state
            print("Loading head from pretraining checkpoint callback state...")
            online_regression_head = None
            
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
                                
                                online_regression_head = create_head(
                                    head_name=callback_head_class,
                                    n_input=z_dim,
                                    n_hidden=hidden_dim,
                                    p=drop_p
                                )
                                
                                # Load the trained weights from pretraining
                                online_regression_head.load_state_dict(callback_state['state_dict'])
                                print("✅ Loaded trained online regression head from pretraining checkpoint")
                                break
                            except Exception as e:
                                print(f"⚠️  Warning: Could not load online regression head: {e}")
                                import traceback
                                traceback.print_exc()
            
            if online_regression_head is not None:
                # Replace TIPBackbone's default classifier with the loaded head
                self.backbone.classifier = online_regression_head
            else:
                print("⚠️  WARNING: No online regression head found in pretraining checkpoint!")
                print("   Using default Linear classifier (random weights)")
                print("   This will result in poor performance")
                self._create_head_from_hparams(hparams)
    
    def _create_head_from_hparams(self, hparams):
        """Create head from hyperparameters (when no checkpoint available)."""
        regression_head_class = getattr(hparams, 'regression_head_class', 'RegressionMLP')
        print(f"Creating regression head from hparams: {regression_head_class}")
        
        z_dim = hparams.multimodal_embedding_dim
        hidden_dim = getattr(hparams, 'embedding_dim', z_dim)
        drop_p = getattr(hparams, 'regression_head_dropout', 0.2)
        
        # Replace default classifier with head from ConstructionCostHead
        self.backbone.classifier = create_head(
            head_name=regression_head_class,
            n_input=z_dim,
            n_hidden=hidden_dim,
            p=drop_p
        )
    
    def forward(self, x, visualize=False):
        """
        Forward pass.
        
        Args:
            x: Tuple of (image, tabular) or (image, tabular, mask)
            visualize: Whether to return attention maps
        
        Returns:
            prediction: (B, 1) - Predicted cost in normalized log space
        """
        return self.backbone(x, visualize=visualize)

