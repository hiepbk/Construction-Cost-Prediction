"""
Configuration file for training parameters
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    train_csv: str = 'data/train_tabular.csv'
    composite_dir: str = 'data/train_composite'
    image_size: int = 224
    val_split: float = 0.2
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    use_augmentation: bool = True


@dataclass
class SatelliteConfig:
    """Satellite imagery model configuration"""
    # SatlasPretrain model identifier
    sentinel2_model_id: str = "Sentinel2_SwinB_SI_MS"  # Options: Sentinel2_SwinB_SI_MS, Sentinel2_SwinB_SI_RGB, Sentinel2_SwinT_SI_MS
    viirs_model_id: str = "Sentinel1_SwinB_SI"  # For VIIRS (1 band)
    use_fpn: bool = True
    feature_dim: int = 512
    freeze_backbone: bool = False  # Set True to freeze satellite backbone (pretrained weights)
    freeze_satellite_model: bool = False  # Set True to freeze entire satellite model (backbone + projection layers)


@dataclass
class TabularConfig:
    """Tabular data model configuration"""
    # FT-Transformer configuration
    d_token: int = 192  # Token embedding dimension
    n_layers: int = 3  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads
    d_ffn_factor: float = 4.0  # FFN dimension factor
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.1
    pooling: str = 'cls'  # 'cls', 'mean', 'max'
    output_dim: int = 512


@dataclass
class FusionConfig:
    """Multi-modal fusion configuration"""
    fusion_type: str = 'cross_attention'  # 'cross_attention' or 'gated'
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class ModelConfig:
    """Complete model configuration"""
    # Output head configuration
    output_hidden_dim: int = 512
    output_dropout: float = 0.1
    
    # Note: num_numerical_features and categorical info will be inferred from data
    # categorical_cardinalities will be computed automatically


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 8
    epochs: int = 100
    device: str = 'cuda'  # 'cuda' or 'cpu'
    
    # TIP Pretraining (optional)
    use_tip_pretraining: bool = False  # Whether to use TIP pretraining before fine-tuning
    tip_pretrain_epochs: int = 50  # Number of pretraining epochs
    tip_projection_dim: int = 256  # Projection dimension for contrastive learning
    tip_temperature: float = 0.07  # Temperature for InfoNCE loss
    tip_label_similarity_threshold: float = 0.1  # Threshold for label-aware contrastive learning
    tip_mask_prob: float = 0.15  # Probability of masking tabular features
    
    # Learning rates
    base_lr: float = 1e-3  # Base learning rate
    tabular_lr: Optional[float] = None  # If None, uses base_lr
    satellite_lr: Optional[float] = None  # If None, uses base_lr * 0.1
    fusion_lr: Optional[float] = None  # If None, uses base_lr
    
    # Optimizer
    optimizer: str = 'adamw'  # 'adamw' or 'adam'
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss and metrics
    loss_type: str = 'msle'  # 'msle' (for RMSLE), 'mse', 'mae', 'huber', 'combined'
    huber_delta: float = 1.0
    loss_weights: dict = None  # For 'combined' loss: e.g., {'msle': 0.8, 'mae': 0.2}
    
    # Regularization
    gradient_clip: float = 1.0
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision (FP16)
    
    # Validation
    val_metric: str = 'rmsle'  # Primary metric for model selection: 'rmsle', 'rmse', 'mae'


@dataclass
class OutputConfig:
    """Output and checkpoint configuration"""
    work_dir: str = 'workdir'  # Base directory for all outputs (will create timestamped subdirectory)
    save_best: bool = True
    save_last: bool = True
    save_frequency: int = 10  # Save checkpoint every N epochs
    log_interval: int = 10  # Print logs every N iterations
    print_lr: bool = True  # Print learning rate in logs
    print_time: bool = True  # Print elapsed time in logs


@dataclass
class Config:
    """Complete configuration"""
    data: DataConfig = None
    satellite: SatelliteConfig = None
    tabular: TabularConfig = None
    fusion: FusionConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    output: OutputConfig = None
    
    def __post_init__(self):
        """Initialize default configs if not provided"""
        if self.data is None:
            self.data = DataConfig()
        if self.satellite is None:
            self.satellite = SatelliteConfig()
        if self.tabular is None:
            self.tabular = TabularConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.output is None:
            self.output = OutputConfig()
        
        # Set default learning rates if not specified
        if self.training.tabular_lr is None:
            self.training.tabular_lr = self.training.base_lr
        if self.training.satellite_lr is None:
            self.training.satellite_lr = self.training.base_lr * 0.1
        if self.training.fusion_lr is None:
            self.training.fusion_lr = self.training.base_lr

