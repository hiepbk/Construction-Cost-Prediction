'''
Satellite Image Encoder for TIP
Supports:
- SatlasPretrain backbone (preferred)
- ResNet with multi-channel input (15 channels: 12 Sentinel-2 + 3 VIIRS)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

# Import satlaspretrain_models package (installed via pip)
try:
    import satlaspretrain_models
    from satlaspretrain_models.model import Weights, Model
    from satlaspretrain_models.utils import Backbone, Head
    SATLAS_AVAILABLE = True
except ImportError:
    SATLAS_AVAILABLE = False
    print("Warning: satlaspretrain_models not available, will use ResNet")


class SatelliteImageEncoder(nn.Module):
    """
    Image encoder for satellite imagery (Sentinel-2 + VIIRS).
    
    Options:
    1. SatlasPretrain (preferred): Uses pretrained SatlasPretrain backbone
    2. ResNet: Standard ResNet adapted for 15-channel input
    """
    
    def __init__(self, args):
        """
        Args:
            args: Config with:
                - model: 'satlas' or 'resnet18'/'resnet50'
                - use_satlas: bool (use SatlasPretrain if available)
                - satellite_feature_dim: int (output feature dimension)
                - embedding_dim: int (for projection head)
        """
        super().__init__()
        
        self.use_satlas = getattr(args, 'use_satlas', False) and SATLAS_AVAILABLE
        self.model_name = getattr(args, 'model', 'resnet50')
        
        # Check if model is 'satlas' or use_satlas flag is set
        if (self.model_name == 'satlas' or self.use_satlas) and SATLAS_AVAILABLE:
            # Use SatlasPretrain
            self.encoder_type = 'satlas'
            self.encoder = self._create_satlas_encoder(args)
            # Satlas outputs (B, feature_dim), need to get pooled features
            self.pooled_dim = getattr(args, 'satellite_feature_dim', 512) * 2  # Sentinel-2 + VIIRS
        elif self.model_name.startswith('resnet'):
            # Use ResNet with multi-channel input
            self.encoder_type = 'resnet'
            self.encoder = self._create_resnet_encoder(args)
            # ResNet outputs feature maps, we'll pool them
            self.pooled_dim = 2048 if '50' in self.model_name else 512
        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Supported: 'satlas', 'resnet18', 'resnet50'")
    
    def _create_satlas_encoder(self, args):
        """Create SatlasPretrain encoder for Sentinel-2 and VIIRS"""
        satellite_feature_dim = getattr(args, 'satellite_feature_dim', 512)
        freeze_backbone = getattr(args, 'freeze_backbone', False)
        use_fpn = getattr(args, 'use_fpn', True)
        sentinel2_model_id = getattr(args, 'sentinel2_model_id', 'Sentinel2_SwinB_SI_MS')
        
        # Initialize weights manager
        weights_manager = Weights()
        
        # Load Sentinel-2 model (12 channels)
        sentinel2_model = weights_manager.get_pretrained_model(
            sentinel2_model_id,
            fpn=use_fpn
        )
        
        # For VIIRS (3 channels), we can use a Sentinel-1 model or create a simple encoder
        # Since VIIRS is nighttime lights, we'll create a simple CNN encoder
        # Alternatively, we can use Sentinel-1 model if available
        viirs_model = self._create_viirs_encoder()
        
        # Combine both encoders
        encoder = MultiSatelliteEncoder(
            sentinel2_model=sentinel2_model,
            viirs_model=viirs_model,
            feature_dim=satellite_feature_dim,
            freeze_backbone=freeze_backbone,
            use_fpn=use_fpn
        )
        return encoder
    
    def _create_viirs_encoder(self):
        """Create a simple CNN encoder for VIIRS (3 channels)"""
        # Simple CNN for VIIRS nighttime lights
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256)  # Output dimension
        )
    
    def _create_resnet_encoder(self, args):
        """Create ResNet encoder adapted for 15-channel input"""
        # Get base ResNet (expects 3 channels)
        base_resnet = torchvision_ssl_encoder(self.model_name, return_all_feature_maps=True)
        
        # Replace first conv layer to accept 15 channels
        if hasattr(base_resnet, 'conv1'):
            # Standard ResNet
            old_conv = base_resnet.conv1
            new_conv = nn.Conv2d(
                15,  # 12 Sentinel-2 + 3 VIIRS
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize: average pretrained weights across input channels
            if old_conv.weight.shape[1] == 3:
                # Repeat pretrained weights 5 times (15/3 = 5)
                with torch.no_grad():
                    new_conv.weight.data = old_conv.weight.data.repeat(1, 5, 1, 1)
            base_resnet.conv1 = new_conv
        elif hasattr(base_resnet, 'features'):
            # ResNet in features module
            if hasattr(base_resnet.features, '0'):
                old_conv = base_resnet.features[0]
                if isinstance(old_conv, nn.Conv2d):
                    new_conv = nn.Conv2d(
                        15,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None
                    )
                    if old_conv.weight.shape[1] == 3:
                        with torch.no_grad():
                            new_conv.weight.data = old_conv.weight.data.repeat(1, 5, 1, 1)
                    base_resnet.features[0] = new_conv
        
        return base_resnet
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through image encoder.
        
        Args:
            x: (B, 15, H, W) - Combined Sentinel-2 (12) + VIIRS (3) channels
        
        Returns:
            features: List of feature maps (for compatibility with TIP)
            Last element is the final feature map
        """
        if self.encoder_type == 'satlas':
            # Split into Sentinel-2 and VIIRS
            sentinel2_full = x[:, :12, :, :]  # (B, 12, H, W)
            viirs = x[:, 12:15, :, :]   # (B, 3, H, W)
            
            # Sentinel2_SwinB_SI_MS expects 9 channels, not 12
            # Our dataset band order: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
            # SatlasPretrain MS model expects 9 bands (standard multispectral selection)
            # Select 9 bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11
            # Skip: B1 (coastal aerosol, 60m), B9 (water vapor, 60m), B12 (SWIR2, 20m)
            # Band indices (0-indexed): [1, 2, 3, 4, 5, 6, 7, 8, 10]
            # This corresponds to: B2, B3, B4, B5, B6, B7, B8, B8A, B11
            band_indices = [1, 2, 3, 4, 5, 6, 7, 8, 10]  # 9 bands
            sentinel2 = sentinel2_full[:, band_indices, :, :].contiguous()  # (B, 9, H, W) - ensure contiguous for DDP
            
            # Get features from Satlas
            features = self.encoder(sentinel2, viirs)  # (B, 2*feature_dim)
            
            # Reshape to match TIP's expected format (list of feature maps)
            # For compatibility, return as list with single element
            # TIP expects: [f1, f2, f3, f4] where last is used
            # We'll create a dummy spatial dimension for compatibility
            B, C = features.shape
            # Reshape to (B, C, 1, 1) to match ResNet output format
            # Use contiguous() to ensure proper memory layout for DDP
            features_spatial = features.unsqueeze(-1).unsqueeze(-1).contiguous()  # (B, C, 1, 1)
            return [features_spatial]  # Return as list for compatibility
        
        elif self.encoder_type == 'resnet':
            # Standard ResNet forward
            return self.encoder(x)
        
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")


class MultiSatelliteEncoder(nn.Module):
    """
    Combines Sentinel-2 and VIIRS encoders into a single encoder.
    """
    def __init__(self, sentinel2_model, viirs_model, feature_dim=512, freeze_backbone=False, use_fpn=True):
        super().__init__()
        self.sentinel2_model = sentinel2_model
        self.viirs_model = viirs_model
        self.feature_dim = feature_dim
        self.use_fpn = use_fpn
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.sentinel2_model.parameters():
                param.requires_grad = False
            for param in self.viirs_model.parameters():
                param.requires_grad = False
        
        # Determine Sentinel-2 output dimension
        # FPN outputs 128 channels, backbone outputs vary by architecture
        # For SwinB: backbone outputs multi-scale features, FPN outputs 128 channels
        if use_fpn:
            s2_dim = 128  # FPN output channels
        else:
            # Backbone only - SwinB outputs features at different scales
            # We'll use the last scale which is typically 1024 for SwinB
            s2_dim = 1024
        
        viirs_dim = 256  # From our simple CNN
        
        # Projection layers
        self.s2_projection = nn.Linear(s2_dim, feature_dim)
        self.viirs_projection = nn.Linear(viirs_dim, feature_dim)
        
    def forward(self, sentinel2, viirs):
        """
        Args:
            sentinel2: (B, 12, H, W) Sentinel-2 image
            viirs: (B, 3, H, W) VIIRS image
        
        Returns:
            features: (B, 2*feature_dim) Combined features
        """
        # Process Sentinel-2
        s2_features = self.sentinel2_model(sentinel2)
        
        # Handle FPN output (list of feature maps) or backbone output
        if isinstance(s2_features, (list, tuple)):
            # FPN returns list, use the last one (highest resolution)
            s2_features = s2_features[-1]
        
        # Pool spatial dimensions if needed
        if s2_features.dim() == 4:
            # Global average pooling
            s2_features = F.adaptive_avg_pool2d(s2_features, (1, 1))
            s2_features = s2_features.flatten(1)  # (B, C)
        
        # Process VIIRS
        viirs_features = self.viirs_model(viirs)  # (B, 256)
        
        # Project to feature_dim
        s2_proj = self.s2_projection(s2_features)  # (B, feature_dim)
        viirs_proj = self.viirs_projection(viirs_features)  # (B, feature_dim)
        
        # Concatenate
        combined = torch.cat([s2_proj, viirs_proj], dim=1)  # (B, 2*feature_dim)
        
        return combined


def create_satellite_image_encoder(args):
    """
    Factory function to create satellite image encoder.
    
    Args:
        args: Config object
    
    Returns:
        encoder: SatelliteImageEncoder instance
    """
    return SatelliteImageEncoder(args)

