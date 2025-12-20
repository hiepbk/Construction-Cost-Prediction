'''
TIP Dataset Adapter for Construction Cost Prediction
Adapts TIP's ContrastiveReconstructImagingAndTabularDataset for:
- Sentinel-2 + VIIRS satellite imagery
- Tabular features (economic, geographic, infrastructure, risk)
- Construction cost regression target
'''
from typing import List, Tuple, Optional
import random
import copy
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import rasterio
from torchvision.transforms import transforms
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

# Import our existing dataset utilities
import sys
# Add project root to path (from src/models/TIP/datasets/ -> project root needs ../../../..)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.data.augmentations import build_satellite_augmentations, apply_augmentations


def convert_to_float(x):
    return x.float()


def convert_satellite_to_tensor(x, **kwargs):
    """Convert satellite numpy array to tensor"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    # Ensure shape is (C, H, W)
    if x.dim() == 4:
        x = x.squeeze(0)
    x = x.permute(2,0,1)  # (H, W, C) -> (C, H, W)
    return x


class ConstructionCostTIPDataset(Dataset):
    """
    TIP-compatible dataset for construction cost prediction.
    
    Generates multiple views for contrastive learning:
    - Image views: augmented and unaugmented satellite imagery
    - Tabular views: original, corrupted, and masked (for TR loss)
    
    Compatible with TIP's three-loss framework:
    - ITC: Image-Tabular Contrastive
    - ITM: Image-Tabular Matching  
    - TR: Tabular Reconstruction
    """
    
    def __init__(
        self,
        csv_path,  # Can be str (path to CSV) or pd.DataFrame
        composite_dir: str,
        field_lengths_tabular: str,  # Path to .pt file with field lengths
        labels_path: str = None,  # Path to .pt file with labels (for pretraining, can be None)
        img_size: int = 224,
        is_train: bool = True,
        corruption_rate: float = 0.3,
        replace_random_rate: float = 0.0,
        replace_special_rate: float = 0.5,
        augmentation_rate: float = 0.95,
        one_hot_tabular: bool = False,
        use_sentinel2: bool = True,
        use_viirs: bool = True,
        live_loading: bool = False,
        augmentation_speedup: bool = True,
        target_log_transform: bool = True,  # Log-transform target for regression
        metadata_path: str = None,  # Path to metadata.pkl file (from preprocessing)
    ):
        """
        Args:
            csv_path: Path to CSV with tabular data and file paths, OR a pandas DataFrame
            composite_dir: Directory containing satellite TIFF files
            field_lengths_tabular: Path to .pt file with field lengths (for TIP tabular encoder)
            labels_path: Path to .pt file with labels (for pretraining contrastive learning)
            img_size: Image size for resizing
            is_train: Whether this is training data
            corruption_rate: Rate for tabular corruption (for TR loss)
            replace_random_rate: Rate for random replacement in masking
            replace_special_rate: Rate for special replacement in masking
            augmentation_rate: Probability of applying augmentation to second image view
            one_hot_tabular: Whether to one-hot encode tabular features
            use_sentinel2: Whether to use Sentinel-2 imagery
            use_viirs: Whether to use VIIRS imagery
            live_loading: Whether to load images on-the-fly (vs preloaded)
            augmentation_speedup: Use albumentations for faster augmentation
            target_log_transform: Log-transform target for regression
            metadata_path: Path to metadata.pkl file (from preprocessing). If None, will try to infer from csv_path.
        """
        self.csv_path = csv_path
        self.composite_dir = composite_dir
        self.img_size = img_size
        self.is_train = is_train
        self.use_sentinel2 = use_sentinel2
        self.use_viirs = use_viirs
        self.live_loading = live_loading
        self.augmentation_speedup = augmentation_speedup
        self.target_log_transform = target_log_transform
        self.metadata_path = metadata_path
        
        # Validate composite_dir exists and contains files (critical check to avoid silent failures)
        if (self.use_sentinel2 or self.use_viirs) and composite_dir:
            if not os.path.exists(composite_dir):
                raise FileNotFoundError(
                    f"❌ CRITICAL ERROR: Composite directory does not exist: {composite_dir}\n"
                    f"   Please check your config file and ensure the path is correct.\n"
                    f"   Expected paths: data/trainval_composite (for train/val) or data/test_composite (for test)"
                )
            if not os.path.isdir(composite_dir):
                raise NotADirectoryError(
                    f"❌ CRITICAL ERROR: Composite path is not a directory: {composite_dir}"
                )
            
            # Check if directory contains any .tif files
            tif_files = list(Path(composite_dir).glob("*.tif"))
            if len(tif_files) == 0:
                print(f"⚠️  WARNING: Composite directory contains no .tif files: {composite_dir}")
                print(f"   This might indicate the wrong directory was specified.")
                print(f"   Expected paths: data/trainval_composite (for train/val) or data/test_composite (for test)")
        
        # Load CSV or use provided DataFrame
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.copy()  # Make a copy to avoid modifying original
            self.csv_path_str = None  # No file path when using DataFrame
        elif isinstance(csv_path, str):
            self.df = pd.read_csv(csv_path)
            self.csv_path_str = csv_path
        else:
            raise TypeError(f"csv_path must be a string (CSV path) or pandas DataFrame, got {type(csv_path)}")
        
        # Load field lengths (required by TIP tabular encoder)
        if isinstance(field_lengths_tabular, str):
            self.field_lengths_tabular = torch.load(field_lengths_tabular)
        else:
            self.field_lengths_tabular = field_lengths_tabular
        
        # Tabular corruption parameters
        self.c = corruption_rate
        self.replace_random_rate = replace_random_rate
        self.replace_special_rate = replace_special_rate
        self.one_hot_tabular = one_hot_tabular
        
        # Process tabular data
        self._process_tabular_data()
        
        # Generate marginal distributions for corruption
        self.generate_marginal_distributions()
        
        # Image augmentation
        self.augmentation_rate = augmentation_rate if is_train else 0.0
        self.satellite_augmentations = []  # Initialize, will be set in _setup_augmentations
        self._setup_augmentations()
        
        # Labels (for pretraining - can be indices for contrastive learning)
        if labels_path and os.path.exists(labels_path):
            self.labels = torch.load(labels_path)
        else:
            # Create dummy labels (sample indices) for contrastive learning
            self.labels = torch.arange(len(self.df), dtype=torch.long)
        
        # Target values (for regression fine-tuning)
        if 'construction_cost_per_m2_usd' in self.df.columns:
            # Store original targets (before any transformation) for logging
            self.targets_original = self.df['construction_cost_per_m2_usd'].values.copy()
            
            targets = self.df['construction_cost_per_m2_usd'].values
            if target_log_transform:
                # Log-transform: log(1 + cost) to handle zeros and reduce skew
                targets = np.log1p(targets)
            self.targets = torch.from_numpy(targets).float()
        else:
            self.targets = None
            self.targets_original = None
        
        # Store data_id for each sample (for logging predictions with data_id)
        if 'data_id' in self.df.columns:
            self.data_ids = self.df['data_id'].values.copy()
        else:
            self.data_ids = None
        
        assert len(self.data_tabular) == len(self.labels)
        if self.targets is not None:
            assert len(self.data_tabular) == len(self.targets)
    
    def _process_tabular_data(self):
        """
        Process tabular data to match TIP's expected format.
        Uses metadata from preprocessing to ensure consistent feature mapping.
        TIP expects: list of lists, where each inner list is [cat_feat1, cat_feat2, ..., num_feat1, num_feat2, ...]
        CRITICAL: Categorical features MUST come first, then numerical (hardcoded in TabularTransformerEncoder)
        """
        # Try to load metadata (from preprocessing) for consistent feature mapping
        # Use explicit metadata_path if provided, otherwise try to infer from csv_path
        if self.metadata_path and os.path.exists(self.metadata_path):
            metadata_path = self.metadata_path
        else:
            # Fallback: try to infer from csv_path (only if it's a string path)
            if self.csv_path_str:
                metadata_path = self.csv_path_str.replace('.csv', '_metadata.pkl')
            else:
                # If using DataFrame, metadata_path must be explicitly provided
                metadata_path = None
        
        if os.path.exists(metadata_path):
            import pickle
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Use metadata to get feature columns (excludes excluded_cols)
            included_feature_cols = metadata['included_feature_cols']  # Ordered: categorical first, then numerical
            excluded_cols = metadata.get('excluded_cols', ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'construction_cost_per_m2_usd'])
            categorical_cols = metadata['categorical_cols']
            numerical_cols = metadata['numerical_cols']
            
            print(f"   Using metadata for feature mapping:")
            print(f"     Included features: {len(included_feature_cols)}")
            print(f"     Excluded columns: {excluded_cols}")
            print(f"     Categorical: {len(categorical_cols)}, Numerical: {len(numerical_cols)}")
            
            # Extract features in correct order (categorical first, then numerical)
            # This matches the preprocessing order exactly
            feature_data = self.df[included_feature_cols].values.astype('float32')
            
            # Split into categorical and numerical based on metadata
            num_cat = len(categorical_cols)
            categorical_data = feature_data[:, :num_cat]  # First num_cat columns
            numerical_data = feature_data[:, num_cat:]  # Remaining columns
            
            # Handle missing values in categorical (shouldn't happen after preprocessing, but safety check)
            categorical_data[categorical_data < 0] = 0
            
            # Normalize numerical features using saved stats (if available)
            if len(numerical_cols) > 0 and 'numerical_mean' in metadata and len(metadata['numerical_mean']) > 0:
                # Load normalization stats from metadata
                numerical_mean = np.array([metadata['numerical_mean'][col] for col in numerical_cols])
                numerical_std = np.array([metadata['numerical_std'][col] for col in numerical_cols])
                numerical_std[numerical_std == 0] = 1.0
                numerical_data = (numerical_data - numerical_mean) / numerical_std
                self.numerical_mean = numerical_mean
                self.numerical_std = numerical_std
            else:
                # Fallback: compute normalization (shouldn't happen if preprocessing was done correctly)
                self.numerical_mean = numerical_data.mean(axis=0) if len(numerical_data) > 0 else np.array([])
                self.numerical_std = numerical_data.std(axis=0) if len(numerical_data) > 0 else np.array([])
                self.numerical_std[self.numerical_std == 0] = 1.0
                if len(numerical_data) > 0:
                    numerical_data = (numerical_data - self.numerical_mean) / self.numerical_std
            
        else:
            # Fallback: process without metadata (for backward compatibility)
            print(f"   WARNING: Metadata not found at {metadata_path}, using fallback processing")
            exclude_cols = ['data_id', 'sentinel2_tiff_file_name', 'viirs_tiff_file_name', 'construction_cost_per_m2_usd']
            feature_cols = [col for col in self.df.columns if col not in exclude_cols]
            
            # Separate numerical and categorical
            numerical_cols = []
            categorical_cols = []
            
            for col in feature_cols:
                if self.df[col].dtype in ['int64', 'int32', 'object', 'category']:
                    if self.df[col].dtype == 'object' or self.df[col].dtype == 'category':
                        n_unique = self.df[col].nunique()
                        if n_unique <= 50:
                            self.df[col] = pd.Categorical(self.df[col]).codes.astype('float32')
                            categorical_cols.append(col)
                        else:
                            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype('float32')
                            numerical_cols.append(col)
                    else:
                        n_unique = self.df[col].nunique()
                        if n_unique <= 50:
                            categorical_cols.append(col)
                        else:
                            numerical_cols.append(col)
                else:
                    numerical_cols.append(col)
            
            # Normalize numerical features
            if numerical_cols:
                numerical_data = self.df[numerical_cols].values.astype('float32')
                self.numerical_mean = numerical_data.mean(axis=0)
                self.numerical_std = numerical_data.std(axis=0)
                self.numerical_std[self.numerical_std == 0] = 1.0
                numerical_data = (numerical_data - self.numerical_mean) / self.numerical_std
            else:
                numerical_data = np.array([]).reshape(len(self.df), 0)
                self.numerical_mean = np.array([])
                self.numerical_std = np.array([])
            
            # Process categorical features
            if categorical_cols:
                categorical_data = self.df[categorical_cols].values.astype('float32')
                categorical_data[categorical_data < 0] = 0
            else:
                categorical_data = np.array([]).reshape(len(self.df), 0)
        
        # Combine: [cat_feat1, cat_feat2, ..., num_feat1, num_feat2, ...]
        # CRITICAL: This order matches TIP's TabularTransformerEncoder expectation
        self.data_tabular = []
        for i in range(len(self.df)):
            row = []
            # Categorical features first (matches field_lengths order)
            if len(categorical_data) > 0:
                row.extend(categorical_data[i].tolist())
            # Then numerical features
            if len(numerical_data) > 0:
                row.extend(numerical_data[i].tolist())
            self.data_tabular.append(row)
        
        # Store column info for reference
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.num_categorical = len(categorical_cols)
        self.num_numerical = len(numerical_cols)
    
    def generate_marginal_distributions(self):
        """
        Generate empirical marginal distribution for tabular corruption.
        Used for replacing masked values during TR loss.
        """
        data = np.array(self.data_tabular)
        self.marginal_distributions = np.transpose(data)
    
    def _setup_augmentations(self):
        """Setup image augmentation transforms with Satlas-specific augmentations"""
        if self.augmentation_speedup:
            # Use albumentations for faster augmentation
            # Default transform: just resize and convert to tensor (no augmentation)
            self.default_transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Lambda(name='convert2tensor', image=convert_satellite_to_tensor)
            ])
            
            # Augmentation transform (for training)
            if self.is_train:
                # Build Satlas-specific augmentations (spectral, noise, etc.)
                # These work on tensors, so we'll apply them after albumentations
                self.satellite_augmentations = build_satellite_augmentations(
                    is_train=True,
                    use_geometric=True,  # RandomRotation90
                    use_spectral=True,   # RandomSpectralJitter, RandomBrightness
                    use_noise=True,      # RandomGaussianNoise
                    use_crop=False       # Don't use crop (too aggressive)
                )
                
                # Albumentations for geometric transforms (resize + flips/rotations)
                # These work on numpy arrays before converting to tensor
                self.albumentations_transform = A.Compose([
                    A.Resize(height=self.img_size, width=self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Lambda(name='convert2tensor', image=convert_satellite_to_tensor)
                ])
            else:
                # No augmentation for validation/test
                self.satellite_augmentations = []
                self.albumentations_transform = self.default_transform
        else:
            # Use torchvision transforms (fallback)
            self.default_transform = transforms.Compose([
                transforms.Resize(size=(self.img_size, self.img_size)),
                transforms.Lambda(convert_to_float)
            ])
            if self.is_train:
                self.satellite_augmentations = build_satellite_augmentations(
                    is_train=True,
                    use_geometric=True,
                    use_spectral=True,
                    use_noise=True,
                    use_crop=False
                )
                self.albumentations_transform = self.default_transform
            else:
                self.satellite_augmentations = []
                self.albumentations_transform = self.default_transform
    
    def _load_satellite_image(self, idx: int) -> torch.Tensor:
        """
        Load and combine Sentinel-2 and VIIRS images.
        Returns: (C, H, W) tensor where C = 12 (Sentinel-2) + 3 (VIIRS) = 15 channels
        """
        row = self.df.iloc[idx]
        
        # Load Sentinel-2 (12 bands)
        if self.use_sentinel2 and pd.notna(row.get('sentinel2_tiff_file_name')):
            sentinel2_path = os.path.join(self.composite_dir, row['sentinel2_tiff_file_name'])
            if not os.path.exists(sentinel2_path):
                raise FileNotFoundError(
                    f"❌ ERROR: Sentinel-2 file not found: {sentinel2_path}\n"
                    f"   Row index: {idx}, data_id: {row.get('data_id', 'unknown')}\n"
                    f"   Check if composite_dir is correct: {self.composite_dir}"
                )
            with rasterio.open(sentinel2_path) as src:
                sentinel2 = src.read()  # (12, H, W)
                # Fill NaN values with 0 (standard practice for deep learning)
                sentinel2 = np.nan_to_num(sentinel2, nan=0.0)
                # Normalize to [0, 1]
                sentinel2 = np.clip(sentinel2 / 10000.0, 0, 1).astype('float32')
                
                # # check if any nan values
                # if np.isnan(sentinel2).any():
                #     print(f"❌ ERROR: Sentinel-2 contains nan values: {sentinel2_path}")
                #     print(f"   Row index: {idx}, data_id: {row.get('data_id', 'unknown')}")
                #     print(f"   Check if composite_dir is correct: {self.composite_dir}")
                #     raise ValueError(f"Sentinel-2 contains nan values: {sentinel2_path}")
        # else:
        #     sentinel2 = np.zeros((12, self.img_size, self.img_size), dtype='float32')
        
        # Load VIIRS (1 band, convert to 3 channels)
        if self.use_viirs and pd.notna(row.get('viirs_tiff_file_name')):
            viirs_path = os.path.join(self.composite_dir, row['viirs_tiff_file_name'])
            if not os.path.exists(viirs_path):
                raise FileNotFoundError(
                    f"❌ ERROR: VIIRS file not found: {viirs_path}\n"
                    f"   Row index: {idx}, data_id: {row.get('data_id', 'unknown')}\n"
                    f"   Check if composite_dir is correct: {self.composite_dir}"
                )
            with rasterio.open(viirs_path) as src:
                viirs = src.read(1)  # (H, W) single band
                # Fill NaN values with 0 (standard practice for deep learning)
                viirs = np.nan_to_num(viirs, nan=0.0)
                # Normalize using 99th percentile (473.52) to preserve most data while handling outliers
                # Clips negative values to 0 (radiance should be non-negative)
                viirs = np.clip(viirs, 0, None)  # Remove negative values first
                viirs = np.clip(viirs / 473.52, 0, 1).astype('float32')
                # Convert to 3 channels (repeat)
                viirs = np.stack([viirs, viirs, viirs], axis=0)  # (3, H, W)
                
                # check if any nan values
                # if np.isnan(viirs).any():
                #     print(f"❌ ERROR: VIIRS contains nan values: {viirs_path}")
                #     print(f"   Row index: {idx}, data_id: {row.get('data_id', 'unknown')}")
                #     print(f"   Check if composite_dir is correct: {self.composite_dir}")
                #     raise ValueError(f"VIIRS contains nan values: {viirs_path}")
        # else:
        #     viirs = np.zeros((3, self.img_size, self.img_size), dtype='float32')
        
        # Combine: (15, H, W)
        combined = np.concatenate([sentinel2, viirs], axis=0)
        
        # Resize if needed
        if combined.shape[1] != self.img_size or combined.shape[2] != self.img_size:
            from PIL import Image
            # Convert to PIL and resize
            # For multi-channel, we'll resize each channel
            resized_channels = []
            for c in range(combined.shape[0]):
                channel = Image.fromarray(combined[c])
                channel = channel.resize((self.img_size, self.img_size), Image.BILINEAR)
                resized_channels.append(np.array(channel))
            combined = np.stack(resized_channels, axis=0)
        
        return torch.from_numpy(combined).float()
    
    def corrupt(self, subject: List[float]) -> List[float]:
        """
        Corrupt tabular features by replacing with values from marginal distribution.
        Used for contrastive learning (first view).
        """
        subject = copy.deepcopy(subject)
        subject = np.array(subject)
        
        indices = random.sample(list(range(len(subject))), int(len(subject) * self.c))
        pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=len(indices))
        subject[indices] = self.marginal_distributions[indices, pick_value_positions]
        return subject.tolist()
    
    def mask(self, subject: List[float]) -> Tuple[List[float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Mask tabular features for reconstruction loss (TR).
        Returns: masked_subject, mask, mask_special, mask_random
        """
        subject = copy.deepcopy(subject)
        subject = np.array(subject)
        
        total_mask_rate = self.replace_random_rate + self.replace_special_rate
        if total_mask_rate > 0:
            indices = random.sample(list(range(len(subject))), round(len(subject) * total_mask_rate))
            num_random = int(len(indices) * self.replace_random_rate / total_mask_rate) if total_mask_rate > 0 else 0
            num_special = len(indices) - num_random
            
            # Replace with random samples from marginal distribution
            if num_random > 0:
                pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=num_random)
                subject[indices[:num_random]] = self.marginal_distributions[indices[:num_random], pick_value_positions]
        else:
            indices = []
            num_random = 0
            num_special = 0
        
        # Create masks
        mask = np.zeros_like(subject, dtype=bool)
        mask_random = np.zeros_like(subject, dtype=bool)
        mask_special = np.zeros_like(subject, dtype=bool)
        
        if len(indices) > 0:
            mask[indices] = True
            if num_random > 0:
                mask_random[indices[:num_random]] = True
            if num_special > 0:
                mask_special[indices[num_special:]] = True
        
        return subject.tolist(), mask, mask_special, mask_random
    
    def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
        """One-hot encode tabular features (if needed)"""
        out = []
        for i in range(len(subject)):
            if self.field_lengths_tabular[i] == 1:
                out.append(subject[i].unsqueeze(0))
            else:
                out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
        return torch.cat(out)
    
    def generate_imaging_views(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Generate two views of satellite image for contrastive learning.
        Uses Satlas-specific augmentations (spectral, noise) combined with geometric transforms.
        Returns: (augmented_views, unaugmented_image)
        """
        im = self._load_satellite_image(index)  # (15, H, W) - already resized to img_size
        
        # print(f"im shape after loading: {im.shape}")

        # Convert to numpy for albumentations (expects H, W, C)
        im_np = im.permute(1, 2, 0).numpy()  # (H, W, 15)
        
        # First view: always augmented (if training)
        if self.is_train:
            # Apply albumentations (geometric: resize, flips, rotations)
            im_tensor = self.albumentations_transform(image=im_np)['image']  # (15, H, W) tensor
            # Apply Satlas-specific augmentations (spectral, noise) on tensor
            if self.satellite_augmentations:
                im_tensor = apply_augmentations(im_tensor, self.satellite_augmentations)
            ims = [im_tensor]
        else:
            ims = [self.default_transform(image=im_np)['image']]
        
        # Second view: augmented with probability augmentation_rate
        if random.random() < self.augmentation_rate and self.is_train:
            # Apply albumentations (geometric: resize, flips, rotations)
            im_tensor = self.albumentations_transform(image=im_np)['image']  # (15, H, W) tensor
            # Apply Satlas-specific augmentations (spectral, noise) on tensor
            if self.satellite_augmentations:
                im_tensor = apply_augmentations(im_tensor, self.satellite_augmentations)
            ims.append(im_tensor)
        else:
            ims.append(self.default_transform(image=im_np)['image'])
        
        # Unaugmented image (for validation) - no augmentation
        orig_im = self.default_transform(image=im_np)['image']
        
        return ims, orig_im
    
    def get_input_size(self) -> int:
        """Returns input size for tabular encoder"""
        if self.one_hot_tabular:
            return int(sum(self.field_lengths_tabular))
        else:
            return len(self.field_lengths_tabular)
    
    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
        """
        Returns:
            imaging_views: [augmented_view1, augmented_view2] - List of (15, H, W) tensors
            tabular_views: [original/corrupted, masked, mask, mask_special] - List of tensors
            label: Sample index (for contrastive learning)
            unaugmented_image: Original image without augmentation
            unaugmented_tabular: Original tabular features (for TR loss)
            target: Regression target (construction_cost_per_m2_usd, log-transformed and normalized) or None
            data_id: Unique identifier for this sample (string, NOT used as a feature, only for logging)
        """
        # Generate image views
        imaging_views, unaugmented_image = self.generate_imaging_views(index)
        
        # Generate tabular views
        if self.c > 0:
            # Corrupted view (for contrastive learning)
            tabular_views = [torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float32)]
        else:
            # Original view
            tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float32)]
        
        # Masked view (for TR loss)
        masked_view, mask, mask_special, mask_random = self.mask(self.data_tabular[index])
        tabular_views.append(torch.from_numpy(np.array(masked_view)).float())
        tabular_views.append(torch.from_numpy(mask).bool())
        tabular_views.append(torch.from_numpy(mask_special).bool())
        
        # One-hot encode if needed
        if self.one_hot_tabular:
            tabular_views = [self.one_hot_encode(tv) for tv in tabular_views[:2]] + tabular_views[2:]
        
        # Label (sample index for contrastive learning)
        label = self.labels[index] if isinstance(self.labels, torch.Tensor) else torch.tensor(self.labels[index], dtype=torch.long)
        
        # Unaugmented tabular (for TR loss) - convert to tensor
        unaugmented_tabular = torch.tensor(self.data_tabular[index], dtype=torch.float32)
        
        # Target value (for regression online evaluation)
        # Always return a tensor (use dummy value if None to avoid collate issues)
        if self.targets is not None:
            target = self.targets[index]
        else:
            # Return dummy tensor with same dtype as targets would have
            target = torch.tensor(0.0, dtype=torch.float32)
        
        # Data ID (for logging predictions with correct identifier)
        # Return as-is (string or list) - will be handled separately in evaluation
        if self.data_ids is not None:
            data_id = str(self.data_ids[index])
        else:
            data_id = f"sample_{index}"
        
        # Target original (ground truth in original USD/m² scale, before any transformation)
        if self.targets_original is not None:
            target_original = torch.tensor(float(self.targets_original[index]), dtype=torch.float32)
        else:
            target_original = torch.tensor(0.0, dtype=torch.float32)
        
        return imaging_views, tabular_views, label, unaugmented_image, unaugmented_tabular, target, target_original, data_id
    
    def __len__(self) -> int:
        return len(self.data_tabular)

