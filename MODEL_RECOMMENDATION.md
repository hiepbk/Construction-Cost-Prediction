# Model Recommendations & Current Architecture

## Current Status: TIP (Tabular-Image Pre-training) ✅

We are using **TIP (Tabular-Image Pre-training)**, a self-supervised pretraining framework for multimodal learning, adapted for construction cost regression.

### TIP Pretraining Pipeline (Two-Stage Training)

**Stage 1: Self-Supervised Pretraining**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  TABULAR MODALITY              │  SATELLITE MODALITY                        │
│  ────────────────────────────  │  ────────────────────────────             │
│  • Categorical: (B, 191)        │  • Sentinel-2: (B, 9, H, W)                │
│  • Continuous: (B, 3)          │  • VIIRS: (B, 3, H, W)                     │
│  • Total: 20 features          │  • Combined: (B, 15, H, W)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼──────────┐      ┌────────────▼────────────┐
        │  TABULAR ENCODER     │      │  IMAGE ENCODER          │
        │  TabularTransformer  │      │  SatelliteImageEncoder  │
        └──────────────────────┘      └─────────────────────────┘
                    │                               │
        ┌───────────▼──────────┐      ┌────────────▼────────────┐
        │ - Embedding(191, 512)│      │ Sentinel-2:            │
        │ - Linear(1→512)     │      │ - Swin-v2-Base         │
        │ - Embedding(20, 512) │      │ - FPN (128D)           │
        │ Output: (B, 20, 512)│      │ - Proj: 128→512         │
        └──────────────────────┘      │                        │
                    │                 │ VIIRS:                 │
        ┌───────────▼──────────┐      │ - CNN: 3→64→256        │
        │ 4 x TransformerBlock│      │ - Proj: 256→512         │
        │ - Masked Attention   │      │                        │
        │ - MLP: 512→2048→512  │      │ Combined: (B, 1024)    │
        │ Output: (B, 20, 512)│      └─────────────────────────┘
        └──────────────────────┘                    │
                    │                 ┌────────────▼────────────┐
        ┌───────────▼──────────┐      │ Projection Head         │
        │ SimCLRProjectionHead │      │ 1024→2048→128           │
        │ 512→512→128          │      └─────────────────────────┘
        └──────────────────────┘                    │
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   MULTIMODAL ENCODER          │
                    │   MultimodalTransformer       │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ - Image Proj: 1024→512        │
                    │ - Tabular: Identity (512)     │
                    │ - 4 x TransformerBlock        │
                    │   • Self-Attention            │
                    │   • Cross-Attention           │
                    │   • MLP: 512→2048→512         │
                    │ Output: (B, 20, 512)          │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   TABULAR PREDICTOR            │
                    │   (for TR loss)               │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ - Cat: Linear(512→191)        │
                    │ - Con: Linear(512→1)          │
                    │ Output: Reconstructed features│
                    └───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   THREE LOSSES                 │
                    │   • ITC: Contrastive (128D)  │
                    │   • ITM: Matching (512D)     │
                    │   • TR: Reconstruction        │
                    └───────────────────────────────┘
```

**Stage 2: Supervised Fine-Tuning**

- Load pretrained encoders (frozen or fine-tuned)
- Replace projection heads with regression head
- Train on log-transformed targets with Huber/MAE loss

### TIP Feature Shape Progression

| Stage | Module | Input Shape | Output Shape | Description |
|-------|--------|-------------|--------------|-------------|
| **Input** | - | - | - | Raw input data |
| | Tabular Categorical | - | `(B, 191)` | 191 categorical features (label encoded) |
| | Tabular Continuous | - | `(B, 3)` | 3 continuous features (normalized) |
| | Sentinel-2 | - | `(B, 9, H, W)` | 9 selected bands |
| | VIIRS | - | `(B, 3, H, W)` | 3 channels |
| **Tabular Encoding** | Embedding + Linear | `(B, 191)` + `(B, 3)` | `(B, 20, 512)` | Feature tokenization |
| | Transformer Blocks | `(B, 20, 512)` | `(B, 20, 512)` | 4 layers with masked attention |
| | Projection Head | `(B, 20, 512)` | `(B, 128)` | SimCLR projection |
| **Image Encoding** | Sentinel-2 Backbone | `(B, 9, H, W)` | `(B, 128)` | Swin-v2-Base + FPN |
| | VIIRS CNN | `(B, 3, H, W)` | `(B, 256)` | Lightweight CNN |
| | Projections | `(B, 128)` + `(B, 256)` | `(B, 1024)` | Concatenated and projected |
| | Projection Head | `(B, 1024)` | `(B, 128)` | SimCLR projection |
| **Multimodal Fusion** | Image Proj | `(B, 1024)` | `(B, 512)` | Linear projection |
| | Multimodal Transformer | `(B, 512)` + `(B, 512)` | `(B, 20, 512)` | 4 layers with cross-attention |
| **Pretraining Output** | ITC Loss | `(B, 128)` + `(B, 128)` | Scalar | Contrastive loss |
| | ITM Loss | `(B, 512)` | Scalar | Matching loss |
| | TR Loss | `(B, 512)` | Scalar | Reconstruction loss |
| **Fine-Tuning** | Regression Head | `(B, 512)` | `(B, 1)` | MLP for regression |

**Note:** `B` = batch size, `H` = image height (typically 224), `W` = image width (typically 224)

## TIP (Tabular-Image Pre-training) Integration ✅

**Status:** Fully integrated and adapted for construction cost regression

**What:** Official TIP implementation (ECCV 2024) - Self-supervised pretraining framework for tabular + image multimodal learning.

**Location:** `src/models/TIP/` (cloned from https://github.com/siyi-wind/TIP)

### TIP Architecture Overview (Actual Implementation)

**Core Components:**

1. **Tabular Encoder** (`TabularTransformerEncoder`):
   - **Input Processing:**
     - Categorical features: `Embedding(191, 512)` - 191 categorical features with 512-dim embeddings
     - Continuous features: `Linear(1, 512)` - Projects each continuous feature to 512D
     - Column embeddings: `Embedding(20, 512)` - Positional embeddings for 20 total features
   - **Transformer Blocks:** 4 layers
     - Self-attention with **masked attention** (handles missing data)
     - QKV projection: `Linear(512→1536)`, output: `Linear(512→512)`
     - MLP: `512→2048→512` with GELU activation
     - LayerNorm and Dropout (0.1) for stability
   - **Projection Head:** `SimCLRProjectionHead`
     - `512→512→128` with BatchNorm and ReLU
   - **Output:** 128D contrastive embedding

2. **Image Encoder** (`SatelliteImageEncoder`):
   - **Sentinel-2 Backbone:** SatlasPretrain Swin-v2-Base
     - Input: 9 channels (selected from 12 Sentinel-2 bands)
     - Architecture: SwinTransformer with 4 stages
       - Stage 1: 128D (2 blocks)
       - Stage 2: 256D (2 blocks)
       - Stage 3: 512D (18 blocks)
       - Stage 4: 1024D (2 blocks)
     - **FPN (Feature Pyramid Network):** Multi-scale feature extraction
       - 4 levels: 128, 256, 512, 1024 → all projected to 128D
       - Upsampling and fusion layers
     - Output: 128D feature maps
   - **VIIRS Encoder:** Lightweight CNN
     - `Conv2d(3, 64)` → BatchNorm → ReLU → AdaptiveAvgPool → `Linear(64→256)`
     - Output: 256D
   - **Projections:**
     - Sentinel-2: `Linear(128→512)`
     - VIIRS: `Linear(256→512)`
   - **Combined:** Concatenated to 1024D (512+512)
   - **Projection Head:** `SimCLRProjectionHead`
     - `1024→2048→128` with BatchNorm and ReLU
   - **Output:** 128D contrastive embedding

3. **Multimodal Encoder** (`MultimodalTransformerEncoder`):
   - **Input Projections:**
     - Image: `Linear(1024→512)` + LayerNorm
     - Tabular: Identity (already 512D)
   - **Transformer Blocks:** 4 layers
     - **Self-Attention:** Standard multi-head attention (512→1536→512)
     - **Cross-Attention:** Cross-modal attention between image and tabular
       - KV projection: `Linear(512→1024)` (from tabular)
       - Q projection: `Linear(512→512)` (from image)
       - Output: `Linear(512→512)`
     - MLP: `512→2048→512` with GELU
     - LayerNorm after each sub-layer
   - **Output:** 512D fused multimodal representation

4. **Tabular Predictor** (`TabularPredictor`):
   - **Categorical Reconstruction:** `Linear(512→191)` - Reconstructs 191 categorical features
   - **Continuous Reconstruction:** `Linear(512→1)` - Reconstructs continuous features
   - Used for **TR (Tabular Reconstruction)** loss during pretraining

### Three Loss Functions (from `TipModel3Loss.py`):

1. **ITC (Image-Tabular Contrastive)** - `CLIPLoss`:
   - CLIP-style contrastive learning
   - Maximizes similarity between matching image-tabular pairs
   - Minimizes similarity between non-matching pairs
   - Temperature-scaled cosine similarity

2. **ITM (Image-Tabular Matching)**:
   - Binary classification task
   - Predicts whether image-tabular pair matches
   - Uses hard negative mining (weighted sampling)

3. **TR (Tabular Reconstruction)** - `ReconstructionLoss`:
   - Reconstructs masked tabular features
   - Categorical: Cross-entropy loss
   - Continuous: MSE loss
   - Only computed on masked features

### Key Files:

- **Models:**
  - `models/Tips/TipModel3Loss.py` - Main TIP model with 3 losses (ITC, ITM, TR)
  - `models/Tip_utils/Tip_pretraining.py` - Base pretraining class
  - `models/Tip_utils/Tip_downstream.py` - Fine-tuning backbone (with regression head)
  - `models/Tip_utils/Transformer.py` - Tabular & Multimodal transformers
  - `models/Tip_utils/SatelliteImageEncoder.py` - **NEW**: SatlasPretrain integration
  - `models/Evaluator_ConstructionCost.py` - **NEW**: Regression fine-tuning module

- **Losses:**
  - `utils/clip_loss.py` - CLIP contrastive loss (ITC)
  - `utils/reconstruct_loss.py` - Tabular reconstruction loss (TR)

- **Datasets:**
  - `datasets/ConstructionCostTIPDataset.py` - **NEW**: Custom dataset for construction cost data
  - `datasets/ContrastiveReconstructImagingAndTabularDataset.py` - Original TIP dataset

- **Training:**
  - `run.py` - Main entry point (uses Hydra config)
  - `trainers/pretrain.py` - Pretraining script (adapted for construction cost)
  - `trainers/evaluate.py` - Fine-tuning/evaluation script (adapted for regression)

- **Preprocessing:**
  - `data/preprocess_construction_cost.py` - **NEW**: Tabular data preprocessing
  - `utils/generate_field_lengths.py` - **NEW**: Generates field_lengths.pt

- **Evaluation:**
  - `utils/ssl_online_regression.py` - **NEW**: Online regression evaluation during pretraining

- **Configuration:**
  - `configs/config_construction_cost_pretrain.yaml` - **NEW**: Pretraining config
  - `configs/config_construction_cost_finetune.yaml` - **NEW**: Fine-tuning config

### Configuration (from `configs/config_construction_cost_pretrain.yaml`):

- **Architecture:**
  - `tabular_embedding_dim`: 512
  - `multimodal_embedding_dim`: 512
  - `tabular_transformer_num_layers`: 4
  - `multimodal_transformer_num_layers`: 4
  - `projection_dim`: 128
  - `num_cat`: 191 (categorical features)
  - `num_con`: 3 (continuous features)
  - `image_encoder_output_dim`: 1024 (Sentinel-2 + VIIRS combined)

- **Training:**
  - `temperature`: 0.1 (for contrastive loss)
  - `corruption_rate`: 0.3 (tabular masking rate)
  - `replace_random_rate`: 0.15 (random replacement)
  - `replace_special_rate`: 0.50 (special token replacement)
  - `batch_size`: 32 (configurable)
  - `lr`: 3e-4
  - `target_log_transform`: True (log-transform construction cost)
  - `online_mlp`: True (online regression evaluation during pretraining)

### Why TIP Helps This Regression Task:

1. **Cross-Modal Alignment**: ITC loss aligns tabular and satellite embeddings
2. **Robustness to Missing Data**: TR loss + masking handles incomplete features
3. **Better Representations**: Multimodal encoder learns joint representations
4. **Transfer Learning**: Pretrained encoders provide better initialization

### Implementation Status:

✅ **Completed:**

1. **Dataset Adaptation** (`ConstructionCostTIPDataset`):
   - Loads Sentinel-2 (12 bands) and VIIRS (1 band) images
   - Concatenates to 15-channel input, selects 9 bands for Sentinel-2
   - Processes tabular data: 191 categorical + 3 continuous features
   - Supports masking for TR loss
   - Handles log-transformed targets
   - Uses preprocessing metadata for consistent encoding

2. **Model Adaptation**:
   - ✅ `SatelliteImageEncoder`: Integrates SatlasPretrain Swin-v2-Base + FPN
   - ✅ `TabularTransformerEncoder`: Uses masked attention for missing data
   - ✅ `MultimodalTransformerEncoder`: Cross-attention fusion
   - ✅ `TabularPredictor`: Reconstruction heads for TR loss

3. **Training Pipeline**:
   - ✅ Stage 1: TIP pretraining (`trainers/pretrain.py`)
     - ITC (Image-Tabular Contrastive) loss
     - ITM (Image-Tabular Matching) loss
     - TR (Tabular Reconstruction) loss
     - Online regression evaluation during pretraining
   - ✅ Stage 2: Fine-tuning (`trainers/evaluate.py`)
     - Regression head with Huber/MAE loss
     - RMSLE as primary metric
     - Supports freezing/unfreezing image backbone

4. **Preprocessing** (`preprocess_construction_cost.py`):
   - Generates clean CSVs with consistent encoding
   - Saves metadata (encoders, normalization stats)
   - Union cardinality strategy for categorical features
   - Handles train/val/test splits

### Key Features:

- **Masked Attention**: TabularTransformerEncoder uses masked attention to handle missing/incomplete tabular features
- **Multi-Scale Features**: FPN extracts features at multiple scales from Sentinel-2
- **Cross-Modal Alignment**: ITC loss aligns satellite and tabular embeddings in shared space
- **Robustness**: TR loss + masking improves model robustness to missing data
- **Online Evaluation**: Regression metrics (MAE, RMSE, RMSLE, R²) computed during pretraining

### References:
- **GitHub:** https://github.com/siyi-wind/TIP
- **Paper:** https://arxiv.org/abs/2407.07582
- **ECCV 2024:** Tabular-Image Pre-training for Multimodal Classification with Incomplete Data

## References

- **TIP (Tabular-Image Pre-training):** [GitHub](https://github.com/siyi-wind/TIP) | [Paper](https://arxiv.org/abs/2407.07582) | ECCV 2024
- **SatlasPretrain:** [GitHub](https://github.com/allenai/satlaspretrain_models) | [Paper](https://arxiv.org/abs/2306.15464)
