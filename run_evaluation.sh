#!/bin/bash
# Evaluation script for Construction Cost Prediction

# Get project root directory
PROJECT_ROOT="/hdd/hiep/CODE/Construction_Cost_Prediction"

# Set paths (absolute)
CHECKPOINT="${PROJECT_ROOT}/work_dir/runs/multimodal/tip_pretrain_construction_cost_1219_1722/checkpoint_best_rmsle_84_0.2719.ckpt"
VAL_CSV="${PROJECT_ROOT}/data/annotation/val/val_clean.csv"
TEST_CSV="${PROJECT_ROOT}/data/annotation/test/test_clean.csv"
COMPOSITE_DIR_TRAINVAL="${PROJECT_ROOT}/data/trainval_composite"
COMPOSITE_DIR_TEST="${PROJECT_ROOT}/data/test_composite"
FIELD_LENGTHS="${PROJECT_ROOT}/data/annotation/field_lengths.pt"
VAL_METADATA="${PROJECT_ROOT}/data/annotation/val/val_clean_metadata.pkl"
TEST_METADATA="${PROJECT_ROOT}/data/annotation/test/test_clean_metadata.pkl"
OUTPUT_DIR="${PROJECT_ROOT}/work_dir/evaluation"

# Run evaluation
cd "${PROJECT_ROOT}/src/models/TIP"
python evaluate_construction_cost.py \
    --checkpoint "$CHECKPOINT" \
    --val_csv "$VAL_CSV" \
    --test_csv "$TEST_CSV" \
    --composite_dir_trainval "$COMPOSITE_DIR_TRAINVAL" \
    --composite_dir_test "$COMPOSITE_DIR_TEST" \
    --field_lengths "$FIELD_LENGTHS" \
    --val_metadata "$VAL_METADATA" \
    --test_metadata "$TEST_METADATA" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda

