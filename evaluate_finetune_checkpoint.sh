#!/bin/bash
# Evaluation command for fine-tuning checkpoint

cd /hdd/hiep/CODE/Construction_Cost_Prediction/src/models/TIP

python evaluate_construction_cost.py \
    --checkpoint /hdd/hiep/CODE/Construction_Cost_Prediction/work_dir/runs/finetune/tip_finetune_construction_cost_1221_1305/fold_1/checkpoint_last_epoch_17.ckpt \
    --val_csv /hdd/hiep/CODE/Construction_Cost_Prediction/data/annotation/val/val_clean.csv \
    --test_csv /hdd/hiep/CODE/Construction_Cost_Prediction/data/annotation/test/test_clean.csv \
    --composite_dir_trainval /hdd/hiep/CODE/Construction_Cost_Prediction/data/trainval_composite \
    --composite_dir_test /hdd/hiep/CODE/Construction_Cost_Prediction/data/test_composite \
    --field_lengths /hdd/hiep/CODE/Construction_Cost_Prediction/data/annotation/field_lengths.pt \
    --val_metadata /hdd/hiep/CODE/Construction_Cost_Prediction/data/annotation/val/val_clean_metadata.pkl \
    --test_metadata /hdd/hiep/CODE/Construction_Cost_Prediction/data/annotation/test/test_clean_metadata.pkl \
    --batch_size 32 \
    --num_workers 2 \
    --device cuda

