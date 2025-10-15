#!/bin/bash
# 快速评估脚本 - 用于测试和调试
# 使用较少的样本以加快评估速度

echo "========================================"
echo "快速评估模式 (用于测试)"
echo "========================================"
echo ""
echo "使用 50 个样本进行快速评估"
echo ""

python scripts/batch_evaluate_models.py \
    --test_data_path datasets/eegdata/bci2a/test_data_embedded.npy \
    --model_dir logs \
    --output_file quick_evaluation_results.json \
    --num_eval_samples 50 \
    --batch_size 16 \
    --image_size 64 \
    --in_channels 22 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --attention_resolutions "16,8"

echo ""
echo "快速评估完成！"

