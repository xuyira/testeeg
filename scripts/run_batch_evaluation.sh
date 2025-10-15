#!/bin/bash
# EEG 扩散模型批量评估脚本
# 使用方法: bash scripts/run_batch_evaluation.sh

# 设置参数
TEST_DATA_PATH="datasets/eegdata/bci2a/test_data_embedded.npy"
MODEL_DIR="logs"
OUTPUT_FILE="evaluation_results.json"

# 评估参数
# NUM_EVAL_SAMPLES 为空表示使用全部测试数据，也可以设置具体数量如 100
NUM_EVAL_SAMPLES=""   # 留空使用全部测试数据，或设置具体数量如 "100"
BATCH_SIZE=16         # 批次大小

# 模型参数（需要与训练时一致）
IMAGE_SIZE=64
IN_CHANNELS=22
NUM_CHANNELS=128
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="16,8"

echo "========================================"
echo "EEG 扩散模型批量评估"
echo "========================================"
echo ""
echo "配置信息："
echo "  测试数据: $TEST_DATA_PATH"
echo "  模型目录: $MODEL_DIR"
echo "  输出文件: $OUTPUT_FILE"
if [ -z "$NUM_EVAL_SAMPLES" ]; then
    echo "  评估样本数: 全部测试数据"
else
    echo "  评估样本数: $NUM_EVAL_SAMPLES"
fi
echo "  批次大小: $BATCH_SIZE"
echo ""
echo "开始评估..."
echo ""

# 构建命令
CMD="python scripts/batch_evaluate_models.py \
    --test_data_path \"$TEST_DATA_PATH\" \
    --model_dir \"$MODEL_DIR\" \
    --output_file \"$OUTPUT_FILE\" \
    --batch_size $BATCH_SIZE \
    --image_size $IMAGE_SIZE \
    --in_channels $IN_CHANNELS \
    --num_channels $NUM_CHANNELS \
    --num_res_blocks $NUM_RES_BLOCKS \
    --attention_resolutions \"$ATTENTION_RESOLUTIONS\""

# 如果指定了样本数，添加参数
if [ ! -z "$NUM_EVAL_SAMPLES" ]; then
    CMD="$CMD --num_eval_samples $NUM_EVAL_SAMPLES"
fi

# 运行评估脚本
eval $CMD

echo ""
echo "评估完成！"
echo "结果已保存至: $MODEL_DIR/$OUTPUT_FILE"

