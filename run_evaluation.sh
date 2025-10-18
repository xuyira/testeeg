#!/bin/bash
# EEG 扩散模型批量评估脚本
# 使用方法: bash run_evaluation.sh

# 设置 GPU（根据需要修改）
export CUDA_VISIBLE_DEVICES=1

# 数据和模型路径
TEST_DATA_PATH="eeg_adapt/datasets/eegdata/bci2a/resub8/test_data_embedded.npy"
MODEL_DIR="eeg_adapt/logs"
OUTPUT_FILE="evaluation_results.json"

# 评估参数
NUM_EVAL_SAMPLES=""  # 留空表示使用全部测试数据
BATCH_SIZE=64
MODEL_TYPE="all"  # 可选: ema, model, all
TIMESTEP_RESPACING=""  # 留空使用全部1000步，设置为"100"可以加速

# 模型参数（必须与训练时一致）
IMAGE_SIZE=64
IN_CHANNELS=22
NUM_CHANNELS=128
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
DROPOUT=0.1
NOISE_SCHEDULE="cosine"
NUM_HEAD_CHANNELS=64
DIFFUSION_STEPS=1000

echo "========================================"
echo "开始评估 EEG 扩散模型"
echo "========================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "测试数据: $TEST_DATA_PATH"
echo "模型目录: $MODEL_DIR"
echo "模型类型: $MODEL_TYPE"
echo "批次大小: $BATCH_SIZE"
if [ -n "$TIMESTEP_RESPACING" ]; then
    echo "采样步数: $TIMESTEP_RESPACING (加速模式)"
else
    echo "采样步数: $DIFFUSION_STEPS (完整模式)"
fi
echo "========================================"

# 构建命令
CMD="python eeg_adapt/process/batch_evaluate_models.py \
  --test_data_path $TEST_DATA_PATH \
  --model_dir $MODEL_DIR \
  --output_file $OUTPUT_FILE \
  --batch_size $BATCH_SIZE \
  --model_type $MODEL_TYPE \
  --image_size $IMAGE_SIZE \
  --in_channels $IN_CHANNELS \
  --num_channels $NUM_CHANNELS \
  --num_res_blocks $NUM_RES_BLOCKS \
  --learn_sigma \
  --use_fp16 \
  --attention_resolutions $ATTENTION_RESOLUTIONS \
  --dropout $DROPOUT \
  --noise_schedule $NOISE_SCHEDULE \
  --num_head_channels $NUM_HEAD_CHANNELS \
  --resblock_updown \
  --use_new_attention_order \
  --use_scale_shift_norm \
  --diffusion_steps $DIFFUSION_STEPS"

# 添加可选参数
if [ -n "$NUM_EVAL_SAMPLES" ]; then
    CMD="$CMD --num_eval_samples $NUM_EVAL_SAMPLES"
fi

if [ -n "$TIMESTEP_RESPACING" ]; then
    CMD="$CMD --timestep_respacing $TIMESTEP_RESPACING"
fi

# 运行评估
echo "执行命令:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "========================================"
echo "评估完成！"
echo "结果已保存至: $MODEL_DIR/$OUTPUT_FILE"
echo "========================================"

