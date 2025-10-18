#!/bin/bash
# EEG 扩散模型批量评估脚本（后台运行版本）
# 使用方法: bash run_evaluation_background.sh

# 设置 GPU（根据需要修改）
export CUDA_VISIBLE_DEVICES=1

# 数据和模型路径
TEST_DATA_PATH="eeg_adapt/datasets/eegdata/bci2a/resub8/test_data_embedded.npy"
MODEL_DIR="eeg_adapt/logs"
OUTPUT_FILE="evaluation_results.json"
LOG_FILE="eval_big.log"

# 评估参数
BATCH_SIZE=64
MODEL_TYPE="all"  # 可选: ema, model, all
TIMESTEP_RESPACING=""  # 留空使用全部1000步，设置为"100"可以加速评估

echo "========================================"
echo "准备在后台运行 EEG 扩散模型评估"
echo "========================================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "测试数据: $TEST_DATA_PATH"
echo "模型目录: $MODEL_DIR"
echo "模型类型: $MODEL_TYPE"
echo "日志文件: $LOG_FILE"
echo "========================================"

# 构建命令（使用默认参数，因为已经在代码中设置）
CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nohup python eeg_adapt/process/batch_evaluate_models.py \
  --test_data_path $TEST_DATA_PATH \
  --model_dir $MODEL_DIR \
  --output_file $OUTPUT_FILE \
  --batch_size $BATCH_SIZE \
  --model_type $MODEL_TYPE"

# 添加可选参数
if [ -n "$TIMESTEP_RESPACING" ]; then
    CMD="$CMD --timestep_respacing $TIMESTEP_RESPACING"
fi

CMD="$CMD > $LOG_FILE 2>&1 &"

# 运行评估
echo "执行命令:"
echo "$CMD"
echo ""

eval $CMD

# 获取进程 ID
sleep 1
PID=$(pgrep -f "batch_evaluate_models.py")

if [ -n "$PID" ]; then
    echo "评估已在后台启动！"
    echo "进程 ID: $PID"
    echo ""
    echo "查看实时日志: tail -f $LOG_FILE"
    echo "查看进程状态: ps aux | grep $PID"
    echo "停止评估: kill $PID"
    echo ""
    echo "结果将保存至: $MODEL_DIR/$OUTPUT_FILE"
else
    echo "警告: 无法找到后台进程"
    echo "请检查日志: cat $LOG_FILE"
fi

echo "========================================"

