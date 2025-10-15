#!/bin/bash
# EEG数据处理和训练完整流程

echo "=== Step 1: 合并原始数据 ==="
python scripts/prepare_eeg_data.py

echo ""
echo "=== Step 2: 使用DelayEmbedder转换数据 ==="
python scripts/embed_eeg_data.py

echo ""
echo "=== Step 3: 开始训练扩散模型 ==="
# 指定使用GPU 0和1
export CUDA_VISIBLE_DEVICES=0,1
# 设置日志和模型保存目录
export OPENAI_LOGDIR=/home/xyr/workspace/testeeg/logs
mkdir -p $OPENAI_LOGDIR
mpirun -n 2 python scripts/eeg_train.py \
    --data_dir datasets/eegdata/bci2a \
    --batch_size 16 \
    --lr 1e-4 \
    --lr_anneal_steps 100000 \
    --image_size 64 \
    --in_channels 22 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --save_interval 5000 \
    --log_interval 100

echo ""
echo "=== 训练完成！ ==="

