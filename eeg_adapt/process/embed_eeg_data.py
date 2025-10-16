"""
使用DelayEmbedder将EEG时间序列数据转换为图像格式
"""

import numpy as np
import torch
import os
import argparse


class DelayEmbedder:
    """Delay embedding transformation"""
    
    def __init__(self, device, seq_len, delay, embedding):
        self.device = device
        self.seq_len = seq_len
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None
    
    def pad_to_square(self, x, mask=0):
        """Pads the input tensor x to make it square along the last two dimensions."""
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (0, max_side - rows, 0, max_side - cols)
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded
    
    def ts_to_img(self, signal, pad=True, mask=0):
        """
        将时间序列转换为图像
        Args:
            signal: (batch, length, features) - EEG数据
            pad: 是否填充到正方形
        Returns:
            x_image: (batch, features, H, W)
        """
        batch, length, features = signal.shape
        if self.seq_len != length:
            self.seq_len = length
        
        x_image = torch.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1
        
        # 处理剩余部分
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1
        
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]
        
        if pad:
            x_image = self.pad_to_square(x_image, mask)
        
        return x_image


def process_single_dataset(data_path, output_path, embedding_size=64, delay=15):
    """
    处理单个数据集文件
    
    Args:
        data_path: 输入数据路径
        output_path: 输出数据路径
        embedding_size: embedding维度，默认64
        delay: delay参数，默认15
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data = np.load(data_path)  # (trials, channels, timepoints)
    
    print(f"原始数据shape: {data.shape}")
    
    # 转换数据格式：(trials, channels, timepoints) -> (trials, timepoints, channels)
    data = np.transpose(data, (0, 2, 1))
    
    # 初始化embedder
    seq_len = data.shape[1]  # timepoints
    embedder = DelayEmbedder(device, seq_len, delay, embedding_size)
    
    # 批量处理数据
    batch_size = 100
    embedded_data = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)
        embedded_batch = embedder.ts_to_img(batch_tensor, pad=True, mask=0)
        embedded_data.append(embedded_batch.cpu().numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  处理进度: {i+len(batch)}/{len(data)}")
    
    # 合并结果
    embedded_data = np.concatenate(embedded_data, axis=0)
    
    print(f"转换后数据shape: {embedded_data.shape}")
    
    # 保存
    np.save(output_path, embedded_data)
    
    print(f"保存到: {output_path}")
    
    return embedded_data.shape


def embed_eeg_data(data_dir="../datasets/eegdata/bci2a", subject_dir=None, embedding_size=64, delay=15):
    """
    将train_data.npy和test_data.npy通过DelayEmbedder转换为图像格式
    
    Args:
        data_dir: 数据基础目录
        subject_dir: 受试者子目录名（例如 "sub1357"），如果为 None 则直接使用 data_dir
        embedding_size: embedding维度，默认64
        delay: delay参数，默认15
    """
    # 确定实际数据目录
    if subject_dir:
        actual_dir = os.path.join(data_dir, subject_dir)
    else:
        actual_dir = data_dir
    
    print("="*80)
    print("EEG 数据嵌入转换")
    print("="*80)
    print(f"数据目录: {actual_dir}\n")
    
    # 处理训练集
    print("【处理训练集】")
    train_data_path = os.path.join(actual_dir, "train_data.npy")
    train_output_path = os.path.join(actual_dir, "train_data_embedded.npy")
    
    if os.path.exists(train_data_path):
        train_shape = process_single_dataset(train_data_path, train_output_path, embedding_size, delay)
    else:
        print(f"⚠️  警告: 找不到训练数据文件 {train_data_path}")
        train_shape = None
    
    # 处理测试集
    print("\n【处理测试集】")
    test_data_path = os.path.join(actual_dir, "test_data.npy")
    test_output_path = os.path.join(actual_dir, "test_data_embedded.npy")
    
    if os.path.exists(test_data_path):
        test_shape = process_single_dataset(test_data_path, test_output_path, embedding_size, delay)
    else:
        print(f"⚠️  警告: 找不到测试数据文件 {test_data_path}")
        test_shape = None
    
    # 打印汇总
    print("\n" + "="*80)
    print("数据嵌入转换完成！")
    print("="*80)
    if train_shape:
        print(f"训练集: {train_shape}")
    if test_shape:
        print(f"测试集: {test_shape}")
    print(f"保存目录: {actual_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 EEG 时间序列数据转换为图像格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理默认目录的数据
  python embed_eeg_data.py
  
  # 处理指定受试者目录的数据 (例如 sub1357)
  python embed_eeg_data.py --subject_dir sub1357
  
  # 指定基础目录和受试者目录
  python embed_eeg_data.py --data_dir datasets/eegdata/bci2a --subject_dir sub34
  
  # 自定义嵌入参数
  python embed_eeg_data.py --subject_dir sub1357 --embedding_size 64 --delay 15
        """
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default='datasets/eegdata/bci2a',
        help='数据基础目录（默认: datasets/eegdata/bci2a）'
    )
    
    parser.add_argument(
        '--subject_dir', '-s',
        type=str,
        default=None,
        help='受试者子目录名（例如: sub1357），如果不指定则直接使用 data_dir'
    )
    
    parser.add_argument(
        '--embedding_size', '-e',
        type=int,
        default=64,
        help='嵌入维度（默认: 64）'
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=15,
        help='时间延迟步长（默认: 15）'
    )
    
    args = parser.parse_args()
    
    # 执行嵌入转换
    embed_eeg_data(
        data_dir=args.data_dir,
        subject_dir=args.subject_dir,
        embedding_size=args.embedding_size,
        delay=args.delay
    )

