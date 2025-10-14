"""
使用DelayEmbedder将EEG时间序列数据转换为图像格式
"""

import numpy as np
import torch
import os


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


def embed_eeg_data(data_dir="../datasets/eegdata/bci2a", embedding_size=64, delay=15):
    """
    将train_data.npy通过DelayEmbedder转换为图像格式
    
    Args:
        data_dir: 数据目录
        embedding_size: embedding维度，默认64
        delay: delay参数，默认15
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data_path = os.path.join(data_dir, "train_data.npy")
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
            print(f"处理进度: {i+len(batch)}/{len(data)}")
    
    # 合并结果
    embedded_data = np.concatenate(embedded_data, axis=0)
    
    print(f"转换后数据shape: {embedded_data.shape}")
    
    # 保存
    output_path = os.path.join(data_dir, "train_data_embedded.npy")
    np.save(output_path, embedded_data)
    
    print(f"保存到: {output_path}")


if __name__ == "__main__":
    embed_eeg_data()

