"""
EEG数据加载器
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EEGDataset(Dataset):
    """EEG数据集，返回已转换为图像格式的数据"""
    
    def __init__(self, data_path, label_path=None, embedder=None):
        """
        Args:
            data_path: 已经通过DelayEmbedder转换后的数据路径 (N, C, H, W)
            label_path: 标签路径（可选）
            embedder: 不使用，保留接口兼容性
        """
        self.data = np.load(data_path)  # (trials, channels, H, W)
        self.labels = np.load(label_path) if label_path else None
        
        # 归一化到[-1, 1]
        data_min = self.data.min()
        data_max = self.data.max()
        self.data = 2 * (self.data - data_min) / (data_max - data_min) - 1
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        out_dict = {}
        if self.labels is not None:
            out_dict["y"] = np.array(self.labels[idx], dtype=np.int64)
        return x, out_dict


def load_eeg_data(
    data_dir,
    batch_size,
    class_cond=False,
    deterministic=False,
):
    """
    创建EEG数据加载器
    
    Args:
        data_dir: 数据目录，包含train_data_embedded.npy和train_label.npy
        batch_size: 批次大小
        class_cond: 是否使用类别条件
        deterministic: 是否确定性顺序
    """
    import os
    
    data_path = os.path.join(data_dir, "train_data_embedded.npy")
    label_path = os.path.join(data_dir, "train_label.npy") if class_cond else None
    
    dataset = EEGDataset(data_path, label_path)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=not deterministic, 
        num_workers=1, 
        drop_last=True
    )
    
    while True:
        yield from loader

