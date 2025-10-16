"""
批量评估 EEG 扩散模型
对指定目录下的所有 .pt 模型进行评估，使用边缘分布差异指标
"""


import os
import numpy as np
import torch
import torch.nn as nn

class Loss:
    """评估损失基类"""
    def __init__(self, name='loss'):
        self.name = name

    def compute(self, x_fake):
        raise NotImplementedError()

    def __call__(self, x_fake):
        loss = self.compute(x_fake)
        if isinstance(loss, tuple):
            return sum([lo.mean() for lo in loss])
        else:
            return loss.mean()


def histogram_torch(x, n_bins, density=True):
    """计算直方图的 PyTorch 实现"""
    x_min = x.min()
    x_max = x.max()
    bins = torch.linspace(x_min, x_max, n_bins + 1)

    # 计算每个 bin 的计数
    hist = torch.zeros(n_bins)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (x >= bins[i]) & (x <= bins[i + 1])
        else:
            mask = (x >= bins[i]) & (x < bins[i + 1])
        hist[i] = mask.sum()

    if density:
        bin_width = (x_max - x_min) / n_bins
        hist = hist / (x.numel() * bin_width)

    return hist, bins


class HistoLoss(Loss):
    """边缘分布直方图损失"""
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            x_i = x_real[..., i].reshape(-1, 1)
            d, b = histogram_torch(x_i, n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


def compute_marginal_loss(x_fake, x_real):
    """
    计算边缘分布损失

    参数:
        x_fake: 生成数据
        x_real: 真实数据

    返回:
        marginal_loss: 边缘分布损失值
    """
    print("\n计算边缘分布损失...")
    marginal_loss = HistoLoss(x_real=x_real, n_bins=50, name='marginal_loss')(x_fake).item()
    print(f"边缘分布损失: {marginal_loss:.6f}")
    return marginal_loss


def load_test_data(data_path):
    """
    加载测试数据

    参数:
        data_path: 数据文件路径

    返回:
        test_data: numpy 数组, shape=(n_trials, 22, 64, 64)
    """
    print(f"从 {data_path} 加载测试数据...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"测试数据文件不存在: {data_path}")

    test_data = np.load(data_path)
    print(f"测试数据形状: {test_data.shape}")
    return test_data


if __name__ == "__main__":
    # 加载数据 - 原始形状应该是 (batch, 1000, 22)
    # 如果不是，请根据实际情况调整 transpose 参数
    print("加载数据...")
    x_fake_np  = np.load('eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_data.npy')
    x_real1_np = np.load('eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy')
    x_real2_np = np.load('eeg_adapt/datasets/eegdata/bci2a/resub8/test_data.npy')
    
    print(f"x_fake 形状: {x_fake_np.shape}")
    print(f"x_real1 形状: {x_real1_np.shape}")
    print(f"x_real2 形状: {x_real2_np.shape}")
    
    # 确保数据格式为 (batch, seq_len, channels)
    # 如果原始数据是 (batch, channels, seq_len)，需要转置为 (batch, seq_len, channels)
    if x_fake_np.shape[1] == 22 and x_fake_np.shape[2] == 1000:
        print("数据格式为 (batch, 22, 1000)，转置为 (batch, 1000, 22)")
        x_fake_np = np.transpose(x_fake_np, (0, 2, 1))
        x_real1_np = np.transpose(x_real1_np, (0, 2, 1))
        x_real2_np = np.transpose(x_real2_np, (0, 2, 1))
    
    # 转换为 torch.Tensor
    x_fake  = torch.from_numpy(x_fake_np).float()
    x_real1 = torch.from_numpy(x_real1_np).float()
    x_real2 = torch.from_numpy(x_real2_np).float()
    
    print(f"\n转换后形状:")
    print(f"x_fake: {x_fake.shape}")
    print(f"x_real1: {x_real1.shape}")
    print(f"x_real2: {x_real2.shape}")
    
    # 计算边缘分布损失
    print("\n" + "="*60)
    print("比较 train_data (resub1234567) vs test_data (resub1234567)")
    print("="*60)
    loss1 = compute_marginal_loss(x_fake, x_real1)
    
    print("\n" + "="*60)
    print("比较 train_data (resub1234567) vs test_data (resub8)")
    print("="*60)
    loss2 = compute_marginal_loss(x_fake, x_real2)
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"resub1234567 train vs test 损失: {loss1:.6f}")
    print(f"resub1234567 train vs resub8 test 损失: {loss2:.6f}")