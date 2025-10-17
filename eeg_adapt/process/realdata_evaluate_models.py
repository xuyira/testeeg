"""
批量评估 EEG 扩散模型
对指定目录下的所有 .pt 模型进行评估，使用边缘分布差异指标和 MMD

功能：
1. 边缘分布直方图损失（HistoLoss）
2. Maximum Mean Discrepancy (MMD) - 使用高斯核的分布差异度量（内存优化版本）

特性：
- ✅ 内存高效的核矩阵计算（使用 torch.cdist）
- ✅ 支持多核混合（更鲁棒的 MMD 估计）
- ✅ 支持无偏估计
- ✅ 自动中位数启发式带宽选择

使用示例：
    # 直接计算两个数据的 MMD
    import torch
    from realdata_evaluate_models import compute_mmd
    
    # 加载数据
    x = torch.randn(100, 1000, 22)  # shape: (batch, seq_len, channels)
    y = torch.randn(100, 1000, 22)
    
    # 计算 MMD（使用默认的多核混合）
    mmd_value = compute_mmd(x, y)
    print(f"MMD: {mmd_value:.6f}")
    
    # 可选：使用单个核和指定的带宽
    mmd_value = compute_mmd(x, y, sigmas=1.0, use_multiple_kernels=False)
    
    # 可选：使用无偏估计
    mmd_value = compute_mmd(x, y, unbiased=True)
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


def gaussian_kernel_matrix(x, y, sigma):
    """
    通过 pairwise squared distances 生成高斯核矩阵
    x: (n, d)
    y: (m, d)
    sigma: float or tensor (标量)
    返回 K: (n, m)
    """
    # 使用 torch.cdist 更节省内存：返回 pairwise L2 距离
    # cdist -> 距离，平方得到 squared distances
    dists = torch.cdist(x, y, p=2.0)
    sq_dists = dists.pow(2)
    # 确保 sigma 是标量 tensor，在相同 device 上
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
    denom = 2.0 * (sigma ** 2)
    return torch.exp(-sq_dists / denom)


def _estimate_sigma_median(x, y):
    """
    基于 pairwise squared distance 的中位数启发式带宽选择（更稳健）
    返回标量 sigma (>0)
    """
    # 计算一部分距离以避免 O(n*m) 内存爆炸：当样本大时可以采样
    n, m = x.shape[0], y.shape[0]
    max_pairs = 10000  # 控制用于 median 的样本数
    # 采样索引
    if n * m > max_pairs:
        # 随机采样一些索引
        idx_x = torch.randint(0, n, (int(max_pairs**0.5),), device=x.device)
        idx_y = torch.randint(0, m, (int(max_pairs**0.5),), device=x.device)
        xs = x[idx_x]
        ys = y[idx_y]
        d = torch.cdist(xs, ys, p=2.0).reshape(-1).pow(2)
    else:
        d = torch.cdist(x, y, p=2.0).reshape(-1).pow(2)
    # 移除 0 与 nan
    d = d[~torch.isnan(d)]
    d = d[d > 0]
    if d.numel() == 0:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    med = torch.median(d)
    sigma = torch.sqrt(med)
    # 如果 sigma 非常小或为0（修复：正确处理 tensor 比较）
    if sigma.item() <= 0 or torch.isnan(sigma):
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return sigma


def compute_mmd(x, y, sigmas=None, use_multiple_kernels=True, unbiased=False, return_tensor=False):
    """
    计算 MMD（默认返回标量 float）
    x: (n, d) 或 (n, seq_len, ch)
    y: (m, d) 或 (m, seq_len, ch)
    sigmas: None 或 list/tuple 或 单个数值。如果 None 且 use_multiple_kernels False，会用 median heuristic。
    use_multiple_kernels: 使用多尺度核（谱）平均
    unbiased: 是否使用无偏估计（去掉自内积对角项）
    return_tensor: 若为 True 返回 tensor（可用于反向传播），否则返回 float
    """
    # flatten 时间序列
    if x.dim() == 3:
        x = x.reshape(x.shape[0], -1)
    if y.dim() == 3:
        y = y.reshape(y.shape[0], -1)

    x = x.float()
    y = y.float()

    if use_multiple_kernels:
        if sigmas is None:
            # 常用多尺度核列表（可替换）
            sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
        elif not isinstance(sigmas, (list, tuple)):
            sigmas = [sigmas]
    else:
        if sigmas is None:
            # 单核时使用 median heuristic
            sigma0 = _estimate_sigma_median(x, y)
            sigmas = [sigma0]

    mmd_total = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    for s in sigmas:
        # 如果 s 是 None，使用估计
        if s is None:
            s = _estimate_sigma_median(x, y)
        # 确保 s 为 tensor 在正确 device
        if not torch.is_tensor(s):
            s = torch.tensor(float(s), device=x.device, dtype=x.dtype)

        Kxx = gaussian_kernel_matrix(x, x, s)
        Kyy = gaussian_kernel_matrix(y, y, s)
        Kxy = gaussian_kernel_matrix(x, y, s)

        if unbiased:
            # 无偏估计：去掉对角项
            n = x.shape[0]
            m = y.shape[0]
            if n > 1:
                sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
            else:
                sum_xx = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            if m > 1:
                sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
            else:
                sum_yy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            sum_xy = Kxy.mean()  # cross terms无需去对角
            mmd_sq = sum_xx + sum_yy - 2.0 * sum_xy
        else:
            # 有偏估计（包括对角）
            mmd_sq = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

        mmd_total += mmd_sq

    mmd_total = mmd_total / len(sigmas)

    # 修正数值小负值
    mmd_total = torch.clamp(mmd_total, min=0.0)

    mmd = torch.sqrt(mmd_total)

    if return_tensor:
        return mmd  # tensor，可用于反向传播
    else:
        return mmd.item()


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
    x_fake_np  = np.load('eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_data_.npy')
    x_real1_np = np.load('eeg_adapt/datasets/eegdata/bci2a/resub8/test_data_gen.npy')
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
    
    # 计算边缘分布损失和 MMD
    print("\n" + "="*60)
    print("比较 train_data (resub1234567) vs test_data_gen (resub8)")
    print("="*60)
    loss1 = compute_marginal_loss(x_fake, x_real1)
    print("\n计算 Maximum Mean Discrepancy (MMD)...")
    mmd1 = compute_mmd(x_fake, x_real1)
    print(f"MMD: {mmd1:.6f}")
    
    print("\n" + "="*60)
    print("比较 train_data (resub1234567) vs test_data (resub8)")
    print("="*60)
    loss2 = compute_marginal_loss(x_fake, x_real2)
    print("\n计算 Maximum Mean Discrepancy (MMD)...")
    mmd2 = compute_mmd(x_fake, x_real2)
    print(f"MMD: {mmd2:.6f}")
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"resub1234567 train vs test_gen 边缘分布损失: {loss1:.6f}, MMD: {mmd1:.6f}")
    print(f"resub1234567 train vs resub8 test 边缘分布损失: {loss2:.6f}, MMD: {mmd2:.6f}")