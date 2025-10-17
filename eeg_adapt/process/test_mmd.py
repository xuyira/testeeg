"""
简单测试 MMD 计算功能
展示如何直接使用 compute_mmd 函数
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from realdata_evaluate_models import compute_mmd


def test_mmd_basic():
    """基础 MMD 测试"""
    print("="*60)
    print("测试 1: 基础 MMD 计算")
    print("="*60)
    
    # 创建两个相似的分布
    torch.manual_seed(42)
    x = torch.randn(100, 1000, 22)
    y = torch.randn(100, 1000, 22)
    
    print(f"数据 x 形状: {x.shape}")
    print(f"数据 y 形状: {y.shape}")
    
    # 计算 MMD
    mmd = compute_mmd(x, y)
    print(f"\nMMD (使用多核): {mmd:.6f}")
    
    # 使用单核
    mmd_single = compute_mmd(x, y, sigma=1.0, use_multiple_kernels=False)
    print(f"MMD (单核, sigma=1.0): {mmd_single:.6f}")


def test_mmd_same_distribution():
    """测试相同分布的 MMD"""
    print("\n" + "="*60)
    print("测试 2: 相同分布的 MMD（应该接近 0）")
    print("="*60)
    
    # 从同一分布采样
    torch.manual_seed(42)
    x = torch.randn(100, 1000, 22)
    torch.manual_seed(42)
    y = torch.randn(100, 1000, 22)
    
    mmd = compute_mmd(x, y)
    print(f"MMD (相同数据): {mmd:.6f}")
    print("✅ 预期接近 0")


def test_mmd_different_distributions():
    """测试不同分布的 MMD"""
    print("\n" + "="*60)
    print("测试 3: 不同分布的 MMD（应该较大）")
    print("="*60)
    
    # 两个不同的分布
    x = torch.randn(100, 1000, 22) * 1.0  # 标准正态分布
    y = torch.randn(100, 1000, 22) * 2.0 + 3.0  # 不同均值和方差
    
    mmd = compute_mmd(x, y)
    print(f"MMD (不同分布): {mmd:.6f}")
    print("✅ 预期较大值")


def test_mmd_with_real_data():
    """使用真实 EEG 数据测试"""
    print("\n" + "="*60)
    print("测试 4: 使用真实 EEG 数据")
    print("="*60)
    
    # 尝试加载真实数据
    data_path1 = "eeg_adapt/datasets/eegdata/bci2a/resub8/test_data.npy"
    data_path2 = "eeg_adapt/datasets/eegdata/bci2a/resub8/test_data_gen.npy"
    
    if os.path.exists(data_path1) and os.path.exists(data_path2):
        x_np = np.load(data_path1)
        y_np = np.load(data_path2)
        
        print(f"数据 1 形状: {x_np.shape}")
        print(f"数据 2 形状: {y_np.shape}")
        
        # 确保格式为 (batch, seq_len, channels)
        if x_np.shape[1] == 22 and x_np.shape[2] == 1000:
            x_np = np.transpose(x_np, (0, 2, 1))
            y_np = np.transpose(y_np, (0, 2, 1))
        
        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()
        
        mmd = compute_mmd(x, y)
        print(f"\nMMD (真实数据 vs 生成数据): {mmd:.6f}")
    else:
        print("⚠️ 真实数据文件不存在，跳过此测试")


if __name__ == "__main__":
    print("\n🧪 MMD 计算功能测试\n")
    
    test_mmd_basic()
    test_mmd_same_distribution()
    test_mmd_different_distributions()
    test_mmd_with_real_data()
    
    print("\n" + "="*60)
    print("✅ 所有测试完成！")
    print("="*60)
    print("\n💡 使用提示:")
    print("   from realdata_evaluate_models import compute_mmd")
    print("   mmd = compute_mmd(data1, data2)")
    print("="*60)

