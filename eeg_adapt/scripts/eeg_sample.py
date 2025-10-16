"""
使用预训练的扩散模型，基于参考 EEG 数据生成新的 EEG 数据
"""

import argparse
import os
import sys
import numpy as np
import torch as th
from tqdm import tqdm
import math

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eeg_adapt.guided_diffusion import dist_util, logger
from eeg_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

# 归一化函数
def normalize_eeg_data(data):
    """将数据归一化到 [-1, 1]"""
    data_min = data.min()
    data_max = data.max()
    return 2 * (data - data_min) / (data_max - data_min) - 1, data_min, data_max

def denormalize_eeg_data(data, data_min, data_max):
    """将数据从 [-1, 1] 还原到原始范围"""
    return (data + 1) / 2 * (data_max - data_min) + data_min


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
        x_padded = th.nn.functional.pad(x, padding, mode='constant', value=mask)
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
        
        x_image = th.zeros((batch, features, self.embedding, self.embedding))
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


def img_to_ts(image_data):
    """
    将图像数据转换为时间序列格式（使用时间延迟嵌入的逆过程）
    
    参数:
        image_data: torch tensor, shape=(batch, 22, 64, 64)
    
    返回:
        time_series: torch tensor, shape=(batch, 1000, 22)
    """
    batch, channels, rows, cols = image_data.shape
    
    # 参数设置
    seq_len = 1000
    delay = 15
    embedding = 64
    
    # 初始化重建的时间序列
    reconstructed_x_time_series = th.zeros((batch, channels, seq_len))
    
    # 重建时间序列（时间延迟嵌入的逆过程）
    for i in range(cols - 1):
        start = i * delay
        end = start + embedding
        reconstructed_x_time_series[:, :, start:end] = image_data[:, :, :, i]
    
    # 处理最后一列（特殊情况）
    start = (cols - 1) * delay
    end = reconstructed_x_time_series[:, :, start:].shape[-1]
    reconstructed_x_time_series[:, :, start:] = image_data[:, :, :end, cols - 1]
    
    # 转换维度: (batch, channels, seq_len) -> (batch, seq_len, channels)
    reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)
    
    return reconstructed_x_time_series


def load_eeg_data(data_path):
    """
    加载 EEG 数据
    
    参数:
        data_path: 数据文件路径
    
    返回:
        data: numpy 数组, shape=(trials, 22, 1000)
    """
    print(f"从 {data_path} 加载 EEG 数据...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    data = np.load(data_path)
    print(f"原始数据形状: {data.shape}")
    
    # 确保数据格式为 (trials, channels, timepoints)
    if data.ndim == 4 and data.shape[1] == 1:
        # 如果是 (trials, 1, channels, timepoints)，去掉第二维
        data = data.squeeze(1)
        print(f"去掉维度1后的形状: {data.shape}")
    
    if data.shape[1] == 1000 and data.shape[2] == 22:
        # 如果是 (trials, timepoints, channels)，转换为 (trials, channels, timepoints)
        data = np.transpose(data, (0, 2, 1))
        print(f"转换数据格式为: {data.shape}")
    
    return data


def create_eeg_reference_loader(data_path, batch_size, embedding_size=64, delay=15):
    """
    创建 EEG 参考数据加载器
    
    参数:
        data_path: EEG 数据路径 (trials, 22, 1000)
        batch_size: 批次大小
        embedding_size: embedding 维度
        delay: delay 参数
    
    返回:
        生成器，每次yield一个batch的图像数据
    """
    # 加载数据
    eeg_data = load_eeg_data(data_path)
    num_trials = len(eeg_data)
    
    print(f"\n📊 数据信息:")
    print(f"   总 trials:    {num_trials}")
    print(f"   数据形状:     {eeg_data.shape}")
    print(f"   通道数:       {eeg_data.shape[1]}")
    print(f"   时间点数:     {eeg_data.shape[2]}")
    
    # 检查数据范围
    data_min_orig = eeg_data.min()
    data_max_orig = eeg_data.max()
    print(f"\n   原始数据范围: [{data_min_orig:.6f}, {data_max_orig:.6f}]")
    print(f"   均值:         {eeg_data.mean():.6f}")
    print(f"   标准差:       {eeg_data.std():.6f}")
    
    # 创建 embedder
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    seq_len = eeg_data.shape[2]  # 1000
    embedder = DelayEmbedder(device, seq_len, delay, embedding_size)
    
    # 智能归一化到 [-1, 1]
    if -1.1 <= data_min_orig and data_max_orig <= 1.1:
        print(f"   ✅ 数据已在 [-1, 1] 范围，跳过归一化")
        eeg_data_normalized = eeg_data
        data_min = data_min_orig
        data_max = data_max_orig
    else:
        print(f"   🔄 归一化数据到 [-1, 1] 范围...")
        eeg_data_normalized, data_min, data_max = normalize_eeg_data(eeg_data)
        print(f"   ✅ 归一化完成: [{eeg_data_normalized.min():.6f}, {eeg_data_normalized.max():.6f}]")
    
    # 批量处理并生成
    num_batches = (num_trials + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_trials)
        actual_batch_size = end_idx - start_idx
        
        # 获取当前批次
        batch_data = eeg_data_normalized[start_idx:end_idx]  # (batch, 22, 1000)
        
        # 转换为 (batch, 1000, 22) 以适配 ts_to_img
        batch_data = np.transpose(batch_data, (0, 2, 1))
        batch_tensor = th.from_numpy(batch_data).float().to(device)
        
        # 转换为图像格式
        batch_images = embedder.ts_to_img(batch_tensor, pad=True, mask=0)
        
        yield batch_images, actual_batch_size, data_min, data_max
    
    return data_min, data_max


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("创建模型...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.D, 2).is_integer(), f"D 必须是 2 的幂次，当前值: {args.D}"
    
    logger.log(f"频率引导参数: D={args.D}, scale={args.scale}")
    if args.N is not None:
        logger.log(f"将从时间步 N={args.N} 开始采样")

    logger.log("加载 EEG 数据并转换为图像格式...")
    data_loader = create_eeg_reference_loader(
        args.eeg_data_path,
        args.batch_size,
        embedding_size=args.embedding_size,
        delay=args.delay,
    )

    logger.log("开始生成样本...")
    all_generated_signals = []
    all_generated_images = []
    count = 0
    
    # 记录归一化参数（使用第一个batch的参数）
    data_min, data_max = None, None

    for batch_images, actual_batch_size, batch_data_min, batch_data_max in data_loader:
        # 记录归一化参数
        if data_min is None:
            data_min = batch_data_min
            data_max = batch_data_max
        
        # 将数据移到设备上
        batch_images = batch_images.to(dist_util.dev())
        
        # 准备模型输入
        model_kwargs = {}
        if args.class_cond:
            # 如果需要类别条件，可以在这里添加
            pass
        
        # 执行采样 (使用基于梯度的频率引导)
        logger.log(f"处理 batch {count+1}, 包含 {actual_batch_size} 个样本...")
        
        # 准备参考图像
        model_kwargs["ref_img"] = batch_images
        
        sample = diffusion.p_sample_loop(
            model,
            (actual_batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=batch_images,
            N=args.N,
            D=args.D,
            scale=args.scale
        )
        
        # 保存生成的图像（可选）
        if args.save_images:
            all_generated_images.append(sample.cpu())
        
        # 转换回时间序列
        logger.log(f"将生成的图像转换回时间序列...")
        generated_signals = img_to_ts(sample)  # (batch, 1000, 22)
        
        # 转换为 (batch, 22, 1000) 格式
        generated_signals = generated_signals.permute(0, 2, 1)
        
        # 添加到结果列表
        all_generated_signals.append(generated_signals.cpu().numpy())
        
        count += 1
        logger.log(f"已生成 {count * args.batch_size} 个样本")

    # 合并所有生成的信号
    all_generated_signals = np.concatenate(all_generated_signals, axis=0)
    
    # 反归一化
    logger.log("反归一化数据...")
    all_generated_signals = denormalize_eeg_data(all_generated_signals, data_min, data_max)
    
    # 保存生成的信号
    output_path = os.path.join(args.save_dir, "test_data_gen.npy")
    np.save(output_path, all_generated_signals)
    logger.log(f"生成的信号已保存到: {output_path}")
    logger.log(f"输出形状: {all_generated_signals.shape}")
    
    # 可选：保存生成的图像
    if args.save_images:
        all_generated_images = th.cat(all_generated_images, dim=0)
        image_output_path = os.path.join(args.save_dir, "generated_eeg_images.npy")
        np.save(image_output_path, all_generated_images.numpy())
        logger.log(f"生成的图像已保存到: {image_output_path}")
    
    logger.log("采样完成！")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
        model_path="",
        eeg_data_path="",  # EEG 数据路径 (trials, 22, 1000)
        save_dir="./eeg_samples",
        embedding_size=64,  # 嵌入维度
        delay=15,  # delay 参数
        save_images=False,  # 是否保存图像格式
        # 频率引导参数（基于梯度的方法）
        D=8,  # 频率引导的下采样倍数 (控制频率级别)
        scale=1.0,  # 频率引导的强度 (梯度缩放系数)
        N=None,  # 可选：从特定时间步开始采样
    )
    defaults.update(model_and_diffusion_defaults())
    # 为 EEG 数据设置默认参数
    defaults["in_channels"] = 22
    defaults["image_size"] = 64
    
    parser = argparse.ArgumentParser(
        description="使用预训练扩散模型和基于梯度的频率引导方法生成 EEG 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # 基本使用 (标准频率引导)
  python eeg_sample.py --model_path models/ema_0.9999_200000.pt \\
                       --eeg_data_path datasets/eegdata/bci2a/resub1234567/test_data.npy \\
                       --save_dir ./output
        """
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
