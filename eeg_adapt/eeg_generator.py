"""
EEG 信号生成器 - 函数式接口
可以在下游任务中直接调用，返回原始信号和生成信号
"""

import numpy as np
import torch as th
from typing import Tuple, Optional, Dict
import os
import sys

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from eeg_adapt.guided_diffusion import dist_util
from eeg_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


class EEGGenerator:
    """
    EEG 信号生成器类
    
    使用示例:
        # 创建生成器
        generator = EEGGenerator(model_path="models/model.pt")
        
        # 生成信号
        original, generated = generator.generate(eeg_data)
        
        # 批量生成
        for original_batch, generated_batch in generator.generate_batches(eeg_data, batch_size=16):
            # 处理每个批次
            pass
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        D: int = 8,
        scale: float = 1.0,
        N: Optional[int] = None,
        embedding_size: int = 64,
        delay: int = 15,
        image_size: int = 64,
        in_channels: int = 22,
        diffusion_steps: int = 1000,
        noise_schedule: str = "linear",
        **model_kwargs
    ):
        """
        初始化 EEG 生成器
        
        Args:
            model_path: 模型权重路径
            device: 指定设备 ("cuda", "cpu", None=自动检测)
            D: 频率引导下采样倍数
            scale: 频率引导强度
            N: 起始时间步（None=从头开始）
            embedding_size: 时间延迟嵌入维度
            delay: 延迟嵌入的delay参数
            image_size: 图像大小
            in_channels: 输入通道数（EEG为22）
            diffusion_steps: 扩散步数
            noise_schedule: 噪声调度
            **model_kwargs: 其他模型参数
        """
        self.model_path = model_path
        self.D = D
        self.scale = scale
        self.N = N
        self.embedding_size = embedding_size
        self.delay = delay
        
        # 设置设备
        if device is None:
            self.device = dist_util.dev()
        else:
            self.device = th.device(device)
        
        print(f"🖥️  使用设备: {self.device}")
        
        # 模型配置（使用完整的默认参数）
        from eeg_adapt.guided_diffusion.script_util import model_and_diffusion_defaults
        
        # 获取所有默认参数
        model_config = model_and_diffusion_defaults()
        
        # 更新关键参数
        model_config.update({
            'image_size': image_size,
            'in_channels': in_channels,
            'num_channels': 128,
            'num_res_blocks': 2,
            'num_heads': 4,
            'num_head_channels': 64,
            'attention_resolutions': "32,16,8",
            'dropout': 0.1,
            'diffusion_steps': diffusion_steps,
            'noise_schedule': noise_schedule,
            'learn_sigma': True,
            'class_cond': False,
            'use_checkpoint': False,
            'use_scale_shift_norm': True,
            'resblock_updown': True,
            'use_fp16': True,
            'use_new_attention_order': True,
            'timestep_respacing': "",
        })
        
        # 应用用户提供的其他参数
        model_config.update(model_kwargs)
        
        # 创建模型
        print(f"📦 加载模型: {model_path}")
        self.model, self.diffusion = create_model_and_diffusion(**model_config)
        self.model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        self.model.to(self.device)
        
        # 如果使用 FP16，需要转换模型
        if model_config.get('use_fp16', False):
            self.model.convert_to_fp16()
            print(f"🔧 已转换为 FP16 模式")
        
        self.model.eval()
        print(f"✅ 模型加载完成")
        
        # 初始化 embedder
        from eeg_adapt.scripts.eeg_sample import DelayEmbedder, img_to_ts
        self.DelayEmbedder = DelayEmbedder
        self.img_to_ts = img_to_ts
    
    def _normalize_data(self, data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """归一化数据到 [-1, 1]"""
        data_min = data.min()
        data_max = data.max()
        
        if -1.1 <= data_min and data_max <= 1.1:
            return data, data_min, data_max
        else:
            normalized = 2 * (data - data_min) / (data_max - data_min) - 1
            return normalized, data_min, data_max
    
    def _denormalize_data(self, data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
        """反归一化数据"""
        return (data + 1) / 2 * (data_max - data_min) + data_min
    
    def _eeg_to_image(self, eeg_data: np.ndarray) -> th.Tensor:
        """
        将 EEG 时间序列转换为图像格式
        
        Args:
            eeg_data: (trials, channels, timepoints) 或 (trials, timepoints, channels)
        
        Returns:
            images: (trials, channels, height, width)
        """
        # 确保数据格式为 (trials, channels, timepoints)
        if eeg_data.ndim == 4 and eeg_data.shape[1] == 1:
            eeg_data = eeg_data.squeeze(1)
        
        if eeg_data.shape[1] == 1000 and eeg_data.shape[2] == 22:
            eeg_data = np.transpose(eeg_data, (0, 2, 1))
        
        # 转换为 (trials, timepoints, channels)
        eeg_data_transposed = np.transpose(eeg_data, (0, 2, 1))
        
        # 创建 embedder
        seq_len = eeg_data.shape[2]
        embedder = self.DelayEmbedder(self.device, seq_len, self.delay, self.embedding_size)
        
        # 转换为图像
        batch_tensor = th.from_numpy(eeg_data_transposed).float().to(self.device)
        images = embedder.ts_to_img(batch_tensor, pad=True, mask=0)
        
        return images
    
    def _image_to_eeg(self, images: th.Tensor) -> np.ndarray:
        """
        将图像格式转换回 EEG 时间序列
        
        Args:
            images: (trials, channels, height, width)
        
        Returns:
            eeg_data: (trials, channels, timepoints)
        """
        signals = self.img_to_ts(images)  # (trials, timepoints, channels)
        signals = signals.permute(0, 2, 1)  # (trials, channels, timepoints)
        return signals.cpu().detach().numpy()
    
    def generate(
        self,
        eeg_data: np.ndarray,
        return_images: bool = False,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成 EEG 信号
        
        Args:
            eeg_data: 输入 EEG 数据
                     格式: (trials, channels, timepoints) 或 (trials, 22, 1000)
            return_images: 是否同时返回图像格式
            verbose: 是否显示详细信息
        
        Returns:
            original_signals: 原始信号 (trials, channels, timepoints)
            generated_signals: 生成信号 (trials, channels, timepoints)
            如果 return_images=True，还会返回:
                (original_signals, generated_signals, original_images, generated_images)
        """
        if verbose:
            print(f"\n🔄 生成 EEG 信号...")
            print(f"   输入形状: {eeg_data.shape}")
        
        # 保存原始数据（用于返回）
        original_data = eeg_data.copy()
        
        # 归一化
        eeg_data_normalized, data_min, data_max = self._normalize_data(eeg_data)
        
        if verbose:
            print(f"   数据范围: [{data_min:.4f}, {data_max:.4f}]")
        
        # 转换为图像
        images = self._eeg_to_image(eeg_data_normalized)
        
        if verbose:
            print(f"   图像形状: {images.shape}")
        
        # 准备模型输入
        model_kwargs = {"ref_img": images}
        
        # 生成
        if verbose:
            print(f"   开始生成... (D={self.D}, scale={self.scale}, N={self.N})")
        
        # ILVR 模式需要梯度，不使用 no_grad
        generated_images = self.diffusion.p_sample_loop(
            self.model,
            images.shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            noise=images,
            N=self.N,
            D=self.D,
            scale=self.scale
        )
        
        # 转换回时间序列
        generated_signals = self._image_to_eeg(generated_images)
        
        # 反归一化
        generated_signals = self._denormalize_data(generated_signals, data_min, data_max)
        
        if verbose:
            print(f"   ✅ 生成完成")
            print(f"   输出形状: {generated_signals.shape}")
        
        if return_images:
            return original_data, generated_signals, images.cpu().detach().numpy(), generated_images.cpu().detach().numpy()
        else:
            return original_data, generated_signals
    
    def generate_batches(
        self,
        eeg_data: np.ndarray,
        batch_size: int = 16,
        verbose: bool = True
    ):
        """
        批量生成 EEG 信号（生成器模式）
        
        Args:
            eeg_data: 输入 EEG 数据 (trials, channels, timepoints)
            batch_size: 批次大小
            verbose: 是否显示详细信息
        
        Yields:
            (original_batch, generated_batch) 元组
        """
        num_trials = len(eeg_data)
        num_batches = (num_trials + batch_size - 1) // batch_size
        
        if verbose:
            print(f"\n🔄 批量生成 EEG 信号...")
            print(f"   总样本数: {num_trials}")
            print(f"   批次大小: {batch_size}")
            print(f"   总批次数: {num_batches}")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_trials)
            
            if verbose:
                print(f"\n   批次 {i+1}/{num_batches} [{start_idx}:{end_idx}]")
            
            batch_data = eeg_data[start_idx:end_idx]
            original, generated = self.generate(batch_data, verbose=False)
            
            yield original, generated


# 便捷函数
def generate_eeg_signals(
    eeg_data: np.ndarray,
    model_path: str,
    device: Optional[str] = None,
    D: int = 8,
    scale: float = 1.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：一行代码生成 EEG 信号
    
    Args:
        eeg_data: 输入 EEG 数据 (trials, channels, timepoints)
        model_path: 模型权重路径
        device: 设备 ("cuda", "cpu", None=自动)
        D: 频率引导下采样倍数
        scale: 频率引导强度
        **kwargs: 其他参数
    
    Returns:
        original_signals: 原始信号
        generated_signals: 生成信号
    
    示例:
        >>> original, generated = generate_eeg_signals(
        ...     eeg_data=test_data,
        ...     model_path="models/model.pt",
        ...     D=4,
        ...     scale=6.0
        ... )
    """
    generator = EEGGenerator(
        model_path=model_path,
        device=device,
        D=D,
        scale=scale,
        **kwargs
    )
    return generator.generate(eeg_data)


if __name__ == "__main__":
    # 测试代码
    print("EEG Generator 测试")
    print("=" * 60)
    
    # 创建示例数据
    test_data = np.random.randn(10, 22, 1000) * 50
    print(f"测试数据形状: {test_data.shape}")
    
    # 方式 1: 使用类
    print("\n方式 1: 使用 EEGGenerator 类")
    generator = EEGGenerator(
        model_path="models/test_model.pt",  # 替换为实际模型路径
        D=8,
        scale=1.0
    )
    original, generated = generator.generate(test_data)
    print(f"原始信号形状: {original.shape}")
    print(f"生成信号形状: {generated.shape}")
    
    # 方式 2: 使用便捷函数
    print("\n方式 2: 使用便捷函数")
    original, generated = generate_eeg_signals(
        eeg_data=test_data,
        model_path="models/test_model.pt",
        D=8,
        scale=1.0
    )
    
    # 方式 3: 批量生成
    print("\n方式 3: 批量生成")
    for i, (orig_batch, gen_batch) in enumerate(generator.generate_batches(test_data, batch_size=4)):
        print(f"  批次 {i+1}: 原始={orig_batch.shape}, 生成={gen_batch.shape}")

