"""
查看 EEG 扩散模型的架构和参数量
"""

import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eeg_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """格式化数字（添加千位分隔符和单位）"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def main():
    print("="*80)
    print("EEG 扩散模型信息")
    print("="*80)
    print()
    
    # 根据您的训练参数创建模型
    config = {
        'image_size': 64,
        'in_channels': 22,           # EEG 通道数
        'num_channels': 128,         # 基础通道数
        'num_res_blocks': 2,         # 每层的残差块数量
        'learn_sigma': True,
        'class_cond': False,
        'use_fp16': True,
        'attention_resolutions': "32,16,8",
        'dropout': 0.1,
        'noise_schedule': 'cosine',
        'num_head_channels': 64,
        'resblock_updown': True,
        'use_new_attention_order': True,
        'use_scale_shift_norm': True,
        'diffusion_steps': 1000,
        'timestep_respacing': "",
    }
    
    print("📋 模型配置:")
    print("-"*80)
    print(f"  模型类型:          UNet (时间步条件扩散模型)")
    print(f"  输入大小:          {config['image_size']}x{config['image_size']}")
    print(f"  输入通道:          {config['in_channels']} (EEG通道数)")
    print(f"  基础通道数:        {config['num_channels']}")
    print(f"  残差块数量/层:     {config['num_res_blocks']}")
    print(f"  注意力分辨率:      {config['attention_resolutions']}")
    print(f"  注意力头通道数:    {config['num_head_channels']}")
    print(f"  Dropout:           {config['dropout']}")
    print(f"  学习 Sigma:        {config['learn_sigma']}")
    print(f"  使用 FP16:         {config['use_fp16']}")
    print(f"  残差上下采样:      {config['resblock_updown']}")
    print(f"  Scale-Shift Norm:  {config['use_scale_shift_norm']}")
    print(f"  噪声调度:          {config['noise_schedule']}")
    print(f"  扩散步数:          {config['diffusion_steps']}")
    print()
    
    # 创建模型
    print("🔨 创建模型...")
    model, diffusion = create_model_and_diffusion(**config)
    
    # 计算参数量
    total_params, trainable_params = count_parameters(model)
    
    print()
    print("="*80)
    print("📊 模型参数统计")
    print("="*80)
    print(f"  总参数量:          {total_params:,} ({format_number(total_params)})")
    print(f"  可训练参数:        {trainable_params:,} ({format_number(trainable_params)})")
    print(f"  内存占用（FP32）:  约 {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  内存占用（FP16）:  约 {total_params * 2 / 1024 / 1024:.2f} MB")
    print()
    
    # 模型架构详情
    print("="*80)
    print("🏗️  模型架构")
    print("="*80)
    print()
    
    # 获取通道倍数
    from eeg_adapt.guided_diffusion.script_util import model_and_diffusion_defaults
    defaults = model_and_diffusion_defaults()
    channel_mult = defaults.get('channel_mult', '')
    if channel_mult == '':
        # 根据 image_size 自动计算
        if config['image_size'] == 64:
            channel_mult = (1, 2, 3, 4)
        elif config['image_size'] == 128:
            channel_mult = (1, 2, 2, 2, 2)
        elif config['image_size'] == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        else:
            channel_mult = (1, 2, 4, 8)
    
    print("UNet 结构:")
    print(f"  基础通道: {config['num_channels']}")
    print(f"  通道倍数: {channel_mult}")
    print()
    
    # 计算每层的通道数
    print("各层通道数:")
    for i, mult in enumerate(channel_mult):
        channels = config['num_channels'] * mult
        resolution = config['image_size'] // (2 ** i)
        has_attention = str(resolution) in config['attention_resolutions'].split(',')
        attention_mark = " [+Attention]" if has_attention else ""
        print(f"  层 {i}: {resolution:3d}x{resolution:3d} → {channels:4d} 通道 × {config['num_res_blocks']} 残差块{attention_mark}")
    
    print()
    print("时间步嵌入维度:", config['num_channels'] * 4)
    
    # 输出通道
    out_channels = config['in_channels'] * 2 if config['learn_sigma'] else config['in_channels']
    print(f"输出通道数: {out_channels} ({'学习方差' if config['learn_sigma'] else '固定方差'})")
    
    print()
    print("="*80)
    print("🎯 模型类型总结")
    print("="*80)
    print()
    print(f"  模型名称:    时间步条件 UNet 扩散模型")
    print(f"  论文来源:    基于 DDPM/Improved DDPM 架构")
    print(f"  改进:        支持 EEG 多通道输入（22通道）")
    print(f"  总参数量:    {format_number(total_params)}")
    print(f"  规模等级:    {'小型模型 (< 100M)' if total_params < 100e6 else '中型模型 (100M-500M)' if total_params < 500e6 else '大型模型 (> 500M)'}")
    print()
    
    # 对比其他模型
    print("="*80)
    print("📈 参数量对比")
    print("="*80)
    print()
    print(f"  您的 EEG 扩散模型:     {format_number(total_params)}")
    print(f"  DDPM (ImageNet):       约 60M")
    print(f"  Stable Diffusion:      约 860M")
    print(f"  DALL-E 2:              约 3.5B")
    print()
    print(f"  您的模型规模相当于 DDPM 的 {total_params/60e6:.2f}x")
    print()
    
    print("="*80)
    print("✅ 分析完成")
    print("="*80)

if __name__ == "__main__":
    main()

