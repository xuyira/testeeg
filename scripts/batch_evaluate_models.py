"""
批量评估 EEG 扩散模型
对指定目录下的所有 .pt 模型进行评估，使用边缘分布差异指标
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import json
from datetime import datetime

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 guided_diffusion 模块
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


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


def generate_samples_from_model(model, diffusion, num_samples, batch_size, in_channels, image_size, device):
    """
    从模型生成样本
    
    参数:
        model: 扩散模型
        diffusion: 扩散过程
        num_samples: 生成样本数量
        batch_size: 批次大小
        in_channels: 输入通道数
        image_size: 图像大小
        device: 设备
    
    返回:
        generated_samples: numpy 数组
    """
    model.eval()
    all_samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"\n生成 {num_samples} 个样本 (batch_size={batch_size}, {num_batches} 个批次)...")
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="生成样本"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # 生成样本
            sample = diffusion.p_sample_loop(
                model,
                (current_batch_size, in_channels, image_size, image_size),
                clip_denoised=True,
                device=device,
                progress=False,
            )
            
            all_samples.append(sample.cpu().numpy())
    
    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples[:num_samples]


def img_to_ts(image_data):
    """
    将图像数据转换为时间序列格式（使用时间延迟嵌入的逆过程）
    
    参数:
        image_data: numpy 数组或 torch tensor, shape=(batch, 22, 64, 64)
    
    返回:
        time_series: numpy 数组, shape=(batch, 1000, 22)
    
    使用参数:
        - seq_len: 1000 (时间序列长度)
        - delay: 15 (时间延迟)
        - embedding: 64 (嵌入维度，即图像高度)
    """
    # 转换为 torch tensor（如果是 numpy）
    is_numpy = isinstance(image_data, np.ndarray)
    if is_numpy:
        img = torch.from_numpy(image_data).float()
    else:
        img = image_data.float()
    
    batch, channels, rows, cols = img.shape
    
    # 参数设置
    seq_len = 1000
    delay = 15
    embedding = 64
    
    # 初始化重建的时间序列
    reconstructed_x_time_series = torch.zeros((batch, channels, seq_len))
    
    # 重建时间序列（时间延迟嵌入的逆过程）
    for i in range(cols - 1):
        start = i * delay
        end = start + embedding
        reconstructed_x_time_series[:, :, start:end] = img[:, :, :, i]
    
    # 处理最后一列（特殊情况）
    start = (cols - 1) * delay
    end = reconstructed_x_time_series[:, :, start:].shape[-1]
    reconstructed_x_time_series[:, :, start:] = img[:, :, :end, cols - 1]
    
    # 转换维度: (batch, channels, seq_len) -> (batch, seq_len, channels)
    reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)
    
    # 转回 numpy（如果输入是 numpy）
    if is_numpy:
        return reconstructed_x_time_series.cpu().numpy()
    else:
        return reconstructed_x_time_series


def evaluate_single_model(model_path, test_data, args, device):
    """
    评估单个模型
    
    参数:
        model_path: 模型文件路径
        test_data: 测试数据
        args: 参数配置
        device: 设备
    
    返回:
        results: 评估结果字典
    """
    print(f"\n{'='*80}")
    print(f"评估模型: {model_path}")
    print(f"{'='*80}")
    
    try:
        # 创建模型和扩散过程
        print("创建模型和扩散过程...")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        
        # 加载模型权重
        print(f"加载模型权重...")
        state_dict = torch.load(model_path, map_location=device)
        
        # 处理可能的 EMA 权重或 DDP 包装
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # 移除 'module.' 前缀（如果存在）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        
        # 确定评估样本数量
        if args.num_eval_samples is None or args.num_eval_samples <= 0:
            # 使用全部测试数据
            num_samples = test_data.shape[0]
            print(f"使用全部测试数据: {num_samples} 个样本")
        else:
            # 使用指定数量，但不超过测试集大小
            num_samples = min(args.num_eval_samples, test_data.shape[0])
            print(f"使用 {num_samples} 个样本进行评估")
        
        # 生成样本
        generated_images = generate_samples_from_model(
            model, diffusion, num_samples, args.batch_size,
            args.in_channels, args.image_size, device
        )
        
        print(f"生成数据形状: {generated_images.shape}")
        
        # 转换为时间序列格式
        print("转换图像数据为时间序列格式...")
        gen_timeseries = img_to_ts(generated_images)
        
        # 准备真实数据
        real_images = test_data[:num_samples]
        real_timeseries = img_to_ts(real_images)
        
        print(f"真实时间序列形状: {real_timeseries.shape}")
        print(f"生成时间序列形状: {gen_timeseries.shape}")
        
        # 转换为 Tensor
        real_tensor = torch.Tensor(real_timeseries).float()
        gen_tensor = torch.Tensor(gen_timeseries).float()
        
        # 计算评估指标
        marginal_loss = compute_marginal_loss(gen_tensor, real_tensor)
        
        print(f"\n最终评估结果:")
        print(f"  边缘分布损失: {marginal_loss:.6f}")
        
        return {
            'model_path': model_path,
            'success': True,
            'marginal_loss': marginal_loss,
            'error': None
        }
        
    except Exception as e:
        print(f"\n错误: 评估模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'model_path': model_path,
            'success': False,
            'marginal_loss': None,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="批量评估 EEG 扩散模型")
    
    # 数据相关参数
    parser.add_argument('--test_data_path', type=str, 
                       default='datasets/eegdata/bci2a/test_data_embedded.npy',
                       help='测试数据路径')
    parser.add_argument('--model_dir', type=str,
                       default='/home/xyr/workspace/testeeg/logs',
                       help='模型目录路径')
    parser.add_argument('--output_file', type=str,
                       default='/home/xyr/workspace/testeeg/logs/evaluation_results.json',
                       help='结果输出文件')
    
    # 评估参数
    parser.add_argument('--num_eval_samples', type=int, default=None,
                       help='用于评估的样本数量')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['ema', 'model', 'all'],
                       help='评估模型类型: ema(仅EMA模型), model(仅普通模型), all(两者都评估)')
    parser.add_argument('--timestep_respacing', type=str, default='',
                       help='采样步数重采样，如"100"表示用100步代替1000步，可大幅加速。留空则使用全部步数')
    
    # 模型参数（需要与训练时一致）
    parser.add_argument('--image_size', type=int, default=64,
                       help='图像大小')
    parser.add_argument('--in_channels', type=int, default=22,
                       help='输入通道数（EEG 通道数）')
    parser.add_argument('--num_channels', type=int, default=128,
                       help='模型通道数')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                       help='残差块数量')
    parser.add_argument('--learn_sigma', action='store_true',
                       help='是否学习 sigma')
    parser.add_argument('--class_cond', action='store_true',
                       help='是否使用类条件')
    parser.add_argument('--use_fp16', action='store_true',
                       help='是否使用 FP16')
    parser.add_argument('--attention_resolutions', type=str, default='16,8',
                       help='注意力分辨率')
    
    # 其他模型默认参数
    defaults = model_and_diffusion_defaults()
    for k, v in defaults.items():
        if k not in ['image_size', 'in_channels', 'num_channels', 'num_res_blocks', 
                     'learn_sigma', 'class_cond', 'use_fp16', 'attention_resolutions']:
            if isinstance(v, bool):
                parser.add_argument(f'--{k}', action='store_true' if not v else 'store_false',
                                  default=v, help=f'{k} (default: {v})')
            else:
                parser.add_argument(f'--{k}', type=type(v), default=v,
                                  help=f'{k} (default: {v})')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据
    test_data = load_test_data(args.test_data_path)
    
    # 查找所有 .pt 模型文件
    all_files = glob.glob(os.path.join(args.model_dir, '**/*.pt'), recursive=True)
    
    # 过滤掉优化器文件（opt*.pt）
    model_files = [f for f in all_files if not os.path.basename(f).startswith('opt')]
    
    # 根据 model_type 参数进一步过滤
    if args.model_type == 'ema':
        model_files = [f for f in model_files if 'ema' in os.path.basename(f).lower()]
    elif args.model_type == 'model':
        model_files = [f for f in model_files if 'ema' not in os.path.basename(f).lower()]
    # args.model_type == 'all' 时不需要额外过滤
    
    model_files.sort()
    
    if len(model_files) == 0:
        print(f"错误: 在 {args.model_dir} 中未找到符合条件的模型文件")
        print(f"  模型类型过滤: {args.model_type}")
        return
    
    print(f"\n找到 {len(model_files)} 个模型文件 (model_type={args.model_type}):")
    for mf in model_files:
        print(f"  - {mf}")
    
    # 评估所有模型
    all_results = []
    
    for model_path in model_files:
        result = evaluate_single_model(model_path, test_data, args, device)
        all_results.append(result)
    
    # 保存结果
    output_path = os.path.join(args.model_dir, args.output_file)
    print(f"\n保存评估结果到: {output_path}")
    
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_data_path': args.test_data_path,
        'model_dir': args.model_dir,
        'num_models_evaluated': len(model_files),
        'num_eval_samples': args.num_eval_samples,
        'results': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print(f"\n{'='*80}")
    print("评估汇总")
    print(f"{'='*80}")
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"成功评估: {len(successful)} 个模型")
    print(f"失败: {len(failed)} 个模型")
    
    if successful:
        print(f"\n成功评估的模型结果:")
        for result in successful:
            model_name = os.path.basename(result['model_path'])
            marginal_loss = result['marginal_loss']
            print(f"\n  {model_name}:")
            print(f"    边缘分布损失: {marginal_loss:.6f}")
    
    if failed:
        print(f"\n失败的模型:")
        for result in failed:
            model_name = os.path.basename(result['model_path'])
            print(f"  - {model_name}: {result['error']}")
    
    print(f"\n完整结果已保存至: {output_path}")


if __name__ == "__main__":
    main()

