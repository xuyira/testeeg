#!/usr/bin/env python3
"""
监控训练进度，判断是否收敛
使用方法：python scripts/monitor_training.py --log_dir logs/
"""

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_file):
    """解析训练日志文件"""
    steps = []
    losses = []
    loss_q0s = []
    grad_norms = []
    
    with open(log_file, 'r') as f:
        content = f.read()
        
    # 匹配所有的step和loss记录
    step_pattern = r'\| step\s+\| ([\d.e\+\-]+)'
    loss_pattern = r'\| loss\s+\| ([\d.e\+\-]+)'
    loss_q0_pattern = r'\| loss_q0\s+\| ([\d.e\+\-]+)'
    grad_norm_pattern = r'\| grad_norm\s+\| ([\d.e\+\-]+)'
    
    step_matches = re.findall(step_pattern, content)
    loss_matches = re.findall(loss_pattern, content)
    loss_q0_matches = re.findall(loss_q0_pattern, content)
    grad_norm_matches = re.findall(grad_norm_pattern, content)
    
    for i in range(min(len(step_matches), len(loss_matches))):
        try:
            steps.append(float(step_matches[i]))
            losses.append(float(loss_matches[i]))
            if i < len(loss_q0_matches):
                loss_q0s.append(float(loss_q0_matches[i]))
            if i < len(grad_norm_matches):
                grad_norms.append(float(grad_norm_matches[i]))
        except:
            continue
    
    return np.array(steps), np.array(losses), np.array(loss_q0s), np.array(grad_norms)


def check_convergence(losses, window=50, threshold=0.01):
    """
    检查是否收敛
    :param losses: loss数组
    :param window: 观察窗口大小
    :param threshold: 相对变化阈值（1%）
    """
    if len(losses) < window + 10:
        return False, "数据点不足，无法判断"
    
    # 计算最近window步的loss均值
    recent_mean = np.mean(losses[-window:])
    # 计算之前window步的loss均值
    previous_mean = np.mean(losses[-2*window:-window])
    
    # 计算相对变化
    if previous_mean > 0:
        relative_change = abs(recent_mean - previous_mean) / previous_mean
        
        if relative_change < threshold:
            return True, f"收敛！最近{window}步loss变化 < {threshold*100}%"
        else:
            return False, f"仍在训练中，最近{window}步loss变化: {relative_change*100:.2f}%"
    
    return False, "无法计算"


def plot_training_curves(steps, losses, loss_q0s, grad_norms, save_path=None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss曲线
    axes[0, 0].plot(steps, losses, 'b-', alpha=0.3, label='Raw Loss')
    if len(losses) > 100:
        # 平滑曲线
        window = min(100, len(losses) // 10)
        smooth_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[:len(smooth_loss)]
        axes[0, 0].plot(smooth_steps, smooth_loss, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Loss_q0曲线
    if len(loss_q0s) > 0:
        axes[0, 1].plot(steps[:len(loss_q0s)], loss_q0s, 'g-', alpha=0.3)
        if len(loss_q0s) > 100:
            window = min(100, len(loss_q0s) // 10)
            smooth_q0 = np.convolve(loss_q0s, np.ones(window)/window, mode='valid')
            smooth_steps = steps[:len(smooth_q0)]
            axes[0, 1].plot(smooth_steps, smooth_q0, 'darkgreen', linewidth=2)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss Q0')
        axes[0, 1].set_title('Loss Q0 (Hardest Timesteps)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # 3. 梯度范数
    if len(grad_norms) > 0:
        axes[1, 0].plot(steps[:len(grad_norms)], grad_norms, 'm-', alpha=0.5)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 最近的Loss详情
    if len(losses) > 1000:
        recent_steps = steps[-1000:]
        recent_losses = losses[-1000:]
        axes[1, 1].plot(recent_steps, recent_losses, 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Recent 1000 Steps - Loss Detail')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("训练曲线已保存到: training_curves.png")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='监控扩散模型训练进度')
    parser.add_argument('--log_dir', type=str, default='logs/', help='日志目录路径')
    parser.add_argument('--log_file', type=str, default=None, help='具体日志文件路径')
    parser.add_argument('--window', type=int, default=50, help='收敛检查窗口大小')
    parser.add_argument('--threshold', type=float, default=0.01, help='收敛阈值（默认1%）')
    args = parser.parse_args()
    
    # 查找日志文件
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_dir = Path(args.log_dir)
        # 尝试找到最新的日志文件
        log_files = list(log_dir.glob('*.txt')) + list(log_dir.glob('*.log'))
        if not log_files:
            print(f"错误：在 {log_dir} 中未找到日志文件")
            print("请手动指定日志文件：--log_file path/to/logfile")
            return
        log_file = sorted(log_files, key=lambda x: x.stat().st_mtime)[-1]
    
    if not log_file.exists():
        print(f"错误：日志文件不存在: {log_file}")
        return
    
    print(f"读取日志文件: {log_file}")
    print("=" * 60)
    
    try:
        steps, losses, loss_q0s, grad_norms = parse_log_file(log_file)
        
        if len(losses) == 0:
            print("警告：未能从日志文件中解析到loss数据")
            return
        
        print(f"\n📊 训练统计:")
        print(f"  总步数: {int(steps[-1])}")
        print(f"  记录点数: {len(losses)}")
        print(f"  当前Loss: {losses[-1]:.6f}")
        print(f"  初始Loss: {losses[0]:.6f}")
        print(f"  Loss下降: {(1 - losses[-1]/losses[0])*100:.2f}%")
        
        if len(grad_norms) > 0:
            print(f"  当前梯度范数: {grad_norms[-1]:.6f}")
        
        print("\n" + "=" * 60)
        
        # 检查收敛
        converged, message = check_convergence(losses, window=args.window, threshold=args.threshold)
        
        print(f"\n🔍 收敛分析 (观察窗口={args.window}步):")
        print(f"  {message}")
        
        if converged:
            print("\n✅ 建议：模型可能已收敛，可以考虑停止训练")
        else:
            # 估算还需要多久
            if len(losses) > 100:
                recent_rate = (losses[-100] - losses[-1]) / 100
                if recent_rate > 0:
                    print(f"\n⏳ Loss仍在下降，建议继续训练")
                    print(f"  近100步平均下降速率: {recent_rate:.8f}/step")
        
        print("\n" + "=" * 60)
        print("\n📈 生成训练曲线...")
        
        # 绘制曲线
        save_path = log_file.parent / "training_curves.png"
        plot_training_curves(steps, losses, loss_q0s, grad_norms, save_path)
        
        print("\n✨ 分析完成！")
        
    except Exception as e:
        print(f"错误：处理日志文件时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

