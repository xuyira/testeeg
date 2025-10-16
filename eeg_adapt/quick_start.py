"""
EEG Generator 快速开始示例
最简单的使用方法
"""

import numpy as np
from eeg_adapt.eeg_generator import generate_eeg_signals, EEGGenerator

# ============================================================================
# 方法 1: 最简单的使用方式（一行代码）
# ============================================================================
def quick_example_1():
    """最简单：一行代码生成"""
    print("=" * 60)
    print("方法 1: 一行代码生成")
    print("=" * 60)
    
    # 加载你的 EEG 数据 (trials, channels, timepoints)
    test_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")
    
    # 一行代码生成！
    original, generated = generate_eeg_signals(
        eeg_data=test_data,
        model_path="models/ema_0.9999_200000.pt"
    )
    
    print(f"✅ 完成！")
    print(f"   原始信号形状: {original.shape}")
    print(f"   生成信号形状: {generated.shape}")
    
    return original, generated


# ============================================================================
# 方法 2: 多次生成（推荐）
# ============================================================================
def quick_example_2():
    """多次生成：创建一次，使用多次"""
    print("\n" + "=" * 60)
    print("方法 2: 多次生成（推荐）")
    print("=" * 60)
    
    # 创建生成器（只需创建一次）
    generator = EEGGenerator(model_path="models/ema_0.9999_200000.pt")
    
    # 可以多次使用
    test_data_1 = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")
    test_data_2 = np.load("eeg_adapt/datasets/eegdata/bci2a/resub8/test_data.npy")
    
    # 生成 1
    print("\n生成数据集 1...")
    original_1, generated_1 = generator.generate(test_data_1)
    print(f"✅ 数据集 1: {generated_1.shape}")
    
    # 生成 2
    print("\n生成数据集 2...")
    original_2, generated_2 = generator.generate(test_data_2)
    print(f"✅ 数据集 2: {generated_2.shape}")
    
    return (original_1, generated_1), (original_2, generated_2)


# ============================================================================
# 方法 3: 数据增强（最常用）
# ============================================================================
def quick_example_3():
    """数据增强：翻倍训练数据"""
    print("\n" + "=" * 60)
    print("方法 3: 数据增强（最常用）")
    print("=" * 60)
    
    # 加载训练数据
    train_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_data.npy")
    train_labels = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_label.npy")
    
    print(f"原始训练数据: {train_data.shape}")
    
    # 生成增强数据
    generator = EEGGenerator(model_path="models/ema_0.9999_200000.pt")
    _, augmented_data = generator.generate(train_data)
    
    # 合并原始和增强数据
    X_train = np.concatenate([train_data, augmented_data], axis=0)
    y_train = np.concatenate([train_labels, train_labels], axis=0)
    
    print(f"增强后训练数据: {X_train.shape}")
    print(f"✅ 数据量翻倍！")
    
    return X_train, y_train


# ============================================================================
# 方法 4: 批量处理（大数据集）
# ============================================================================
def quick_example_4():
    """批量处理：避免内存溢出"""
    print("\n" + "=" * 60)
    print("方法 4: 批量处理（大数据集）")
    print("=" * 60)
    
    generator = EEGGenerator(model_path="models/ema_0.9999_200000.pt")
    
    large_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")
    
    all_generated = []
    
    # 逐批生成
    for original_batch, generated_batch in generator.generate_batches(
        large_data, 
        batch_size=32  # 根据你的显存调整
    ):
        all_generated.append(generated_batch)
        print(f"  批次形状: {generated_batch.shape}")
    
    # 合并所有批次
    all_generated = np.concatenate(all_generated, axis=0)
    print(f"\n✅ 总共生成: {all_generated.shape}")
    
    return all_generated


# ============================================================================
# 运行示例
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 EEG Generator 快速开始")
    print("=" * 60)
    print("\n请确保:")
    print("  1. 模型文件存在: models/ema_0.9999_200000.pt")
    print("  2. 数据文件存在: eeg_adapt/datasets/eegdata/bci2a/...")
    print("\n" + "=" * 60)
    
    input("\n按 Enter 键开始运行示例...")
    
    # 运行所有示例
    try:
        # 示例 1
        original, generated = quick_example_1()
        
        # 示例 2
        results = quick_example_2()
        
        # 示例 3
        X_train, y_train = quick_example_3()
        
        # 示例 4
        all_generated = quick_example_4()
        
        print("\n" + "=" * 60)
        print("✅ 所有示例运行成功！")
        print("=" * 60)
        print("\n下一步:")
        print("  - 查看 eeg_adapt/GENERATOR_API.md 了解详细文档")
        print("  - 查看 eeg_adapt/examples/use_generator.py 了解更多示例")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("\n请确保模型和数据文件路径正确")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

