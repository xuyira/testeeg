"""
EEG Generator 使用示例
展示如何在下游任务中使用 EEG 生成器
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_adapt.eeg_generator import EEGGenerator, generate_eeg_signals


def example_1_basic_usage():
    """示例 1: 基本使用"""
    print("=" * 80)
    print("示例 1: 基本使用")
    print("=" * 80)
    
    # 加载测试数据
    test_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")
    print(f"测试数据形状: {test_data.shape}")
    
    # 使用便捷函数生成
    original, generated = generate_eeg_signals(
        eeg_data=test_data,
        model_path="models/ema_0.9999_200000.pt",
        D=8,
        scale=1.0
    )
    
    print(f"\n结果:")
    print(f"  原始信号: {original.shape}")
    print(f"  生成信号: {generated.shape}")
    
    return original, generated


def example_2_class_usage():
    """示例 2: 使用类接口（推荐用于多次生成）"""
    print("\n" + "=" * 80)
    print("示例 2: 使用类接口（适合多次调用）")
    print("=" * 80)
    
    # 创建生成器（只需要创建一次）
    generator = EEGGenerator(
        model_path="models/ema_0.9999_200000.pt",
        D=8,
        scale=1.0,
        device="cuda"  # 或 "cpu"
    )
    
    # 加载多个数据集并生成
    test_data_1 = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")
    test_data_2 = np.load("eeg_adapt/datasets/eegdata/bci2a/resub8/test_data.npy")
    
    # 生成 1
    print("\n生成数据集 1...")
    original_1, generated_1 = generator.generate(test_data_1)
    
    # 生成 2
    print("\n生成数据集 2...")
    original_2, generated_2 = generator.generate(test_data_2)
    
    return generator, (original_1, generated_1), (original_2, generated_2)


def example_3_batch_generation():
    """示例 3: 批量生成（处理大数据集）"""
    print("\n" + "=" * 80)
    print("示例 3: 批量生成（适合大数据集）")
    print("=" * 80)
    
    generator = EEGGenerator(
        model_path="models/ema_0.9999_200000.pt",
        D=8,
        scale=1.0
    )
    
    test_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")
    
    # 批量生成并处理
    all_originals = []
    all_generated = []
    
    for original_batch, generated_batch in generator.generate_batches(
        test_data, 
        batch_size=32,  # 根据显存调整
        verbose=True
    ):
        # 在这里可以对每个批次进行处理
        all_originals.append(original_batch)
        all_generated.append(generated_batch)
    
    # 合并所有批次
    all_originals = np.concatenate(all_originals, axis=0)
    all_generated = np.concatenate(all_generated, axis=0)
    
    print(f"\n最终结果:")
    print(f"  原始信号: {all_originals.shape}")
    print(f"  生成信号: {all_generated.shape}")
    
    return all_originals, all_generated


def example_4_downstream_classification():
    """示例 4: 下游分类任务"""
    print("\n" + "=" * 80)
    print("示例 4: 用于下游分类任务")
    print("=" * 80)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # 创建生成器
    generator = EEGGenerator(
        model_path="models/ema_0.9999_200000.pt",
        D=8,
        scale=1.0
    )
    
    # 加载训练和测试数据
    train_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_data.npy")
    train_labels = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_label.npy")
    test_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")
    test_labels = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_label.npy")
    
    # 生成增强数据
    print("\n生成训练数据增强...")
    _, train_augmented = generator.generate(train_data, verbose=False)
    
    # 合并原始数据和增强数据
    X_train = np.concatenate([train_data, train_augmented], axis=0)
    y_train = np.concatenate([train_labels, train_labels], axis=0)
    
    # 将数据展平用于分类器
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = test_data.reshape(test_data.shape[0], -1)
    
    print(f"\n训练数据形状: {X_train_flat.shape}")
    print(f"测试数据形状: {X_test_flat.shape}")
    
    # 训练分类器
    print("\n训练分类器...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_flat, y_train)
    
    # 评估
    y_pred = clf.predict(X_test_flat)
    accuracy = accuracy_score(test_labels, y_pred)
    
    print(f"\n分类结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"\n详细报告:")
    print(classification_report(test_labels, y_pred))
    
    return clf, accuracy


def example_5_different_guidance_strengths():
    """示例 5: 比较不同的引导强度"""
    print("\n" + "=" * 80)
    print("示例 5: 比较不同引导强度的效果")
    print("=" * 80)
    
    test_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/test_data.npy")[:10]  # 只用10个样本测试
    
    configurations = [
        {"D": 8, "scale": 0.5, "name": "弱引导"},
        {"D": 8, "scale": 1.0, "name": "标准引导"},
        {"D": 4, "scale": 6.0, "name": "强引导"},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n测试配置: {config['name']} (D={config['D']}, scale={config['scale']})")
        
        generator = EEGGenerator(
            model_path="models/ema_0.9999_200000.pt",
            D=config['D'],
            scale=config['scale']
        )
        
        original, generated = generator.generate(test_data, verbose=False)
        
        # 计算相似度（示例：使用相关系数）
        similarity = np.mean([
            np.corrcoef(orig.flatten(), gen.flatten())[0, 1]
            for orig, gen in zip(original, generated)
        ])
        
        results[config['name']] = {
            'generated': generated,
            'similarity': similarity
        }
        
        print(f"  平均相似度: {similarity:.4f}")
    
    return results


def example_6_integrate_with_pytorch():
    """示例 6: 与 PyTorch 模型集成"""
    print("\n" + "=" * 80)
    print("示例 6: 与 PyTorch 深度学习模型集成")
    print("=" * 80)
    
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    
    # 自定义数据集（使用生成的数据）
    class AugmentedEEGDataset(Dataset):
        def __init__(self, original_data, generated_data, labels):
            self.data = np.concatenate([original_data, generated_data], axis=0)
            self.labels = np.concatenate([labels, labels], axis=0)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return (
                torch.FloatTensor(self.data[idx]),
                torch.LongTensor([self.labels[idx]])[0]
            )
    
    # 加载数据
    train_data = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_data.npy")
    train_labels = np.load("eeg_adapt/datasets/eegdata/bci2a/resub1234567/train_label.npy")
    
    # 生成增强数据
    generator = EEGGenerator(
        model_path="models/ema_0.9999_200000.pt",
        D=8,
        scale=1.0
    )
    
    print("\n生成训练数据增强...")
    _, train_augmented = generator.generate(train_data, verbose=False)
    
    # 创建数据集和数据加载器
    dataset = AugmentedEEGDataset(train_data, train_augmented, train_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"批次数: {len(dataloader)}")
    
    # 示例：训练一个简单的 CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(22, 64, kernel_size=5)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, 4)  # 4类分类
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN()
    print(f"\n模型: {model}")
    
    # 训练循环示例
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\n开始训练（示例：1个epoch）...")
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return model, dataloader


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EEG Generator 使用示例集合")
    print("=" * 80)
    print("\n请选择要运行的示例:")
    print("  1. 基本使用")
    print("  2. 类接口使用（多次生成）")
    print("  3. 批量生成（大数据集）")
    print("  4. 下游分类任务")
    print("  5. 比较不同引导强度")
    print("  6. 与 PyTorch 模型集成")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-6): ")
    
    if choice == "1":
        example_1_basic_usage()
    elif choice == "2":
        example_2_class_usage()
    elif choice == "3":
        example_3_batch_generation()
    elif choice == "4":
        example_4_downstream_classification()
    elif choice == "5":
        example_5_different_guidance_strengths()
    elif choice == "6":
        example_6_integrate_with_pytorch()
    elif choice == "0":
        print("退出")
    else:
        print("无效选项")

