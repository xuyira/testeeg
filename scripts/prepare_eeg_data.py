"""
准备bci2a数据集，分别保存train和test数据
"""

import numpy as np
import os
import argparse

def prepare_bci2a_data(data_dir="../datasets/eegdata/bci2a", subject_ids=None, resplit=False):
    """
    分别合并指定受试者的train_data_original和test_data
    保存为独立的train和test文件到特定子目录
    
    Args:
        data_dir: 数据目录
        subject_ids: 受试者编号列表，例如 [1, 2, 3] 或 None（默认处理全部1-9）
        resplit: 是否重新划分数据集（True: 80/20划分, False: 使用原始划分）
    """
    # 如果没有指定，默认处理所有受试者 1-9
    if subject_ids is None:
        subject_ids = list(range(1, 10))
    
    # 创建子目录名称（基于受试者编号和划分方式）
    subject_str = ''.join(map(str, subject_ids))
    prefix = "resub" if resplit else "sub"
    output_dir = os.path.join(data_dir, f"{prefix}{subject_str}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    all_labels = []
    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []
    
    print(f"准备处理的受试者: {subject_ids}")
    print(f"划分方式: {'80/20 重新划分' if resplit else '使用原始训练/测试划分'}")
    print(f"输出目录: {output_dir}\n")
    
    for subject_id in subject_ids:
        # 读取训练数据（从原始数据目录）
        train_data = np.load(os.path.join(data_dir, f"subject{subject_id}_train_data_original.npy"))
        train_label = np.load(os.path.join(data_dir, f"subject{subject_id}_train_label_original.npy"))
        
        # 读取测试数据
        test_data = np.load(os.path.join(data_dir, f"subject{subject_id}_test_data.npy"))
        test_label = np.load(os.path.join(data_dir, f"subject{subject_id}_test_label.npy"))
        
        print(f"Subject {subject_id}: train shape {train_data.shape}, test shape {test_data.shape}")
        
        if resplit:
            # 合并训练和测试数据
            combined_data = np.concatenate([train_data, test_data], axis=0)
            combined_labels = np.concatenate([train_label, test_label], axis=0)
            all_data.append(combined_data)
            all_labels.append(combined_labels)
        else:
            # 分别收集训练和测试数据
            all_train_data.append(train_data)
            all_train_labels.append(train_label)
            all_test_data.append(test_data)
            all_test_labels.append(test_label)
    
    if resplit:
        # 合并所有数据
        all_data_combined = np.concatenate(all_data, axis=0)
        all_labels_combined = np.concatenate(all_labels, axis=0)
        
        print(f"\n合并后总数据形状: {all_data_combined.shape}")
        
        # 随机打乱
        np.random.seed(42)  # 固定随机种子以便复现
        indices = np.random.permutation(len(all_data_combined))
        all_data_combined = all_data_combined[indices]
        all_labels_combined = all_labels_combined[indices]
        
        # 80/20 划分
        split_idx = int(len(all_data_combined) * 0.8)
        train_data_combined = all_data_combined[:split_idx]
        train_labels_combined = all_labels_combined[:split_idx]
        test_data_combined = all_data_combined[split_idx:]
        test_labels_combined = all_labels_combined[split_idx:]
        
        print(f"按 80/20 划分:")
        print(f"  训练集: {len(train_data_combined)} 样本 (80%)")
        print(f"  测试集: {len(test_data_combined)} 样本 (20%)")
    else:
        # 拼接训练数据
        train_data_combined = np.concatenate(all_train_data, axis=0)
        train_labels_combined = np.concatenate(all_train_labels, axis=0)
        
        # 拼接测试数据
        test_data_combined = np.concatenate(all_test_data, axis=0)
        test_labels_combined = np.concatenate(all_test_labels, axis=0)
    
    # 去掉多余的维度: (N, 1, 22, 1000) -> (N, 22, 1000)
    if train_data_combined.ndim == 4 and train_data_combined.shape[1] == 1:
        train_data_combined = train_data_combined.squeeze(1)
        print(f"训练数据去掉维度1后的shape: {train_data_combined.shape}")
    
    if test_data_combined.ndim == 4 and test_data_combined.shape[1] == 1:
        test_data_combined = test_data_combined.squeeze(1)
        print(f"测试数据去掉维度1后的shape: {test_data_combined.shape}")
    
    # 保存训练数据到子目录
    np.save(os.path.join(output_dir, "train_data.npy"), train_data_combined)
    np.save(os.path.join(output_dir, "train_label.npy"), train_labels_combined)
    
    # 保存测试数据到子目录
    np.save(os.path.join(output_dir, "test_data.npy"), test_data_combined)
    np.save(os.path.join(output_dir, "test_label.npy"), test_labels_combined)
    
    print(f"\n数据准备完成！")
    print(f"\n训练集:")
    print(f"  train_data.npy shape: {train_data_combined.shape}")
    print(f"  train_label.npy shape: {train_labels_combined.shape}")
    print(f"\n测试集:")
    print(f"  test_data.npy shape: {test_data_combined.shape}")
    print(f"  test_label.npy shape: {test_labels_combined.shape}")
    print(f"\n保存路径: {output_dir}")
    
    return output_dir  # 返回输出目录路径

def parse_subject_ids(subject_str):
    """
    解析受试者编号字符串
    
    支持格式:
        - "1,2,3" -> [1, 2, 3]
        - "1-3" -> [1, 2, 3]
        - "1,3-5,7" -> [1, 3, 4, 5, 7]
    """
    if not subject_str:
        return None
    
    subject_ids = []
    parts = subject_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 处理范围，例如 "1-3"
            start, end = map(int, part.split('-'))
            subject_ids.extend(range(start, end + 1))
        else:
            # 处理单个数字
            subject_ids.append(int(part))
    
    return sorted(list(set(subject_ids)))  # 去重并排序


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="准备 BCI2A 数据集，分离训练集和测试集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理所有受试者 (1-9)，使用原始划分
  python prepare_eeg_data.py
  
  # 处理指定受试者 (3 和 4)，使用原始划分 → 保存到 sub34/
  python prepare_eeg_data.py --subjects 3,4
  
  # 处理受试者 3,4，使用 80/20 重新划分 → 保存到 resub34/
  python prepare_eeg_data.py --subjects 3,4 --resplit
  
  # 处理受试者范围 (1 到 6)
  python prepare_eeg_data.py --subjects 1-6
  
  # 混合使用 + 80/20 划分 → 保存到 resub13457/
  python prepare_eeg_data.py --subjects 1,3-5,7 --resplit
  
  # 指定数据目录
  python prepare_eeg_data.py --data_dir /path/to/data --subjects 1,2,3
        """
    )
    
    parser.add_argument(
        '--subjects', '-s',
        type=str,
        default=None,
        help='受试者编号，支持逗号分隔和范围（例如: "1,2,3" 或 "1-6" 或 "1,3-5,7"），默认处理全部 1-9'
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default='datasets/eegdata/bci2a',
        help='数据目录路径（默认: datasets/eegdata/bci2a）'
    )
    
    parser.add_argument(
        '--resplit', '-r',
        action='store_true',
        help='重新划分数据集为 80%% 训练 / 20%% 测试（默认使用原始划分，保存到 sub* 目录；使用此选项保存到 resub* 目录）'
    )
    
    args = parser.parse_args()
    
    # 解析受试者编号
    subject_ids = parse_subject_ids(args.subjects)
    
    if subject_ids:
        print(f"指定受试者: {subject_ids}")
    else:
        print("使用默认设置: 处理所有受试者 (1-9)")
    
    # 执行数据准备
    prepare_bci2a_data(data_dir=args.data_dir, subject_ids=subject_ids, resplit=args.resplit)

