"""
合并bci2a数据集的train和test数据
"""

import numpy as np
import os

def prepare_bci2a_data(data_dir="../datasets/eegdata/bci2a", num_subjects=6):
    """
    合并subject 1-6的train_data_original和test_data
    """
    all_train_data = []
    all_train_labels = []
    
    for subject_id in range(1, num_subjects + 1):
        # 读取训练数据
        train_data = np.load(os.path.join(data_dir, f"subject{subject_id}_train_data_original.npy"))
        train_label = np.load(os.path.join(data_dir, f"subject{subject_id}_train_label_original.npy"))
        
        # 读取测试数据
        test_data = np.load(os.path.join(data_dir, f"subject{subject_id}_test_data.npy"))
        test_label = np.load(os.path.join(data_dir, f"subject{subject_id}_test_label.npy"))
        
        # 合并
        all_train_data.append(train_data)
        all_train_data.append(test_data)
        all_train_labels.append(train_label)
        all_train_labels.append(test_label)
        
        print(f"Subject {subject_id}: train shape {train_data.shape}, test shape {test_data.shape}")
    
    # 拼接所有数据
    combined_data = np.concatenate(all_train_data, axis=0)
    combined_labels = np.concatenate(all_train_labels, axis=0)
    
    # 去掉多余的维度: (N, 1, 22, 1000) -> (N, 22, 1000)
    if combined_data.ndim == 4 and combined_data.shape[1] == 1:
        combined_data = combined_data.squeeze(1)
        print(f"去掉维度1后的shape: {combined_data.shape}")
    
    # 保存
    np.save(os.path.join(data_dir, "train_data.npy"), combined_data)
    np.save(os.path.join(data_dir, "train_label.npy"), combined_labels)
    
    print(f"\n合并完成！")
    print(f"train_data.npy shape: {combined_data.shape}")
    print(f"train_label.npy shape: {combined_labels.shape}")
    print(f"保存路径: {data_dir}")

if __name__ == "__main__":
    prepare_bci2a_data()

