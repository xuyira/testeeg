"""
EEG Conformer with Data Augmentation using Diffusion Generator

改进点:
1. 支持指定单个受试者测试（通过命令行参数）
2. 集成 EEG 扩散生成器进行数据增强
3. 将原始数据和生成数据在通道维度拼接（22 → 44通道）
"""

import argparse
import os
import numpy as np
import math
import random
import datetime
import time
import scipy.io
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.utils.data import Dataset
from torch import Tensor
from torch.autograd import Variable

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 EEG Generator
from eeg_adapt.eeg_generator import EEGGenerator


# ==================== 模型组件 ====================

class PatchEmbedding(nn.Module):
    """支持可变通道数的 Patch Embedding"""
    def __init__(self, in_channels=22, emb_size=40):
        super().__init__()
        
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (in_channels, 1), (1, 1)),  # 动态通道数
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    """支持可变通道数的 Conformer"""
    def __init__(self, in_channels=22, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


# ==================== 实验类 ====================

class ExP():
    def __init__(self, test_subject, total_sub=9, use_generator=True, 
                 model_path=None, gen_D=8, gen_scale=1.0, gen_N=None):
        """
        Args:
            test_subject: 测试受试者编号 (1-9)
            total_sub: 总受试者数量
            use_generator: 是否使用扩散生成器增强
            model_path: 扩散模型路径
            gen_D: 生成器 D 参数
            gen_scale: 生成器 scale 参数
            gen_N: 生成器 N 参数（起始时间步）
        """
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 1000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.test_subject = test_subject
        self.total_sub = total_sub
        
        self.use_generator = use_generator
        self.gen_D = gen_D
        self.gen_scale = gen_scale
        self.gen_N = gen_N
        
        # 早停参数
        self.patience = 200
        self.min_delta = 0.0001

        self.start_epoch = 0
        self.root = './data/standard_2a_data/'

        # 创建结果目录
        os.makedirs('./results', exist_ok=True)
        self.log_write = open(f"./results/log_subject{self.test_subject}.txt", "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        # 根据是否使用生成器决定输入通道数
        in_channels = 44 if use_generator else 22
        self.model = Conformer(in_channels=in_channels).cuda()
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        
        # 初始化生成器
        if use_generator:
            if model_path is None:
                raise ValueError("使用生成器时必须提供 model_path")
            print(f"\n{'='*60}")
            print(f"🚀 初始化 EEG 扩散生成器")
            print(f"   模型路径: {model_path}")
            print(f"   D={gen_D}, scale={gen_scale}, N={gen_N}")
            print(f"   输入通道: {in_channels} (22原始 + 22生成)")
            print(f"{'='*60}\n")
            
            self.generator = EEGGenerator(
                model_path=model_path,
                D=gen_D,
                scale=gen_scale,
                N=gen_N,
                device='cuda',
                image_size=64,
                in_channels=22,
                diffusion_steps=1000,
                noise_schedule='cosine'
            )
        else:
            self.generator = None
            print(f"\n⚠️  不使用生成器，标准 22 通道模式\n")

    def augment_with_generator(self, data):
        """
        使用扩散生成器增强数据
        
        Args:
            data: (batch, 1, 22, 1000) numpy array
        
        Returns:
            augmented_data: (batch, 1, 44, 1000) torch tensor
        """
        if not self.use_generator or self.generator is None:
            # 不使用生成器，直接返回原始数据
            return torch.from_numpy(data).cuda().float()
        
        # 移除 channel 维度: (batch, 1, 22, 1000) -> (batch, 22, 1000)
        data_squeezed = data.squeeze(1)
        
        # 生成数据
        with torch.no_grad():
            original, generated = self.generator.generate(
                data_squeezed, 
                verbose=False
            )
        
        # 拼接原始和生成: (batch, 22, 1000) + (batch, 22, 1000) -> (batch, 44, 1000)
        concatenated = np.concatenate([original, generated], axis=1)
        
        # 添加 channel 维度: (batch, 44, 1000) -> (batch, 1, 44, 1000)
        concatenated = np.expand_dims(concatenated, axis=1)
        
        # 转换为 tensor
        return torch.from_numpy(concatenated).cuda().float()

    def interaug(self, timg, label):
        """S&R 数据增强"""
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        # 应用生成器增强
        aug_data = self.augment_with_generator(aug_data)
        
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        """加载 LOSO 数据"""
        train_data_list = []
        train_label_list = []
        
        # 加载训练数据（除测试受试者外的所有受试者）
        for subject_id in range(1, self.total_sub + 1):
            if subject_id == self.test_subject:
                continue
            
            for data_type in ['T', 'E']:
                file_path = self.root + 'A0%d%s.mat' % (subject_id, data_type)
                print(f'Loading training data: {file_path}')
                
                data_mat = scipy.io.loadmat(file_path)
                data = data_mat['data']
                label = data_mat['label']
                
                data = np.transpose(data, (2, 1, 0))
                data = np.expand_dims(data, axis=1)
                label = np.transpose(label)
                
                train_data_list.append(data)
                train_label_list.append(label[0])
        
        self.allData = np.concatenate(train_data_list, axis=0)
        self.allLabel = np.concatenate(train_label_list, axis=0)
        
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]
        
        # 加载测试数据
        test_data_list = []
        test_label_list = []
        
        for data_type in ['T', 'E']:
            file_path = self.root + 'A0%d%s.mat' % (self.test_subject, data_type)
            print(f'Loading test data: {file_path}')
            
            data_mat = scipy.io.loadmat(file_path)
            data = data_mat['data']
            label = data_mat['label']
            
            data = np.transpose(data, (2, 1, 0))
            data = np.expand_dims(data, axis=1)
            label = np.transpose(label)
            
            test_data_list.append(data)
            test_label_list.append(label[0])
        
        self.testData = np.concatenate(test_data_list, axis=0)
        self.testLabel = np.concatenate(test_label_list, axis=0)
        
        # 标准化
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        
        print(f'Training data shape: {self.allData.shape}, Training labels shape: {self.allLabel.shape}')
        print(f'Test data shape: {self.testData.shape}, Test labels shape: {self.testLabel.shape}')
        
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        # 训练数据增强（应用生成器）
        print(f"\n🔄 对训练数据应用生成器增强...")
        img_augmented = self.augment_with_generator(img)
        print(f"   增强后训练数据形状: {img_augmented.shape}")
        
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img_augmented, label)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True
        )

        # 测试数据增强（应用生成器）
        print(f"🔄 对测试数据应用生成器增强...")
        test_data_augmented = self.augment_with_generator(test_data)
        print(f"   增强后测试数据形状: {test_data_augmented.shape}\n")
        
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data_augmented, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=True
        )

        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50, 
            verbose=True, min_lr=1e-6
        )

        test_data_augmented = Variable(test_data_augmented.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        
        best_epoch = 0
        epochs_no_improve = 0

        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # 数据增强 (S&R增强 + 生成器增强)
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 测试
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data_augmented)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                
                self.scheduler.step(acc)
                current_lr = self.optimizer.param_groups[0]['lr']

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc,
                      '  LR: %.8f' % current_lr)

                self.log_write.write(str(e) + "    " + str(acc) + "    LR: " + str(current_lr) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                
                # 早停
                if acc > bestAcc + self.min_delta:
                    bestAcc = acc
                    best_epoch = e
                    epochs_no_improve = 0
                    Y_true = test_label
                    Y_pred = y_pred
                    torch.save(self.model.module.state_dict(), f'best_model_subject{self.test_subject}.pth')
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.patience:
                    print(f'Early stopping triggered! No improvement for {self.patience} epochs.')
                    print(f'Best accuracy {bestAcc:.6f} at epoch {best_epoch}')
                    break

        # 保存最终模型
        torch.save(self.model.module.state_dict(), f'final_model_subject{self.test_subject}.pth')
        print(f'\nTraining completed at epoch {e}')
        print(f'Best model was at epoch {best_epoch} with accuracy {bestAcc:.6f}')
        
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.log_write.write('Best epoch: ' + str(best_epoch) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    parser = argparse.ArgumentParser(description="EEG Conformer with Diffusion Generator")
    
    # 受试者参数
    parser.add_argument('--test_subject', type=int, default=None,
                       help='指定测试受试者编号 (1-9)。如果不指定，则测试所有受试者')
    parser.add_argument('--total_subjects', type=int, default=9,
                       help='总受试者数量')
    
    # 生成器参数
    parser.add_argument('--use_generator', action='store_true', default=False,
                       help='是否使用扩散生成器增强数据')
    parser.add_argument('--generator_model', type=str, default=None,
                       help='扩散模型路径（使用生成器时必须提供）')
    parser.add_argument('--gen_D', type=int, default=8,
                       help='生成器 D 参数（频率引导下采样倍数）')
    parser.add_argument('--gen_scale', type=float, default=1.0,
                       help='生成器 scale 参数（频率引导强度）')
    parser.add_argument('--gen_N', type=int, default=None,
                       help='生成器 N 参数（ILVR起始时间步，None表示从头开始）')
    
    # GPU 设置
    parser.add_argument('--gpus', type=str, default='0',
                       help='使用的 GPU 编号，用逗号分隔，如 "0,1"')
    
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # 设置随机种子
    seed_n = np.random.randint(2021)
    print(f'🎲 Random seed: {seed_n}')
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    
    # 确定测试受试者列表
    if args.test_subject is not None:
        # 测试单个受试者
        test_subjects = [args.test_subject]
        print(f"\n{'='*60}")
        print(f"🎯 单受试者测试模式")
        print(f"   测试受试者: Subject {args.test_subject}")
        print(f"{'='*60}\n")
    else:
        # 测试所有受试者
        test_subjects = list(range(1, args.total_subjects + 1))
        print(f"\n{'='*60}")
        print(f"🎯 全受试者测试模式 (LOSO)")
        print(f"   总受试者数: {args.total_subjects}")
        print(f"{'='*60}\n")
    
    # 结果记录
    result_write = open("./results/sub_result.txt", "w")
    result_write.write(f"Seed: {seed_n}\n")
    result_write.write(f"Use Generator: {args.use_generator}\n")
    if args.use_generator:
        result_write.write(f"Generator Model: {args.generator_model}\n")
        result_write.write(f"Generator D: {args.gen_D}, Scale: {args.gen_scale}, N: {args.gen_N}\n")
    result_write.write(f"\n{'='*60}\n\n")
    
    best_acc_sum = 0
    aver_acc_sum = 0
    all_Y_true = []
    all_Y_pred = []
    
    # 测试每个受试者
    for subject_id in test_subjects:
        starttime = datetime.datetime.now()
        
        print(f"\n{'='*60}")
        print(f"📊 Training Subject {subject_id}")
        print(f"{'='*60}")
        
        exp = ExP(
            test_subject=subject_id,
            total_sub=args.total_subjects,
            use_generator=args.use_generator,
            model_path=args.generator_model,
            gen_D=args.gen_D,
            gen_scale=args.gen_scale,
            gen_N=args.gen_N
        )
        
        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        
        print(f'\n✅ Subject {subject_id} - Best Accuracy: {bestAcc:.6f}')
        
        result_write.write(f'Subject {subject_id}:\n')
        result_write.write(f'  Best accuracy: {bestAcc:.6f}\n')
        result_write.write(f'  Average accuracy: {averAcc:.6f}\n')
        
        endtime = datetime.datetime.now()
        duration = endtime - starttime
        print(f'⏱️  Duration: {duration}')
        result_write.write(f'  Duration: {duration}\n\n')
        
        best_acc_sum += bestAcc
        aver_acc_sum += averAcc
        all_Y_true.append(Y_true)
        all_Y_pred.append(Y_pred)
    
    # 计算总体结果
    num_subjects = len(test_subjects)
    avg_best = best_acc_sum / num_subjects
    avg_aver = aver_acc_sum / num_subjects
    
    print(f"\n{'='*60}")
    print(f"🎉 Final Results")
    print(f"{'='*60}")
    print(f"Average Best Accuracy:  {avg_best:.6f}")
    print(f"Average Aver Accuracy:  {avg_aver:.6f}")
    print(f"{'='*60}\n")
    
    result_write.write(f"\n{'='*60}\n")
    result_write.write(f"Final Results:\n")
    result_write.write(f"  Average Best Accuracy: {avg_best:.6f}\n")
    result_write.write(f"  Average Aver Accuracy: {avg_aver:.6f}\n")
    result_write.write(f"{'='*60}\n")
    result_write.close()


if __name__ == "__main__":
    print(f"⏰ Start time: {time.asctime(time.localtime(time.time()))}")
    main()
    print(f"⏰ End time: {time.asctime(time.localtime(time.time()))}")

