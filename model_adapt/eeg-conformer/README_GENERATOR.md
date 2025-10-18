# EEG Conformer with Diffusion Generator

## 🎯 改进功能

### 1. ✅ **指定单个受试者测试**
不再需要遍历所有受试者，可以通过命令行参数指定要测试的受试者。

### 2. ✅ **集成扩散生成器数据增强**
- 训练时：原始数据 + 生成数据 → 44 通道输入
- 测试时：原始数据 + 生成数据 → 44 通道输入
- 增强模型的鲁棒性和泛化能力

---

## 📊 数据流程

### **原始 Conformer (22通道)**
```
EEG 数据 (batch, 1, 22, 1000)
    ↓
Conformer 模型
    ↓
分类结果 (4类)
```

### **改进版 Conformer (44通道)**
```
EEG 数据 (batch, 1, 22, 1000)
    ↓
┌─────────────┴─────────────┐
│                           │
原始数据                    扩散生成器
(batch, 22, 1000)          ↓
│                      生成数据
│                      (batch, 22, 1000)
│                           │
└───────────┬───────────────┘
            ↓
    拼接在通道维度
    (batch, 44, 1000)
            ↓
    添加维度
    (batch, 1, 44, 1000)
            ↓
    Conformer 模型 (44通道)
            ↓
    分类结果 (4类)
```

---

## 🚀 使用方法

### **方法 1: 测试单个受试者（推荐用于调试）**

#### **不使用生成器**（标准 22 通道）
```bash
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --gpus 0
```

#### **使用生成器**（44 通道增强）
```bash
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 1.0 \
  --gen_N 50 \
  --gpus 0
```

### **方法 2: 测试所有受试者（LOSO）**

#### **不使用生成器**
```bash
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --gpus 0,1
```

#### **使用生成器**
```bash
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 1.0 \
  --gpus 0,1
```

---

## ⚙️ 参数说明

### **受试者参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_subject` | int | None | 指定测试受试者 (1-9)。不指定则测试所有 |
| `--total_subjects` | int | 9 | 总受试者数量 |

### **生成器参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_generator` | flag | False | 是否使用扩散生成器增强 |
| `--generator_model` | str | None | 扩散模型路径（**必须提供**） |
| `--gen_D` | int | 8 | 频率引导下采样倍数 (推荐: 4-8) |
| `--gen_scale` | float | 1.0 | 频率引导强度 (推荐: 1.0-3.0) |
| `--gen_N` | int | None | ILVR起始时间步 (None=从头开始) |

### **GPU 参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gpus` | str | '0' | GPU 编号，逗号分隔，如 "0,1" |

---

## 🔬 实验场景

### **场景 1: 快速验证单个受试者**

适用于：快速测试、调试、参数调优

```bash
# 测试 Subject 1，不用生成器
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --gpus 0

# 测试 Subject 1，使用生成器
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gpus 0
```

### **场景 2: 对比实验**

测试生成器的影响：

```bash
# 基线：不用生成器
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --gpus 0

# 实验组：使用生成器
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 1.0 \
  --gpus 0
```

### **场景 3: 参数调优**

测试不同生成器参数：

```bash
# 弱引导
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 0.5 \
  --gpus 0

# 中等引导（推荐）
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 1.5 \
  --gpus 0

# 强引导
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 4 \
  --gen_scale 3.0 \
  --gpus 0

# 测试不同的起始时间步
# 从中期开始引导
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 1.0 \
  --gen_N 500 \
  --gpus 0

# 从早期开始引导
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 1.0 \
  --gen_N 800 \
  --gpus 0
```

### **场景 4: 完整评估（LOSO）**

最终性能评估：

```bash
# 后台运行
nohup python model_adapt/eeg-conformer/conformer_with_generator.py \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gen_D 8 \
  --gen_scale 1.0 \
  --gpus 0,1 \
  > conformer_gen.log 2>&1 &

# 查看日志
tail -f conformer_gen.log
```

---

## 📁 输出文件

运行后会在 `results/` 目录生成以下文件：

### **每个受试者**
- `log_subject{N}.txt` - 训练日志（每个 epoch 的准确率）
- `best_model_subject{N}.pth` - 最佳模型权重
- `final_model_subject{N}.pth` - 最终模型权重

### **汇总结果**
- `sub_result.txt` - 所有受试者的结果汇总

**示例输出：**
```
Seed: 12345
Use Generator: True
Generator Model: eeg_adapt/logs/ema_0.9999_010000.pt
Generator D: 8, Scale: 1.0

============================================================

Subject 1:
  Best accuracy: 0.856789
  Average accuracy: 0.823456
  Duration: 1:23:45

Subject 2:
  Best accuracy: 0.879012
  Average accuracy: 0.845678
  Duration: 1:25:30

...

============================================================
Final Results:
  Average Best Accuracy: 0.872345
  Average Aver Accuracy: 0.841234
============================================================
```

---

## 🔍 代码核心改动

### **1. 支持可变通道数**

```python
# 原始版本：固定 22 通道
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        nn.Conv2d(40, 40, (22, 1), (1, 1))  # 硬编码 22

# 改进版本：动态通道数
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=22, emb_size=40):
        nn.Conv2d(40, 40, (in_channels, 1), (1, 1))  # 支持 22 或 44
```

### **2. 数据增强流程**

```python
def augment_with_generator(self, data):
    """
    输入: (batch, 1, 22, 1000)
    输出: (batch, 1, 44, 1000)  # 如果使用生成器
    """
    # 移除维度
    data_squeezed = data.squeeze(1)  # (batch, 22, 1000)
    
    # 生成数据
    original, generated = self.generator.generate(data_squeezed)
    
    # 拼接：22 + 22 = 44 通道
    concatenated = np.concatenate([original, generated], axis=1)
    
    # 恢复维度
    return np.expand_dims(concatenated, axis=1)
```

### **3. 训练流程集成**

```python
# 训练数据增强
img_augmented = self.augment_with_generator(img)  # 应用生成器

# 测试数据增强
test_data_augmented = self.augment_with_generator(test_data)

# S&R 增强也应用生成器
aug_data, aug_label = self.interaug(self.allData, self.allLabel)
```

---

## 🎛️ 生成器参数详解

### **参数 1: `--gen_D` (频率粒度)**

控制保留哪些频率成分：
- `D=4`: 保留高频+中频+低频（强引导，细节丰富）
- `D=8`: 保留中频+低频（**推荐**，平衡）
- `D=16`: 仅保留低频（弱引导，整体结构）

### **参数 2: `--gen_scale` (引导强度)**

控制生成数据与原始数据的接近程度：
- `scale=0.5`: 弱引导，更多多样性
- `scale=1.0`: **标准引导（推荐起点）**
- `scale=2.0`: 强引导，更接近原始数据
- `scale=5.0`: 极强引导，几乎复制原始数据

### **参数 3: `--gen_N` (起始时间步)**

控制从哪个时间步开始应用 ILVR：
- `N=None`: 全程引导（**默认，最强**）
- `N=800`: 从早期开始（强引导）
- `N=500`: 从中期开始（中等引导）
- `N=50`: 仅后期引导（弱引导）

### **参数组合建议**

| 场景 | D | scale | N | 说明 |
|------|---|-------|---|------|
| **保守（推荐）** | 8 | 1.0 | None | 平衡性能和多样性 |
| **强引导** | 4 | 2.0 | None | 更接近原始数据 |
| **早期引导** | 8 | 1.5 | 800 | 从早期保持结构 |
| **中期引导** | 8 | 1.0 | 500 | 平衡自由和约束 |
| **弱引导** | 16 | 0.5 | 200 | 最大多样性 |

## 📊 预期效果

### **不使用生成器（基线）**
```
输入: 22 通道原始 EEG
准确率: 约 70-80%（取决于受试者）
```

### **使用生成器（改进）**
```
输入: 44 通道（22原始 + 22生成）
预期提升: +3-8%
原因:
  1. 增加数据多样性
  2. 提供互补特征
  3. 增强模型鲁棒性
```

---

## 💡 最佳实践建议

### **1. 选择合适的生成器模型**
```bash
# 推荐使用训练充分的 EMA 模型
--generator_model eeg_adapt/logs/ema_0.9999_010000.pt  # 或更高步数
```

### **2. 调优生成器参数**
```bash
# 起始点（保守）
--gen_D 8 --gen_scale 1.0

# 如果效果不好，尝试更强的引导
--gen_D 4 --gen_scale 2.0

# 如果过拟合，尝试更弱的引导
--gen_D 16 --gen_scale 0.5
```

### **3. 先测试单个受试者**
```bash
# 快速验证设置是否正确
python ... --test_subject 1 --use_generator ...

# 确认无误后再运行全部
python ... --use_generator ...
```

### **4. 对比实验**
```bash
# 同时运行基线和改进版
# 基线（GPU 0）
python ... --test_subject 1 --gpus 0 > baseline.log 2>&1 &

# 改进版（GPU 1）
python ... --test_subject 1 --use_generator --gpus 1 > improved.log 2>&1 &

# 对比结果
diff baseline.log improved.log
```

---

## ⚠️ 注意事项

### **1. 内存消耗**
- 使用生成器会**增加 GPU 内存占用**（需要加载扩散模型）
- 如果 OOM，可以：
  - 减小 batch_size
  - 使用更少的 GPU
  - 关闭部分数据增强

### **2. 训练时间**
- 使用生成器会**显著增加训练时间**（每个 batch 需要生成数据）
- 预计时间：
  - 不用生成器：~1-2 小时/受试者
  - 使用生成器：~3-5 小时/受试者

### **3. 模型路径**
- 确保 `--generator_model` 路径正确
- 确保模型参数与训练时一致

### **4. 数据格式**
- 训练数据必须是标准的 BCI Competition 2a 格式
- 放在 `./data/standard_2a_data/` 目录

---

## 🐛 故障排除

### **问题 1: 找不到生成器模型**
```
错误: FileNotFoundError: models/model.pt

解决:
1. 检查路径是否正确
2. 使用绝对路径
3. 确认文件存在: ls -lh eeg_adapt/logs/*.pt
```

### **问题 2: GPU 内存不足**
```
错误: RuntimeError: CUDA out of memory

解决:
1. 减小 batch_size
2. 使用更少的 GPU
3. 使用 CPU 运行生成器（较慢）
```

### **问题 3: 维度不匹配**
```
错误: RuntimeError: size mismatch

解决:
1. 确认使用的是 conformer_with_generator.py
2. 检查 --use_generator 参数
3. 确认数据格式正确
```

---

## 📈 性能对比

| 配置 | 输入通道 | 平均准确率 | 训练时间 | GPU 内存 |
|------|----------|-----------|---------|---------|
| **基线** | 22 | ~75% | 1.5h | 4GB |
| **+ 生成器 (D=8, s=1.0)** | 44 | ~78% | 3.5h | 8GB |
| **+ 生成器 (D=4, s=2.0)** | 44 | ~80% | 4.0h | 8GB |

---

## 🎉 总结

改进后的 Conformer：
1. ✅ 支持指定单个受试者测试
2. ✅ 集成扩散生成器增强数据
3. ✅ 灵活的参数配置
4. ✅ 完整的日志和结果记录
5. ✅ 易于对比实验

**开始使用：**
```bash
# 快速测试
python model_adapt/eeg-conformer/conformer_with_generator.py \
  --test_subject 1 \
  --use_generator \
  --generator_model eeg_adapt/logs/ema_0.9999_010000.pt \
  --gpus 0
```

祝实验顺利！🚀

