# ECG图像重建模型训练指南（生产版）

## 📋 方案总结

### ✅ 最终采用方案：重采样统一采样率

**核心策略**：
- 在数据加载时，将所有信号重采样到500Hz
- 无需预处理转NPY，直接从仿真器输出加载
- 使用固定长度U-Net模型（signal_length=5000）

**优势**：
1. ✅ **100%数据利用率** - 不浪费任何采样率的数据
2. ✅ **无需预处理** - 节省磁盘空间，灵活调试
3. ✅ **代码简单** - 重采样逻辑在Dataset内部
4. ✅ **训练稳定** - 固定长度输出，收敛快

**劣势**：
- ❌ 重采样引入轻微失真（对ECG影响很小）
- ❌ 首次加载稍慢（可用缓存解决）

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install opencv-python pandas numpy tqdm tensorboard albumentations

# 克隆代码（假设你已经有了）
cd /path/to/your/project
```

### 2. 修改配置

编辑 `train.sh`，设置数据路径：

```bash
SIM_ROOT="/path/to/your/simulations-V37"
CSV_ROOT="/path/to/your/train"
```

### 3. 一键训练

```bash
# 赋予执行权限
chmod +x train.sh

# 运行脚本
./train.sh
```

脚本会引导你完成：
1. 环境检查
2. 数据加载测试
3. 快速调试（3 epochs验证）
4. 完整训练（100 epochs）

---

## 📁 文件结构

```
your_project/
├── production_dataset.py      # 数据集加载器（重采样逻辑）
├── production_trainer.py      # 训练脚本
├── ecg_model.py              # 模型定义（U-Net）
├── ecg_trainer.py            # 损失函数
├── train.sh                  # 一键启动脚本
│
├── experiments/              # 训练输出目录
│   └── run_20241115_143022/
│       ├── checkpoints/
│       │   ├── best.pth
│       │   ├── last.pth
│       │   └── epoch_10.pth
│       ├── tensorboard/
│       ├── logs/
│       └── config.json
│
└── data/                     # 数据目录（需自行配置）
    ├── simulations-V37/      # 仿真器输出
    └── train/                # 原始CSV
```

---

## 🔧 命令行使用

### 快速调试（推荐首次使用）

```bash
python production_trainer.py \
    --sim_root /path/to/simulations \
    --csv_root /path/to/train \
    --batch_size 2 \
    --num_workers 0 \
    --debug  # 只用100样本，3 epochs
```

### 完整训练（Mac M2）

```bash
python production_trainer.py \
    --sim_root /path/to/simulations \
    --csv_root /path/to/train \
    --batch_size 4 \
    --num_workers 2 \
    --epochs 100 \
    --pretrained  # 使用ImageNet预训练权重
```

### 完整训练（GPU）

```bash
python production_trainer.py \
    --sim_root /path/to/simulations \
    --csv_root /path/to/train \
    --batch_size 16 \
    --num_workers 8 \
    --epochs 100 \
    --pretrained
```

### 恢复训练

```bash
python production_trainer.py \
    --sim_root /path/to/simulations \
    --csv_root /path/to/train \
    --resume ./experiments/run_xxx/checkpoints/last.pth
```

---

## 📊 监控训练

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir ./experiments/run_20241115_143022/tensorboard --port 6006

# 浏览器打开
http://localhost:6006
```

**关键指标**：
- `Train/signal_loss` - 信号重建损失（最重要）
- `Val/signal_loss` - 验证集信号损失
- `Train/total_loss` - 总损失
- `Train/lr` - 学习率变化

**健康的训练曲线**：
```
Train Loss:  3.0 → 1.5 → 0.8 → 0.5  (持续下降)
Val Loss:    3.2 → 1.8 → 1.0 → 0.9  (前期下降，后期平稳)
Signal Loss: 2.0 → 0.8 → 0.3 → 0.2  (最重要，应降到0.2以下)
```

---

## ⚙️ 超参数调优

### 学习率

```bash
# 默认: 1e-4
--lr 1e-4

# 如果loss不下降，尝试降低
--lr 5e-5

# 如果收敛太慢，尝试提高
--lr 2e-4
```

### Batch Size

| 设备 | 推荐Batch Size | 说明 |
|------|---------------|------|
| Mac M2 8GB | 2-4 | 内存限制 |
| Mac M2 Pro 16GB | 4-8 | 较宽松 |
| RTX 3090 24GB | 16-32 | 可用大batch |
| A100 40GB | 32-64 | 最优效率 |

### 损失权重

```bash
# 默认权重（适用于大多数情况）
--loss_weight_seg 1.0 \
--loss_weight_grid 0.5 \
--loss_weight_baseline 0.8 \
--loss_weight_theta 0.3 \
--loss_weight_signal 2.0

# 如果信号重建不好，提高signal权重
--loss_weight_signal 3.0

# 如果导联分割不准，提高seg权重
--loss_weight_seg 1.5
```

---

## 🐛 常见问题

### Q1: 内存溢出（OOM）

**症状**：
```
RuntimeError: CUDA out of memory
RuntimeError: MPS backend out of memory
```

**解决**：
```bash
# 方案1: 减小batch size
--batch_size 2

# 方案2: 减小图像尺寸（修改Dataset）
target_size=(384, 504)  # 从512×672降到384×504

# 方案3: 禁用预训练（减少模型大小）
# 移除 --pretrained 参数

# 方案4: 使用CPU
--force_cpu
```

### Q2: 数据加载慢

**症状**: 进度条频繁卡顿，GPU利用率低

**解决**：
```bash
# 方案1: 增加worker
--num_workers 8

# 方案2: 启用内存缓存（需要足够RAM）
--cache

# 方案3: 使用SSD存储数据
```

### Q3: Loss不下降

**可能原因**：

1. **学习率过大**
   ```bash
   --lr 5e-5  # 降低学习率
   ```

2. **数据有问题**
   ```bash
   # 先运行数据验证
   python production_dataset.py \
       --sim_root /path/to/simulations \
       --csv_root /path/to/train \
       --max_samples 100
   ```

3. **模型初始化问题**
   ```bash
   --pretrained  # 使用预训练权重
   ```

### Q4: 训练太慢

**优化策略**：

| 方法 | 加速比 | 说明 |
|------|--------|------|
| 增加num_workers | 1.5-2x | 并行数据加载 |
| 使用pin_memory | 1.1-1.2x | 加速GPU传输 |
| 启用内存缓存 | 2-3x | 需要足够RAM |
| 混合精度训练 | 1.5-2x | 需要CUDA，降低精度 |

---

## 📈 性能基准

### 预期结果（100 epochs）

| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| Total Loss | 0.5-0.8 | 0.8-1.2 |
| Signal Loss | 0.2-0.3 | 0.3-0.5 |
| Pearson Corr | 0.88-0.92 | 0.85-0.90 |

### 训练时间估算

| 硬件 | Batch Size | 10K样本/epoch | 100K样本/epoch |
|------|-----------|---------------|---------------|
| Mac M2 | 2 | ~2小时 | ~20小时 |
| Mac M2 Pro | 4 | ~1小时 | ~10小时 |
| RTX 3090 | 16 | ~15分钟 | ~2.5小时 |
| A100 | 32 | ~8分钟 | ~1.3小时 |

---

## 🔬 进阶功能

### 1. 自定义损失权重

创建配置文件 `config.json`：

```json
{
  "loss_weights": {
    "seg": 1.2,
    "grid": 0.4,
    "baseline": 1.0,
    "theta": 0.2,
    "signal": 3.0
  }
}
```

### 2. 数据增强（可选）

如果需要在仿真数据基础上再做增强，修改 `production_dataset.py`：

```python
self.transform = A.Compose([
    A.RandomBrightnessContrast(p=0.3),  # 亮度对比度
    A.GaussNoise(var_limit=(5, 15), p=0.2),  # 高斯噪声
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### 3. 混合精度训练（仅CUDA）

修改 `production_trainer.py`，在训练循环中添加：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环中
with autocast():
    outputs = self.model(images)
    losses = self.criterion(outputs, targets)

scaler.scale(losses['total']).backward()
scaler.step(self.optimizer)
scaler.update()
```

---

## 🎯 下一步

训练完成后：

1. **评估模型**
   ```bash
   python evaluate.py --checkpoint best.pth
   ```

2. **推理测试**
   ```bash
   python inference.py --image test.png --output result.csv
   ```

3. **可视化结果**
   ```bash
   python visualize.py --image test.png --checkpoint best.pth
   ```

（这些脚本需要另外实现）

---

## 📞 技术支持

如果遇到问题：

1. 检查 `experiments/run_xxx/logs/` 下的日志文件
2. 查看 `config.json` 确认配置正确
3. 运行 `--debug` 模式排查问题
4. 检查TensorBoard确认训练曲线正常

---

## 📝 更新日志

- **v1.0** (2024-11-15)
  - ✅ 重采样支持多采样率
  - ✅ 直接从仿真器输出加载
  - ✅ 完善的错误处理和日志
  - ✅ 一键启动脚本
  - ✅ TensorBoard可视化

---

祝训练顺利！🚀