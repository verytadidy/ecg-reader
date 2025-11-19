# ECG Simulator V46-Ultimate 完整文档

## 📋 目录

1. [函数调用拓扑图](#函数调用拓扑图)
2. [数据结构说明](#数据结构说明)
3. [核心函数详解](#核心函数详解)
4. [使用指南](#使用指南)
5. [迭代更新指南](#迭代更新指南)

---

## 🔄 函数调用拓扑图

### 主流程拓扑

```
main()
  │
  ├─> argparse.ArgumentParser()                    # 命令行参数解析
  │
  ├─> pd.read_csv(TRAIN_CSV)                       # 加载元数据
  │
  ├─> 生成任务列表: [(ecg_id, var_idx), ...]
  │
  ├─> multiprocessing.Pool()                       # 多进程执行
  │     │
  │     └─> process_one_id_ultimate()              # 🔥 单任务处理入口
  │           │
  │           ├─> pd.read_csv(csv_path)            # 加载ECG信号
  │           │
  │           ├─> sample_physical_params()         # 采样物理参数
  │           │
  │           ├─> render_clean_ecg_ultimate()      # 🔥 主渲染函数
  │           │     │
  │           │     ├─> generate_paper_texture()   # 生成纸张纹理
  │           │     │
  │           │     ├─> render_layout_3x4_plus_II_ultimate()  # 🔥 布局渲染
  │           │     │     │
  │           │     │     ├─> render_calibration_pulse()      # 定标脉冲
  │           │     │     │
  │           │     │     ├─> render_lead_text()              # 导联文字
  │           │     │     │
  │           │     │     ├─> draw_lead_separator()           # 分隔符
  │           │     │     │
  │           │     │     └─> cv2.polylines()                 # 绘制波形
  │           │     │
  │           │     └─> 图像融合 (Alpha Blending)
  │           │
  │           ├─> generate_scanner_background()    # 扫描仪背景
  │           │
  │           ├─> apply_degradation_pipeline_ultimate()  # 🔥 退化管道
  │           │     │
  │           │     ├─> add_printer_halftone()     # 打印半调
  │           │     ├─> add_screen_moire()         # 屏幕摩尔纹
  │           │     ├─> add_stains()               # 污渍
  │           │     ├─> add_motion_blur()          # 运动模糊
  │           │     ├─> add_jpeg_compression()     # JPEG压缩
  │           │     └─> 几何变换 (旋转+透视)
  │           │
  │           ├─> transform_bbox()                 # 坐标变换
  │           │
  │           ├─> save_ground_truth_signals()      # 保存真值信号
  │           │
  │           └─> create_metadata_ultimate()       # 生成元数据
  │
  └─> validate_sample_output()                     # 验证输出
```

---

## 📦 数据结构说明

### 1. 输出文件结构

```
output_dir/
└── {ecg_id}_v{var_idx:02d}_{layout_type}_{degradation_type}/
    ├── {id}_dirty.png                    # RGB图像 (H, W, 3) uint8
    ├── {id}_label_baseline.npy           # 基线热图 (12, H, W) uint8
    ├── {id}_label_text_multi.npy         # 文字掩码 (13, H, W) uint8
    ├── {id}_label_wave.npy               # 波形掩码 (H, W) uint8
    ├── {id}_label_auxiliary.npy          # 辅助掩码 (1, H, W) uint8
    ├── {id}_label_paper_speed.npy        # 纸速掩码 (1, H, W) uint8
    ├── {id}_label_gain.npy               # 增益掩码 (1, H, W) uint8
    ├── {id}_gt_signals.json              # 真值信号 JSON
    └── {id}_metadata.json                # 元数据 JSON
```

**示例文件名：**
```
262_v00_3x4+1_0005_dirty.png
262_v00_3x4+1_0005_label_baseline.npy
...
```

### 2. NPY掩码详细说明

#### 2.1 `label_baseline.npy`
```python
shape: (12, H, W)
dtype: uint8
value: 0-255

# 通道索引对应导联
channel_map = {
    0: 'I',    1: 'II',   2: 'III',
    3: 'aVR',  4: 'aVL',  5: 'aVF',
    6: 'V1',   7: 'V2',   8: 'V3',
    9: 'V4',  10: 'V5',  11: 'V6'
}

# 用途: 标记每个导联的基线位置（水平线）
# 值越大表示基线置信度越高
```

#### 2.2 `label_text_multi.npy`
```python
shape: (13, H, W)
dtype: uint8
value: 0-255

# 通道语义
channels = {
    0: 'background',      # 背景（自动生成 = 255 - sum(1:12)）
    1: 'lead_I',          # 导联I的文字
    2: 'lead_II',
    3: 'lead_III',
    4: 'lead_aVR',
    5: 'lead_aVL',
    6: 'lead_aVF',
    7: 'lead_V1',
    8: 'lead_V2',
    9: 'lead_V3',
    10: 'lead_V4',
    11: 'lead_V5',
    12: 'lead_V6'
}

# 用途: 精细分割每个导联的文字标签位置
# 训练渐进式模型的"细层"任务
```

#### 2.3 `label_wave.npy`
```python
shape: (H, W)
dtype: uint8
value: 1-12 (0=背景)

# 语义编码
pixel_value_map = {
    0: 'background',
    1: 'lead_I',
    2: 'lead_II',
    3: 'lead_III',
    4: 'lead_aVR',
    5: 'lead_aVL',
    6: 'lead_aVF',
    7: 'lead_V1',
    8: 'lead_V2',
    9: 'lead_V3',
    10: 'lead_V4',
    11: 'lead_V5',
    12: 'lead_V6'
}

# 用途: 标记每个像素属于哪个导联的波形
# 单通道语义分割标签
```

#### 2.4 `label_auxiliary.npy`
```python
shape: (1, H, W)
dtype: uint8
value: 0-255

# 包含内容
contents = [
    '定标脉冲 (calibration pulse)',
    '导联分隔符 (lead separators)',
    '页眉文字 (header text)'
]

# 用途: 标记辅助标记物的位置
# 训练OCR和布局分析任务
```

#### 2.5 `label_paper_speed.npy` & `label_gain.npy`
```python
shape: (1, H, W)
dtype: uint8
value: 0-255

# paper_speed: 标记 "25.0mm/s" 文字位置
# gain: 标记 "10.0mm/mV" 文字位置

# 用途: OCR目标检测和识别
# 物理参数提取
```

### 3. JSON数据结构

#### 3.1 `gt_signals.json`
```json
{
  "fs": 500,                          // 采样率 (Hz)
  "signals": {
    "I": [0.123, 0.145, ...],         // 1250个点 (2.5s)
    "II": [0.098, 0.112, ...],        // 5000个点 (10s)
    "III": [0.087, 0.091, ...],       // 1250个点
    "aVR": [-0.123, -0.145, ...],
    "aVL": [0.045, 0.056, ...],
    "aVF": [0.034, 0.042, ...],
    "V1": [0.023, 0.028, ...],
    "V2": [0.045, 0.051, ...],
    "V3": [0.078, 0.089, ...],
    "V4": [0.112, 0.123, ...],
    "V5": [0.098, 0.107, ...],
    "V6": [0.087, 0.095, ...]
  },
  "durations": {
    "I": 2.5,
    "II": 10.0,                       // 长导联
    "III": 2.5,
    // ... 其余导联
  },
  "lengths": {
    "I": 1250,                        // 2.5s * 500Hz
    "II": 5000,                       // 10s * 500Hz
    "III": 1250,
    // ... 其余导联
  }
}
```

#### 3.2 `metadata.json`
```json
{
  "ecg_id": "262",
  "fs": 500,
  "sig_len": 5000,
  "layout_type": "3x4+1",             // 布局类型
  "degradation_type": "0005",         // 退化类型
  
  "physical_params": {
    "paper_speed_mm_s": 25.0,         // 纸速
    "gain_mm_mv": 10.0,               // 增益
    "effective_px_per_mm": 20.5,      // 实际像素/毫米
    "effective_px_per_mv": 205.0      // 实际像素/毫伏
  },
  
  "ocr_targets": {
    "paper_speed": {
      "value": 25.0,
      "bbox": [1100, 1650, 1250, 1680]  // [x1, y1, x2, y2]
    },
    "gain": {
      "value": 10.0,
      "bbox": [1450, 1650, 1600, 1680]
    },
    "calibration_pulses": [
      [150, 300, 190, 500],           // 第1行脉冲bbox
      [150, 650, 190, 850],           // 第2行脉冲bbox
      [150, 1000, 190, 1200],         // 第3行脉冲bbox
      [150, 1400, 190, 1600]          // 长导联脉冲bbox
    ]
  },
  
  "lead_rois": {
    "I": {
      "bbox": [200, 250, 750, 450],   // 导联区域
      "text_bbox": [210, 280, 240, 310],  // 文字bbox
      "baseline_y": 350,              // 基线y坐标
      "time_range": [0.0, 2.5]        // 时间范围(秒)
    },
    "II": {
      "bbox": [200, 600, 750, 800],
      "text_bbox": [210, 630, 250, 660],
      "baseline_y": 700,
      "time_range": [0.0, 2.5]
    },
    // ... 其余10个导联
    "II_long": {                      // 长导联特殊处理
      "bbox": [200, 1350, 2000, 1550],
      "text_bbox": [210, 1380, 250, 1410],
      "baseline_y": 1450,
      "time_range": [0.0, 10.0]       // 10秒长导联
    }
  },
  
  "image_size": {
    "height": 2040,                   // 底板高度
    "width": 2640                     // 底板宽度
  },
  
  "paper_offset": {
    "x": 150,                         // 纸张粘贴x偏移
    "y": 200                          // 纸张粘贴y偏移
  },
  
  "paper_color_bgr": [255, 252, 250], // 纸张颜色
  
  "geometric_transform": [             // 可选: 3x3变换矩阵
    [0.998, 0.012, 5.2],
    [-0.013, 0.997, 3.8],
    [0.0, 0.0, 1.0]
  ]
}
```

---

## 🔧 核心函数详解

### 1. `render_clean_ecg_ultimate()`

**功能**: 主渲染函数，生成清晰的ECG图像和所有标注

**输入参数**:
```python
df: pd.DataFrame              # ECG信号数据
layout_type: str              # '3x4+1', '3x4', '6x2', '12x1'
params: dict                  # 物理参数
fs: int                       # 采样率
sig_len: int                  # 信号长度
```

**输出返回**:
```python
(
    clean_img,                # (H, W, 3) RGB图像
    base,                     # (H, W, 3) 网格底图
    wave_label,               # (H, W) 波形语义掩码
    text_masks,               # (13, H, W) 文字掩码
    alpha_auxiliary,          # (1, H, W) 辅助掩码
    baseline_heatmaps,        # (12, H, W) 基线热图
    ps_mask,                  # (1, H, W) 纸速掩码
    gain_mask,                # (1, H, W) 增益掩码
    paper_color,              # (3,) BGR颜色
    metadata_params,          # dict 物理参数
    lead_rois,                # dict 导联RoI
    ocr_targets               # dict OCR目标
)
```

**关键步骤**:
```python
# 1. 生成网格底图
grid_base = generate_paper_texture(h, w, paper_color, grid_img=temp_base)

# 2. 初始化内容图层 (白色背景)
sig_rgb = np.full((h, w, 3), 255, dtype=np.uint8)

# 3. 渲染布局
render_layout_3x4_plus_II_ultimate(...)

# 4. Alpha混合融合
combined_alpha = np.maximum(wave_mask_binary, alpha_auxiliary[0])
combined_alpha = np.maximum(combined_alpha, text_mask_combined)
combined_alpha = np.maximum(combined_alpha, ps_mask[0])
combined_alpha = np.maximum(combined_alpha, gain_mask[0])

alpha_mask = combined_alpha[..., None] / 255.0
clean_img = base * (1 - alpha_mask) + sig_rgb * alpha_mask
```

### 2. `render_layout_3x4_plus_II_ultimate()`

**功能**: 渲染3x4+长导联布局

**核心逻辑**:
```python
# 1. 渲染定标脉冲 (每行一个)
for r in range(3):
    render_calibration_pulse(...)

# 2. 渲染12个短导联
for lead, (r, c) in layout_leads.items():
    # 绘制波形
    cv2.polylines(sig_rgb, [pts], ink_color, thick)
    cv2.polylines(wave_label, [pts], lead_id, thick)
    
    # 绘制基线
    cv2.line(baseline_heatmaps[lead_id-1], ...)
    
    # 绘制文字
    render_lead_text(sig_rgb, text_masks, lead, ...)
    
    # 保存RoI
    lead_rois_dict[lead] = {...}

# 3. 绘制分隔符
for c in range(1, 4):
    for r in range(3):
        draw_lead_separator(...)

# 4. 渲染长导联
render_long_lead(...)
```

### 3. `apply_degradation_pipeline_ultimate()`

**功能**: 应用图像退化和几何变换

**退化类型**:
```python
degradation_effects = {
    'CLEAN': None,                     # 无退化
    'PRINTED_COLOR': add_printer_halftone(),  # 彩色打印
    'PRINTED_BW': add_printer_halftone() + grayscale,  # 黑白打印
    'PHOTO_PRINT': add_motion_blur() + jpeg_compression,  # 拍照打印件
    'PHOTO_SCREEN': add_screen_moire() + jpeg_compression,  # 拍照屏幕
    'STAINED': add_stains(),           # 污渍
    'DAMAGED': add_severe_damage(),    # 严重损坏
    'MOLD_COLOR': add_mold_spots(),    # 彩色霉斑
    'MOLD_BW': add_mold_spots() + grayscale  # 黑白霉斑
}
```

**几何变换**:
```python
# 1. 旋转: -5° ~ +5°
M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)

# 2. 透视: 四角随机偏移 ±2%
M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 3. 合并变换
M_geo = M_persp @ M_rot_3x3

# 4. 应用到图像和所有掩码
dirty_img = cv2.warpPerspective(img, M_geo, ...)
warped_masks = cv2.warpPerspective(masks, M_geo, ...)
```

### 4. `transform_bbox()`

**功能**: 坐标变换（原始纸张 → 最终图像）

**变换流程**:
```python
# 1. 应用粘贴偏移
pts_offset = pts + [x_offset, y_offset]

# 2. 应用几何变换
pts_transformed = cv2.perspectiveTransform(pts_offset, M_geo)

# 3. 计算新的AABB
new_bbox = [min_x, min_y, max_x, max_y]
```

**使用场景**:
- OCR目标bbox变换
- 导联RoI bbox变换
- 定标脉冲bbox变换

---

## 📚 使用指南

### 1. 基础使用

```bash
# 安装依赖
pip install numpy pandas opencv-python tqdm

# 快速测试 (5个样本)
python ecg_simulator_v46_ultimate.py --limit 5 --debug --validate

# 生产运行 (所有数据，8进程)
python ecg_simulator_v46_ultimate.py --workers 8 --variations 3
```

### 2. 命令行参数

```bash
--workers N        # 并行worker数量 (默认4)
--limit N          # 限制处理的ECG ID数量
--debug            # 单进程调试模式
--variations N     # 每个ID生成的变体数量 (默认3)
--validate         # 运行验证检查
```

### 3. 数据加载示例

```python
import numpy as np
import cv2
import json

# 加载单个样本
sample_id = "262_v00_3x4+1_0005"
sample_dir = f"./output/{sample_id}"

# 1. 加载图像
img = cv2.imread(f"{sample_dir}/{sample_id}_dirty.png")

# 2. 加载掩码
baseline = np.load(f"{sample_dir}/{sample_id}_label_baseline.npy")  # (12, H, W)
text_mask = np.load(f"{sample_dir}/{sample_id}_label_text_multi.npy")  # (13, H, W)
wave_mask = np.load(f"{sample_dir}/{sample_id}_label_wave.npy")  # (H, W)
aux_mask = np.load(f"{sample_dir}/{sample_id}_label_auxiliary.npy")  # (1, H, W)

# 3. 加载JSON
with open(f"{sample_dir}/{sample_id}_gt_signals.json", 'r') as f:
    gt_signals = json.load(f)

with open(f"{sample_dir}/{sample_id}_metadata.json", 'r') as f:
    metadata = json.load(f)

# 4. 提取训练标签
# 粗层标签
coarse_baseline = baseline.max(axis=0, keepdims=True)  # (1, H, W)

# 时间范围
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
time_ranges = np.array([
    metadata['lead_rois'][lead]['time_range'] 
    for lead in lead_names
])  # (12, 2)

# 真值信号
gt_signal_array = np.zeros((12, 5000), dtype=np.float32)
for i, lead in enumerate(lead_names):
    sig = gt_signals['signals'][lead]
    if sig is not None:
        gt_signal_array[i, :len(sig)] = sig
```

### 4. PyTorch Dataset示例

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path

class ECGImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.samples = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        sample_id = sample_dir.name
        
        # 加载图像
        img = cv2.imread(str(sample_dir / f"{sample_id}_dirty.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 加载掩码
        baseline = np.load(sample_dir / f"{sample_id}_label_baseline.npy")
        text_mask = np.load(sample_dir / f"{sample_id}_label_text_multi.npy")
        aux_mask = np.load(sample_dir / f"{sample_id}_label_auxiliary.npy")
        
        # 加载元数据
        with open(sample_dir / f"{sample_id}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # 加载真值信号
        with open(sample_dir / f"{sample_id}_gt_signals.json", 'r') as f:
            gt_signals = json.load(f)
        
        # 转换为tensor
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        baseline = torch.from_numpy(baseline).float()
        text_mask = torch.from_numpy(text_mask).long()
        aux_mask = torch.from_numpy(aux_mask).float()
        
        # 提取时间范围
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        time_ranges = torch.tensor([
            metadata['lead_rois'][lead]['time_range']
            for lead in lead_names
        ])
        
        return {
            'image': img,
            'baseline': baseline,
            'text_mask': text_mask,
            'aux_mask': aux_mask,
            'time_ranges': time_ranges,
            'metadata': metadata,
            'gt_signals': gt_signals
        }
```

---

## 🔄 迭代更新指南

### 1. 添加新的布局类型

**步骤1**: 在常量中添加布局配置
```python
# 在 LAYOUT_CONFIGS 中添加
LayoutType.LAYOUT_6X2_NEW = "6x2_new"

LAYOUT_CONFIGS[LayoutType.LAYOUT_6X2_NEW] = {
    'leads': {
        # 定义导联位置 (row, col)
        'I': (0, 0), 'II': (0, 1), ...
    },
    'long_lead': None,  # 或指定长导联
    'rows': 6,
    'cols': 2
}
```

**步骤2**: 实现渲染函数
```python
def render_layout_6x2_new_ultimate(df, sig_rgb, wave_label, text_masks, 
                                   alpha_auxiliary, baseline_heatmaps, 
                                   params, ink_color, font, fs,
                                   render_params, lead_rois_dict, 
                                   calibration_pulse_bboxes):
    # 参考 render_layout_3x4_plus_II_ultimate() 的实现
    pass
```

**步骤3**: 在主渲染函数中调用
```python
# 在 render_clean_ecg_ultimate() 中添加
elif layout_type == LayoutType.LAYOUT_6X2_NEW:
    render_layout_6x2_new_ultimate(...)
```

### 2. 添加新的退化类型

**步骤1**: 定义退化类型
```python
class DegradationType:
    # ... 现有类型
    WATER_DAMAGE = "0013"  # 新增
```

**步骤2**: 实现退化效果函数
```python
def add_water_damage(img):
    """添加水渍效果"""
    h, w = img.shape[:2]
    
    # 创建水渍形状
    water_mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(1, 3)):
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(w//6, w//3), random.randint(h//6, h//3))
        angle = random.randint(0, 180)
        cv2.ellipse(water_mask, center, axes, angle, 0, 360, 1.0, -1)
    
    # 模糊边缘
    water_mask = cv2.GaussianBlur(water_mask, (51, 51), 0)
    
    # 应用褪色效果
    fade_factor = 0.7
    result = img.astype(np.float32)
    result = result + (255 - result) * water_mask[..., None] * (1 - fade_factor)
    
    return np.clip(result, 0, 255).astype(np.uint8)
```

**步骤3**: 集成到退化管道
```python
# 在 apply_degradation_pipeline_ultimate() 中添加
elif degradation_type == DegradationType.WATER_DAMAGE:
    img = add_water_damage(img)
```

### 3. 修改物理参数范围

```python
def sample_physical_params(layout_type):
    # 修改增益范围
    if layout_type == LayoutType.LAYOUT_3X4_PLUS_II:
        gain_mm_mv = random.choice([5.0, 10.0, 20.0])  # 添加20.0
    
    # 修改纸速范围
    paper_speed_mm_s = random.choice([12.5, 25.0, 50.0])  # 添加12.5
    
    return {
        'paper_speed_mm_s': paper_speed_mm_s,
        'gain_mm_mv': gain_mm_mv,
        'lead_durations': {'long': 10.0, 'short': 2.5}
    }
```

