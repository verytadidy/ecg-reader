这是一份针对 **ECG 仿真器 V39 (Semantic & Realistic Update)** 的技术文档。这份文档旨在帮助你快速回顾核心逻辑、参数配置及输出格式，便于后续迭代开发。

-----

# ECG Image Digitization Simulator - Technical Documentation

**Version:** V39
**Date:** 2025-11-15
**Status:** Production Ready (Training Data Generation)

## 1\. 项目概述

该仿真器用于将 1D ECG 信号数据（CSV格式）批量转换为 **仿真的 2D 扫描心电图纸图像**。它生成的不仅是逼真的“脏”数据（用于模型输入），还生成精确的**语义分割掩码**（Semantic Segmentation Mask）和其他辅助标签（用于监督训练）。

核心目标：**生成高逼真度、多样化布局、带有精确语义标注的合成数据。**

-----

## 2\. 核心功能特性 (Key Features)

### 2.1 多布局支持 (Multi-Layout Generation)

支持四种临床常用布局，并根据配置概率随机采样：

  * **3x4 + 1 (II):** 标准 12 导联，底部带一条长 II 导联（最常用）。
  * **3x4:** 纯 12 导联网格，无长导联。
  * **6x2:** 两列布局，每列 6 个导联（V39 优化了增益）。
  * **12x1:** 单列长条布局（模拟动态或长记录）。

### 2.2 物理参数仿真 (Physical Simulation)

模拟真实心电图机的物理参数，而非简单的图像缩放：

  * **纸速:** 25mm/s 或 50mm/s。
  * **增益 (Gain):**
      * 标准布局：5mm/mV 或 10mm/mV。
      * **6x2 (V39特性):** 加权随机 (50% 10mm/mV, 40% 5mm/mV, 10% 20mm/mV)，避免波形重叠。
      * 12x1：根据概率分布偏向 5mm/mV。
  * **分辨率:** 随机采样 `px_per_mm` (18.0 \~ 22.0)，模拟不同扫描分辨率。

### 2.3 语义分割标注 (Semantic Segmentation)

**V38/V39 重大变更：** 波形标签 (`label_wave`) 不再是二值图 (0/255)。

  * **背景:** 0
  * **导联波形:** 1 \~ 12 (对应 I, II, III, aVR, ..., V6)。
  * **用途:** 支持模型直接学习区分不同导联的波形线，解决波形交叉时的归属问题。

### 2.4 复杂退化引擎 (Degradation Engine)

模拟真实世界的“脏”数据特征：

  * **纹理:** 纸张纤维、网格线、不均匀光照。
  * **伪影:** 污渍 (Coffee stains)、霉斑、折痕、撕裂孔洞。
  * **噪声:** 打印机半色调 (Halftone)、屏幕摩尔纹 (Moire)、JPEG 压缩噪声、运动模糊。
  * **几何:** 随机旋转、透视变换、纸张在扫描仪底板上的随机平移。

### 2.5 细节随机化 (V39 Update)

  * **悬浮分隔符:** 列分隔符 (Tick marks) 有 50% 概率不画在基线上，而是悬浮在波形上方。
  * **文字扰动:** 导联名称的位置会有轻微的随机抖动。

-----

## 3\. 输入与输出架构

### 3.1 输入数据

  * **Root Dir:** `BASE_DATA_DIR`
  * **Metadata:** `train.csv` (必须包含 `id`, `fs`, `sig_len`)。
  * **Signals:** `train/{id}/{id}.csv` (包含 I, II, ..., V6 等列)。

### 3.2 输出结构

输出目录为 `OUTPUT_DIR/{id}_v{var}_{layout}_{deg}/`，每个变体包含：

| 文件名后缀 | 类型 | 描述 |
| :--- | :--- | :--- |
| `_dirty.png` | **Input** | 最终的仿真图像（带退化、背景、透视变换）。 |
| `_label_wave.png` | **Target** | **语义分割掩码**。像素值 0-12，无抗锯齿 (Nearest Neighbor)。 |
| `_label_grid.png` | Target | 网格线的二值掩码。 |
| `_label_other.png` | Target | 文本（ID、标签）和定标脉冲的掩码。 |
| `_label_baseline.png` | Aux | 零电位基线的掩码。 |
| `_metadata.json` | Meta | 包含所有生成参数（增益、纸速、几何变换矩阵、导联ID映射表等）。 |

-----

## 4\. 核心代码模块说明

为方便后续修改，请关注以下关键函数：

### 4.1 参数控制

  * **`CONFIG` 字典:** 控制布局概率 (`LAYOUT_DISTRIBUTION`) 和退化类型概率。
  * **`sample_physical_params_v37`:** **(修改热点)** 控制不同布局下的增益 (`gain_mm_mv`) 和纸速采样逻辑。

### 4.2 渲染逻辑 (Renderer)

  * **`render_clean_ecg_v37`:** 主渲染入口，负责画纸张、网格、页眉页脚。
  * **`render_layout_*` 系列函数:**
      * 核心逻辑：计算时间轴 -\> 切片 DataFrame -\> 坐标映射 -\> OpenCV 绘图。
      * **V39特性:** `separator_style` (centered/floating) 在这里实现。
      * **语义掩码:** 绘图时使用 `lead_id` 而非 255 绘制到 `wave_label_semantic_mask`。

### 4.3 退化逻辑 (Augmentor)

  * **`apply_degradation_pipeline_v32`:** 串联各种退化效果。
  * **关键注意:** 在进行几何变换（透视/旋转）时，`wave` (语义掩码) 必须使用 `cv2.INTER_NEAREST` 插值，以防止引入非整数的像素值（如 1.5），破坏类别 ID。

### 4.4 主流程 (Pipeline)

  * **`process_one_id_v37`:**
    1.  读取 CSV 和 Metadata (fs, sig\_len)。
    2.  调用 Render 生成 "Clean Paper"。
    3.  将 Paper 贴到 "Scanner Bed" (背景) 上。
    4.  调用 Degradation 生成 "Dirty Image"。
    5.  保存所有图片和 JSON。

-----

## 5\. 快速上手与调试

### 运行命令

```bash
# 默认运行 (使用多进程)
python simulator.py

# 调试模式 (单进程，显示进度条和详细错误堆栈)
python simulator.py --debug --limit 10

# 指定 Worker 数量
python simulator.py --workers 8
```

### 常见修改场景指南

1.  **修改 6x2 布局的增益分布:**

      * 定位到 `sample_physical_params_v37` 函数。
      * 修改 `elif layout_type == LayoutType.LAYOUT_6X2:` 下的 `weights` 数组。

2.  **调整列分隔符的悬浮概率:**

      * 定位到 `render_layout_3x4_v37`, `_3x4_plus_II`, `_6x2` 函数。
      * 搜索 `separator_style = random.choice(...)` 进行修改。

3.  **添加新的退化效果:**

      * 编写新的 `add_xxx_effect(img)` 函数。
      * 在 `apply_degradation_pipeline_v32` 中添加调用逻辑。

4.  **修改导联 ID 映射:**

      * 修改全局字典 `LEAD_TO_ID_MAP`。

-----

## 6\. 版本变更日志 (Change Log)

  * **V37:** 修复采样率 (fs) 与信号长度 (sig\_len) 不匹配导致的波形拉伸/压缩问题。从 `train.csv` 动态读取真实 fs。
  * **V38:** 引入语义分割。波形标签从二值图升级为 12 类语义图。
  * **V39 (Current):**
      * **真实性增强:** 6x2 布局增益调整（降低 20mm/mV 概率）。
      * **多样性增强:** 引入 "Floating Tick Marks"（悬浮分隔符）。
      * **元数据:** `metadata.json` 现包含 `lead_to_id_map` 以方便下游 dataloader 解析。


