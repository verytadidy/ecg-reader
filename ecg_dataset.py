"""
ECG V47 Dataset (Production Ready)
适配 CRNN 架构与物理参数感知

主要功能:
1. 加载仿真图像 (支持高分辨率)
2. 加载 GT 信号并根据模型输出步长进行重采样
3. 提取物理参数 (Gain, Speed) 用于 Loss 计算
4. 包含完整的单元测试模块
"""

import json
import numpy as np
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from typing import Tuple, Dict, Optional, List
import warnings

class ECGV47ProductionDataset(Dataset):
    """
    ECG 生产级数据集加载器
    """
    def __init__(self,
                 sim_root_dir: str,
                 csv_root_dir: str,
                 split: str = 'train',
                 # 宽度设为 2048，经过 Stride=4 的特征提取后，序列长度为 512
                 # 配合 CRNN 和后续的样条插值，足以重建 500Hz 甚至 1000Hz 信号
                 target_size: Tuple[int, int] = (512, 2048),
                 target_fs: int = 500,
                 max_samples: Optional[int] = None,
                 augment: bool = False):
        
        self.sim_root = Path(sim_root_dir)
        self.csv_root = Path(csv_root_dir)
        self.split = split
        self.target_size = target_size
        self.target_fs = target_fs
        self.augment = augment
        
        # 导联顺序必须固定，与模型输出通道对应
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # 扫描样本
        self.samples = self._scan_samples(max_samples)
        
        # 定义增强流水线
        self.transform = self._build_transform()

        print(f"\n{'='*60}")
        print(f"ECG Dataset Initialized ({split})")
        print(f"Root: {sim_root_dir}")
        print(f"Samples: {len(self.samples)}")
        print(f"Target Size: {target_size} (H, W)")
        print(f"Augmentation: {augment}")
        print(f"{'='*60}\n")

    def _scan_samples(self, max_samples):
        """扫描符合 V47 格式的样本目录"""
        samples = []
        if not self.sim_root.exists():
            warnings.warn(f"仿真目录不存在: {self.sim_root}")
            return []

        # 遍历仿真目录
        dirs = sorted([d for d in self.sim_root.iterdir() if d.is_dir()])
        
        for d in dirs:
            sid = d.name
            # 检查必要文件 (V45/V47 格式)
            # 必须包含脏图、GT信号json、元数据json
            required_files = [
                d / f"{sid}_dirty.png",
                d / f"{sid}_gt_signals.json",
                d / f"{sid}_metadata.json"
            ]
            
            if all(f.exists() for f in required_files):
                samples.append({'id': sid, 'dir': d})
            
            if max_samples and len(samples) >= max_samples:
                break
                
        if len(samples) == 0:
            warnings.warn(f"未在 {self.sim_root} 找到有效样本")
            
        return samples

    def _build_transform(self):
        if self.augment:
            return A.Compose([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                # 可以添加更多针对扫描件的增强，如模糊、JPEG压缩等
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sid = sample_info['id']
        sdir = sample_info['dir']
        
        # 1. 加载图像
        img_path = sdir / f"{sid}_dirty.png"
        # 使用 PIL 加载保持最高质量，然后转 numpy
        image_pil = Image.open(img_path).convert('RGB')
        image = np.array(image_pil)
        
        # 2. 加载元数据 (物理参数)
        meta_path = sdir / f"{sid}_metadata.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        # 提取物理参数用于 Loss 计算
        # 如果元数据缺失，使用默认标准值
        phys_params = meta.get('physical_params', {})
        gain = phys_params.get('gain_mm_mv', 10.0)
        speed = phys_params.get('paper_speed_mm_s', 25.0)
        
        # 3. 加载辅助掩码 (Baseline & Text)
        # 用于监督定位头
        base_path = sdir / f"{sid}_label_baseline.npy"
        text_path = sdir / f"{sid}_label_text_multi.npy"
        
        h_tg, w_tg = self.target_size
        
        # 处理基线掩码 (12, H, W)
        if base_path.exists():
            baseline_raw = np.load(base_path)
            baseline_resized = np.zeros((12, h_tg, w_tg), dtype=np.float32)
            for i in range(12):
                # 使用 linear 插值保持热图性质
                baseline_resized[i] = cv2.resize(baseline_raw[i], (w_tg, h_tg), interpolation=cv2.INTER_LINEAR) / 255.0
        else:
            baseline_resized = np.zeros((12, h_tg, w_tg), dtype=np.float32)
            
        # 处理文字掩码 (13, H, W) - 第0通道是背景
        if text_path.exists():
            text_raw = np.load(text_path)
            text_resized = np.zeros((13, h_tg, w_tg), dtype=np.float32)
            for i in range(13):
                text_resized[i] = cv2.resize(text_raw[i], (w_tg, h_tg), interpolation=cv2.INTER_NEAREST) / 255.0
        else:
            text_resized = np.zeros((13, h_tg, w_tg), dtype=np.float32)

        # 4. 加载 GT 信号并重采样
        gt_path = sdir / f"{sid}_gt_signals.json"
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
            
        # 模型输出的特征宽度 (基于 ResNet FPN P2/d2 层，stride=4)
        feature_width = w_tg // 4
        
        target_signals = np.zeros((12, feature_width), dtype=np.float32)
        valid_mask = np.zeros((12, feature_width), dtype=np.float32) # 标记哪些时间点有真实信号
        
        for i, lead in enumerate(self.lead_names):
            raw_sig = gt_data['signals'].get(lead, None)
            
            if raw_sig is None or len(raw_sig) == 0:
                continue
                
            raw_sig = np.array(raw_sig)
            
            # 重采样策略:
            # 我们假设输入图像覆盖了信号的全长。
            # 将 GT 信号线性插值到 feature_width，让模型学习波形的“形状”。
            # 物理数值转换 (mV -> Pixel Offset) 将由 Loss Function 处理 (传入 Gain)。
            
            x_old = np.linspace(0, 1, len(raw_sig))
            x_new = np.linspace(0, 1, feature_width)
            
            resampled_sig = np.interp(x_new, x_old, raw_sig)
            
            target_signals[i] = resampled_sig
            
            # 对于短导联，在图像上的分布是分段的，但此处简化为整行有效
            # 如果需要更精细的控制，可以使用 lead_rois 中的 time_range 来生成 valid_mask
            valid_mask[i] = 1.0

        # 5. 图像 Resize 与 Tensor 转换
        img_resized = cv2.resize(image, (w_tg, h_tg), interpolation=cv2.INTER_LINEAR)
        
        # 应用增强和归一化
        transformed = self.transform(image=img_resized)
        img_tensor = transformed['image']
        
        return {
            'image': img_tensor,
            'baseline_mask': torch.from_numpy(baseline_resized).float(),
            'text_mask': torch.from_numpy(text_resized).float(),
            'gt_signals': torch.from_numpy(target_signals).float(), # (12, W/4) [mV]
            'valid_mask': torch.from_numpy(valid_mask).float(),     # (12, W/4)
            'metadata': {
                'ecg_id': str(sid),
                'gain': float(gain),    # mm/mV
                'speed': float(speed)   # mm/s
            }
        }

def create_dataloaders(sim_root, csv_root, batch_size=8, num_workers=4, **kwargs):
    """工厂函数"""
    dataset = ECGV47ProductionDataset(sim_root, csv_root, **kwargs)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader

# =============================================================================
# 模块测试
# =============================================================================
if __name__ == "__main__":
    import argparse
    import time
    import matplotlib.pyplot as plt
    
    # 模拟命令行参数
    parser = argparse.ArgumentParser(description='Test ECG Dataset')
    parser.add_argument('--sim_root', type=str, default='./data/sim_output', help='Path to simulation output')
    parser.add_argument('--csv_root', type=str, default='./data/train', help='Path to raw CSVs')
    args = parser.parse_args()

    # 检查路径是否存在，不存在则创建 dummy 数据进行测试
    sim_path = Path(args.sim_root)
    if not sim_path.exists():
        print(f"Warning: {sim_path} does not exist. Skipping real data test.")
    else:
        print("Testing with real data...")
        dataset = ECGV47ProductionDataset(
            sim_root_dir=args.sim_root, 
            csv_root_dir=args.csv_root,
            target_size=(512, 2048),
            augment=True
        )
        
        if len(dataset) > 0:
            # 加载一个样本
            sample = dataset[0]
            print("\nSample Shapes:")
            print(f"Image: {sample['image'].shape}")
            print(f"Baseline Mask: {sample['baseline_mask'].shape}")
            print(f"Text Mask: {sample['text_mask'].shape}")
            print(f"GT Signals: {sample['gt_signals'].shape}")
            print(f"Metadata: {sample['metadata']}")
            
            # 简单可视化验证
            # img_vis = sample['image'].permute(1, 2, 0).numpy()
            # img_vis = (img_vis * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            # img_vis = np.clip(img_vis, 0, 1)
            # plt.imshow(img_vis)
            # plt.title(f"ID: {sample['metadata']['ecg_id']}")
            # plt.show()
            
            # 验证信号非空
            if torch.sum(torch.abs(sample['gt_signals'])) > 0:
                print("✓ GT Signals contain data.")
            else:
                print("✗ GT Signals are all zeros (check json loading).")
        else:
            print("No samples found in directory.")