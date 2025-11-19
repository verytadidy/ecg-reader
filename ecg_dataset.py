"""
ECG V45 生产就绪数据集加载器

特点:
1. ✅ 支持多采样率（重采样到500Hz统一处理）
2. ✅ 直接从仿真器输出加载V45格式
3. ✅ 高效缓存机制（可选）
4. ✅ 完善的错误处理和数据验证
5. ✅ 支持所有V45标注（纸速OCR、增益OCR、物理约束）
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from typing import Optional, Tuple, Dict, List
import warnings
from PIL import Image


class ECGV45ProductionDataset(Dataset):
    """
    ECG V45 生产就绪数据集
    
    特性:
    - 自动重采样到统一采样率（默认500Hz）
    - 完整支持V45标注格式
    - 数据验证和错误恢复
    - 可选的内存缓存
    """
    
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def __init__(self,
                 sim_root_dir: str,
                 csv_root_dir: str,
                 split: str = 'train',
                 target_size: Tuple[int, int] = (512, 672),
                 target_fs: int = 500,
                 max_samples: Optional[int] = None,
                 cache_in_memory: bool = False,
                 load_fine_labels: bool = True,
                 load_ocr_labels: bool = True,
                 augment: bool = False):
        """
        Args:
            sim_root_dir: 仿真数据根目录
            csv_root_dir: 原始CSV数据根目录
            split: 'train' 或 'val' 或 'test'
            target_size: 统一resize尺寸 (H, W)
            target_fs: 目标采样率（所有信号重采样到此采样率）
            max_samples: 最大样本数（用于快速测试）
            cache_in_memory: 是否缓存到内存（需要足够RAM）
            load_fine_labels: 是否加载细粒度标注
            load_ocr_labels: 是否加载OCR标注（纸速、增益）
            augment: 是否使用数据增强（仅训练时）
        """
        self.sim_root = Path(sim_root_dir)
        self.csv_root = Path(csv_root_dir)
        self.split = split
        self.target_size = target_size
        self.target_fs = target_fs
        self.cache_in_memory = cache_in_memory
        self.load_fine = load_fine_labels
        self.load_ocr = load_ocr_labels
        self.augment = augment and (split == 'train')
        
        # 扫描所有有效样本
        self.samples = self._scan_samples(max_samples)
        
        # 内存缓存（可选）
        self.cache = {} if cache_in_memory else None
        
        # 数据变换
        self.transform = self._build_transform()
        
        # 统计信息
        self._print_statistics()
    
    def _build_transform(self):
        """构建数据变换pipeline"""
        if self.augment:
            # 训练时增强
            transform = A.Compose([
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.ISONoise(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.Blur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # 验证/测试时只归一化
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return transform
    
    def _scan_samples(self, max_samples: Optional[int]) -> List[Dict]:
        """扫描并验证所有样本"""
        samples = []
        
        for sample_dir in sorted(self.sim_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            
            sample_id = sample_dir.name
            
            # V45必需文件
            required_files = [
                f"{sample_id}_dirty.png",
                f"{sample_id}_label_wave.png",
                f"{sample_id}_label_baseline_coarse.npy",
                f"{sample_id}_metadata.json"
            ]
            
            # 检查细粒度标注
            if self.load_fine:
                required_files.extend([
                    f"{sample_id}_label_baseline_fine.npy",
                    f"{sample_id}_label_text_multi.npy",
                    f"{sample_id}_label_auxiliary.npy",
                    f"{sample_id}_label_grid_fine.npy"
                ])
            
            # 检查OCR标注（V45新增）
            if self.load_ocr:
                required_files.extend([
                    f"{sample_id}_label_paper_speed.npy",
                    f"{sample_id}_label_gain.npy"
                ])
            
            # 验证文件完整性
            if not all((sample_dir / f).exists() for f in required_files):
                missing = [f for f in required_files if not (sample_dir / f).exists()]
                warnings.warn(f"样本 {sample_id} 缺失文件: {missing}，跳过")
                continue
            
            # 加载并验证元数据
            try:
                metadata_path = sample_dir / f"{sample_id}_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # 验证关键字段
                required_keys = ['ecg_id', 'fs', 'sig_len', 'physical_params']
                if not all(k in metadata for k in required_keys):
                    warnings.warn(f"样本 {sample_id} 元数据不完整，跳过")
                    continue
                
                samples.append({
                    'sample_dir': sample_dir,
                    'sample_id': sample_id,
                    'metadata': metadata
                })
                
                if max_samples and len(samples) >= max_samples:
                    break
                    
            except Exception as e:
                warnings.warn(f"样本 {sample_id} 元数据读取失败: {e}")
                continue
        
        if len(samples) == 0:
            raise RuntimeError(f"在 {self.sim_root} 未找到有效样本！")
        
        return samples
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        print(f"\n{'='*80}")
        print(f"ECG V45 数据集加载完成 (split={self.split})")
        print(f"{'='*80}")
        print(f"总样本数: {len(self.samples)}")
        print(f"目标尺寸: {self.target_size[0]}×{self.target_size[1]}")
        print(f"目标采样率: {self.target_fs} Hz")
        print(f"细粒度标注: {'✓' if self.load_fine else '✗'}")
        print(f"OCR标注: {'✓' if self.load_ocr else '✗'}")
        print(f"数据增强: {'✓' if self.augment else '✗'}")
        
        # 统计分布
        fs_counts = {}
        layout_counts = {}
        deg_counts = {}
        paper_speed_counts = {}
        gain_counts = {}
        corruption_counts = {'none': 0, 'has_corruption': 0}
        
        for sample in self.samples:
            meta = sample['metadata']
            
            # 采样率
            fs = meta['fs']
            fs_counts[fs] = fs_counts.get(fs, 0) + 1
            
            # 布局
            layout = meta['layout_type']
            layout_counts[layout] = layout_counts.get(layout, 0) + 1
            
            # 退化类型
            deg = meta['degradation_type']
            deg_counts[deg] = deg_counts.get(deg, 0) + 1
            
            # 物理参数
            paper_speed = meta['physical_params']['paper_speed_mm_s']
            paper_speed_counts[paper_speed] = paper_speed_counts.get(paper_speed, 0) + 1
            
            gain = meta['physical_params']['gain_mm_mv']
            gain_counts[gain] = gain_counts.get(gain, 0) + 1
            
            # 导联污损（V45）
            if 'lead_corruption' in meta and len(meta['lead_corruption']) > 0:
                corruption_counts['has_corruption'] += 1
            else:
                corruption_counts['none'] += 1
        
        print(f"\n原始采样率分布:")
        for fs, count in sorted(fs_counts.items()):
            pct = count / len(self.samples) * 100
            status = "→ 重采样" if fs != self.target_fs else "✓ 保持"
            print(f"  {fs:3d}Hz: {count:5d} ({pct:5.1f}%) {status}")
        
        print(f"\n布局类型分布:")
        for layout, count in sorted(layout_counts.items()):
            pct = count / len(self.samples) * 100
            print(f"  {layout:15s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n退化类型分布 (Top 5):")
        for deg, count in sorted(deg_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = count / len(self.samples) * 100
            print(f"  {deg:15s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n纸速分布 ⭐⭐⭐⭐⭐:")
        for speed, count in sorted(paper_speed_counts.items()):
            pct = count / len(self.samples) * 100
            print(f"  {speed:5.1f} mm/s: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n增益分布 ⭐⭐⭐:")
        for gain, count in sorted(gain_counts.items()):
            pct = count / len(self.samples) * 100
            print(f"  {gain:5.1f} mm/mV: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n导联污损统计 (V45):")
        for status, count in corruption_counts.items():
            pct = count / len(self.samples) * 100
            print(f"  {status:15s}: {count:5d} ({pct:5.1f}%)")
        
        if self.cache_in_memory:
            estimated_memory = len(self.samples) * 15  # ~15MB/sample
            print(f"\n⚠️  内存缓存已启用，预计占用 ~{estimated_memory}MB RAM")
        
        print(f"{'='*80}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """加载单个样本"""
        # 检查缓存
        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]
        
        sample_info = self.samples[idx]
        sample_dir = sample_info['sample_dir']
        sample_id = sample_info['sample_id']
        metadata = sample_info['metadata']
        
        try:
            # ========== 1. 加载图像数据 ==========
            dirty_path = sample_dir / f"{sample_id}_dirty.png"
            image = np.array(Image.open(dirty_path).convert('RGB'))
            
            # 波形分割掩码
            wave_mask = np.array(Image.open(sample_dir / f"{sample_id}_label_wave.png"))
            
            # 粗粒度基线
            baseline_coarse = np.load(sample_dir / f"{sample_id}_label_baseline_coarse.npy")
            
            # ========== 2. 加载细粒度标注（可选）==========
            if self.load_fine:
                baseline_fine = np.load(sample_dir / f"{sample_id}_label_baseline_fine.npy")
                text_multi = np.load(sample_dir / f"{sample_id}_label_text_multi.npy")
                auxiliary = np.load(sample_dir / f"{sample_id}_label_auxiliary.npy")
                grid_fine = np.load(sample_dir / f"{sample_id}_label_grid_fine.npy")
            else:
                H, W = image.shape[:2]
                baseline_fine = np.zeros((12, H, W), dtype=np.uint8)
                text_multi = np.zeros((13, H, W), dtype=np.uint8)
                auxiliary = np.zeros((1, H, W), dtype=np.uint8)
                grid_fine = np.zeros((1, H, W), dtype=np.uint8)
            
            # ========== 3. 加载OCR标注（V45）==========
            if self.load_ocr:
                paper_speed_mask = np.load(sample_dir / f"{sample_id}_label_paper_speed.npy")
                gain_mask = np.load(sample_dir / f"{sample_id}_label_gain.npy")
            else:
                H, W = image.shape[:2]
                paper_speed_mask = np.zeros((1, H, W), dtype=np.uint8)
                gain_mask = np.zeros((1, H, W), dtype=np.uint8)
            
            # ========== 4. Resize到统一尺寸 ==========
            h_target, w_target = self.target_size
            
            image_resized = cv2.resize(image, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
            wave_mask_resized = cv2.resize(wave_mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
            
            # Resize所有掩码
            baseline_coarse_resized = cv2.resize(
                baseline_coarse[0], (w_target, h_target), interpolation=cv2.INTER_LINEAR
            )[np.newaxis, ...]
            
            baseline_fine_resized = np.zeros((12, h_target, w_target), dtype=np.float32)
            for i in range(12):
                baseline_fine_resized[i] = cv2.resize(
                    baseline_fine[i], (w_target, h_target), interpolation=cv2.INTER_LINEAR
                )
            
            text_multi_resized = np.zeros((13, h_target, w_target), dtype=np.float32)
            for i in range(13):
                text_multi_resized[i] = cv2.resize(
                    text_multi[i], (w_target, h_target), interpolation=cv2.INTER_LINEAR
                )
            
            auxiliary_resized = cv2.resize(
                auxiliary[0], (w_target, h_target), interpolation=cv2.INTER_LINEAR
            )[np.newaxis, ...]
            
            grid_fine_resized = cv2.resize(
                grid_fine[0], (w_target, h_target), interpolation=cv2.INTER_NEAREST
            )[np.newaxis, ...]
            
            paper_speed_resized = cv2.resize(
                paper_speed_mask[0], (w_target, h_target), interpolation=cv2.INTER_LINEAR
            )[np.newaxis, ...]
            
            gain_resized = cv2.resize(
                gain_mask[0], (w_target, h_target), interpolation=cv2.INTER_LINEAR
            )[np.newaxis, ...]
            
            # ========== 5. 加载原始GT信号 ==========
            ecg_id = metadata['ecg_id']
            original_fs = metadata['fs']
            sig_len = metadata['sig_len']
            
            csv_path = self.csv_root / ecg_id / f"{ecg_id}.csv"
            gt_signal = self._load_and_resample_signal(
                csv_path, original_fs, sig_len, self.target_fs
            )
            
            # ========== 6. 数据变换 ==========
            transformed = self.transform(image=image_resized)
            image_tensor = transformed['image']
            
            # ========== 7. 转换为Tensor ==========
            # 构建标准的lead_rois（如果不存在）
            lead_rois = metadata.get('lead_rois', {})
            if not lead_rois:
                # 创建默认的lead_rois
                lead_rois = self._create_default_lead_rois(metadata)
            
            result = {
                # 图像
                'image': image_tensor,
                
                # 波形分割
                'wave_mask': torch.from_numpy(wave_mask_resized).long(),
                
                # 基线标注
                'baseline_coarse': torch.from_numpy(baseline_coarse_resized).float() / 255.0,
                'baseline_fine': torch.from_numpy(baseline_fine_resized).float() / 255.0,
                
                # 文字和辅助
                'text_multi': torch.from_numpy(text_multi_resized).float() / 255.0,
                'auxiliary': torch.from_numpy(auxiliary_resized).float() / 255.0,
                'grid_fine': torch.from_numpy(grid_fine_resized).float() / 255.0,
                
                # OCR标注（V45）
                'paper_speed_mask': torch.from_numpy(paper_speed_resized).float() / 255.0,
                'gain_mask': torch.from_numpy(gain_resized).float() / 255.0,
                
                # GT信号
                'gt_signal': gt_signal,
                
                # 元数据
                'metadata': {
                    'sample_id': sample_id,
                    'ecg_id': ecg_id,
                    'original_fs': original_fs,
                    'target_fs': self.target_fs,
                    'physical_params': metadata['physical_params'],
                    'lead_rois': lead_rois,
                    'ocr_targets': metadata.get('ocr_targets', {}),
                    'lead_corruption': metadata.get('lead_corruption', {}),
                }
            }
            
            # 缓存到内存
            if self.cache_in_memory:
                self.cache[idx] = result
            
            return result
            
        except Exception as e:
            warnings.warn(f"加载样本 {sample_id} 失败: {e}，返回dummy样本")
            return self._get_dummy_sample()
    
    def _load_and_resample_signal(self, csv_path: Path, 
                                  original_fs: int, 
                                  sig_len: int, 
                                  target_fs: int) -> torch.Tensor:
        """
        加载并重采样信号到目标采样率
        
        关键修复：正确处理CSV中的NaN值
        - 长导联（如II）: 完整10秒数据
        - 短导联: 只有部分时间段有数据，其余为NaN
        
        Args:
            csv_path: CSV文件路径
            original_fs: 原始采样率
            sig_len: 原始信号长度
            target_fs: 目标采样率
        
        Returns:
            signal: (T, 12) tensor，T = target_fs * 10
        """
        try:
            df = pd.read_csv(csv_path)
            
            signal_list = []
            
            for lead in self.LEAD_NAMES:
                if lead in df.columns:
                    sig = df[lead].values[:sig_len]
                    
                    # 🔥 关键修复：将NaN替换为0
                    sig = np.nan_to_num(sig, nan=0.0)
                    
                    # 填充到标准长度
                    if len(sig) < sig_len:
                        sig = np.pad(sig, (0, sig_len - len(sig)), 
                                   mode='constant', constant_values=0)
                else:
                    # 导联不存在，用全0
                    sig = np.zeros(sig_len)
                
                signal_list.append(sig)
            
            signal = np.stack(signal_list, axis=1)  # (T_original, 12)
            
            # 二次检查NaN
            if np.isnan(signal).any():
                warnings.warn(f"CSV {csv_path.name} 中发现NaN，已替换为0")
                signal = np.nan_to_num(signal, nan=0.0)
            
            # 重采样到目标采样率
            if original_fs != target_fs:
                signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
                signal_tensor = signal_tensor.permute(0, 2, 1)  # (1, 12, T)
                
                target_length = int(target_fs * 10)  # 10秒
                signal_resampled = F.interpolate(
                    signal_tensor,
                    size=target_length,
                    mode='linear',
                    align_corners=True
                )
                
                signal_resampled = signal_resampled.permute(0, 2, 1).squeeze(0)
            else:
                signal_resampled = torch.from_numpy(signal).float()
            
            # 最终检查
            if torch.isnan(signal_resampled).any():
                warnings.warn(f"重采样后发现NaN，强制替换为0")
                signal_resampled = torch.nan_to_num(signal_resampled, nan=0.0)
            
            return signal_resampled
            
        except Exception as e:
            warnings.warn(f"加载信号失败 {csv_path}: {e}，返回零信号")
            target_length = int(target_fs * 10)
            return torch.zeros(target_length, 12)
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """返回dummy样本（用于错误恢复）"""
        h, w = self.target_size
        target_length = int(self.target_fs * 10)
        
        return {
            'image': torch.zeros(3, h, w),
            'wave_mask': torch.zeros(h, w, dtype=torch.long),
            'baseline_coarse': torch.zeros(1, h, w),
            'baseline_fine': torch.zeros(12, h, w),
            'text_multi': torch.zeros(13, h, w),
            'auxiliary': torch.zeros(1, h, w),
            'grid_fine': torch.zeros(1, h, w),
            'paper_speed_mask': torch.zeros(1, h, w),
            'gain_mask': torch.zeros(1, h, w),
            'gt_signal': torch.zeros(target_length, 12),
            'metadata': {'sample_id': 'dummy', 'ecg_id': 'dummy'}
        }


def create_dataloaders(
    sim_root: str,
    csv_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    train_split: float = 0.9,
    target_fs: int = 500,
    target_size: Tuple[int, int] = (512, 672),
    max_samples: Optional[int] = None,
    cache_in_memory: bool = False,
    load_fine_labels: bool = True,
    load_ocr_labels: bool = True
) -> Tuple:
    """
    创建训练和验证DataLoader
    
    Returns:
        train_loader, val_loader
    """
    # 创建完整数据集
    full_dataset = ECGV45ProductionDataset(
        sim_root_dir=sim_root,
        csv_root_dir=csv_root,
        split='train',
        target_size=target_size,
        target_fs=target_fs,
        max_samples=max_samples,
        cache_in_memory=cache_in_memory,
        load_fine_labels=load_fine_labels,
        load_ocr_labels=load_ocr_labels,
        augment=True
    )
    
    # 划分训练/验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 为验证集禁用增强
    val_dataset.dataset.augment = False
    
    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # 避免最后一个batch太小
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(val_dataset)} 样本")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Workers: {num_workers}\n")
    
    return train_loader, val_loader


# ========== 测试代码 ==========

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='测试ECG V45数据集')
    parser.add_argument('--sim_root', type=str, required=True, help='仿真数据根目录')
    parser.add_argument('--csv_root', type=str, required=True, help='CSV数据根目录')
    parser.add_argument('--max_samples', type=int, default=100, help='最大样本数')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Worker数量')
    parser.add_argument('--cache', action='store_true', help='启用内存缓存')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ECG V45 数据集加载测试")
    print("="*80)
    
    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(
        sim_root=args.sim_root,
        csv_root=args.csv_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_fs=500,
        max_samples=args.max_samples,
        cache_in_memory=args.cache,
        load_fine_labels=True,
        load_ocr_labels=True
    )
    
    # 测试加载
    print("测试数据加载...")
    start = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Image: {batch['image'].shape}")
        print(f"  Wave mask: {batch['wave_mask'].shape}")
        print(f"  Baseline coarse: {batch['baseline_coarse'].shape}")
        print(f"  Baseline fine: {batch['baseline_fine'].shape}")
        print(f"  Text multi: {batch['text_multi'].shape}")
        print(f"  Paper speed mask: {batch['paper_speed_mask'].shape} ⭐⭐⭐⭐⭐")
        print(f"  Gain mask: {batch['gain_mask'].shape} ⭐⭐⭐")
        print(f"  GT signal: {batch['gt_signal'].shape}")
        
        # 验证数据完整性
        assert not torch.isnan(batch['image']).any(), "Image contains NaN"
        assert not torch.isnan(batch['gt_signal']).any(), "Signal contains NaN"
        assert batch['wave_mask'].max() <= 12, "Wave mask invalid"
        
        # 检查OCR掩码
        paper_speed_coverage = (batch['paper_speed_mask'] > 0.5).float().mean()
        gain_coverage = (batch['gain_mask'] > 0.5).float().mean()
        print(f"  Paper speed coverage: {paper_speed_coverage.item()*100:.2f}%")
        print(f"  Gain coverage: {gain_coverage.item()*100:.2f}%")
    
    elapsed = time.time() - start
    print(f"\n✓ 加载10个batch耗时: {elapsed:.2f}秒 (平均 {elapsed/10:.3f}s/batch)")
    
    # 测试验证集
    print("\n测试验证集...")
    val_batch = next(iter(val_loader))
    print(f"✓ 验证集batch形状: {val_batch['image'].shape}")
    
    # 测试缓存效果
    if args.cache:
        print("\n测试缓存效果（第二次加载）...")
        start = time.time()
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break
        elapsed2 = time.time() - start
        print(f"✓ 第二次加载耗时: {elapsed2:.2f}秒")
        print(f"✓ 加速比: {elapsed/elapsed2:.2f}x")
    
    print("\n" + "="*80)
    print("✓ 数据集测试通过！")
    print("="*80)