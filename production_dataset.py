"""
生产就绪的ECG数据集加载器

特点:
1. ✅ 支持多采样率（重采样到500Hz统一处理）
2. ✅ 直接从仿真器输出加载（无需预处理）
3. ✅ 高效缓存机制（可选）
4. ✅ 完善的错误处理
5. ✅ 数据验证
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
from typing import Optional, Tuple, Dict
import warnings


class ECGProductionDataset(Dataset):
    """
    生产就绪的ECG数据集
    
    特性:
    - 自动重采样到统一采样率（默认500Hz）
    - 数据验证和错误恢复
    - 可选的内存缓存
    """
    def __init__(self,
                 sim_root_dir: str,
                 csv_root_dir: str,
                 target_size: Tuple[int, int] = (512, 672),
                 target_fs: int = 500,
                 max_samples: Optional[int] = None,
                 cache_in_memory: bool = False,
                 split: str = 'train'):
        """
        Args:
            sim_root_dir: 仿真数据根目录
            csv_root_dir: 原始CSV数据根目录
            target_size: 统一resize尺寸 (H, W)
            target_fs: 目标采样率（所有信号重采样到此采样率）
            max_samples: 最大样本数（用于快速测试）
            cache_in_memory: 是否缓存到内存（需要足够RAM）
            split: 'train' 或 'val'
        """
        self.sim_root = Path(sim_root_dir)
        self.csv_root = Path(csv_root_dir)
        self.target_size = target_size
        self.target_fs = target_fs
        self.split = split
        self.cache_in_memory = cache_in_memory
        
        # 扫描所有有效样本
        self.samples = self._scan_samples(max_samples)
        
        # 内存缓存（可选）
        self.cache = {} if cache_in_memory else None
        
        # 数据变换（仅归一化，仿真器已做退化）
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 统计信息
        self._print_statistics()
    
    def _scan_samples(self, max_samples: Optional[int]) -> list:
        """扫描并验证所有样本"""
        samples = []
        
        for var_dir in self.sim_root.iterdir():
            if not var_dir.is_dir():
                continue
            
            variation_id = var_dir.name
            
            # 检查必需文件
            required_files = [
                f"{variation_id}_dirty.png",
                f"{variation_id}_label_wave.png",
                f"{variation_id}_label_baseline.npy",
                f"{variation_id}_label_grid.png",
                f"{variation_id}_metadata.json"
            ]
            
            if not all((var_dir / f).exists() for f in required_files):
                warnings.warn(f"样本 {variation_id} 文件不完整，跳过")
                continue
            
            # 加载元数据验证
            try:
                with open(var_dir / f"{variation_id}_metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                # 验证关键字段
                if not all(k in metadata for k in ['ecg_id', 'fs', 'sig_len']):
                    warnings.warn(f"样本 {variation_id} 元数据不完整，跳过")
                    continue
                
                samples.append({
                    'var_dir': var_dir,
                    'variation_id': variation_id,
                    'metadata': metadata
                })
                
                if max_samples and len(samples) >= max_samples:
                    break
                    
            except Exception as e:
                warnings.warn(f"样本 {variation_id} 元数据读取失败: {e}")
                continue
        
        if len(samples) == 0:
            raise RuntimeError(f"在 {self.sim_root} 未找到有效样本！")
        
        return samples
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        print(f"\n{'='*70}")
        print(f"ECG数据集加载完成 (split={self.split})")
        print(f"{'='*70}")
        print(f"总样本数: {len(self.samples)}")
        print(f"目标尺寸: {self.target_size[0]}×{self.target_size[1]}")
        print(f"目标采样率: {self.target_fs} Hz")
        
        # 统计原始采样率分布
        fs_counts = {}
        layout_counts = {}
        deg_counts = {}
        
        for sample in self.samples:
            fs = sample['metadata']['fs']
            layout = sample['metadata']['layout_type']
            deg = sample['metadata']['degradation_type']
            
            fs_counts[fs] = fs_counts.get(fs, 0) + 1
            layout_counts[layout] = layout_counts.get(layout, 0) + 1
            deg_counts[deg] = deg_counts.get(deg, 0) + 1
        
        print(f"\n原始采样率分布:")
        for fs, count in sorted(fs_counts.items()):
            pct = count / len(self.samples) * 100
            status = "→ 重采样" if fs != self.target_fs else "✓ 保持"
            print(f"  {fs:3d}Hz: {count:5d} ({pct:5.1f}%) {status}")
        
        print(f"\n布局类型分布:")
        for layout, count in sorted(layout_counts.items()):
            pct = count / len(self.samples) * 100
            print(f"  {layout:15s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n退化类型分布:")
        for deg, count in sorted(deg_counts.items()):
            pct = count / len(self.samples) * 100
            print(f"  {deg:15s}: {count:5d} ({pct:5.1f}%)")
        
        if self.cache_in_memory:
            estimated_memory = len(self.samples) * 10  # 粗略估算 ~10MB/sample
            print(f"\n⚠️  内存缓存已启用，预计占用 ~{estimated_memory}MB RAM")
        
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """加载单个样本"""
        # 检查缓存
        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]
        
        sample_info = self.samples[idx]
        var_dir = sample_info['var_dir']
        variation_id = sample_info['variation_id']
        metadata = sample_info['metadata']
        
        try:
            # 1. 加载图像数据
            dirty_img = cv2.imread(str(var_dir / f"{variation_id}_dirty.png"))
            wave_seg = cv2.imread(str(var_dir / f"{variation_id}_label_wave.png"), cv2.IMREAD_GRAYSCALE)
            baseline_heatmaps = np.load(str(var_dir / f"{variation_id}_label_baseline.npy"))
            grid_mask = cv2.imread(str(var_dir / f"{variation_id}_label_grid.png"), cv2.IMREAD_GRAYSCALE)
            
            # 几何变换（可选）
            geometric_transform = metadata.get('geometric_transform', None)
            if geometric_transform is not None:
                theta_gt = torch.tensor(geometric_transform, dtype=torch.float32)[:2, :]
            else:
                theta_gt = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
            
            # 2. Resize到统一尺寸
            h_target, w_target = self.target_size
            
            dirty_resized = cv2.resize(dirty_img, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
            wave_seg_resized = cv2.resize(wave_seg, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
            grid_mask_resized = cv2.resize(grid_mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
            
            baseline_resized = np.zeros((12, h_target, w_target), dtype=np.float32)
            for i in range(12):
                baseline_resized[i] = cv2.resize(baseline_heatmaps[i], (w_target, h_target), interpolation=cv2.INTER_LINEAR)
            
            # 3. 加载原始信号
            ecg_id = metadata['ecg_id']
            original_fs = metadata['fs']
            sig_len = metadata['sig_len']
            
            csv_path = self.csv_root / ecg_id / f"{ecg_id}.csv"
            gt_signal = self._load_and_resample_signal(csv_path, original_fs, sig_len, self.target_fs)

            signal_mask = create_signal_mask_from_csv(csv_path, sig_len, num_leads=12)
            
            # 4. 数据变换
            transformed = self.transform(image=dirty_resized)
            image = transformed['image']
            
            wave_seg_tensor = torch.from_numpy(wave_seg_resized).long()
            grid_mask_tensor = torch.from_numpy(grid_mask_resized).float() / 255.0
            baseline_tensor = torch.from_numpy(baseline_resized).float()
            
            result = {
                'image': image,
                'wave_seg': wave_seg_tensor,
                'grid_mask': grid_mask_tensor.unsqueeze(0),
                'baseline_heatmaps': baseline_tensor,
                'theta_gt': theta_gt,
                'gt_signal': gt_signal,
                'metadata': {
                    'variation_id': variation_id,
                    'ecg_id': ecg_id,
                    'original_fs': original_fs,
                    'target_fs': self.target_fs,
                    'physical_params': metadata['physical_params']
                }
            }
            
            # 缓存到内存
            if self.cache_in_memory:
                self.cache[idx] = result
            
            return result
            
        except Exception as e:
            warnings.warn(f"加载样本 {variation_id} 失败: {e}，返回空样本")
            # 返回一个dummy样本避免训练中断
            return self._get_dummy_sample()
    
    def _load_and_resample_signal(self, csv_path: Path, original_fs: int, sig_len: int, target_fs: int) -> torch.Tensor:
        """
        加载并重采样信号到目标采样率
        
        🔥 修复：正确处理CSV中的NaN值
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
        df = pd.read_csv(csv_path)
        
        # 提取12导联
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        signal_list = []
        
        for lead in leads:
            if lead in df.columns:
                sig = df[lead].values[:sig_len]
                
                # 🔥 关键修复：处理NaN值
                # 方案1: 将NaN替换为0（表示该时间段没有信号）
                sig = np.nan_to_num(sig, nan=0.0)
                
                # 如果信号不足长度，填充0
                if len(sig) < sig_len:
                    sig = np.pad(sig, (0, sig_len - len(sig)), mode='constant', constant_values=0)
            else:
                # 导联完全不存在，用全0
                sig = np.zeros(sig_len)
            
            signal_list.append(sig)
        
        signal = np.stack(signal_list, axis=1)  # (T_original, 12)
        
        # 🔥 二次检查：确保没有NaN进入重采样
        if np.isnan(signal).any():
            warnings.warn(f"CSV {csv_path.name} 中发现NaN，已替换为0")
            signal = np.nan_to_num(signal, nan=0.0)
        
        # 重采样到目标采样率
        if original_fs != target_fs:
            signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)  # (1, T_original, 12)
            signal_tensor = signal_tensor.permute(0, 2, 1)  # (1, 12, T_original)
            
            target_length = target_fs * 10  # 10秒
            signal_resampled = F.interpolate(
                signal_tensor,
                size=target_length,
                mode='linear',
                align_corners=True
            )  # (1, 12, target_length)
            
            signal_resampled = signal_resampled.permute(0, 2, 1).squeeze(0)  # (target_length, 12)
        else:
            signal_resampled = torch.from_numpy(signal).float()
        
        # 🔥 最终检查：确保输出没有NaN
        if torch.isnan(signal_resampled).any():
            warnings.warn(f"重采样后发现NaN，强制替换为0")
            signal_resampled = torch.nan_to_num(signal_resampled, nan=0.0)
        
        return signal_resampled
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """返回一个dummy样本（用于错误恢复）"""
        h, w = self.target_size
        return {
            'image': torch.zeros(3, h, w),
            'wave_seg': torch.zeros(h, w, dtype=torch.long),
            'grid_mask': torch.zeros(1, h, w),
            'baseline_heatmaps': torch.zeros(12, h, w),
            'theta_gt': torch.eye(2, 3),
            'gt_signal': torch.zeros(self.target_fs * 10, 12),
            'metadata': {'variation_id': 'dummy'}
        }


def create_dataloaders(sim_root: str,
                       csv_root: str,
                       batch_size: int = 4,
                       num_workers: int = 4,
                       train_split: float = 0.9,
                       target_fs: int = 500,
                       max_samples: Optional[int] = None,
                       cache_in_memory: bool = False) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建训练和验证DataLoader的便捷函数
    
    Returns:
        train_loader, val_loader
    """
    # 创建完整数据集
    full_dataset = ECGProductionDataset(
        sim_root_dir=sim_root,
        csv_root_dir=csv_root,
        target_size=(512, 672),
        target_fs=target_fs,
        max_samples=max_samples,
        cache_in_memory=cache_in_memory,
        split='train'
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
    
    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),  # 保持worker进程
        prefetch_factor=2 if num_workers > 0 else None  # 预加载2个batch
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
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print()
    
    return train_loader, val_loader


# ========== 测试代码 ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试数据集加载')
    parser.add_argument('--sim_root', type=str, required=True)
    parser.add_argument('--csv_root', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--cache', action='store_true', help='启用内存缓存')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("数据集加载测试")
    print("="*70)
    
    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(
        sim_root=args.sim_root,
        csv_root=args.csv_root,
        batch_size=4,
        num_workers=2,
        target_fs=500,
        max_samples=args.max_samples,
        cache_in_memory=args.cache
    )
    
    # 测试加载速度
    import time
    
    print("测试训练集加载速度...")
    start = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只测试10个batch
            break
        
        print(f"Batch {i+1}:")
        print(f"  Image: {batch['image'].shape}")
        print(f"  Signal: {batch['gt_signal'].shape}")
        print(f"  原始fs: {batch['metadata']['original_fs']}")
        
        # 验证重采样正确性
        for j in range(len(batch['gt_signal'])):
            expected_length = batch['metadata']['target_fs'][j] * 10
            actual_length = batch['gt_signal'][j].shape[0]
            assert actual_length == expected_length, f"信号长度不匹配: {actual_length} vs {expected_length}"
    
    elapsed = time.time() - start
    print(f"\n✓ 加载10个batch耗时: {elapsed:.2f}秒 (平均 {elapsed/10:.3f}s/batch)")
    
    # 测试缓存效果
    if args.cache:
        print("\n测试缓存效果（第二次加载）...")
        start = time.time()
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break
        elapsed2 = time.time() - start
        print(f"✓ 第二次加载10个batch耗时: {elapsed2:.2f}秒")
        print(f"✓ 加速比: {elapsed/elapsed2:.2f}x")
    
    print("\n" + "="*70)
    print("✓ 数据集测试通过！")
    print("="*70)