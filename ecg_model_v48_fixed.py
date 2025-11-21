"""
ECG V48 Dataset (完全修复版)
修复内容:
1. ✅ 加载 label_wave.npy (波形分割标签)
2. ✅ 加载 label_auxiliary.npy (辅助标记)
3. ✅ 加载 OCR 掩码 (paper_speed, gain)
4. ✅ 使用 metadata.time_range 精确对齐信号
5. ✅ 优化内存缓存策略
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
from typing import Tuple, Dict, Optional
import warnings

class ECGV48FixedDataset(Dataset):
    def __init__(self,
                 sim_root_dir: str,
                 csv_root_dir: str,
                 split: str = 'train',
                 target_size: Tuple[int, int] = (512, 2048),
                 target_fs: int = 500,
                 max_samples: Optional[int] = None,
                 augment: bool = False,
                 cache_images: bool = True):
        
        self.sim_root = Path(sim_root_dir)
        self.csv_root = Path(csv_root_dir)
        self.split = split
        self.target_size = target_size
        self.target_fs = target_fs
        self.augment = augment
        self.cache_images = cache_images
        
        self.cache = {}
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        self.samples = self._scan_samples(max_samples)
        self.transform = self._build_transform()

        print(f"\n{'='*60}")
        print(f"ECG Dataset V48 Fixed ({split})")
        print(f"Samples: {len(self.samples)}")
        print(f"Target Size: {target_size}")
        print(f"Image Cache: {'✅ ON' if cache_images else '⚠️ OFF'}")
        print(f"{'='*60}\n")

    def _scan_samples(self, max_samples):
        samples = []
        if not self.sim_root.exists():
            warnings.warn(f"Sim root not found: {self.sim_root}")
            return []
        
        dirs = sorted([d for d in self.sim_root.iterdir() if d.is_dir()])
        for d in dirs:
            sid = d.name
            required_files = [
                f"{sid}_dirty.png",
                f"{sid}_gt_signals.json",
                f"{sid}_metadata.json"
            ]
            if all((d / f).exists() for f in required_files):
                samples.append({'id': sid, 'dir': d})
            if max_samples and len(samples) >= max_samples:
                break
        return samples

    def _build_transform(self):
        if self.augment:
            return A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
                A.CoarseDropout(max_holes=8, max_height=40, max_width=40, p=0.3),
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

    def _load_file(self, file_path, file_type, sample_id):
        """带缓存的文件加载"""
        cache_key = f"{sample_id}_{file_path.name}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        data = None
        try:
            if file_type == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif file_type == 'npy':
                if file_path.exists():
                    data = np.load(file_path)
            elif file_type == 'img':
                data = np.array(Image.open(file_path).convert('RGB'))
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            return None
            
        # 缓存策略
        if data is not None:
            if file_type == 'img':
                if self.cache_images:
                    self.cache[cache_key] = data
            else:
                self.cache[cache_key] = data
            
        return data

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sid = sample_info['id']
        sdir = sample_info['dir']
        
        h_tg, w_tg = self.target_size
        
        # ========== 1. 加载图像 ==========
        img_path = sdir / f"{sid}_dirty.png"
        image = self._load_file(img_path, 'img', sid)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        orig_h, orig_w = image.shape[:2]
        
        # ========== 2. 加载元数据 ==========
        meta = self._load_file(sdir / f"{sid}_metadata.json", 'json', sid)
        phys_params = meta.get('physical_params', {})
        gain = phys_params.get('gain_mm_mv', 10.0)
        speed = phys_params.get('paper_speed_mm_s', 25.0)
        lead_rois = meta.get('lead_rois', {})
        
        # ========== 3. 加载基线掩码 (12, H, W) ==========
        baseline_raw = self._load_file(sdir / f"{sid}_label_baseline.npy", 'npy', sid)
        if baseline_raw is not None and baseline_raw.shape[0] == 12:
            baseline_resized = np.zeros((12, h_tg, w_tg), dtype=np.float32)
            for i in range(12):
                baseline_resized[i] = cv2.resize(baseline_raw[i], (w_tg, h_tg), 
                                                  interpolation=cv2.INTER_LINEAR) / 255.0
        else:
            baseline_resized = np.zeros((12, h_tg, w_tg), dtype=np.float32)
            
        # ========== 4. 加载文字掩码 (13, H, W) ==========
        text_raw = self._load_file(sdir / f"{sid}_label_text_multi.npy", 'npy', sid)
        if text_raw is not None and text_raw.shape[0] == 13:
            text_resized = np.zeros((13, h_tg, w_tg), dtype=np.float32)
            for i in range(13):
                text_resized[i] = cv2.resize(text_raw[i], (w_tg, h_tg), 
                                              interpolation=cv2.INTER_NEAREST) / 255.0
        else:
            text_resized = np.zeros((13, h_tg, w_tg), dtype=np.float32)

        # ========== 5. 🆕 加载波形分割掩码 (H, W) ==========
        wave_seg_raw = self._load_file(sdir / f"{sid}_label_wave.npy", 'npy', sid)
        if wave_seg_raw is not None:
            wave_seg_resized = cv2.resize(wave_seg_raw, (w_tg, h_tg), 
                                           interpolation=cv2.INTER_NEAREST)
        else:
            wave_seg_resized = np.zeros((h_tg, w_tg), dtype=np.uint8)
            
        # ========== 6. 🆕 加载辅助掩码 (1, H, W) ==========
        aux_raw = self._load_file(sdir / f"{sid}_label_auxiliary.npy", 'npy', sid)
        if aux_raw is not None and len(aux_raw.shape) == 3:
            aux_resized = cv2.resize(aux_raw[0], (w_tg, h_tg), 
                                      interpolation=cv2.INTER_LINEAR) / 255.0
        else:
            aux_resized = np.zeros((h_tg, w_tg), dtype=np.float32)
            
        # ========== 7. 🆕 加载 OCR 掩码 ==========
        ps_raw = self._load_file(sdir / f"{sid}_label_paper_speed.npy", 'npy', sid)
        if ps_raw is not None and len(ps_raw.shape) == 3:
            ps_resized = cv2.resize(ps_raw[0], (w_tg, h_tg), 
                                     interpolation=cv2.INTER_LINEAR) / 255.0
        else:
            ps_resized = np.zeros((h_tg, w_tg), dtype=np.float32)
            
        gain_raw = self._load_file(sdir / f"{sid}_label_gain.npy", 'npy', sid)
        if gain_raw is not None and len(gain_raw.shape) == 3:
            gain_resized = cv2.resize(gain_raw[0], (w_tg, h_tg), 
                                       interpolation=cv2.INTER_LINEAR) / 255.0
        else:
            gain_resized = np.zeros((h_tg, w_tg), dtype=np.float32)

        # ========== 8. 🆕 精确信号对齐 (使用 time_range) ==========
        gt_data = self._load_file(sdir / f"{sid}_gt_signals.json", 'json', sid)
        
        feature_width = w_tg // 4  # 特征图宽度
        target_signals = np.zeros((12, feature_width), dtype=np.float32)
        valid_mask = np.zeros((12, feature_width), dtype=np.float32)
        
        for i, lead in enumerate(self.lead_names):
            # 从 GT JSON 加载原始信号
            raw_sig = gt_data['signals'].get(lead, None)
            if raw_sig is None or len(raw_sig) == 0:
                continue
            raw_sig = np.array(raw_sig, dtype=np.float32)
            
            # 🔥 关键修复: 使用 metadata 的 time_range
            if lead in lead_rois and lead_rois[lead] is not None:
                time_range = lead_rois[lead].get('time_range', [0.0, 2.5])
            else:
                time_range = [0.0, 2.5]  # 默认短导联
            
            time_start, time_end = time_range
            duration = time_end - time_start
            
            # 计算信号在特征图上的起止位置
            # 根据时间比例映射到特征图宽度
            total_duration = 10.0  # 假设整个图像横跨 10 秒 (长导联的时长)
            start_ratio = time_start / total_duration
            end_ratio = time_end / total_duration
            
            start_idx = int(start_ratio * feature_width)
            end_idx = int(end_ratio * feature_width)
            segment_len = end_idx - start_idx
            
            if segment_len > 0:
                # 重采样信号到特征图长度
                x_old = np.linspace(0, 1, len(raw_sig))
                x_new = np.linspace(0, 1, segment_len)
                resampled = np.interp(x_new, x_old, raw_sig)
                
                # 填充到目标数组
                target_signals[i, start_idx:end_idx] = resampled
                valid_mask[i, start_idx:end_idx] = 1.0
            
        # ========== 9. 图像增强与转换 ==========
        img_resized = cv2.resize(image, (w_tg, h_tg), interpolation=cv2.INTER_LINEAR)
        img_tensor = self.transform(image=img_resized)['image']
        
        # ========== 10. 返回完整数据 ==========
        return {
            'image': img_tensor,  # (3, H, W)
            
            # 分割标签
            'baseline_mask': torch.from_numpy(baseline_resized).float(),  # (12, H, W)
            'text_mask': torch.from_numpy(text_resized).float(),          # (13, H, W)
            'wave_segmentation': torch.from_numpy(wave_seg_resized).long(),  # (H, W)
            'auxiliary_mask': torch.from_numpy(aux_resized).float(),      # (H, W)
            
            # OCR 标签
            'paper_speed_mask': torch.from_numpy(ps_resized).float(),     # (H, W)
            'gain_mask': torch.from_numpy(gain_resized).float(),          # (H, W)
            
            # 信号回归标签
            'gt_signals': torch.from_numpy(target_signals).float(),       # (12, W/4)
            'valid_mask': torch.from_numpy(valid_mask).float(),           # (12, W/4)
            
            # 元数据
            'metadata': {
                'ecg_id': str(sid),
                'gain': float(gain),
                'speed': float(speed)
            }
        }


def create_dataloaders(sim_root, csv_root, batch_size=8, num_workers=4, 
                       train_split=0.9, **kwargs):
    """
    创建训练和验证 DataLoader
    """
    # 清理参数
    augment_flag = kwargs.pop('augment', None)
    cache_flag = kwargs.pop('cache_data', True)
    
    # 创建数据集
    train_ds = ECGV48FixedDataset(
        sim_root, csv_root, split='train', 
        augment=True,  # 训练集开启增强
        cache_images=cache_flag,
        **kwargs
    )
    
    val_ds = ECGV48FixedDataset(
        sim_root, csv_root, split='val',
        augment=False,  # 验证集关闭增强
        cache_images=cache_flag,
        **kwargs
    )
    
    # 划分训练/验证集
    dataset_size = len(train_ds)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(train_split * dataset_size))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    
    print(f"Dataset Split: Train={len(train_indices)}, Val={len(val_indices)}")
    
    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices)
    
    # DataLoader 配置
    worker_count = num_workers if num_workers > 0 else 0
    prefetch = 4 if worker_count > 0 else None
    
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=worker_count, pin_memory=True, drop_last=True,
        persistent_workers=(worker_count > 0), prefetch_factor=prefetch
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=worker_count, pin_memory=True, drop_last=False,
        persistent_workers=(worker_count > 0), prefetch_factor=prefetch
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_root', type=str, required=True)
    parser.add_argument('--csv_root', type=str, required=True)
    args = parser.parse_args()
    
    if Path(args.sim_root).exists():
        print("Testing dataset...")
        loader, _ = create_dataloaders(
            args.sim_root, args.csv_root, 
            batch_size=2, max_samples=10
        )
        
        for batch in loader:
            print("\nBatch keys:", batch.keys())
            print(f"Image: {batch['image'].shape}")
            print(f"Baseline: {batch['baseline_mask'].shape}")
            print(f"Wave Seg: {batch['wave_segmentation'].shape}")
            print(f"Auxiliary: {batch['auxiliary_mask'].shape}")
            print(f"Signals: {batch['gt_signals'].shape}")
            print(f"Valid Mask: {batch['valid_mask'].shape}")
            break
        
        print("\n✓ Dataset test passed!")