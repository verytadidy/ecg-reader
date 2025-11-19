"""
ECG V45 多任务损失函数 (修正版)

包含7个任务:
1. 粗粒度基线 (coarse_baseline) -> 自动适配 H/16
2. 时间范围估计 (time_range)
3. 导联文字 (text) ⭐⭐⭐ -> 自动适配 H/4
4. 辅助元素 (auxiliary) -> 自动适配 H/4
5. 纸速OCR (paper_speed) ⭐⭐⭐⭐⭐ -> 自动适配 H/4
6. 增益OCR (gain) ⭐⭐⭐ -> 自动适配 H/4
7. 细粒度基线 (lead_baseline) ⭐⭐⭐⭐ -> 自动适配 H/4

修正: 增加自动尺寸对齐，解决多尺度输出与GT尺寸不一致的问题
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


class ProgressiveLeadLocalizationLoss(nn.Module):
    """
    渐进式导联定位的多任务损失
    """
    
    def __init__(self,
                 weight_coarse_baseline: float = 1.0,
                 weight_time_range: float = 0.5,
                 weight_text: float = 2.0,
                 weight_auxiliary: float = 1.0,
                 weight_paper_speed: float = 5.0,
                 weight_gain: float = 3.0,
                 weight_lead_baseline: float = 2.0,
                 use_focal_loss: bool = True):
        super().__init__()
        
        self.weight_coarse_baseline = weight_coarse_baseline
        self.weight_time_range = weight_time_range
        self.weight_text = weight_text
        self.weight_auxiliary = weight_auxiliary
        self.weight_paper_speed = weight_paper_speed
        self.weight_gain = weight_gain
        self.weight_lead_baseline = weight_lead_baseline
        self.use_focal_loss = use_focal_loss
        
        print(f"\n{'='*80}")
        print("损失函数权重配置 (已启用多尺度自动对齐):")
        print(f"{'='*80}")
        print(f"  纸速OCR:         {weight_paper_speed:.1f}  ⭐⭐⭐⭐⭐")
        print(f"  增益OCR:         {weight_gain:.1f}  ⭐⭐⭐")
        print(f"  导联文字:        {weight_text:.1f}  ⭐⭐⭐")
        print(f"  细粒度基线:      {weight_lead_baseline:.1f}  ⭐⭐⭐⭐")
        print(f"  粗粒度基线:      {weight_coarse_baseline:.1f}")
        print(f"  辅助元素:        {weight_auxiliary:.1f}")
        print(f"  时间范围:        {weight_time_range:.1f}")
        print(f"{'='*80}\n")

    def _resize_target_if_needed(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        [核心修复] 自动将Target下采样到Prediction的尺寸
        
        Args:
            pred: (B, C, h, w)
            target: (B, C, H, W)
            
        Returns:
            resized_target: (B, C, h, w)
        """
        if pred.shape[-2:] != target.shape[-2:]:
            # 使用 nearest 模式保持掩码的二值特性 (0/1)
            return F.interpolate(target, size=pred.shape[-2:], mode='nearest')
        return target
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        device = next(iter(outputs.values())).device
        
        # ========== 1. 粗粒度基线损失 (H/16) ==========
        if 'coarse_baseline' in outputs and 'baseline_coarse' in targets:
            loss_coarse = self.baseline_loss(
                outputs['coarse_baseline'],
                targets['baseline_coarse']
            )
            losses['coarse_baseline'] = loss_coarse * self.weight_coarse_baseline
        
        # ========== 2. 时间范围损失 (L1) ==========
        if 'time_ranges' in outputs and 'metadata' in targets:
            # 获取当前预测的 Batch Size
            current_bs = outputs['time_ranges'].shape[0]
            
            time_ranges_gt = self.extract_time_ranges_from_metadata(
                targets['metadata'], 
                device,
                current_batch_size=current_bs  # <--- 传入这个新参数
            )
            
            loss_time = F.l1_loss(outputs['time_ranges'], time_ranges_gt)
            losses['time_range'] = loss_time * self.weight_time_range
        
        # ========== 3. 导联文字损失 (H/4) ==========
        if 'text_masks' in outputs and 'text_multi' in targets:
            loss_text = self.multi_channel_mask_loss(
                outputs['text_masks'],
                targets['text_multi']
            )
            losses['text'] = loss_text * self.weight_text
        
        # ========== 4. 辅助元素损失 (H/4) ==========
        if 'auxiliary_mask' in outputs and 'auxiliary' in targets:
            loss_aux = self.baseline_loss(
                outputs['auxiliary_mask'],
                targets['auxiliary']
            )
            losses['auxiliary'] = loss_aux * self.weight_auxiliary
        
        # ========== 5. 纸速OCR损失 (H/4) ==========
        if 'paper_speed_mask' in outputs and 'paper_speed_mask' in targets:
            loss_paper_speed = self.ocr_loss(
                outputs['paper_speed_mask'],
                targets['paper_speed_mask']
            )
            losses['paper_speed'] = loss_paper_speed * self.weight_paper_speed
        
        # ========== 6. 增益OCR损失 (H/4) ==========
        if 'gain_mask' in outputs and 'gain_mask' in targets:
            loss_gain = self.ocr_loss(
                outputs['gain_mask'],
                targets['gain_mask']
            )
            losses['gain'] = loss_gain * self.weight_gain
        
        # ========== 7. 细粒度导联基线损失 (H/4) ==========
        if 'lead_baselines' in outputs and 'baseline_fine' in targets:
            loss_lead_baseline = self.multi_channel_mask_loss(
                outputs['lead_baselines'],
                targets['baseline_fine']
            )
            losses['lead_baseline'] = loss_lead_baseline * self.weight_lead_baseline
        
        # ========== 总损失 ==========
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def baseline_loss(self, 
                      pred: torch.Tensor, 
                      target: torch.Tensor) -> torch.Tensor:
        # 1. 自动对齐尺寸
        target = self._resize_target_if_needed(pred, target)
        
        # 2. BCE Loss
        bce = F.binary_cross_entropy(pred, target, reduction='mean')
        
        # 3. Dice Loss (传入已对齐的target)
        dice = self.dice_loss(pred, target)
        
        return bce + dice
    
    def multi_channel_mask_loss(self,
                                pred: torch.Tensor,
                                target: torch.Tensor) -> torch.Tensor:
        # 1. 整体对齐尺寸 (比在循环中每次对齐更高效)
        target = self._resize_target_if_needed(pred, target)
        
        num_channels = pred.shape[1]
        loss = 0
        
        for i in range(num_channels):
            # 这里直接调用，target已经对齐
            loss += self.baseline_loss(
                pred[:, i:i+1], 
                target[:, i:i+1]
            )
        
        return loss / num_channels
    
    def ocr_loss(self,
                 pred: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        # OCR Loss 也会调用 focal 或 baseline，那里会处理对齐
        # 但为了安全起见，这里也可以显式对齐
        target = self._resize_target_if_needed(pred, target)
        
        if self.use_focal_loss:
            return self.focal_loss(pred, target, alpha=0.25, gamma=2.0)
        else:
            return self.baseline_loss(pred, target)
    
    def dice_loss(self,
                  pred: torch.Tensor,
                  target: torch.Tensor,
                  smooth: float = 1e-5) -> torch.Tensor:
        # 1. 自动对齐尺寸
        target = self._resize_target_if_needed(pred, target)
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_coeff = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice_coeff
    
    def focal_loss(self,
                   pred: torch.Tensor,
                   target: torch.Tensor,
                   alpha: float = 0.25,
                   gamma: float = 2.0) -> torch.Tensor:
        # 1. 自动对齐尺寸
        target = self._resize_target_if_needed(pred, target)
        
        # BCE
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # pt = p if target == 1 else 1-p
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # Focal weight
        focal_weight = alpha * (1 - pt) ** gamma
        
        # Focal loss
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()
    
    def extract_time_ranges_from_metadata(self,
                                          metadata_list: list,
                                          device: torch.device,
                                          current_batch_size: int = None) -> torch.Tensor:
        """
        从metadata提取时间范围 (终极鲁棒版: 支持List, Tuple 和 Dict)
        """
        LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # 1. [修复] 处理 DataLoader 自动 Collate 产生的字典结构
        # 如果 metadata_list 是 {'lead_rois': [sample1, sample2...]} 这种结构
        if isinstance(metadata_list, dict):
            if 'lead_rois' in metadata_list:
                metadata_list = metadata_list['lead_rois']
            else:
                # 如果是其他未知的字典结构，无法提取，只能回退到默认值
                # 创建一个空列表，长度为 current_batch_size (如果有) 或 0
                fallback_len = current_batch_size if current_batch_size is not None else 0
                metadata_list = [None] * fallback_len

        # 2. [修复] 强制对齐 Batch Size (仅对序列类型切片)
        if isinstance(metadata_list, (list, tuple)):
            if current_batch_size is not None and len(metadata_list) > current_batch_size:
                metadata_list = metadata_list[:current_batch_size]
            
        batch_time_ranges = []
        
        for idx, metadata in enumerate(metadata_list):
            final_rois = {} # 最终提取出的 ROI 字典
            
            # 3. [修复] 解析单个样本
            # 情况 A: None 或 NaN
            if metadata is None:
                pass
            elif isinstance(metadata, float) and (metadata != metadata): # NaN
                pass
            
            # 情况 B: JSON 字符串 (通常来自 CSV 读取)
            elif isinstance(metadata, str):
                s_meta = metadata.strip()
                if s_meta and s_meta.lower() != 'nan':
                    try:
                        if "'" in s_meta: s_meta = s_meta.replace("'", '"')
                        parsed = json.loads(s_meta)
                        if isinstance(parsed, dict):
                            final_rois = parsed.get('lead_rois', {})
                    except Exception:
                        if idx == 0: print(f"Warning: JSON parse failed")

            # 情况 C: 已经是字典 (来自 Collate 后的结果)
            elif isinstance(metadata, dict):
                # 如果字典里有 'lead_rois' Key，取里面的；否则假设整个字典就是 ROI
                final_rois = metadata.get('lead_rois', metadata)
            
            # 4. 提取时间范围 (应用默认值)
            time_ranges = []
            for lead_name in LEAD_NAMES:
                # 检查 final_rois 中是否有该导联的时间数据
                if lead_name in final_rois and 'time_range' in final_rois[lead_name]:
                    # 确保 time_range 是列表且长度为2
                    tr = final_rois[lead_name]['time_range']
                    if isinstance(tr, (list, tuple)) and len(tr) >= 2:
                         time_ranges.append([float(tr[0]), float(tr[1])])
                    else:
                         # 格式不对，回退默认
                         time_ranges.append([0.0, 10.0] if lead_name == 'II' else [0.0, 2.5])
                else:
                    # 默认值
                    if lead_name == 'II':
                        time_range = [0.0, 10.0]
                    else:
                        time_range = [0.0, 2.5]
                    time_ranges.append(time_range)
            
            batch_time_ranges.append(time_ranges)
        
        return torch.tensor(batch_time_ranges, dtype=torch.float32, device=device)
    
class CombinedLossWithConstraints(nn.Module):
    """
    带物理约束的损失函数 (保持不变，物理约束是在1D信号上计算，无需Resize)
    """
    
    def __init__(self,
                 base_loss: ProgressiveLeadLocalizationLoss,
                 weight_constraint: float = 0.1):
        super().__init__()
        self.base_loss = base_loss
        self.weight_constraint = weight_constraint
        
        if weight_constraint > 0:
            print(f"✓ 物理约束已启用，权重: {weight_constraint}")
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        total_loss, loss_dict = self.base_loss(outputs, targets)
        
        if self.weight_constraint > 0 and 'signals' in outputs:
            constraint_loss = self.physical_constraint_loss(
                outputs['signals'],
                targets['metadata']
            )
            
            if constraint_loss is not None:
                loss_dict['constraint'] = constraint_loss * self.weight_constraint
                total_loss = total_loss + constraint_loss * self.weight_constraint
        
        return total_loss, loss_dict
    
    def physical_constraint_loss(self,
                                 signals: list,
                                 metadata_list: list) -> torch.Tensor:
        # ... (代码与之前相同，省略以节省空间) ...
        total_loss = 0
        num_constraints = 0
        
        for batch_idx, signals_batch in enumerate(signals):
            metadata = metadata_list[batch_idx]
            lead_corruption = metadata.get('lead_corruption', {})
            if len(lead_corruption) == 0:
                continue 
            
            if len(signals_batch) != 12:
                continue
            
            # Einthoven
            sig_I = signals_batch[0]
            sig_II = signals_batch[1]
            sig_III = signals_batch[2]
            
            min_len = min(len(sig_I), len(sig_II), len(sig_III))
            if min_len > 0:
                einthoven_error = F.l1_loss(
                    sig_I[:min_len] + sig_III[:min_len],
                    sig_II[:min_len]
                )
                total_loss += einthoven_error
                num_constraints += 1
            
            # Augmented Sum
            sig_aVR = signals_batch[3]
            sig_aVL = signals_batch[4]
            sig_aVF = signals_batch[5]
            
            min_len = min(len(sig_aVR), len(sig_aVL), len(sig_aVF))
            if min_len > 0:
                augmented_sum = sig_aVR[:min_len] + sig_aVL[:min_len] + sig_aVF[:min_len]
                augmented_error = F.l1_loss(
                    augmented_sum,
                    torch.zeros_like(augmented_sum)
                )
                total_loss += augmented_error
                num_constraints += 1
        
        if num_constraints == 0:
            return None
        
        return total_loss / num_constraints

# ============================================================
# 测试代码 (验证多尺度自动对齐)
# ============================================================

if __name__ == '__main__':
    print("="*80)
    print("ECG V45 损失函数测试 (含多尺度对齐验证)")
    print("="*80)
    
    # 1. 初始化损失函数
    criterion = ProgressiveLeadLocalizationLoss(
        weight_coarse_baseline=1.0,
        weight_text=2.0,
        weight_paper_speed=5.0,
        use_focal_loss=True
    )
    
    # 2. 构造多尺度不匹配的数据 (模拟真实报错场景)
    B = 2
    H_orig, W_orig = 512, 512  # 原始输入尺寸
    
    print(f"原始输入尺寸: ({H_orig}, {W_orig})")
    
    # === 模型输出 (经过下采样的特征图) ===
    outputs = {
        # H/16 (32x32) - 之前报错的地方
        'coarse_baseline': torch.rand(B, 1, H_orig//16, W_orig//16),
        
        # H/4 (128x128)
        'text_masks': torch.rand(B, 13, H_orig//4, W_orig//4),
        'paper_speed_mask': torch.rand(B, 1, H_orig//4, W_orig//4),
        'gain_mask': torch.rand(B, 1, H_orig//4, W_orig//4),
        'lead_baselines': torch.rand(B, 12, H_orig//4, W_orig//4),
        
        # 向量数据
        'time_ranges': torch.rand(B, 12, 2)
    }
    
    # === 标签数据 (全分辨率) ===
    targets = {
        # H (512x512) - 标签通常是原图大小
        'baseline_coarse': torch.rand(B, 1, H_orig, W_orig),
        'text_multi': torch.rand(B, 13, H_orig, W_orig),
        'paper_speed_mask': torch.rand(B, 1, H_orig, W_orig),
        'gain_mask': torch.rand(B, 1, H_orig, W_orig),
        'baseline_fine': torch.rand(B, 12, H_orig, W_orig),
        
        # Metadata
        'metadata': [{'lead_rois': {}} for _ in range(B)]
    }
    
    print("\n数据形状检查:")
    print(f"  Coarse Pred (H/16): {outputs['coarse_baseline'].shape}")
    print(f"  Coarse GT   (H/1):  {targets['baseline_coarse'].shape}")
    print(f"  Text Pred   (H/4):  {outputs['text_masks'].shape}")
    print(f"  Text GT     (H/1):  {targets['text_multi'].shape}")
    
    # 3. 运行计算 (如果不报错，说明自动 Resize 生效)
    print("\n正在计算损失 (测试自动Resize)...")
    try:
        total_loss, loss_dict = criterion(outputs, targets)
        print("✓ 计算成功！没有发生尺寸不匹配错误。")
        
        print("\n损失明细:")
        print("-" * 60)
        for k, v in loss_dict.items():
            print(f"  {k:20s}: {v.item():.4f}")
            
    except RuntimeError as e:
        print(f"\n❌ 计算失败: {e}")
        print("提示: 请检查 _resize_target_if_needed 是否正确被调用")
        exit(1)

    # 4. 单独验证 Focal Loss 的对齐能力
    print("\n测试 Focal Loss 对齐...")
    focal_pred = torch.rand(B, 1, 32, 32) # 小图
    focal_target = torch.randint(0, 2, (B, 1, 128, 128)).float() # 大图
    
    try:
        fl = criterion.focal_loss(focal_pred, focal_target)
        print(f"✓ Focal Loss 计算成功: {fl.item():.4f}")
    except Exception as e:
        print(f"❌ Focal Loss 失败: {e}")

    print("\n" + "="*80)
    print("所有测试通过")
    print("="*80)