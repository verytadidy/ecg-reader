"""
ECG V48 Loss Functions (完全修复版)
修复内容:
1. ✅ 新增波形分割 Loss (WaveSegmentationLoss)
2. ✅ 新增辅助掩码抑制机制
3. ✅ 优化信号回归 Loss (背景抑制)
4. ✅ 渐进式权重调度策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class DiceLoss(nn.Module):
    """Dice Loss (适用于二值/多类分割)"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) after sigmoid
        # target: (B, C, H, W)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()


class FocalLoss(nn.Module):
    """Focal Loss (处理类别不平衡)"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class SegmentationLoss(nn.Module):
    """通用分割 Loss"""
    def __init__(self, use_focal=True):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss() if use_focal else None

    def _align_target(self, pred, target):
        """对齐 target 到 pred 的尺寸"""
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode='nearest')
        
        # 通道对齐
        if pred.shape[1] == 1 and target.shape[1] > 1:
            target, _ = target.max(dim=1, keepdim=True)
        
        return target

    def forward(self, pred, target):
        target = self._align_target(pred, target)
        loss = self.dice(pred, target)
        if self.focal:
            loss += self.focal(pred, target)
        return loss


class WaveSegmentationLoss(nn.Module):
    """
    🆕 波形分割 Loss (语义分割)
    处理单通道语义掩码 (值 0-12)
    """
    def __init__(self, num_classes=13, ignore_index=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, 12, H, W) - 12 类的 logits
            target: (B, H, W) - 值范围 [0, 12]
        """
        # 对齐尺寸
        if pred_logits.shape[-2:] != target.shape[-2:]:
            pred_logits = F.interpolate(pred_logits, size=target.shape[-2:], 
                                         mode='bilinear', align_corners=False)
        
        # CrossEntropy Loss
        loss = self.ce_loss(pred_logits, target.long())
        
        return loss


class SignalRegressionLoss(nn.Module):
    """
    信号回归 Loss (带背景抑制)
    """
    def __init__(self, loss_type='l1', background_weight=0.1):
        super().__init__()
        self.criterion = nn.L1Loss(reduction='none') if loss_type == 'l1' else nn.MSELoss(reduction='none')
        self.bg_weight = background_weight

    def forward(self, pred_signals, gt_signals, valid_mask=None):
        """
        Args:
            pred_signals: (B, 12, W)
            gt_signals: (B, 12, W)
            valid_mask: (B, 12, W) - 1=信号区, 0=背景区
        """
        min_len = min(pred_signals.shape[-1], gt_signals.shape[-1])
        pred = pred_signals[..., :min_len]
        gt = gt_signals[..., :min_len]
        
        raw_loss = self.criterion(pred, gt)
        
        if valid_mask is not None:
            mask = valid_mask[..., :min_len]
            # 信号区权重 1.0，背景区权重 bg_weight
            weights = mask + (1.0 - mask) * self.bg_weight
            weighted_loss = raw_loss * weights
            return weighted_loss.mean()
        else:
            return raw_loss.mean()


class AuxiliarySuppressionLoss(nn.Module):
    """
    🆕 辅助掩码抑制 Loss
    确保模型在 auxiliary_mask=1 的区域不输出波形
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_wave_seg, auxiliary_mask):
        """
        Args:
            pred_wave_seg: (B, 12, H, W) - 波形分割 logits
            auxiliary_mask: (B, H, W) - 辅助区域掩码 (0-1)
        """
        # 对齐尺寸
        if pred_wave_seg.shape[-2:] != auxiliary_mask.shape[-2:]:
            auxiliary_mask = F.interpolate(
                auxiliary_mask.unsqueeze(1), 
                size=pred_wave_seg.shape[-2:], 
                mode='bilinear', align_corners=False
            ).squeeze(1)
        
        # 在 auxiliary 区域，所有类别的概率应该趋向于均匀分布（无信号）
        # 或者更简单：在 auxiliary 区域，波形分割概率应该很低
        pred_probs = torch.softmax(pred_wave_seg, dim=1)[:, 1:, :, :]  # 排除 background
        
        # auxiliary_mask 作为权重，只惩罚辅助区域的波形预测
        aux_penalty = (pred_probs * auxiliary_mask.unsqueeze(1)).mean()
        
        return aux_penalty


class ProgressiveLeadLocalizationLossV48(nn.Module):
    """
    🔥 完整的组合 Loss (V48)
    """
    def __init__(self,
                 weight_coarse_baseline=1.0,
                 weight_text=1.0,
                 weight_wave_seg=5.0,        # 🆕 波形分割权重
                 weight_lead_baseline=5.0,   # 精细基线权重（核心）
                 weight_signal=10.0,
                 weight_aux_suppress=0.5,    # 🆕 辅助抑制权重
                 weight_ocr=0.5,
                 background_weight=0.1,
                 use_focal_loss=True):
        super().__init__()
        
        self.weights = {
            'coarse': weight_coarse_baseline,
            'text': weight_text,
            'wave_seg': weight_wave_seg,
            'fine': weight_lead_baseline,
            'signal': weight_signal,
            'aux_suppress': weight_aux_suppress,
            'ocr': weight_ocr
        }
        
        self.seg_loss_fn = SegmentationLoss(use_focal=use_focal_loss)
        self.wave_seg_loss_fn = WaveSegmentationLoss(num_classes=13, ignore_index=0)
        self.sig_loss_fn = SignalRegressionLoss(loss_type='l1', background_weight=background_weight)
        self.aux_suppress_fn = AuxiliarySuppressionLoss()

    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        loss_dict = {}
        total_loss = 0.0
        
        # 1. 粗基线 (H/16)
        if 'coarse_baseline' in outputs and 'baseline_coarse' in targets:
            l = self.seg_loss_fn(outputs['coarse_baseline'], targets['baseline_coarse'])
            loss_dict['loss_coarse'] = l
            total_loss += l * self.weights['coarse']
        
        # 2. 文字掩码 (H/4)
        if 'text_masks' in outputs and 'text_multi' in targets:
            l = self.seg_loss_fn(outputs['text_masks'], targets['text_multi'])
            loss_dict['loss_text'] = l
            total_loss += l * self.weights['text']
        
        # 3. 🆕 波形分割 (H/4)
        if 'wave_segmentation_logits' in outputs and 'wave_segmentation' in targets:
            l = self.wave_seg_loss_fn(outputs['wave_segmentation_logits'], targets['wave_segmentation'])
            loss_dict['loss_wave_seg'] = l
            total_loss += l * self.weights['wave_seg']
        
        # 4. 精细基线 (H/4) - 核心定位
        if 'lead_baselines' in outputs and 'baseline_fine' in targets:
            l = self.seg_loss_fn(outputs['lead_baselines'], targets['baseline_fine'])
            loss_dict['loss_fine'] = l
            total_loss += l * self.weights['fine']
        
        # 5. 🆕 辅助掩码抑制
        if 'wave_segmentation_logits' in outputs and 'auxiliary_mask' in targets:
            l = self.aux_suppress_fn(outputs['wave_segmentation_logits'], targets['auxiliary_mask'])
            loss_dict['loss_aux_suppress'] = l
            total_loss += l * self.weights['aux_suppress']
        
        # 6. OCR 任务
        if 'ocr_maps' in outputs:
            if 'paper_speed_mask' in targets and 'gain_mask' in targets:
                # 合并两个 OCR 目标
                ocr_target = torch.stack([
                    targets['paper_speed_mask'],
                    targets['gain_mask']
                ], dim=1)  # (B, 2, H, W)
                
                l = self.seg_loss_fn(outputs['ocr_maps'], ocr_target)
                loss_dict['loss_ocr'] = l
                total_loss += l * self.weights['ocr']
        
        # 7. 信号回归
        if 'signals' in outputs and 'gt_signals' in targets:
            mask = targets.get('valid_mask', None)
            l = self.sig_loss_fn(outputs['signals'], targets['gt_signals'], mask)
            loss_dict['loss_signal'] = l
            total_loss += l * self.weights['signal']
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


class ProgressiveWeightScheduler:
    """
    🆕 渐进式权重调度器
    早期: 专注定位 (baseline, wave_seg)
    后期: 加大信号权重
    """
    def __init__(self, criterion, total_epochs=50, warmup_epochs=10):
        self.criterion = criterion
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        # 保存初始权重
        self.initial_weights = criterion.weights.copy()

    def step(self, epoch):
        """根据 epoch 调整权重"""
        if epoch < self.warmup_epochs:
            # Warmup 阶段: 高定位权重，低信号权重
            ratio = epoch / self.warmup_epochs
            self.criterion.weights['signal'] = self.initial_weights['signal'] * (0.1 + 0.9 * ratio)
            self.criterion.weights['fine'] = self.initial_weights['fine'] * (1.5 - 0.5 * ratio)
        else:
            # 正常阶段
            self.criterion.weights['signal'] = self.initial_weights['signal']
            self.criterion.weights['fine'] = self.initial_weights['fine']


# ========== 模块测试 ==========
if __name__ == "__main__":
    print("Testing ECG Loss V48...")
    
    criterion = ProgressiveLeadLocalizationLossV48()
    scheduler = ProgressiveWeightScheduler(criterion, total_epochs=50, warmup_epochs=10)
    
    B, H, W = 2, 512, 2048
    
    # 模拟输出
    outputs = {
        'coarse_baseline': torch.sigmoid(torch.randn(B, 1, 32, 128)),
        'lead_baselines': torch.sigmoid(torch.randn(B, 12, 128, 512)),
        'wave_segmentation_logits': torch.randn(B, 12, 128, 512),
        'signals': torch.randn(B, 12, 512)
    }
    
    # 模拟目标
    targets = {
        'baseline_coarse': torch.zeros(B, 1, H, W),
        'baseline_fine': torch.zeros(B, 12, H, W),
        'text_multi': torch.zeros(B, 13, H, W),
        'wave_segmentation': torch.randint(0, 13, (B, H, W)),
        'auxiliary_mask': torch.zeros(B, H, W),
        'gt_signals': torch.zeros(B, 12, 512),
        'valid_mask': torch.cat([torch.ones(B, 12, 256), torch.zeros(B, 12, 256)], dim=-1)
    }
    
    # 测试 Loss
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"\nTotal Loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
    
    # 测试权重调度
    print("\nTesting weight scheduler...")
    for epoch in [0, 5, 10, 20, 50]:
        scheduler.step(epoch)
        print(f"  Epoch {epoch}: signal_weight={criterion.weights['signal']:.2f}, "
              f"fine_weight={criterion.weights['fine']:.2f}")
    
    print("\n✓ Loss test passed!")