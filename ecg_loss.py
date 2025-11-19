"""
ECG V47 Loss Functions (CRNN & Segmentation)

包含:
1. SegmentationLoss: 负责基线、文字、OCR的定位 (支持多尺度自动对齐)
2. SignalLoss: 负责波形数值回归 (支持 Mask 机制忽略 Padding)
3. CompositeLoss: 组合上述两者
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        loss = (1 - ((2. * intersection + self.smooth) / 
                     (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)))
        return loss.mean()

class FocalLoss(nn.Module):
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
    """
    通用分割损失：结合 Dice Loss 和 Focal Loss
    自动处理 Prediction 和 Target 的尺寸不匹配问题
    """
    def __init__(self, use_focal=True):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss() if use_focal else None

    def _align_target(self, pred, target):
        """将 Target 下采样到 Prediction 的尺寸"""
        if pred.shape[-2:] != target.shape[-2:]:
            # 使用 nearest 保持 0/1 掩码特性
            return F.interpolate(target, size=pred.shape[-2:], mode='nearest')
        return target

    def forward(self, pred, target):
        # 1. 对齐尺寸
        target = self._align_target(pred, target)
        
        # 2. 计算损失
        loss = self.dice(pred, target)
        
        if self.focal:
            # Focal Loss 需要 BCE 的输入在 (0, 1) 之间，pred 已经是 Sigmoid 后的
            loss += self.focal(pred, target)
            
        return loss

class SignalRegressionLoss(nn.Module):
    """
    信号回归损失
    仅在 valid_mask 为 1 的区域计算 Loss (忽略 Padding)
    """
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.criterion = nn.L1Loss(reduction='none') if loss_type == 'l1' else nn.MSELoss(reduction='none')

    def forward(self, pred_signals, gt_signals, valid_mask=None):
        """
        Args:
            pred_signals: (B, 12, W)
            gt_signals: (B, 12, W)
            valid_mask: (B, 12, W) or None
        """
        # 确保长度对齐 (Model输出可能因卷积取整与Target略有不同)
        min_len = min(pred_signals.shape[-1], gt_signals.shape[-1])
        pred = pred_signals[..., :min_len]
        gt = gt_signals[..., :min_len]
        
        # 计算逐点 Loss
        loss = self.criterion(pred, gt)
        
        # 应用掩码
        if valid_mask is not None:
            mask = valid_mask[..., :min_len]
            loss = loss * mask
            # 避免除以零
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()

class ProgressiveLeadLocalizationLoss(nn.Module):
    """
    总损失函数
    """
    def __init__(self, 
                 weight_coarse_baseline=1.0,
                 weight_text=1.0,
                 weight_paper_speed=1.0,
                 weight_lead_baseline=1.0,
                 weight_signal=10.0, # 信号回归通常数值较小，给予较大权重
                 use_focal_loss=True):
        super().__init__()
        
        self.weights = {
            'coarse': weight_coarse_baseline,
            'text': weight_text,
            'ocr': weight_paper_speed, # 共用 OCR 权重
            'fine': weight_lead_baseline,
            'signal': weight_signal
        }
        
        self.seg_loss_fn = SegmentationLoss(use_focal=use_focal_loss)
        self.sig_loss_fn = SignalRegressionLoss(loss_type='l1')

    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        losses = {}
        total_loss = 0.0
        
        # 1. 粗基线损失 (H/16)
        if 'coarse_baseline' in outputs:
            l = self.seg_loss_fn(outputs['coarse_baseline'], targets['baseline_coarse'])
            losses['loss_coarse'] = l
            total_loss += l * self.weights['coarse']
            
        # 2. 文字掩码损失 (H/4)
        if 'text_masks' in outputs:
            l = self.seg_loss_fn(outputs['text_masks'], targets['text_multi'])
            losses['loss_text'] = l
            total_loss += l * self.weights['text']
            
        # 3. OCR 损失 (Paper Speed & Gain) (H/4)
        if 'paper_speed_mask' in outputs: # 假设 targets 中暂无对应 GT，这里仅作示例
            # 如果 dataset 提供了 paper_speed mask，则计算
            # 目前 ecg_dataset.py 返回的是 metadata，没有 OCR mask GT
            # 实际训练中应在 Dataset 生成 OCR GT，这里先跳过或使用全黑/全白作为 Dummy 避免报错
            pass 
            
        # 4. 精细基线损失 (H/4) -> 最关键的定位信息
        if 'lead_baselines' in outputs:
            # 使用 baseline_fine (即 baseline_mask) 作为 GT
            l = self.seg_loss_fn(outputs['lead_baselines'], targets['baseline_fine'])
            losses['loss_fine'] = l
            total_loss += l * self.weights['fine']
            
        # 5. 信号回归损失
        if 'signals' in outputs and 'gt_signals' in targets:
            l = self.sig_loss_fn(
                outputs['signals'], 
                targets['gt_signals'], 
                targets.get('valid_mask', None)
            )
            losses['loss_signal'] = l
            total_loss += l * self.weights['signal']
            
        return total_loss, losses

# =============================================================================
# 模块测试
# =============================================================================
if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    device = torch.device('cpu')
    criterion = ProgressiveLeadLocalizationLoss()
    
    # Mock Data
    B = 2
    outputs = {
        'coarse_baseline': torch.sigmoid(torch.randn(B, 1, 32, 128)), # H/16
        'text_masks': torch.sigmoid(torch.randn(B, 13, 128, 512)),    # H/4
        'lead_baselines': torch.sigmoid(torch.randn(B, 12, 128, 512)),# H/4
        'signals': torch.randn(B, 12, 512)                            # W/4
    }
    
    targets = {
        'baseline_coarse': torch.randint(0, 2, (B, 1, 512, 2048)).float(), # Original Size
        'text_multi': torch.randint(0, 2, (B, 13, 512, 2048)).float(),
        'baseline_fine': torch.randint(0, 2, (B, 12, 512, 2048)).float(),
        'gt_signals': torch.randn(B, 12, 512),
        'valid_mask': torch.ones(B, 12, 512)
    }
    
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"Total Loss: {loss.item():.4f}")
    print("Components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
        
    print("\n✓ Loss function test passed!")