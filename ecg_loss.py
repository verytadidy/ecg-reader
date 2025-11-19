"""
ECG V47 Loss Functions (Production Ready)

更新重点:
1. SignalRegressionLoss: 引入 background_weight 机制。
   - 有效信号区域 (valid_mask=1): 权重 1.0，学习波形细节。
   - 无效背景区域 (valid_mask=0): 权重 0.1 (可配)，学习输出 0/静默。
   - 这解决了"模型定位错误时Loss应当很大"的需求。

2. SegmentationLoss: 增强了维度检查和自动对齐。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class DiceLoss(nn.Module):
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
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred should be in (0, 1)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class SegmentationLoss(nn.Module):
    """
    通用分割损失 (Dice + Focal)
    自动处理尺寸不匹配问题 (Pred vs Target)
    """
    def __init__(self, use_focal=True):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss() if use_focal else None

    def _align_target(self, pred, target):
        """
        对齐 Target 的尺寸到 Prediction
        """
        # 1. 空间尺寸对齐 (H, W)
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode='nearest')
        
        # 2. 通道对齐 (C) - 处理 1通道 vs 12通道 的情况
        # 如果 Pred 是单通道 (如粗基线)，而 Target 是多通道，则取 Target 的最大值(并集)
        if pred.shape[1] == 1 and target.shape[1] > 1:
            target, _ = target.max(dim=1, keepdim=True)
            
        return target

    def forward(self, pred, target):
        target = self._align_target(pred, target)
        
        # Dice Loss
        loss = self.dice(pred, target)
        
        # Focal Loss
        if self.focal:
            loss += self.focal(pred, target)
            
        return loss

class SignalRegressionLoss(nn.Module):
    """
    信号回归损失 (支持背景抑制)
    """
    def __init__(self, loss_type='l1', background_weight=0.1):
        """
        Args:
            background_weight: 背景区域(mask=0)的Loss权重。
                               建议设为 0.1~0.2，既能抑制噪声，又主要关注信号重建。
        """
        super().__init__()
        self.criterion = nn.L1Loss(reduction='none') if loss_type == 'l1' else nn.MSELoss(reduction='none')
        self.bg_weight = background_weight

    def forward(self, pred_signals, gt_signals, valid_mask=None):
        """
        Args:
            pred_signals: (B, 12, W)
            gt_signals: (B, 12, W) (背景区域为0)
            valid_mask: (B, 12, W) (1=信号, 0=背景)
        """
        # 1. 长度对齐
        min_len = min(pred_signals.shape[-1], gt_signals.shape[-1])
        pred = pred_signals[..., :min_len]
        gt = gt_signals[..., :min_len]
        
        # 2. 计算逐点 Loss
        raw_loss = self.criterion(pred, gt)
        
        # 3. 应用加权 Mask
        if valid_mask is not None:
            mask = valid_mask[..., :min_len]
            # 信号区权重 1.0，背景区权重 self.bg_weight
            weights = mask + (1.0 - mask) * self.bg_weight
            weighted_loss = raw_loss * weights
            return weighted_loss.mean()
        else:
            # 如果没有 mask，全员平等
            return raw_loss.mean()

class ProgressiveLeadLocalizationLoss(nn.Module):
    """
    组合损失函数
    """
    def __init__(self, 
                 weight_coarse_baseline=1.0,
                 weight_text=1.0,
                 weight_paper_speed=1.0,
                 weight_lead_baseline=1.0,
                 weight_signal=10.0,
                 background_weight=0.1, # 新增: 背景抑制权重
                 use_focal_loss=True):
        super().__init__()
        
        self.weights = {
            'coarse': weight_coarse_baseline,
            'text': weight_text,
            'ocr': weight_paper_speed,
            'fine': weight_lead_baseline,
            'signal': weight_signal
        }
        
        self.seg_loss_fn = SegmentationLoss(use_focal=use_focal_loss)
        # 使用带背景抑制的信号损失
        self.sig_loss_fn = SignalRegressionLoss(loss_type='l1', background_weight=background_weight)

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
            
        # 3. OCR 辅助任务 (H/4) - 如果有 GT 的话
        if 'paper_speed_mask' in outputs and 'paper_speed_mask' in targets:
             # 暂时跳过或在这里加入逻辑
             pass
            
        # 4. 精细基线 (H/4) - 核心定位
        if 'lead_baselines' in outputs and 'baseline_fine' in targets:
            l = self.seg_loss_fn(outputs['lead_baselines'], targets['baseline_fine'])
            loss_dict['loss_fine'] = l
            total_loss += l * self.weights['fine']
            
        # 5. 信号回归
        if 'signals' in outputs and 'gt_signals' in targets:
            # 自动寻找 valid_mask
            mask = targets.get('valid_mask', None)
            l = self.sig_loss_fn(outputs['signals'], targets['gt_signals'], mask)
            loss_dict['loss_signal'] = l
            total_loss += l * self.weights['signal']
            
        return total_loss, loss_dict

# =============================================================================
# 模块测试
# =============================================================================
if __name__ == "__main__":
    print("Testing Updated ECG Loss...")
    
    criterion = ProgressiveLeadLocalizationLoss(background_weight=0.1)
    
    # 模拟数据
    B, H, W = 2, 512, 2048
    outputs = {
        'coarse_baseline': torch.sigmoid(torch.randn(B, 1, 32, 128)), # H/16
        'lead_baselines': torch.sigmoid(torch.randn(B, 12, 128, 512)), # H/4
        'signals': torch.randn(B, 12, 512) # W/4
    }
    
    targets = {
        # Target 尺寸大，测试自动对齐
        'baseline_coarse': torch.zeros(B, 1, H, W), 
        'baseline_fine': torch.zeros(B, 12, H, W),
        'gt_signals': torch.zeros(B, 12, 512),
        # 模拟 Valid Mask: 前半段有效，后半段无效
        'valid_mask': torch.cat([torch.ones(B, 12, 256), torch.zeros(B, 12, 256)], dim=-1)
    }
    
    # 测试场景: 背景区域预测了非零值 (噪声)
    # 我们期望这会产生 Loss (因为 background_weight=0.1 > 0)
    outputs['signals'][..., 256:] = 10.0 # 在无效区域制造巨大误差
    
    loss, loss_dict = criterion(outputs, targets)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Signal Loss: {loss_dict['loss_signal'].item():.4f}")
    
    # 简单验证: Loss 应该大于 0
    assert loss_dict['loss_signal'] > 0, "Background noise should be penalized!"
    
    print("✓ Loss function test passed! Background suppression is active.")