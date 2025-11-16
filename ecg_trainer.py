"""
修复版损失函数 - 支持导联有效时间掩码

关键改进:
1. 添加signal_mask参数，标记每个导联的有效时间段
2. 只在有效区域计算损失
3. 避免模型被迫学习填充的0值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLossWithMask(nn.Module):
    """
    支持导联时间掩码的多任务损失函数
    
    新增功能:
    - signal_mask: (B, 12, T) bool tensor，标记每个导联每个时间点是否有效
    - 只在有效区域计算信号损失
    """
    def __init__(self, loss_weights: dict = None):
        super().__init__()
        
        self.weights = loss_weights or {
            'seg': 1.0,
            'grid': 0.5,
            'baseline': 0.8,
            'theta': 0.3,
            'signal': 2.0
        }
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss(reduction='none')  # 🔥 改为none，手动处理mask
    
    def dice_loss(self, pred, target, num_classes):
        """多类Dice损失（不变）"""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        return 1 - dice.mean()
    
    def masked_mae_loss(self, pred, target, mask):
        """
        带掩码的MAE损失
        
        Args:
            pred: (B, 12, T)
            target: (B, 12, T)
            mask: (B, 12, T) bool，True表示有效区域
        
        Returns:
            loss: scalar
        """
        # 只在有效区域计算误差
        mae = torch.abs(pred - target)  # (B, 12, T)
        masked_mae = mae * mask.float()  # 无效区域置0
        
        # 计算平均损失（除以有效点数）
        num_valid = mask.float().sum() + 1e-7
        loss = masked_mae.sum() / num_valid
        
        return loss
    
    def masked_pearson_loss(self, pred, target, mask):
        """
        带掩码的Pearson相关系数损失
        
        Args:
            pred: (B, 12, T)
            target: (B, 12, T)
            mask: (B, 12, T) bool
        
        Returns:
            loss: scalar (1 - 平均相关系数)
        """
        B, num_leads, T = pred.shape
        
        correlations = []
        
        for b in range(B):
            for lead in range(num_leads):
                lead_mask = mask[b, lead]  # (T,)
                
                # 跳过完全无效的导联
                if lead_mask.sum() < 10:  # 至少需要10个有效点
                    continue
                
                # 提取有效区域
                pred_valid = pred[b, lead, lead_mask]  # (N_valid,)
                target_valid = target[b, lead, lead_mask]  # (N_valid,)
                
                # 计算Pearson相关系数
                pred_mean = pred_valid.mean()
                target_mean = target_valid.mean()
                
                pred_centered = pred_valid - pred_mean
                target_centered = target_valid - target_mean
                
                numerator = (pred_centered * target_centered).sum()
                pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-6)
                target_std = torch.sqrt((target_centered ** 2).sum() + 1e-6)
                denominator = pred_std * target_std + 1e-6
                
                corr = numerator / denominator
                corr = torch.clamp(corr, -1.0, 1.0)
                
                correlations.append(corr)
        
        if len(correlations) == 0:
            # 如果没有有效导联，返回最大损失
            return torch.tensor(1.0, device=pred.device)
        
        # 返回负平均相关系数
        avg_corr = torch.stack(correlations).mean()
        return 1 - avg_corr
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from model
                - wave_seg: (B, 13, H, W)
                - grid_mask: (B, 1, H, W)
                - baseline_heatmaps: (B, 12, H/16, W/16)
                - theta: (B, 2, 3)
                - signal: (B, 12, T)
            
            targets: dict containing
                - wave_seg: (B, H, W)
                - grid_mask: (B, 1, H, W)
                - baseline_heatmaps: (B, 12, H, W)
                - theta_gt: (B, 2, 3)
                - gt_signal: (B, T, 12)
                - signal_mask: (B, T, 12) bool [新增]
        """
        losses = {}
        
        # ========== 1-4. 分割/网格/基线/几何损失（不变）==========
        wave_seg_pred = outputs['wave_seg']
        wave_seg_target = targets['wave_seg']
        
        ce_seg = self.ce_loss(wave_seg_pred, wave_seg_target)
        dice_seg = self.dice_loss(wave_seg_pred, wave_seg_target, num_classes=13)
        losses['seg'] = (ce_seg + dice_seg) * self.weights['seg']
        
        grid_pred = outputs['grid_mask']
        grid_target = targets['grid_mask']
        bce_grid = self.bce_loss(grid_pred, grid_target)
        intersection = (grid_pred * grid_target).sum()
        union = grid_pred.sum() + grid_target.sum()
        dice_grid = 1 - (2.0 * intersection + 1e-7) / (union + 1e-7)
        losses['grid'] = (bce_grid + dice_grid) * self.weights['grid']
        
        baseline_pred = outputs['baseline_heatmaps']
        baseline_target = targets['baseline_heatmaps']
        B, num_leads, H_pred, W_pred = baseline_pred.shape
        baseline_target_down = F.interpolate(
            baseline_target, size=(H_pred, W_pred),
            mode='bilinear', align_corners=True
        )
        losses['baseline'] = self.bce_loss(baseline_pred, baseline_target_down) * self.weights['baseline']
        
        if 'theta' in outputs and outputs['theta'] is not None:
            theta_pred = outputs['theta']
            theta_target = targets['theta_gt']
            losses['theta'] = torch.nn.functional.l1_loss(theta_pred, theta_target) * self.weights['theta']
        else:
            losses['theta'] = torch.tensor(0.0, device=baseline_pred.device)
        
        # ========== 5. 信号重建损失（修复版）==========
        signal_pred = outputs['signal']  # (B, 12, T)
        signal_target = targets['gt_signal'].transpose(1, 2)  # (B, 12, T)
        signal_mask = targets['signal_mask'].transpose(1, 2)  # (B, 12, T)
        
        # 归一化（只在有效区域）
        signal_pred_norm = self._normalize_signal_masked(signal_pred, signal_mask)
        signal_target_norm = self._normalize_signal_masked(signal_target, signal_mask)
        
        # 🔥 关键修复：使用掩码损失
        mae_signal = self.masked_mae_loss(signal_pred_norm, signal_target_norm, signal_mask)
        corr_signal = self.masked_pearson_loss(signal_pred_norm, signal_target_norm, signal_mask)
        
        losses['signal'] = (mae_signal + corr_signal) * self.weights['signal']
        
        # ========== 总损失 ==========
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _normalize_signal_masked(self, signal, mask):
        """
        对每个导联在有效区域独立归一化
        
        Args:
            signal: (B, 12, T)
            mask: (B, 12, T) bool
        
        Returns:
            signal_norm: (B, 12, T)
        """
        B, num_leads, T = signal.shape
        signal_norm = torch.zeros_like(signal)
        
        for b in range(B):
            for lead in range(num_leads):
                lead_mask = mask[b, lead]
                
                if lead_mask.sum() < 2:  # 至少2个点才能归一化
                    continue
                
                # 只在有效区域计算min/max
                valid_signal = signal[b, lead, lead_mask]
                min_val = valid_signal.min()
                max_val = valid_signal.max()
                
                if max_val - min_val < 1e-6:  # 常数信号
                    signal_norm[b, lead] = 0.0
                else:
                    # 归一化到[-1, 1]
                    signal_norm[b, lead] = 2 * (signal[b, lead] - min_val) / (max_val - min_val + 1e-8) - 1
                    
                    # 无效区域置0（可选，因为损失计算会忽略）
                    signal_norm[b, lead, ~lead_mask] = 0.0
        
        return signal_norm


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("="*70)
    print("带掩码的损失函数测试")
    print("="*70)
    
    loss_fn = MultiTaskLossWithMask()
    
    B, H, W, T = 2, 512, 672, 5000
    
    outputs = {
        'wave_seg': torch.randn(B, 13, H, W, requires_grad=True),
        'grid_mask': torch.sigmoid(torch.randn(B, 1, H, W, requires_grad=True)),
        'baseline_heatmaps': torch.sigmoid(torch.randn(B, 12, H//16, W//16, requires_grad=True)),
        'theta': torch.randn(B, 2, 3, requires_grad=True),
        'signal': torch.randn(B, 12, T, requires_grad=True)
    }
    
    # 🔥 模拟真实掩码：II导联全时间，其他导联部分时间
    signal_mask = torch.zeros(B, T, 12, dtype=torch.bool)
    
    # II导联（索引1）：完整10秒
    signal_mask[:, :, 1] = True
    
    # 其他导联：只有2.5-5.0秒有数据
    signal_mask[:, 1250:2500, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] = True
    
    print(f"\n掩码统计:")
    for lead in range(12):
        valid_ratio = signal_mask[:, :, lead].float().mean().item()
        print(f"  Lead {lead:2d}: {valid_ratio*100:.1f}% 有效")
    
    targets = {
        'wave_seg': torch.randint(0, 13, (B, H, W)),
        'grid_mask': torch.rand(B, 1, H, W),
        'baseline_heatmaps': torch.rand(B, 12, H, W),
        'theta_gt': torch.randn(B, 2, 3),
        'gt_signal': torch.randn(B, T, 12),
        'signal_mask': signal_mask  # 🔥 新增
    }
    
    # 计算损失
    losses = loss_fn(outputs, targets)
    
    print("\n损失值:")
    for key, value in losses.items():
        print(f"  {key:15s}: {value.item():.4f}")
    
    # 测试反向传播
    print("\n测试反向传播...")
    losses['total'].backward()
    print("✓ 反向传播成功")
    
    print("\n" + "="*70)
    print("✓ 测试通过！")
    print("="*70)