"""
ECG V47 Model Architecture
基于 CRNN 和 渐进式导联定位 (Locate-Crop-Decode)

核心组件:
1. ResNet50-FPN: 提取高分辨率特征 (d2 stride=4)
2. Localization Heads: 预测每个导联的基线位置
3. RoI Extractor: 基于基线从特征图动态裁剪条带
4. CRNN Decoder: 垂直CNN压缩 + 水平BiGRU序列建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List, Tuple, Optional

class CRNNLeadDecoder(nn.Module):
    """
    CRNN 解码器
    输入: (B*12, C, H_roi, W_roi) - 裁剪出的特征条带
    输出: (B*12, W_roi) - 预测的 1D 信号 (像素偏移量)
    """
    def __init__(self, in_channels=256, hidden_size=128, roi_height=32):
        super().__init__()
        
        # 1. CNN Encoder: 垂直方向特征压缩
        # 目标: 将高度 H_roi (32) 压缩到 1，同时保留波形特征
        self.cnn = nn.Sequential(
            # Layer 1: 32 -> 16
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1)), 
            
            # Layer 2: 16 -> 8
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Layer 3: 8 -> 1 (垂直全卷积)
            nn.Conv2d(64, 64, kernel_size=(8, 1), padding=0),
            nn.BatchNorm2d(64), nn.ReLU(True)
        )
        
        # 2. RNN Sequence Modeling: 水平方向时序建模
        # 处理波形上下文，平滑噪声
        self.rnn = nn.GRU(
            input_size=64, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Regressor Head
        # 输出: 像素偏移量 (相对于基线)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch_Size * 12, C, H, W)
        
        # CNN 提取
        # out: (B', 64, 1, W)
        feat = self.cnn(x)
        
        # 调整维度适配 RNN
        # (B', 64, W) -> (B', W, 64)
        feat = feat.squeeze(2).permute(0, 2, 1)
        
        # RNN 建模
        # out: (B', W, hidden*2)
        rnn_out, _ = self.rnn(feat)
        
        # 回归预测
        # out: (B', W, 1) -> (B', W)
        signal = self.head(rnn_out).squeeze(-1)
        
        return signal


class ProgressiveLeadLocalizationModel(nn.Module):
    """
    主模型架构
    """
    def __init__(self, 
                 num_leads=12, 
                 roi_height=32,  # 在特征图上的裁剪高度
                 pretrained=True):
        super().__init__()
        self.num_leads = num_leads
        self.roi_height = roi_height
        
        # ================== 1. Backbone + FPN ==================
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 提取各层 (Stem, c1, c2, c3, c4)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4
        
        # FPN Lateral Connections
        self.lat4 = nn.Conv2d(2048, 256, 1)
        self.lat3 = nn.Conv2d(1024, 256, 1)
        self.lat2 = nn.Conv2d(512, 256, 1)
        self.lat1 = nn.Conv2d(256, 256, 1)
        
        # FPN Smooth Layers
        self.smooth = nn.Conv2d(256, 256, 3, padding=1)
        
        # ================== 2. Localization Heads ==================
        # 粗定位 (H/16)
        self.head_coarse = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 1, 1), nn.Sigmoid()
        )
        
        # 文字定位 (H/4)
        self.head_text = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 13, 1), nn.Sigmoid() # 12 leads + background
        )
        
        # OCR Heads (辅助任务)
        self.head_ocr = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 2, 1), nn.Sigmoid() # [Speed, Gain] masks
        )
        
        # 精细基线融合头
        # Input: Feature(256) + Coarse(1) + Text(13)
        self.head_fusion = nn.Sequential(
            nn.Conv2d(256 + 1 + 13, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, num_leads, 1), nn.Sigmoid()
        )
        
        # ================== 3. Decoder ==================
        self.decoder = CRNNLeadDecoder(in_channels=256, roi_height=roi_height)

    def _upsample_add(self, x, y):
        """FPN 上采样相加辅助函数"""
        return F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False) + y

    def extract_rois(self, feature_map, baselines):
        """
        动态 RoI 提取核心逻辑
        
        Args:
            feature_map: (B, 256, H, W) - FPN 特征图 (d2)
            baselines: (B, 12, H, W) - 预测的基线热图
            
        Returns:
            rois: (B*12, 256, roi_height, W)
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        # 1. 从热图中计算 Y 轴中心 (Soft Argmax)
        # 将热图沿 W 轴平均，得到 Y 轴分布
        y_dist = baselines.mean(dim=3) # (B, 12, H)
        
        # 生成 [-1, 1] 的坐标网格
        pos_y = torch.linspace(-1, 1, H, device=device).view(1, 1, H)
        
        # 计算期望: Sum(P * y) / Sum(P)
        weights = y_dist / (y_dist.sum(dim=2, keepdim=True) + 1e-6)
        centers_y = (weights * pos_y).sum(dim=2) # (B, 12)
        
        # 2. 构建 Grid Sample 的采样网格
        # 我们需要为每个 (Batch, Lead) 构建一个网格
        # Grid Shape: (B*12, roi_height, W, 2)
        
        # 归一化半高
        half_h = (self.roi_height / H)
        
        # 基础 Y 网格: [-half_h, half_h]
        base_grid_y = torch.linspace(-half_h, half_h, self.roi_height, device=device)
        base_grid_y = base_grid_y.view(1, 1, self.roi_height, 1)
        
        # 扩展 centers_y 以匹配网格维度
        # centers_y: (B, 12) -> (B*12, 1, 1, 1)
        centers_flat = centers_y.view(-1, 1, 1, 1)
        
        # 最终 Y 坐标: center + local_offset
        final_grid_y = centers_flat + base_grid_y
        final_grid_y = final_grid_y.expand(-1, -1, -1, W) # 沿 W 轴复制
        
        # 基础 X 网格: [-1, 1]
        base_grid_x = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
        final_grid_x = base_grid_x.expand(B*self.num_leads, -1, self.roi_height, -1)
        
        # 拼接得到采样 Grid (x, y)
        grid = torch.cat([final_grid_x, final_grid_y], dim=1)
        grid = grid.permute(0, 2, 3, 1) # (B*12, roi_height, W, 2)
        
        # 3. 执行采样
        # 特征图需要 repeat: (B, C, H, W) -> (B*12, C, H, W)
        feat_repeat = feature_map.repeat_interleave(self.num_leads, dim=0)
        
        rois = F.grid_sample(feat_repeat, grid, align_corners=False)
        
        return rois

    def forward(self, x):
        # x: (B, 3, H, W)
        
        # ================== Encoder (ResNet) ==================
        # C1 (H/4), C2 (H/4), C3 (H/8), C4 (H/16), C5 (H/32)
        c1 = self.enc0(x)
        c2 = self.enc1(c1)
        c3 = self.enc2(c2)
        c4 = self.enc3(c3)
        c5 = self.enc4(c4)
        
        # ================== FPN ==================
        p5 = self.lat4(c5)
        p4 = self.lat3(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lat2(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lat1(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        
        # 融合层: d2 (H/4) 是主要的高分辨率特征
        d2 = self.smooth(p2) 
        # d4 (H/16) 用于粗定位
        d4 = F.avg_pool2d(d2, kernel_size=4, stride=4) 
        
        # ================== Localization Heads ==================
        coarse_baseline = self.head_coarse(d4)
        text_masks = self.head_text(d2)
        ocr_maps = self.head_ocr(d2)
        
        # 精细基线预测
        # 上采样粗基线到 H/4
        coarse_up = F.interpolate(coarse_baseline, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        # 融合特征
        fusion_feat = torch.cat([d2, coarse_up, text_masks], dim=1)
        lead_baselines = self.head_fusion(fusion_feat) # (B, 12, H/4, W/4)
        
        outputs = {
            'coarse_baseline': coarse_baseline,
            'text_masks': text_masks,
            'ocr_maps': ocr_maps,
            'lead_baselines': lead_baselines
        }
        
        # ================== Decode Signals ==================
        # 仅在训练或推理模式下需要信号时运行
        # 可以通过 self.training 自动判断，或者显式调用
        if True: 
            # 1. 提取 RoIs (B*12, 256, 32, W/4)
            lead_rois = self.extract_rois(d2, lead_baselines)
            
            # 2. CRNN 解码 (B*12, W/4)
            raw_signals = self.decoder(lead_rois)
            
            # 3. Reshape 回 (B, 12, W/4)
            B = x.shape[0]
            W_feat = d2.shape[-1]
            pred_signals = raw_signals.view(B, self.num_leads, W_feat)
            
            # 输出的是像素偏移量，需要除以 Gain (在 Loss 中做)
            outputs['signals'] = pred_signals 
            
        return outputs

# =============================================================================
# 模块测试
# =============================================================================
if __name__ == "__main__":
    print("Testing ProgressiveLeadLocalizationModel...")
    
    # 1. 创建模型
    model = ProgressiveLeadLocalizationModel(num_leads=12, pretrained=False)
    model.eval()
    
    # 2. 创建 Dummy 输入 (B=2, C=3, H=512, W=2048)
    # 注意宽度 2048 适配 test.csv 中的高频信号需求
    dummy_input = torch.randn(2, 3, 512, 2048)
    
    print(f"Input Shape: {dummy_input.shape}")
    
    # 3. 前向传播
    with torch.no_grad():
        outputs = model(dummy_input)
        
    # 4. 验证输出形状
    print("\nOutput Shapes:")
    print(f"Coarse Baseline: {outputs['coarse_baseline'].shape} (Expected: B, 1, H/16, W/16)")
    print(f"Lead Baselines: {outputs['lead_baselines'].shape} (Expected: B, 12, H/4, W/4)")
    print(f"Signals: {outputs['signals'].shape} (Expected: B, 12, W/4)")
    
    # 检查信号长度
    expected_len = 2048 // 4
    assert outputs['signals'].shape[-1] == expected_len, f"Signal length mismatch: got {outputs['signals'].shape[-1]}, expected {expected_len}"
    
    print("\n✓ Model test passed!")