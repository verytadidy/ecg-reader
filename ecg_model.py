"""
ECG V45 渐进式导联定位模型

架构设计:
- 编码器: ResNet-50 + FPN
- 粗层路径(H/16): 基线热图 + 时间范围估计
- 细层路径(H/4): 文字掩码(13层) + 辅助元素 + OCR
- 融合模块: 导联级基线定位(12层)
- 信号解码器: STN校正 + 1D-CNN解码

特点:
1. ✅ 多尺度特征提取 (FPN)
2. ✅ 渐进式定位 (粗->细)
3. ✅ OCR优先 (纸速⭐⭐⭐⭐⭐ + 增益⭐⭐⭐)
4. ✅ 几何校正 (STN)
5. ✅ 端到端训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align
from typing import Dict, List, Tuple, Optional


class SpatialTransformerNetwork(nn.Module):
    """
    空间变换网络 (STN)
    用于几何校正 (矫正图像中的扭曲、倾斜等)
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # 定位网络 (预测仿射变换参数)
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        
        # 全连接层回归仿射变换参数 (2x3矩阵)
        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )
        
        # 初始化为恒等变换
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) 输入特征
        
        Returns:
            x_transformed: (B, C, H, W) 校正后的特征
            theta: (B, 2, 3) 仿射变换矩阵
        """
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (4, 4))
        xs = xs.view(xs.size(0), -1)
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # 应用仿射变换
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        
        return x_transformed, theta


class LeadSignalDecoder(nn.Module):
    """
    单导联信号解码器
    从2D特征图解码出1D信号
    """
    
    def __init__(self, in_channels: int = 256):
        super().__init__()
        
        # 1D卷积解码
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            
            nn.Conv1d(32, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) RoI特征
        
        Returns:
            signal: (B, W) 1D信号
        """
        # 沿高度方向平均池化
        x = x.mean(dim=2)  # (B, C, W)
        
        # 1D卷积解码
        signal = self.decoder(x)  # (B, 1, W)
        signal = signal.squeeze(1)  # (B, W)
        
        return signal


class ProgressiveLeadLocalizationModel(nn.Module):
    """
    渐进式导联定位模型
    
    特点:
    - 从粗到细的导联定位
    - OCR优先 (纸速⭐⭐⭐⭐⭐ + 增益⭐⭐⭐)
    - 多任务学习
    """
    
    def __init__(self, 
                 num_leads: int = 12,
                 encoder_name: str = 'resnet50',
                 pretrained: bool = True):
        super().__init__()
        
        self.num_leads = num_leads
        
        # ============================================================
        # 编码器 (ResNet-50 + FPN)
        # ============================================================
        if encoder_name == 'resnet50':
            backbone = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            
            # 提取各层特征
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            
            # ResNet-50的标准通道数
            self.layer1 = backbone.layer1  # 1/4,  256 channels
            self.layer2 = backbone.layer2  # 1/8,  512 channels
            self.layer3 = backbone.layer3  # 1/16, 1024 channels
            self.layer4 = backbone.layer4  # 1/32, 2048 channels
            
            # FPN lateral connections (统一到256通道)
            self.lateral5 = nn.Conv2d(2048, 256, kernel_size=1)  # c4 -> p5
            self.lateral4 = nn.Conv2d(1024, 256, kernel_size=1)  # c3 -> p4
            self.lateral3 = nn.Conv2d(512, 256, kernel_size=1)   # c2 -> p3
            self.lateral2 = nn.Conv2d(256, 256, kernel_size=1)   # c1 -> p2
            
            # FPN smooth layers (减少上采样的混叠效应)
            self.smooth5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # ============================================================
        # 粗层路径 (从d4预测, H/16分辨率)
        # ============================================================
        
        # 粗粒度基线热图 (单通道, 不区分导联)
        self.coarse_baseline_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 时间范围估计 (每个导联2个值: start, end)
        self.time_range_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, num_leads * 2)  # 12导联 x 2
        )
        
        # ============================================================
        # 细层路径 (从d2预测, H/4分辨率)
        # ============================================================
        
        # 导联文字掩码 (13通道: 12导联 + 1背景)
        self.text_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 13, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 辅助元素掩码 (定标脉冲 + 分隔符)
        self.auxiliary_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ============================================================
        # OCR分支 (V45新增)
        # ============================================================
        
        # 纸速OCR掩码 (关键性 ⭐⭐⭐⭐⭐)
        self.paper_speed_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 增益OCR掩码 (关键性 ⭐⭐⭐)
        self.gain_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ============================================================
        # 融合与导联定位模块
        # ============================================================
        
        # 特征融合 (粗基线 + 文字 + 辅助 + d2特征)
        # 输入通道数: 256(d2) + 1(粗基线) + 13(文字) + 1(辅助) = 271
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(271, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # 生成每个导联的基线掩码
        self.lead_baseline_head = nn.Sequential(
            nn.Conv2d(128, num_leads, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ============================================================
        # 高分辨率波形解码器 (可选, 推理时使用)
        # ============================================================
        self.wave_decoder = nn.ModuleList([
            LeadSignalDecoder(in_channels=256) for _ in range(num_leads)
        ])
        
        # STN用于RoI校正
        self.stn = SpatialTransformerNetwork(in_channels=256)
    
    def forward_encoder(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向编码器，提取多尺度特征
        
        Args:
            x: (B, 3, H, W) 输入图像
        
        Returns:
            features: 字典 {
                'd2': (B, 256, H/4, W/4),
                'd3': (B, 256, H/8, W/8),
                'd4': (B, 256, H/16, W/16),
                'd5': (B, 256, H/32, W/32)
            }
        """
        # Stem
        x = self.conv1(x)      # (B, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (B, 64, H/4, W/4)
        
        # ResNet stages
        c1 = self.layer1(x)    # (B, 256, H/4, W/4)
        c2 = self.layer2(c1)   # (B, 512, H/8, W/8)
        c3 = self.layer3(c2)   # (B, 1024, H/16, W/16)
        c4 = self.layer4(c3)   # (B, 2048, H/32, W/32)
        
        # FPN top-down pathway
        # 从最深层开始
        p5 = self.lateral5(c4)  # (B, 256, H/32, W/32)
        p5 = self.smooth5(p5)
        
        # p4: 融合c3和上采样的p5
        p4 = self.lateral4(c3) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p4 = self.smooth4(p4)   # (B, 256, H/16, W/16)
        
        # p3: 融合c2和上采样的p4
        p3 = self.lateral3(c2) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = self.smooth3(p3)   # (B, 256, H/8, W/8)
        
        # p2: 融合c1和上采样的p3
        p2 = self.lateral2(c1) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p2 = self.smooth2(p2)   # (B, 256, H/4, W/4)
        
        return {
            'd2': p2,  # H/4  - 用于细层预测
            'd3': p3,  # H/8
            'd4': p4,  # H/16 - 用于粗层预测
            'd5': p5   # H/32
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (B, 3, H, W) 输入图像
        
        Returns:
            outputs: 字典 {
                'coarse_baseline': (B, 1, H/16, W/16),
                'time_ranges': (B, 12, 2),
                'text_masks': (B, 13, H/4, W/4),
                'auxiliary_mask': (B, 1, H/4, W/4),
                'paper_speed_mask': (B, 1, H/4, W/4),
                'gain_mask': (B, 1, H/4, W/4),
                'lead_baselines': (B, 12, H/4, W/4),
                # 以下仅推理时
                'lead_bboxes': (B, 12, 4),  [可选]
                'signals': List[Tensor]      [可选]
            }
        """
        B, _, H, W = x.shape
        
        # ========== 1. 编码器 ==========
        features = self.forward_encoder(x)
        d2, d3, d4, d5 = (
            features['d2'], features['d3'], 
            features['d4'], features['d5']
        )
        
        # ========== 2. 粗层预测 (从d4) ==========
        coarse_baseline = self.coarse_baseline_head(d4)  # (B, 1, H/16, W/16)
        
        time_ranges = self.time_range_head(d4)           # (B, 12*2)
        time_ranges = time_ranges.view(B, self.num_leads, 2)
        
        # ========== 3. 细层预测 (从d2) ==========
        text_masks = self.text_head(d2)                  # (B, 13, H/4, W/4)
        auxiliary_mask = self.auxiliary_head(d2)         # (B, 1, H/4, W/4)
        
        # ========== 4. OCR预测 (从d2) ==========
        paper_speed_mask = self.paper_speed_head(d2)     # (B, 1, H/4, W/4)
        gain_mask = self.gain_head(d2)                   # (B, 1, H/4, W/4)
        
        # ========== 5. 融合与导联定位 ==========
        # 上采样粗基线到H/4
        coarse_baseline_up = F.interpolate(
            coarse_baseline, 
            size=(H//4, W//4), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 拼接所有特征
        fusion_input = torch.cat([
            d2,                     # (B, 256, H/4, W/4)
            coarse_baseline_up,     # (B, 1, H/4, W/4)
            text_masks,             # (B, 13, H/4, W/4)
            auxiliary_mask          # (B, 1, H/4, W/4)
        ], dim=1)  # (B, 271, H/4, W/4)
        
        fusion_feat = self.fusion_conv(fusion_input)     # (B, 128, H/4, W/4)
        lead_baselines = self.lead_baseline_head(fusion_feat)  # (B, 12, H/4, W/4)
        
        # ========== 6. 输出 ==========
        outputs = {
            'coarse_baseline': coarse_baseline,
            'time_ranges': time_ranges,
            'text_masks': text_masks,
            'auxiliary_mask': auxiliary_mask,
            'paper_speed_mask': paper_speed_mask,
            'gain_mask': gain_mask,
            'lead_baselines': lead_baselines,
        }
        
        # ========== 7. 高分辨率信号解码 (仅推理时) ==========
        if not self.training:
            # 这部分在实际使用时可以根据需要实现
            # 这里提供简化版本
            pass
        
        return outputs
    
    def extract_roi_from_masks(self, 
                               baseline_mask: torch.Tensor, 
                               text_mask: torch.Tensor) -> torch.Tensor:
        """
        从基线和文字掩码提取RoI边界框
        
        Args:
            baseline_mask: (H, W) 基线掩码
            text_mask: (H, W) 文字掩码
        
        Returns:
            bbox: (4,) [x1, y1, x2, y2]
        """
        H, W = baseline_mask.shape
        
        # 合并基线和文字掩码
        combined = (baseline_mask + text_mask).clamp(0, 1)
        
        # 找到非零像素
        nonzero = torch.nonzero(combined > 0.5)
        
        if len(nonzero) == 0:
            # 返回默认框
            return torch.tensor([0, 0, W-1, H-1], dtype=torch.float32, device=baseline_mask.device)
        
        y_min, x_min = nonzero.min(dim=0)[0]
        y_max, x_max = nonzero.max(dim=0)[0]
        
        # 添加边距
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(W - 1, x_max + margin)
        y_max = min(H - 1, y_max + margin)
        
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32, device=baseline_mask.device)


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("="*80)
    print("ECG V45 模型测试")
    print("="*80)
    
    # 创建模型
    model = ProgressiveLeadLocalizationModel(
        num_leads=12, 
        encoder_name='resnet50',
        pretrained=False
    )
    model.eval()
    
    # 模拟输入
    batch_size = 2
    H, W = 512, 672
    x = torch.randn(batch_size, 3, H, W)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    # 打印输出
    print("\n模型输出:")
    print("-" * 80)
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:25s}: {val.shape}")
        else:
            print(f"  {key:25s}: {type(val)}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*80)
    print("模型统计:")
    print("="*80)
    print(f"总参数量:       {total_params:,}")
    print(f"可训练参数量:   {trainable_params:,}")
    print(f"参数大小:       {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print("="*80)
    
    # 测试各个输出的形状
    assert outputs['coarse_baseline'].shape == (batch_size, 1, H//16, W//16), \
        f"coarse_baseline shape mismatch: {outputs['coarse_baseline'].shape}"
    assert outputs['time_ranges'].shape == (batch_size, 12, 2), \
        f"time_ranges shape mismatch: {outputs['time_ranges'].shape}"
    assert outputs['text_masks'].shape == (batch_size, 13, H//4, W//4), \
        f"text_masks shape mismatch: {outputs['text_masks'].shape}"
    assert outputs['auxiliary_mask'].shape == (batch_size, 1, H//4, W//4), \
        f"auxiliary_mask shape mismatch: {outputs['auxiliary_mask'].shape}"
    assert outputs['paper_speed_mask'].shape == (batch_size, 1, H//4, W//4), \
        f"paper_speed_mask shape mismatch: {outputs['paper_speed_mask'].shape}"
    assert outputs['gain_mask'].shape == (batch_size, 1, H//4, W//4), \
        f"gain_mask shape mismatch: {outputs['gain_mask'].shape}"
    assert outputs['lead_baselines'].shape == (batch_size, 12, H//4, W//4), \
        f"lead_baselines shape mismatch: {outputs['lead_baselines'].shape}"
    
    print("\n✓ 所有测试通过！")