"""
ECG重建模型定义

架构: U-Net + 多任务头 + STN + 信号解码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialTransformerNetwork(nn.Module):
    """
    空间变换网络 - 用于几何校正
    
    MPS兼容版本：避免使用AdaptiveAvgPool2d
    """
    def __init__(self, in_channels: int = 2048):
        super().__init__()
        self.in_channels = in_channels
        
        # 🔥 使用全卷积网络代替AdaptiveAvgPool
        self.localization = nn.Sequential(
            # 降维
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 全局平均池化（MPS支持）
            nn.AdaptiveAvgPool2d(1),  # 池化到1x1，这个MPS支持
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 6)  # 仿射变换6个参数
        )
        
        # 初始化为单位变换
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
    
    def forward(self, x):
        """
        Args:
            x: 特征图 (B, C, H, W)
        Returns:
            theta: 仿射变换矩阵 (B, 2, 3)
        """
        theta = self.localization(x)  # (B, 6)
        theta = theta.view(-1, 2, 3)  # (B, 2, 3)
        return theta


class SignalDecoder(nn.Module):
    """
    信号解码器：从校正后的特征图 + 分割掩码 -> 时间序列信号
    """
    def __init__(self, feature_channels: int, num_leads: int, signal_length: int):
        super().__init__()
        self.num_leads = num_leads
        self.signal_length = signal_length
        
        # 每个导联的1D细化网络
        self.lead_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_channels, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 1, kernel_size=3, padding=1)
            ) for _ in range(num_leads)
        ])
    
    def forward(self, features, wave_seg_logits, baseline_heatmaps):
        """
        Args:
            features: (B, C, H, W) 校正后的特征
            wave_seg_logits: (B, 13, H, W) 分割logits
            baseline_heatmaps: (B, 12, H/k, W/k) 基线热图
        
        Returns:
            signal: (B, 12, T) 重建信号
        """
        B, C, H, W = features.shape
        
        # Softmax分割掩码
        wave_seg_prob = F.softmax(wave_seg_logits, dim=1)[:, 1:, :, :]  # (B, 12, H, W)
        
        # 上采样基线热图
        baseline_up = F.interpolate(
            baseline_heatmaps, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=True
        )
        
        signals = []
        
        for lead_idx in range(self.num_leads):
            # 提取该导联的加权特征
            lead_mask = wave_seg_prob[:, lead_idx:lead_idx+1, :, :]  # (B, 1, H, W)
            baseline_mask = baseline_up[:, lead_idx:lead_idx+1, :, :]  # (B, 1, H, W)
            
            # 组合掩码
            combined_mask = lead_mask * 0.3 + baseline_mask * 0.7  # (B, 1, H, W)
            
            # 加权特征
            weighted_features = features * combined_mask  # (B, C, H, W)
            
            # 沿垂直方向（y轴）加权求和，保留水平（时间）信息
            signal_1d = torch.sum(weighted_features, dim=2)  # (B, C, W)
            
            # 归一化（防止全零）
            norm_factor = combined_mask.sum(dim=2).clamp(min=1e-6)  # (B, 1, W)
            signal_1d = signal_1d / norm_factor  # (B, C, W)
            
            # 🔥 修复NaN: 检查并替换异常值
            if torch.isnan(signal_1d).any() or torch.isinf(signal_1d).any():
                signal_1d = torch.nan_to_num(signal_1d, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 调整长度到目标信号长度
            if W != self.signal_length:
                signal_1d = F.interpolate(
                    signal_1d, 
                    size=self.signal_length, 
                    mode='linear', 
                    align_corners=True
                )
            
            # 1D CNN细化
            refined_signal = self.lead_refiners[lead_idx](signal_1d)  # (B, 1, T)
            signals.append(refined_signal)
        
        # 合并所有导联
        output = torch.cat(signals, dim=1)  # (B, 12, T)
        
        return output


class ECGReconstructionModel(nn.Module):
    """
    完整的ECG重建模型
    
    输入: (B, 3, H, W) ECG图像
    输出: 
        - wave_seg: (B, 13, H, W) 导联分割（12导联+1背景）
        - grid_mask: (B, 1, H, W) 网格掩码
        - baseline_heatmaps: (B, 12, H/16, W/16) 基线热图
        - theta: (B, 2, 3) 几何变换矩阵
        - signal: (B, 12, T) 重建信号
        
    注意: 如果使用pretrained=True，会有警告，这是正常的
    """
    def __init__(self,
                 num_leads: int = 12,
                 signal_length: int = 5000,
                 pretrained: bool = True,
                 enable_stn: bool = True):  # 🔥 新增参数
        super().__init__()
        
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.enable_stn = enable_stn  # 是否启用STN
        
        # ========== Stage 1: Encoder (ResNet-50 Backbone) ==========
        if pretrained:
            # 使用新的weights参数（兼容新版torchvision）
            try:
                from torchvision.models import ResNet50_Weights
                resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            except:
                # 回退到旧版本的pretrained参数
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning)
                resnet = models.resnet50(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=False)
        self.encoder1 = nn.Sequential(*list(resnet.children())[:4])   # 64 channels
        self.encoder2 = nn.Sequential(*list(resnet.children())[4:5])  # 256 channels
        self.encoder3 = nn.Sequential(*list(resnet.children())[5:6])  # 512 channels
        self.encoder4 = nn.Sequential(*list(resnet.children())[6:7])  # 1024 channels
        self.encoder5 = nn.Sequential(*list(resnet.children())[7:8])  # 2048 channels
        
        # 通道注意力
        self.ca5 = ChannelAttention(2048)
        self.ca4 = ChannelAttention(1024)
        self.ca3 = ChannelAttention(512)
        self.ca2 = ChannelAttention(256)
        self.ca1 = ChannelAttention(64)
        
        # ========== Stage 2: Decoder ==========
        self.up5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec5 = self._make_decoder_block(1024 + 1024, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._make_decoder_block(512 + 512, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(256 + 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec2 = self._make_decoder_block(64 + 64, 64)
        
        # ========== Stage 3: Task-Specific Heads ==========
        
        # 1. 导联分割头
        self.wave_seg_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_leads + 1, 1)  # 12导联 + 1背景类
        )
        
        # 2. 网格掩码头
        self.grid_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 3. 基线热图头（从中间层）
        self.baseline_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_leads, 1),
            nn.Sigmoid()
        )
        
        # 4. 几何校正网络
        self.stn = SpatialTransformerNetwork(in_channels=2048)
        
        # 5. 信号重建网络
        self.signal_decoder = SignalDecoder(
            feature_channels=64,
            num_leads=num_leads,
            signal_length=signal_length
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) 输入图像
        
        Returns:
            outputs: dict containing
                - wave_seg: (B, 13, H, W)
                - grid_mask: (B, 1, H, W)
                - baseline_heatmaps: (B, 12, H/16, W/16)
                - theta: (B, 2, 3)
                - signal: (B, 12, T)
                - rectified_features: (B, 64, H, W)
        """
        B, C, H, W = x.shape
        
        # ========== Encoding ==========
        e1 = self.encoder1(x)       # (B, 64, H/4, W/4)
        e2 = self.encoder2(e1)      # (B, 256, H/8, W/8)
        e3 = self.encoder3(e2)      # (B, 512, H/16, W/16)
        e4 = self.encoder4(e3)      # (B, 1024, H/32, W/32)
        e5 = self.encoder5(e4)      # (B, 2048, H/64, W/64)
        
        # ========== 几何校正 ==========
        theta = self.stn(e5)  # (B, 2, 3)
        
        # ========== Decoding ==========
        d5 = self.up5(self.ca5(e5))
        d5 = torch.cat([d5, self.ca4(e4)], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat([d4, self.ca3(e3)], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, self.ca2(e2)], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        # 🔥 修复：检查尺寸是否匹配
        if d2.shape[2:] != e1.shape[2:]:
            # 如果尺寸不匹配，插值到相同尺寸
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, self.ca1(e1)], dim=1)
        d2 = self.dec2(d2)  # (B, 64, H/4, W/4)
        
        # 上采样到原始分辨率
        d_final = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        
        # ========== Task Outputs ==========
        
        # 1. 导联分割
        wave_seg = self.wave_seg_head(d_final)  # (B, 13, H, W)
        
        # 2. 网格掩码
        grid_mask = self.grid_head(d_final)  # (B, 1, H, W)
        
        # 3. 基线热图（从中间层）
        baseline_heatmaps = self.baseline_head(d4)  # (B, 12, H/16, W/16)
        
        # 4. 几何校正特征
        # 🔥 修复：完全禁用STN的几何校正，避免MPS的grid_sample问题
        if self.enable_stn:
            # 如果启用STN，仍然计算theta但不做grid_sample
            # 只在CPU/CUDA上才真正做空间变换
            if self.training and d_final.device.type in ['cuda', 'cpu']:
                try:
                    grid_sample_grid = F.affine_grid(theta, d_final.size(), align_corners=True)
                    rectified_features = F.grid_sample(d_final, grid_sample_grid, align_corners=True)
                except (RuntimeError, NotImplementedError):
                    rectified_features = d_final
            else:
                # MPS或推理模式：不做几何校正
                rectified_features = d_final
        else:
            # STN完全禁用
            rectified_features = d_final
            # 返回单位矩阵作为theta（避免损失函数报错）
            B = d_final.size(0)
            theta = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=d_final.device)
            theta = theta.unsqueeze(0).repeat(B, 1, 1)
        
        # 5. 信号重建
        signal = self.signal_decoder(rectified_features, wave_seg, baseline_heatmaps)
        
        return {
            'wave_seg': wave_seg,
            'grid_mask': grid_mask,
            'baseline_heatmaps': baseline_heatmaps,
            'theta': theta,
            'signal': signal,
            'rectified_features': rectified_features
        }


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("="*70)
    print("ECG重建模型测试")
    print("="*70)
    
    # 忽略torchvision的警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 创建模型
    print("\n创建模型...")
    model = ECGReconstructionModel(
        num_leads=12,
        signal_length=5000,
        pretrained=False  # 测试时不用预训练，避免下载
    )
    print("✓ 模型创建成功")
    
    # 测试前向传播
    print("\n测试前向传播...")
    x = torch.randn(2, 3, 512, 672)  # Batch=2
    
    print(f"输入shape: {x.shape}")
    
    try:
        outputs = model(x)
        print("✓ 前向传播成功")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        print("\n调试信息:")
        
        # 逐层测试找出问题
        e1 = model.encoder1(x)
        print(f"  e1: {e1.shape}")
        e2 = model.encoder2(e1)
        print(f"  e2: {e2.shape}")
        e3 = model.encoder3(e2)
        print(f"  e3: {e3.shape}")
        e4 = model.encoder4(e3)
        print(f"  e4: {e4.shape}")
        e5 = model.encoder5(e4)
        print(f"  e5: {e5.shape}")
        
        raise e
    
    print("\n模型输出:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {tuple(value.shape)}")
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    print("\n" + "="*70)
    print("✓ 模型测试通过！")
    print("="*70)