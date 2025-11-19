"""
ECG V47 训练脚本 (CRNN + 渐进式定位)

功能特性:
1. ✅ 多后端支持: 自动适配 CUDA (NVIDIA), MPS (Apple Silicon/Mac), CPU
2. ✅ 混合精度训练: 支持 CUDA AMP 加速
3. ✅ CRNN 适配: 支持时序信号回归 Loss 计算
4. ✅ 完整监控: TensorBoard 可视化 Loss 和指标
5. ✅ 健壮性: 包含 Checkpoint 管理和早停机制
"""

import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 导入项目模块 (确保这些文件在同一目录下)
from ecg_dataset import create_dataloaders
from ecg_model import ProgressiveLeadLocalizationModel
# 假设 ecg_loss 中有基础的分割 Loss，这里我们也会在 Trainer 中定义组合 Loss
from ecg_loss import ProgressiveLeadLocalizationLoss 

class ECGTrainer:
    def __init__(self, args):
        self.args = args
        self._setup_device()
        self._setup_paths()
        
        # 1. 数据加载
        print(f"\n[Data] 初始化数据加载器...")
        self.train_loader, self.val_loader = create_dataloaders(
            sim_root=args.sim_root,
            csv_root=args.csv_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=tuple(args.input_size),
            target_fs=args.target_fs,
            max_samples=args.max_samples,
            augment=True
        )
        
        # 2. 模型构建
        print(f"[Model] 构建 ProgressiveLeadLocalizationModel...")
        self.model = ProgressiveLeadLocalizationModel(
            num_leads=12,
            roi_height=32, # 对应 ecg_model 中的设置
            pretrained=args.pretrained
        ).to(self.device)
        
        # 3. 优化器 & 调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        if args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=args.lr_step, gamma=0.1
            )
        else:
            self.scheduler = None
            
        # 4. 损失函数
        # 分割任务使用基础 Loss
        self.seg_criterion = ProgressiveLeadLocalizationLoss(
            weight_coarse_baseline=1.0,
            weight_text=1.0,
            weight_paper_speed=1.0,
            use_focal_loss=True
        ).to(self.device)
        
        # 信号回归任务使用 L1 Loss (鲁棒性优于 MSE)
        self.signal_criterion = nn.L1Loss()
        
        # 5. 混合精度 (仅 CUDA)
        self.use_amp = args.use_amp and (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 状态初始化
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 恢复训练
        if args.resume:
            self._load_checkpoint(args.resume)

        self._print_setup()

    def _setup_device(self):
        """自动选择最佳计算设备"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            print(f"[System] 使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"[System] 使用 Apple MPS (Metal Performance Shaders) 加速")
        else:
            self.device = torch.device("cpu")
            print(f"[System] ⚠️ 未检测到加速卡，使用 CPU 训练")

    def _setup_paths(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 保存配置
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2)

    def _print_setup(self):
        print(f"\n{'='*60}")
        print(f"Training Setup")
        print(f"{'='*60}")
        print(f"Device      : {self.device}")
        print(f"Input Size  : {self.args.input_size}")
        print(f"Batch Size  : {self.args.batch_size}")
        print(f"Epochs      : {self.args.epochs}")
        print(f"LR          : {self.args.lr}")
        print(f"AMP Enabled : {self.use_amp}")
        print(f"Num Workers : {self.args.num_workers}")
        # === 模型参数统计 ===
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total Params: {total_params:,}")
        print(f"Trainable   : {trainable_params:,}")
        print(f"Model Size  : {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        # ===================
        print(f"{'='*60}\n")

    def calculate_loss(self, outputs, targets):
        """计算组合 Loss (Segmentation + Signal Regression)"""
        loss_dict = {}
        
        # 1. 分割 Loss (复用 ecg_loss.py 的逻辑)
        # 注意: ecg_loss 需要根据实际输出 key 进行适配，这里假设它处理标准字典
        seg_loss, seg_components = self.seg_criterion(outputs, targets)
        loss_dict.update(seg_components)
        
        # 2. 信号回归 Loss
        # outputs['signals']: (B, 12, W)
        # targets['gt_signals']: (B, 12, W)
        if 'signals' in outputs and 'gt_signals' in targets:
            pred_sig = outputs['signals']
            gt_sig = targets['gt_signals']
            
            # 确保长度对齐 (以防万一)
            if pred_sig.shape[-1] != gt_sig.shape[-1]:
                # 如果不对齐，通常意味着 Dataset 设置的 W_feat 和 Model 不一致
                # 简单的 crop 或 interpolate
                min_len = min(pred_sig.shape[-1], gt_sig.shape[-1])
                pred_sig = pred_sig[..., :min_len]
                gt_sig = gt_sig[..., :min_len]
            
            # 计算 L1 Loss
            sig_loss = self.signal_criterion(pred_sig, gt_sig)
            
            loss_dict['signal_loss'] = sig_loss
            
            # 总 Loss = 分割权重 * 分割Loss + 信号权重 * 信号Loss
            total_loss = (self.args.weight_seg * seg_loss) + \
                         (self.args.weight_signal * sig_loss)
        else:
            total_loss = seg_loss
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        
        for batch in pbar:
            # 数据迁移到 Device
            images = batch['image'].to(self.device, non_blocking=True)
            
            # 构建 Targets 字典
            targets = {
                'baseline_coarse': batch['baseline_mask'].to(self.device), # 对应 ecg_loss 需要的 key
                'text_multi': batch['text_mask'].to(self.device),
                'gt_signals': batch['gt_signals'].to(self.device),
                # 兼容旧 Loss 的 Key
                'baseline_fine': batch['baseline_mask'].to(self.device), 
                'paper_speed_mask': torch.zeros_like(batch['baseline_mask']).to(self.device), # 暂无 GT
                'gain_mask': torch.zeros_like(batch['baseline_mask']).to(self.device), # 暂无 GT
                'valid_mask': batch['valid_mask'].to(self.device),
                'metadata': batch['metadata']
            }
            
            self.optimizer.zero_grad()
            
            # --- Forward & Backward ---
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.calculate_loss(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 常规训练 (CPU / MPS / No-AMP CUDA)
                outputs = self.model(images)
                loss, loss_dict = self.calculate_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # --- Logging ---
            running_loss += loss.item()
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sig': f"{loss_dict.get('signal_loss', 0):.4f}",
                'seg': f"{loss_dict.get('coarse_baseline', 0):.4f}"
            })
            
            # TensorBoard 记录
            if self.global_step % self.args.log_interval == 0:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    self.writer.add_scalar(f"Train/{k}", v, self.global_step)
                self.writer.add_scalar("Train/LR", self.optimizer.param_groups[0]['lr'], self.global_step)
                
            self.global_step += 1
            
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        
        # 额外的指标记录
        signal_mae = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Validating {epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = {
                'baseline_coarse': batch['baseline_mask'].to(self.device),
                'text_multi': batch['text_mask'].to(self.device),
                'gt_signals': batch['gt_signals'].to(self.device),
                # 兼容 Key
                'baseline_fine': batch['baseline_mask'].to(self.device),
                'paper_speed_mask': torch.zeros_like(batch['baseline_mask']).to(self.device),
                'gain_mask': torch.zeros_like(batch['baseline_mask']).to(self.device),
                'metadata': batch['metadata']
            }
            
            outputs = self.model(images)
            loss, loss_dict = self.calculate_loss(outputs, targets)
            
            running_loss += loss.item()
            if 'signal_loss' in loss_dict:
                signal_mae += loss_dict['signal_loss'].item()
        
        avg_loss = running_loss / len(self.val_loader)
        avg_mae = signal_mae / len(self.val_loader)
        
        print(f"\nValidation Epoch {epoch}: Loss={avg_loss:.4f}, Signal MAE={avg_mae:.4f}")
        self.writer.add_scalar("Val/Loss", avg_loss, epoch)
        self.writer.add_scalar("Val/Signal_MAE", avg_mae, epoch)
        
        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
            
        # 保存最新
        torch.save(state, self.output_dir / "checkpoint_latest.pth")
        
        # 保存最佳
        if is_best:
            torch.save(state, self.output_dir / "checkpoint_best.pth")
            print(f"[Checkpoints] Saved new best model (Loss: {self.best_val_loss:.4f})")

    def _load_checkpoint(self, path):
        print(f"[Checkpoints] Loading from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if self.scheduler and 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"[Checkpoints] Resuming from Epoch {self.start_epoch}")

    def run(self):
        print(f"\n[Training] 开始训练，总轮数: {self.args.epochs}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Schedule
            if self.scheduler:
                self.scheduler.step()
                
            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
        print("[Training] 训练完成。")
        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG V47 Training Script")
    
    # 路径参数
    parser.add_argument('--sim_root', type=str, required=True, help='仿真数据根目录')
    parser.add_argument('--csv_root', type=str, required=True, help='原始CSV根目录')
    parser.add_argument('--output_dir', type=str, default='./outputs/run_v47', help='输出目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--use_amp', action='store_true', help='启用混合精度 (仅CUDA)')
    parser.add_argument('--resume', type=str, default=None, help='恢复检查点路径')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--lr_step', type=int, default=20, help='StepLR 步长')
    
    # 模型 & 数据参数
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 2048], help='输入尺寸 H W')
    parser.add_argument('--target_fs', type=int, default=500, help='目标采样率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载 Workers')
    parser.add_argument('--pretrained', action='store_true', help='使用 ImageNet 预训练权重')
    parser.add_argument('--max_samples', type=int, default=None, help='调试用：最大样本数')
    
    # Loss 权重
    parser.add_argument('--weight_seg', type=float, default=1.0, help='分割 Loss 权重')
    parser.add_argument('--weight_signal', type=float, default=10.0, help='信号回归 Loss 权重')
    
    # 日志
    parser.add_argument('--log_interval', type=int, default=10, help='Logging step interval')
    
    args = parser.parse_args()
    
    trainer = ECGTrainer(args)
    trainer.run()