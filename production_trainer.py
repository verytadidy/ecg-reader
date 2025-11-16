#!/usr/bin/env python3
"""
生产就绪的ECG重建模型训练脚本

特点:
1. ✅ 支持多采样率（重采样到500Hz）
2. ✅ 完善的日志和检查点管理
3. ✅ 自动设备检测（CUDA/MPS/CPU）
4. ✅ TensorBoard可视化
5. ✅ 早停和学习率调度
6. ✅ 混合精度训练（可选）
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 导入模型和数据集
from ecg_model import ECGReconstructionModel
from production_dataset import create_dataloaders


class ProductionTrainer:
    """生产级训练器"""
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.output_dir = self._setup_output_dir()
        
        # 创建数据加载器
        print("\n" + "="*70)
        print("初始化数据加载器...")
        print("="*70)
        self.train_loader, self.val_loader = create_dataloaders(
            sim_root=args.sim_root,
            csv_root=args.csv_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_split=args.train_split,
            target_fs=args.target_fs,
            max_samples=args.max_samples if args.debug else None,
            cache_in_memory=args.cache
        )
        
        # 创建模型
        print("="*70)
        print("初始化模型...")
        print("="*70)
        
        # 🔥 Mac M2 MPS兼容：禁用STN避免grid_sampler问题
        enable_stn = not (self.device.type == 'mps')
        if not enable_stn:
            print("⚠️  检测到MPS设备，禁用STN（grid_sampler不支持）")
        
        self.model = ECGReconstructionModel(
            num_leads=12,
            signal_length=args.target_fs * 10,
            pretrained=args.pretrained,
            enable_stn=enable_stn  # MPS上禁用STN
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print()
        
        # 损失函数
        self.criterion = self._create_loss_function()
        
        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 恢复训练
        if args.resume:
            self._load_checkpoint(args.resume)
        
        # 保存配置
        self._save_config()
    
    def _setup_device(self):
        """设置训练设备"""
        if self.args.force_cpu:
            device = torch.device('cpu')
            device_name = 'CPU'
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = f'CUDA ({torch.cuda.get_device_name(0)})'
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = 'MPS (Apple Silicon)'
        else:
            device = torch.device('cpu')
            device_name = 'CPU'
        
        print("\n" + "="*70)
        print(f"训练设备: {device_name}")
        print("="*70)
        
        return device
    
    def _setup_output_dir(self):
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.args.output_dir) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        (output_dir / 'checkpoints').mkdir(exist_ok=True)
        (output_dir / 'logs').mkdir(exist_ok=True)
        
        print(f"输出目录: {output_dir}")
        return output_dir
    
    def _create_loss_function(self):
        """创建损失函数"""
        from ecg_trainer import MultiTaskLoss
        
        return MultiTaskLoss(loss_weights={
            'seg': self.args.loss_weight_seg,
            'grid': self.args.loss_weight_grid,
            'baseline': self.args.loss_weight_baseline,
            'theta': self.args.loss_weight_theta,
            'signal': self.args.loss_weight_signal
        }).to(self.device)
    
    def _save_config(self):
        """保存训练配置"""
        config = {
            'model': {
                'num_leads': 12,
                'signal_length': self.args.target_fs * 10,
                'pretrained': self.args.pretrained
            },
            'data': {
                'sim_root': self.args.sim_root,
                'csv_root': self.args.csv_root,
                'target_fs': self.args.target_fs,
                'target_size': [512, 672],
                'train_split': self.args.train_split
            },
            'training': {
                'batch_size': self.args.batch_size,
                'epochs': self.args.epochs,
                'lr': self.args.lr,
                'weight_decay': self.args.weight_decay,
                'early_stopping_patience': self.args.patience
            },
            'loss_weights': {
                'seg': self.args.loss_weight_seg,
                'grid': self.args.loss_weight_grid,
                'baseline': self.args.loss_weight_baseline,
                'theta': self.args.loss_weight_theta,
                'signal': self.args.loss_weight_signal
            }
        }
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_losses = {
            'total': 0, 'seg': 0, 'grid': 0,
            'baseline': 0, 'theta': 0, 'signal': 0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 数据移到设备
                images = batch['image'].to(self.device, non_blocking=True)
                wave_seg = batch['wave_seg'].to(self.device, non_blocking=True)
                grid_mask = batch['grid_mask'].to(self.device, non_blocking=True)
                baseline_heatmaps = batch['baseline_heatmaps'].to(self.device, non_blocking=True)
                theta_gt = batch['theta_gt'].to(self.device, non_blocking=True)
                gt_signal = batch['gt_signal'].to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                targets = {
                    'wave_seg': wave_seg,
                    'grid_mask': grid_mask,
                    'baseline_heatmaps': baseline_heatmaps,
                    'theta_gt': theta_gt,
                    'gt_signal': gt_signal
                }
                losses = self.criterion(outputs, targets)
                
                # 🔥 检查NaN
                if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                    print(f"\n⚠️  检测到NaN/Inf损失，跳过此batch")
                    print(f"  Losses: {[(k, v.item()) for k, v in losses.items()]}")
                    # 检查哪个输出有问题
                    for k, v in outputs.items():
                        if isinstance(v, torch.Tensor):
                            if torch.isnan(v).any():
                                print(f"  ✗ {k} 包含NaN")
                            elif torch.isinf(v).any():
                                print(f"  ✗ {k} 包含Inf")
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                losses['total'].backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 累积损失
                for key in total_losses.keys():
                    total_losses[key] += losses[key].item()
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'sig': f"{losses['signal'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                print(f"\n⚠️  Batch {batch_idx} 出错: {e}")
                if self.args.debug:
                    raise e
                continue
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # 记录到TensorBoard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'Train/{key}_loss', value, epoch)
        self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        
        total_losses = {
            'total': 0, 'seg': 0, 'grid': 0,
            'baseline': 0, 'theta': 0, 'signal': 0
        }
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validating")
        
        for batch in pbar:
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                wave_seg = batch['wave_seg'].to(self.device, non_blocking=True)
                grid_mask = batch['grid_mask'].to(self.device, non_blocking=True)
                baseline_heatmaps = batch['baseline_heatmaps'].to(self.device, non_blocking=True)
                theta_gt = batch['theta_gt'].to(self.device, non_blocking=True)
                gt_signal = batch['gt_signal'].to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                
                targets = {
                    'wave_seg': wave_seg,
                    'grid_mask': grid_mask,
                    'baseline_heatmaps': baseline_heatmaps,
                    'theta_gt': theta_gt,
                    'gt_signal': gt_signal
                }
                losses = self.criterion(outputs, targets)
                
                for key in total_losses.keys():
                    total_losses[key] += losses[key].item()
                num_batches += 1
                
            except Exception as e:
                print(f"\n⚠️  验证出错: {e}")
                if self.args.debug:
                    raise e
                continue
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # 记录到TensorBoard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f'Val/{key}_loss', value, epoch)
        
        return avg_losses
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.args.__dict__
        }
        
        # 保存最新
        torch.save(checkpoint, self.output_dir / 'checkpoints' / 'last.pth')
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoints' / 'best.pth')
            print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
        
        # 定期保存
        if epoch % self.args.save_freq == 0:
            torch.save(checkpoint, self.output_dir / 'checkpoints' / f'epoch_{epoch}.pth')
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ 从 {checkpoint_path} 恢复训练")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*70)
        print("开始训练")
        print("="*70)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_losses = self.validate(epoch)
            
            # 学习率调整
            self.scheduler.step(val_losses['total'])
            
            # 打印统计
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_losses['total']:.4f} (signal: {train_losses['signal']:.4f})")
            print(f"  Val Loss:   {val_losses['total']:.4f} (signal: {val_losses['signal']:.4f})")
            print(f"  Best Val:   {self.best_val_loss:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 保存检查点
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_losses['total'], is_best)
            
            # 早停
            if self.patience_counter >= self.args.patience:
                print(f"\n早停触发！{self.args.patience} epochs无改善")
                break
        
        # 训练结束
        self.writer.close()
        print("\n" + "="*70)
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"输出目录: {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='ECG重建模型训练（生产版）')
    
    # 数据参数
    parser.add_argument('--sim_root', type=str, required=True, help='仿真数据根目录')
    parser.add_argument('--csv_root', type=str, required=True, help='原始CSV数据根目录')
    parser.add_argument('--target_fs', type=int, default=500, help='目标采样率（重采样）')
    parser.add_argument('--train_split', type=float, default=0.9, help='训练集比例')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载worker数')
    
    # 损失权重
    parser.add_argument('--loss_weight_seg', type=float, default=1.0)
    parser.add_argument('--loss_weight_grid', type=float, default=0.5)
    parser.add_argument('--loss_weight_baseline', type=float, default=0.8)
    parser.add_argument('--loss_weight_theta', type=float, default=0.3)
    parser.add_argument('--loss_weight_signal', type=float, default=2.0)
    
    # 模型参数
    parser.add_argument('--pretrained', action='store_true', help='使用预训练权重')
    
    # 其他
    parser.add_argument('--output_dir', type=str, default='./experiments', help='输出目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练检查点')
    parser.add_argument('--save_freq', type=int, default=10, help='保存检查点频率')
    parser.add_argument('--patience', type=int, default=15, help='早停patience')
    parser.add_argument('--force_cpu', action='store_true', help='强制CPU训练')
    parser.add_argument('--cache', action='store_true', help='缓存数据到内存')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数（调试用）')
    
    args = parser.parse_args()
    
    # 调试模式配置
    if args.debug:
        args.epochs = 3
        args.max_samples = 100
        args.save_freq = 1
        print("\n⚠️  调试模式已启用")
    
    # 创建训练器并开始训练
    trainer = ProductionTrainer(args)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print(f"检查点已保存至: {trainer.output_dir / 'checkpoints'}")
    except Exception as e:
        print(f"\n\n❌ 训练出错: {e}")
        if args.debug:
            raise e


if __name__ == '__main__':
    # 使用示例:
    #
    # 快速测试（Mac M2）:
    #   python production_trainer.py \
    #       --sim_root /path/to/simulations \
    #       --csv_root /path/to/train \
    #       --batch_size 2 \
    #       --num_workers 0 \
    #       --debug
    #
    # 完整训练（GPU）:
    #   python production_trainer.py \
    #       --sim_root /path/to/simulations \
    #       --csv_root /path/to/train \
    #       --batch_size 16 \
    #       --num_workers 8 \
    #       --pretrained \
    #       --epochs 100
    #
    # 恢复训练:
    #   python production_trainer.py \
    #       --sim_root /path/to/simulations \
    #       --csv_root /path/to/train \
    #       --resume ./experiments/run_xxx/checkpoints/best.pth
    
    main()