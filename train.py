"""
ECG V45 训练脚本

特点:
1. ✅ 支持混合精度训练（AMP）
2. ✅ 完整的检查点管理
3. ✅ TensorBoard可视化
4. ✅ 多GPU训练支持
5. ✅ 渐进式学习率调度
6. ✅ 早停机制
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from pathlib import Path
import time
import numpy as np

# 假设模型和数据集在同一目录
from ecg_dataset import create_dataloaders
from ecg_model import ProgressiveLeadLocalizationModel
from ecg_loss import ProgressiveLeadLocalizationLoss, CombinedLossWithConstraints


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')
        
        # 构建模型
        print("构建模型...")
        self.model = ProgressiveLeadLocalizationModel(
            num_leads=12,
            encoder_name='resnet50',
            pretrained=args.pretrained
        ).to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1 and args.multi_gpu:
            print(f"使用 {torch.cuda.device_count()} 个GPU训练")
            self.model = nn.DataParallel(self.model)
        
        # 构建损失函数
        base_criterion = ProgressiveLeadLocalizationLoss(
            weight_coarse_baseline=args.weight_coarse_baseline,
            weight_time_range=args.weight_time_range,
            weight_text=args.weight_text,
            weight_auxiliary=args.weight_auxiliary,
            weight_paper_speed=args.weight_paper_speed,
            weight_gain=args.weight_gain,
            weight_lead_baseline=args.weight_lead_baseline,
            use_focal_loss=args.use_focal_loss
        )
        
        if args.use_physical_constraints:
            self.criterion = CombinedLossWithConstraints(
                base_criterion,
                weight_constraint=args.weight_constraint
            )
        else:
            self.criterion = base_criterion
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        if args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.lr_step,
                gamma=0.1
            )
        elif args.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # 混合精度训练
        self.use_amp = args.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # 数据加载器
        print("\n加载数据集...")
        self.train_loader, self.val_loader = create_dataloaders(
            sim_root=args.sim_root,
            csv_root=args.csv_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_split=args.train_split,
            target_fs=args.target_fs,
            target_size=tuple(args.input_size),
            max_samples=args.max_samples,
            cache_in_memory=args.cache_in_memory,
            load_fine_labels=True,
            load_ocr_labels=True
        )
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # 保存配置
        self._save_config()
        
        print(f"\n{'='*80}")
        print(f"训练器初始化完成")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Mixed precision: {'✓' if self.use_amp else '✗'}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
    
    def _save_config(self):
        """保存训练配置"""
        config = vars(self.args)
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {self.current_epoch}/{self.args.epochs}',
                   leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['image'].to(self.device, non_blocking=True)
            
            targets = {
                'baseline_coarse': batch['baseline_coarse'].to(self.device, non_blocking=True),
                'baseline_fine': batch['baseline_fine'].to(self.device, non_blocking=True),
                'text_multi': batch['text_multi'].to(self.device, non_blocking=True),
                'auxiliary': batch['auxiliary'].to(self.device, non_blocking=True),
                'paper_speed_mask': batch['paper_speed_mask'].to(self.device, non_blocking=True),
                'gain_mask': batch['gain_mask'].to(self.device, non_blocking=True),
                'metadata': batch['metadata']
            }
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                if self.args.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准训练
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += val.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ps': f'{loss_dict.get("paper_speed", torch.tensor(0)).item():.4f}',
                'text': f'{loss_dict.get("text", torch.tensor(0)).item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 定期记录到TensorBoard
            if self.global_step % self.args.log_interval == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                for key, val in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', val.item(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        
        total_loss = 0
        loss_components = {}
        
        # 额外指标
        metrics = {
            'paper_speed_iou': [],
            'gain_iou': [],
            'text_iou': [],
            'baseline_iou': []
        }
        
        for batch in tqdm(self.val_loader, desc='Validating', leave=False):
            images = batch['image'].to(self.device, non_blocking=True)
            
            targets = {
                'baseline_coarse': batch['baseline_coarse'].to(self.device, non_blocking=True),
                'baseline_fine': batch['baseline_fine'].to(self.device, non_blocking=True),
                'text_multi': batch['text_multi'].to(self.device, non_blocking=True),
                'auxiliary': batch['auxiliary'].to(self.device, non_blocking=True),
                'paper_speed_mask': batch['paper_speed_mask'].to(self.device, non_blocking=True),
                'gain_mask': batch['gain_mask'].to(self.device, non_blocking=True),
                'metadata': batch['metadata']
            }
            
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += val.item()
            
            # 计算IoU指标
            metrics['paper_speed_iou'].append(
                self._calculate_iou(outputs['paper_speed_mask'], targets['paper_speed_mask'])
            )
            metrics['gain_iou'].append(
                self._calculate_iou(outputs['gain_mask'], targets['gain_mask'])
            )
            metrics['text_iou'].append(
                self._calculate_iou(outputs['text_masks'].mean(dim=1, keepdim=True), 
                                   targets['text_multi'].mean(dim=1, keepdim=True))
            )
            metrics['baseline_iou'].append(
                self._calculate_iou(outputs['lead_baselines'].mean(dim=1, keepdim=True),
                                   targets['baseline_fine'].mean(dim=1, keepdim=True))
            )
        
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in loss_components.items()}
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        return avg_loss, avg_components, avg_metrics
    
    def _calculate_iou(self, pred, target, threshold=0.5):
        """计算IoU"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.item()
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            print(f"  ✓ 保存最佳模型 (val_loss={self.best_val_loss:.4f})")
        
        # 定期保存
        if self.current_epoch % self.args.save_interval == 0:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pth')
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载scaler
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 加载训练状态
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ 从epoch {self.current_epoch}恢复训练")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*80)
        print("开始训练...")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.args.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_components = self.train_epoch()
            
            # 验证
            val_loss, val_components, val_metrics = self.validate()
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录到TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            for key, val in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', val, epoch)
            
            # 打印统计
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            print(f"{'='*80}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"  ├─ Paper Speed: {train_components.get('paper_speed', 0):.4f} ⭐⭐⭐⭐⭐")
            print(f"  ├─ Gain: {train_components.get('gain', 0):.4f} ⭐⭐⭐")
            print(f"  ├─ Text: {train_components.get('text', 0):.4f}")
            print(f"  └─ Lead Baseline: {train_components.get('lead_baseline', 0):.4f}")
            print(f"\nVal Loss: {val_loss:.4f}")
            print(f"  ├─ Paper Speed: {val_components.get('paper_speed', 0):.4f} ⭐⭐⭐⭐⭐")
            print(f"  └─ Gain: {val_components.get('gain', 0):.4f} ⭐⭐⭐")
            print(f"\nMetrics:")
            print(f"  ├─ Paper Speed IoU: {val_metrics['paper_speed_iou']:.4f}")
            print(f"  ├─ Gain IoU: {val_metrics['gain_iou']:.4f}")
            print(f"  ├─ Text IoU: {val_metrics['text_iou']:.4f}")
            print(f"  └─ Baseline IoU: {val_metrics['baseline_iou']:.4f}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            self.save_checkpoint(is_best=is_best)
            
            # 早停
            if self.args.early_stop > 0 and self.early_stop_counter >= self.args.early_stop:
                print(f"\n⚠️  早停触发 ({self.early_stop_counter} epochs without improvement)")
                break
        
        # 训练结束
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("训练完成！")
        print("="*80)
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最佳模型保存在: {self.output_dir / 'checkpoint_best.pth'}")
        print("="*80 + "\n")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train ECG V45 Model')
    
    # 数据参数
    parser.add_argument('--sim_root', type=str, required=True, help='仿真数据根目录')
    parser.add_argument('--csv_root', type=str, required=True, help='CSV数据根目录')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--input_size', nargs=2, type=int, default=[512, 672], help='输入尺寸 [H, W]')
    parser.add_argument('--target_fs', type=int, default=500, help='目标采样率')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数(用于测试)')
    
    # 模型参数
    parser.add_argument('--pretrained', action='store_true', help='使用预训练ResNet')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--train_split', type=float, default=0.9, help='训练集比例')
    
    # 损失权重
    parser.add_argument('--weight_coarse_baseline', type=float, default=1.0)
    parser.add_argument('--weight_time_range', type=float, default=0.5)
    parser.add_argument('--weight_text', type=float, default=2.0)
    parser.add_argument('--weight_auxiliary', type=float, default=1.0)
    parser.add_argument('--weight_paper_speed', type=float, default=5.0, help='⭐⭐⭐⭐⭐')
    parser.add_argument('--weight_gain', type=float, default=3.0, help='⭐⭐⭐')
    parser.add_argument('--weight_lead_baseline', type=float, default=2.0)
    parser.add_argument('--use_focal_loss', action='store_true', help='使用Focal Loss')
    
    # 物理约束
    parser.add_argument('--use_physical_constraints', action='store_true', help='使用物理约束')
    parser.add_argument('--weight_constraint', type=float, default=0.1)
    
    # 优化器和调度器
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'])
    parser.add_argument('--lr_step', type=int, default=30, help='StepLR的步长')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    
    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--cache_in_memory', action='store_true', help='缓存数据到内存')
    parser.add_argument('--multi_gpu', action='store_true', help='使用多GPU训练')
    
    # 日志和保存
    parser.add_argument('--log_interval', type=int, default=10, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=10, help='检查点保存间隔')
    parser.add_argument('--early_stop', type=int, default=0, help='早停patience (0=禁用)')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()