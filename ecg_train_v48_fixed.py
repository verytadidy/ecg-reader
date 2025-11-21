"""
ECG V48 训练脚本 - MPS 优化版 + 详细日志
集成了完整的 loss 分解打印功能
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
import gc

# 导入模块
from ecg_dataset_v48_fixed import create_dataloaders
from ecg_model_v48_mps_optimized import ProgressiveLeadLocalizationModelV48Lite
from ecg_loss_v48_fixed import ProgressiveLeadLocalizationLossV48, ProgressiveWeightScheduler


# ========== 颜色输出 ==========
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text, color=Colors.ENDC):
    """彩色打印"""
    print(f"{color}{text}{Colors.ENDC}")


def format_number(value, decimals=4):
    """格式化数字"""
    if isinstance(value, torch.Tensor):
        value = value.item()
    return f"{value:.{decimals}f}"


def get_loss_status(loss_val):
    """根据 loss 值返回颜色和状态符号"""
    if loss_val < 0.2:
        return Colors.GREEN, "✓"
    elif loss_val < 0.5:
        return Colors.YELLOW, "~"
    else:
        return Colors.RED, "✗"


# ========== 主训练器 ==========
class ECGTrainerV48MPS:
    def __init__(self, args):
        self.args = args
        self._setup_device()
        self._setup_paths()
        
        # 数据加载
        print_colored(f"\n[Data] 初始化数据加载器...", Colors.BLUE)
        self.train_loader, self.val_loader = create_dataloaders(
            sim_root=args.sim_root,
            csv_root=args.csv_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=tuple(args.input_size),
            target_fs=args.target_fs,
            max_samples=args.max_samples,
            train_split=0.9
        )
        
        # 模型
        print_colored(f"[Model] 构建 MPS 优化模型...", Colors.BLUE)
        self.model = ProgressiveLeadLocalizationModelV48Lite(
            num_leads=12,
            roi_height=32,
            pretrained=args.pretrained
        ).to(self.device)
        
        if self.device.type == 'mps':
            print_colored(f"  ✓ 使用整数切片 RoI 提取（避免 grid_sample backward 回退）", Colors.GREEN)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        if args.scheduler == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        else:
            self.lr_scheduler = None
        
        # 损失函数
        self.criterion = ProgressiveLeadLocalizationLossV48(
            weight_coarse_baseline=1.0,
            weight_text=1.0,
            weight_wave_seg=5.0,
            weight_lead_baseline=5.0,
            weight_signal=10.0,
            weight_aux_suppress=0.5,
            weight_ocr=0.5,
            background_weight=0.1,
            use_focal_loss=True
        ).to(self.device)
        
        self.weight_scheduler = ProgressiveWeightScheduler(
            self.criterion, 
            total_epochs=args.epochs, 
            warmup_epochs=args.warmup_epochs
        )
        
        # 状态
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.loss_history = defaultdict(list)
        
        if args.resume:
            self._load_checkpoint(args.resume)
        
        self._print_setup()

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print_colored(f"[System] 使用 CUDA: {torch.cuda.get_device_name(0)}", Colors.GREEN)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print_colored(f"[System] 使用 MPS (优化版，无 grid_sample 回退)", Colors.GREEN)
        else:
            self.device = torch.device("cpu")
            print_colored(f"[System] 使用 CPU", Colors.YELLOW)

    def _setup_paths(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2)

    def _print_setup(self):
        print_colored("\n" + "="*80, Colors.CYAN)
        print_colored("ECG V48 MPS 优化版训练系统 (含详细日志)", Colors.CYAN)
        print_colored("="*80, Colors.CYAN)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"""
设备配置:
  Device:        {self.device}
  Batch Size:    {self.args.batch_size}
  Workers:       {self.args.num_workers}
  
模型配置:
  总参数:        {total_params:,} ({total_params/1e6:.1f}M)
  可训练:        {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}
  
训练配置:
  Epochs:        {self.args.epochs}
  Warmup:        {self.args.warmup_epochs}
  学习率:        {self.args.lr}
  
数据配置:
  训练集:        {len(self.train_loader)} batches
  验证集:        {len(self.val_loader)} batches
""")
        print_colored("="*80 + "\n", Colors.CYAN)

    def build_targets(self, batch):
        coarse_target = batch['baseline_mask'].max(dim=1, keepdim=True)[0]
        
        targets = {
            'baseline_coarse': coarse_target.to(self.device),
            'baseline_fine': batch['baseline_mask'].to(self.device),
            'text_multi': batch['text_mask'].to(self.device),
            'wave_segmentation': batch['wave_segmentation'].to(self.device),
            'auxiliary_mask': batch['auxiliary_mask'].to(self.device),
            'paper_speed_mask': batch['paper_speed_mask'].to(self.device),
            'gain_mask': batch['gain_mask'].to(self.device),
            'gt_signals': batch['gt_signals'].to(self.device),
            'valid_mask': batch['valid_mask'].to(self.device),
            'metadata': batch['metadata']
        }
        
        return targets

    def train_epoch(self, epoch):
        """训练一个 epoch，打印详细日志"""
        self.model.train()
        self.weight_scheduler.step(epoch)
        
        loss_stats = defaultdict(list)
        running_loss = 0.0
        
        # 获取当前权重
        weights = self.criterion.weights
        print_colored(f"\n--- Epoch {epoch}/{self.args.epochs} ---", Colors.BOLD)
        print_colored(f"Loss 权重配置:", Colors.CYAN)
        for k, v in weights.items():
            print(f"  {k:20s}: {v:.4f}")
        
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            targets = self.build_targets(batch)
            
            del batch
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播
            with torch.autocast(device_type=self.device.type, enabled=self.args.use_amp):
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)
            
            del outputs, images
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    loss_stats[k].append(v.item())
            
            # 定期清理内存
            if (self.global_step + 1) % 20 == 0:
                if self.device.type == 'mps':
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                gc.collect()
            
            pbar.set_postfix({
                'total': f"{loss.item():.4f}",
                'sig': f"{loss_dict.get('loss_signal', torch.tensor(0.0)).item():.4f}",
                'fine': f"{loss_dict.get('loss_fine', torch.tensor(0.0)).item():.4f}",
            })
            
            if self.global_step % self.args.log_interval == 0:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        self.writer.add_scalar(f"Train/{k}", v.item(), self.global_step)
            
            self.global_step += 1
        
        # 计算平均值
        avg_losses = {k: np.mean(v) for k, v in loss_stats.items()}
        avg_total = running_loss / len(self.train_loader)
        
        # 打印详细总结
        self._print_epoch_summary("Train", epoch, avg_losses, avg_total, weights)
        
        # 写入 tensorboard
        for k, v in avg_losses.items():
            self.writer.add_scalar(f"Train/{k}", v, epoch)
        self.writer.add_scalar("Train/total_loss", avg_total, epoch)
        
        # 保存历史
        self.loss_history['train_total'].append(avg_total)
        for k, v in avg_losses.items():
            self.loss_history[f'train_{k}'].append(v)
        
        return avg_total

    def _print_epoch_summary(self, mode, epoch, losses_dict, total_loss, weights=None):
        """打印 epoch 总结"""
        if mode == "Train":
            color = Colors.GREEN
            symbol = "📚"
        else:
            color = Colors.YELLOW
            symbol = "✅"
        
        print_colored(f"\n{symbol} {mode} Epoch {epoch} Summary:", color)
        print_colored("─" * 80, color)
        
        # 分组打印 loss
        loss_groups = {
            '定位任务': ['loss_coarse', 'loss_fine'],
            '分割任务': ['loss_text', 'loss_wave_seg'],
            '回归任务': ['loss_signal'],
            '约束任务': ['loss_aux_suppress', 'loss_ocr'],
        }
        
        total_weighted = 0.0
        if weights is None:
            weights = self.criterion.weights
        
        for group_name, loss_keys in loss_groups.items():
            group_losses = {k: losses_dict.get(k, 0.0) for k in loss_keys}
            group_losses = {k: v for k, v in group_losses.items() if v > 0}
            
            if group_losses:
                print(f"\n  {group_name}:")
                for loss_name, loss_val in group_losses.items():
                    weight = weights.get(loss_name, 1.0)
                    weighted_val = loss_val * weight
                    total_weighted += weighted_val
                    
                    status_color, status = get_loss_status(loss_val)
                    
                    print(f"{status_color}    {status} {loss_name:25s}: {format_number(loss_val, 6)} "
                          f"(w={weight:.1f}, wtd={format_number(weighted_val, 6)}){Colors.ENDC}")
        
        print(f"\n  {'─' * 76}")
        print_colored(f"  🎯 Total Loss: {format_number(total_loss, 6)}", Colors.BOLD)
        print_colored("─" * 80, color)

    @torch.no_grad()
    def validate(self, epoch):
        """验证，打印详细指标"""
        self.model.eval()
        
        running_loss = 0.0
        loss_stats = defaultdict(list)
        signal_metrics = {
            'corr': [],
            'mae': [],
            'rmse': [],
        }
        
        pbar = tqdm(self.val_loader, desc=f"Val  {epoch}", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = self.build_targets(batch)
            
            del batch
            
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    loss_stats[k].append(v.item())
            
            # 计算信号指标
            if 'signals' in outputs:
                pred_sig = outputs['signals'].cpu().numpy()
                gt_sig = targets['gt_signals'].cpu().numpy()
                mask = targets['valid_mask'].cpu().numpy()
                
                for b in range(pred_sig.shape[0]):
                    for l in range(12):
                        valid_idx = mask[b, l] > 0.5
                        if valid_idx.sum() > 10:
                            try:
                                # 相关系数
                                corr, _ = pearsonr(pred_sig[b, l, valid_idx],
                                                   gt_sig[b, l, valid_idx])
                                if not np.isnan(corr):
                                    signal_metrics['corr'].append(corr)
                                
                                # MAE
                                mae = np.mean(np.abs(pred_sig[b, l, valid_idx] - gt_sig[b, l, valid_idx]))
                                signal_metrics['mae'].append(mae)
                                
                                # RMSE
                                rmse = np.sqrt(np.mean((pred_sig[b, l, valid_idx] - gt_sig[b, l, valid_idx])**2))
                                signal_metrics['rmse'].append(rmse)
                            except:
                                pass
            
            del outputs, images, targets
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'corr': f"{np.mean(signal_metrics['corr']) if signal_metrics['corr'] else 0:.4f}",
            })
        
        avg_loss = running_loss / len(self.val_loader)
        avg_losses = {k: np.mean(v) for k, v in loss_stats.items()}
        
        # 计算平均指标
        avg_corr = np.mean(signal_metrics['corr']) if signal_metrics['corr'] else 0.0
        avg_mae = np.mean(signal_metrics['mae']) if signal_metrics['mae'] else 0.0
        avg_rmse = np.mean(signal_metrics['rmse']) if signal_metrics['rmse'] else 0.0
        
        # 打印验证总结
        print_colored(f"\n✅ Val Epoch {epoch} Summary:", Colors.YELLOW)
        print_colored("─" * 80, Colors.YELLOW)
        
        # Loss 详情
        loss_groups = {
            '定位任务': ['loss_coarse', 'loss_fine'],
            '分割任务': ['loss_text', 'loss_wave_seg'],
            '回归任务': ['loss_signal'],
            '约束任务': ['loss_aux_suppress', 'loss_ocr'],
        }
        
        for group_name, loss_keys in loss_groups.items():
            group_losses = {k: avg_losses.get(k, 0.0) for k in loss_keys}
            group_losses = {k: v for k, v in group_losses.items() if v > 0}
            
            if group_losses:
                print(f"\n  {group_name}:")
                for loss_name, loss_val in group_losses.items():
                    weight = self.criterion.weights.get(loss_name, 1.0)
                    status_color, status = get_loss_status(loss_val)
                    
                    print(f"{status_color}    {status} {loss_name:25s}: {format_number(loss_val, 6)} "
                          f"(w={weight:.1f}){Colors.ENDC}")
        
        # 信号指标
        corr_status = Colors.GREEN if avg_corr > 0.85 else (Colors.YELLOW if avg_corr > 0.70 else Colors.RED)
        print(f"\n  信号质量指标:")
        print(f"{corr_status}    📊 相关系数 (Correlation): {format_number(avg_corr, 6)}{Colors.ENDC}")
        print(f"    📊 平均误差 (MAE):        {format_number(avg_mae, 6)}")
        print(f"    📊 均方根误差 (RMSE):     {format_number(avg_rmse, 6)}")
        
        print(f"\n  {'─' * 76}")
        print_colored(f"  🎯 Total Loss: {format_number(avg_loss, 6)}", Colors.BOLD)
        print_colored("─" * 80, Colors.YELLOW)
        
        # 写入 tensorboard
        self.writer.add_scalar("Val/total_loss", avg_loss, epoch)
        for k, v in avg_losses.items():
            self.writer.add_scalar(f"Val/{k}", v, epoch)
        self.writer.add_scalar("Val/signal_correlation", avg_corr, epoch)
        self.writer.add_scalar("Val/signal_mae", avg_mae, epoch)
        self.writer.add_scalar("Val/signal_rmse", avg_rmse, epoch)
        
        # 保存历史
        self.loss_history['val_total'].append(avg_loss)
        self.loss_history['val_corr'].append(avg_corr)
        for k, v in avg_losses.items():
            self.loss_history[f'val_{k}'].append(v)
        
        return avg_loss, avg_corr

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }
        
        torch.save(state, self.output_dir / "checkpoint_latest.pth")
        if is_best:
            torch.save(state, self.output_dir / "checkpoint_best.pth")
            print_colored(f"💾 已保存最佳模型 (Loss: {self.best_val_loss:.4f})", Colors.GREEN)

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    def run(self):
        print_colored(f"\n🚀 开始训练！", Colors.GREEN)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_corr = self.validate(epoch)
            
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print_colored(f"\n🎉 新的最佳验证 Loss: {self.best_val_loss:.4f}\n", Colors.GREEN)
            
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print_colored(f"\n✅ 训练完成！最佳 Loss: {self.best_val_loss:.4f}", Colors.GREEN)
        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sim_root', type=str, required=True)
    parser.add_argument('--csv_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/mps_opt')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 2048])
    parser.add_argument('--target_fs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()
    
    trainer = ECGTrainerV48MPS(args)
    trainer.run()