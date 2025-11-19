"""
ECG V48 训练脚本 - MPS 优化版
主要修改: 使用 MPS 优化的模型，避免 grid_sample backward 回退
"""

import torch.nn.functional as F

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

# 🔥 关键修改：导入 MPS 优化版模型
from ecg_dataset_v48_fixed import create_dataloaders
from ecg_model_v48_mps_optimized import ProgressiveLeadLocalizationModelV48MPS
from ecg_loss_v48_fixed import ProgressiveLeadLocalizationLossV48, ProgressiveWeightScheduler


class ECGTrainerV48MPS:
    def __init__(self, args):
        self.args = args
        self._setup_device()
        self._setup_paths()
        
        # 数据加载
        print(f"\n[Data] 初始化数据加载器...")
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
        
        # 🔥 使用 MPS 优化模型
        print(f"[Model] 构建 MPS 优化模型...")
        self.model = ProgressiveLeadLocalizationModelV48MPS(
            num_leads=12,
            roi_height=32,
            pretrained=args.pretrained
        ).to(self.device)
        
        if self.device.type == 'mps':
            print(f"  ✓ 使用整数切片 RoI 提取（避免 grid_sample backward 回退）")
        
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
        
        if args.resume:
            self._load_checkpoint(args.resume)
        
        self._print_setup()

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[System] 使用 CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"[System] 使用 MPS (优化版，无 grid_sample 回退)")
        else:
            self.device = torch.device("cpu")
            print(f"[System] 使用 CPU")

    def _setup_paths(self):
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2)

    def _print_setup(self):
        print(f"\n{'='*60}")
        print(f"Training Setup V48 MPS-Optimized")
        print(f"{'='*60}")
        print(f"Device      : {self.device}")
        print(f"Optimization: Fast RoI Extraction (No MPS Fallback)")
        print(f"Batch Size  : {self.args.batch_size}")
        print(f"Epochs      : {self.args.epochs}")
        print(f"Warmup      : {self.args.warmup_epochs}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Params: {total_params:,}")
        print(f"{'='*60}\n")

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
        self.model.train()
        self.weight_scheduler.step(epoch)
        
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            targets = self.build_targets(batch)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'sig': f"{loss_dict.get('loss_signal', torch.tensor(0.0)).item():.4f}",
                'seg': f"{loss_dict.get('loss_fine', torch.tensor(0.0)).item():.4f}"
            })
            
            if self.global_step % self.args.log_interval == 0:
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        self.writer.add_scalar(f"Train/{k}", v.item(), self.global_step)
            
            self.global_step += 1
        
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        
        running_loss = 0.0
        signal_corr = []
        
        for batch in tqdm(self.val_loader, desc=f"Val {epoch}"):
            images = batch['image'].to(self.device)
            targets = self.build_targets(batch)
            
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            
            if 'signals' in outputs:
                pred_sig = outputs['signals'].cpu().numpy()
                gt_sig = targets['gt_signals'].cpu().numpy()
                mask = targets['valid_mask'].cpu().numpy()
                
                for b in range(pred_sig.shape[0]):
                    for l in range(12):
                        valid_idx = mask[b, l] > 0.5
                        if valid_idx.sum() > 10:
                            try:
                                corr, _ = pearsonr(pred_sig[b, l, valid_idx], 
                                                   gt_sig[b, l, valid_idx])
                                if not np.isnan(corr):
                                    signal_corr.append(corr)
                            except:
                                pass
        
        avg_loss = running_loss / len(self.val_loader)
        avg_corr = np.mean(signal_corr) if signal_corr else 0.0
        
        print(f"\nValidation Epoch {epoch}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Signal Corr: {avg_corr:.4f}")
        
        self.writer.add_scalar("Val/Loss", avg_loss, epoch)
        self.writer.add_scalar("Val/Signal_Correlation", avg_corr, epoch)
        
        return avg_loss

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
            print(f"💾 Saved best model (Loss: {self.best_val_loss:.4f})")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    def run(self):
        print(f"\n[Training] 🚀 开始训练 (MPS 优化版)")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print(f"\n✅ 训练完成！Best Val Loss: {self.best_val_loss:.4f}")
        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sim_root', type=str, required=True)
    parser.add_argument('--csv_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/mps_opt')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 2048])
    parser.add_argument('--target_fs', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()
    
    trainer = ECGTrainerV48MPS(args)
    trainer.run()