"""
ECG V48 独立评估脚本 - 完全重写版本
不依赖任何旧代码，避免缓存问题
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import warnings

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ============ 颜色输出 ============
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_c(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")


# ============ 主类 ============
class ECGEvaluator:
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
    
    def __init__(self, checkpoint, sim_root, csv_root, sample_id=None, output_dir='./eval_results'):
        self.checkpoint = checkpoint
        self.sim_root = Path(sim_root)
        self.csv_root = Path(csv_root)
        self.sample_id = sample_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print_c(f"\n使用设备: {self.device}\n", Colors.GREEN)
        
        # 加载模型
        self.model = self._load_model()
        # 加载数据集
        self.dataset = self._load_dataset()
    
    def _load_model(self):
        """加载模型"""
        print_c("加载模型...", Colors.CYAN)
        try:
            from ecg_model_v48_mps_optimized import ProgressiveLeadLocalizationModelV48Lite
        except:
            from ecg_model_v48_fixed import ProgressiveLeadLocalizationModelV48LiteFixed as ProgressiveLeadLocalizationModelV48Lite
        
        model = ProgressiveLeadLocalizationModelV48Lite(num_leads=12, roi_height=32, pretrained=False).to(self.device)
        checkpoint = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print_c("✓ 模型加载成功\n", Colors.GREEN)
        return model
    
    def _load_dataset(self):
        """加载数据集"""
        print_c("加载数据集...", Colors.CYAN)
        from ecg_dataset_v48_fixed import ECGV48FixedDataset
        dataset = ECGV48FixedDataset(
            sim_root_dir=str(self.sim_root),
            csv_root_dir=str(self.csv_root),
            split='val',
            target_size=(512, 2048),
            augment=False,
            cache_images=False
        )
        print_c(f"✓ 数据集加载成功 ({len(dataset)} 样本)\n", Colors.GREEN)
        return dataset
    
    def _get_sample_idx(self):
        """获取样本索引"""
        if self.sample_id:
            for idx, s in enumerate(self.dataset.samples):
                if s['id'] == self.sample_id:
                    return idx
            print_c(f"✗ 未找到样本: {self.sample_id}\n", Colors.RED)
            return None
        else:
            idx = np.random.randint(0, len(self.dataset))
            sid = self.dataset.samples[idx]['id']
            print_c(f"随机选中样本 (ID={sid}, idx={idx})\n", Colors.GREEN)
            return idx
    
    def _resample_1d(self, x, target_len):
        """1D 信号重采样"""
        if len(x) == target_len:
            return x
        x_old = np.linspace(0, 1, len(x))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, x)
    
    def _align_and_scale(self, pred, gt, valid_mask):
        """对齐、缩放和计算指标"""
        pred = np.atleast_1d(pred).flatten().astype(np.float32)
        gt = np.atleast_1d(gt).flatten().astype(np.float32)
        valid_mask = np.atleast_1d(valid_mask).flatten().astype(np.float32)
        
        # 统一长度 - 使用 GT 长度作为标准
        target_len = len(gt)
        pred = self._resample_1d(pred, target_len)
        valid_mask = self._resample_1d(valid_mask, target_len)
        
        # 提取有效区间
        valid_idx = valid_mask > 0.5
        if valid_idx.sum() < 10:
            return pred, 0.0, 0.0, 0.0
        
        v_pred = pred[valid_idx]
        v_gt = gt[valid_idx]
        
        # 最优缩放
        try:
            a = np.cov(v_pred, v_gt)[0, 1] / (np.var(v_pred) + 1e-6)
            b = v_gt.mean() - a * v_pred.mean()
        except:
            a, b = 1.0, 0.0
        
        pred_scaled = a * v_pred + b
        
        # 相关系数
        try:
            corr, _ = pearsonr(pred_scaled, v_gt)
        except:
            corr = 0.0
        
        mae = np.abs(pred_scaled - v_gt).mean()
        rmse = np.sqrt(((pred_scaled - v_gt) ** 2).mean())
        
        # 返回完整缩放后的预测
        pred_full_scaled = a * pred + b
        
        return pred_full_scaled, corr, mae, rmse
    
    def run(self):
        """运行评估"""
        idx = self._get_sample_idx()
        if idx is None:
            return
        
        # 加载样本
        print_c("加载样本...", Colors.CYAN)
        batch = self.dataset[idx]
        sample_id = self.dataset.samples[idx]['id']
        
        image = batch['image'].unsqueeze(0).to(self.device)
        
        # 推理
        print_c("推理...", Colors.CYAN)
        with torch.no_grad():
            outputs = self.model(image, return_signals=True)
        
        # 提取数据
        print_c("处理数据...", Colors.CYAN)
        wave_seg = outputs['wave_segmentation_logits'][0].cpu().numpy()  # (12, H, W)
        baseline = outputs['lead_baselines'][0].cpu().numpy()  # (12, H, W)
        signals_pred = outputs['signals'][0].cpu().numpy()  # (12, W_signal)
        
        signals_gt = batch['gt_signals'].squeeze(0).numpy()  # (12, W_gt)
        valid_mask = batch['valid_mask'].squeeze(0).numpy()  # (12, W_gt)
        
        img = batch['image'][0].cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        print_c(f"数据形状: pred={signals_pred.shape}, gt={signals_gt.shape}\n", Colors.CYAN)
        
        # 创建可视化
        print_c("创建可视化...", Colors.CYAN)
        
        # 1. 导联分割
        fig1 = self._plot_segmentation(img, wave_seg, baseline)
        
        # 2. 波形对比
        fig2, metrics = self._plot_waveforms(signals_pred, signals_gt, valid_mask)
        
        # 3. 指标总结
        fig3 = self._plot_metrics(metrics)
        
        # 保存
        print_c("保存结果...", Colors.CYAN)
        fig1.savefig(self.output_dir / f"{sample_id}_01_segmentation.png", dpi=150, bbox_inches='tight')
        fig2.savefig(self.output_dir / f"{sample_id}_02_waveforms.png", dpi=150, bbox_inches='tight')
        fig3.savefig(self.output_dir / f"{sample_id}_03_metrics.png", dpi=150, bbox_inches='tight')
        print_c(f"✓ 保存到 {self.output_dir}\n", Colors.GREEN)
        
        # 打印报告
        self._print_report(metrics, sample_id)
        
        plt.show()
    
    def _plot_segmentation(self, img, wave_seg, baseline):
        """绘制分割结果"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 原始图像
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 波形分割
        H, W = wave_seg.shape[1], wave_seg.shape[2]
        wave_seg_viz = np.zeros((H, W, 3), dtype=np.uint8)
        argmax = wave_seg.argmax(axis=0)
        for i in range(12):
            mask = argmax == i
            wave_seg_viz[mask] = (np.array(self.COLORS[i][:3]) * 255).astype(np.uint8)
        
        axes[1].imshow(wave_seg_viz)
        axes[1].set_title('Lead Segmentation', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        patches = [mpatches.Patch(facecolor=self.COLORS[i][:3], label=self.LEAD_NAMES[i]) for i in range(12)]
        axes[1].legend(handles=patches, loc='upper right', fontsize=8, ncol=2)
        
        # 基线
        baseline_viz = np.zeros((baseline.shape[1], baseline.shape[2]))
        for i in range(12):
            baseline_viz += baseline[i] * (i + 1)
        
        im = axes[2].imshow(baseline_viz, cmap='jet', vmin=0, vmax=12)
        axes[2].set_title('Baseline Detection', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], label='Lead Index')
        
        plt.tight_layout()
        return fig
    
    def _plot_waveforms(self, pred, gt, valid_mask):
        """绘制波形对比"""
        fig, axes = plt.subplots(12, 1, figsize=(20, 18))
        fig.suptitle('ECG Waveform Comparison\nBlue=Predicted, Red=Ground Truth', fontsize=14, fontweight='bold')
        
        metrics = []
        
        for i, (ax, name) in enumerate(zip(axes, self.LEAD_NAMES)):
            pred_aligned, corr, mae, rmse = self._align_and_scale(pred[i], gt[i], valid_mask[i])
            
            metrics.append({'lead': name, 'corr': corr, 'mae': mae, 'rmse': rmse})
            
            x = np.arange(len(pred_aligned))
            ax.plot(x, pred_aligned, 'b-', label='Pred', linewidth=2, alpha=0.8)
            ax.plot(x, gt[i], 'r-', label='GT', linewidth=2, alpha=0.8)
            
            # 标记有效区间
            valid_idx = valid_mask[i] > 0.5
            if np.any(valid_idx):
                ax.fill_between(x, np.min([pred_aligned.min(), gt[i].min()]), 
                               np.max([pred_aligned.max(), gt[i].max()]),
                               where=valid_idx, alpha=0.1, color='green', label='Valid')
            
            color = 'green' if corr > 0.92 else 'orange' if corr > 0.85 else 'red'
            marker = '✓' if corr > 0.92 else '~' if corr > 0.85 else '✗'
            ax.set_title(f'{marker} {name}: Corr={corr:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}', 
                        color=color, fontweight='bold', fontsize=10)
            ax.set_ylabel('Amp', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, metrics
    
    def _plot_metrics(self, metrics):
        """绘制指标总结"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        leads = [m['lead'] for m in metrics]
        corrs = [m['corr'] for m in metrics]
        maes = [m['mae'] for m in metrics]
        rmses = [m['rmse'] for m in metrics]
        
        # 相关系数
        colors = ['green' if c > 0.92 else 'orange' if c > 0.85 else 'red' for c in corrs]
        axes[0].bar(leads, corrs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].axhline(0.92, color='green', linestyle='--', label='Excellent (0.92)', linewidth=2)
        axes[0].axhline(0.85, color='orange', linestyle='--', label='Good (0.85)', linewidth=2)
        axes[0].set_title('Signal Correlation', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Correlation', fontsize=10)
        axes[0].set_ylim([0, 1.05])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE
        axes[1].bar(leads, maes, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAE', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        # RMSE
        axes[2].bar(leads, rmses, color='salmon', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[2].set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('RMSE', fontsize=10)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def _print_report(self, metrics, sample_id):
        """打印报告"""
        print_c(f"\n{'='*70}", Colors.CYAN)
        print_c(f"评估报告 - {sample_id}", Colors.CYAN)
        print_c(f"{'='*70}", Colors.CYAN)
        
        print(f"\n{'导联':<8} {'相关系数':<15} {'MAE':<15} {'RMSE':<15}")
        print(f"{'-'*55}")
        
        corr_avg = mae_avg = rmse_avg = 0.0
        for m in metrics:
            print(f"{m['lead']:<8} {m['corr']:<15.6f} {m['mae']:<15.6f} {m['rmse']:<15.6f}")
            corr_avg += m['corr']
            mae_avg += m['mae']
            rmse_avg += m['rmse']
        
        n = len(metrics)
        corr_avg /= n
        mae_avg /= n
        rmse_avg /= n
        
        print(f"{'-'*55}")
        print(f"{'平均':<8} {corr_avg:<15.6f} {mae_avg:<15.6f} {rmse_avg:<15.6f}")
        print(f"{'='*70}\n")
        
        if corr_avg > 0.92:
            rating = "⭐⭐⭐⭐⭐ 优秀"
            color = Colors.GREEN
        elif corr_avg > 0.88:
            rating = "⭐⭐⭐⭐ 很好"
            color = Colors.GREEN
        elif corr_avg > 0.80:
            rating = "⭐⭐⭐ 良好"
            color = Colors.YELLOW
        else:
            rating = "⭐⭐ 中等"
            color = Colors.YELLOW
        
        print_c(f"总体评分: {rating}", color)
        print_c(f"平均相关系数: {corr_avg:.6f}\n", color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sim_root', type=str, required=True)
    parser.add_argument('--csv_root', type=str, required=True)
    parser.add_argument('--sample_id', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    
    args = parser.parse_args()
    
    evaluator = ECGEvaluator(
        checkpoint=args.checkpoint,
        sim_root=args.sim_root,
        csv_root=args.csv_root,
        sample_id=args.sample_id,
        output_dir=args.output_dir
    )
    evaluator.run()