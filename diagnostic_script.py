"""
ECG V48 数据流诊断脚本
检查:
1. Dataset 输出的信号和有效掩码
2. 模型输出的信号长度
3. 重采样和对齐逻辑
4. 基线输出
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 导入相关模块
from ecg_dataset_v48_fixed import ECGV48FixedDataset

try:
    from ecg_model_v48_mps_optimized import ProgressiveLeadLocalizationModelV48Lite
except:
    from ecg_model_v48_fixed import ProgressiveLeadLocalizationModelV48LiteFixed as ProgressiveLeadLocalizationModelV48Lite

# ============ 诊断函数 ============

def diagnose_dataset(dataset, idx=0):
    """诊断 Dataset 输出"""
    print("\n" + "="*80)
    print("【1】Dataset 诊断")
    print("="*80)
    
    batch = dataset[idx]
    sample_id = dataset.samples[idx]['id']
    
    print(f"\n样本 ID: {sample_id}")
    print(f"\n输出数据形状:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<25s}: {v.shape}")
    
    # 检查信号
    signals_gt = batch['gt_signals']  # (12, W_gt)
    valid_mask = batch['valid_mask']  # (12, W_gt)
    
    print(f"\n【信号分析】")
    print(f"  GT 信号形状: {signals_gt.shape}")
    print(f"  有效掩码形状: {valid_mask.shape}")
    print(f"  GT 信号范围: [{signals_gt.min():.4f}, {signals_gt.max():.4f}]")
    print(f"  GT 信号均值: {signals_gt.mean():.4f}, 方差: {signals_gt.std():.4f}")
    
    # 逐导联检查
    print(f"\n【逐导联检查】")
    print(f"{'导联':<8} {'信号数':<12} {'有效数':<12} {'信号范围':<25}")
    print(f"{'-'*60}")
    
    for i in range(min(3, 12)):  # 只检查前 3 个
        n_valid = (valid_mask[i] > 0.5).sum().item()
        sig_range = f"[{signals_gt[i].min():.3f}, {signals_gt[i].max():.3f}]"
        print(f"{i:<8} {signals_gt[i].shape[0]:<12} {n_valid:<12} {sig_range:<25}")
    
    return batch, sample_id


def diagnose_model(model, batch, device):
    """诊断模型输出"""
    print("\n" + "="*80)
    print("【2】模型输出诊断")
    print("="*80)
    
    image = batch['image'].unsqueeze(0).to(device)
    
    print(f"\n输入形状: {image.shape}")
    
    with torch.no_grad():
        outputs = model(image, return_signals=True)
    
    print(f"\n输出数据形状:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<30s}: {v.shape}")
    
    # 检查信号输出 (转移到 CPU)
    signals_pred = outputs['signals'][0].cpu().numpy()  # (12, W_pred)
    baseline_pred = outputs['lead_baselines'][0].cpu().numpy()  # (12, H, W)
    
    print(f"\n【预测信号分析】")
    print(f"  预测信号形状: {signals_pred.shape}")
    print(f"  预测信号范围: [{signals_pred.min():.4f}, {signals_pred.max():.4f}]")
    print(f"  预测信号均值: {signals_pred.mean():.4f}, 方差: {signals_pred.std():.4f}")
    
    print(f"\n【基线预测分析】")
    print(f"  基线形状: {baseline_pred.shape}")
    print(f"  基线范围: [{baseline_pred.min():.4f}, {baseline_pred.max():.4f}]")
    print(f"  基线均值: {baseline_pred.mean():.4f}, 方差: {baseline_pred.std():.4f}")
    
    # 检查是否全零
    if baseline_pred.max() < 1e-6:
        print(f"  ⚠️ 警告: 基线全为零!")
    
    return signals_pred, baseline_pred


def diagnose_resampling(signals_pred, signals_gt, valid_mask):
    """诊断重采样逻辑"""
    print("\n" + "="*80)
    print("【3】重采样和对齐诊断")
    print("="*80)
    
    print(f"\n【长度信息】")
    print(f"  预测信号长度: {signals_pred.shape[1]}")
    print(f"  GT 信号长度: {signals_gt.shape[1]}")
    print(f"  有效掩码长度: {valid_mask.shape[1]}")
    
    # 第一条导联详细分析
    pred = signals_pred[0]  # 一维
    gt = signals_gt[0] if isinstance(signals_gt, np.ndarray) else signals_gt[0].numpy()
    valid = valid_mask[0] if isinstance(valid_mask, np.ndarray) else valid_mask[0].numpy()
    
    print(f"\n【第一条导联详细分析】")
    print(f"  预测长度: {len(pred)}")
    print(f"  GT 长度: {len(gt)}")
    print(f"  有效掩码长度: {len(valid)}")
    
    # 重采样
    target_len = len(gt)
    if len(pred) != target_len:
        x_old = np.linspace(0, 1, len(pred))
        x_new = np.linspace(0, 1, target_len)
        pred_resampled = np.interp(x_new, x_old, pred)
        print(f"\n  重采样: {len(pred)} -> {target_len}")
        print(f"  重采样前范围: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"  重采样后范围: [{pred_resampled.min():.4f}, {pred_resampled.max():.4f}]")
    else:
        pred_resampled = pred
    
    # 有效掩码重采样
    if len(valid) != target_len:
        x_old = np.linspace(0, 1, len(valid))
        x_new = np.linspace(0, 1, target_len)
        valid_resampled = np.interp(x_new, x_old, valid)
        print(f"\n  有效掩码重采样: {len(valid)} -> {target_len}")
    else:
        valid_resampled = valid
    
    # 计算对齐指标
    valid_idx = valid_resampled > 0.5
    n_valid = valid_idx.sum()
    
    print(f"\n【对齐后统计】")
    print(f"  有效样本数: {n_valid} / {target_len}")
    print(f"  有效比例: {n_valid / target_len * 100:.1f}%")
    
    if n_valid > 10:
        # 最优缩放
        v_pred = pred_resampled[valid_idx]
        v_gt = gt[valid_idx]
        
        try:
            a = np.cov(v_pred, v_gt)[0, 1] / (np.var(v_pred) + 1e-6)
            b = v_gt.mean() - a * v_pred.mean()
        except:
            a, b = 1.0, 0.0
        
        pred_scaled = a * v_pred + b
        
        # 相关系数
        try:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(pred_scaled, v_gt)
        except:
            corr = 0.0
        
        mae = np.abs(pred_scaled - v_gt).mean()
        rmse = np.sqrt(((pred_scaled - v_gt) ** 2).mean())
        
        print(f"\n【对齐质量指标】")
        print(f"  缩放系数 a: {a:.6f}")
        print(f"  缩放偏差 b: {b:.6f}")
        print(f"  相关系数: {corr:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        
        return corr, mae, rmse
    else:
        print(f"  ⚠️ 警告: 有效数据不足!")
        return 0.0, float('inf'), float('inf')


def diagnose_visualization():
    """生成诊断图"""
    print("\n" + "="*80)
    print("【4】生成诊断图")
    print("="*80)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 模拟数据用于演示
    x = np.linspace(0, 1, 512)
    gt_signal = np.sin(2 * np.pi * 5 * x) + 0.1 * np.random.randn(512)
    pred_signal_256 = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 256)) + 0.2 * np.random.randn(256)
    valid_mask = np.concatenate([np.ones(350), np.zeros(162)])
    
    # 重采样
    x_old = np.linspace(0, 1, 256)
    x_new = np.linspace(0, 1, 512)
    pred_resampled = np.interp(x_new, x_old, pred_signal_256)
    
    # 缩放
    valid_idx = valid_mask > 0.5
    v_pred = pred_resampled[valid_idx]
    v_gt = gt_signal[valid_idx]
    a = np.cov(v_pred, v_gt)[0, 1] / (np.var(v_pred) + 1e-6)
    b = v_gt.mean() - a * v_pred.mean()
    pred_scaled = a * pred_resampled + b
    
    # 绘图 1: 重采样过程
    axes[0].plot(np.linspace(0, 1, 256), pred_signal_256, 'b.-', label='Original (256)', alpha=0.7)
    axes[0].plot(x, pred_resampled, 'g-', label='Resampled (512)', linewidth=2)
    axes[0].plot(x, gt_signal, 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
    axes[0].fill_between(x, gt_signal.min(), gt_signal.max(), where=valid_idx, alpha=0.1, color='green', label='Valid region')
    axes[0].set_title('Resampling Process', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘图 2: 缩放对齐
    axes[1].plot(x, pred_scaled, 'b-', label='Pred (scaled)', linewidth=2, alpha=0.8)
    axes[1].plot(x, gt_signal, 'r-', label='Ground Truth', linewidth=2, alpha=0.8)
    axes[1].fill_between(x, gt_signal.min(), gt_signal.max(), where=valid_idx, alpha=0.1, color='green')
    
    from scipy.stats import pearsonr
    corr, _ = pearsonr(pred_scaled[valid_idx], gt_signal[valid_idx])
    
    axes[1].set_title(f'After Scaling (Corr={corr:.4f})', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./diagnostic_resampling.png', dpi=150, bbox_inches='tight')
    print(f"✓ 诊断图已保存: ./diagnostic_resampling.png")
    
    return fig


# ============ 主函数 ============

def main(checkpoint_path, sim_root, csv_root):
    print("\n" + "="*80)
    print("ECG V48 完整数据流诊断")
    print("="*80)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 1. Dataset 诊断
    dataset = ECGV48FixedDataset(
        sim_root_dir=sim_root,
        csv_root_dir=csv_root,
        split='val',
        target_size=(512, 2048),
        augment=False,
        cache_images=False
    )
    
    # 随机选择一个样本
    idx = np.random.randint(0, len(dataset))
    batch, sample_id = diagnose_dataset(dataset, idx)
    
    # 2. 模型诊断
    model = ProgressiveLeadLocalizationModelV48Lite(
        num_leads=12, roi_height=32, pretrained=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    signals_pred, baseline_pred = diagnose_model(model, batch, device)
    
    # 3. 重采样诊断
    signals_gt = batch['gt_signals'].numpy()
    valid_mask = batch['valid_mask'].numpy()
    
    corr, mae, rmse = diagnose_resampling(signals_pred, signals_gt, valid_mask)
    
    # 4. 生成诊断图
    diagnose_visualization()
    
    print("\n" + "="*80)
    print("诊断完成!")
    print("="*80)
    print(f"\n关键发现:")
    print(f"  ✓ 样本 ID: {sample_id}")
    print(f"  ✓ 第一导联相关系数: {corr:.6f}")
    print(f"  ✓ 第一导联 MAE: {mae:.6f}")
    print(f"  ✓ 第一导联 RMSE: {rmse:.6f}")
    
    if baseline_pred.max() < 1e-6:
        print(f"  ⚠️ 基线预测全为零 - 这可能是一个问题!")
    
    if corr < 0.5:
        print(f"  ⚠️ 相关系数很低 - 检查对齐逻辑!")
    
    print(f"\n输出文件:")
    print(f"  ✓ ./diagnostic_resampling.png - 重采样过程可视化")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sim_root', type=str, required=True)
    parser.add_argument('--csv_root', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args.checkpoint, args.sim_root, args.csv_root)