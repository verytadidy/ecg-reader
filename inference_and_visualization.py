#!/usr/bin/env python3
"""
ECG重建模型推理和可视化脚本

用途：只需要PNG图像输入，输出重建的波形并可视化对比
适用场景：实际测试集（无CSV真值），人工评估几个样本
"""

import os
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# 导入模型
from ecg_model import ECGReconstructionModel


class ECGInference:
    """ECG推理器"""
    
    def __init__(self, checkpoint_path, device='auto', target_fs=500):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 'auto', 'cuda', 'mps', 'cpu'
            target_fs: 目标采样率
        """
        self.target_fs = target_fs
        self.signal_length = target_fs * 10  # 10秒
        
        # 设置设备
        self.device = self._setup_device(device)
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        
        # 导联名称
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        print(f"✓ 推理器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  采样率: {self.target_fs} Hz")
        print(f"  信号长度: {self.signal_length} 点")
    
    def _setup_device(self, device):
        """设置设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        print(f"\n加载模型: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 创建模型（MPS上禁用STN）
        enable_stn = not (self.device.type == 'mps')
        model = ECGReconstructionModel(
            num_leads=12,
            signal_length=self.signal_length,
            pretrained=False,
            enable_stn=enable_stn
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 打印检查点信息
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")
        
        return model
    
    def preprocess_image(self, image_path, target_size=(512, 672)):
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            target_size: (H, W)
        
        Returns:
            tensor: (1, 3, H, W)
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (target_size[1], target_size[0]))
        
        # 转换为tensor并归一化
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        img_tensor = img_tensor / 255.0
        
        # Normalize（使用ImageNet统计）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        return img_tensor
    
    @torch.no_grad()
    def predict(self, image_path, return_all_outputs=False):
        """
        预测单张图像
        
        Args:
            image_path: 图像路径
            return_all_outputs: 是否返回所有中间输出
        
        Returns:
            dict: {
                'signals': (12, T) numpy array - 重建的12导联信号
                'lead_names': list - 导联名称
                'fs': int - 采样率
                'image': numpy array - 原始图像
                (可选) 'seg_mask', 'grid_mask', 'baseline_heatmap' 等中间输出
            }
        """
        # 预处理
        img_tensor = self.preprocess_image(image_path).to(self.device)
        
        # 推理
        outputs = self.model(img_tensor)
        
        # 提取重建信号（模型返回的键名是 'signal'）
        signals = outputs['signal'][0].cpu().numpy()  # (12, T)
        
        # 准备返回结果
        result = {
            'signals': signals,
            'lead_names': self.lead_names,
            'fs': self.target_fs,
            'image': cv2.imread(str(image_path))
        }
        
        # 返回中间输出（用于可视化）
        if return_all_outputs:
            result['seg_mask'] = torch.sigmoid(outputs['wave_seg'][0]).cpu().numpy()  # (12, H, W)
            result['grid_mask'] = torch.sigmoid(outputs['grid_mask'][0, 0]).cpu().numpy()  # (H, W)
            result['baseline_heatmap'] = outputs['baseline_heatmaps'][0].cpu().numpy()  # (12, H, W)
            
            if 'theta' in outputs:
                result['theta'] = outputs['theta'][0].cpu().numpy()  # (2, 3)
        
        return result
    
    def predict_batch(self, image_paths):
        """批量预测"""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"⚠️  处理 {image_path} 失败: {e}")
                results.append(None)
        return results


class ECGVisualizer:
    """ECG可视化器"""
    
    @staticmethod
    def plot_comparison_with_gt(pred_result, gt_csv_path, save_path=None, show=True):
        """
        对比预测和真值（训练集用）
        
        Args:
            pred_result: predict()返回的结果
            gt_csv_path: 真值CSV路径
            save_path: 保存路径
            show: 是否显示
        """
        # 读取真值
        gt_df = pd.read_csv(gt_csv_path)
        lead_names = pred_result['lead_names']
        
        # 创建图表
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for i, lead_name in enumerate(lead_names):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            
            # 真值
            gt_signal = gt_df[lead_name].values
            
            # 预测值（可能需要重采样）
            pred_signal = pred_result['signals'][i]
            
            # 如果长度不一致，重采样预测信号
            if len(pred_signal) != len(gt_signal):
                pred_signal_resampled = np.interp(
                    np.linspace(0, len(pred_signal), len(gt_signal)),
                    np.arange(len(pred_signal)),
                    pred_signal
                )
            else:
                pred_signal_resampled = pred_signal
            
            # 计算MSE和相关系数
            mse = np.mean((pred_signal_resampled - gt_signal) ** 2)
            corr = np.corrcoef(pred_signal_resampled, gt_signal)[0, 1]
            
            # 绘图
            time_axis = np.arange(len(gt_signal)) / pred_result['fs']
            ax.plot(time_axis, gt_signal, 'g-', linewidth=1.5, alpha=0.7, label='Ground Truth')
            ax.plot(time_axis, pred_signal_resampled, 'r--', linewidth=1.5, alpha=0.7, label='Prediction')
            
            ax.set_title(f'{lead_name} | MSE: {mse:.4f} | Corr: {corr:.3f}', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude (mV)', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('ECG Reconstruction: Prediction vs Ground Truth', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存对比图: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_prediction_only(pred_result, save_path=None, show=True):
        """
        只绘制预测结果（测试集用，无真值）
        
        Args:
            pred_result: predict()返回的结果
            save_path: 保存路径
            show: 是否显示
        """
        lead_names = pred_result['lead_names']
        signals = pred_result['signals']
        fs = pred_result['fs']
        
        # 创建图表
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for i, lead_name in enumerate(lead_names):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            
            signal = signals[i]
            time_axis = np.arange(len(signal)) / fs
            
            # 计算简单统计
            signal_min = signal.min()
            signal_max = signal.max()
            signal_mean = signal.mean()
            signal_std = signal.std()
            
            # 绘图
            ax.plot(time_axis, signal, 'b-', linewidth=1.5)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
            
            ax.set_title(f'{lead_name} | Mean: {signal_mean:.3f} | Std: {signal_std:.3f}', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude (mV)', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 标注极值
            ax.text(0.02, 0.98, f'Min: {signal_min:.3f}\nMax: {signal_max:.3f}',
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('ECG Reconstruction: Predicted Signals', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存预测图: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_intermediate_outputs(pred_result, save_path=None, show=True):
        """
        可视化中间输出（分割掩码、网格、基线热图）
        
        Args:
            pred_result: predict(return_all_outputs=True)返回的结果
            save_path: 保存路径
            show: 是否显示
        """
        if 'seg_mask' not in pred_result:
            raise ValueError("需要 return_all_outputs=True 才能可视化中间输出")
        
        # 准备数据
        image = pred_result['image']
        seg_mask = pred_result['seg_mask']  # (12, H, W)
        grid_mask = pred_result['grid_mask']  # (H, W)
        baseline_heatmap = pred_result['baseline_heatmap']  # (12, H, W)
        
        # 创建图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 原始图像
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # 2. 网格掩码
        ax2 = plt.subplot(3, 3, 2)
        ax2.imshow(grid_mask, cmap='hot')
        ax2.set_title('Grid Mask')
        ax2.axis('off')
        
        # 3. 所有导联分割掩码叠加
        ax3 = plt.subplot(3, 3, 3)
        seg_combined = seg_mask.max(axis=0)  # 取所有导联的最大值
        ax3.imshow(seg_combined, cmap='hot')
        ax3.set_title('Combined Segmentation Mask')
        ax3.axis('off')
        
        # 4-6. 前3个导联的分割掩码
        for i in range(3):
            ax = plt.subplot(3, 3, 4 + i)
            ax.imshow(seg_mask[i], cmap='hot')
            ax.set_title(f'Seg Mask - {pred_result["lead_names"][i]}')
            ax.axis('off')
        
        # 7-9. 前3个导联的基线热图
        for i in range(3):
            ax = plt.subplot(3, 3, 7 + i)
            ax.imshow(baseline_heatmap[i], cmap='jet')
            ax.set_title(f'Baseline Heatmap - {pred_result["lead_names"][i]}')
            ax.axis('off')
        
        plt.suptitle('Intermediate Outputs Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存中间输出图: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_full_pipeline(image_path, pred_result, gt_csv_path=None, save_dir=None):
        """
        完整可视化流程（一张图片包含所有信息）
        
        Args:
            image_path: 输入图像路径
            pred_result: predict(return_all_outputs=True)返回的结果
            gt_csv_path: 真值CSV路径（可选）
            save_dir: 保存目录
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 绘制中间输出
        if 'seg_mask' in pred_result:
            save_path_inter = save_dir / f"{Path(image_path).stem}_intermediate.png" if save_dir else None
            ECGVisualizer.plot_intermediate_outputs(pred_result, save_path=save_path_inter, show=False)
        
        # 2. 绘制预测波形
        if gt_csv_path:
            save_path_comp = save_dir / f"{Path(image_path).stem}_comparison.png" if save_dir else None
            ECGVisualizer.plot_comparison_with_gt(pred_result, gt_csv_path, save_path=save_path_comp, show=False)
        else:
            save_path_pred = save_dir / f"{Path(image_path).stem}_prediction.png" if save_dir else None
            ECGVisualizer.plot_prediction_only(pred_result, save_path=save_path_pred, show=False)


def main():
    parser = argparse.ArgumentParser(description='ECG重建模型推理和可视化')
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径（单张）或目录（批量）')
    
    # 可选参数
    parser.add_argument('--gt_csv', type=str, default=None,
                       help='真值CSV路径（可选，用于对比）')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='推理设备')
    parser.add_argument('--target_fs', type=int, default=500,
                       help='目标采样率')
    parser.add_argument('--save_csv', action='store_true',
                       help='保存预测结果为CSV')
    parser.add_argument('--show_intermediate', action='store_true',
                       help='可视化中间输出')
    parser.add_argument('--no_display', action='store_true',
                       help='不显示图表（只保存）')
    
    args = parser.parse_args()
    
    # 创建推理器
    print("="*70)
    print("初始化推理器")
    print("="*70)
    inferencer = ECGInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        target_fs=args.target_fs
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 推理
    print("\n" + "="*70)
    print("开始推理")
    print("="*70)
    
    image_path = Path(args.image)
    
    if image_path.is_file():
        # 单张图像
        print(f"处理图像: {image_path}")
        
        result = inferencer.predict(
            image_path,
            return_all_outputs=args.show_intermediate
        )
        
        # 保存CSV
        if args.save_csv:
            csv_path = output_dir / f"{image_path.stem}_prediction.csv"
            df = pd.DataFrame(result['signals'].T, columns=result['lead_names'])
            df.to_csv(csv_path, index=False)
            print(f"✓ 保存预测CSV: {csv_path}")
        
        # 可视化
        print("\n可视化结果...")
        ECGVisualizer.plot_full_pipeline(
            image_path=str(image_path),
            pred_result=result,
            gt_csv_path=args.gt_csv,
            save_dir=output_dir
        )
        
        # 显示最终波形图
        if not args.no_display:
            if args.gt_csv:
                ECGVisualizer.plot_comparison_with_gt(result, args.gt_csv)
            else:
                ECGVisualizer.plot_prediction_only(result)
    
    elif image_path.is_dir():
        # 批量处理目录
        image_files = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg'))
        print(f"找到 {len(image_files)} 张图像")
        
        for img_file in image_files[:5]:  # 只处理前5张
            print(f"\n处理: {img_file.name}")
            
            try:
                result = inferencer.predict(img_file, return_all_outputs=False)
                
                # 保存CSV
                if args.save_csv:
                    csv_path = output_dir / f"{img_file.stem}_prediction.csv"
                    df = pd.DataFrame(result['signals'].T, columns=result['lead_names'])
                    df.to_csv(csv_path, index=False)
                
                # 可视化
                save_path = output_dir / f"{img_file.stem}_prediction.png"
                ECGVisualizer.plot_prediction_only(result, save_path=save_path, show=False)
                
            except Exception as e:
                print(f"  ⚠️  处理失败: {e}")
    
    else:
        raise ValueError(f"无效路径: {image_path}")
    
    print("\n" + "="*70)
    print("推理完成！")
    print(f"结果保存至: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    # 使用示例：
    #
    # 1. 单张图像（无真值）：
    #   python inference_and_visualization.py \
    #       --checkpoint ./experiments/run_xxx/checkpoints/best.pth \
    #       --image ./test/262.png \
    #       --output_dir ./inference_results
    #
    # 2. 单张图像（有真值对比）：
    #   python inference_and_visualization.py \
    #       --checkpoint ./experiments/run_xxx/checkpoints/best.pth \
    #       --image ./train/262/262.png \
    #       --gt_csv ./train/262/262.csv \
    #       --output_dir ./inference_results \
    #       --show_intermediate
    #
    # 3. 批量处理（保存CSV）：
    #   python inference_and_visualization.py \
    #       --checkpoint ./experiments/run_xxx/checkpoints/best.pth \
    #       --image ./test \
    #       --output_dir ./inference_results \
    #       --save_csv \
    #       --no_display
    
    main()