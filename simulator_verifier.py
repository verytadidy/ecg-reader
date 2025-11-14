"""
ECG仿真数据验证工具 V25
用途：检查生成的仿真数据质量，确保标签与图像对齐
"""

import numpy as np
import pandas as pd
import cv2
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ============================
# 1. 数据完整性检查
# ============================
def check_data_integrity(output_dir):
    """检查所有样本是否完整"""
    print("=" * 70)
    print("数据完整性检查")
    print("=" * 70)
    
    all_samples = [d for d in os.listdir(output_dir) 
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('0')]
    
    print(f"找到 {len(all_samples)} 个样本目录\n")
    
    incomplete_samples = []
    required_files = ['_dirty.png', '_label_grid.png', '_label_wave.png',
                     '_label_other.png', '_label_baseline.png', '_metadata.json']
    
    for sample_id in all_samples:
        sample_dir = os.path.join(output_dir, sample_id)
        missing_files = []
        
        for suffix in required_files:
            filepath = os.path.join(sample_dir, f"{sample_id}{suffix}")
            if not os.path.exists(filepath):
                missing_files.append(suffix)
        
        if missing_files:
            incomplete_samples.append((sample_id, missing_files))
    
    if incomplete_samples:
        print(f"⚠️  发现 {len(incomplete_samples)} 个不完整的样本:")
        for sample_id, missing in incomplete_samples[:10]:
            print(f"  {sample_id}: 缺失 {missing}")
        if len(incomplete_samples) > 10:
            print(f"  ... 还有 {len(incomplete_samples) - 10} 个")
    else:
        print("✅ 所有样本文件完整")
    
    return len(incomplete_samples) == 0

# ============================
# 2. 标签对齐检查
# ============================
def check_label_alignment(output_dir, num_samples=10):
    """检查标签与图像是否对齐"""
    print("\n" + "=" * 70)
    print("标签对齐检查")
    print("=" * 70)
    
    all_samples = [d for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('0')]
    
    if len(all_samples) == 0:
        print("⚠️  未找到样本")
        return False
    
    # 随机选择样本
    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    alignment_issues = []
    
    for sample_id in selected:
        sample_dir = os.path.join(output_dir, sample_id)
        
        dirty_img = cv2.imread(os.path.join(sample_dir, f"{sample_id}_dirty.png"))
        wave_mask = cv2.imread(os.path.join(sample_dir, f"{sample_id}_label_wave.png"), 0)
        
        if dirty_img is None or wave_mask is None:
            continue
        
        # 检查尺寸是否匹配
        if dirty_img.shape[:2] != wave_mask.shape[:2]:
            alignment_issues.append((sample_id, "尺寸不匹配"))
            continue
        
        # 检查波形掩码是否在有效范围内
        wave_pixels = np.sum(wave_mask > 0)
        total_pixels = wave_mask.shape[0] * wave_mask.shape[1]
        wave_ratio = wave_pixels / total_pixels
        
        if wave_ratio < 0.005 or wave_ratio > 0.3:
            alignment_issues.append((sample_id, f"波形占比异常: {wave_ratio:.3f}"))
    
    if alignment_issues:
        print(f"⚠️  发现 {len(alignment_issues)} 个对齐问题:")
        for sample_id, issue in alignment_issues:
            print(f"  {sample_id}: {issue}")
    else:
        print(f"✅ 检查的 {len(selected)} 个样本标签对齐正常")
    
    return len(alignment_issues) == 0

# ============================
# 3. 物理参数验证
# ============================
def validate_physical_params(output_dir, num_samples=50):
    """验证物理参数的分布"""
    print("\n" + "=" * 70)
    print("物理参数分布验证")
    print("=" * 70)
    
    all_samples = [d for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('0')]
    
    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    paper_speeds = []
    gains = []
    px_per_mms = []
    
    for sample_id in selected:
        metadata_path = os.path.join(output_dir, sample_id, f"{sample_id}_metadata.json")
        if not os.path.exists(metadata_path):
            continue
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        params = metadata['physical_params']
        paper_speeds.append(params['paper_speed_mm_s'])
        gains.append(params['gain_mm_mv'])
        px_per_mms.append(params['px_per_mm'])
    
    print(f"检查了 {len(paper_speeds)} 个样本的物理参数\n")
    
    # 纸速分布
    print("纸速分布 (mm/s):")
    paper_speed_counts = {}
    for speed in paper_speeds:
        paper_speed_counts[speed] = paper_speed_counts.get(speed, 0) + 1
    for speed, count in sorted(paper_speed_counts.items()):
        print(f"  {speed:5.1f} mm/s: {count:3d} ({count/len(paper_speeds)*100:5.1f}%)")
    
    # 增益分布
    print("\n增益分布 (mm/mV):")
    gain_counts = {}
    for gain in gains:
        gain_counts[gain] = gain_counts.get(gain, 0) + 1
    for gain, count in sorted(gain_counts.items()):
        print(f"  {gain:5.1f} mm/mV: {count:3d} ({count/len(gains)*100:5.1f}%)")
    
    # 分辨率统计
    print(f"\n分辨率 (px/mm):")
    print(f"  最小: {min(px_per_mms):.2f}")
    print(f"  最大: {max(px_per_mms):.2f}")
    print(f"  平均: {np.mean(px_per_mms):.2f}")
    
    # 验证是否符合预期
    valid = True
    if set(paper_speeds) != {25.0, 50.0}:
        print("⚠️  纸速不在预期范围 [25.0, 50.0]")
        valid = False
    if set(gains) != {5.0, 10.0, 20.0}:
        print("⚠️  增益不在预期范围 [5.0, 10.0, 20.0]")
        valid = False
    if min(px_per_mms) < 18.0 or max(px_per_mms) > 22.0:
        print("⚠️  分辨率不在预期范围 [18.0, 22.0]")
        valid = False
    
    if valid:
        print("\n✅ 物理参数分布正常")
    
    return valid

# ============================
# 4. 布局分布检查
# ============================
def check_layout_distribution(output_dir):
    """检查布局类型的分布"""
    print("\n" + "=" * 70)
    print("布局分布检查")
    print("=" * 70)
    
    all_samples = [d for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('0')]
    
    layout_counts = {}
    degradation_counts = {}
    
    for sample_id in all_samples:
        metadata_path = os.path.join(output_dir, sample_id, f"{sample_id}_metadata.json")
        if not os.path.exists(metadata_path):
            continue
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        layout = metadata['layout_type']
        degradation = metadata['degradation_type']
        
        layout_counts[layout] = layout_counts.get(layout, 0) + 1
        degradation_counts[degradation] = degradation_counts.get(degradation, 0) + 1
    
    total = len(all_samples)
    
    print(f"检查了 {total} 个样本\n")
    
    print("布局分布:")
    for layout, count in sorted(layout_counts.items()):
        print(f"  {layout:15s}: {count:5d} ({count/total*100:5.1f}%)")
    
    print("\n退化分布:")
    for degradation, count in sorted(degradation_counts.items()):
        print(f"  {degradation:15s}: {count:5d} ({count/total*100:5.1f}%)")
    
    return True

# ============================
# 5. 可视化检查
# ============================
def visualize_samples(output_dir, num_samples=4, save_path=None):
    """可视化样本和标签"""
    print("\n" + "=" * 70)
    print("生成可视化")
    print("=" * 70)
    
    all_samples = [d for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('0')]
    
    if len(all_samples) == 0:
        print("⚠️  未找到样本")
        return
    
    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_id in enumerate(selected):
        sample_dir = os.path.join(output_dir, sample_id)
        
        # 读取图像
        dirty_img = cv2.imread(os.path.join(sample_dir, f"{sample_id}_dirty.png"))
        grid_img = cv2.imread(os.path.join(sample_dir, f"{sample_id}_label_grid.png"))
        wave_mask = cv2.imread(os.path.join(sample_dir, f"{sample_id}_label_wave.png"), 0)
        other_mask = cv2.imread(os.path.join(sample_dir, f"{sample_id}_label_other.png"), 0)
        baseline_mask = cv2.imread(os.path.join(sample_dir, f"{sample_id}_label_baseline.png"), 0)
        
        # 读取元数据
        metadata_path = os.path.join(sample_dir, f"{sample_id}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 转换颜色
        dirty_img = cv2.cvtColor(dirty_img, cv2.COLOR_BGR2RGB)
        grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
        
        # 显示
        axes[idx, 0].imshow(dirty_img)
        axes[idx, 0].set_title(f"Dirty Image\n{metadata['layout_type']}\n{metadata['degradation_type']}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(grid_img)
        axes[idx, 1].set_title("Grid Label")
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(wave_mask, cmap='hot')
        axes[idx, 2].set_title("Wave Mask")
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(other_mask, cmap='hot')
        axes[idx, 3].set_title("Other Mask")
        axes[idx, 3].axis('off')
        
        axes[idx, 4].imshow(baseline_mask, cmap='hot')
        axes[idx, 4].set_title("Baseline Mask")
        axes[idx, 4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化已保存至: {save_path}")
    else:
        plt.show()

# ============================
# 6. 导联时序验证
# ============================
def validate_lead_timing(output_dir, original_csv_dir, num_samples=10):
    """验证导联时序是否正确（Lead II=10s, 其他=2.5s）"""
    print("\n" + "=" * 70)
    print("导联时序验证")
    print("=" * 70)
    
    all_samples = [d for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('0')]
    
    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    timing_issues = []
    
    for sample_id in selected:
        # 解析 ecg_id
        parts = sample_id.split('_')
        ecg_id = parts[0]
        
        # 读取原始CSV
        csv_path = os.path.join(original_csv_dir, ecg_id, f"{ecg_id}.csv")
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path)
        fs = len(df) / 10.0  # 采样率
        
        # 读取元数据
        metadata_path = os.path.join(output_dir, sample_id, f"{sample_id}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 检查配置
        expected_lead_II = metadata['physical_params']['lead_II_duration_s']
        expected_others = metadata['physical_params']['other_leads_duration_s']
        
        if expected_lead_II != 10.0:
            timing_issues.append((sample_id, f"Lead II 时长错误: {expected_lead_II}"))
        if expected_others != 2.5:
            timing_issues.append((sample_id, f"其他导联时长错误: {expected_others}"))
    
    if timing_issues:
        print(f"⚠️  发现 {len(timing_issues)} 个时序问题:")
        for sample_id, issue in timing_issues:
            print(f"  {sample_id}: {issue}")
    else:
        print(f"✅ 检查的 {len(selected)} 个样本时序配置正确")
    
    return len(timing_issues) == 0

# ============================
# 7. 主验证流程
# ============================
def run_full_validation(output_dir, original_csv_dir=None, save_viz=True):
    """运行完整验证流程"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "ECG 仿真数据验证报告" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = {}
    
    # 1. 数据完整性
    results['integrity'] = check_data_integrity(output_dir)
    
    # 2. 标签对齐
    results['alignment'] = check_label_alignment(output_dir, num_samples=20)
    
    # 3. 物理参数
    results['physics'] = validate_physical_params(output_dir, num_samples=100)
    
    # 4. 布局分布
    results['layout'] = check_layout_distribution(output_dir)
    
    # 5. 时序验证（如果提供了原始CSV目录）
    if original_csv_dir:
        results['timing'] = validate_lead_timing(output_dir, original_csv_dir, num_samples=20)
    
    # 6. 可视化
    if save_viz:
        viz_path = os.path.join(output_dir, "validation_visualization.png")
        visualize_samples(output_dir, num_samples=4, save_path=viz_path)
    
    # 汇总报告
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():20s}: {status}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有验证通过！数据质量良好")
    else:
        print("⚠️  部分验证失败，请检查上述问题")
    print("=" * 70)
    
    return all_passed

# ============================
# 使用示例
# ============================
if __name__ == "__main__":
    # 配置路径
    OUTPUT_DIR = "/Volumes/movie/work/physionet-ecg-image-digitization-simulations-V25"
    ORIGINAL_CSV_DIR = "/Volumes/movie/work/physionet-ecg-image-digitization/train"
    
    # 运行验证
    run_full_validation(
        output_dir=OUTPUT_DIR,
        original_csv_dir=ORIGINAL_CSV_DIR,
        save_viz=True
    )