"""
检查CSV文件中的NaN分布

用于验证：
1. 哪些导联有NaN
2. NaN出现在什么位置
3. 是否符合"短导联只有部分时间段"的预期
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def analyze_csv(csv_path: Path):
    """分析单个CSV文件"""
    df = pd.read_csv(csv_path)
    
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    result = {
        'ecg_id': csv_path.parent.name,
        'total_rows': len(df),
        'leads': {}
    }
    
    for lead in leads:
        if lead not in df.columns:
            result['leads'][lead] = {
                'exists': False,
                'nan_count': 0,
                'nan_ratio': 0.0,
                'valid_range': None
            }
            continue
        
        sig = df[lead].values
        nan_mask = np.isnan(sig)
        nan_count = np.sum(nan_mask)
        valid_mask = ~nan_mask
        
        # 找到有效数据的范围
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            valid_start = valid_indices[0]
            valid_end = valid_indices[-1]
            valid_range = (valid_start, valid_end)
        else:
            valid_range = None
        
        result['leads'][lead] = {
            'exists': True,
            'nan_count': int(nan_count),
            'nan_ratio': float(nan_count / len(sig)),
            'valid_count': int(np.sum(valid_mask)),
            'valid_range': valid_range
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='检查CSV中的NaN分布')
    parser.add_argument('--csv_root', type=str, required=True, help='CSV根目录')
    parser.add_argument('--max_samples', type=int, default=10, help='检查的样本数')
    
    args = parser.parse_args()
    
    csv_root = Path(args.csv_root)
    
    print("="*70)
    print("CSV数据NaN分析")
    print("="*70)
    print()
    
    # 收集所有CSV
    csv_files = []
    for ecg_dir in csv_root.iterdir():
        if ecg_dir.is_dir():
            csv_file = ecg_dir / f"{ecg_dir.name}.csv"
            if csv_file.exists():
                csv_files.append(csv_file)
                if len(csv_files) >= args.max_samples:
                    break
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    print()
    
    # 统计信息
    lead_nan_stats = defaultdict(list)
    
    for csv_path in csv_files:
        result = analyze_csv(csv_path)
        
        print(f"样本: {result['ecg_id']} (总行数: {result['total_rows']})")
        print("  导联信息:")
        
        for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            info = result['leads'][lead]
            
            if not info['exists']:
                print(f"    {lead:3s}: ✗ 不存在")
                continue
            
            nan_ratio = info['nan_ratio']
            lead_nan_stats[lead].append(nan_ratio)
            
            if nan_ratio > 0:
                valid_range = info['valid_range']
                if valid_range:
                    start, end = valid_range
                    print(f"    {lead:3s}: ⚠️  {info['nan_count']:4d} NaN ({nan_ratio*100:5.1f}%), "
                          f"有效范围: [{start:4d}, {end:4d}]")
                else:
                    print(f"    {lead:3s}: ✗ 全是NaN")
            else:
                print(f"    {lead:3s}: ✓ 无NaN ({info['valid_count']} 有效点)")
        
        print()
    
    # 汇总统计
    print("="*70)
    print("汇总统计")
    print("="*70)
    print()
    
    print("各导联的NaN比例分布:")
    for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        if lead not in lead_nan_stats:
            continue
        
        ratios = lead_nan_stats[lead]
        avg_ratio = np.mean(ratios)
        max_ratio = np.max(ratios)
        min_ratio = np.min(ratios)
        
        if avg_ratio > 0:
            print(f"  {lead:3s}: 平均 {avg_ratio*100:5.1f}% NaN (范围: {min_ratio*100:.1f}%-{max_ratio*100:.1f}%)")
        else:
            print(f"  {lead:3s}: ✓ 无NaN")
    
    print()
    print("="*70)
    print("结论:")
    print("="*70)
    
    # 判断是否有"长导联 vs 短导联"的模式
    long_lead_candidates = []
    short_lead_candidates = []
    
    for lead, ratios in lead_nan_stats.items():
        avg_ratio = np.mean(ratios)
        if avg_ratio < 0.05:  # < 5% NaN
            long_lead_candidates.append(lead)
        elif avg_ratio > 0.70:  # > 70% NaN
            short_lead_candidates.append(lead)
    
    if long_lead_candidates:
        print(f"✓ 检测到长导联（完整数据）: {', '.join(long_lead_candidates)}")
    
    if short_lead_candidates:
        print(f"⚠️  检测到短导联（部分数据）: {', '.join(short_lead_candidates)}")
        print("   → 这些导联的NaN会被替换为0")
    
    if not long_lead_candidates and not short_lead_candidates:
        print("✓ 所有导联数据都比较完整")
    
    print()
    print("💡 建议:")
    if short_lead_candidates:
        print("  - Dataset代码已添加 np.nan_to_num() 来处理NaN")
        print("  - NaN会被替换为0，表示该时间段无信号")
        print("  - 这是正确的处理方式，不会影响训练")
    else:
        print("  - 数据质量良好，无需特殊处理")


if __name__ == "__main__":
    # 使用示例:
    # python check_csv_nan.py --csv_root /path/to/train --max_samples 20
    main()