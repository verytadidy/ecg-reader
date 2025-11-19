"""
训练集物理约束分析脚本

分析真实ECG数据是否满足:
1. Einthoven定律: I + III = II
2. 加压导联和: aVR + aVL + aVF = 0
3. 统计约束满足程度的分布
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# ============================
# 1. 约束检查函数
# ============================
def check_einthoven_law(df):
    """
    检查 Einthoven定律: I + III = II
    
    返回:
        satisfied: bool, 是否满足
        relative_error: float, 归一化误差
        max_absolute_error: float, 最大绝对误差(mV)
    """
    leads_required = ['I', 'II', 'III']
    if not all(lead in df.columns for lead in leads_required):
        return None, None, None
    
    # --- 修改点 ---
    # 1. 选取相关列并丢弃 *任何* 包含NaN的行
    #    这样我们只在 I, II, III 均有效的时间点进行比较
    df_valid = df[leads_required].dropna()
    
    # 2. 检查是否有剩余数据
    if df_valid.empty:
        return None, None, None # 没有可供比较的有效数据点
    
    I = df_valid['I'].values
    II = df_valid['II'].values
    III = df_valid['III'].values
    # --- 修改结束 ---
    
    # 计算误差
    lhs = I + III
    rhs = II
    absolute_error = np.abs(lhs - rhs)
    
    # 归一化误差 (相对于II的最大幅度)
    max_amplitude = max(np.abs(II).max(), 1e-6)
    relative_error = absolute_error.mean() / max_amplitude
    max_absolute_error = absolute_error.max()
    
    # 判断是否满足 (阈值10%)
    satisfied = relative_error < 0.10
    
    return satisfied, relative_error, max_absolute_error

def check_augmented_sum(df):
    """
    检查加压导联和: aVR + aVL + aVF = 0
    
    返回:
        satisfied: bool, 是否满足
        relative_error: float, 归一化误差
        max_absolute_error: float, 最大绝对误差(mV)
    """
    leads_required = ['aVR', 'aVL', 'aVF']
    if not all(lead in df.columns for lead in leads_required):
        return None, None, None
    
    # --- 修改点 ---
    # 1. 选取相关列并丢弃NaN
    df_valid = df[leads_required].dropna()
    
    # 2. 检查是否有剩余数据
    if df_valid.empty:
        return None, None, None # 没有可供比较的有效数据点
    
    aVR = df_valid['aVR'].values
    aVL = df_valid['aVL'].values
    aVF = df_valid['aVF'].values
    # --- 修改结束 ---
    
    # 求和应该为0
    sum_signal = aVR + aVL + aVF
    absolute_error = np.abs(sum_signal)
    
    # 归一化误差 (相对于aVF的最大幅度)
    max_amplitude = max(np.abs(aVF).max(), 1e-6)
    relative_error = absolute_error.mean() / max_amplitude
    max_absolute_error = absolute_error.max()
    
    # 判断是否满足 (阈值10%)
    satisfied = relative_error < 0.10
    
    return satisfied, relative_error, max_absolute_error

def check_lead_derivations(df):
    """
    检查所有可推导的导联关系
    
    返回:
        results: dict, {derivation_name: (satisfied, error)}
    """
    results = {}
    
    # 检查 III = II - I
    leads_einthoven = ['I', 'II', 'III']
    if all(lead in df.columns for lead in leads_einthoven):
        # --- 修改点 ---
        df_valid = df[leads_einthoven].dropna()
        if not df_valid.empty:
        # --- 修改结束 ---
            III_derived = df_valid['II'].values - df_valid['I'].values
            III_actual = df_valid['III'].values
            error = np.abs(III_derived - III_actual)
            max_amplitude = max(np.abs(III_actual).max(), 1e-6)
            relative_error = error.mean() / max_amplitude
            results['III_from_I_II'] = (relative_error < 0.10, relative_error)
    
    # 检查 I = II - III
    if all(lead in df.columns for lead in leads_einthoven):
        # --- 修改点 ---
        df_valid = df[leads_einthoven].dropna()
        if not df_valid.empty:
        # --- 修改结束 ---
            I_derived = df_valid['II'].values - df_valid['III'].values
            I_actual = df_valid['I'].values
            error = np.abs(I_derived - I_actual)
            max_amplitude = max(np.abs(I_actual).max(), 1e-6)
            relative_error = error.mean() / max_amplitude
            results['I_from_II_III'] = (relative_error < 0.10, relative_error)
    
    # 检查 aVF = -aVR - aVL
    leads_augmented = ['aVR', 'aVL', 'aVF']
    if all(lead in df.columns for lead in leads_augmented):
        # --- 修改点 ---
        df_valid = df[leads_augmented].dropna()
        if not df_valid.empty:
        # --- 修改结束 ---
            aVF_derived = -df_valid['aVR'].values - df_valid['aVL'].values
            aVF_actual = df_valid['aVF'].values
            error = np.abs(aVF_derived - aVF_actual)
            max_amplitude = max(np.abs(aVF_actual).max(), 1e-6)
            relative_error = error.mean() / max_amplitude
            results['aVF_from_aVR_aVL'] = (relative_error < 0.10, relative_error)
    
    return results

# ============================
# 2. 批量分析函数
# ============================
def analyze_single_ecg(csv_path):
    """
    分析单个ECG文件
    
    返回:
        analysis: dict, 分析结果
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Einthoven定律
        einthoven_satisfied, einthoven_error, einthoven_max_error = check_einthoven_law(df)
        
        # 加压导联和
        augmented_satisfied, augmented_error, augmented_max_error = check_augmented_sum(df)
        
        # 推导关系
        derivations = check_lead_derivations(df)
        
        return {
            'ecg_id': csv_path.stem,
            'einthoven': {
                'satisfied': einthoven_satisfied,
                'relative_error': einthoven_error,
                'max_absolute_error_mV': einthoven_max_error
            },
            'augmented_sum': {
                'satisfied': augmented_satisfied,
                'relative_error': augmented_error,
                'max_absolute_error_mV': augmented_max_error
            },
            'derivations': derivations,
            'has_all_limb_leads': all(lead in df.columns for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']),
            'has_all_chest_leads': all(lead in df.columns for lead in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
        }
    except Exception as e:
        return {
            'ecg_id': csv_path.stem,
            'error': str(e)
        }

def analyze_dataset(train_dir, train_csv_path, sample_size=None, random_seed=42):
    """
    分析整个数据集
    
    参数:
        train_dir: 训练数据目录
        train_csv_path: train.csv路径
        sample_size: 抽样数量 (None表示全部)
        random_seed: 随机种子
    """
    train_dir = Path(train_dir)
    train_meta = pd.read_csv(train_csv_path)
    
    # 抽样
    if sample_size is not None:
        train_meta = train_meta.sample(n=min(sample_size, len(train_meta)), 
                                       random_state=random_seed)
    
    print(f"分析样本数: {len(train_meta)}")
    print("=" * 70)
    
    # 批量分析
    results = []
    for idx, row in tqdm(train_meta.iterrows(), total=len(train_meta), desc="分析ECG"):
        ecg_id = str(row['id'])
        csv_path = train_dir / ecg_id / f"{ecg_id}.csv"
        
        if csv_path.exists():
            result = analyze_single_ecg(csv_path)
            results.append(result)
    
    return results, train_meta

# ============================
# 3. 统计与可视化
# ============================
def compute_statistics(results):
    """
    计算统计指标
    """
    stats = {
        'total_samples': len(results),
        'einthoven': {
            'available': 0,
            'satisfied': 0,
            'errors': []
        },
        'augmented_sum': {
            'available': 0,
            'satisfied': 0,
            'errors': []
        },
        'derivations': {},
        'missing_data': {
            'missing_limb_leads': 0,
            'missing_chest_leads': 0
        }
    }
    
    for result in results:
        if 'error' in result:
            continue
        
        # Einthoven
        if result['einthoven']['satisfied'] is not None:
            stats['einthoven']['available'] += 1
            if result['einthoven']['satisfied']:
                stats['einthoven']['satisfied'] += 1
            if result['einthoven']['relative_error'] is not None:
                stats['einthoven']['errors'].append(result['einthoven']['relative_error'])
        
        # Augmented sum
        if result['augmented_sum']['satisfied'] is not None:
            stats['augmented_sum']['available'] += 1
            if result['augmented_sum']['satisfied']:
                stats['augmented_sum']['satisfied'] += 1
            if result['augmented_sum']['relative_error'] is not None:
                stats['augmented_sum']['errors'].append(result['augmented_sum']['relative_error'])
        
        # Derivations
        for deriv_name, (satisfied, error) in result['derivations'].items():
            if deriv_name not in stats['derivations']:
                stats['derivations'][deriv_name] = {
                    'satisfied': 0,
                    'total': 0,
                    'errors': []
                }
            stats['derivations'][deriv_name]['total'] += 1
            if satisfied:
                stats['derivations'][deriv_name]['satisfied'] += 1
            stats['derivations'][deriv_name]['errors'].append(error)
        
        # Missing data
        if not result['has_all_limb_leads']:
            stats['missing_data']['missing_limb_leads'] += 1
        if not result['has_all_chest_leads']:
            stats['missing_data']['missing_chest_leads'] += 1
    
    return stats

def print_statistics(stats):
    """
    打印统计结果
    """
    print("\n" + "=" * 70)
    print("约束满足情况统计")
    print("=" * 70)
    
    # Einthoven定律
    print("\n1. Einthoven定律 (I + III = II)")
    print("-" * 70)
    if stats['einthoven']['available'] > 0:
        satisfaction_rate = stats['einthoven']['satisfied'] / stats['einthoven']['available']
        print(f"   可用样本: {stats['einthoven']['available']}/{stats['total_samples']}")
        print(f"   满足约束: {stats['einthoven']['satisfied']} ({satisfaction_rate*100:.2f}%)")
        
        errors = np.array(stats['einthoven']['errors'])
        print(f"   相对误差:")
        print(f"     均值: {errors.mean():.6f}")
        print(f"     中位数: {np.median(errors):.6f}")
        print(f"     标准差: {errors.std():.6f}")
        print(f"     最大值: {errors.max():.6f}")
        print(f"     90分位: {np.percentile(errors, 90):.6f}")
        print(f"     95分位: {np.percentile(errors, 95):.6f}")
    else:
        print("   ⚠️ 无可用样本")
    
    # 加压导联和
    print("\n2. 加压导联和 (aVR + aVL + aVF = 0)")
    print("-" * 70)
    if stats['augmented_sum']['available'] > 0:
        satisfaction_rate = stats['augmented_sum']['satisfied'] / stats['augmented_sum']['available']
        print(f"   可用样本: {stats['augmented_sum']['available']}/{stats['total_samples']}")
        print(f"   满足约束: {stats['augmented_sum']['satisfied']} ({satisfaction_rate*100:.2f}%)")
        
        errors = np.array(stats['augmented_sum']['errors'])
        print(f"   相对误差:")
        print(f"     均值: {errors.mean():.6f}")
        print(f"     中位数: {np.median(errors):.6f}")
        print(f"     标准差: {errors.std():.6f}")
        print(f"     最大值: {errors.max():.6f}")
        print(f"     90分位: {np.percentile(errors, 90):.6f}")
        print(f"     95分位: {np.percentile(errors, 95):.6f}")
    else:
        print("   ⚠️ 无可用样本")
    
    # 推导关系
    print("\n3. 导联推导关系")
    print("-" * 70)
    for deriv_name, deriv_stats in stats['derivations'].items():
        satisfaction_rate = deriv_stats['satisfied'] / deriv_stats['total']
        errors = np.array(deriv_stats['errors'])
        print(f"   {deriv_name}:")
        print(f"     满足率: {satisfaction_rate*100:.2f}% ({deriv_stats['satisfied']}/{deriv_stats['total']})")
        print(f"     误差中位数: {np.median(errors):.6f}")
    
    # 缺失数据
    print("\n4. 数据完整性")
    print("-" * 70)
    print(f"   缺失肢体导联: {stats['missing_data']['missing_limb_leads']}/{stats['total_samples']}")
    print(f"   缺失胸导联: {stats['missing_data']['missing_chest_leads']}/{stats['total_samples']}")

def plot_error_distribution(stats, save_path=None):
    """
    绘制误差分布图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Einthoven误差分布
    ax1 = axes[0, 0]
    if len(stats['einthoven']['errors']) > 0:
        errors = np.array(stats['einthoven']['errors'])
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(0.10, color='r', linestyle='--', linewidth=2, label='10% 阈值')
        ax1.set_xlabel('相对误差')
        ax1.set_ylabel('样本数')
        ax1.set_title('Einthoven定律误差分布 (I + III = II)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 加压导联和误差分布
    ax2 = axes[0, 1]
    if len(stats['augmented_sum']['errors']) > 0:
        errors = np.array(stats['augmented_sum']['errors'])
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax2.axvline(0.10, color='r', linestyle='--', linewidth=2, label='10% 阈值')
        ax2.set_xlabel('相对误差')
        ax2.set_ylabel('样本数')
        ax2.set_title('加压导联和误差分布 (aVR + aVL + aVF = 0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 误差CDF
    ax3 = axes[1, 0]
    if len(stats['einthoven']['errors']) > 0:
        errors_ein = np.array(stats['einthoven']['errors'])
        sorted_errors = np.sort(errors_ein)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax3.plot(sorted_errors, cdf, label='Einthoven', linewidth=2)
    
    if len(stats['augmented_sum']['errors']) > 0:
        errors_aug = np.array(stats['augmented_sum']['errors'])
        sorted_errors = np.sort(errors_aug)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax3.plot(sorted_errors, cdf, label='Augmented Sum', linewidth=2)
    
    ax3.axvline(0.10, color='r', linestyle='--', linewidth=2, label='10% 阈值')
    ax3.set_xlabel('相对误差')
    ax3.set_ylabel('累积概率')
    ax3.set_title('误差累积分布函数 (CDF)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 满足率对比
    ax4 = axes[1, 1]
    categories = []
    satisfaction_rates = []
    
    if stats['einthoven']['available'] > 0:
        categories.append('Einthoven\n定律')
        satisfaction_rates.append(stats['einthoven']['satisfied'] / stats['einthoven']['available'] * 100)
    
    if stats['augmented_sum']['available'] > 0:
        categories.append('加压导联\n求和')
        satisfaction_rates.append(stats['augmented_sum']['satisfied'] / stats['augmented_sum']['available'] * 100)
    
    for deriv_name, deriv_stats in stats['derivations'].items():
        categories.append(deriv_name.replace('_', '\n'))
        satisfaction_rates.append(deriv_stats['satisfied'] / deriv_stats['total'] * 100)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax4.bar(categories, satisfaction_rates, color=colors[:len(categories)], alpha=0.7, edgecolor='black')
    ax4.axhline(90, color='g', linestyle='--', linewidth=2, label='90% 目标')
    ax4.set_ylabel('满足率 (%)')
    ax4.set_title('约束满足率对比')
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")
    
    plt.show()

# ============================
# 4. 主函数
# ============================
def main():
    """
    主分析流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='分析ECG训练集的物理约束满足情况')
    parser.add_argument('--train_dir', type=str, required=True,
                       help='训练数据目录 (包含各个ECG ID的子文件夹)')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='train.csv文件路径')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='抽样数量 (默认: 全部)')
    parser.add_argument('--output', type=str, default='constraint_analysis.json',
                       help='输出JSON文件路径')
    parser.add_argument('--plot', type=str, default='constraint_analysis.png',
                       help='输出图表文件路径')
    
    args = parser.parse_args()
    
    # 分析数据集
    results, train_meta = analyze_dataset(
        args.train_dir,
        args.train_csv,
        sample_size=args.sample_size
    )
    
    # 计算统计
    stats = compute_statistics(results)
    
    # 打印统计
    print_statistics(stats)
    
    # 绘制图表
    plot_error_distribution(stats, save_path=args.plot)
    
    # 保存详细结果
    output_data = {
        'statistics': {
            'total_samples': stats['total_samples'],
            'einthoven': {
                'available': stats['einthoven']['available'],
                'satisfied': stats['einthoven']['satisfied'],
                'satisfaction_rate': stats['einthoven']['satisfied'] / stats['einthoven']['available'] 
                    if stats['einthoven']['available'] > 0 else 0,
                'error_mean': float(np.mean(stats['einthoven']['errors'])) 
                    if len(stats['einthoven']['errors']) > 0 else None,
                'error_median': float(np.median(stats['einthoven']['errors'])) 
                    if len(stats['einthoven']['errors']) > 0 else None,
                'error_std': float(np.std(stats['einthoven']['errors'])) 
                    if len(stats['einthoven']['errors']) > 0 else None
            },
            'augmented_sum': {
                'available': stats['augmented_sum']['available'],
                'satisfied': stats['augmented_sum']['satisfied'],
                'satisfaction_rate': stats['augmented_sum']['satisfied'] / stats['augmented_sum']['available'] 
                    if stats['augmented_sum']['available'] > 0 else 0,
                'error_mean': float(np.mean(stats['augmented_sum']['errors'])) 
                    if len(stats['augmented_sum']['errors']) > 0 else None,
                'error_median': float(np.median(stats['augmented_sum']['errors'])) 
                    if len(stats['augmented_sum']['errors']) > 0 else None,
                'error_std': float(np.std(stats['augmented_sum']['errors'])) 
                    if len(stats['augmented_sum']['errors']) > 0 else None
            }
        },
        'sample_results': results[:100]  # 保存前100个样本的详细结果
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n详细结果已保存至: {args.output}")
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""
使用示例:

# 分析全部数据
python analyze_constraints.py \
    --train_dir /path/to/train \
    --train_csv /path/to/train.csv \
    --output constraint_analysis.json \
    --plot constraint_analysis.png

# 抽样1000个样本
python analyze_constraints.py \
    --train_dir /path/to/train \
    --train_csv /path/to/train.csv \
    --sample_size 1000 \
    --output constraint_analysis_sample.json \
    --plot constraint_analysis_sample.png
"""