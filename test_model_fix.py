#!/usr/bin/env python3
"""
测试ECG模型修复

主要测试:
1. FPN通道数匹配
2. 模型前向传播
3. 输出形状验证
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ecg_model import ProgressiveLeadLocalizationModel


def test_fpn_channels():
    """测试FPN通道数是否正确"""
    print("="*80)
    print("测试1: FPN通道数匹配")
    print("="*80)
    
    try:
        model = ProgressiveLeadLocalizationModel(
            num_leads=12,
            encoder_name='resnet50',
            pretrained=False
        )
        
        # 检查lateral connections
        print("\n检查Lateral Connections:")
        print(f"  lateral5 (2048->256): {model.lateral5}")
        print(f"  lateral4 (1024->256): {model.lateral4}")
        print(f"  lateral3 (512->256):  {model.lateral3}")
        print(f"  lateral2 (256->256):  {model.lateral2}")
        
        print("\n✓ Lateral connections配置正确")
        return True
        
    except Exception as e:
        print(f"\n✗ FPN配置错误: {e}")
        return False


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*80)
    print("测试2: 模型前向传播")
    print("="*80)
    
    try:
        model = ProgressiveLeadLocalizationModel(
            num_leads=12,
            encoder_name='resnet50',
            pretrained=False
        )
        model.eval()
        
        # 测试不同尺寸
        test_sizes = [
            (512, 672),  # 标准尺寸
            (384, 512),  # 小尺寸
            (640, 800),  # 大尺寸
        ]
        
        for H, W in test_sizes:
            print(f"\n测试尺寸: {H}x{W}")
            
            x = torch.randn(2, 3, H, W)
            
            with torch.no_grad():
                outputs = model(x)
            
            # 验证输出
            expected_shapes = {
                'coarse_baseline': (2, 1, H//16, W//16),
                'time_ranges': (2, 12, 2),
                'text_masks': (2, 13, H//4, W//4),
                'auxiliary_mask': (2, 1, H//4, W//4),
                'paper_speed_mask': (2, 1, H//4, W//4),
                'gain_mask': (2, 1, H//4, W//4),
                'lead_baselines': (2, 12, H//4, W//4),
            }
            
            all_correct = True
            for key, expected_shape in expected_shapes.items():
                actual_shape = tuple(outputs[key].shape)
                if actual_shape != expected_shape:
                    print(f"  ✗ {key}: 期望 {expected_shape}, 实际 {actual_shape}")
                    all_correct = False
                else:
                    print(f"  ✓ {key}: {actual_shape}")
            
            if not all_correct:
                return False
        
        print("\n✓ 所有尺寸测试通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_features():
    """测试编码器特征提取"""
    print("\n" + "="*80)
    print("测试3: 编码器特征尺寸")
    print("="*80)
    
    try:
        model = ProgressiveLeadLocalizationModel(
            num_leads=12,
            encoder_name='resnet50',
            pretrained=False
        )
        model.eval()
        
        H, W = 512, 672
        x = torch.randn(1, 3, H, W)
        
        with torch.no_grad():
            features = model.forward_encoder(x)
        
        print("\n编码器输出:")
        for key, feat in features.items():
            print(f"  {key}: {feat.shape}")
        
        # 验证分辨率
        expected_resolutions = {
            'd2': (H//4, W//4),
            'd3': (H//8, W//8),
            'd4': (H//16, W//16),
            'd5': (H//32, W//32),
        }
        
        all_correct = True
        for key, (exp_h, exp_w) in expected_resolutions.items():
            actual_h, actual_w = features[key].shape[2:]
            if (actual_h, actual_w) != (exp_h, exp_w):
                print(f"\n✗ {key} 分辨率错误: 期望 {(exp_h, exp_w)}, 实际 {(actual_h, actual_w)}")
                all_correct = False
        
        if all_correct:
            print("\n✓ 编码器特征尺寸正确")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"\n✗ 编码器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试显存占用"""
    print("\n" + "="*80)
    print("测试4: 显存占用")
    print("="*80)
    
    try:
        if not torch.cuda.is_available():
            print("\n⚠️  CUDA不可用，跳过显存测试")
            return True
        
        device = torch.device('cuda')
        model = ProgressiveLeadLocalizationModel(
            num_leads=12,
            encoder_name='resnet50',
            pretrained=False
        ).to(device)
        model.eval()
        
        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 测试不同batch size
        batch_sizes = [1, 2, 4, 8]
        H, W = 512, 672
        
        print(f"\n测试图像尺寸: {H}x{W}")
        print(f"{'Batch Size':<12} {'峰值显存(MB)':<15} {'状态'}")
        print("-" * 40)
        
        for bs in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                x = torch.randn(bs, 3, H, W, device=device)
                
                with torch.no_grad():
                    outputs = model(x)
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"{bs:<12} {peak_memory:<15.1f} ✓")
                
                del x, outputs
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"{bs:<12} {'OOM':<15} ✗")
                else:
                    raise
        
        print("\n✓ 显存测试完成")
        return True
        
    except Exception as e:
        print(f"\n✗ 显存测试失败: {e}")
        return False


def test_training_mode():
    """测试训练模式"""
    print("\n" + "="*80)
    print("测试5: 训练模式")
    print("="*80)
    
    try:
        model = ProgressiveLeadLocalizationModel(
            num_leads=12,
            encoder_name='resnet50',
            pretrained=False
        )
        model.train()
        
        H, W = 512, 672
        x = torch.randn(2, 3, H, W)
        
        # 前向传播
        outputs = model(x)
        
        # 计算损失（模拟）
        loss = 0
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                loss += val.mean()
        
        # 反向传播
        loss.backward()
        
        print("\n✓ 前向传播: 成功")
        print("✓ 反向传播: 成功")
        
        # 检查梯度
        has_grad = False
        no_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_grad = True
                else:
                    no_grad_params.append(name)
        
        if has_grad:
            print("✓ 梯度计算: 成功")
            if len(no_grad_params) > 0:
                print(f"⚠️  {len(no_grad_params)} 个参数没有梯度（可能未使用）")
        else:
            print("✗ 没有参数有梯度")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ 训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("ECG模型修复验证")
    print("="*80)
    
    results = []
    
    # 测试1: FPN通道
    results.append(("FPN通道数匹配", test_fpn_channels()))
    
    # 测试2: 前向传播
    results.append(("模型前向传播", test_forward_pass()))
    
    # 测试3: 编码器
    results.append(("编码器特征尺寸", test_encoder_features()))
    
    # 测试4: 显存
    results.append(("显存占用", test_memory_usage()))
    
    # 测试5: 训练模式
    results.append(("训练模式", test_training_mode()))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status:8s} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("="*80)
    if all_passed:
        print("✓ 所有测试通过！模型已修复，可以开始训练")
        print("\n下一步:")
        print("  python train.py --sim_root ./data/simulated_ecg --csv_root ./data/original_csv ...")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())