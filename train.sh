#!/bin/bash
# train_v48.sh - ECG V48 完全修复版训练一键启动脚本

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 数据路径（请修改为您的实际 V48 数据路径）
# 注意：V48 数据集需包含以下文件：
#   - *_dirty.png
#   - *_gt_signals.json
#   - *_metadata.json
#   - *_label_baseline.npy
#   - *_label_text_multi.npy
#   - *_label_wave.npy (新增)
#   - *_label_auxiliary.npy (新增)
#   - *_label_paper_speed.npy (新增)
#   - *_label_gain.npy (新增)
SIM_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization-simulations-V47"
CSV_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization/train"

# 训练参数
EPOCHS=50
WARMUP_EPOCHS=10        # V48 新增：权重调度 warmup
BATCH_SIZE=8
LR=1e-4
NUM_WORKERS=4
TARGET_FS=500

# 输入尺寸（V48 优化）
INPUT_SIZE="512 2048"  # H W 

# 输出目录
OUTPUT_DIR="./outputs"

# V48 移除了手动设置的 weight_seg/weight_signal
# 这些权重由 Loss 函数内部管理，并通过 ProgressiveWeightScheduler 自动调整

# ==================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_cyan() {
    echo -e "${CYAN}$1${NC}"
}

# 检查必需文件
check_requirements() {
    print_header "环境检查 (V48)"
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "未找到python"
        exit 1
    fi
    print_info "Python: $(python --version 2>&1)"
    
    # 检查必需的Python包
    echo ""
    echo "检查Python依赖..."
    
    # 检查PyTorch版本
    if python -c "import torch" &> /dev/null; then
        TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null)
        print_info "PyTorch: $TORCH_VERSION"
    else
        print_error "未安装 PyTorch"
        echo "安装命令: pip install torch torchvision"
        exit 1
    fi
    
    # 检查 torchvision (V48 需要 roi_align)
    if python -c "import torchvision" &> /dev/null; then
        TV_VERSION=$(python -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null)
        print_info "torchvision: $TV_VERSION"
    else
        print_error "未安装 torchvision (V48 必需)"
        exit 1
    fi
    
    # 检查 scipy (计算相关系数)
    if python -c "import scipy" &> /dev/null; then
        print_info "scipy: installed"
    else
        print_warning "未安装 scipy (验证指标会受限)"
        echo "安装命令: pip install scipy"
    fi

    # 检测计算设备
    echo ""
    if python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        print_info "检测到CUDA: $CUDA_VERSION"
        print_cyan "  混合精度训练: 可用 (使用 --use_amp)"
    elif python -c "import torch; assert torch.backends.mps.is_available()" &> /dev/null; then
        print_info "检测到Apple Silicon MPS"
        print_cyan "  MPS Fallback: 已自动启用"
    else
        print_warning "仅CPU模式（训练速度较慢）"
    fi
    
    # 检查核心代码文件 (V48 版本)
    echo ""
    echo "检查V48代码文件..."
    REQUIRED_FILES=(
        "ecg_dataset_v48_fixed.py" 
        "ecg_model_v48_fixed.py" 
        "ecg_loss_v48_fixed.py" 
        "ecg_train_v48_fixed.py"
    )
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "缺少文件: $file"
            echo "请确保使用 V48 完全修复版代码"
            exit 1
        else
            print_info "$file"
        fi
    done
    
    # 检查数据目录
    echo ""
    echo "检查数据目录..."
    if [ ! -d "$SIM_ROOT" ]; then
        print_error "仿真数据目录不存在: $SIM_ROOT"
        echo "请修改脚本中的 SIM_ROOT 变量"
        exit 1
    fi
    print_info "仿真数据: $SIM_ROOT"
    
    if [ ! -d "$CSV_ROOT" ]; then
        print_error "CSV数据目录不存在: $CSV_ROOT"
        echo "请修改脚本中的 CSV_ROOT 变量"
        exit 1
    fi
    print_info "CSV数据: $CSV_ROOT"
    
    # 统计样本数 (V48文件结构检查)
    echo ""
    echo "检查V48数据完整性..."
    
    # 检查关键文件
    GT_COUNT=$(find "$SIM_ROOT" -maxdepth 2 -name "*_gt_signals.json" 2>/dev/null | wc -l)
    WAVE_COUNT=$(find "$SIM_ROOT" -maxdepth 2 -name "*_label_wave.npy" 2>/dev/null | wc -l)
    AUX_COUNT=$(find "$SIM_ROOT" -maxdepth 2 -name "*_label_auxiliary.npy" 2>/dev/null | wc -l)
    
    if [ "$GT_COUNT" -eq 0 ]; then
        print_error "未找到 *_gt_signals.json 文件"
        echo "请确认 SIM_ROOT 指向正确的数据集"
        exit 1
    fi
    print_info "找到 $GT_COUNT 个样本"
    
    if [ "$WAVE_COUNT" -eq 0 ]; then
        print_warning "未找到 *_label_wave.npy (V48 关键文件)"
        echo "这可能是旧版数据集，建议重新生成"
    else
        print_info "波形分割标签: $WAVE_COUNT 个文件"
    fi
    
    if [ "$AUX_COUNT" -eq 0 ]; then
        print_warning "未找到 *_label_auxiliary.npy"
    else
        print_info "辅助掩码标签: $AUX_COUNT 个文件"
    fi
    
    echo ""
}

# 测试数据加载
test_data_loading() {
    print_header "数据加载测试 (V48)"
    
    echo "测试前10个样本的数据加载..."
    echo ""
    
    # 使用 V48 Dataset 的测试功能
    python ecg_dataset_v48_fixed.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT"
    
    if [ $? -eq 0 ]; then
        echo ""
        print_info "数据加载测试通过！"
        echo ""
        print_cyan "V48 新增字段已验证:"
        print_cyan "  ✓ wave_segmentation (H, W)"
        print_cyan "  ✓ auxiliary_mask (H, W)"
        print_cyan "  ✓ paper_speed_mask (H, W)"
        print_cyan "  ✓ gain_mask (H, W)"
        echo ""
    else
        print_error "数据加载失败，请检查:"
        echo "  1. 数据路径是否正确"
        echo "  2. 数据格式是否符合V48规范"
        echo "  3. 是否包含所有必需的 .npy 文件"
        exit 1
    fi
}

# 快速调试训练
debug_train() {
    print_header "快速调试训练 (V48)"
    
    echo "配置:"
    echo "  样本数: 50"
    echo "  Epochs: 5"
    echo "  Warmup: 2"
    echo "  Batch Size: 4"
    echo "  Workers: 0"
    echo "  Workers: 0"
    echo ""
    
    DEBUG_OUTPUT="$OUTPUT_DIR/debug_v48"
    mkdir -p "$DEBUG_OUTPUT"
    
    python ecg_train_v48_fixed.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --output_dir "$DEBUG_OUTPUT" \
        --max_samples 50 \
        --epochs 5 \
        --warmup_epochs 2 \
        --batch_size 4 \
        --lr $LR \
        --num_workers 0 \
        --input_size $INPUT_SIZE \
        --target_fs $TARGET_FS \
        --pretrained \
        --log_interval 5 \
        --save_interval 2
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        print_info "调试训练完成！"
        print_info "检查点: $DEBUG_OUTPUT/checkpoint_latest.pth"
        echo ""
        print_cyan "查看训练日志:"
        echo "  tensorboard --logdir $DEBUG_OUTPUT/logs --port 6006"
        echo ""
    else
        print_error "调试训练失败（退出码: $EXIT_CODE）"
        exit $EXIT_CODE
    fi
}

# 完整训练
full_train() {
    print_header "完整训练 (V48)"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TRAIN_OUTPUT="$OUTPUT_DIR/v48_exp_${TIMESTAMP}"
    
    echo "训练配置:"
    echo "  输出目录: $TRAIN_OUTPUT"
    echo "  Epochs: $EPOCHS"
    echo "  Warmup Epochs: $WARMUP_EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LR"
    echo "  Workers: $NUM_WORKERS"
    echo "  Input Size: $INPUT_SIZE"
    echo ""
    echo "V48 特性:"
    echo "  ✓ 波形分割监督 (WaveSegmentationLoss)"
    echo "  ✓ 辅助掩码抑制 (AuxiliarySuppressionLoss)"
    echo "  ✓ 可微 RoI 提取 (grid_sample)"
    echo "  ✓ 渐进式权重调度 (ProgressiveWeightScheduler)"
    echo "  ✓ OCR 任务监督 (paper_speed + gain)"
    echo ""
    
    mkdir -p "$TRAIN_OUTPUT"
    
    # 询问是否使用混合精度
    USE_AMP=""
    if python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        read -p "是否启用混合精度训练 (AMP)? [Y/n] " yn
        case $yn in
            [Nn]*) USE_AMP="" ;;
            *) USE_AMP="--use_amp" ;;
        esac
    fi
    
    python ecg_train_v48_fixed.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --output_dir "$TRAIN_OUTPUT" \
        --epochs $EPOCHS \
        --warmup_epochs $WARMUP_EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --num_workers $NUM_WORKERS \
        --input_size $INPUT_SIZE \
        --target_fs $TARGET_FS \
        --pretrained \
        --scheduler cosine \
        --save_interval 5 \
        --log_interval 10 \
        $USE_AMP
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        print_info "训练完成！"
        echo ""
        print_info "查看训练曲线:"
        echo "  tensorboard --logdir $TRAIN_OUTPUT/logs --port 6006"
        echo ""
        print_info "最佳模型:"
        echo "  $TRAIN_OUTPUT/checkpoint_best.pth"
        echo ""
    else
        print_error "训练失败（退出码: $EXIT_CODE）"
        exit $EXIT_CODE
    fi
}

# 恢复训练
resume_train() {
    print_header "恢复训练 (V48)"
    
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint_latest.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        print_error "未找到检查点文件"
        echo "请手动指定:"
        echo "  python ecg_train_v48_fixed.py --resume /path/to/checkpoint.pth ..."
        exit 1
    fi
    
    print_info "找到检查点: $LATEST_CHECKPOINT"
    RESUME_DIR=$(dirname "$LATEST_CHECKPOINT")
    
    echo ""
    read -p "是否恢复训练？[y/N] " yn
    read -p "是否恢复训练？[y/N] " yn
    case $yn in
        [Yy]*)
            python ecg_train_v48_fixed.py \
                --sim_root "$SIM_ROOT" \
                --csv_root "$CSV_ROOT" \
                --output_dir "$RESUME_DIR" \
                --resume "$LATEST_CHECKPOINT" \
                --epochs $EPOCHS \
                --warmup_epochs $WARMUP_EPOCHS \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --num_workers $NUM_WORKERS \
                --input_size $INPUT_SIZE \
                --target_fs $TARGET_FS \
                --weight_seg $WEIGHT_SEG \
                --weight_signal $WEIGHT_SIGNAL
            
            if [ $? -eq 0 ]; then
                print_info "训练完成！"
            else
                print_error "训练失败"
                exit 1
            fi
            ;;
        *)
            print_info "已取消"
            exit 0
            ;;
    esac
}

# 验证单个检查点
validate_checkpoint() {
    print_header "验证模型 (V48)"
    
    # 查找最佳模型
    BEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint_best.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$BEST_CHECKPOINT" ]; then
        print_error "未找到 checkpoint_best.pth"
        exit 1
    fi
    
    print_info "验证模型: $BEST_CHECKPOINT"
    echo ""
    
    # 创建验证脚本
    cat > /tmp/validate_v48.py << 'EOF'
import sys
import torch
from pathlib import Path
from ecg_train_v48_fixed import ECGTrainerV48
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--sim_root', type=str, required=True)
parser.add_argument('--csv_root', type=str, required=True)
args = parser.parse_args()

# 加载检查点
checkpoint = torch.load(args.checkpoint, map_location='cpu')
print(f"Checkpoint Epoch: {checkpoint['epoch']}")
print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
print(f"\nCriterion Weights:")
for k, v in checkpoint.get('criterion_weights', {}).items():
    print(f"  {k}: {v:.2f}")
EOF
    
    python /tmp/validate_v48.py \
        --checkpoint "$BEST_CHECKPOINT" \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT"
    
    rm /tmp/validate_v48.py
    echo ""
}

# 主菜单
show_menu() {
    echo ""
    print_header "ECG V48 完全修复版训练系统"
    echo ""
    print_cyan "数据路径:"
    echo "  仿真数据: $SIM_ROOT"
    echo "  CSV数据: $CSV_ROOT"
    echo ""
    print_cyan "V48 新特性:"
    echo "  • 波形分割监督 (12类语义分割)"
    echo "  • 辅助掩码抑制 (定标脉冲/分隔符)"
    echo "  • OCR 任务 (纸速/增益识别)"
    echo "  • 可微 RoI 提取 (端到端梯度)"
    echo "  • 渐进式权重调度 (前 $WARMUP_EPOCHS 轮 warmup)"
    echo ""
    echo "请选择操作:"
    echo "  1) 检查环境和数据 (推荐首次运行)"
    echo "  2) 测试数据加载"
    echo "  3) 快速调试训练 (5 epochs, 50样本)"
    echo "  4) 完整训练 ($EPOCHS epochs)"
    echo "  5) 恢复训练"
    echo "  6) 验证最佳模型"
    echo "  7) 一键运行 (检查→测试→调试→训练)"
    echo "  0) 退出"
    echo ""
    read -p "输入选项 [0-7]: " choice
    
    case $choice in
        1) check_requirements ;;
        2) check_requirements; test_data_loading ;;
        3) check_requirements; debug_train ;;
        4) check_requirements; full_train ;;
        5) check_requirements; resume_train ;;
        6) validate_checkpoint ;;
        7) 
           check_requirements
           test_data_loading
           debug_train
           full_train
           ;;
        0) print_info "已退出"; exit 0 ;;
        *) print_error "无效选项"; exit 1 ;;
    esac
}

# ==================== 主程序 ====================

# 检查必需文件是否存在
if [ ! -f "ecg_train_v48_fixed.py" ]; then
    print_error "请在包含 ecg_train_v48_fixed.py 的目录下运行此脚本"
    exit 1
fi

if [ $# -gt 0 ]; then
    case $1 in
        check) check_requirements ;;
        test) check_requirements; test_data_loading ;;
        debug) check_requirements; debug_train ;;
        train) check_requirements; full_train ;;
        resume) check_requirements; resume_train ;;
        validate) validate_checkpoint ;;
        all) check_requirements; test_data_loading; debug_train; full_train ;;
        *) 
            echo "用法: $0 [check|test|debug|train|resume|validate|all]"
            echo ""
            echo "命令说明:"
            echo "  check    - 检查环境和数据"
            echo "  test     - 测试数据加载"
            echo "  debug    - 快速调试训练"
            echo "  train    - 完整训练"
            echo "  resume   - 恢复训练"
            echo "  validate - 验证最佳模型"
            echo "  all      - 全自动流程"
            exit 1 
            ;;
    esac
else
    show_menu
fi