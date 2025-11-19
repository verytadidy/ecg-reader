#!/bin/bash
# train.sh - ECG V47 CRNN模型训练一键启动脚本

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 数据路径（请修改为您的实际 V47 数据路径）
# 注意：必须使用包含 gt_signals.json 的数据集
SIM_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization-simulations-V47"
CSV_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization/train"

# 训练参数
EPOCHS=50
BATCH_SIZE=8
LR=1e-4
NUM_WORKERS=4
TARGET_FS=500

# 关键修改：输入宽度提升至 2048 以支持时序解码
INPUT_SIZE="512 2048"  # H W 

# 输出目录
OUTPUT_DIR="./outputs"

# 损失权重（V47 新版配置）
# Seg: 负责基线、文字、OCR定位
# Signal: 负责波形数值回归 (L1 Loss)
WEIGHT_SEG=1.0
WEIGHT_SIGNAL=10.0    # 信号回归通常数值较小，给高权重以平衡

# ==================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# 检查必需文件
check_requirements() {
    print_header "环境检查"
    
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
        exit 1
    fi

    # 检测计算设备
    echo ""
    if python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
        print_info "检测到CUDA: $CUDA_VERSION"
    elif python -c "import torch; assert torch.backends.mps.is_available()" &> /dev/null; then
        print_info "检测到Apple Silicon MPS"
    else
        print_warning "仅CPU模式（训练速度较慢）"
    fi
    
    # 检查核心代码文件
    echo ""
    echo "检查代码文件..."
    REQUIRED_FILES=("ecg_dataset.py" "ecg_model.py" "ecg_loss.py" "train.py")
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "缺少文件: $file"
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
    
    # 统计样本数 (V47文件结构检查)
    # 检查是否包含 gt_signals.json
    SAMPLE_COUNT=$(find "$SIM_ROOT" -maxdepth 2 -name "*_gt_signals.json" 2>/dev/null | wc -l)
    if [ "$SAMPLE_COUNT" -eq 0 ]; then
        print_warning "未找到 *_gt_signals.json 文件，请确认 SIM_ROOT 指向的是 V47 数据集"
    else
        print_info "找到 $SAMPLE_COUNT 个有效训练样本"
    fi
    
    echo ""
}

# 测试数据加载
test_data_loading() {
    print_header "数据加载测试"
    
    echo "测试前50个样本的数据加载..."
    echo ""
    
    # 使用 ecg_dataset.py 的 main 函数进行测试
    python ecg_dataset.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --max_samples 50 \
        --batch_size 4 \
        --num_workers 0
    
    if [ $? -eq 0 ]; then
        echo ""
        print_info "数据加载测试通过！"
        echo ""
    else
        print_error "数据加载失败，请检查:"
        echo "  1. 数据路径是否正确"
        echo "  2. 数据格式是否符合V47规范 (需包含 gt_signals.json)"
        exit 1
    fi
}

# 快速调试训练
debug_train() {
    print_header "快速调试训练"
    
    echo "配置:"
    echo "  样本数: 50"
    echo "  Epochs: 3"
    echo "  Batch Size: 4"
    echo "  Workers: 0"
    echo ""
    
    DEBUG_OUTPUT="$OUTPUT_DIR/debug_run"
    mkdir -p "$DEBUG_OUTPUT"
    
    # 更新后的参数列表
    python train.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --output_dir "$DEBUG_OUTPUT" \
        --max_samples 50 \
        --epochs 3 \
        --batch_size 4 \
        --lr $LR \
        --num_workers 0 \
        --input_size $INPUT_SIZE \
        --target_fs $TARGET_FS \
        --pretrained \
        --weight_seg $WEIGHT_SEG \
        --weight_signal $WEIGHT_SIGNAL
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        print_info "调试训练完成！"
        print_info "检查点: $DEBUG_OUTPUT/checkpoint_latest.pth"
        echo ""
    else
        print_error "调试训练失败（退出码: $EXIT_CODE）"
        exit $EXIT_CODE
    fi
}

# 完整训练
full_train() {
    print_header "完整训练"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TRAIN_OUTPUT="$OUTPUT_DIR/exp_${TIMESTAMP}"
    
    echo "训练配置:"
    echo "  输出目录: $TRAIN_OUTPUT"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LR"
    echo "  Workers: $NUM_WORKERS"
    echo "  Input Size: $INPUT_SIZE"
    echo ""
    echo "损失权重:"
    echo "  分割权重: $WEIGHT_SEG"
    echo "  信号权重: $WEIGHT_SIGNAL"
    echo ""
    
    mkdir -p "$TRAIN_OUTPUT"
    
    # 更新后的参数列表
    python train.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --output_dir "$TRAIN_OUTPUT" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --num_workers $NUM_WORKERS \
        --input_size $INPUT_SIZE \
        --target_fs $TARGET_FS \
        --pretrained \
        --scheduler cosine \
        --weight_seg $WEIGHT_SEG \
        --weight_signal $WEIGHT_SIGNAL \
        --save_interval 5 \
        --log_interval 10
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        print_info "训练完成！"
        echo ""
        print_info "启动TensorBoard:"
        echo "  tensorboard --logdir $TRAIN_OUTPUT/logs --port 6006"
        echo ""
    else
        print_error "训练失败（退出码: $EXIT_CODE）"
        exit $EXIT_CODE
    fi
}

# 恢复训练
resume_train() {
    print_header "恢复训练"
    
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint_latest.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        print_error "未找到检查点文件"
        echo "请手动指定:"
        echo "  python train.py --resume /path/to/checkpoint.pth ..."
        exit 1
    fi
    
    print_info "找到检查点: $LATEST_CHECKPOINT"
    RESUME_DIR=$(dirname "$LATEST_CHECKPOINT")
    
    echo ""
    read -p "是否恢复训练？[y/N] " yn
    case $yn in
        [Yy]*)
            # 更新后的参数列表
            python train.py \
                --sim_root "$SIM_ROOT" \
                --csv_root "$CSV_ROOT" \
                --output_dir "$RESUME_DIR" \
                --resume "$LATEST_CHECKPOINT" \
                --epochs $EPOCHS \
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

# 主菜单
show_menu() {
    echo ""
    print_header "ECG V47 CRNN 模型训练"
    echo ""
    echo "数据路径:"
    echo "  仿真数据: $SIM_ROOT"
    echo "  CSV数据: $CSV_ROOT"
    echo ""
    echo "请选择操作:"
    echo "  1) 检查环境和数据"
    echo "  2) 测试数据加载"
    echo "  3) 快速调试训练（3 epochs, 50样本）"
    echo "  4) 完整训练"
    echo "  5) 恢复训练"
    echo "  6) 一键运行（检查→测试→调试→训练）"
    echo "  0) 退出"
    echo ""
    read -p "输入选项 [0-6]: " choice
    
    case $choice in
        1) check_requirements ;;
        2) check_requirements; test_data_loading ;;
        3) check_requirements; debug_train ;;
        4) check_requirements; full_train ;;
        5) check_requirements; resume_train ;;
        6) 
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
if [ ! -f "train.py" ]; then
    print_error "请在包含 train.py 的目录下运行此脚本"
    exit 1
fi

if [ $# -gt 0 ]; then
    case $1 in
        check) check_requirements ;;
        test) check_requirements; test_data_loading ;;
        debug) check_requirements; debug_train ;;
        train) check_requirements; full_train ;;
        resume) check_requirements; resume_train ;;
        all) check_requirements; test_data_loading; debug_train; full_train ;;
        *) echo "用法: $0 [check|test|debug|train|resume|all]"; exit 1 ;;
    esac
else
    show_menu
fi