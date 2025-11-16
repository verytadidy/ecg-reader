#!/bin/bash
# train.sh - ECG模型训练一键启动脚本

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 数据路径（必须修改）
SIM_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization-simulations-V37"
CSV_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization/train"

# 训练参数
EPOCHS=20
BATCH_SIZE=4
LR=1e-4
NUM_WORKERS=4
TARGET_FS=500  # 所有数据重采样到500Hz

# 输出目录
OUTPUT_DIR="./experiments"

# ==================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}======================================================================${NC}"
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
    print_header "检查环境"
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        print_error "未找到python"
        exit 1
    fi
    print_info "python: $(python --version)"
    
    # 检查PyTorch
    if ! python -c "import torch" &> /dev/null; then
        print_error "未安装PyTorch"
        echo "安装命令: pip install torch torchvision torchaudio"
        exit 1
    fi
    print_info "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    
    # 检查数据目录
    if [ ! -d "$SIM_ROOT" ]; then
        print_error "仿真数据目录不存在: $SIM_ROOT"
        exit 1
    fi
    print_info "仿真数据: $SIM_ROOT"
    
    if [ ! -d "$CSV_ROOT" ]; then
        print_error "CSV数据目录不存在: $CSV_ROOT"
        exit 1
    fi
    print_info "CSV数据: $CSV_ROOT"
    
    # 统计样本数
    SAMPLE_COUNT=$(find "$SIM_ROOT" -maxdepth 1 -type d | wc -l)
    print_info "找到 $SAMPLE_COUNT 个仿真样本"
    
    echo ""
}

# 测试数据加载
test_data_loading() {
    print_header "测试数据加载"
    
    echo "正在测试前100个样本..."
    python production_dataset.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --max_samples 100
    
    if [ $? -eq 0 ]; then
        print_info "数据加载测试通过"
        echo ""
    else
        print_error "数据加载失败，请检查数据"
        exit 1
    fi
}

# 快速调试训练
debug_train() {
    print_header "快速调试训练（3 epochs, 100样本）"
    
    python production_trainer.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --target_fs $TARGET_FS \
        --batch_size 2 \
        --num_workers 0 \
        --output_dir "$OUTPUT_DIR" \
        --debug
    
    if [ $? -eq 0 ]; then
        print_info "调试训练完成"
        echo ""
    else
        print_error "调试训练失败"
        exit 1
    fi
}

# 完整训练
full_train() {
    print_header "开始完整训练"
    
    echo "训练配置:"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Learning Rate: $LR"
    echo "  Workers: $NUM_WORKERS"
    echo "  Target FS: ${TARGET_FS}Hz（所有数据重采样）"
    echo ""
    
    # 检测设备
    if python -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
        DEVICE="CUDA"
        print_info "检测到GPU，使用CUDA加速"
    elif python -c "import torch; assert torch.backends.mps.is_available()" &> /dev/null; then
        DEVICE="MPS"
        print_info "检测到Apple Silicon，使用MPS加速"
    else
        DEVICE="CPU"
        print_warning "使用CPU训练（速度较慢）"
    fi
    echo ""
    
    # 启动训练
    python production_trainer.py \
        --sim_root "$SIM_ROOT" \
        --csv_root "$CSV_ROOT" \
        --target_fs $TARGET_FS \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --num_workers $NUM_WORKERS \
        --output_dir "$OUTPUT_DIR" \
        --pretrained \
        --save_freq 10 \
        --patience 15
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_info "训练完成！"
        echo ""
        print_info "检查点保存在: $OUTPUT_DIR/run_*/checkpoints/"
        print_info "启动TensorBoard查看训练曲线:"
        echo "  tensorboard --logdir $OUTPUT_DIR/run_*/tensorboard"
    else
        print_error "训练失败（退出码: $EXIT_CODE）"
        exit $EXIT_CODE
    fi
}

# 主菜单
show_menu() {
    echo ""
    print_header "ECG重建模型训练"
    echo ""
    echo "请选择操作:"
    echo "  1) 检查环境"
    echo "  2) 测试数据加载"
    echo "  3) 快速调试（3 epochs, 100样本）"
    echo "  4) 完整训练"
    echo "  5) 一键运行（检查+测试+调试+训练）"
    echo "  0) 退出"
    echo ""
    read -p "输入选项 [0-5]: " choice
    
    case $choice in
        1)
            check_requirements
            ;;
        2)
            check_requirements
            test_data_loading
            ;;
        3)
            check_requirements
            debug_train
            ;;
        4)
            check_requirements
            full_train
            ;;
        5)
            check_requirements
            test_data_loading
            
            echo ""
            read -p "数据测试通过，是否继续调试训练？[y/N] " yn
            case $yn in
                [Yy]*)
                    debug_train
                    ;;
                *)
                    print_info "跳过调试训练"
                    ;;
            esac
            
            echo ""
            read -p "是否开始完整训练？[y/N] " yn
            case $yn in
                [Yy]*)
                    full_train
                    ;;
                *)
                    print_info "已取消"
                    exit 0
                    ;;
            esac
            ;;
        0)
            print_info "已退出"
            exit 0
            ;;
        *)
            print_error "无效选项"
            exit 1
            ;;
    esac
}

# ==================== 主程序 ====================

# 检查是否在正确目录
if [ ! -f "production_trainer.py" ]; then
    print_error "请在包含 production_trainer.py 的目录下运行此脚本"
    exit 1
fi

# 如果提供了命令行参数，直接执行
if [ $# -gt 0 ]; then
    case $1 in
        check)
            check_requirements
            ;;
        test)
            check_requirements
            test_data_loading
            ;;
        debug)
            check_requirements
            debug_train
            ;;
        train)
            check_requirements
            full_train
            ;;
        *)
            echo "用法: $0 [check|test|debug|train]"
            exit 1
            ;;
    esac
else
    # 交互式菜单
    show_menu
fi