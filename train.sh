#!/bin/bash
# train.sh - ECG V45模型训练一键启动脚本

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 数据路径（必须修改为您的实际路径）
SIM_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization-simulations-V45"
CSV_ROOT="/Volumes/movie/work/physionet-ecg-image-digitization/train"

# 训练参数
EPOCHS=100
BATCH_SIZE=8
LR=1e-4
NUM_WORKERS=4
TARGET_FS=500
INPUT_SIZE="512 672"  # H W

# 输出目录
OUTPUT_DIR="./outputs"

# 损失权重（V45关键配置）
WEIGHT_PAPER_SPEED=5.0  # ⭐⭐⭐⭐⭐ 关键
WEIGHT_GAIN=3.0         # ⭐⭐⭐
WEIGHT_TEXT=2.0
WEIGHT_LEAD_BASELINE=2.0

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
    
    REQUIRED_PACKAGES=("torch" "torchvision" "numpy" "opencv-python" "albumentations" "pandas" "tqdm" "tensorboard")
    MISSING_PACKAGES=()
    
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        # 默认逻辑：将横杠替换为下划线
        pkg_name=$(echo $pkg | sed 's/-/_/g')
        
        # --- 修复开始：特殊处理导入名称不一致的包 ---
        if [ "$pkg" == "opencv-python" ]; then
            import_name="cv2"
        elif [ "$pkg" == "scikit-learn" ]; then
            import_name="sklearn"  # 预防以后用到 sklearn
        elif [ "$pkg" == "Pillow" ]; then
            import_name="PIL"      # 预防以后用到 Pillow
        else
            import_name="$pkg_name"
        fi
        # --- 修复结束 ---

        # 使用正确的 import_name 进行检查
        if ! python -c "import ${import_name}" &> /dev/null; then
            MISSING_PACKAGES+=("$pkg")
            print_error "✗ 缺少依赖: $pkg (尝试导入: $import_name 失败)"
        else
            print_info "✓ $pkg"
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo ""
        print_error "缺少以下依赖包，请先安装:"
        echo "  pip install ${MISSING_PACKAGES[@]}"
        exit 1
    fi
    
    # 检查PyTorch版本
    TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null)
    print_info "PyTorch: $TORCH_VERSION"
    
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
    
    # 统计样本数
    SAMPLE_COUNT=$(find "$SIM_ROOT" -maxdepth 1 -type d -name "*_v*" 2>/dev/null | wc -l)
    print_info "找到 $SAMPLE_COUNT 个仿真样本目录"
    
    echo ""
}

# 测试数据加载
test_data_loading() {
    print_header "数据加载测试"
    
    echo "测试前50个样本的数据加载..."
    echo ""
    
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
        echo "  2. 数据格式是否符合V45规范"
        echo "  3. CSV文件是否存在"
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
    echo "  Workers: 0 (避免多进程问题)"
    echo ""
    
    DEBUG_OUTPUT="$OUTPUT_DIR/debug_run"
    mkdir -p "$DEBUG_OUTPUT"
    
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
        --use_focal_loss \
        --weight_paper_speed $WEIGHT_PAPER_SPEED \
        --weight_gain $WEIGHT_GAIN \
        --weight_text $WEIGHT_TEXT \
        --weight_lead_baseline $WEIGHT_LEAD_BASELINE
    
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
    echo "  Target FS: ${TARGET_FS}Hz"
    echo "  Input Size: $INPUT_SIZE"
    echo ""
    echo "损失权重:"
    echo "  纸速OCR: $WEIGHT_PAPER_SPEED ⭐⭐⭐⭐⭐"
    echo "  增益OCR: $WEIGHT_GAIN ⭐⭐⭐"
    echo "  导联文字: $WEIGHT_TEXT"
    echo "  细粒度基线: $WEIGHT_LEAD_BASELINE"
    echo ""
    
    # 创建输出目录
    mkdir -p "$TRAIN_OUTPUT"
    
    # 保存配置
    cat > "$TRAIN_OUTPUT/config.txt" << EOF
训练配置 - $(date)
========================
数据:
  SIM_ROOT: $SIM_ROOT
  CSV_ROOT: $CSV_ROOT

训练参数:
  EPOCHS: $EPOCHS
  BATCH_SIZE: $BATCH_SIZE
  LR: $LR
  NUM_WORKERS: $NUM_WORKERS
  TARGET_FS: $TARGET_FS
  INPUT_SIZE: $INPUT_SIZE

损失权重:
  WEIGHT_PAPER_SPEED: $WEIGHT_PAPER_SPEED
  WEIGHT_GAIN: $WEIGHT_GAIN
  WEIGHT_TEXT: $WEIGHT_TEXT
  WEIGHT_LEAD_BASELINE: $WEIGHT_LEAD_BASELINE
EOF
    
    # 启动训练
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
        --use_focal_loss \
        --scheduler cosine \
        --weight_paper_speed $WEIGHT_PAPER_SPEED \
        --weight_gain $WEIGHT_GAIN \
        --weight_text $WEIGHT_TEXT \
        --weight_lead_baseline $WEIGHT_LEAD_BASELINE \
        --save_interval 10 \
        --log_interval 10 \
        --early_stop 20
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        print_info "训练完成！"
        echo ""
        print_info "模型检查点: $TRAIN_OUTPUT/checkpoint_best.pth"
        print_info "TensorBoard日志: $TRAIN_OUTPUT/logs"
        echo ""
        print_info "启动TensorBoard查看训练曲线:"
        echo "  tensorboard --logdir $TRAIN_OUTPUT/logs --port 6006"
        echo ""
        print_info "运行推理测试:"
        echo "  python inference.py --checkpoint $TRAIN_OUTPUT/checkpoint_best.pth --image /path/to/test.png --output_dir ./results"
        echo ""
    else
        print_error "训练失败（退出码: $EXIT_CODE）"
        echo ""
        print_warning "请检查:"
        echo "  1. 显存/内存是否充足（尝试减小 BATCH_SIZE）"
        echo "  2. 数据路径是否正确"
        echo "  3. 查看错误日志: $TRAIN_OUTPUT/logs/"
        exit $EXIT_CODE
    fi
}

# 恢复训练
resume_train() {
    print_header "恢复训练"
    
    # 查找最新的检查点
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint_latest.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        print_error "未找到检查点文件"
        echo "请手动指定检查点路径:"
        echo "  python train.py --resume /path/to/checkpoint.pth ..."
        exit 1
    fi
    
    print_info "找到检查点: $LATEST_CHECKPOINT"
    
    RESUME_DIR=$(dirname "$LATEST_CHECKPOINT")
    
    echo ""
    read -p "是否从此检查点恢复训练？[y/N] " yn
    case $yn in
        [Yy]*)
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
                --target_fs $TARGET_FS
            
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
    print_header "ECG V45 模型训练"
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
            resume_train
            ;;
        6)
            # 一键运行
            check_requirements
            
            echo ""
            read -p "环境检查完成，继续测试数据加载？[Y/n] " yn
            case $yn in
                [Nn]*)
                    print_info "跳过数据测试"
                    ;;
                *)
                    test_data_loading
                    ;;
            esac
            
            echo ""
            read -p "是否运行快速调试训练？[Y/n] " yn
            case $yn in
                [Nn]*)
                    print_info "跳过调试训练"
                    ;;
                *)
                    debug_train
                    ;;
            esac
            
            echo ""
            read -p "是否开始完整训练？[Y/n] " yn
            case $yn in
                [Nn]*)
                    print_info "已取消完整训练"
                    exit 0
                    ;;
                *)
                    full_train
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

# 显示脚本信息
cat << "EOF"
  _____ ____ ____   __     _____ ____  
 | ____/ ___/ ___| /  \   |_   _|  _ \ 
 |  _|| |  | |  _ | () |    | | | |_) |
 | |__| |__| |_| | \__/     | | |  _ < 
 |_____\____\____|           |_| |_| \_\
                                         
 ECG V45 渐进式导联定位模型训练
 Version: 1.0
 
EOF

# 检查是否在正确目录
if [ ! -f "train.py" ]; then
    print_error "请在包含 train.py 的目录下运行此脚本"
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
        resume)
            check_requirements
            resume_train
            ;;
        all)
            check_requirements
            test_data_loading
            debug_train
            full_train
            ;;
        *)
            echo "用法: $0 [check|test|debug|train|resume|all]"
            echo ""
            echo "命令说明:"
            echo "  check  - 检查环境和数据"
            echo "  test   - 测试数据加载"
            echo "  debug  - 快速调试训练"
            echo "  train  - 完整训练"
            echo "  resume - 恢复训练"
            echo "  all    - 一键运行所有步骤"
            exit 1
            ;;
    esac
else
    # 交互式菜单
    show_menu
fi