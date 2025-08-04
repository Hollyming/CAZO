#!/bin/bash

# Hessian Low-Rank Property Verification Experiment Runner
# Hessian低秩特性验证实验运行脚本

set -e  # 遇到错误时退出

# 默认配置
DATA_CORRUPTION="/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/datasets/imagenet-c/imagenet-c"
OUTPUT_DIR="./hessian_experiments"
BATCH_SIZE=32
MAX_BATCHES=100
GPU=0
ADAPTER_LAYERS="3"
REDUCTION_FACTOR=384
SEED=42

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 帮助信息
show_help() {
    cat << EOF
Hessian低秩特性验证实验运行脚本

用法:
    $0 [选项]

选项:
    -d, --data-corruption PATH    ImageNet-C数据集路径 (默认: $DATA_CORRUPTION)
    -o, --output DIR             输出目录 (默认: $OUTPUT_DIR)
    -b, --batch-size SIZE        批次大小 (默认: $BATCH_SIZE)
    -m, --max-batches NUM        最大分析批次数 (默认: $MAX_BATCHES)
    -g, --gpu GPU_ID             GPU设备ID (默认: $GPU)
    -a, --adapter-layers LAYERS  Adapter层位置 (默认: $ADAPTER_LAYERS)
    -r, --reduction-factor NUM   Adapter降维因子 (默认: $REDUCTION_FACTOR)
    -s, --seed SEED              随机种子 (默认: $SEED)
    --quick                      快速模式 (batch_size=8, max_batches=20)
    --full                       完整模式 (batch_size=32, max_batches=100)
    --dry-run                    试运行模式，只检查环境不实际运行
    -h, --help                   显示此帮助信息

示例:
    # 基础运行
    $0

    # 指定数据路径
    $0 -d /path/to/imagenet-c

    # 快速验证
    $0 --quick

    # 完整实验
    $0 --full -o ./full_experiment

    # 多层adapter
    $0 -a "9,10,11" -r 8

    # 试运行
    $0 --dry-run
EOF
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        print_error "Python未安装或不在PATH中"
        exit 1
    fi
    
    # 检查必要的包
    local required_packages=("torch" "timm" "scipy" "sklearn" "matplotlib" "seaborn" "numpy")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_error "缺少必要的Python包: ${missing_packages[*]}"
        print_info "请安装: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    print_success "Python环境检查通过"
}

# 检查数据集
check_dataset() {
    print_info "检查ImageNet-C数据集..."
    
    if [ ! -d "$DATA_CORRUPTION" ]; then
        print_error "ImageNet-C数据集目录不存在: $DATA_CORRUPTION"
        exit 1
    fi
    
    local target_dir="$DATA_CORRUPTION/gaussian_noise/5"
    if [ ! -d "$target_dir" ]; then
        print_error "找不到目标数据目录: $target_dir"
        print_info "请确保ImageNet-C数据集结构正确："
        print_info "  $DATA_CORRUPTION/"
        print_info "    └── gaussian_noise/"
        print_info "        └── 5/"
        exit 1
    fi
    
    # 检查数据文件数量
    local file_count=$(find "$target_dir" -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" | wc -l)
    if [ "$file_count" -eq 0 ]; then
        print_warning "在 $target_dir 中未找到图像文件"
    else
        print_success "找到 $file_count 个图像文件"
    fi
    
    print_success "数据集检查通过"
}

# 检查GPU
check_gpu() {
    print_info "检查GPU环境..."
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        print_info "检测到 $gpu_count 个GPU设备"
        
        if [ "$GPU" -ge "$gpu_count" ]; then
            print_warning "指定的GPU ID ($GPU) 超出可用范围 (0-$((gpu_count-1)))"
            print_info "将使用GPU 0"
            GPU=0
        fi
        
        # 检查GPU内存
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=$GPU)
        print_info "GPU $GPU 内存: ${gpu_memory}MB"
        
        if [ "$gpu_memory" -lt 4000 ]; then
            print_warning "GPU内存较少，建议减小batch_size"
        fi
    else
        print_warning "未检测到NVIDIA GPU，将使用CPU运行 (可能很慢)"
        GPU=-1  # 使用CPU
    fi
}

# 创建输出目录
prepare_output_dir() {
    print_info "准备输出目录: $OUTPUT_DIR"
    
    mkdir -p "$OUTPUT_DIR"
    
    if [ ! -w "$OUTPUT_DIR" ]; then
        print_error "输出目录不可写: $OUTPUT_DIR"
        exit 1
    fi
    
    print_success "输出目录准备完成"
}

# 运行实验
run_experiment() {
    print_info "开始运行Hessian低秩特性验证实验"
    print_info "配置参数:"
    print_info "  数据路径: $DATA_CORRUPTION"
    print_info "  输出目录: $OUTPUT_DIR"
    print_info "  批次大小: $BATCH_SIZE"
    print_info "  最大批次: $MAX_BATCHES"
    print_info "  GPU设备: $GPU"
    print_info "  Adapter层: $ADAPTER_LAYERS"
    print_info "  降维因子: $REDUCTION_FACTOR"
    print_info "  随机种子: $SEED"
    
    # 构建Python命令（执行当前目录下的脚本）
    local python_cmd="python ./run_hessian_experiment.py"
    python_cmd="$python_cmd --data_corruption '$DATA_CORRUPTION'"
    python_cmd="$python_cmd --output '$OUTPUT_DIR'"
    python_cmd="$python_cmd --batch_size $BATCH_SIZE"
    python_cmd="$python_cmd --max_batches $MAX_BATCHES"
    python_cmd="$python_cmd --adapter_layers '$ADAPTER_LAYERS'"
    python_cmd="$python_cmd --reduction_factor $REDUCTION_FACTOR"
    python_cmd="$python_cmd --seed $SEED"
    python_cmd="$python_cmd --workers 4"
    
    if [ "$GPU" -ge 0 ]; then
        python_cmd="$python_cmd --gpu $GPU"
    fi
    
    print_info "执行命令: $python_cmd"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 运行实验
    if eval "$python_cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        print_success "实验完成！用时: ${minutes}分${seconds}秒"
        
        # 查找最新的实验结果目录
        local latest_dir=$(find "$OUTPUT_DIR" -type d -name "hessian_tta_experiment_*" | sort | tail -1)
        if [ -n "$latest_dir" ]; then
            print_success "实验结果保存在: $latest_dir"
            print_info "主要文件:"
            print_info "  📊 可视化图表: $latest_dir/figures/"
            print_info "  📄 实验报告: $latest_dir/EXPERIMENT_REPORT.md"
            print_info "  📈 详细数据: $latest_dir/results/"
            
            # 如果有markdown报告，显示关键结果
            local md_report="$latest_dir/EXPERIMENT_REPORT.md"
            if [ -f "$md_report" ]; then
                print_info "实验摘要:"
                echo "----------------------------------------"
                head -20 "$md_report" | tail -15
                echo "----------------------------------------"
            fi
        fi
    else
        print_error "实验失败！请检查错误信息"
        exit 1
    fi
}

# 试运行模式
dry_run() {
    print_info "=== 试运行模式 ==="
    check_python_env
    check_dataset
    check_gpu
    prepare_output_dir
    
    print_success "所有检查通过！"
    print_info "实际运行命令："
    echo "  $0 $(echo "$@" | sed 's/--dry-run//')"
}

# 解析命令行参数
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-corruption)
            DATA_CORRUPTION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--max-batches)
            MAX_BATCHES="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -a|--adapter-layers)
            ADAPTER_LAYERS="$2"
            shift 2
            ;;
        -r|--reduction-factor)
            REDUCTION_FACTOR="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --quick)
            BATCH_SIZE=8
            MAX_BATCHES=20
            print_info "启用快速模式: batch_size=$BATCH_SIZE, max_batches=$MAX_BATCHES"
            shift
            ;;
        --full)
            BATCH_SIZE=32
            MAX_BATCHES=100
            print_info "启用完整模式: batch_size=$BATCH_SIZE, max_batches=$MAX_BATCHES"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主执行流程
main() {
    echo "=================================="
    echo "🔬 Hessian低秩特性验证实验"
    echo "=================================="
    
    if [ "$DRY_RUN" = true ]; then
        dry_run
    else
        check_python_env
        check_dataset
        check_gpu
        prepare_output_dir
        run_experiment
    fi
}

# 运行主函数
main 