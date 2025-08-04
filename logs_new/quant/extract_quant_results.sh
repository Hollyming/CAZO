#!/bin/bash

# Extract results from quantized model experiments
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/quant"
BASE_LOG_DIR="${PROJECT_ROOT}/logs_new/quant"
RESULTS_FILE="${BASE_LOG_DIR}/quant_experiments_results.csv"

# Create CSV header
echo "algorithm,accuracy,ece,model_type,gpu_used" > ${RESULTS_FILE}

algorithms=(no_adapt lame t3a foa zo_base cazo)

for algorithm in "${algorithms[@]}"; do
    echo "Extracting results for ${algorithm}..."
    
    # Find log files
    LOG_PATTERN="${BASE_OUTPUT_DIR}/${algorithm}/${algorithm}_quant8_bs64"*"/"*"-log.txt"
    
    if ls ${LOG_PATTERN} 1> /dev/null 2>&1; then
        LOG_FILE=$(ls ${LOG_PATTERN} | head -1)
        
        # Extract final results
        ACC_LINE=$(tail -30 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
        ECE_LINE=$(tail -30 ${LOG_FILE} | grep -E "(mean ece|ECE)" | tail -1)
        
        ACC_VALUE=$(echo "$ACC_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        ECE_VALUE=$(echo "$ECE_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        # Try to extract GPU info from log
        GPU_INFO=$(grep -E "CUDA_VISIBLE_DEVICES|Starting.*on GPU" ${LOG_FILE} | head -1 | grep -oE '[0-9]+' | head -1)
        
        echo "${algorithm},${ACC_VALUE:-N/A},${ECE_VALUE:-N/A},quantized_8bit,${GPU_INFO:-N/A}" >> ${RESULTS_FILE}
        echo "  ✓ ${algorithm}: ACC=${ACC_VALUE:-N/A}, ECE=${ECE_VALUE:-N/A}, GPU=${GPU_INFO:-N/A}"
    else
        echo "  ✗ No log file found for ${algorithm}"
        echo "${algorithm},N/A,N/A,quantized_8bit,N/A" >> ${RESULTS_FILE}
    fi
done

echo ""
echo "Quantized results extracted to: ${RESULTS_FILE}"
echo "Preview:"
cat ${RESULTS_FILE}
