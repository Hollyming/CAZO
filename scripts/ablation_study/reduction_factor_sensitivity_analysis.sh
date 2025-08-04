#!/bin/bash

# Extract results from completed experiments (both accuracy and ECE)
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/reduction_factor_sensitivity"
SUMMARY_LOG="${PROJECT_ROOT}/logs_new/reduction_factor_sensitivity/reduction_factor_sensitivity_summary_fixed.log"

echo "Extracting results from completed experiments..." > ${SUMMARY_LOG}
echo "Format: reduction_factor=X (bottleneck=Y): accuracy=Z.ZZ%, ece=W.WW%" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

reduction_factors=(384 256 192 128 96 48)

for reduction_factor in "${reduction_factors[@]}"; do
    bottleneck_dim=$((768 / reduction_factor))
    
    # 构建正确的路径
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/reduction_factor_${reduction_factor}"
    tag="_reduction_factor_sensitivity_${reduction_factor}_bs64_lr0.01_pertub20_adapter3_parallel_eps0.1_nu0.8"
    
    LOG_FILE="${OUTPUT_DIR}/cazo${tag}/"*"-log.txt"
    
    echo "Checking: ${LOG_FILE}"
    
    if ls ${LOG_FILE} 1> /dev/null 2>&1; then
        # 提取准确率结果
        ACC_RESULT=$(tail -30 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
        ECE_RESULT=$(tail -30 ${LOG_FILE} | grep -E "(mean ece|ECE)" | tail -1)
        
        # 从结果中提取数值
        ACC_VALUE=$(echo "$ACC_RESULT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        ECE_VALUE=$(echo "$ECE_RESULT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        if [[ -n "$ACC_VALUE" && -n "$ECE_VALUE" ]]; then
            COMBINED_RESULT="accuracy=${ACC_VALUE}%, ece=${ECE_VALUE}%"
            echo "reduction_factor=${reduction_factor} (bottleneck=${bottleneck_dim}): ${COMBINED_RESULT}" >> ${SUMMARY_LOG}
            echo "✓ Found result for reduction_factor=${reduction_factor}: ${COMBINED_RESULT}"
        else
            echo "reduction_factor=${reduction_factor} (bottleneck=${bottleneck_dim}): accuracy=${ACC_VALUE:-N/A}%, ece=${ECE_VALUE:-N/A}%" >> ${SUMMARY_LOG}
            echo "⚠ Partial result for reduction_factor=${reduction_factor}: ACC=${ACC_VALUE:-N/A}, ECE=${ECE_VALUE:-N/A}"
        fi
    else
        echo "✗ Log file not found for reduction_factor=${reduction_factor}"
        echo "  Searched: ${LOG_FILE}"
        echo "reduction_factor=${reduction_factor} (bottleneck=${bottleneck_dim}): accuracy=N/A%, ece=N/A%" >> ${SUMMARY_LOG}
    fi
done

echo "" >> ${SUMMARY_LOG}
echo "Results extraction completed at $(date)" >> ${SUMMARY_LOG}

echo ""
echo "Results summary saved to: ${SUMMARY_LOG}"