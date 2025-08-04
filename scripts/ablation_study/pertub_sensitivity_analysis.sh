#!/bin/bash

# Extract results from completed pertub sensitivity experiments (both accuracy and ECE)
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/cazo_pertub_sensitivity"
SUMMARY_LOG="${PROJECT_ROOT}/logs_new/cazo_pertub_sensitivity/cazo_pertub_sensitivity_summary_fixed.log"

echo "Extracting results from completed pertub sensitivity experiments..." > ${SUMMARY_LOG}
echo "Format: pertub=X: accuracy=Y.YY%, ece=Z.ZZ%" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Pertub values from the original experiment
pertub_values=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30)

for pertub in "${pertub_values[@]}"; do
    echo "Checking pertub=${pertub}..."
    
    # 构建正确的路径
    tag="_pertub_sensitivity_${pertub}_bs64_lr0.01_adapter3_rf384_parallel_eps0.1_nu0.8"
    
    LOG_FILE="${BASE_OUTPUT_DIR}/cazo${tag}/"*"-log.txt"
    
    echo "  Searching: ${LOG_FILE}"
    
    if ls ${LOG_FILE} 1> /dev/null 2>&1; then
        # 提取准确率和ECE结果
        ACC_RESULT=$(tail -30 ${LOG_FILE} | grep -E "(mean acc|Top-1 Accuracy)" | tail -1)
        ECE_RESULT=$(tail -30 ${LOG_FILE} | grep -E "(mean ece|ECE)" | tail -1)
        
        # 从结果中提取数值
        ACC_VALUE=$(echo "$ACC_RESULT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        ECE_VALUE=$(echo "$ECE_RESULT" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        if [[ -n "$ACC_VALUE" && -n "$ECE_VALUE" ]]; then
            COMBINED_RESULT="accuracy=${ACC_VALUE}%, ece=${ECE_VALUE}%"
            echo "pertub=${pertub}: ${COMBINED_RESULT}" >> ${SUMMARY_LOG}
            echo "  ✓ Found result: ${COMBINED_RESULT}"
        else
            echo "pertub=${pertub}: accuracy=${ACC_VALUE:-N/A}%, ece=${ECE_VALUE:-N/A}%" >> ${SUMMARY_LOG}
            echo "  ⚠ Partial result: ACC=${ACC_VALUE:-N/A}, ECE=${ECE_VALUE:-N/A}"
        fi
    else
        echo "  ✗ Log file not found"
        echo "pertub=${pertub}: accuracy=N/A%, ece=N/A%" >> ${SUMMARY_LOG}
    fi
done

echo "" >> ${SUMMARY_LOG}
echo "Results extraction completed at $(date)" >> ${SUMMARY_LOG}

echo ""
echo "Results summary saved to: ${SUMMARY_LOG}"