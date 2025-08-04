#!/bin/bash

# Extract results from completed adapter layer sensitivity experiments (both accuracy and ECE)
PROJECT_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_2/user_mingjzhang/Workspace/CAZO
BASE_OUTPUT_DIR="${PROJECT_ROOT}/outputs_new/adapter_layer_sensitivity"
SUMMARY_LOG="${PROJECT_ROOT}/logs_new/adapter_layer_sensitivity/adapter_layer_sensitivity_summary_fixed.log"

echo "Extracting results from completed adapter layer sensitivity experiments..." > ${SUMMARY_LOG}
echo "Format: adapter_layer=X (transformer_block=Y): accuracy=Z.ZZ%, ece=W.WW%" >> ${SUMMARY_LOG}
echo "Adapter layers: 0-11 correspond to ViT-Base/16 Transformer Blocks: 1-12" >> ${SUMMARY_LOG}
echo "============================================" >> ${SUMMARY_LOG}

# Adapter layer values from the original experiment (0-11)
adapter_layers=(0 1 2 3 4 5 6 7 8 9 10 11)

for adapter_layer in "${adapter_layers[@]}"; do
    transformer_block=$((adapter_layer + 1))
    echo "Checking adapter_layer=${adapter_layer} (Transformer Block ${transformer_block})..."
    
    # 构建正确的路径
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/adapter_layer_${adapter_layer}"
    tag="_adapter_layer_sensitivity_${adapter_layer}_bs64_lr0.01_pertub20_rf384_parallel_eps0.1_nu0.8"
    
    LOG_FILE="${OUTPUT_DIR}/cazo${tag}/"*"-log.txt"
    
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
            echo "adapter_layer=${adapter_layer} (transformer_block=${transformer_block}): ${COMBINED_RESULT}" >> ${SUMMARY_LOG}
            echo "  ✓ Found result: ${COMBINED_RESULT}"
        else
            echo "adapter_layer=${adapter_layer} (transformer_block=${transformer_block}): accuracy=${ACC_VALUE:-N/A}%, ece=${ECE_VALUE:-N/A}%" >> ${SUMMARY_LOG}
            echo "  ⚠ Partial result: ACC=${ACC_VALUE:-N/A}, ECE=${ECE_VALUE:-N/A}"
        fi
    else
        echo "  ✗ Log file not found"
        echo "adapter_layer=${adapter_layer} (transformer_block=${transformer_block}): accuracy=N/A%, ece=N/A%" >> ${SUMMARY_LOG}
    fi
done

echo "" >> ${SUMMARY_LOG}
echo "Results extraction completed at $(date)" >> ${SUMMARY_LOG}

echo ""
echo "Results summary saved to: ${SUMMARY_LOG}"