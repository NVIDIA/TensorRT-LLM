#!/bin/bash

# ============================================================================
# Results Parsing Only Script
# ============================================================================

# Configuration (match your main script)
EG_DIR=/tmp/lora-eg
TP_SIZES=(1 2 4)

echo ""
echo "###################################################################"
echo "#                        RESULTS SUMMARY                         #"
echo "###################################################################"
echo ""

parse_results() {
    local tp=$1
    local scenario=$2
    local log_path="$3"

    echo "TP=${tp} - ${scenario}:"

    if [ ${tp} -eq 1 ]; then
        # Single GPU logs
        if [ -f "${log_path}/output.log" ]; then
            grep -E "(token_throughput|seq_throughput|total_latency|avg_sequence_latency|avg_time_to_first_token|avg_inter_token_latency)" "${log_path}/output.log" 2>/dev/null || echo "  No metrics found"
        else
            echo "  Log file not found: ${log_path}/output.log"
        fi
    else
        # Multi-GPU MPI logs
        if [ -d "${log_path}" ]; then
            find ${log_path} -name "stdout" -exec grep -E "(token_throughput|seq_throughput|total_latency|avg_sequence_latency|avg_time_to_first_token|avg_inter_token_latency)" {} \; 2>/dev/null || echo "  No metrics found"
        else
            echo "  Log directory not found: ${log_path}"
        fi
    fi
    echo ""
}

for TP in ${TP_SIZES[@]}; do
    echo "========== TP=${TP} Results =========="
    parse_results ${TP} "Scenario 1 (No LoRA Engine)" "${EG_DIR}/logs/scenario1-no-lora-engine-tp-${TP}"
    parse_results ${TP} "Scenario 2 (LoRA Engine, No LoRA Requests)" "${EG_DIR}/logs/scenario2-lora-engine-no-lora-requests-tp-${TP}"
    parse_results ${TP} "Scenario 3 (LoRA Engine, LoRA Requests)" "${EG_DIR}/logs/scenario3-lora-engine-lora-requests-tp-${TP}"
done

echo ""
echo "###################################################################"
echo "#                    RESULTS PARSING COMPLETE!                   #"
echo "###################################################################"
echo ""
