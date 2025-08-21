#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
SERVER_IPADDR=${TRITONSERVER_IPADDR:=localhost}
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
BACKEND_ROOT=${BACKEND_ROOT:="/opt/tritonserver/tensorrtllm_backend"}
DATASET="$PWD/simple_data.json"
TOOLS_DIR=${BACKEND_ROOT}/tools
STREAM_DIR=${BACKEND_ROOT}/inflight_batcher_llm/client
MODEL_DIR="$PWD/triton_model_repo"
SERVER=/opt/tritonserver/bin/tritonserver
TOKENIZER_DIR=${BACKEND_ROOT}/ci/L0_backend_trtllm/tokenizer
BASE_DIR=${BACKEND_ROOT}/ci/L0_backend_trtllm
BASE_METRICS_VERIFICATION_TEST=base_metrics_verification_tests.py
BASE_METRICS_VERIFICATION_TEST_NAME=base_metrics_verification_tests
BASE_METRICS_VERIFICATION_LOG="base_metrics_verification.log"
CUSTOM_METRICS_VERIFICATION_TEST=custom_metrics_verification_tests.py
CUSTOM_METRICS_VERIFICATION_LOG="custom_metrics_verification.log"
SERVER_PID=0
SLEEP_DURATION=3

# Force environment to use python version 3
apt update -q=2 \
    && apt install -y python-is-python3

# Helpers ===============================
function replace_config_tags {
  tag_to_replace="${1}"
  new_value="${2}"
  config_file_path="${3}"
  sed -i "s|${tag_to_replace}|${new_value}|g" ${config_file_path}

}

function run_server {
  SERVER_ARGS="${1}"
  python3 ${BACKEND_ROOT}/scripts/launch_triton_server.py ${SERVER_ARGS} > ${SERVER_LOG} 2>&1 &
  sleep 2 # allow time to obtain the pid(s)
  # Read PIDs into an array, trimming whitespaces
  readarray -t SERVER_PID < <(pgrep -s 0 "tritonserver")
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local wait_time_secs="${1:-30}"; shift
    local spids=("$@");

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        # Multi-GPU will spawn multiple pids
        for pid in "${spids[@]}"; do
            if ! kill -0 $pid > /dev/null 2>&1; then
                echo "=== Server not running."
                WAIT_RET=1
                return
            fi
        done

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} ${SERVER_IPADDR}:8000/v2/health/ready`
        set -e
        if [ "$code" == "200" ]; then
            code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":1}' localhost:8000/v2/logging`
            assert_curl_success "Failed to change log settings necessary for verification" ${BASH_LINENO}
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

function reset_model_repo {
    rm -rf triton_model_repo/
    mkdir ${MODEL_DIR}
}

function kill_server {
    pgrep tritonserver | xargs kill -SIGINT
}

function wait_for_server_terminated {
    local wait_time_secs="${1:-30}"; shift
    local spids=("$@");
    for pid in "${spids[@]}"; do
        WAIT_RET=1
        echo "Waiting for proc ${pid} to terminate..."
        local wait_secs=$wait_time_secs
        until test $wait_secs -eq 0 ; do
            if ! (kill -0 $pid) > /dev/null 2>&1; then
                WAIT_RET=0
                break
            fi
            sleep 1
            ((wait_secs--));
        done
        if [ "$WAIT_RET" != "0" ]; then
            # Cleanup
            kill $SERVER_PID > /dev/null 2>&1 || true
            echo -e "\n***\n*** Failed to wait for server to terminated $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi
    done
    ps aux
    if pgrep --runstates R,S,D,I -x "trtllmExecutorW" > /dev/null; then
        echo -e "Worker process still exists - failed to terminate"
        exit 1
    fi
}

function assert_curl_success {
  message="${1}"
  original_line_no="${2}"
  if [ "$code" != "200" ]; then
    cat ./curl.out
    cat ${SERVER_LOG}
    echo -e "\n***\n*** ${message} : line ${original_line_no}\n***"
    RET=1
    return 1
  fi
  return 0
}

# =======================================

prerun_kill_triton_server () {
    pkill -9 -f trtllmExecutorWorker || true
    pkill -9 -f tritonserver
}

# Kill titonserver if it is still pending from previous test
prerun_kill_triton_server || true

rm -f *.log *.out *.txt
# Generate TRT_LLM engines and install dependencies
source ./generate_engines.sh
pip3 install --upgrade tritonclient[all] pandas tabulate

export AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)

RET=0

NUM_GPUS_TO_TEST=("1" "2" "4")
for NUM_GPU in "${NUM_GPUS_TO_TEST[@]}"; do
    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPU" ]; then
        break
    fi

    SERVER_ARGS="--world_size=${NUM_GPU} --model_repo=${MODEL_DIR}"

    reset_model_repo

    cp -r ${BACKEND_ROOT}/all_models/inflight_batcher_llm/* ${MODEL_DIR}
    rm -rf ${MODEL_DIR}/tensorrt_llm_bls
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/ensemble/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${preprocessing_instance_count}' '1' "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${max_queue_size}' '0' "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${max_queue_delay_microseconds}' '50000' "${MODEL_DIR}/preprocessing/config.pbtxt"
    replace_config_tags '${decoupled_mode}' 'False' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${max_queue_size}' "0" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${batching_strategy}' 'INVALID' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${engine_dir}' "${MODEL_DIR}/tensorrt_llm/1/inflight_${NUM_GPU}_gpu/" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${max_queue_delay_microseconds}' "50000" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${triton_backend}' "tensorrtllm" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${encoder_input_features_data_type}' "TYPE_FP16" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${prompt_embedding_table_data_type}' 'TYPE_FP16' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_DIR}/postprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_DIR}/postprocessing/config.pbtxt"
    replace_config_tags '${postprocessing_instance_count}' '1' "${MODEL_DIR}/postprocessing/config.pbtxt"
    replace_config_tags '${logits_datatype}' 'TYPE_FP32' "${MODEL_DIR}/ensemble/config.pbtxt"
    replace_config_tags '${logits_datatype}' 'TYPE_FP32' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    # Copy the engine and place it into the model folder
    cp -r ${BASE_DIR}/engines/inflight_${NUM_GPU}_gpu/ triton_model_repo/tensorrt_llm/1

    # Invalid GPT model Type
    SERVER_LOG="./${NUM_GPU}gpu_invalid_batch_strat.log"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}

    # Expect invalid GPT model type error to be gracefully handled
    if [ `grep -c "Invalid gpt_model_type" $SERVER_LOG` == "0" ]; then
        echo -e "\n***\n*** GPT model type error not handled gracefully: line ${LINENO}\n***"
        cat $SERVER_LOG
        exit 1
    fi

    wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}

    # inflight batching OFF (V1)
    # streaming OFF
    SERVER_LOG="./${NUM_GPU}gpu_v1_no_streaming_server.log"
    replace_config_tags 'INVALID' 'V1' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}

    # Expect invalid GPT model type error to be gracefully handled
    if [ `grep -c "Static batching type is deprecated" $SERVER_LOG` == "0" ]; then
        echo -e "\n***\n*** GPT model type error not handled gracefully: line ${LINENO}\n***"
        cat $SERVER_LOG
        exit 1
    fi

    # inflight batching ON
    # streaming OFF
    SERVER_LOG="./${NUM_GPU}gpu_IFB_no_streaming_server.log"
    replace_config_tags 'V1' 'inflight_fused_batching' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set -e
    python3 ${TOOLS_DIR}/inflight_batcher_llm/benchmark_core_model.py \
        --max-input-len=500 \
        dataset --dataset=${DATASET} \
        --tokenizer-dir=${TOKENIZER_DIR}

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing inflight batching benchmark_core_model test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    set -e
    python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_test.py \
        --max-input-len=500 \
        --dataset=${DATASET}

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing inflight batching end-to-end test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    # Make sure the metrics is retrieved after the server has updated the metrics internally
    sleep ${SLEEP_DURATION}
    curl localhost:8002/metrics -o ${NUM_GPU}gpu_IFB_no_stream_metrics.out

    kill_server
    wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}

    # Start a clean server to verify token metrics are being
    # reported correctly
    SERVER_LOG="./${NUM_GPU}gpu_token_metrics.log"
    replace_config_tags 'decoupled: False' 'decoupled: True' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    set -e

    #Based on prompt below
    export STREAM_INPUT_SIZE=3
    export STREAM_OUTPUT_SIZE=50
    python3 ${STREAM_DIR}/end_to_end_grpc_client.py \
        --prompt="My name is" \
        --streaming \
        -o=$STREAM_OUTPUT_SIZE

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing inflight batching end-to-end test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
        RET=1
    fi

    # Make sure the metrics is retrieved after the server has updated the metrics internally
    sleep ${SLEEP_DURATION}
    curl localhost:8002/metrics -o end_to_end_token_metrics.out

    set +e
    kill_server
    wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    replace_config_tags 'decoupled: True' 'decoupled: False' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    # Start a clean server to verify base metrics are being
    # reported correctly
    SERVER_LOG="./${NUM_GPU}gpu_IFB_no_streaming_base_metrics.log"
    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    set -e

    set +e
    BACKEND_ROOT=${BACKEND_ROOT} python3 -m unittest ${BASE_METRICS_VERIFICATION_TEST_NAME}.TRTLLMBaseMetricsTest.test_end_to_end >> ${BASE_METRICS_VERIFICATION_LOG} 2>&1
    if [ $? -ne 0 ]; then
        cat ${BASE_METRICS_VERIFICATION_LOG}
        echo -e "\n***\n*** Error executing base metrics verification test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        RET=1
    fi
    set -e

    set +e

    kill_server
    wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}

    # Start a clean server to verify base metrics are being
    # reported correctly
    SERVER_LOG="./${NUM_GPU}gpu_IFB_no_streaming_base_metrics.log"
    replace_config_tags '${max_beam_width}' "2" "${MODEL_DIR}/tensorrt_llm/config.pbtxt"
    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    set -e

    set +e
    BACKEND_ROOT=${BACKEND_ROOT} python3 -m unittest ${BASE_METRICS_VERIFICATION_TEST_NAME}.TRTLLMBaseMetricsTest.test_end_to_end_beam_width >> ${BASE_METRICS_VERIFICATION_LOG} 2>&1
    if [ $? -ne 0 ]; then
        cat ${BASE_METRICS_VERIFICATION_LOG}
        echo -e "\n***\n*** Error executing base metrics verification test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        RET=1
    fi
    set -e

    set +e

    kill_server
    wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}

    # World size must be 1 when using multi-model
    if [ "${NUM_GPU}" == "0" ]; then
        # Multi-model
        SERVER_LOG="./${NUM_GPU}gpu_multi_model.log"
        run_server "${SERVER_ARGS} --multi-model"
        wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
        if [ "$WAIT_RET" != "0" ]; then
            # Cleanup
            kill $SERVER_PID > /dev/null 2>&1 || true
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi
        set -e

        python3 ${TOOLS_DIR}/inflight_batcher_llm/end_to_end_test.py \
            --max-input-len=500 \
            --dataset=${DATASET}

        if [ $? -ne 0 ]; then
            cat $SERVER_LOG
            echo -e "\n***\n*** Error executing inflight batching end-to-end test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
            kill_server
            wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
            RET=1
        fi
        set +e

        # Make sure the metrics is retrieved after the server has updated the metrics internally
        sleep ${SLEEP_DURATION}
        curl localhost:8002/metrics -o ${NUM_GPU}gpu_multi_model_metrics.out

        kill_server
        wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    fi

    # inflight batching ON
    # streaming ON
    SERVER_LOG="./${NUM_GPU}gpu_IFB_streaming_server.log"
    replace_config_tags 'decoupled: False' 'decoupled: True' "${MODEL_DIR}/tensorrt_llm/config.pbtxt"

    run_server "${SERVER_ARGS}"
    wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set -e
    python3 ${STREAM_DIR}/end_to_end_grpc_client.py \
        --prompt="My name is"

    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Error executing inflight batching end-to-end streaming test with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
        kill_server
        wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
        RET=1
    fi
    set +e

    # Make sure the metrics is retrieved after the server has updated the metrics internally
    sleep ${SLEEP_DURATION}
    curl localhost:8002/metrics -o ${NUM_GPU}gpu_IFB_stream_metrics.out

    kill_server
    wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
    # Per-request metrics stats
    # Use large number of tokens for KV cache reuse
    echo '{"text_input": "Machine learning is a field of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. It involves the use of data and algorithms to imitate the way humans learn, gradually improving its accuracy. Machine learning is used in a variety of applications such as email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers.", "max_tokens": 50, "pad_id": 2, "end_id": 2, "return_perf_metrics": true }' > tmp.txt
    echo "Machine learning is a field of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit instructions. It involves the use of data and algorithms to imitate the way humans learn, gradually improving its accuracy. Machine learning is used in a variety of applications such as email filtering, detection of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers." > prompt.txt

    # Test the tensorrtllm model with different backends
    for TRITON_BACKEND in tensorrtllm python; do
        for DECOUPLED_TRIAL in non-decoupled decoupled; do
            reset_model_repo
            cp -r ${BACKEND_ROOT}/all_models/inflight_batcher_llm/* ${MODEL_DIR}
            # Copy the engine and place it into the model folder
            cp -r ${BASE_DIR}/engines/inflight_${NUM_GPU}_gpu/ triton_model_repo/tensorrt_llm/1
            ENGINE_DIR=${MODEL_DIR}/tensorrt_llm/1/inflight_${NUM_GPU}_gpu/
            TRITON_MAX_BATCH_SIZE=64
            INSTANCE_COUNT=1
            MAX_QUEUE_DELAY_MS=0
            MAX_QUEUE_SIZE=0
            FILL_TEMPLATE_SCRIPT=${BACKEND_ROOT}/tools/fill_template.py
            if [ "${DECOUPLED_TRIAL}" == "non-decoupled" ]; then
                DECOUPLED_MODE=false
            else
                DECOUPLED_MODE=true
            fi

            python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_DIR}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:TYPE_FP32
            python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_DIR}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
            python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_DIR}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_batching,max_queue_size:${MAX_QUEUE_SIZE},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,request_stats_max_iterations:10,exclude_input_in_output:True,enable_kv_cache_reuse:True,encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32
            python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_DIR}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
            python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_DIR}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:TYPE_FP32

            for ENDPOINT in generate grpc inflight_batcher_llm; do
                SERVER_LOG="./${NUM_GPU}gpu_perf_metrics_${TRITON_BACKEND}_${ENDPOINT}_${DECOUPLED_TRIAL}_server.log"
                CLIENT_LOG="./${NUM_GPU}gpu_perf_metrics_${TRITON_BACKEND}_${ENDPOINT}_${DECOUPLED_TRIAL}_client.log"
                run_server "${SERVER_ARGS}"
                wait_for_server_ready ${SERVER_TIMEOUT} ${SERVER_PID[@]}

                for ITER in 1 2; do
                    if [ "$ITER" == "1" ]; then
                        EXPECTED_KV_CACHE_ALLOC_NEW_BLOCKS=4
                        EXPECTED_KV_CACHE_ALLOC_TOTAL_BLOCKS=4
                        EXPECTED_KV_CACHE_REUSED_BLOCKS=0
                    else
                        EXPECTED_KV_CACHE_ALLOC_NEW_BLOCKS=1
                        EXPECTED_KV_CACHE_ALLOC_TOTAL_BLOCKS=1
                        EXPECTED_KV_CACHE_REUSED_BLOCKS=4
                    fi
                    EXPECTED_ACCEPTANCE_RATE=0.0
                    EXPECTED_TOTAL_ACCEPTED_DRAFT_TOKENS=0
                    EXPECTED_TOTAL_DRAFT_TOKENS=0

                    if [ "$WAIT_RET" != "0" ]; then
                        # Cleanup
                        kill $SERVER_PID > /dev/null 2>&1 || true
                        echo -e "\n***\n*** Failed to start $SERVER\n***"
                        cat $SERVER_LOG
                        exit 1
                    fi

                    if [ "$ENDPOINT" == "generate" ]; then
                        # Generate endpoint
                        # Test with both ensemble and tensorrt_llm_bls models
                        if [ "$ITER" == "1" ]; then
                            MODEL="ensemble"
                        else
                            MODEL="tensorrt_llm_bls"
                        fi

                        set +e
                        if [ "${DECOUPLED_TRIAL}" == "non-decoupled" ]; then
                            code=`curl -s -w %{http_code} -o ./curl.out -d @tmp.txt localhost:8000/v2/models/${MODEL}/generate`
                        else
                            # Remove the "data:" prefix from the response to avoid parsing issues
                            code=$(curl -s -w %{http_code} -o ./curl.out -d @tmp.txt localhost:8000/v2/models/${MODEL}/generate_stream && sed -i 's/^data: //' ./curl.out)
                        fi

                        if [ "$code" != "200" ]; then
                            cat ./curl.out
                            echo -e "\n***\n*** Test Failed\n***"
                            RET=1
                        fi
                        set -e

                        kv_cache_alloc_new_blocks=$(jq '.kv_cache_alloc_new_blocks' curl.out)
                        kv_cache_alloc_total_blocks=$(jq '.kv_cache_alloc_total_blocks' curl.out)
                        kv_cache_reused_blocks=$(jq '.kv_cache_reused_blocks' curl.out)
                        arrival_time_ns=$(jq '.arrival_time_ns' curl.out)
                        first_scheduled_time_ns=$(jq '.first_scheduled_time_ns' curl.out)
                        first_token_time_ns=$(jq '.first_token_time_ns' curl.out)
                        last_token_time_ns=$(jq '.last_token_time_ns' curl.out)
                        acceptance_rate=$(jq '.acceptance_rate' curl.out)
                        total_accepted_draft_tokens=$(jq '.total_accepted_draft_tokens' curl.out)
                        total_draft_tokens=$(jq '.total_draft_tokens' curl.out)
                    else
                        STREAMING_FLAG=""
                        if [ "${DECOUPLED_TRIAL}" == "decoupled" ]; then
                            STREAMING_FLAG="--streaming"
                        fi
                        if [ "$ENDPOINT" == "grpc" ]; then
                            set +e
                            python3 ${STREAM_DIR}/end_to_end_grpc_client.py -v --prompt="$(cat prompt.txt)" --return-perf-metrics ${STREAMING_FLAG} > ${CLIENT_LOG} 2>&1
                            if [ $? -ne 0 ]; then
                                cat $SERVER_LOG
                                echo -e "\n***\n*** Error executing end_to_end_grpc_client.py with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
                                kill_server
                                wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
                                RET=1
                            fi
                            set -e
                        elif [ "$ENDPOINT" == "inflight_batcher_llm" ]; then
                            set +e
                            python3 ${STREAM_DIR}/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir ${TOKENIZER_DIR} \
                                        --return-perf-metrics --text "$(cat prompt.txt)" ${STREAMING_FLAG} > ${CLIENT_LOG} 2>&1
                            if [ $? -ne 0 ]; then
                                cat $SERVER_LOG
                                echo -e "\n***\n*** Error executing inflight_batcher_llm_client.py with ${NUM_GPU}GPU(s): line ${LINENO}\n***"
                                kill_server
                                wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
                                RET=1
                            fi
                            set -e
                        fi
                        kv_cache_alloc_new_blocks=$(grep "kv_cache_alloc_new_blocks" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        kv_cache_alloc_total_blocks=$(grep "kv_cache_alloc_total_blocks" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        kv_cache_reused_blocks=$(grep "kv_cache_reused_blocks" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        arrival_time_ns=$(grep "arrival_time_ns" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        first_scheduled_time_ns=$(grep "first_scheduled_time_ns" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        first_token_time_ns=$(grep "first_token_time_ns" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        last_token_time_ns=$(grep "last_token_time_ns" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        acceptance_rate=$(grep "acceptance_rate" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        total_accepted_draft_tokens=$(grep "total_accepted_draft_tokens" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                        total_draft_tokens=$(grep "total_draft_tokens" ${CLIENT_LOG} | head -n 1 | awk '{print $2}')
                    fi

                    if [[ "$kv_cache_alloc_new_blocks" -ne "$EXPECTED_KV_CACHE_ALLOC_NEW_BLOCKS" || \
                        "$kv_cache_alloc_total_blocks" -ne "$EXPECTED_KV_CACHE_ALLOC_TOTAL_BLOCKS" || \
                        "$kv_cache_reused_blocks" -ne "$EXPECTED_KV_CACHE_REUSED_BLOCKS" || \
                        "$acceptance_rate" != "$EXPECTED_ACCEPTANCE_RATE" || \
                        "$total_accepted_draft_tokens" -ne "$EXPECTED_TOTAL_ACCEPTED_DRAFT_TOKENS" || \
                        "$total_draft_tokens" -ne "$EXPECTED_TOTAL_DRAFT_TOKENS" ]]; then
                        echo "Test failed for ${ENDPOINT} with ${NUM_GPU}GPU(s):"
                        [[ "$kv_cache_alloc_new_blocks" -ne "$EXPECTED_KV_CACHE_ALLOC_NEW_BLOCKS" ]] && \
                            echo "  kv_cache_alloc_new_blocks: expected $EXPECTED_KV_CACHE_ALLOC_NEW_BLOCKS, got $kv_cache_alloc_new_blocks"
                        [[ "$kv_cache_alloc_total_blocks" -ne "$EXPECTED_KV_CACHE_ALLOC_TOTAL_BLOCKS" ]] && \
                            echo "  kv_cache_alloc_total_blocks: expected $EXPECTED_KV_CACHE_ALLOC_TOTAL_BLOCKS, got $kv_cache_alloc_total_blocks"
                        [[ "$kv_cache_reused_blocks" -ne "$EXPECTED_KV_CACHE_REUSED_BLOCKS" ]] && \
                            echo "  kv_cache_reused_blocks: expected $EXPECTED_KV_CACHE_REUSED_BLOCKS, got $kv_cache_reused_blocks"
                        [[ "$acceptance_rate" != "$EXPECTED_ACCEPTANCE_RATE" ]] && \
                            echo "  acceptance_rate: expected $EXPECTED_ACCEPTANCE_RATE, got $acceptance_rate"
                        [[ "$total_accepted_draft_tokens" -ne "$EXPECTED_TOTAL_ACCEPTED_DRAFT_TOKENS" ]] && \
                            echo "  total_accepted_draft_tokens: expected $EXPECTED_TOTAL_ACCEPTED_DRAFT_TOKENS, got $total_accepted_draft_tokens"
                        [[ "$total_draft_tokens" -ne "$EXPECTED_TOTAL_DRAFT_TOKENS" ]] && \
                            echo "  total_draft_tokens: expected $EXPECTED_TOTAL_DRAFT_TOKENS, got $total_draft_tokens"
                        RET=1
                    fi

                    if ! [[ $arrival_time_ns =~ ^-?[0-9]+$ ]] || [ $arrival_time_ns -eq 0 ]; then
                        echo "Arrival time $arrival_time_ns is not valid, expected positive integer value"
                        RET=1
                    fi
                    if ! [[ $first_scheduled_time_ns =~ ^-?[0-9]+$ ]] || [ $first_scheduled_time_ns  -eq 0 ]; then
                        echo "First scheduled time $first_scheduled_time_ns is not valid, expected positive integer value"
                        RET=1
                    fi
                    if ! [[ $first_token_time_ns =~ ^-?[0-9]+$ ]] || [ $first_token_time_ns -eq 0 ]; then
                        echo "First token time $first_token_time_ns is not valid, expected positive integer value"
                        RET=1
                    fi
                    if [ "${DECOUPLED_TRIAL}" == "decoupled" ]; then
                        # Allow 0 for streaming mode
                        if ! [[ $last_token_time_ns =~ ^-?[0-9]+$ ]]; then
                            echo "Last token time: expected 0; got $last_token_time_ns"
                            RET=1
                        fi
                    else
                        if ! [[ $last_token_time_ns =~ ^-?[0-9]+$ ]] || [ $last_token_time_ns -eq 0 ]; then
                            echo "Last token time $last_token_time_ns is not valid, expected positive integer value"
                            RET=1
                        fi
                    fi
                done
                kill_server
                wait_for_server_terminated ${SERVER_TIMEOUT} ${SERVER_PID[@]}
                # Add a delay to make sure the memory is freed before starting the next test
                sleep 10
            done
        done
    done
done


set +e
# Verify TRT LLM statistics are being properly reported as custom metrics
python3 ${CUSTOM_METRICS_VERIFICATION_TEST} >> ${CUSTOM_METRICS_VERIFICATION_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat ${CUSTOM_METRICS_VERIFICATION_LOG}
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
