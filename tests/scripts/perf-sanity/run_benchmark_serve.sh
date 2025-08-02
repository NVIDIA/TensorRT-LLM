set -ex
export TQDM_MININTERVAL=1000
export PRINT_ITER_LOG=false
nvidia-smi

echo "TRT-LLM GIT COMMIT": $TRT_LLM_GIT_COMMIT

# The script must run under PWD as TRT-LLM source code root, because it uses ../tensorrt_llm/serve/scripts/benchmark_serving.py
output_folder=$1
pushd ${output_folder} # all the dataset and the yaml are stored inside the folder

# Sweep configs, change here if you want to run less or more
declare -a models=(
  "R1-0528-FP4:nvidia/DeepSeek-R1-0528-FP4:/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-0528-FP4:4:true:2048:0.9"
  "R1-FP8:deepseek-ai/DeepSeek-R1：/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1/:8:true:2048:0.8"
  "70B-FP8:nvidia/Llama-3.1-70B-Instruct-FP8:/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-70B-Instruct-FP8:4:false:5500:0.9"
  "Scout-FP4:/home/scratch.trt_llm_data/llm-models/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4:/home/scratch.trt_llm_data/llm-models/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4:4:false:2048:0.85"
  #"405B-FP8:nvidia/Llama-3.1-405B-Instruct-FP8:/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-405B-Instruct-FP8:4:false:5500"
  #"70B-FP16:meta-llama/Llama-3.1-70B:/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Meta-Llama-3.1-70B:4:false:5500"
)
concurrency_levels=(4) #1024 256 128 64 16 4)
ISL=1024
OSL=1024


  # Function to wait for server to be ready
wait_for_server() {
  target_server_pid=$1
  local max_attempts=360
  local attempt=1
  local server_ready=false

  echo "Waiting for trtllm-serve to be ready..."

  while [ $attempt -le $max_attempts ]; do
    # Check if the server is still running
    if ! kill -0 $target_server_pid 2>/dev/null; then
      echo "Error: Server process has died"
      return 1
    fi

    # Try to connect to the server and check HTTP status code 200
    http_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null)
    if [ "$http_status" = "200" ]; then
      echo "Server is ready! HTTP status: $http_status"
      server_ready=true
      break
    fi

    echo "Attempt $attempt/$max_attempts: Server not ready yet (HTTP status: ${http_status:-N/A}), waiting..."
    sleep 10
    ((attempt++))
  done

  if [ "$server_ready" = false ]; then
    echo "Error: Server did not become ready after $max_attempts attempts"
    return 1
  fi

  return 0
}

# Loop through models and run benchmarks
for model_info in "${models[@]}"; do
  IFS=':' read -r model_label model_name model_path GPUS adp max_num_tokens kv_fraction <<< "$model_info"
  # Parse model info (format: model_label:model_name:model_path:gpus)
  echo "========================================================="
  echo "Running benchmark for $model_label model: $model_name"
  echo "========================================================="

  # Create the extra config file
  cat >/tmp/extra-llm-api-config.yml<<EOF
  print_iter_log: ${PRINT_ITER_LOG}
  cuda_graph_config:
      enable_padding: true
      batch_sizes:
        - 1
        - 2
        - 4
        - 8
        - 16
        - 32
        - 64
        - 128
        - 256
        - 384
        - 512
        - 1024
        - 2048
  enable_attention_dp: ${adp}
EOF

  MODEL=""
  if [ -e ${model_path} ]; then # only use local data path when it is detected
    MODEL="${model_path}"
  else
    MODEL="{model_name}"
  fi

  cat  /tmp/extra-llm-api-config.yml

  # Start the server and capture its PID
  mpirun -n 1 --oversubscribe --allow-run-as-root \
  trtllm-serve ${MODEL} --backend pytorch --tp_size ${GPUS} --ep_size ${GPUS} --max_batch_size 1024 --max_num_tokens ${max_num_tokens} --kv_cache_free_gpu_memory_fraction ${kv_fraction} --extra_llm_api_options /tmp/extra-llm-api-config.yml &
  server_pid=$!

  # Wait for the server to be ready before proceeding
  wait_for_server $server_pid || {
    echo "Failed to start server, killing process and exiting"
    kill $server_pid 2>/dev/null
    exit 1
  }

  for concurrency in "${concurrency_levels[@]}"
  do
      num_prompts=$((concurrency * 10))
      echo "Running benchmark with concurrency: $concurrency and num-prompts: $num_prompts"

      python -m tensorrt_llm.serve.scripts.benchmark_serving \
          --model ${MODEL} \
          --dataset-name random \
          --random-ids \
          --num-prompts "$num_prompts" \
          --random-input-len ${ISL} \
          --random-output-len ${OSL} \
          --random-range-ratio 0.0 \
          --ignore-eos \
          --percentile-metrics ttft,tpot,itl,e2el \
          --max-concurrency "$concurrency" |& tee serve.${model_label}.${concurrency}.log

      echo "Completed benchmark with concurrency: $concurrency"
      echo "-----------------------------------------"
  done

  # Cleanup: Kill the server process when done with this model
  echo "Stopping server for $model_label"
  kill $server_pid
  wait $server_pid 2>/dev/null || true
  sleep 5  # Give it some time to clean up resources

  echo "Benchmark completed for $model_label model: $model_name"
  echo ""

done

echo "All benchmarks completed!"
popd
