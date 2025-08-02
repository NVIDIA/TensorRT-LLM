set -ex
export TQDM_MININTERVAL=1000
export PRINT_ITER_LOG=true
nvidia-smi

echo "TRT-LLM GIT COMMIT": $TRT_LLM_GIT_COMMIT

# The script must run under PWD as TRT-LLM source code root, because it uses ../benchmarks/cpp/prepare_dataset.py
output_folder=$1
pushd ${output_folder} # all the dataset and the yaml are stored inside the folder

# Define an array of models to benchmark
declare -a models=(
  "70B-FP8:nvidia/Llama-3.1-70B-Instruct-FP8:/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-70B-Instruct-FP8:4:false"
  "Scout-FP4:/home/scratch.trt_llm_data/llm-models/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4:/home/scratch.trt_llm_data/llm-models/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4:4:false"
  "R1-0528-FP4:nvidia/DeepSeek-R1-0528-FP4:/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-0528-FP4:4:true"
#  "405B-FP8:nvidia/Llama-3.1-405B-Instruct-FP8:/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-405B-Instruct-FP8:4:false"
)

# num_requests concurrency isl osl
declare -a data_configs=(
  40:4:1024:1024
  160:16:1024:1024
  640:64:1024:1024
  1280:128:1024:1024
  2560:256:1024:1024
  4096:1024:1024:1024
)

# Loop through models and run benchmarks
for model_info in "${models[@]}"; do
  IFS=':' read -r model_label model_name model_path GPUS adp <<< "$model_info"
for req_info in "${data_configs[@]}"; do
  IFS=":" read -r num_requests concurrency isl osl <<< "${req_info}"

  # Parse model info (format: model_label:model_name:model_path:gpus)
  echo "========================================================="
  echo "Running benchmark for $model_label model: $model_name"
  echo "========================================================="

  # Create the extra config file
  cat >/tmp/extra-llm-api-config.yml<<EOF
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
  print_iter_log: ${PRINT_ITER_LOG}
  enable_attention_dp: ${adp}
EOF

  # Generate dataset if it doesn't exist
  dataset_file=/tmp/"${model_label}_${num_requests}_isl${isl}_osl${osl}.json"
  if [ ! -f "$dataset_file" ]; then
    echo "Generating dataset for $model_name: $dataset_file"
    python ../benchmarks/cpp/prepare_dataset.py --tokenizer=$model_name --stdout token-norm-dist --num-requests=$num_requests \
        --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0 > $dataset_file
  fi

  LOCAL_DATA_PATH_OPTIONS=""
  if [ -e ${model_path} ]; then # only use local data path when it is detected
    LOCAL_DATA_PATH_OPTIONS="--model_path ${model_path}"
  fi

  cat  /tmp/extra-llm-api-config.yml

  # Run benchmark for this model
  mpirun -n 1 --oversubscribe --allow-run-as-root \
  trtllm-bench --model $model_name ${LOCAL_DATA_PATH_OPTIONS} \
    throughput --dataset $dataset_file --backend pytorch --extra_llm_api_options /tmp/extra-llm-api-config.yml \
      --tp ${GPUS} \
      --ep ${GPUS} \
      --warmup 0 \
      --num_requests ${num_requests} \
      --concurrency ${concurrency} \
      --kv_cache_free_gpu_mem_fraction 0.90 |& tee -a bench.${model_label}.${concurrency}.log

  echo "Benchmark completed for $model_label model: $model_name"
  echo ""

done
done

echo "All benchmarks completed!"
popd
