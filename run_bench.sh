BATCH_PER_GPU=64
NUM_GPU=8
CONCURRENCY=$((BATCH_PER_GPU * NUM_GPU))

# uncomment following lines to capture nsys profile

# TLLM_PROFILE_START_STOP=300-310 nsys profile \
#  -o profile_4k_512batch_5_30_0080 -f true -t 'cuda,nvtx,python-gil' -c cudaProfilerApi --cuda-graph-trace node \
#  -e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json --trace-fork-before-exec=true \
trtllm-bench \
 -m /workspaces/tensorrt_llm/hf-ckpt \
 --model_path /workspaces/tensorrt_llm/hf-ckpt \
 throughput \
 --tp $NUM_GPU --ep $NUM_GPU --warmup 0 \
 --dataset dataset4k.txt \
 --backend pytorch \
 --max_batch_size $CONCURRENCY --max_num_tokens 5220 --num_requests $CONCURRENCY --concurrency $CONCURRENCY \
 --kv_cache_free_gpu_mem_fraction 0.85 \
 --extra_llm_api_options ./extra-llm-api-config.yml \
 | tee log_4k512b_5_30_0039_3.txt

# python parse_iter_log.py --file log_4k512b_5_29_0080.txt --concurrency 256 --enable_dp --gpu_num 8
