cd /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others

export TRTLLM_ENABLE_KV_CACHE_PRECISION_CONVERSION=1
export TRTLLM_DISABLE_SELECTIVE_CACHE_TRANSFER=1
export UCX_LOG_LEVEL=info
export TLLM_LOG_LEVEL=INFO

pytest /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others/test_mixed_kv_transceiver.py \
  -k "ctx_fp16_gen_fp8" -v -s | tee log.log

# pytest /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others/test_mixed_kv_transceiver.py -v -s --test-only test_kv_cache_transceiver_single_process[ctx_fp16_gen_fp8-mha]

# pytest /home/scratch.timothyg_gpu/TensorRT-LLM/tests/unittest/others/test_mixed_kv_transceiver.py -v -s
