export PATH=$PATH:/home/dev_user/.local/bin/

# CTX_MODEL=/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct
CTX_MODEL=/home/scratch.timothyg_gpu/weights/Llama-3.1-8B-Instruct-FP8-NO-KV-QUANT

# GEN_MODEL=/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct
# GEN_MODEL=/home/scratch.timothyg_gpu/weights/Llama-3.1-8B-Instruct-NVFP4-NO-KV-QUANT
GEN_MODEL=/home/scratch.timothyg_gpu/weights/Llama-3.1-8B-Instruct-FP8-NO-KV-QUANT

echo "Launching context 1"

# Utility: choose a random free TCP port for the ZeroMQ IPC channel
rand_port() {
  python - <<'PY'
import socket, random
while True:
    port = random.randint(20000, 40000)
    with socket.socket() as s:
        try:
            s.bind(("", port))
            print(port)
            break
        except OSError:
            pass
PY
}

# Launch a context-only worker (for embedding / “ctx” tasks)
launch_ctx() {
  local gpu=$1 http_port=$2 log_file=$3
  TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR="tcp://127.0.0.1:$(rand_port)" \
  CUDA_VISIBLE_DEVICES=${gpu} \
  trtllm-serve ${CTX_MODEL} \
      --host localhost --port ${http_port} \
      --extra_llm_api_options ./ctx_extra-llm-api-config.yaml \
      &> "${log_file}" &
}

# Launch a generation worker (for token generation / “gen” tasks)
launch_gen() {
  local gpu=$1 http_port=$2 log_file=$3
  TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR="tcp://127.0.0.1:$(rand_port)" \
  CUDA_VISIBLE_DEVICES=${gpu} \
  trtllm-serve ${GEN_MODEL} \
      --host localhost --port ${http_port} \
      --extra_llm_api_options ./gen_extra-llm-api-config.yaml \
      &> "${log_file}" &
}

echo "Launching context servers"

# Start two context servers
launch_ctx 0 8401 log_ctx_0
# launch_ctx 1 8402 log_ctx_1

echo "Launching gen servers"

# Start one generation server
launch_gen 1 8403 log_gen_0

echo "Launching controller/proxy server"

# Finally start the disaggregated controller/proxy
trtllm-serve disaggregated -c disagg_config.yaml &

echo "Servers launched"