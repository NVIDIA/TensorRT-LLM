export PATH=$PATH:/home/dev_user/.local/bin/

LOG=/home/scratch.timothyg_gpu/data/log/serve_log/test_serve.log
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
nohup python /home/scratch.timothyg_gpu/TensorRT-LLM/tensorrt_llm/commands/serve.py "${MODEL}" --host localhost --port 8500 \
    > "${LOG}" 2>&1 &

echo "Server started. Logs saved to ${LOG}"