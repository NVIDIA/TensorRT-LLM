#!/bin/bash
dataset="template_trtllm_openai_completions.json"
output_folder="output_loadgen"
port=8000
host="localhost"
max_count=256
model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
streaming="False"
input_tokens=128
output_tokens=128
concurrency=32

infserver_loadgen ${dataset} \
    --output_dir "${output_folder}" \
    --set dataset.input_tokens:int="${input_tokens}" \
    --set dataset.output_tokens:int="${output_tokens}" \
    --set dataset.max_count:int="${max_count}" \
    --set dataset.model_name:str="${model_name}" \
    --set dataset.max_concurrent_requests:int="${concurrency}" \
    --set inference_server.host:str="${host}" \
    --set inference_server.port:int="${port}" \
    --set post_processors[0].model_name:str="${model_name}" \
    --set timing_strategy.desired_rps:float="-1" \
    --set inference_server.inference_server_config.stream:bool="${streaming}"
