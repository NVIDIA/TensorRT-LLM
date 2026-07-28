# Qwen3 (PyTorch Backend)

This guide covers running Qwen3 / Qwen3-Next on the TensorRT LLM **PyTorch backend**.
HuggingFace checkpoints are loaded directly — there is no checkpoint-conversion or
engine-build step. For the LLM Python API see
[examples/llm-api](../../../llm-api/), and for serving see
[`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html).

## Qwen3

TensorRT LLM now supports Qwen3, the latest version of the Qwen model series. This guide walks you through the examples to run the Qwen3 models using NVIDIA's TensorRT LLM framework with the PyTorch backend. According to the support matrix, TensorRT LLM provides comprehensive support for various Qwen3 model variants including:

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- Qwen3-30B-A3B
- Qwen3-235B-A22B

Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html) for how to build TensorRT LLM from source and start a TRT-LLM docker container if needed.

> [!NOTE]
> This guide assumes that you replace placeholder values (e.g. `<YOUR_MODEL_DIR>`) with the appropriate paths.

### Downloading the Model Weights

Qwen3 model weights are available on [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B). To download the weights, execute the following commands (replace `<YOUR_MODEL_DIR>` with the target directory where you want the weights stored):

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-30B-A3B <YOUR_MODEL_DIR>
```

### Quick start

#### Run a single inference

To quickly run Qwen3, [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py):

```bash
python3 examples/llm-api/quickstart_advanced.py --model_dir Qwen3-30B-A3B/ --kv_cache_fraction 0.6
```

### Evaluation

1. Evaluate accuracy on the MMLU dataset:

```bash
trtllm-eval --model=Qwen3-32B/ --tokenizer=Qwen3-32B/ --backend=pytorch mmlu --dataset_path=./datasets/mmlu/
[05/01/2025-13:56:15] [TRT-LLM] [I] MMLU weighted average accuracy: 79.09 (14042)
```

```bash
trtllm-eval --model=Qwen3-30B-A3B/ --tokenizer=Qwen3-30B-A3B/ --backend=pytorch mmlu --dataset_path=./datasets/mmlu/
[05/05/2025-11:33:02] [TRT-LLM] [I] MMLU weighted average accuracy: 79.44 (14042)
```

2. Evaluate accuracy on GSM8K dataset:

```bash
trtllm-eval --model=Qwen3-30B-A3B/ --tokenizer=Qwen3-30B-A3B/ --backend=pytorch gsm8k --dataset_path=./datasets/openai/gsm8k/
[05/05/2025-12:05:40] [TRT-LLM] [I] lm-eval gsm8k results (scores normalized to range 0~100):
|Tasks|Version|     Filter     |n-shot|  Metric   |   | Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|------:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |84.3063|±  |1.0019|
|     |       |strict-match    |     5|exact_match|↑  |88.6277|±  |0.8745|

```

### Model Quantization

To quantize the Qwen3 model for use with the PyTorch backend, we'll use NVIDIA's Model Optimizer (ModelOpt) tool. Follow these steps:

```bash
# Clone the Model Optimizer (ModelOpt)
git clone https://github.com/NVIDIA/Model-Optimizer.git
pushd Model-Optimizer

# install the ModelOpt
pip install -e .

# Quantize the Qwen3-235B-A22B model by nvfp4
# By default, the checkpoint would be stored in `Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-235B-A22B_nvfp4_hf/`.
./examples/llm_ptq/scripts/huggingface_example.sh --model Qwen3-235B-A22B/ --quant nvfp4 --export_fmt hf

# Quantize the Qwen3-32B model by fp8_pc_pt
# By default, the checkpoint would be stored in `Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-32B_fp8_pc_pt_hf/`.
./examples/llm_ptq/scripts/huggingface_example.sh --model Qwen3-32B/ --quant fp8_pc_pt --export_fmt hf
popd
```

### Benchmark

To run the benchmark, we suggest using the `trtllm-bench` tool. Please refer to the following script on B200:

```bash
#!/bin/bash

folder_model=Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-235B-A22B_nvfp4_hf/
path_config=config.yml
num_gpus=8
ep_size=8
max_input_len=1024
max_batch_size=512
# We want to limit the number of prefill requests to 1 with in-flight batching.
max_num_tokens=$(( max_input_len + max_batch_size - 1 ))
kv_cache_free_gpu_mem_fraction=0.9
concurrency=128

path_data=./aa_prompt_isl_1k_osl_2k_qwen3_10000samples.txt

# Setup the extra configuration for llm-api
echo -e "disable_overlap_scheduler: false
print_iter_log: true
cuda_graph_config:
  batch_sizes: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128]
enable_attention_dp: true " > ${path_config}

# Run trtllm-bench with pytorch backend
mpirun --allow-run-as-root --oversubscribe -n 1 \
trtllm-bench --model ${folder_model} --model_path ${folder_model} throughput \
  --backend pytorch \
  --max_batch_size ${max_batch_size} \
  --max_num_tokens ${max_num_tokens} \
  --dataset ${path_data} \
  --tp ${num_gpus}\
  --ep ${ep_size} \
  --kv_cache_free_gpu_mem_fraction ${kv_cache_free_gpu_mem_fraction} \
  --config ${path_config} \
  --concurrency ${concurrency} \
  --num_requests $(( concurrency * 5 )) \
  --warmup 0 \
  --streaming
```

We suggest benchmarking with a real dataset. It will prevent from having improperly distributed tokens in the MoE. Here, we use the `aa_prompt_isl_1k_osl_2k_qwen3_10000samples.txt` dataset. It has 10000 samples with an average input length of 1024 and an average output length of 2048. If you don't have a dataset (this or another) and you want to run the benchmark, you can use the following command to generate a random dataset:

```bash
folder_model=Model-Optimizer/examples/llm_ptq/saved_models_Qwen3-235B-A22B_nvfp4_hf/
min_input_len=1024
min_output_len=2048
concurrency=128
path_data=random_data.txt

trtllm-bench\
    --model=${folder_model} \
    prepare-dataset --output ${path_data} \
    token-norm-dist --num-requests=$(( concurrency * 5 )) \
    --input-mean=${min_input_len} --output-mean=${min_output_len} --input-stdev=0 --output-stdev=0
```

### Serving

#### Recommended Performance Settings

We maintain YAML configuration files with recommended performance settings in the [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) directory. These config files are present in the TensorRT LLM container at the path `/app/tensorrt_llm/examples/configs`. You can use these out-of-the-box, or adjust them to your specific use case.

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/qwen3.yaml
```

#### trtllm-serve

To serve the model using `trtllm-serve`:

```bash
trtllm-serve Qwen3-30B-A3B/ --port 8000 --config ${EXTRA_LLM_API_FILE}
```

To query the server, you can start with a `curl` command:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen3-30B-A3B/",
      "prompt": "Please describe what is Qwen.",
      "max_tokens": 12,
      "temperature": 0
  }'
```
#### Disaggregated Serving

To serve the model in disaggregated mode, you should launch context and generation servers using `trtllm-serve`.

For example, you can launch a single context server on port 8001 with:

```bash
export TRTLLM_USE_UCX_KVCACHE=1
export TRTLLM_DIR=/app/tensorrt_llm
export EXTRA_LLM_API_FILE="${TRTLLM_DIR}/examples/configs/curated/qwen3-disagg-prefill.yaml"

trtllm-serve Qwen3-30B-A3B/ --port 8001 --config ${EXTRA_LLM_API_FILE} &> output_ctx &
```

And you can launch two generation servers on port 8002 and 8003 with:

```bash
export TRTLLM_USE_UCX_KVCACHE=1
export TRTLLM_DIR=/app/tensorrt_llm
export EXTRA_LLM_API_FILE="${TRTLLM_DIR}/examples/configs/curated/qwen3.yaml"

for port in {8002..8003}; do \
trtllm-serve Qwen3-30B-A3B/ --port ${port} --config ${EXTRA_LLM_API_FILE} &> output_gen_${port} & \
done
```

Finally, you can launch the disaggregated server which will accept requests from the client and do
the orchestration between the context and generation servers with:

```bash
cat >./disagg-config.yml <<EOF
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
      - "localhost:8002"
EOF

trtllm-serve disaggregated -c disagg-config.yaml
```

To query the server, you can start with a `curl` command:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "Qwen3-30B-A3B/",
      "prompt": "Please describe what is Qwen.",
      "max_tokens": 12,
      "temperature": 0
  }'
```

Note that the optimal disaggregated serving configuration (i.e. tp/pp/ep mappings, number of ctx/gen instances, etc.) will depend
on the request parameters, the number of concurrent requests and the GPU type. It is recommended to experiment to identify optimal
settings for your specific use case.

#### Eagle3

Qwen3 now supports Eagle3 (Speculative Decoding with Eagle3). To enable Eagle3 on Qwen3, you need to set the following arguments when running `trtllm-bench` or `trtllm-serve`:

- `speculative_config.decoding_type: Eagle3`
  Set the decoding type to `Eagle3` to enable Eagle3 speculative decoding.
- `speculative_config.max_draft_len: 3`
  Set the maximum number of draft tokens generated per step (this value can be adjusted as needed).
- `speculative_config.speculative_model: <HUGGINGFACE ID / LOCAL PATH>`
  Specify the Eagle3 draft model either as a Huggingface model ID or a local path. You can find ready-to-use Eagle3 draft models at https://huggingface.co/collections/nvidia/speculative-decoding-modules.

Currently, there are some limitations when enabling Eagle3:

1. `attention_dp` is not supported. Please disable it or do not set the related flag (it is disabled by default).
2. If you want to use `enable_block_reuse`, the kv cache type of the target model and the draft model must be the same. Since the draft model only supports fp16/bf16, you need to disable `enable_block_reuse` when using fp8 kv cache.

Example `config.yml` snippet for Eagle3:

```bash
echo "
enable_attention_dp: false
speculative_config:
    decoding_type: Eagle3
    max_draft_len: 3
    speculative_model: <HUGGINGFACE ID / LOCAL PATH>
kv_cache_config:
    enable_block_reuse: false
" >> ${path_config}
```

For further details, please refer to [speculative-decoding.md](../../../../docs/source/legacy/advanced/speculative-decoding.md)

### Dynamo

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.
Dynamo supports TensorRT LLM as one of its inference engine. For details on how to use TensorRT LLM with Dynamo please refer to [LLM Deployment Examples using TensorRT-LLM](https://github.com/ai-dynamo/dynamo/blob/main/examples/tensorrt_llm/README.md)

## Qwen3-Next

Below is the command to run the Qwen3-Next model.

```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_advanced.py --model_dir /Qwen3-Next-80B-A3B-Thinking --kv_cache_fraction 0.6 --disable_kv_cache_reuse --max_batch_size 1 --tp_size 4

```

### NVFP4 quantization

TRTLLM supports NVFP4 precision with blocksize=16 for both activations and GEMM weights.
To run the Qwen3-Next model on NVFP4 precision, use the following command
```bash
mpirun -n 1 --allow-run-as-root --oversubscribe python3 examples/llm-api/quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --kv_cache_fraction 0.6 --disable_kv_cache_reuse --max_batch_size 1 --tp_size 2 --trust_remote_code

```

## Notes and Troubleshooting

- **Model Directory:** Update `<YOUR_MODEL_DIR>` with the actual path where the model weights reside.
- **GPU Memory:** Adjust `--max_batch_size` and `--max_num_tokens` if you encounter out-of-memory errors.
- **Configuration Files:** Verify that the configuration files are correctly formatted to avoid runtime issues.

## Credits
This Qwen model example exists thanks to Tlntin (TlntinDeng01@gmail.com) and zhaohb (zhaohbcloud@126.com).
