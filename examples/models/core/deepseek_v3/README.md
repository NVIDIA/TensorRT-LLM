# DeepSeek‑V3 and DeepSeek-R1

This guide walks you through the examples to run the DeepSeek‑V3/DeepSeek-R1 models using NVIDIA's TensorRT LLM framework with the PyTorch backend.
**DeepSeek-R1 and DeepSeek-V3 share exact same model architecture other than weights differences, and share same code path in TensorRT-LLM, for brevity we only provide one model example, the example command to be used interchangeably by only replacing the model name to the other one**.

To benchmark the model with best configurations, refer to [DeepSeek R1 benchmarking blog](../../../../docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md).

Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html) for how to build TensorRT LLM from source and start a TRT-LLM docker container.

> [!NOTE]
> This guide assumes that you replace placeholder values (e.g. `<YOUR_MODEL_DIR>`) with the appropriate paths.


## Table of Contents


- [DeepSeek‑V3 and DeepSeek-R1](#deepseekv3-and-deepseek-r1)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Downloading the Model Weights](#downloading-the-model-weights)
  - [Quick Start](#quick-start)
    - [Run a single inference](#run-a-single-inference)
    - [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
      - [Relaxed acceptance](#relaxed-acceptance)
    - [Long context support](#long-context-support)
      - [ISL-64k-OSL-1024](#isl-64k-osl-1024)
      - [ISL-128k-OSL-1024](#isl-128k-osl-1024)
  - [Evaluation](#evaluation)
  - [Serving](#serving)
    - [trtllm-serve](#trtllm-serve)
      - [B200 FP4 min-latency config](#b200-fp4-min-latency-config)
      - [B200 FP4 max-throughput config](#b200-fp4-max-throughput-config)
      - [B200 FP8 min-latency config](#b200-fp8-min-latency-config)
      - [B200 FP8 max-throughput config](#b200-fp8-max-throughput-config)
      - [Launch trtllm-serve OpenAI-compatible API server](#launch-trtllm-serve-openai-compatible-api-server)
    - [Disaggregated Serving](#disaggregated-serving)
    - [Dynamo](#dynamo)
    - [tensorrtllm\_backend for triton inference server (Prototype)](#tensorrtllm_backend-for-triton-inference-server-prototype)
  - [Advanced Usages](#advanced-usages)
    - [Multi-node](#multi-node)
      - [mpirun](#mpirun)
      - [Slurm](#slurm)
      - [Example: Multi-node benchmark on GB200 Slurm cluster](#example-multi-node-benchmark-on-gb200-slurm-cluster)
    - [DeepGEMM](#deepgemm)
    - [FlashMLA](#flashmla)
    - [FP8 KV Cache and MLA](#fp8-kv-cache-and-mla)
    - [W4AFP8](#w4afp8)
      - [Activation calibration](#activation-calibration)
      - [Weight quantization and assembling](#weight-quantization-and-assembling)
    - [KV Cache Reuse](#kv-cache-reuse)
    - [Chunked Prefill](#chunked-prefill)
  - [Notes and Troubleshooting](#notes-and-troubleshooting)
  - [Known Issues](#known-issues)


## Hardware Requirements

DeepSeek-v3 has 671B parameters which needs about 671GB GPU memory for FP8 weights, and needs more memories for activation tensors and KV cache.
The minimum hardware requirements for running DeepSeek V3/R1 at FP8/FP4/W4A8 are listed as follows.

| GPU  | DeepSeek-V3/R1 FP8 | DeepSeek-V3/R1 FP4 | DeepSeek-V3/R1 W4A8 |
| -------- | ------- | -- | -- |
| H100 80GB | 16 | N/A | 8 |
| H20 141GB | 8 | N/A | 4 |
| H20 96GB | 8  | N/A | 4 |
| H200 | 8     | N/A | 4 |
| B200/GB200| Not supported yet, WIP | 4 (8 GPUs is recommended for best perf) | Not supported yet, WIP |

Ampere architecture (SM80 & SM86) is not supported.


## Downloading the Model Weights

DeepSeek‑v3 model weights are available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3). To download the weights, execute the following commands (replace `<YOUR_MODEL_DIR>` with the target directory where you want the weights stored):

```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3 <YOUR_MODEL_DIR>
```


## Quick Start

### Run a single inference
To quickly run DeepSeek-V3, [examples/llm-api/quickstart_advanced.py](../llm-api/quickstart_advanced.py):

```bash
cd examples/llm-api
python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --tp_size 8
```

The model will be run by PyTorch backend and generate outputs like:
```
Processed requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.66it/s]
Prompt: 'Hello, my name is', Generated text: " Dr. Anadale. I teach philosophy at Mount St. Mary's University in Emmitsburg, Maryland. This is the first in a series of videos"
Prompt: 'The president of the United States is', Generated text: ' the head of state and head of government of the United States, indirectly elected to a four-year term via the Electoral College. The officeholder leads the executive branch'
Prompt: 'The capital of France is', Generated text: ' Paris. Paris is one of the most famous and iconic cities in the world, known for its rich history, culture, art, fashion, and cuisine. It'
Prompt: 'The future of AI is', Generated text: ' a topic of great interest and speculation. While it is impossible to predict the future with certainty, there are several trends and possibilities that experts have discussed regarding the future'
```

### Multi-Token Prediction (MTP)
To run with MTP, use [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py) with additional options, see
```bash
cd examples/llm-api
python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --spec_decode_algo MTP --spec_decode_max_draft_len N
```

`N` is the number of MTP modules. When `N` is equal to `0`, which means that MTP is not used (default). When `N` is greater than `0`, which means that `N` MTP modules are enabled. In the current implementation, the weight of each MTP module is shared.

#### Relaxed acceptance
**NOTE: This feature can only be used for DeepSeek R1.**
When verifying and receiving draft tokens, there are two ways:
- Strict acceptance: (default)

  The draft token is accepted only when it is exactly the same as the token sampled by the target model based on the Top-1 strategy.
- Relaxed acceptance:

  For the reasoning model (such as DeepSeek R1), the generation may consist of two phases: `thinking phase` and `actual output` (`<think>[thinking phase]</think>[actual output]`).

  **During the thinking phase**, if we enable relaxed acceptance, the draft token can be accepted when it is in a candidate. This candidate is generated based on the logits and the below 2 knobs.
  - Knob 1: Top-N. The top-N tokens are sampled from logits.
  - Knob 2: Probability threshold (delta). Based on Top-N candidates, only those tokens with a probability greater than the Top-1's probability - delta can remain in the candidate set.

  During the non-thinking phase, we still use the strict acceptance.

  This is a relaxed way of verification and comparison, which can improve the acceptance rate and bring positive speedup.

  Here is an example. We allow the first 15 (`--relaxed_topk 15`) tokens to be used as the initial candidate set, and use delta (`--relaxed_delta 0.5`) to filter out tokens with a large probability gap, which may be semantically different from the top-1 token.

  ```bash
  cd examples/llm-api
  python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --spec_decode_algo MTP --spec_decode_max_draft_len N --use_relaxed_acceptance_for_thinking --relaxed_topk 15 --relaxed_delta 0.5
  ```

### Long context support
DeepSeek-V3 model can support up to 128k context length. The following shows how to benchmark 64k and 128k input_seq_length using trtllm-bench on B200.
To avoid OOM (out of memory) error, you need to adjust the values of "--max_batch_size", "--max_num_tokens" and "--kv_cache_free_gpu_mem_fraction".
#### ISL-64k-OSL-1024
```bash
DS_R1_NVFP4_MODEL_PATH=/path/to/DeepSeek-R1
python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py \
        --stdout --tokenizer ${DS_R1_NVFP4_MODEL_PATH} \
        token-norm-dist \
        --input-mean 65536 --output-mean 1024 \
        --input-stdev 0 --output-stdev 0 \
        --num-requests 24 > /tmp/benchmarking_64k.txt

cat <<EOF > /tmp/extra-llm-api-config.yml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 4, 8, 12]
EOF

trtllm-bench -m deepseek-ai/DeepSeek-R1 --model_path ${DS_R1_NVFP4_MODEL_PATH} throughput \
        --tp 8 --ep 8 \
        --warmup 0 \
        --dataset /tmp/benchmarking_64k.txt \
        --max_batch_size 12 \
        --max_num_tokens 65548 \
        --kv_cache_free_gpu_mem_fraction 0.6 \
        --extra_llm_api_options /tmp/extra-llm-api-config.yml
```

#### ISL-128k-OSL-1024
```bash
DS_R1_NVFP4_MODEL_PATH=/path/to/DeepSeek-R1
python /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py \
        --stdout --tokenizer ${DS_R1_NVFP4_MODEL_PATH} \
        token-norm-dist \
        --input-mean 131072 --output-mean 1024 \
        --input-stdev 0 --output-stdev 0 \
        --num-requests 4 > /tmp/benchmarking_128k.txt

cat <<EOF > /tmp/extra-llm-api-config.yml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2]
moe_config:
  max_num_tokens: 16384
EOF

trtllm-bench -m deepseek-ai/DeepSeek-R1 --model_path ${DS_R1_NVFP4_MODEL_PATH} throughput \
        --tp 8 --ep 8 \
        --warmup 0 \
        --dataset /tmp/benchmarking_128k.txt \
        --max_batch_size 2 \
        --max_num_tokens 131074 \
        --kv_cache_free_gpu_mem_fraction 0.3 \
        --extra_llm_api_options /tmp/extra-llm-api-config.yml
```

## Evaluation

Evaluate the model accuracy using `trtllm-eval`.

1. (Optional) Prepare an advanced configuration file:
```bash
cat >./extra-llm-api-config.yml <<EOF
enable_attention_dp: true
EOF
```

2. Evaluate accuracy on the [MMLU](https://people.eecs.berkeley.edu/~hendrycks/data.tar) dataset:
```bash
trtllm-eval --model  <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options ./extra-llm-api-config.yml \
  mmlu
```

3. Evaluate accuracy on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset:
```bash
trtllm-eval --model  <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options ./extra-llm-api-config.yml \
  gsm8k
```

4. Evaluate accuracy on the [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) dataset:
```bash
# Ensure signing up a huggingface account with access to the GPQA dataset

trtllm-eval --model  <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options ./extra-llm-api-config.yml \
  gpqa_diamond \
  --apply_chat_template
```

## Serving
### trtllm-serve

Below are example B200 serving configurations for both min-latency and max-throughput in FP4 and FP8. If you want to explore configurations, see the [blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md). **Treat these as starting points—tune for your model and workload to achieve the best performance.**

To serve the model using `trtllm-serve`:

#### B200 FP4 min-latency config
```bash
cat >./extra-llm-api-config.yml <<EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 1024
enable_attention_dp: false
kv_cache_config:
    dtype: fp8
stream_interval: 10
EOF
```

#### B200 FP4 max-throughput config
```bash
cat >./extra-llm-api-config.yml <<EOF
cuda_graph_config:
  enable_padding: true
  batch_sizes:
  - 1024
  - 896
  - 512
  - 256
  - 128
  - 64
  - 32
  - 16
  - 8
  - 4
  - 2
  - 1
kv_cache_config:
  dtype: fp8
stream_interval: 10
enable_attention_dp: true
EOF
```

#### B200 FP8 min-latency config
```bash
cat >./extra-llm-api-config.yml <<EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 1024
enable_attention_dp: false
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
stream_interval: 10
moe_config:
    backend: DEEPGEMM
    max_num_tokens: 37376
EOF
```

#### B200 FP8 max-throughput config
```bash
cat >./extra-llm-api-config.yml <<EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 512
enable_attention_dp: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
stream_interval: 10
moe_config:
    backend: DEEPGEMM
EOF
```
#### Launch trtllm-serve OpenAI-compatible API server
```bash
trtllm-serve \
  deepseek-ai/DeepSeek-R1 \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 1024 \
  --max_num_tokens 8192 \
  --tp_size 8 \
  --ep_size 8 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.9 \
  --extra_llm_api_options ./extra-llm-api-config.yml
```
It's possible seeing OOM issues on some configs. Considering reducing `kv_cache_free_gpu_mem_fraction` to a smaller value as a workaround. We're working on the investigation and addressing the problem. If you are using max-throughput config, reduce `max_num_tokens` to `3072` to avoid OOM issues.

To query the server, you can start with a `curl` command:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "deepseek-ai/DeepSeek-R1",
      "prompt": "Where is New York?",
      "max_tokens": 16,
      "temperature": 0
  }'
```

For DeepSeek-R1 FP4, use the model name `nvidia/DeepSeek-R1-FP4-v2`.  
For DeepSeek-V3, use the model name `deepseek-ai/DeepSeek-V3`.

### Disaggregated Serving

To serve the model in disaggregated mode, you should launch context and generation servers using `trtllm-serve`.

For example, you can launch a single context server on port 8001 with:

```bash
export TRTLLM_USE_UCX_KVCACHE=1

cat >./ctx-extra-llm-api-config.yml <<EOF
print_iter_log: true
enable_attention_dp: true
EOF

trtllm-serve \
  deepseek-ai/DeepSeek-V3 \
  --host localhost \
  --port 8001 \
  --backend pytorch \
  --max_batch_size 161 \
  --max_num_tokens 1160 \
  --tp_size 8 \
  --ep_size 8 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --extra_llm_api_options ./ctx-extra-llm-api-config.yml &> output_ctx &
```

And you can launch two generation servers on port 8002 and 8003 with:

```bash
export TRTLLM_USE_UCX_KVCACHE=1

cat >./gen-extra-llm-api-config.yml <<EOF
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
print_iter_log: true
enable_attention_dp: true
EOF

for port in {8002..8003}; do \
trtllm-serve \
  deepseek-ai/DeepSeek-V3 \
  --host localhost \
  --port ${port} \
  --backend pytorch \
  --max_batch_size 161 \
  --max_num_tokens 1160 \
  --tp_size 8 \
  --ep_size 8 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --extra_llm_api_options ./gen-extra-llm-api-config.yml \
  &> output_gen_${port} & \
done
```

Finally, you can launch the disaggregated server which will accept requests from the client and do
the orchestration between the context and generation servers with:

```bash
cat >./disagg-config.yaml <<EOF
hostname: localhost
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
      - "localhost:8001"
generation_servers:
  num_instances: 2
  urls:
      - "localhost:8002"
      - "localhost:8003"
EOF

trtllm-serve disaggregated -c disagg-config.yaml
```

To query the server, you can start with a `curl` command:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "deepseek-ai/DeepSeek-V3",
      "prompt": "Where is New York?",
      "max_tokens": 16,
      "temperature": 0
  }'
```

For DeepSeek-R1, use the model name `deepseek-ai/DeepSeek-R1`.

Note that the optimal disaggregated serving configuration (i.e. tp/pp/ep mappings, number of ctx/gen instances, etc.) will depend
on the request parameters, the number of concurrent requests and the GPU type. It is recommended to experiment to identify optimal
settings for your specific use case.

### Dynamo

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.
Dynamo supports TensorRT LLM as one of its inference engine. For details on how to use TensorRT LLM with Dynamo please refer to [LLM Deployment Examples using TensorRT-LLM](https://github.com/ai-dynamo/dynamo/blob/main/examples/tensorrt_llm/README.md)

### tensorrtllm_backend for triton inference server (Prototype)
To serve the model using [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend.git), make sure the version is v0.19+ in which the pytorch path is added as a prototype feature.

The model configuration file is located at https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/llmapi/tensorrt_llm/1/model.yaml

```bash
model: <replace with the deepseek model or path to the checkpoints>
backend: "pytorch"
```
Additional configs similar to `extra-llm-api-config.yml` can be added to the yaml file and will be used to configure the LLM model. At the minimum, `tensor_parallel_size` needs to be set to 8 on H200 and B200 machines and 16 on H100.

The initial loading of the model can take around one hour and the following runs will take advantage of the weight caching.

To send requests to the server, try:
```bash
curl -X POST localhost:8000/v2/models/tensorrt_llm/generate -d '{"text_input": "Hello, my name is", "sampling_param_temperature":0.8, "sampling_param_top_p":0.95}' | sed 's/^data: //' | jq
```
Available parameters for the requests are listed in https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/all_models/llmapi/tensorrt_llm/config.pbtxt.


## Advanced Usages
### Multi-node
TensorRT LLM supports multi-node inference. You can use mpirun or Slurm to launch multi-node jobs. We will use two nodes for this example.

#### mpirun
mpirun requires each node to have passwordless ssh access to the other node. We need to setup the environment inside the docker container. Run the container with host network and mount the current directory as well as model directory to the container.

```bash
# use host network
IMAGE=<YOUR_IMAGE>
NAME=test_2node_docker
# host1
docker run -it --name ${NAME}_host1 --ipc=host --gpus=all --network host --privileged --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace -v <YOUR_MODEL_DIR>:/models/DeepSeek-V3 -w /workspace ${IMAGE}
# host2
docker run -it --name ${NAME}_host2 --ipc=host --gpus=all --network host --privileged --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace -v <YOUR_MODEL_DIR>:/models/DeepSeek-V3 -w /workspace ${IMAGE}
```

Set up ssh inside the container

```bash
apt-get update && apt-get install -y openssh-server

# modify /etc/ssh/sshd_config
PermitRootLogin yes
PubkeyAuthentication yes
# modify /etc/ssh/sshd_config, change default port 22 to another unused port
port 2233

# modify /etc/ssh
```

Generate ssh key on host1 and copy to host2, vice versa.

```bash
# on host1
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@<HOST2>
# on host2
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@<HOST1>

# restart ssh service on host1 and host2
service ssh restart # or
/etc/init.d/ssh restart # or
systemctl restart ssh
```

You can use the following example to test mpi communication between two nodes:
```cpp
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Hello world from rank %d out of %d processors\n", world_rank, world_size);

    MPI_Finalize();
    return 0;
}
```

Compile and run the hello world program on two nodes:
```bash
mpicc -o mpi_hello_world mpi_hello_world.c
mpirun -np 2 -H <HOST1>:1,<HOST2>:1 -mca plm_rsh_args "-p 2233" ./mpi_hello_world
# Hello world from rank 11 out of 16 processors
# Hello world from rank 13 out of 16 processors
# Hello world from rank 12 out of 16 processors
# Hello world from rank 15 out of 16 processors
# Hello world from rank 14 out of 16 processors
# Hello world from rank 10 out of 16 processors
# Hello world from rank 9 out of 16 processors
# Hello world from rank 8 out of 16 processors
# Hello world from rank 5 out of 16 processors
# Hello world from rank 2 out of 16 processors
# Hello world from rank 4 out of 16 processors
# Hello world from rank 6 out of 16 processors
# Hello world from rank 3 out of 16 processors
# Hello world from rank 1 out of 16 processors
# Hello world from rank 7 out of 16 processors
# Hello world from rank 0 out of 16 processors
```

Prepare the dataset and configuration files on two nodes as mentioned above. Then you can run the benchmark on two nodes using mpirun:

```bash
mpirun \
--output-filename bench_log_2node_ep8_tp16_attndp_on_sl1000 \
-H <HOST1>:8,<HOST2>:8 \
-mca plm_rsh_args "-p 2233" \
--allow-run-as-root -n 16 \
trtllm-llmapi-launch trtllm-bench --model deepseek-ai/DeepSeek-V3 --model_path /models/DeepSeek-V3 throughput --max_batch_size 161 --max_num_tokens 1160 --dataset /workspace/tensorrt_llm/dataset_isl1000.txt --tp 16 --ep 8 --kv_cache_free_gpu_mem_fraction 0.95 --extra_llm_api_options /workspace/tensorrt_llm/extra-llm-api-config.yml --concurrency 4096 --streaming
```

#### Slurm
```bash
  srun -N 2 -w [NODES] \
  --output=benchmark_2node.log \
  --ntasks 16 --ntasks-per-node=8 \
  --mpi=pmix --gres=gpu:8 \
  --container-image=<CONTAINER_IMG> \
  --container-mounts=/workspace:/workspace \
  --container-workdir /workspace \
  bash -c "trtllm-llmapi-launch trtllm-bench --model deepseek-ai/DeepSeek-V3 --model_path <YOUR_MODEL_DIR> throughput --max_batch_size 161 --max_num_tokens 1160 --dataset /workspace/dataset.txt --tp 16 --ep 4 --kv_cache_free_gpu_mem_fraction 0.95 --extra_llm_api_options ./extra-llm-api-config.yml"
```


#### Example: Multi-node benchmark on GB200 Slurm cluster

Step 1: Prepare dataset and `extra-llm-api-config.yml`.
```bash
python3 /path/to/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
    --tokenizer=/path/to/DeepSeek-R1 \
    --stdout token-norm-dist --num-requests=49152 \
    --input-mean=1024 --output-mean=2048 --input-stdev=0 --output-stdev=0 > /tmp/dataset.txt

cat >/path/to/TensorRT-LLM/extra-llm-api-config.yml <<EOF
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
print_iter_log: true
enable_attention_dp: true
EOF
```

Step 2: Prepare `benchmark.slurm`.
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --time=02:00:00
#SBATCH --job-name=<job_name>

srun --container-image=${container_image} --container-mounts=${mount_dir}:${mount_dir} --mpi=pmix \
    --output ${logdir}/bench_%j_%t.srun.out \
    bash benchmark.sh
```

Step 3: Prepare `benchmark.sh`.
```bash
#!/bin/bash
cd /path/to/TensorRT-LLM
# pip install build/tensorrt_llm*.whl
if [ $SLURM_LOCALID == 0 ];then
    pip install build/tensorrt_llm*.whl
    echo "Install dependencies on rank 0."
else
    echo "Sleep 60 seconds on other ranks."
    sleep 60
fi

export PATH=${HOME}/.local/bin:${PATH}
export PYTHONPATH=/path/to/TensorRT-LLM
DS_R1_NVFP4_MODEL_PATH=/path/to/DeepSeek-R1  # optional

trtllm-llmapi-launch trtllm-bench \
    --model deepseek-ai/DeepSeek-R1 \
    --model_path $DS_R1_NVFP4_MODEL_PATH \
    throughput \
    --num_requests 49152 \
    --max_batch_size 384 --max_num_tokens 1536 \
    --concurrency 3072 \
    --dataset /path/to/dataset.txt \
    --tp 8 --pp 1 --ep 8 --kv_cache_free_gpu_mem_fraction 0.85 \
    --extra_llm_api_options ./extra-llm-api-config.yml --warmup 0
```

Step 4: Submit the job to Slurm cluster to launch the benchmark by executing:
```
sbatch --nodes=2 --ntasks=8 --ntasks-per-node=4 benchmark.slurm
```


### DeepGEMM
TensorRT LLM uses DeepGEMM for DeepSeek-V3/R1, which provides significant e2e performance boost on Hopper GPUs. DeepGEMM can be disabled by setting the environment variable `TRTLLM_DG_ENABLED` to `0`:

DeepGEMM-related behavior can be controlled by the following environment variables:

| Environment Variable | Description |
| ----------------------------- | ----------- |
| `TRTLLM_DG_ENABLED` | When set to `0`, disable DeepGEMM. |
| `TRTLLM_DG_JIT_DEBUG` | When set to `1`, enable JIT debugging. |
| `TRTLLM_DG_JIT_USE_NVCC` | When set to `1`, use NVCC instead of NVRTC to compile the kernel, which has slightly better performance but requires CUDA Toolkit (>=12.3) and longer compilation time.|
| `TRTLLM_DG_JIT_DUMP_CUBIN` | When set to `1`, dump the cubin file. This is only effective with NVRTC since NVCC will always dump the cubin file. NVRTC-based JIT will store the generated kernels in memory by default. If you want to persist the kernels across multiple runs, you can either use this variable or use NVCC. |

#### MOE GEMM Optimization

For Mixture of Experts (MOE) GEMM operations, TensorRT-LLM's DeepGEMM includes the optimized `fp8_gemm_kernel_swapAB` kernel. This kernel is automatically selected based on the input dimensions and GPU type:

- On H20 GPUs (SM count = 78): Uses `fp8_gemm_kernel_swapAB` when the expected m_per_expert is less than 64
- On H100/H200 GPUs: Uses `fp8_gemm_kernel_swapAB` when the expected m_per_expert is less than 32
- Otherwise, uses the original `fp8_gemm_kernel`

This automatic selection provides better performance for different workload sizes across various Hopper GPUs. In our test cases, the `fp8_gemm_kernel_swapAB` kernel achieves up to 1.8x speedup for individual kernels on H20 GPUs and up to 1.3x speedup on H100 GPUs.

#### Dense GEMM Optimization

The same optimization has been extended to Dense GEMM operations. For regular dense matrix multiplications:

- On all Hopper GPUs (H20, H100, H200): Uses `fp8_gemm_kernel_swapAB` when the m is less than 32
- Otherwise, uses the original `fp8_gemm_kernel`

This optimization delivers significant performance improvements for small batch sizes. Our benchmarks show that the `fp8_gemm_kernel_swapAB` kernel achieves up to 1.7x speedup on H20 GPUs and up to 1.8x speedup on H100 GPUs for certain matrix dimensions.

```bash
#single-node
trtllm-bench \
      --model deepseek-ai/DeepSeek-V3 \
      --model_path /models/DeepSeek-V3 \
      throughput \
      --max_batch_size ${MAX_BATCH_SIZE} \
      --max_num_tokens ${MAX_NUM_TOKENS} \
      --dataset dataset.txt \
      --tp 8 \
      --ep 8 \
      --kv_cache_free_gpu_mem_fraction 0.9 \
      --extra_llm_api_options /workspace/extra-llm-api-config.yml \
      --concurrency ${CONCURRENCY} \
      --num_requests ${NUM_REQUESTS} \
      --streaming \
      --report_json "${OUTPUT_FILENAME}.json"

# multi-node
mpirun -H <HOST1>:8,<HOST2>:8 \
      -n 16 \
      -x "TRTLLM_DG_ENABLED=1" \
      -x "CUDA_HOME=/usr/local/cuda" \
      trtllm-llmapi-launch trtllm-bench \
      --model deepseek-ai/DeepSeek-V3 \
      --model_path /models/DeepSeek-V3 \
      throughput \
      --max_batch_size ${MAX_BATCH_SIZE} \
      --max_num_tokens ${MAX_NUM_TOKENS} \
      --dataset dataset.txt \
      --tp 16 \
      --ep 16 \
      --kv_cache_free_gpu_mem_fraction 0.9 \
      --extra_llm_api_options /workspace/extra-llm-api-config.yml \
      --concurrency ${CONCURRENCY} \
      --num_requests ${NUM_REQUESTS} \
      --streaming \
      --report_json "${OUTPUT_FILENAME}.json"
```

### FlashMLA
TensorRT LLM has already integrated FlashMLA in the PyTorch backend. It is enabled automatically when running DeepSeek-V3/R1.

### FP8 KV Cache and MLA

FP8 KV Cache and MLA quantization could be enabled, which delivers two key performance advantages:
- Compression of the latent KV cache enables larger batch sizes, resulting in higher throughput;
- MLA kernel of the generation phase is accelerated by FP8 arithmetic and reduced KV cache memory access.

FP8 KV Cache and MLA is supported on Hopper and Blackwell. The accuracy loss is small, with GSM8k accuracy drop less than 1%.
- On Hopper we use the [FP8 FlashMLA kernel](https://github.com/deepseek-ai/FlashMLA/pull/54) from community.
- On Blackwell we use the kernel generated from an internal code-gen based solution called `trtllm-gen`.

You can enable FP8 MLA through either of these methods:

**Option 1: Checkpoint config**

TensorRT LLM automatically detects the `hf_quant_config.json` file in the model directory, which configures both GEMM and KV cache quantization. For example, see the FP4 DeepSeek-R1 checkpoint [configuration](https://huggingface.co/nvidia/DeepSeek-R1-FP4/blob/main/hf_quant_config.json) provided by [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

To enable FP8 MLA, modify the `kv_cache_quant_algo` property. The following shows the config for DeepSeek's block-wise FP8 GEMM quantization + FP8 MLA:

```json
{
  "quantization": {
    "quant_algo": "FP8_BLOCK_SCALES",
    "kv_cache_quant_algo": "FP8"
  }
}
```

**Option 2: PyTorch backend config**

Alternatively, configure FP8 MLA through the `kv_cache_dtype` of the PyTorch backend config. An example is to use `--kv_cache_dtype` of `quickstart_advanced.py`. Also, you can edit `extra-llm-api-config.yml` consumed by `--extra_llm_api_options` of `trtllm-serve`, `trtllm-bench` and so on:
```yaml
# ...
kv_cache_dtype: fp8
# ...
```

### W4AFP8

TensorRT LLM supports W(INT)4-A(FP)8 for DeepSeek on __Hopper__. Activations and weights are quantized at per-tensor and per-group (1x128) granularity respectively for MoE, and FP8 block scaling is preserved for dense layers.

We provide a pre-quantized checkpoint for DeepSeek-R1 W4AFP8 at [HF model hub](https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8).

```bash
python quickstart_advanced.py --model_dir <W4AFP8 Checkpoint> --tp_size 8
```
Or you can follow the steps to generate one by yourselves.

#### Activation calibration

[ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) is used for calibrating activations of MoE layers. We provide a calibrated file at [HF model hub](https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main/act_scales.safetensors) or you can run the following commands to generate by yourselves.

```bash
# Make sure for enough GPU resources (8xH200s) to run the following commands
PATH_OF_DEEPSEEK_R1=/llm-models/DeepSeek-R1/DeepSeek-R1

# Install ModelOpt from source
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer/ && cd modelopt
pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com

# Clone DeepSeek-V3 (base model of R1) Github repository for FP8 inference,
git clone https://github.com/deepseek-ai/DeepSeek-V3.git && cd DeepSeek-V3 && git checkout 1398800

# Convert the HF checkpoint to a specific format for DeepSeek
python inference/convert.py --hf-ckpt-path $PATH_OF_DEEPSEEK_R1 --save-path ds_r1 --n-experts 256 --model-parallel 8 && cd ..

# Do per-tensor fp8 calibration
torchrun --nproc-per-node 8 --master_port=12346 ptq.py --model_path DeepSeek-V3/ds_r1 --config DeepSeek-V3/inference/configs/config_671B.json --quant_cfg FP8_DEFAULT_CFG --output_path ds_r1_fp8_per_tensor_calibration && cd ../..
```

#### Weight quantization and assembling

You can run the following bash to quantize weights and generate the full checkpoint.
```bash
#!/bin/bash
HF_MODEL_DIR=/models/DeepSeek-R1/DeepSeek-R1/
OUTPUT_DIR=/workspace/ckpt/
# Safetensors or ModelOpt exported FP8 checkpoint path is accepted
# e.g. ACT_SCALES=ds_r1_fp8_per_tensor_calibration
ACT_SCALES=/workspace/act_scales.safetensors

if [ ! -d "convert_logs" ]; then
    mkdir convert_logs
fi

pids=()
for i in 0 1 2 3 4 5 6 7
do
    python examples/quantization/quantize_mixed_precision_moe.py --model_dir $HF_MODEL_DIR --output_dir $OUTPUT_DIR --act_scales $ACT_SCALES --parts 9 --rank $i > convert_logs/log_$i 2>&1 &
    pids+=($!)
done

python examples/quantization/quantize_mixed_precision_moe.py --model_dir $HF_MODEL_DIR --output_dir $OUTPUT_DIR --act_scales $ACT_SCALES --parts 9 --rank 8 > convert_logs/log_8 2>&1
pids+=($!)

for pid in ${pids[@]}; do
    wait $pid
done

echo "All processes completed!"
```

The converted checkpoint could be used as `<YOUR_MODEL_DIR>` and consumed by other commands.

### KV Cache Reuse
KV cache reuse is supported for MLA on SM90, SM100 and SM120. It is enabled by default. Due to extra operations like memcpy and GEMMs, GPU memory consumption may be higher and the E2E performance may have regression in some cases. Users could pass `KvCacheConfig(enable_block_reuse=False)` to LLM API to disable it.

### Chunked Prefill
Chunked Prefill is supported for MLA only on SM90 and SM100 currently. You should add `--enable_chunked_prefill` to enable it. The GPU memory consumption is highly correlated with `max_num_tokens` and `max_batch_size`. If encountering out-of-memory errors, you may make these values smaller. (`max_num_tokens` must be divisible by kv cache's `tokens_per_block`)

More specifically, we can imitate what we did in the [Quick Start](#quick-start):

``` bash
cd examples/llm-api
python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --enable_chunked_prefill
```

## Notes and Troubleshooting

- **Model Directory:** Update `<YOUR_MODEL_DIR>` with the actual path where the model weights reside.
- **GPU Memory:** Adjust `--max_batch_size` and `--max_num_tokens` if you encounter out-of-memory errors.
- **Logs:** Check `/workspace/trt_bench.log` for detailed performance information and troubleshooting messages.
- **Configuration Files:** Verify that the configuration files are correctly formatted to avoid runtime issues.
