# DeepSeek‑V3 and DeepSeek-R1

This guide walks you through the examples to run the DeepSeek‑V3/DeepSeek-R1 models using NVIDIA's TensorRT-LLM framework with the PyTorch backend.
**DeepSeek-R1 and DeepSeek-V3 share exact same model architecture other than weights differences, and share same code path in TensorRT-LLM, for brevity we only provide one model example, the example command to be used interchangeablely by only replacing the model name to the other one**.

To benchmark the model with best configurations, refer to [DeepSeek R1 benchmarking blog](../../docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md).

Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html) for how to build TensorRT-LLM from source and start a TRT-LLM docker container.

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
    - [Run evaluation on GPQA dataset](#run-evaluation-on-gpqa-dataset)
  - [Serving](#serving)
  - [Advanced Usages](#advanced-usages)
    - [Multi-node](#multi-node)
      - [mpirun](#mpirun)
      - [Slurm](#slurm)
      - [Example: Multi-node benchmark on GB200 Slurm cluster](#example-multi-node-benchmark-on-gb200-slurm-cluster)
    - [DeepGEMM](#deepgemm)
    - [FlashMLA](#flashmla)
    - [FP8 KV Cache and MLA](#fp8-kv-cache-and-mla)
  - [Notes and Troubleshooting](#notes-and-troubleshooting)


## Hardware Requirements

DeepSeek-v3 has 671B parameters which needs about 671GB GPU memory for FP8 weights, and needs more memories for activation tensors and KV cache.
The minimum hardware requirements for running DeepSeek V3/R1 FP8&FP4 are listed as follows.

| GPU  | DeepSeek-V3/R1 FP8 | DeepSeek-V3/R1 FP4 |
| -------- | ------- | -- |
| H100 80GB | 16 | N/A |
| H20 141GB | 8 | N/A |
| H20 96GB | 8  | N/A |
| H200 | 8     | N/A |
| B200/GB200| Not supported yet, WIP | 4 (8 GPUs is recommended for best perf) |

Ampere architecture (SM80 & SM86) is not supported.


## Downloading the Model Weights

DeepSeek‑v3 model weights are available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3). To download the weights, execute the following commands (replace `<YOUR_MODEL_DIR>` with the target directory where you want the weights stored):

```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3 <YOUR_MODEL_DIR>
```


## Quick Start

### Run a single inference
To quickly run DeepSeek-V3, [examples/pytorch/quickstart_advanced.py](../pytorch/quickstart_advanced.py):

```bash
cd examples/pytorch
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
To run with MTP, use [examples/pytorch/quickstart_advanced.py](../pytorch/quickstart_advanced.py) with additional options, see
```bash
cd examples/pytorch
python quickstart_advanced.py --model_dir <YOUR_MODEL_DIR> --spec_decode_algo MTP --spec_decode_nextn N
```

`N` is the number of MTP modules. When `N` is equal to `0`, which means that MTP is not used (default). When `N` is greater than `0`, which means that `N` MTP modules are enabled. In the current implementation, the weight of each MTP module is shared.

### Run evaluation on GPQA dataset
Download the dataset first
1. Sign up a huggingface account and request the access to the gpqa dataset: https://huggingface.co/datasets/Idavidrein/gpqa
2. Download the csv file from https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/gpqa_diamond.csv

Evaluate on GPQA dataset.
```
python examples/gpqa_llmapi.py \
  --hf_model_dir <YOUR_MODEL_DIR> \
  --data_dir <DATASET_PATH> \
  --tp_size 8 \
  --use_cuda_graph \
  --enable_overlap_scheduler \
  --concurrency 32 \
  --batch_size 32 \
  --max_num_tokens 4096
```

## Serving

To serve the model using `trtllm-serve`:

```bash
cat >./extra-llm-api-config.yml <<EOF
pytorch_backend_config:
    use_cuda_graph: true
    cuda_graph_padding_enabled: true
    cuda_graph_batch_sizes:
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
    enable_overlap_scheduler: true
enable_attention_dp: true
EOF

trtllm-serve \
  deepseek-ai/DeepSeek-V3 \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 161 \
  --max_num_tokens 1160 \
  --tp_size 8 \
  --ep_size 8 \
  --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --extra_llm_api_options ./extra-llm-api-config.yml
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

## Advanced Usages
### Multi-node
TensorRT-LLM supports multi-node inference. You can use mpirun or Slurm to launch multi-node jobs. We will use two nodes for this example.

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
trtllm-llmapi-launch trtllm-bench --model deepseek-ai/DeepSeek-V3 --model_path /models/DeepSeek-V3 throughput --backend pytorch --max_batch_size 161 --max_num_tokens 1160 --dataset /workspace/tensorrt_llm/dataset_isl1000.txt --tp 16 --ep 8 --kv_cache_free_gpu_mem_fraction 0.95 --extra_llm_api_options /workspace/tensorrt_llm/extra-llm-api-config.yml --concurrency 4096 --streaming
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
  bash -c "trtllm-llmapi-launch trtllm-bench --model deepseek-ai/DeepSeek-V3 --model_path <YOUR_MODEL_DIR> throughput --backend pytorch --max_batch_size 161 --max_num_tokens 1160 --dataset /workspace/dataset.txt --tp 16 --ep 4 --kv_cache_free_gpu_mem_fraction 0.95 --extra_llm_api_options ./extra-llm-api-config.yml"
```


#### Example: Multi-node benchmark on GB200 Slurm cluster

Step 1: Prepare dataset and `extra-llm-api-config.yml`.
```bash
python3 /path/to/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
    --tokenizer=/path/to/DeepSeek-R1 \
    --stdout token-norm-dist --num-requests=49152 \
    --input-mean=1024 --output-mean=2048 --input-stdev=0 --output-stdev=0 > /tmp/dataset.txt

cat >/path/to/TensorRT-LLM/extra-llm-api-config.yml <<EOF
pytorch_backend_config:
    use_cuda_graph: true
    cuda_graph_padding_enabled: true
    cuda_graph_batch_sizes:
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
    enable_overlap_scheduler: true
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
    throughput --backend pytorch \
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
TensorRT-LLM uses DeepGEMM for DeepSeek-V3/R1, which provides significant e2e performance boost on Hopper GPUs. DeepGEMM can be disabled by setting the environment variable `TRTLLM_DG_ENABLED` to `0`:

DeepGEMM-related behavior can be controlled by the following environment variables:

| Environment Variable | Description |
| ----------------------------- | ----------- |
| `TRTLLM_DG_ENABLED` | When set to `0`, disable DeepGEMM. |
| `TRTLLM_DG_JIT_DEBUG` | When set to `1`, enable JIT debugging. |
| `TRTLLM_DG_JIT_USE_NVCC` | When set to `1`, use NVCC instead of NVRTC to compile the kernel, which has slightly better performance but requires CUDA Toolkit (>=12.3) and longer compilation time.|
| `TRTLLM_DG_JIT_DUMP_CUBIN` | When set to `1`, dump the cubin file. This is only effective with NVRTC since NVCC will always dump the cubin file. NVRTC-based JIT will store the generated kernels in memory by default. If you want to persist the kernels across multiple runs, you can either use this variable or use NVCC. |

```bash
#single-node
trtllm-bench \
      --model deepseek-ai/DeepSeek-V3 \
      --model_path /models/DeepSeek-V3 \
      throughput \
      --backend pytorch \
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
      --backend pytorch \
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
TensorRT-LLM has already integrated FlashMLA in the PyTorch backend. It is enabled automatically when running DeepSeek-V3/R1.

### FP8 KV Cache and MLA

FP8 KV Cache and MLA quantization could be enabled, which delivers two key performance advantages:
- Compression of the latent KV cache enables larger batch sizes, resulting in higher throughput;
- MLA kernel of the generation phase is accelerated by FP8 arithmetic and reduced KV cache memory access.

FP8 KV Cache and MLA is supported on Hopper and Blackwell.
- On Hopper we use the [FP8 FlashMLA kernel](https://github.com/deepseek-ai/FlashMLA/pull/54) from community. The accuracy loss is small, with GSM8k accuracy drop less than 1%.
- On Blackwell we use the kernel generated from an internal code-gen based solution called `trtllm-gen`. Note that FP8 MLA on Blackwell currently suffers from accuracy issues and there are ongoing efforts to solve it.

You can enable FP8 MLA through either of these methods:

**Option 1: Checkpoint config**

TensorRT-LLM automatically detects the `hf_quant_config.json` file in the model directory, which configures both GEMM and KV cache quantization. For example, see the FP4 DeepSeek-R1 checkpoint [configuration](https://huggingface.co/nvidia/DeepSeek-R1-FP4/blob/main/hf_quant_config.json) provided by [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

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
pytorch_backend_config:
  kv_cache_dtype: fp8
  # ...
```

## Notes and Troubleshooting

- **Model Directory:** Update `<YOUR_MODEL_DIR>` with the actual path where the model weights reside.
- **GPU Memory:** Adjust `--max_batch_size` and `--max_num_tokens` if you encounter out-of-memory errors.
- **Logs:** Check `/workspace/trt_bench.log` for detailed performance information and troubleshooting messages.
- **Configuration Files:** Verify that the configuration files are correctly formatted to avoid runtime issues.
