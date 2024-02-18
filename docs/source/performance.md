# Performance of TensorRT-LLM

This document summarizes performance measurements of TensorRT-LLM on H100
(Hopper), L40S (Ada) and A100 (Ampere) GPUs for a few key models.

The data in the following tables is provided as a reference point to help users
validate observed performance. It should not be considered as the peak
performance that can be delivered by TensorRT-LLM.

## Methodology

The different performance numbers below were collected using the methodology
described in the benchmarks [folder](source:benchmarks/).

## Peak Throughput

The below tables provide reference data at large batch sizes, representing
high throughput offline tasks.

This data has been updated for v0.6.1, unless specified.

### H100 GPUs (FP8)

| Model                        | Batch Size | TP (1)    | Input Length | Output Length | Throughput (out tok/s/GPU) |
| :--------------------------- | :--------- | :-------- | :----------- | :------------ | -------------------------: |
| GPT-J 6B                     | 1024       | 1         | 128          | 128           |                     26,150 |
| GPT-J 6B                     | 120        | 1         | 128          | 2048          |                      8,011 |
| GPT-J 6B                     | 64         | 1         | 2048         | 128           |                      2,551 |
| GPT-J 6B                     | 64         | 1         | 2048         | 2048          |                      3,327 |
|                              |            |           |              |               |                            |
| LLaMA 7B                     | 768        | 1         | 128          | 128           |                     19,694 |
| LLaMA 7B                     | 112        | 1         | 128          | 2048          |                      6,818 |
| LLaMA 7B                     | 80         | 1         | 2048         | 128           |                      2,244 |
| LLaMA 7B                     | 48         | 1         | 2048         | 2048          |                      2,740 |
|                              |            |           |              |               |                            |
| LLaMA 70B                    | 1024       | 2         | 128          | 128           |                      2,657 |
| LLaMA 70B                    | 480        | 4         | 128          | 2048          |                      1,486 |
| LLaMA 70B                    | 96         | 2         | 2048         | 128           |                        306 |
| LLaMA 70B                    | 64         | 2         | 2048         | 2048          |                        547 |
|                              |            |           |              |               |                            |
| Falcon 180B                  | 1024       | 4         | 128          | 128           |                        987 |
| Falcon 180B                  | 1024       | 8         | 128          | 2048          |                        724 |
| Falcon 180B                  | 64         | 4         | 2048         | 128           |                        112 |
| Falcon 180B                  | 64         | 4         | 2048         | 2048          |                        264 |

### L40S GPUs (FP8)<sup>*</sup>

<sup> * The following data is from TensorRT-LLM v0.5. </sup>


| Model                        | Batch Size | TP (1)    | Input Length | Output Length | Throughput (out tok/s/GPU) |
| :--------------------------- | :--------- | :-------- | :----------- | :------------ | ---------------------: |
| GPT-J 6B                     | 64         | 1         | 128          | 128           |                  3,630 |
| GPT-J 6B                     | 64         | 1         | 128          | 2048          |                  1,859 |
| GPT-J 6B                     | 32         | 1         | 2048         | 128           |                    616 |
| GPT-J 6B                     | 32         | 1         | 2048         | 2048          |                    757 |
|                              |            |           |              |               |                        |
| LLaMA 7B                     | 64         | 1         | 128          | 128           |                  3,240 |
| LLaMA 7B                     | 64         | 1         | 128          | 2048          |                  1,622 |
| LLaMA 7B                     | 32         | 1         | 2048         | 128           |                    581 |
| LLaMA 7B                     | 16         | 1         | 2048         | 2048          |                    531 |


### A100 GPUs (FP16)

| Model                        | Batch Size | TP (1)    | Input Length | Output Length | Throughput (out tok/s/GPU) |
| :--------------------------- | :--------- | :-------- | :----------- | :------------ | ---------------------: |
| GPT-J 6B                     | 512        | 1         | 128          | 128           |                  6,374 |
| GPT-J 6B                     | 120        | 2         | 128          | 2048          |                  2,192 |
| GPT-J 6B                     | 60         | 1         | 2048         | 128           |                    670 |
| GPT-J 6B                     | 64         | 2         | 2048         | 2048          |                    903 |
|                              |            |           |              |               |                        |
| LLaMA 7B                     | 384        | 1         | 128          | 128           |                  5,586 |
| LLaMA 7B                     | 60         | 1         | 128          | 2048          |                  1,928 |
| LLaMA 7B                     | 52         | 1         | 2048         | 128           |                    591 |
| LLaMA 7B                     | 64         | 2         | 2048         | 2048          |                    782 |
|                              |            |           |              |               |                        |
| LLaMA 70B                    | 1280       | 4         | 128          | 128           |                    670 |
| LLaMA 70B                    | 240        | 4         | 128          | 2048          |                    525 |
| LLaMA 70B                    | 120        | 4         | 2048         | 128           |                     79 |
|                              |            |           |              |               |                        |
| Falcon 180B                  | 1024       | 8         | 128          | 128           |                    232 |
| Falcon 180B                  | 128        | 8         | 128          | 2048          |                    180 |

(1) TP stands for Tensor Parallelism.

## Low Latency<sup>**</sup>

<sup> ** The following data is from TensorRT-LLM v0.5. Low latency numbers will soon be updated to reflect real time latency with infight-batching.</sup>

The below tables provide reference data at batch size 1 for first token
latency, representing end-user's perceived latency for online streaming
tasks.

### H100 GPUs (FP8)

| Model                        | Batch Size | TP (1)    | Input Length | 1st Token Latency (ms) |
| :--------------------------- | :--------- | :-------- | :----------- | ---------------------: |
| GPT-J 6B                     | 1          | 1         | 128          |                      7 |
| GPT-J 6B                     | 1          | 1         | 2048         |                     29 |
|                              |            |           |              |                        |
| LLaMA 7B                     | 1          | 1         | 128          |                      7 |
| LLaMA 7B                     | 1          | 1         | 2048         |                     36 |
|                              |            |           |              |                        |
| LLaMA 70B                    | 1          | 4         | 128          |                     26 |
| LLaMA 70B                    | 1          | 4         | 2048         |                    109 |
|                              |            |           |              |                        |
| Falcon 180B                  | 1          | 8         | 128          |                     27 |
| Falcon 180B                  | 1          | 8         | 2048         |                    205 |

### L40S GPUs (FP8)

| Model                        | Batch Size | TP (1)    | Input Length | 1st Token Latency (ms) |
| :--------------------------- | :--------- | :-------- | :----------- | ---------------------: |
| GPT-J 6B                     | 1          | 1         | 128          |                     12 |
| GPT-J 6B                     | 1          | 1         | 2048         |                     71 |
|                              |            |           |              |                        |
| LLaMA 7B                     | 1          | 1         | 128          |                     14 |
| LLaMA 7B                     | 1          | 1         | 2048         |                     73 |

### A100 GPUs (FP16)

| Model                        | Batch Size | TP (1)    | Input Length | 1st Token Latency (ms) |
| :--------------------------- | :--------- | :-------- | :----------- | ---------------------: |
| GPT-J 6B                     | 1          | 1         | 128          |                     12 |
| GPT-J 6B                     | 1          | 1         | 2048         |                    129 |
|                              |            |           |              |                        |
| LLaMA 7B                     | 1          | 1         | 128          |                     16 |
| LLaMA 7B                     | 1          | 1         | 2048         |                    133 |
|                              |            |           |              |                        |
| LLaMA 70B                    | 1          | 4         | 128          |                     47 |
| LLaMA 70B                    | 1          | 4         | 2048         |                    377 |
|                              |            |           |              |                        |
| Falcon 180B                  | 1          | 8         | 128          |                     61 |
| Falcon 180B                  | 1          | 8         | 2048         |                    509 |

(1) TP stands for Tensor Parallelism.


## Known Issues

The following issues are being addressed to improve the efficiency of TensorRT-LLM.

### Fused Matmul + Gated-SiLU (LLaMA)

The current implementation combines two Matmul operations into one Matmul followed by
a separate SwiGLU kernel (when `--use_fused_mlp` is enabled). The future release will
include a more efficient implementation that runs single Matmul + SwiGLU fused kernel.


## Reproducing Benchmarked Results

### Building the TensorRT-LLM Container

---
In order to benchmark TensorRT-LLM, you will need to follow the [Quick Start](../../README.md#quick-start)
build process to create a baseline container for building a wheel. Additionally, the development
container needs a copy of the source code to build the wheel and the benchmarking script. Create the
right build environment, use the following :

```shell
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs install
git lfs pull
make -C docker build
make -C docker run LOCAL_USER=1
```

> [!WARNING]
> If you have elevated privileges on your system, then skip the `make -C docker run LOCAL_USER=1`
command above as it may make it so that you cannot access some required system libraries within the
container because the build forces your UID and GID to match those that are set for your non-elevated
user. There are cases where the container will be booted as root (i.e. on some SLURM systems with
the pyxis plugin) which will cause libraries to be missing.

If you are benchmarking in a shared environment, you need to specify the GPU indices that you would
like the container to use, otherwise the Makefile defaults to loading the container with all GPUs on
the system. For example, if you only have the 4 higher indices of GPUs on your system you can
configure it using the following example:

```shell
NV_GPU=0,1,2,3
make -C docker run LOCAL_USER=1 GPU_OPTS='--gpus \"device=${NV_GPU}\"'
```

Additionally, if you'd like to mount external storage to access persistent storage, or previously
built engines, you can mount directories as follows (simply replace `source` and `destination` with
the appropriate paths):

```shell
make -C docker run LOCAL_USER=1 DOCKER_RUN_ARGS="-v /source:/destination"
```

Once the container starts, you'll need to build the wheel and the benchmarking scripts. From the
code root (the default directory when the container is loaded), the following commands will build
the TensorRT-LLM wheel, install dependencies, and build the benchmark scripts:

```shell
python3 ./scripts/build_wheel.py --benchmarks --trt_root /usr/local/tensorrt
pip install ./build/tensorrt_llm*.whl
```

## Methodology

### Engine Building Setups

Each engine needs to be built before they can be benchmarked, and requires the source code for each
of their respective build scripts. For smaller models, it is fine to build the engine on the fly in
container; however, for larger engines it is recommended to pre-build and mount a directory with the
engine because engine files are quite large and take time to repeatedly build. Additionally, built
engines can be used for input lengths, output lengths, and batch sizes *up to* their build options
meaning you can use an engine to benchmark multiple input configurations.

In order to benchmark the various networks, our engine building scheme is as follows:
- For the GPT-J, Llama2-7b, and Llama2-70b benchmarks were ran using a single-setting engine build
for each network configured for our maximum expected throughput.
- For Falcon-180B, where memory limits and model size have a higher impact for running the model,
our benchmarks transition to a per-configuration engine build.

Below we document how to benchmark each model on an H100-HBM3-80GB system and reproduce the throughput
numbers we document on our [Performance section](#performance of-tensorrt-llm).

### Running on A100

To run the benchmarks below on A100, you will need to remove the below fp8 quantization field from each
config json file, because FP8 computation is a feature in H100 and newer GPUs.
```json
"quantization": {
	"quant_algo": "FP8",
	"kv_cache_quant_algo": "FP8"
}
```

### Reproducing First Token Latency

In order to test the latency to the first token, you can build the engines as specified below (or
with the tweaks specified above on A100) -- once built as described in the
[build steps](#engine-building-setups) above, you can then benchmark with a single output token in
order to find the time to first token latency. We provide the appropriate command lines below for
each of the benchmarked models, but you can use this same method to benchmark other models available
in [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

## Benchmarking per Model

> [!WARNING]
> In some cases, using Group Query Attention (GQA) can improve performance of some networks. These
kernels are currently experimental and not enabled by default. In order to enable them, simply run
`export TRTLLM_ENABLE_XQA=1` in your shell. The kernels are an inference runtime optimization, so
previously built engines should still function. For the benchmarks below, we have enabled GQA where
our tests displayed performance benefits. If your network is not listed below, be sure to try both
GQA-enabled and GQA-disabled configurations to find the configuration that works best.
For more details see our documentation about [GPT Attention](./gpt_attention.md#generation-phase).

### GPT-J 6B

---
Prepare a config json file `/tmp/engines/gptj/ckpt_config.json`:
```json
{
    "architecture": "GPTJForCausalLM",
    "dtype": "float16",
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "hidden_size": 4096,
    "norm_epsilon": 1e-05,
    "vocab_size": 50400,
    "position_embedding_type": "rope_gptj",
    "max_position_embeddings": 2048,
    "hidden_act": "gelu",
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    },
    "rotary_dim": 64
}
```

Build an engine:
```shell
trtllm-build --model_config /tmp/engines/gptj/ckpt_config.json \
	--output_dir /tmp/engines/gptj \
	--context_fmha enable \
	--gpt_attention_plugin float16 \
	--max_batch_size 64 \
	--max_input_len 2048 \
	--max_output_len 2048 \
	--strongly_typed
```

#### Throughput Benchmark

```shell
in_out_sizes=("64:128,128" "64:128,2048" "64:2048,128" "64:2048,2048")
for in_out in ${in_out_sizes[@]}
do
	batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
	in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
	echo "BS: $batch_size, ISL/OSL: $in_out_dims"

	./cpp/build/benchmarks/gptSessionBenchmark --model gptj --engine_dir /tmp/engines/gptj/ --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len $in_out_dims
done
```

#### First Token Latency Benchmark

```shell
in_out_sizes=("64:128,1" "64:2048,1")
for in_out in ${in_out_sizes[@]}
do
	batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
	in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
	echo "BS: $batch_size, ISL/OSL: $in_out_dims"

	./cpp/build/benchmarks/gptSessionBenchmark --model gptj --engine_dir /tmp/engines/gptj/ --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len $in_out_dims
done
```


### Llama2-7b

---
Prepare a config json file `/tmp/engines/llama/7b/ckpt_config.json`:
```json
{
    "architecture": "LlamaForCausalLM",
    "dtype": "float16",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_key_value_heads": 32,
    "vocab_size": 32000,
    "position_embedding_type": "rope_gpt_neox",
    "max_position_embeddings": 4096,
    "hidden_act": "silu",
    "rotary_base": 10000.0,
    "rotary_scaling": null,
    "norm_epsilon": 1e-05,
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    }
}
```

Build an engine:
```shell
pip install -r examples/llama/requirements.txt
trtllm-build --model_config /tmp/engines/llama/7b/ckpt_config.json \
	--output_dir /tmp/engines/llama/7b \
	--remove_input_padding enable \
	--context_fmha enable \
	--gpt_attention_plugin float16 \
	--max_batch_size 64 \
	--max_input_len 2048 \
	--max_output_len 2048 \
	--strongly_typed
```

#### Throughput Benchmark

```shell
in_out_sizes=("64:128,128" "64:128,2048" "64:2048,128" "32:2048,2048")
for in_out in ${in_out_sizes[@]}
do
	batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
	in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
	echo "BS: $batch_size, ISL/OSL: $in_out_dims"

	./cpp/build/benchmarks/gptSessionBenchmark --model llama --engine_dir /tmp/engines/llama/7b --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len $in_out_dims
done
```
#### First Token Latency Benchmark

```shell
in_out_sizes=("64:128,1" "32:2048,1")
for in_out in ${in_out_sizes[@]}
do
	batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
	in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
	echo "BS: $batch_size, ISL/OSL: $in_out_dims"

	./cpp/build/benchmarks/gptSessionBenchmark --model llama --engine_dir /tmp/engines/llama/7b --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len $in_out_dims
done
```

### Llama2-70b

---
Prepare a config json file `/tmp/engines/llama/70b/ckpt_config.json`:
```json
{
    "architecture": "LlamaForCausalLM",
    "dtype": "float16",
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "position_embedding_type": "rope_gpt_neox",
    "max_position_embeddings": 4096,
    "hidden_act": "silu",
    "rotary_base": 10000.0,
    "rotary_scaling": null,
    "norm_epsilon": 1e-05,
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    },
    "mapping": {
        "world_size": 4,
        "tp_size": 4,
        "pp_size": 1
    }
}
```

Build an engine:
```shell
pip install -r examples/llama/requirements.txt
trtllm-build --model_config /tmp/engines/llama/70b/ckpt_config.json \
	--output_dir /tmp/engines/llama/70b \
	--workers 4 \
	--remove_input_padding enable \
	--context_fmha enable \
	--gpt_attention_plugin float16 \
	--max_batch_size 64 \
	--max_input_len 2048 \
	--max_output_len 2048 \
	--strongly_typed
```

#### Throughput Benchmark

```shell
export TRTLLM_ENABLE_XQA=1
in_out_sizes=("64:128,128" "64:128,2048" "64:2048,128" "64:2048,2048")
for in_out in ${in_out_sizes[@]}
do
	batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
	in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
	echo "BS: $batch_size, ISL/OSL: $in_out_dims"

	mpirun -n 4 --allow-run-as-root --oversubscribe ./cpp/build/benchmarks/gptSessionBenchmark --model llama --engine_dir /tmp/engines/llama/70b --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len $in_out_dims
done
```

#### First Token Latency Benchmark

```shell
export TRTLLM_ENABLE_XQA=1
in_out_sizes=("64:128,1" "64:128,1")
for in_out in ${in_out_sizes[@]}
do
	batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
	in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
	echo "BS: $batch_size, ISL/OSL: $in_out_dims"

	mpirun -n 4 --allow-run-as-root --oversubscribe ./cpp/build/benchmarks/gptSessionBenchmark --model llama --engine_dir /tmp/engines/llama/70b --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len $in_out_dims
done
```


### Falcon-180B

---

Benchmarking Falcon-180B requires a custom engine per batch size, input/output sequence length due
to the large footprint of the model and the large input size of 2048. You can build and benchmark
each engine one at a time with the following loop.

Prepare a config json file `/tmp/engines/falcon/180b/ckpt_config.json`:
```json
{
    "architecture": "FalconForCausalLM",
    "dtype": "float16",
    "num_hidden_layers": 80,
    "num_attention_heads": 232,
    "num_key_value_heads": 8,
    "hidden_size": 14848,
    "norm_epsilon": 1e-05,
    "vocab_size": 65024,
    "position_embedding_type": "rope_gpt_neox",
    "max_position_embeddings": 2048,
    "hidden_act": "gelu",
    "use_parallel_embedding": false,
    "embedding_sharding_dim": 0,
    "share_embedding_table": false,
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    },
    "mapping": {
        "world_size": 8,
        "tp_size": 8,
        "pp_size": 1
    },
    "bias": false,
    "parallel_attention": true,
    "new_decoder_architecture": true
}
```

```shell
export TRTLLM_ENABLE_XQA=1
# Benchmark specific batch size:isl:osl combinations.
in_out_sizes=("96:128,128" "96:128,2048" "64:2048,128")
for in_out in ${in_out_sizes[@]}
do
	batch_size=$(echo $in_out | awk -F':' '{ print $1 }')
	in_out_dims=$(echo $in_out | awk -F':' '{ print $2 }')
	isl=$(echo $in_out_dims | awk -F',' '{ print $1 }')
	osl=$(echo $in_out_dims | awk -F',' '{ print $2 }')
	engine_path="/tmp/engines/falcon/180b/${batch_size}_${isl}_${osl}"
	echo "BS: $batch_size, ISL/OSL: ${isl},${osl}"

	# Build the specific engine for the BS,ISL,OSL combination
	trtllm-build --model_config /tmp/engines/falcon/180b/ckpt_config.json \
		--output_dir $engine_path \
		--workers 8 \
		--remove_input_padding enable \
		--context_fmha enable \
		--gpt_attention_plugin float16 \
		--gemm_plugin float16 \
		--paged_kv_cache enable \
		--max_batch_size $batch_size \
		--max_input_len $isl \
		--max_output_len $osl \
		--strongly_typed

	# Throughput benchmark
	mpirun -n 8 --allow-run-as-root --oversubscribe ./cpp/build/benchmarks/gptSessionBenchmark --model falcon --engine_dir $engine_path --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len "${isl},${osl}"
	# Time to first token benchmark
	mpirun -n 8 --allow-run-as-root --oversubscribe ./cpp/build/benchmarks/gptSessionBenchmark --model falcon --engine_dir $engine_path --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len "${isl},1"

	# The Falcon-180b engine is quite large, remove after the benchmark to free up space
	# Remove this line if you'd like to save the engines.
	rm -r $engine_path
done
```
