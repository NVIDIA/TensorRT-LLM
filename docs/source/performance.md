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

To run the benchmarks below on A100, you will need to remove the `--enable_fp8 --fp8_kv_cache` options
from each engine build command because FP8 computation is a feature in H100 and newer GPUs.

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
```shell
python examples/gptj/build.py \
	--enable_context_fmha \
	--parallel_build \
	--output_dir /tmp/engines/gptj \
	--dtype float16 \
	--use_gpt_attention_plugin float16 \
	--world_size 1 \
	--max_batch_size 64 \
	--max_input_len 2048 \
	--max_output_len 2048 \
	--hidden_act gelu \
	--enable_fp8 \
	--fp8_kv_cache \
	--strongly_typed \
	--n_layer 28 \
	--n_head 16 \
	--n_embd 4096 \
	--n_positions 2048 \
	--enable_two_optimization_profiles
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
```shell
pip install -r examples/llama/requirements.txt
python examples/llama/build.py \
	--remove_input_padding \
	--enable_context_fmha \
	--parallel_build \
	--output_dir /tmp/engines/llama/7b \
	--dtype float16 \
	--use_gpt_attention_plugin float16 \
	--world_size 1 \
	--tp_size 1 \
	--pp_size 1 \
	--max_batch_size 64 \
	--max_input_len 2048 \
	--max_output_len 2048 \
	--enable_fp8 \
	--fp8_kv_cache \
	--strongly_typed \
	--n_layer 32 \
	--n_head 32 \
	--n_embd 4096 \
	--inter_size 11008 \
	--vocab_size 32000 \
	--n_positions 4096 \
	--hidden_act silu
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

```shell
pip install -r examples/llama/requirements.txt
python examples/llama/build.py \
	--remove_input_padding \
	--enable_context_fmha \
	--parallel_build \
	--output_dir /tmp/engines/llama/70b \
	--dtype float16 \
	--use_gpt_attention_plugin float16 \
	--world_size 4 \
	--tp_size 4 \
	--pp_size 1 \
	--max_batch_size 64 \
	--max_input_len 2048 \
	--max_output_len 2048 \
	--enable_fp8 \
	--fp8_kv_cache \
	--strongly_typed \
	--n_layer 80 \
	--n_head 64 \
	--n_kv_head 8 \
	--n_embd 8192 \
	--inter_size 28672 \
	--vocab_size 32000 \
	--n_positions 4096 \
	--hidden_act silu \
	--ffn_dim_multiplier 1.3 \
	--multiple_of 4096
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
	python examples/falcon/build.py \
		--use_inflight_batching \
		--paged_kv_cache \
		--remove_input_padding \
		--enable_context_fmha \
		--parallel_build \
		--output_dir $engine_path \
		--dtype float16 \
		--use_gemm_plugin float16 \
		--use_gpt_attention_plugin float16 \
		--world_size 8 \
		--tp 8 \
		--max_batch_size $batch_size \
		--max_input_len $isl \
		--max_output_len $osl \
		--enable_fp8 \
		--fp8_kv_cache \
		--n_layer 80 \
		--n_head 232 \
		--n_kv_head 8 \
		--n_embd 14848 \
		--vocab_size 65024 \
		--new_decoder_architecture
	# Throughput benchmark
	mpirun -n 8 --allow-run-as-root --oversubscribe ./cpp/build/benchmarks/gptSessionBenchmark --model falcon --engine_dir $engine_path --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len "${isl},${osl}"
	# Time to first token benchmark
	mpirun -n 8 --allow-run-as-root --oversubscribe ./cpp/build/benchmarks/gptSessionBenchmark --model falcon --engine_dir $engine_path --warm_up 1 --batch_size $batch_size --duration 0 --num_runs 5 --input_output_len "${isl},1"

	# The Falcon-180b engine is quite large, remove after the benchmark to free up space
	# Remove this line if you'd like to save the engines.
	rm -r $engine_path
done
```
