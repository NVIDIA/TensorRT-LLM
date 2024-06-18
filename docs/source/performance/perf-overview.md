(perf-overview)=

> [!IMPORTANT]
> As of TensorRT-LLM v0.10, these performance benchmarks have changed methodology to utilize in-flight batching and
no longer utilize static benchmarking. These numbers are initial measurements and are expected to improve in future
releases.

# Overview

This document summarizes performance measurements of TensorRT-LLM on H100
(Hopper), L40S (Ada) and A100 (Ampere) GPUs for a few key models.

The data in the following tables is provided as a reference point to help users
validate observed performance. It should not be considered as the peak
performance that can be delivered by TensorRT-LLM.

## Known Issues

The following issues are being addressed to improve the efficiency of TensorRT-LLM.

### Fused Matmul + Gated-SiLU (LLaMA)

The current implementation combines two Matmul operations into one Matmul followed by
a separate SwiGLU kernel (when `--use_fused_mlp` is enabled). There is also a more
efficient implementation that runs single Matmul + SwiGLU fused kernel for FP8 on Hopper
(when `--use_fused_mlp --gemm_swiglu_plugin fp8` is enabled). The gemm_swiglu_plugin
will support more data types and GPU architectures in the future release.

## Throughput Measurements

The below table shows performance data where a local inference client is fed requests at an infinite rate (no delay between messages),
and shows the throughput client-server scenario under maximum load.


The performance numbers below were collected using the steps described in this document.

**All data in the table below was generated using version 0.10.0 and presents token throughput in tokens/second.**

|              |                          |               |                 |                |                |                |         |         |
| ------------ | ------------------------ | ------------- | --------------- | -------------- | -------------- | -------------- | ------- | ------- |
|              |                          | **GPU**       | H200 141GB HBM3 | H100 80GB HBM3 | H100 80GB HBM3 | A100-SXM4-80GB | L40S    | L40S    |
|              |                          | **Precision** | FP8             | FP8            | FP16           | FP16           | FP8     | FP16    |
| **Model**    | **Input/Output Lengths** | **TP**        |                 |                |                |                |         |         |
| GPTJ 6B      | 128/128                  | 1             | 40633.96        | 34955.29       | 11206.68       | 5966.69        | 6997.91 | 3448.53 |
|              | 2048/128                 | 1             | 2937.91         | 2800.37        | 1354.56        | 682.27         | 747.43  | 352.4   |
|              | 128/2048                 | 1             | 9039.72         | 54939.48       | 3896.8         | 2225.09        | 2041.52 | 896.04  |
|              | 2048/2048                | 1             | 5437.97         | 3663.26        | 1498.04        | 882.61         |         |         |
| LLaMA v2 7B  | 128/128                  | 1             | 18229.3         | 16985.6        | 10725.31       | 5303.5         | 6121.1  | 3139.62 |
|              | 2048/128                 | 1             | 2496.92         | 2355.47        | 1235.4         | 585.6          | 642.24  | 311.82  |
|              | 128/2048                 | 1             | 7612.25         | 6679.36        | 3399.43        | 1903.4         | 1749.4  |         |
|              | 2048/2048                | 1             | 3259.74         | 2805.32        | 1335.51        |                |         |         |
| LLaMA v3 8B  | 128/128                  | 1             | 16708.84        | 16708.53       | 12085.78       | 5853.96        | 8273.8  | 5207.01 |
|              | 2048/128                 | 1             | 2478.94         | 2427.09        | 1604.7         | 737.81         | 1021.64 | 622.15  |
|              | 128/2048                 | 1             | 8367.88         | 8013.55        | 6208.23        | 3385.71        | 4568.17 | 2134.72 |
|              | 2048/2048                | 1             | 3674.33         | 3500.48        | 2776.31        | 1514.04        | 1546.84 | 899.2   |
| Mixtral 8x7B | 128/128                  | 2             | 16959.49        | 16051.88       | 12376.52       | 5120.41        |         |         |
|              |                          | 4             |                 |                |                |                |         | 5271.48 |
|              | 2048/128                 | 2             | 2423.99         | 2276.6         | 1717.37        | 636.5          |         |         |
|              |                          | 4             |                 |                |                |                |         | 654.36  |
|              | 128/2048                 | 2             | 12944.52        | 11997.24       | 7864.88        | 3946.92        |         |         |
|              |                          | 4             |                 |                |                |                |         | 4650.16 |
|              | 2048/2048                | 2             | 6208.97         | 5498.33        | 3722.56        | 1834.36        |         |         |
|              |                          | 4             |                 |                |                |                |         | 2262.57 |
| LLaMA v2 70B | 128/128                  | 1             | 4055.97         | 2134.52        |                |                |         |         |
|              |                          | 2             | 6299.21         | 6035.36        |                | 963.14         | 980.31  |         |
|              |                          | 4             | 8758.45         | 8148.67        | 5454.76        | 2394.12        | 1450.61 | 838.03  |
|              |                          | 8             | 10261.44        | 9385.26        | 7491.94        | 3683.42        | 1387.91 | 1204.32 |
|              | 2048/128                 | 1             | 493.87          | 222.16         |                |                |         |         |
|              |                          | 2             | 784.47          | 757.55         |                | 114.9          | 111.24  |         |
|              |                          | 4             | 1164.15         | 1083.25        | 695.33         | 292.77         | 171.68  | 102.49  |
|              |                          | 8             | 1441.26         | 1346.9         | 1016.58        | 456.46         | 163.76  | 145.41  |
|              | 128/2048                 | 1             | 3199.9          | 635.32         |                |                |         |         |
|              |                          | 2             | 6747            | 4710.45        |                |                |         |         |
|              |                          | 4             | 10960.72        | 8485.56        | 3686.63        | 2047.67        | 1368.09 |         |
|              |                          | 8             | 17250.73        | 12333.24       | 7927.16        | 4166.36        | 1667.57 | 1186.38 |
|              | 2048/2048                | 1             | 1734.58         |                |                |                |         |         |
|              |                          | 2             | 3455.34         | 2267.45        |                |                |         |         |
|              |                          | 4             | 6141.39         | 4019.31        | 1814.78        | 1046           |         |         |
|              |                          | 8             | 9271.77         | 7061.32        | 3658.42        | 2210.84        | 771.23  | 614.74  |
| LLaMA v3 70B | 128/128                  | 1             | 3988.96         |                |                |                |         |         |
|              |                          | 2             | 6155.26         | 5835.57        |                |                |         |         |
|              |                          | 4             | 8454.74         | 7945.64        | 5210.19        | 2405.44        | 1280.9  |         |
|              |                          | 8             | 9893.18         | 9308.51        | 7126.51        | 3621.25        | 1367.56 | 1164.88 |
|              | 2048/128                 | 1             | 491.79          |                |                |                |         |         |
|              |                          | 2             | 783.26          | 751.14         |                |                |         |         |
|              |                          | 4             | 1154.66         | 1074.31        | 691.99         | 295.87         | 171.16  |         |
|              |                          | 8             | 1434.86         | 1337.36        | 1010.5         | 455.18         | 165.06  | 143.92  |
|              | 128/2048                 | 1             | 3015.16         |                |                |                |         |         |
|              |                          | 2             | 6758.32         | 4130.4         |                |                |         |         |
|              |                          | 4             | 10532.1         | 7730.54        | 3246.34        | 1974.04        | 1232.53 |         |
|              |                          | 8             | 16467.79        | 11680.94       | 7205.34        | 4091.45        | 1514.93 | 1034.07 |
|              | 2048/2048                | 1             | 1654.25         |                |                |                |         |         |
|              |                          | 2             | 3271.6          | 1976.76        |                |                |         |         |
|              |                          | 4             | 6113.93         | 3685.74        | 1612.11        | 992.74         |         |         |
|              |                          | 8             | 8986.3          | 6443.85        | 3523.17        | 2118.89        | 691.62  |         |
| Falcon 180B  | 128/128                  | 4             | 3810.55         | 3698.71        |                |                |         |         |
|              |                          | 8             | 5946.89         | 5608.59        | 3954.58        | 1754.14        | 1243.33 |         |
|              | 2048/128                 | 4             | 525.6           | 510.85         |                |                |         |         |
|              |                          | 8             | 848.4           | 813.95         | 535.41         | 221.39         | 145.35  |         |
|              | 128/2048                 | 4             | 2883.67         | 2495.62        |                |                |         |         |
|              |                          | 8             | 5388.34         | 4796.47        | 3051.89        | 1684.6         | 1359.42 |         |
|              | 2048/2048                | 4             | 1376.61         | 952.25         |                |                |         |         |
|              |                          | 8             | 2495.66         | 2421.77        | 896.28         |                | 609.65  |         |

*TP stands for Tensor Parallelism*

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

The following tables are references for commands that are used as part of the benchmarking process.

### Commands

| Stage | Description | Command |
| :- | - | - |
| [Build](#engine-building) | Build a TensorRT-LLM engine | `trtllm-build --model_config $model_cfg --strongly_typed --output_dir $engine_dir --max_batch_size 2048 --max_input_len 2048 --max_seq_len 6144 --workers $tp_size --max_num_tokens 2048 --use_paged_context_fmha enable --multiple_profiles enable` |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset | `benchmarks/cpp/prepare_dataset.py --output=$dataset_file --tokenizer=$model_name token-norm-dist --num-requests=2000 --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `mpirun -n $tp_size --allow-run-as-root --oversubscribe cpp/build/benchmarks/gptManagerBenchmark --engine_dir $engine_dir --type IFB --dataset $dataset_file --scheduler_policy max_utilization --kv_cache_free_gpu_mem_fraction 0.9 --output_csv $results_csv --request_rate -1.0 --enable_chunked_context --streaming --warm_up 0` |

### Variables

| Name | Description |
| :- | - |
| `$isl` | Benchmark input sequence length. |
|`$osl` | Benchmark output sequence length. |
| `$tp_size` | Number of GPUs to run the benchmark with |
| `$engine_dir` | Location to store built engine file (can be deleted after running benchmarks). |
| `$model_cfg` | Name of the model configuration JSON file to use for building. |
| `$model_name` | HuggingFace model name eg. meta-llama/Llama-2-7b-hf or use the path to a local weights directory |
| `$dataset_file` | Location of the dataset file generated by `prepare_dataset.py` |
| `$results_csv` | Path to store end results to. |


### Engine Building

All benchmarks were run using a single engine with a configuration that is capable of handling the
maximum sequence lengths encountered during benchmarking. For each benchmark, regardless of input/output
sequence length, you can reuse the single engine to run all tests. Each engine will be built with a paged
KV cache and in-flight batching enabled. For more information see the
[documentation about in-flight batching](../overview.md#in-flight-batching-and-paged-attention).

In order to build an engine you will need to run the following command by specifying a configuration file
for the model that you would like to build (see [below](#network-configuration-files)). The general build
command is as follows:

```shell
trtllm-build --model_config $model_cfg --strongly_typed --output_dir $engine_dir --max_batch_size 2048 --max_input_len 2048 --max_seq_len 6144 --workers $tp_size --max_num_tokens 2048 --use_paged_context_fmha enable --multiple_profiles enable
```

Some notes about the command:
- `--workers` affects the number of threads that build the engine file and does not necessarily need to match
the TP size. Make sure to set the tensor parallelism in the `$model_cfg` JSON file. See [below](#network-configuration-files)
- You can run benchmarks for datasets that fit within the bounds of the `max_input_len` and `max_seq_len` parameters.

### Engine Configuration Files

In order to configure the TensorRT-LLM build process for benchmarking, you need to provide
`trtllm-build` a configuration file that specifies the following the network configuration, parallelism
mapping, and quantization options.

Below we document how to benchmark each model on an H100-HBM3-80GB system and reproduce the throughput
numbers we document on our [Performance section](#performance of-tensorrt-llm).

> [!Important]
> In order to change the parallelism for a build, you need to modify the `mapping` dictionary in your configuration file. The settings
must conform to the following condition: `world_size == tp_size * pp_size`.

> [!Note]
> All configurations below are set to run utilizing FP8 by default. If you would like to run on an A100 system, see our notes about [disabling FP8 quantization](#running-on-a100).


### Network Configuration Files

Each network has its own configuration file. All networks are configured to run using FP8 quantization
by default.

<table>
<tr>
<td> Model </td> <td> Configuration File (FP8) </td>
</tr>
<tr>
<td> EleutherAI/gpt-j-6b </td>
<td>

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
    "hidden_act": "gelu_new",
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    },
    "rotary_dim": 64,
    "kv_dtype": "float16"
}
```

</td>
</tr>
<tr>
<td> tiiuae/falcon-180B </td>
<td>

```json
{
    "architecture": "FalconForCausalLM",
    "dtype": "bfloat16",
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
    "new_decoder_architecture": true,
    "kv_dtype": "float16"
}
```

</td>
</tr>
<tr>
<td> meta-llama/Llama-2-7b-hf </td>
<td>

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
    },
    "kv_dtype": "float16"
}
```

</td>
</tr>
</tr>
<tr>
<td> meta-llama/Llama-2-70b-hf </td>
<td>

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
    },
    "kv_dtype": "float16"
}
```

</td>
</tr>
<tr>
<td> meta-llama/Meta-Llama-3-8B </td>
<td>

```json
{
    "architecture": "LlamaForCausalLM",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "hidden_act": "silu",
    "norm_epsilon": 1e-05,
    "dtype": "float16",
    "position_embedding_type": "rope_gpt_neox",
    "intermediate_size": 28672,
    "rotary_base": 500000.0,
    "rope_theta": 500000.0,
    "rotary_scaling": null,
    "mapping": {
        "world_size": 1,
        "tp_size": 1,
        "pp_size": 1
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    },
    "kv_dtype": "float16"
}
```

</td>
</tr>
<tr>
<td> meta-llama/Meta-Llama-3-70B </td>
<td>

```json
{
    "architecture": "LlamaForCausalLM",
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
    "hidden_act": "silu",
    "dtype": "float16",
    "norm_epsilon": 1e-05,
    "position_embedding_type": "rope_gpt_neox",
    "intermediate_size": 14336,
    "rotary_base": 500000.0,
    "rope_theta": 500000.0,
    "rotary_scaling": null,
    "mapping": {
        "world_size": 4,
        "tp_size": 4,
        "pp_size": 1
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    },
    "kv_dtype": "float16"
}
```

</td>
</tr>
<tr>
<td> mistralai/Mixtral-8x7B-v0.1 </td>
<td>

```json
{
    "architecture": "MixtralForCausalLM",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "norm_epsilon": 1e-05,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "head_size": 128,
    "hidden_act": "swiglu",
    "dtype": "float16",
    "position_embedding_type": "rope_gpt_neox",
    "intermediate_size": 14336,
    "moe_num_experts": 8,
    "moe_top_k": 2,
    "rotary_base": 1000000.0,
    "rope_theta": 1000000.0,
    "mapping": {
        "world_size": 1,
        "tp_size": 1,
        "pp_size": 1
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    },
    "kv_dtype": "float16"
}
```

</td>
</tr>
</table>



### Running on A100

To run the benchmarks on A100, you will need to undefine or remove the following
quantization fields from each config json file, because FP8 computation is a feature in H100 and newer GPUs.
```json
"quantization": {
	"quant_algo": null,
	"kv_cache_quant_algo": null,
}
```

## Preparing a Dataset

In order to prepare a dataset, you can use the provided [script](../../../benchmarks/cpp/prepare_dataset.py).
To generate a synthetic dataset, run the following command:

```shell
benchmarks/cpp/prepare_dataset.py --output=$dataset_file --tokenizer=$model_name token-norm-dist --num-requests=2000 --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0
```

The command will generate a JSON file located at the path specified `$dataset_file` where all requests are of the same
input/output sequence length combinations. The script works by using the tokenizer to retrieve the vocabulary size and
randomly sample token IDs from it to create entirely random sequences. In the command above, all requests will be uniform
because the standard deviations for both input and output sequences are set to 0.

## Running the Benchmark

To run the benchmark with the generated data set, simply run the following command from the root of the
TensorRT-LLM repository. See the [variables](#variables) section for reference on variable values.

```shell
mpirun -n $tp_size --allow-run-as-root --oversubscribe cpp/build/benchmarks/gptManagerBenchmark --engine_dir $engine_dir --type IFB --dataset $dataset_file --scheduler_policy max_utilization --kv_cache_free_gpu_mem_fraction 0.9 --output_csv $results_csv --request_rate -1.0 --enable_chunked_context --streaming --warm_up 0
```

The command will run the `gptManagerBenchmark` binary that will report the throughput and other metrics as part of its output
that can be compared with the table in the [Performance section](#peak-throughput) of this README.
