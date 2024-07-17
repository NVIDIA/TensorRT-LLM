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

**All data in the table below was generated using version 0.11.0 and presents token throughput in tokens/second.**

|              |                          |               |                 |             |                |                |                |          |
| ------------ | ------------------------ | ------------- | --------------- | ----------- | -------------- | -------------- | -------------- | -------- |
|              |                          | **GPU**       | H200 141GB HBM3 | GH200 120GB | H100 80GB HBM3 | H100 80GB HBM3 | A100-SXM4-80GB | L40S     |
|              |                          | **Precision** | FP8             | FP8         | FP8            | FP16           | FP16           | FP8      |
| **Model**    | **Input/Output Lengths** | **TP**        |                 |             |                |                |                |          |
| GPTJ 6B      | 128/128                  | 1             | 25116.54        | 24998.09    | 24456.84       | 13328.96       | 6168.8         | 7737.44  |
|              | 2048/128                 | 1             | 2845.4          | 2840.46     | 2781.11        | 1410.81        | 662            | 83.46    |
|              | 128/2048                 | 1             | 8165.91         | 7936.16     | 7643.02        | 3503.41        | 2213.44        | 1927.91  |
|              | 2048/2048                | 1             | 3560.37         | 3197.21     | 3081.26        | 1326.79        | 893.43         |          |
| LLaMA v2 7B  | 128/128                  | 1             | 19695.41        | 19509.49    | 17684.88       | 11605.69       | 5286.1         | 6655.52  |
|              | 2048/128                 | 1             | 2471.89         | 2401.29     | 2342.71        | 1173.81        | 558.56         | 644.72   |
|              | 128/2048                 | 1             | 7867.28         | 6689.51     | 6814.72        | 3074.4         | 1813.79        | 1591.51  |
|              | 2048/2048                | 1             | 3215.63         | 3015.84     | 2820.31        | 1289.87        | 716.55         | 653.19   |
| LLaMA v3 8B  | 128/128                  | 1             | 29084.05        | 29197.48    | 27781.28       | 15225.75       | 6450.88        | 8929.6   |
|              | 2048/128                 | 1             | 3699.64         | 3780.47     | 3555.57        | 1844.38        | 775.18         | 1052.3   |
|              | 128/2048                 | 1             | 23723.81        | 22055.94    | 17894.85       | 8415.67        | 4837.47        | 4497.21  |
|              | 2048/2048                | 1             | 11193.29        | 8877.13     | 8398.71        | 3996.93        | 2271.65        | 1911.63  |
| Mistral 7B   | 128/128                  | 1             | 31618.59        | 31868.45    | 30400.21       | 16108.11       | 6749.91        | 10237.23 |
|              | 2048/128                 | 1             | 3791.1          | 3795.27     | 3618.11        | 1896.76        | 783.94         | 1126.08  |
|              | 128/2048                 | 1             | 25646.02        | 20491.88    | 20518.75       | 10018.54       | 5358.28        | 5441.98  |
|              | 2048/2048                | 1             | 12068.11        | 9462.96     | 9504.59        | 4383.42        | 2465.77        | 2213.69  |
| LLaMA v2 70B | 128/128                  | 2             | 6652.29         | 5619.41     | 6502.44        |                |                |          |
|              |                          | 4             | 10921.65        | 11043       | 10448.46       | 6219.11        | 2487.78        | 1549.09  |
|              |                          | 8             | 15878.34        |             | 14781.66       | 10093.27       | 4233.24        | 1497.68  |
|              | 2048/128                 | 2             | 766.38          | 647.73      | 747.14         |                |                |          |
|              |                          | 4             | 1296.75         | 1298.94     | 1231.26        | 714.07         | 285.9          | 179.19   |
|              |                          | 8             | 1930.16         |             | 1808.02        | 1230.66        | 494.29         | 176.24   |
|              | 128/2048                 | 2             | 7014.86         | 4844.17     | 5267.56        |                |                |          |
|              |                          | 4             | 13365.86        | 11596.55    | 9202.42        | 3787.24        | 2267.02        | 1772.45  |
|              |                          | 8             | 18861.53        |             | 17085.82       | 7846.64        | 5096.52        | 2290.99  |
|              | 2048/2048                | 2             | 3554.71         | 2843.31     | 2457.73        |                |                |          |
|              |                          | 4             | 6604.37         | 5969.11     | 4586.99        | 1994.1         | 1137.22        | 890.83   |
|              |                          | 8             | 10034.12        |             | 7647.54        | 4347.09        | 2152.35        | 1130.36  |
| LLaMA v3 70B | 128/128                  | 4             |                 | 9872.81     |                |                |                |          |
|              |                          | 8             | 15255           |             | 13853.05       |                | 4033.42        |          |
|              | 2048/128                 | 4             |                 | 1284.88     |                |                |                |          |
|              |                          | 8             | 1918.47         |             | 1738.94        |                | 476.42         |          |
|              | 128/2048                 | 4             |                 | 9996.88     |                |                |                |          |
|              |                          | 8             | 19071.39        |             | 10887.34       |                | 3373.71        |          |
|              | 2048/2048                | 4             |                 | 4985.31     |                |                |                |          |
|              |                          | 8             | 9387.81         |             | 6029.39        |                | 1824.06        |          |
| Mixtral 8x7B | 128/128                  | 2             | 26317.73        | 21768.19    | 24770.44       | 11821.14       | 5522.43        |          |
|              | 2048/128                 | 2             | 3181.76         | 2545.52     | 2973.11        | 1391.28        | 636.77         |          |
|              | 128/2048                 | 2             | 30105.61        | 23643.33    | 22120.85       | 6337.02        | 3698.23        |          |
|              | 2048/2048                | 2             | 15002.42        | 11683.11    | 11486.66       | 3024.95        | 1710.53        |          |
| Falcon 180B  | 128/128                  | 4             | 5647.01         |             | 5568.91        |                |                |          |
|              |                          | 8             | 9304.06         |             | 8885.39        |                | 2171.78        |          |
|              | 2048/128                 | 4             | 670.99          | 693.82      | 667.8          |                |                |          |
|              |                          | 8             | 1103.18         |             | 1065.16        |                | 238.61         |          |
|              | 128/2048                 | 4             | 8358.01         | 6655.38     | 6376.89        |                |                |          |
|              |                          | 8             | 14514.24        |             | 12447.25       |                | 2657.9         |          |
|              | 2048/2048                | 4             | 4169.39         | 3415.05     | 3412.09        |                |                |          |
|              |                          | 8             | 7524.11         |             | 6326.46        |                | 1392.31        |          |

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
| [Build](#engine-building) | Build a TensorRT-LLM engine | `trtllm-build --model_config $model_cfg --use_fused_mlp --gpt_attention_plugin float16 --output_dir $engine_dir --max_batch_size $max_batch_size --max_input_len 2048 --max_output_len 2048 --reduce_fusion disable --workers $tp_size --max_num_tokens $max_num_tokens --use_paged_context_fmha enable --multiple_profiles enable` |
| [Dataset](#preparing-a-dataset) | Create a synthetic dataset | `benchmarks/cpp/prepare_dataset.py --output=$dataset_file --tokenizer=$model_name token-norm-dist --num-requests=2000 --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0` |
| [Run](#running-the-benchmark) | Run a benchmark with a dataset | `mpirun -n $tp_size --allow-run-as-root --oversubscribe cpp/build/benchmarks/gptManagerBenchmark --engine_dir $engine_dir --type IFB --dataset $dataset_file --eos_id -1 --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.99 --output_csv result.csv --request_rate -1.0 --enable_chunked_context --warm_up 0` |

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
| `$max_batch_size` | Absolute maximum number of concurrent requests an engine can handle during one iteration. |
| `$max_num_tokens` | Maximum number of total tokens an engine can handle during one iteration. |


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
trtllm-build --model_config $model_cfg --use_fused_mlp --gpt_attention_plugin float16 --output_dir $engine_dir --max_batch_size $max_batch_size --max_input_len 2048 --max_output_len 2048 --reduce_fusion disable --workers $tp_size --max_num_tokens $max_num_tokens --use_paged_context_fmha enable --multiple_profiles enable
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


### Network Configuration Files and Settings

Each network has its own configuration file. All networks are configured to run using FP8 quantization by default. Additionally, each network has a specific tuning for the
`$max_batch_size` and `$max_num_tokens` parameters -- at times varying for some
input and output sequence legnths within the same model.

> ![Note]
> General settings are specified by "General" in the "ISL/OSL" column. For special
> cases, specific input and output sequence lengths will be specified.

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

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 1          | 2048              | 128               | General  |
| 1          | 2048              | 2048              | 128, 128 |


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

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 4          | 8192              | 4096              | General  |
| 8          | 8192              | 2048              | General  |

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

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 1          | 8192              | 4096              | General  |

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

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 2          | 2048              | 2048              | General  |
| 4          | 8192              | 4096              | General  |
| 4          | 8192              | 256               | 128, 4096|
| 8          | 16384             | 8192              | General  |
| 8          | 16384             | 1024              | 128, 2048|

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
    "intermediate_size": 14336,
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

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 1          | 8192              | 2048              | General  |

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
    "intermediate_size": 28672,
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

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 4          | 1024              | 2048              | General  |
| 8          | 16384             | 8192              | General  |

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

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 2          | 3072              | 2048              | General  |
| 4          | 8192              | 8192              | General  |

</td>
</tr>
<tr>
<td> mistralai/Mistral-7B-v0.1 </td>
<td>

```json
{
    "architecture": "MistralForCausalLM",
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "norm_epsilon": 1e-05,
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "hidden_act": "silu",
    "dtype": "float16",
    "logits_dtype": "float32",
    "position_embedding_type": "rope_gpt_neox",
    "use_parallel_embedding": false,
    "embedding_sharding_dim": 0,
    "share_embedding_table": false,
    "intermediate_size": 14336,
    "use_prompt_tuning": false,
    "mapping": {
        "world_size": 1,
        "tp_size": 1,
        "pp_size": 1
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8"
    }
}
```

| `$tp_size` | `$max_num_tokens` | `$max_batch_size` |  ISL/OSL |
| ---------- | ----------------- | ----------------- | -------- |
| 1          | 8192              | 4098              | General  |

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
benchmarks/cpp/prepare_dataset.py --output=$dataset_file --tokenizer=$model_name token-norm-dist --num-requests=$num_requests --input-mean=$isl --output-mean=$osl --input-stdev=0 --output-stdev=0
```

The command will generate a JSON file located at the path specified `$dataset_file` where all requests are of the same
input/output sequence length combinations. The script works by using the tokenizer to retrieve the vocabulary size and
randomly sample token IDs from it to create entirely random sequences. In the command above, all requests will be uniform
because the standard deviations for both input and output sequences are set to 0.


For each input and output sequence length combination, the table below details the `$num_requests` that were used. For
shorter input and output lengths, a larger number of messages were used to guarantee that the system hit a steady state
because requests enter and exit the system at a much faster rate. For longer input/output sequence lengths, requests
remain in the system longer and therefore require less requests to achieve steady state.


| Input Length | Output Length | $num_requests      |
| ------------ | ------------- | ------------------ |
| 128          | 128           | 30000              |
| 128          | 2048          | 3000               |
| 128          | 4096          | 1500               |
| 2048         | 128           | 3000               |
| 2048         | 2048          | 1500               |


## Running the Benchmark

To run the benchmark with the generated data set, simply run the following command from the root of the
TensorRT-LLM repository. See the [variables](#variables) section for reference on variable values.

```shell
mpirun -n $tp_size --allow-run-as-root --oversubscribe cpp/build/benchmarks/gptManagerBenchmark --engine_dir $engine_dir --type IFB --dataset $dataset_file --eos_id -1 --scheduler_policy guaranteed_no_evict --kv_cache_free_gpu_mem_fraction 0.99 --output_csv result.csv --request_rate -1.0 --enable_chunked_context --warm_up 0
```

> [!Warning] GH200 benchmarks
> For GH200 benchmarks, the command above must be modified to use `--kv_cache_free_gpu_mem_fraction 0.95` to avoid an out of memory scenario.

The command will run the `gptManagerBenchmark` binary that will report the throughput and other metrics as part of its output
that can be compared with the table in the [Performance section](#peak-throughput) of this README.
