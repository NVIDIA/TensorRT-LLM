(perf-benchmarking)=

# TensorRT-LLM Benchmarking

```{important}
This benchmarking suite is a work in progress.
Expect breaking API changes.
```

TensorRT-LLM provides the `trtllm-bench` CLI, a packaged benchmarking utility that aims to make it
easier for users to reproduce our officially published [performance overiew](./perf-overview.md#throughput-measurements). `trtllm-bench` provides the follows:

- A streamlined way to build tuned engines for benchmarking for a variety of models and platforms.
- An entirely Python workflow for benchmarking.
- Ability to benchmark various flows and features within TensorRT-LLM.

`trtllm-bench` executes all benchmarks using [in-flight batching] -- for more information see
the [this section](../advanced/gpt-attention.md#in-flight-batching) that describes the concept
in further detail.

## Throughput Benchmarking

### Limitations and Caveats

#### Validated Networks for Benchmarking

While `trtllm-bench` should be able to run any network that TensorRT-LLM supports, the following are the list
that have been validated extensively and is the same listing as seen on the
[Performance Overview](./perf-overview.md) page.

- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [tiiuae/falcon-180B](https://huggingface.co/tiiuae/falcon-180B)
- [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
- [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
- [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)
- [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- [meta-llama/Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
- [mistralai/Mixtral-8x7B-v0.1-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1-Instruct)

```{tip}
`trtllm-bench` can automatically download the model from Hugging Face Model Hub.
Export your token in the `HF_TOKEN` environment variable.
```

#### Supported Quantization Modes

`trtllm-bench` supports the following quantization modes:

- None (no quantization applied)
- `FP8`
- `NVFP4`

For more information about quantization, refer to [](../reference/precision.md) and
the [support matrix](../reference/precision.md#support-matrix) of the supported quantization methods for each network.

```{tip}
Although TensorRT-LLM supports more quantization modes than listed above, `trtllm-bench` currently only configures for
a smaller subset.
```

### Quickstart

This quick start focuses on running a short max throughput benchmark on
`meta-llama/Llama-3.1-8B` on a synthetic dataset with a uniform distribution of prompts with ISL:OSL
of 128:128.
To run the benchmark from start to finish, run the following commands:

```shell
python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-3.1-8B token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 3000 > /tmp/synthetic_128_128.txt
trtllm-bench --model meta-llama/Llama-3.1-8B build --dataset /tmp/synthetic_128_128.txt --quantization FP8
trtllm-bench --model meta-llama/Llama-3.1-8B throughput --dataset /tmp/synthetic_128_128.txt --engine_dir /tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1
```

After the benchmark completes, `trtllm-bench` prints a summary with summary metrics.

```shell
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-3.1-8B
Engine Directory:       /tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1
TensorRT-LLM Version:   0.17.0
Dtype:                  bfloat16
KV Cache Dtype:         FP8
Quantization:           FP8
Max Input Length:       256
Max Sequence Length:    256

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 4096
Max Runtime Tokens:     8192
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   90.00%
Issue Rate (req/sec):   5.0689E+14

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0000
Average Output Length (tokens): 128.0000
Token Throughput (tokens/sec):  28390.4265
Request Throughput (req/sec):   221.8002
Total Latency (ms):             13525.6862

===========================================================
```

### Workflow

The workflow for `trtllm-bench` is composed of the following steps:

1. Prepare a dataset to drive the inflight batching benchmark.
2. Build a benchmark engine using `trtllm-bench build` subcommand (not required for [PyTorch flow](#running-with-the-pytorch-workflow)).
3. Run the max throughput benchmark using the `trtllm-bench throughput` subcommand or low latency benchmark using the `trtllm-bench latency` subcommand.


#### Preparing a Dataset

The throughput benchmark utilizes a fixed JSON schema to specify requests. The schema is defined as follows:

| Key             | Required |     Type      | Description                                     |
| :-------------- | :------: | :-----------: | :---------------------------------------------- |
| `task_id`       |    Y     |    String     | Unique identifier for the request.              |
| `prompt`        |    N*    |    String     | Input text for a generation request.            |
| `input_ids`     |    Y*    | List[Integer] | List of logits that make up the request prompt. |
| `output_tokens` |    Y     |    Integer    | Number of generated tokens for this request.    |

```{tip}
\* Specifying `prompt` or `input_ids` is required. However, you can not have both prompts and logits (`input_ids`)
defined at the same time. If you specify `input_ids`, the `prompt` entry is ignored for request generation.
```

Refer to the following examples of valid entries for the benchmark:

- Entries with a human-readable prompt and no logits.

  ```json
  {"task_id": 1, "prompt": "Generate an infinite response to the following: This is the song that never ends, it goes on and on my friend.", "output_tokens": 1000}
  {"task_id": 2, "prompt": "Generate an infinite response to the following: Na, na, na, na", "output_tokens": 1000}
  ```

- Entries which contain logits.

  ```json
  {"task_id":0,"input_ids":[863,22056,25603,11943,8932,13195,3132,25032,21747,22213],"output_tokens":128}
  {"task_id":1,"input_ids":[14480,13598,15585,6591,1252,8259,30990,26778,7063,30065,21764,11023,1418],"output_tokens":128}
  ```

```{tip}
Specify each entry on one line.
To simplify passing the data, a complete JSON entry is on each line so that the benchmarker
can simply read a line and assume a complete entry. When creating a dataset, be sure that a complete
JSON entry is on every line.
```

In order to prepare a synthetic dataset, you can use the provided script in the `benchmarks/cpp`
directory. For example, to generate a synthetic dataset of 1000 requests with a uniform ISL/OSL of
128/128 for [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), run:

```shell
benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-3.1-8B token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1000 > /tmp/synthetic_128_128.txt
```

### Building a Benchmark Engine

#### Default Build Behavior
The `trtllm-bench` CLI tool provides the `build` subcommand to build the TRT-LLM engines for max throughput benchmark.
To build an engine for benchmarking, you can specify the dataset generated with `prepare_dataset.py` through `--dataset` option.
By default, `trtllm-bench`'s tuning heuristic uses the high-level statistics of the dataset (average ISL/OSL, max sequence length)
to optimize engine build settings. The following command builds an FP8 quantized engine optimized using the dataset's ISL/OSL.

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B build --quantization FP8 --dataset /tmp/synthetic_128_128.txt
```

#### Other Build Modes

The build subcommand also provides other ways to build the engine where users have larger control over the tuning values.

- Build engine with self-defined tuning values:
You specify the tuning values to build the engine with by setting `--max_batch_size` and `--max_num_tokens` directly.
`max_batch_size` and `max_num_tokens` control the maximum number of requests and tokens that can be scheduled in each iteration.
If no value is specified, the default `max_batch_size` and `max_num_tokens` values of `2048` and `8192` are used.
The following command builds an FP8 quantized engine by specifying the engine tuning values.

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B build --quantization FP8 --max_seq_len 4096 --max_batch_size 1024 --max_num_tokens 2048
```

- [Experimental] Build engine with target ISL/OSL for optimization:
In this experimental mode, you can provide hints to `trtllm-bench`'s tuning heuristic to optimize the engine on specific ISL and OSL targets.
Generally, the target ISL and OSL aligns with the average ISL and OSL of the dataset, but you can experiment with different values to optimize the engine using this mode.
The following command builds an FP8 quantized engine and optimizes for ISL:OSL targets of 128:128.

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B build --quantization FP8 --max_seq_len 4096 --target_isl 128 --target_osl 128
```


#### Parallelism Mapping Support
The `trtllm-bench build` subcommand supports combinations of tensor-parallel (TP) and pipeline-parallel (PP) mappings as long as the world size (`tp_size x pp_size`) `<=` `8`. The parallelism mapping in build subcommad is controlled by `--tp_size` and `--pp_size` options. The following command builds an engine with TP2-PP2 mapping.

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B build --quantization FP8 --dataset /tmp/synthetic_128_128.txt --tp_size 2 --pp_size 2
```


#### Example of Build Subcommand Output:
The output of the `build` subcommand looks similar to the snippet below (for `meta-llama/Llama-3.1-8B`):

```shell
user@387b12598a9e:/scratch/code/trt-llm/tekit_2025$ trtllm-bench --model meta-llama/Llama-3.1-8B build --dataset /tmp/synthetic_128_128.txt --quantization FP8
[TensorRT-LLM] TensorRT-LLM version: 0.17.0
[01/18/2025-00:55:14] [TRT-LLM] [I] Found dataset.
[01/18/2025-00:55:14] [TRT-LLM] [I]
===========================================================
= DATASET DETAILS
===========================================================
Max Input Sequence Length:      128
Max Output Sequence Length:     128
Max Sequence Length:    256
Target (Average) Input Sequence Length: 128
Target (Average) Output Sequence Length:        128
Number of Sequences:    3000
===========================================================


[01/18/2025-00:55:14] [TRT-LLM] [I] Max batch size and max num tokens are not provided, use tuning heuristics or pre-defined setting from trtllm-bench.
[01/18/2025-00:55:14] [TRT-LLM] [I] Estimated total available memory for KV cache: 132.37 GB
[01/18/2025-00:55:14] [TRT-LLM] [I] Estimated total KV cache memory: 125.75 GB
[01/18/2025-00:55:14] [TRT-LLM] [I] Estimated max number of requests in KV cache memory: 8048.16
[01/18/2025-00:55:14] [TRT-LLM] [I] Estimated max batch size (after fine-tune): 4096
[01/18/2025-00:55:14] [TRT-LLM] [I] Estimated max num tokens (after fine-tune): 8192
[01/18/2025-00:55:14] [TRT-LLM] [I] Set dtype to bfloat16.
[01/18/2025-00:55:14] [TRT-LLM] [I] Set multiple_profiles to True.
[01/18/2025-00:55:14] [TRT-LLM] [I] Set use_paged_context_fmha to True.
[01/18/2025-00:55:14] [TRT-LLM] [I] Set use_fp8_context_fmha to True.
[01/18/2025-00:55:14] [TRT-LLM] [I]
===========================================================
= ENGINE BUILD INFO
===========================================================
Model Name:             meta-llama/Llama-3.1-8B
Model Path:             None
Workspace Directory:    /tmp
Engine Directory:       /tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1

===========================================================
= ENGINE CONFIGURATION DETAILS
===========================================================
Max Sequence Length:            256
Max Batch Size:                 4096
Max Num Tokens:                 8192
Quantization:                   FP8
KV Cache Dtype:                 FP8
===========================================================

Loading Model: [1/3]    Downloading HF model
Downloaded model to /data/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
Time: 0.321s
Loading Model: [2/3]    Loading HF model to memory
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:59<00:00, 14.79s/it]
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████| 287113/287113 [00:06<00:00, 41375.57 examples/s]
Generating validation split: 100%|█████████████████████████████████████████████████████████████████████████████████| 13368/13368 [00:00<00:00, 41020.63 examples/s]
Generating test split: 100%|███████████████████████████████████████████████████████████████████████████████████████| 11490/11490 [00:00<00:00, 41607.11 examples/s]
Inserted 675 quantizers
/usr/local/lib/python3.12/dist-packages/modelopt/torch/quantization/model_quant.py:71: DeprecationWarning: forward_loop should take model as argument, but got forward_loop without any arguments. This usage will be deprecated in future versions.
  warnings.warn(
Disable lm_head quantization for TRT-LLM export due to deployment limitations.
current rank: 0, tp rank: 0, pp rank: 0
Time: 122.568s
Loading Model: [3/3]    Building TRT-LLM engine
/usr/local/lib/python3.12/dist-packages/tensorrt/__init__.py:85: DeprecationWarning: Context managers for TensorRT types are deprecated. Memory will be freed automatically when the reference count reaches 0.
  warnings.warn(
Time: 53.820s
Loading model done.
Total latency: 176.709s

<snip verbose logging>

===========================================================
ENGINE SAVED: /tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1
===========================================================
```

The engine in this case will be written to `/tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1` (the end of the log).


### Max Throughput Benchmark

The `trtllm-bench` command line tool provides a max throughput benchmark that is accessible via the
`throughput` subcommand. This benchmark tests a TensorRT-LLM engine or PyTorch backend under maximum load to provide an
upper bound throughput number.

#### How the Benchmarker Works

The benchmarker reads a data file where a single line contains
a complete JSON request entry as specified in [](#preparing-a-dataset).
The process that the benchmarker is as follows:

1. Iterate over all input requests. If `logits` is specified, construct the request using the specified
list of logits. Otherwise, tokenize the `prompt` with as specified by `--model $HF_MODEL_NAME`.
1. Submit the dataset to the TensorRT-LLM `Executor` API as fast as possible (offline mode).
1. Wait for all requests to return, compute statistics, and then report results.

To run the benchmarker, run the following commands with the [engine](#building-a-benchmark-engine) and
[dataset](#preparing-a-dataset) generated from previous steps:

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B throughput --dataset /tmp/synthetic_128_128.txt --engine_dir /tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1
[TensorRT-LLM] TensorRT-LLM version: 0.17.0
[01/18/2025-01:01:13] [TRT-LLM] [I] Preparing to run throughput benchmark...
[01/18/2025-01:01:13] [TRT-LLM] [I] Setting up throughput benchmark.

<snip verbose logging>

[01/18/2025-01:01:26] [TRT-LLM] [I] Setting up for warmup...
[01/18/2025-01:01:26] [TRT-LLM] [I] Running warmup.
[01/18/2025-01:01:26] [TRT-LLM] [I] Starting benchmarking async task.
[01/18/2025-01:01:26] [TRT-LLM] [I] Starting benchmark...
[01/18/2025-01:01:26] [TRT-LLM] [I] Request submission complete. [count=2, time=0.0000s, rate=121847.20 req/s]
[01/18/2025-01:01:28] [TRT-LLM] [I] Benchmark complete.
[01/18/2025-01:01:28] [TRT-LLM] [I] Stopping LLM backend.
[01/18/2025-01:01:28] [TRT-LLM] [I] Cancelling all 0 tasks to complete.
[01/18/2025-01:01:28] [TRT-LLM] [I] All tasks cancelled.
[01/18/2025-01:01:28] [TRT-LLM] [I] LLM Backend stopped.
[01/18/2025-01:01:28] [TRT-LLM] [I] Warmup done.
[01/18/2025-01:01:28] [TRT-LLM] [I] Starting benchmarking async task.
[01/18/2025-01:01:28] [TRT-LLM] [I] Starting benchmark...
[01/18/2025-01:01:28] [TRT-LLM] [I] Request submission complete. [count=3000, time=0.0012s, rate=2590780.97 req/s]
[01/18/2025-01:01:42] [TRT-LLM] [I] Benchmark complete.
[01/18/2025-01:01:42] [TRT-LLM] [I] Stopping LLM backend.
[01/18/2025-01:01:42] [TRT-LLM] [I] Cancelling all 0 tasks to complete.
[01/18/2025-01:01:42] [TRT-LLM] [I] All tasks cancelled.
[01/18/2025-01:01:42] [TRT-LLM] [I] LLM Backend stopped.
[01/18/2025-01:01:42] [TRT-LLM] [I]

===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-3.1-8B
Engine Directory:       /tmp/meta-llama/Llama-3.1-8B/tp_1_pp_1
TensorRT-LLM Version:   0.17.0
Dtype:                  bfloat16
KV Cache Dtype:         FP8
Quantization:           FP8
Max Input Length:       256
Max Sequence Length:    256

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 4096
Max Runtime Tokens:     8192
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   90.00%
Issue Rate (req/sec):   5.0689E+14

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0000
Average Output Length (tokens): 128.0000
Token Throughput (tokens/sec):  28390.4265
Request Throughput (req/sec):   221.8002
Total Latency (ms):             13525.6862

===========================================================

[01/18/2025-01:01:42] [TRT-LLM] [I] Thread proxy_dispatch_result_thread stopped.
[TensorRT-LLM][INFO] Refreshed the MPI local session
```

### Running with the PyTorch Workflow

To benchmark the PyTorch backend (`tensorrt_llm._torch`), use the following command with [dataset](#preparing-a-dataset) generated from previous steps. With the PyTorch flow, you will not need to
run `trtllm-bench build`; the `throughput` benchmark initializes the backend by tuning against the
dataset provided via `--dataset` (or the other build mode settings described [above](#other-build-modes)).
Note that CUDA graph is enabled by default. You can add additional pytorch config with
`--extra_llm_api_options` followed by the path to a YAML file. For more details, please refer to the
help text by running the command with `--help`.

```{tip}
The command below specifies the `--model_path` option. The model path is optional and used only when you want to run a locally
stored checkpoint. When using `--model_path`, the `--model` is still required for reporting reasons and in order to look up parameters
for build heuristics.
```

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B --model_path /Ckpt/Path/To/Llama-3.1-8B throughput --dataset /tmp/synthetic_128_128.txt --backend pytorch

# Example output
<snip verbose logging>
===========================================================
= PyTorch backend
===========================================================
Model:                  meta-llama/Llama-3.1-8B
Model Path:             /Ckpt/Path/To/Llama-3.1-8B
TensorRT-LLM Version:   0.17.0
Dtype:                  bfloat16
KV Cache Dtype:         None
Quantization:           FP8

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 2048
Max Runtime Tokens:     4096
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   90.00%
Issue Rate (req/sec):   7.6753E+14

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0000
Average Output Length (tokens): 128.0000
Token Throughput (tokens/sec):  20685.5510
Request Throughput (req/sec):   161.6059
Total Latency (ms):             18563.6825

```

#### Quantization in the PyTorch Flow

In order to run a quantized run with `trtllm-bench` utilizing the PyTorch flow, you will need to use a pre-quantized
To run a quantized benchmark with `trtllm-bench` utilizing the PyTorch flow, you will need to use a pre-quantized
checkpoint. For the Llama-3.1 models, TensorRT-LLM provides the following checkpoints via HuggingFace:

- [`nvidia/Llama-3.1-8B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)
- [`nvidia/Llama-3.1-70B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8)
- [`nvidia/Llama-3.1-405B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8)

`trtllm-bench` utilizes the `hf_quant_config.json` file present in the pre-quantized checkpoints above. The configuration
file is present in checkpoints quantized with [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
and describes the compute and KV cache quantization that checkpoint was compiled with. For example, from the checkpoints
above:

```json
{
    "producer": {
        "name": "modelopt",
        "version": "0.23.0rc1"
    },
    "quantization": {
        "quant_algo": "FP8",
        "kv_cache_quant_algo": null
    }
```

The checkpoints above are quantized to run with a compute precision of `FP8` and default to no KV cache quantization (full
`FP16` cache). When running `trtllm-bench throughput`. The benchmark will select a KV cache quantization that is best suited
for the compute precision in the checkpoint automatically if `kv_cache_quant_algo` is specified as `null`, otherwise it will
be forced to match the specified non-null KV cache quantization. The following are the mappings that `trtllm-bench` will
follow when a checkpoint does not specify a KV cache quantization algorithm:

| Checkpoint Compute Quant | Checkpoint KV Cache Quant | `trtllm-bench` | Note |
| - | - | - | - |
| `null` | `null` | `null` | In this case, a quantization config doesn't exist. |
| `FP8` | `FP8` | `FP8` | Matches the checkpoint |
| `FP8` | `null` | `FP8` | Set to `FP8` via benchmark |
| `NVFP4` | `null` | `FP8` | Set to `FP8` via benchmark |

If you would like to force the KV cache quantizaton, you can specify the following in the YAML file to force the precision
when the checkpoint precision is `null`:

```yaml
pytorch_backend_config:
  kv_cache_dtype: "fp8"
```

```{tip}
The two valid values for `kv_cache_dtype` are `auto` and `fp8`.
```

## Low Latency Benchmark

The low latency benchmark follows a similar workflow to the [throughput benchmark](#max-throughput-benchmark)
but requires building the engine separately from `trtllm-bench`. Low latency benchmarks has the following modes:

- A single-request low-latency engine
- A Medusa-enabled speculative-decoding engine

### Low Latency TensorRT-LLM Engine for Llama-3 70B

To build a low-latency engine for the latency benchmark, run the following quantize and build commands.
The `$checkpoint_dir` is the path to the [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) Hugging Face checkpoint in your cache or downloaded to a specific location with the [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
To prepare a dataset, follow the same process as specified in [](#preparing-a-dataset).

#### Benchmarking a non-Medusa Low Latency Engine

To quantize the checkpoint:

```shell
cd tensorrt_llm/examples/llama
python ../quantization/quantize.py \
    --model_dir $checkpoint_dir \
    --dtype bfloat16 \
    --qformat fp8 \
    --kv_cache_dtype fp8 \
    --output_dir /tmp/meta-llama/Meta-Llama-3-70B/checkpoint \
    --calib_size 512 \
    --tp_size $tp_size
```

then build,

```shell
trtllm-build \
    --checkpoint_dir /tmp/meta-llama/Meta-Llama-3-70B/checkpoint \
    --use_fused_mlp enable \
    --gpt_attention_plugin bfloat16 \
    --output_dir /tmp/meta-llama/Meta-Llama-3-70B/engine \
    --max_batch_size 1 \
    --max_seq_len $(($isl+$osl)) \
    --reduce_fusion enable \
    --gemm_plugin fp8 \
    --workers $tp_size \
    --use_fp8_context_fmha enable \
    --max_num_tokens $isl \
    --use_paged_context_fmha disable \
    --multiple_profiles enable
```

After the engine is built, run the low-latency benchmark:

```shell
env TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG=1 \
  TRTLLM_MMHA_KERNEL_BLOCK_SIZE=256 \
  TRTLLM_MMHA_BLOCKS_PER_SEQUENCE=32 \
  FORCE_MULTI_BLOCK_MODE=ON \
  TRTLLM_ENABLE_PDL=1 \
  trtllm-bench --model meta-llama/Meta-Llama-3-70B \
  latency \
  --dataset $DATASET_PATH \
  --engine_dir /tmp/meta-llama/Meta-Llama-3-70B/engine
```

### Building a Medusa Low-Latency Engine

To build a Medusa-enabled engine requires checkpoints that contain Medusa heads.
NVIDIA provides TensorRT-LLM checkpoints on the [NVIDIA](https://huggingface.co/nvidia) page on Hugging Face.
The checkpoints are pre-quantized and can be directly built after downloading them with the
[huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
After you download the checkpoints, run the following command. Make sure to
specify the `$tp_size` supported by your Medusa checkpoint and the path to its stored location `$checkpoint_dir`.
Additionally, `$max_seq_len` should be set to the model's maximum position embedding.

Using Llama-3.1 70B as an example, for a tensor parallel 8 and bfloat16 dtype:

```shell
tp_size=8
max_seq_len=131072
trtllm-build --checkpoint_dir $checkpoint_dir \
    --speculative_decoding_mode medusa \
    --max_batch_size 1 \
    --gpt_attention_plugin bfloat16 \
    --max_seq_len $max_seq_len \
    --output_dir /tmp/meta-llama/Meta-Llama-3.1-70B/medusa/engine \
    --use_fused_mlp enable \
    --paged_kv_cache enable \
    --use_paged_context_fmha disable \
    --multiple_profiles enable \
    --reduce_fusion enable \
    --use_fp8_context_fmha enable \
    --workers $tp_size \
    --low_latency_gemm_plugin fp8
```

After the engine is built, you need to define the Medusa choices.
The choices are specified with a YAML file like the following example (`medusa.yaml`):

```yaml
- [0]
- [0, 0]
- [1]
- [0, 1]
- [2]
- [0, 0, 0]
- [1, 0]
- [0, 2]
- [3]
- [0, 3]
- [4]
- [0, 4]
- [2, 0]
- [0, 5]
- [0, 0, 1]
```

To run the Medusa-enabled engine, run the following command:

```shell
env TRTLLM_ENABLE_PDL=1 \
  UB_ONESHOT=1 \
  UB_TP_SIZE=$tp_size \
  TRTLLM_ENABLE_PDL=1 \
  TRTLLM_PDL_OVERLAP_RATIO=0.15 \
  TRTLLM_PREFETCH_RATIO=-1 \
  trtllm-bench --model meta-llama/Meta-Llama-3-70B \
  latency \
  --dataset $DATASET_PATH \
  --engine_dir /tmp/meta-llama/Meta-Llama-3-70B/medusa/engine \
  --medusa_choices medusa.yml
```

## Summary

The following table summarizes the commands needed for running benchmarks:

| Scenario | Phase | Command |
| - | - | - |
| Dataset | Preparation | `python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer $HF_MODEL token-norm-dist --input-mean $ISL --output-mean $OSL --input-stdev 0 --output-stdev 0 --num-requests $NUM_REQUESTS > $DATASET_PATH` |
| Throughput | Build | `trtllm-bench --model $HF_MODEL build --dataset $DATASET_PATH` |
| Throughput | Benchmark | `trtllm-bench --model $HF_MODEL throughput --dataset $DATASET_PATH --engine_dir $ENGINE_DIR` |
| Latency | Build | See [section about building low latency engines](#low-latency-tensorrt-llm-engine-for-llama-3-70b) |
| Non-Medusa Latency | Benchmark | `trtllm-bench --model $HF_MODEL latency --dataset $DATASET_PATH --engine_dir $ENGINE_DIR` |
| Medusa Latency | Benchmark | `trtllm-bench --model $HF_MODEL latency --dataset $DATASET_PATH --engine_dir $ENGINE_DIR --medusa_choices $MEDUSA_CHOICES` |

where,

`$HF_MODEL`
: The Hugging Face name of a model.

`$NUM_REQUESTS`
: The number of requests to generate.

`$DATASET_PATH`
: The path where the dataset was written when preparing the dataset.

`$ENGINE_DIR`
: The engine directory as printed by `trtllm-bench build`.

`$MEDUSA_CHOICES`
: A YAML config representing the Medusa tree for the benchmark.
