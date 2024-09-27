# TensorRT-LLM Benchmarking

> [!WARNING] Work in Progress
> This benchmarking suite is a current work in progress and is prone to large changes.

TensorRT-LLM provides a packaged benchmarking utility that is accessible via the `trtllm-bench` CLI tool.

#### Supported Networks for Benchmarking

- [`tiiuae/falcon-180B`](https://huggingface.co/tiiuae/falcon-180B)
- [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [`meta-llama/Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [`meta-llama/Meta-Llama-3-70B`](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
- [`EleutherAI/gpt-j-6b`](https://huggingface.co/EleutherAI/gpt-j-6b)
- [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [`mistralai/Mixtral-8x7B-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)


#### Support Quantization Modes

TensorRT-LLM supports a number of quanization modes. For more information about quantization, see the
[documentation](https://nvidia.github.io/TensorRT-LLM/reference/precision.html).

- None (no quantization applied)
- W8A16
- W4A16
- W4A16_AWQ
- W4A8_AWQ
- W4A16_GPTQ
- FP8
- INT8

> [!NOTE] Please see the supported quantization methods for each network [here](https://nvidia.github.io/TensorRT-LLM/reference/precision.html#support-matrix)


## Inflight Benchmarking with a Dataset

This section covers how to benchmark TensorRT-LLM using inflight batching.


### Quickstart

For this quick start guide, we will focus on running a short max throughput benchmark on
`meta-llama/Llama-2-7b-hf` on a syntehtic dataset with a uniform distribution of prompts with ISL:OSL
of 128:128. In order to run the benchmark from start to finish simply run the following commands:

```shell
python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 3000 > /tmp/synthetic_128_128.txt
trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8
trtllm-bench --model meta-llama/Llama-2-7b-hf throughput --dataset /tmp/synthetic_128_128.txt --engine_dir /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
```

And that's it! Once the benchmark completes, a summary will be printed with summary metrics.

```shell
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0
Dtype:                  float16
KV Cache Dtype:         FP8
Quantization:           FP8
Max Input Length:       2048
Max Sequence Length:    4098

===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
Max Runtime Batch Size: 4096
Max Runtime Tokens:     8192
Scheduling Policy:      Guaranteed No Evict
KV Memory Percentage:   99.0%
Issue Rate (req/sec):   3.680275266452667e+18
===========================================================
= STATISTICS
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0
Average Output Length (tokens): 128.0
Token Throughput (tokens/sec):  23405.927228471104
Request Throughput (req/sec):   182.8588064724305
Total Latency (seconds):        16.406100739
===========================================================
```

### Workflow

The workflow for `trtllm-bench` is composed of the following steps:

1. Prepare a dataset to drive the inflight batching benchmark.
2. Build a benchmark engine using `trtllm-bench build` subcommand.
3. Run the max throughput benchmark using the `trtllm-bench throughput` subcommand.

#### Preparing a Dataset

The inflight benchmark utilizes a fixed JSON schema so that it is simple and
straightforward to specify requests. The schema is defined as follows:

| Key | Required | Type | Description |
| :- | :-: | :-: | :- |
| `task_id`| Y | String | Unique identifier for the request. |
| `prompt` | N* | String | Input text for a generation request. |
| `logits` | N* | List[Integer] | List of logits that make up the request prompt. |
| `output_tokens` | Y | Integer | Number of generated tokens for this request. |

> [!NOTE] Prompt and logits are mutually exclusive*
> While having both `prompt` and `logits` is not required, at least one is required.
> If `logits` are specified, the `prompt` entry is ignored for request generation.

Examples of valid entries for the inflight benchmark are:

- Entries with a human-readable prompt and no logits.
```json
{"task_id": 1, "prompt": "Generate an infinite response to the following: This is the song that never ends, it goes on and on my friend.", "output_tokens": 1000}
{"task_id": 2, "prompt": "Generate an infinite response to the following: Na, na, na, na", "output_tokens": 1000}
```

- Entries which contain logits.
```json
{"task_id":0,"logits":[863,22056,25603,11943,8932,13195,3132,25032,21747,22213],"output_tokens":128}
{"task_id":1,"logits":[14480,13598,15585,6591,1252,8259,30990,26778,7063,30065,21764,11023,1418],"output_tokens":128}
```

> [!INFO] A whole entry is on a line!
> To make the passing of data simpler, a complete JSON entry is on each line so that the benchmarker
> can simply read a line and assume a complete entry. When creating a dataset, be sure that a complete
> JSON entry is on every line.

#### Using `prepare_dataset` to Create Synthetic Datasets

In order to prepare a synthetic dataset, you can use the provided script in the `benchmarks/cpp`
directory. For example, to generate a synthetic dataset of 1000 requests with a uniform ISL/OSL of
128/128 for [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b), simply run:

```shell
benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1000 > /tmp/synthetic_128_128.txt
```

You can pipe the above command to a file to reuse the same dataset, or simply pipe its output to the
benchmark script (example below).

### Building a Benchmark Engine

The second thing you'll need once you have a dataset is an engine to benchmark against. In order to
build a pre-configured engine for one of the supported ISL:OSL combinations, you can run the following
using the dataset you generated with `prepare_dataset.py` to build an FP8 quantized engine:

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8
```

or manually set a max sequence length that you plan to run with specifically:

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf build --max_seq_len 256 --quantization FP8
```

> [!NOTE] `trtllm-bench build` reproduces benchmark engines for performance study. These engine
configurations are not guaranteed to be optimal for all cases and should be viewed as reproducers
for the benchmark data we provide on our [Performance Overview](../docs/source/performance/perf-overview.md).

Looking a little closer, the `build` sub-command
will perform a lookup and build an engine using those reference settings. The
look up table directly corresponds to the performance table found in our
[Performance Overview](../docs/source/performance/perf-overview.md#throughput-measurements). The
output of the `build` sub-command looks similar to the snippet below (for `meta-llama/Llama-2-7b-hf`):

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8
[TensorRT-LLM] TensorRT-LLM version: 0.12.0
[08/12/2024-19:13:06] [TRT-LLM] [I] Found dataset.
[08/12/2024-19:13:07] [TRT-LLM] [I]
===========================================================
= DATASET DETAILS
===========================================================
Max Input Sequence Length:      128
Max Output Sequence Length:     128
Max Sequence Length:    256
Number of Sequences:    3000
===========================================================


[08/12/2024-19:13:07] [TRT-LLM] [I] Set multiple_profiles to True.
[08/12/2024-19:13:07] [TRT-LLM] [I] Set use_paged_context_fmha to True.
[08/12/2024-19:13:07] [TRT-LLM] [I] Set use_fp8_context_fmha to True.
[08/12/2024-19:13:07] [TRT-LLM] [I]
===========================================================
= ENGINE BUILD INFO
===========================================================
Model Name:             meta-llama/Llama-2-7b-hf
Workspace Directory:    /tmp
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1

===========================================================
= ENGINE CONFIGURATION DETAILS
===========================================================
Max Sequence Length:            256
Max Batch Size:                 4096
Max Num Tokens:                 8192
Quantization:                   FP8
===========================================================

Loading Model: [1/3]    Downloading HF model
Downloaded model to /data/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
Time: 0.115s
Loading Model: [2/3]    Loading HF model to memory
current rank: 0, tp rank: 0, pp rank: 0
Time: 60.786s
Loading Model: [3/3]    Building TRT-LLM engine
Time: 163.331s
Loading model done.
Total latency: 224.232s
[TensorRT-LLM][INFO] Engine version 0.12.0 found in the config file, assuming engine(s) built by new builder API.

<snip verbose logging>

[08/12/2024-19:17:09] [TRT-LLM] [I]

===========================================================
ENGINE SAVED: /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
===========================================================
```

The engine in this case will be written to `/tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1` (the end of the log).

### Running a Max Throughput Benchmark

The `trtllm-bench` command line tool provides a max throughput benchmark that is accessible via the
`throughput` subcommand. This benchmark tests a TensorRT-LLM engine under maximum load to provide an
upper bound throughput number.

#### How the Benchmarker Works

The benchmarker will read in a data file or standard input (stdin) as a stream where a single line contains
a complete JSON request entry. The process that the benchmarker is as follows:

1. Iterate over all input requests. If `logits` is specified, construct the request using the specified
list of logits. Otherwise, tokenize the `prompt` with as specified by `--model $HF_MODEL_NAME`.
3. Submit the dataset to the TensorRT-LLM `Executor` API at as fast of a rate as possible (offline mode).
4. Wait for all requests to return, compute statistics, then report out results.

To run the benchmarker, run the following with the [engine](#building-a-benchmark-engine) and
[dataset](#preparing-a-dataset) generated above:

```shell
trtllm-bench --model meta-llama/Llama-2-7b-hf throughput --dataset /tmp/synthetic_128_128.txt --engine_dir /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
[TensorRT-LLM] TensorRT-LLM version: 0.12.0
[08/12/2024-19:36:48] [TRT-LLM] [I] Preparing to run throughput benchmark...
[08/12/2024-19:36:49] [TRT-LLM] [I] Setting up benchmarker and infrastructure.
[08/12/2024-19:36:49] [TRT-LLM] [I] Ready to start benchmark.
[08/12/2024-19:36:49] [TRT-LLM] [I] Initializing Executor.
[TensorRT-LLM][INFO] Engine version 0.12.0 found in the config file, assuming engine(s) built by new builder API.

<snip verbose logging>

[TensorRT-LLM][INFO] Executor instance created by worker
[08/12/2024-19:36:58] [TRT-LLM] [I] Starting response daemon...
[08/12/2024-19:36:58] [TRT-LLM] [I] Executor started.
[08/12/2024-19:36:58] [TRT-LLM] [I] Request serving started.
[08/12/2024-19:36:58] [TRT-LLM] [I] Starting statistics collection.
[08/12/2024-19:36:58] [TRT-LLM] [I] Benchmark started.
[08/12/2024-19:36:58] [TRT-LLM] [I] Collecting live stats...
[08/12/2024-19:36:59] [TRT-LLM] [I] Request serving stopped.
[08/12/2024-19:37:19] [TRT-LLM] [I] Collecting last stats...
[08/12/2024-19:37:19] [TRT-LLM] [I] Ending statistics collection.
[08/12/2024-19:37:19] [TRT-LLM] [I] Stop received.
[08/12/2024-19:37:19] [TRT-LLM] [I] Stopping response parsing.
[08/12/2024-19:37:19] [TRT-LLM] [I] Collecting last responses before shutdown.
[08/12/2024-19:37:19] [TRT-LLM] [I] Completed request parsing.
[08/12/2024-19:37:19] [TRT-LLM] [I] Parsing stopped.
[08/12/2024-19:37:19] [TRT-LLM] [I] Request generator successfully joined.
[08/12/2024-19:37:19] [TRT-LLM] [I] Statistics process successfully joined.
[08/12/2024-19:37:19] [TRT-LLM] [I]
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0
Dtype:                  float16
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
KV Memory Percentage:   90.0%
Issue Rate (req/sec):   2.0827970096792666e+19
===========================================================
= STATISTICS
===========================================================
Number of requests:             3000
Average Input Length (tokens):  128.0
Average Output Length (tokens): 128.0
Token Throughput (tokens/sec):  18886.813971319196
Request Throughput (req/sec):   147.55323415093122
Total Latency (seconds):        20.331645167
===========================================================

[TensorRT-LLM][INFO] Orchestrator sendReq thread exiting
[TensorRT-LLM][INFO] Orchestrator recv thread exiting
[TensorRT-LLM][INFO] Leader sendThread exiting
[TensorRT-LLM][INFO] Leader recvReq thread exiting
[TensorRT-LLM][INFO] Refreshed the MPI local session
```

## Summary

In summary, the general process for reproducing a benchmark point is as follows:

- Prepare a dataset: `python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer $HF_MODEL token-norm-dist --input-mean $ISL --output-mean $OSL --input-stdev 0 --output-stdev 0 --num-requests $NUM_REQUESTS > $DATASET_PATH`
- Build engine: `trtllm-bench --model $HF_MODEL build --dataset $DATASET_PATH`
- Benchmark engine: trtllm-bench --model $HF_MODEL throughput --dataset $DATASET_PATH --engine_dir $ENGINE_DIR`

where,
- `$HF_MODEL` is the Huggingface name of a model.
- `$NUM_REQUESTS` is the number of requests to generate.
- `$DATASET_PATH` is the path where the dataset was written when preparing the dataset.
- `$ENGINE_DIR` the engine directory as printed by `trtllm-bench build`.
