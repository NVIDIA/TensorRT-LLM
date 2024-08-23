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

#### Support Quantization Modes

TensorRT-LLM supports a number of quanization modes. For more information about quantization, see the
[documentation](https://nvidia.github.io/TensorRT-LLM/precision.html).

- None (no quantization applied)
- W8A16
- W4A16
- W4A16_AWQ
- W4A8_AWQ
- W4A16_GPTQ
- FP8
- INT8

> [!NOTE] Please see the supported quantization methods for each network [here](https://nvidia.github.io/TensorRT-LLM/precision.html#support-matrix)


## Inflight Benchmarking with a Dataset

This section covers how to benchmark TensorRT-LLM using inflight batching.


### Quickstart

For this quick start guide, we will focus on running a short max throughput benchmark on
`meta-llama/Llama-2-7b-hf` on a syntehtic dataset with a uniform distribution of prompts with ISL:OSL
of 128:128. In order to run the benchmark from start to finish simply run the following commands:

```shell
python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1400 > /tmp/synthetic_128_128.txt
trtllm-bench --model meta-llama/Llama-2-7b-hf build --dataset /tmp/synthetic_128_128.txt --quantization FP8
trtllm-bench --model meta-llama/Llama-2-7b-hf throughput --dataset /tmp/synthetic_128_128.txt --engine-path /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
```

And that's it! Once the benchmark completes, a summary will be printed with summary metrics.

```
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0.dev2024073000
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
benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1000 > $PATH_TO_DATASET
```

You can pipe the above command to a file to reuse the same dataset, or simply pipe its output to the
benchmark script (example below).

### Building a Benchmark Engine

The second thing you'll need once you have a dataset is an engine to benchmark against. In order to
build a pre-configured engine for one of the supported ISL:OSL combinations, you can run the following
using the dataset you generated with `prepare_dataset.py` to build an FP8 quantized engine:

```shell
trtllm-bench --model $HF_MODEL_NAME build --dataset $PATH_TO_DATASET --quantization FP8
```

or manually set a max sequence length thatL you plan to run with specifically:

```shell
trtllm-bench --model $HF_MODEL_NAME build --max_seq_len $MAX_SEQ_LEN --quantization FP8
```

The engine in this case will be written to the `/tmp/$HF_MODEL_NAME/tp_1_pp_1/` directory.

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

To run the benchmarker, run the following with the engine and dataset generated above:

```
trtllm-bench --model $HF_MODEL_NAME throughput --dataset $PATH_TO_DATASET --engine_dir /tmp/$HF_MODEL_NAME/tp_1_pp_1/
```

When the benchmark runs, you will see output similar to the following:

```
Preparing to run throughput benchmark...
Setting up benchmarker and infrastructure.
Initializing Throughput Benchmark. [rate=%d req/s]
Ready to start benchmark.
Initializing Executor.
[TensorRT-LLM][INFO] Engine version 0.12.0.dev2024073000 found in the config file, assuming engine(s) built by new builder API.
[TensorRT-LLM][INFO] Initializing MPI with thread mode 3
[TensorRT-LLM][INFO] Initialized MPI
[TensorRT-LLM][INFO] Engine version 0.12.0.dev2024073000 found in the config file, assuming engine(s) built by new builder API.
[TensorRT-LLM][INFO] MPI size: 1, MPI local size: 1, rank: 0
[TensorRT-LLM][INFO] Rank 0 is using GPU 0
[TensorRT-LLM][INFO] TRTGptModel maxNumSequences: 4096
[TensorRT-LLM][INFO] TRTGptModel maxBatchSize: 4096
[TensorRT-LLM][INFO] TRTGptModel maxBeamWidth: 1
[TensorRT-LLM][INFO] TRTGptModel maxSequenceLen: 4098
[TensorRT-LLM][INFO] TRTGptModel maxDraftLen: 0
[TensorRT-LLM][INFO] TRTGptModel mMaxAttentionWindowSize: 4098
[TensorRT-LLM][INFO] TRTGptModel enableTrtOverlap: 0
[TensorRT-LLM][INFO] TRTGptModel normalizeLogProbs: 1
[TensorRT-LLM][INFO] TRTGptModel maxNumTokens: 8192
[TensorRT-LLM][INFO] TRTGptModel maxInputLen: 4097  = maxSequenceLen - 1 since chunked context is enabled
[TensorRT-LLM][INFO] Capacity Scheduler Policy: GUARANTEED_NO_EVICT
[TensorRT-LLM][INFO] Context Chunking Scheduler Policy: FIRST_COME_FIRST_SERVED
[TensorRT-LLM][INFO] Loaded engine size: 6214 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 928.77 MiB for execution context memory.
[TensorRT-LLM][INFO] [MS] Running engine with multi stream info
[TensorRT-LLM][INFO] [MS] Number of aux streams is 1
[TensorRT-LLM][INFO] [MS] Number of total worker streams is 2
[TensorRT-LLM][INFO] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 6166 (MiB)
[TensorRT-LLM][INFO] [MS] Running engine with multi stream info
[TensorRT-LLM][INFO] [MS] Number of aux streams is 1
[TensorRT-LLM][INFO] [MS] Number of total worker streams is 2
[TensorRT-LLM][INFO] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 6166 (MiB)
[TensorRT-LLM][INFO] Switching optimization profile from: 0 to 1. Please ensure there are no enqueued operations pending in this context prior to switching profiles
[TensorRT-LLM][INFO] [MS] Running engine with multi stream info
[TensorRT-LLM][INFO] [MS] Number of aux streams is 1
[TensorRT-LLM][INFO] [MS] Number of total worker streams is 2
[TensorRT-LLM][INFO] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 6166 (MiB)
[TensorRT-LLM][INFO] Switching optimization profile from: 0 to 2. Please ensure there are no enqueued operations pending in this context prior to switching profiles
[TensorRT-LLM][INFO] [MS] Running engine with multi stream info
[TensorRT-LLM][INFO] [MS] Number of aux streams is 1
[TensorRT-LLM][INFO] [MS] Number of total worker streams is 2
[TensorRT-LLM][INFO] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 6166 (MiB)
[TensorRT-LLM][INFO] Switching optimization profile from: 0 to 3. Please ensure there are no enqueued operations pending in this context prior to switching profiles
[TensorRT-LLM][INFO] [MS] Running engine with multi stream info
[TensorRT-LLM][INFO] [MS] Number of aux streams is 1
[TensorRT-LLM][INFO] [MS] Number of total worker streams is 2
[TensorRT-LLM][INFO] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 6166 (MiB)
[TensorRT-LLM][INFO] Switching optimization profile from: 0 to 4. Please ensure there are no enqueued operations pending in this context prior to switching profiles
[TensorRT-LLM][INFO] [MS] Running engine with multi stream info
[TensorRT-LLM][INFO] [MS] Number of aux streams is 1
[TensorRT-LLM][INFO] [MS] Number of total worker streams is 2
[TensorRT-LLM][INFO] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 6166 (MiB)
[TensorRT-LLM][INFO] Switching optimization profile from: 0 to 5. Please ensure there are no enqueued operations pending in this context prior to switching profiles
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 1.14 GB GPU memory for runtime buffers.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 4.35 GB GPU memory for decoder.
[TensorRT-LLM][INFO] Memory usage when calculating max tokens in paged kv cache: total: 79.10 GiB, available: 63.62 GiB
[TensorRT-LLM][INFO] Number of blocks in KV cache primary pool: 4607
[TensorRT-LLM][INFO] Number of blocks in KV cache secondary pool: 0, onboard blocks to primary memory before reuse: true
[TensorRT-LLM][INFO] Max KV cache pages per sequence: 65
[TensorRT-LLM][INFO] Number of tokens per block: 64.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 62.99 GiB for max tokens in paged KV cache (294848).
[TensorRT-LLM][INFO] Executor instance created by worker
Starting response daemon...Executor started.

Request serving started.
Starting statistics collection.
Collecting live stats...
Benchmark started.
Request serving stopped.
Collecting last stats...
Ending statistics collection.
Stop received.
Stopping response parsing.
Collecting last responses before shutdown.
Completed request parsing.
Parsing stopped.
Request generator successfully joined.
Statistics process successfully joined.
===========================================================
= ENGINE DETAILS
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
Engine Directory:       /tmp/meta-llama/Llama-2-7b-hf/tp_1_pp_1
TensorRT-LLM Version:   0.12.0.dev2024073000
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

Benchmark Shutdown called!
Shutting down ExecutorServer.
[TensorRT-LLM][INFO] Orchestrator sendReq thread exiting
[TensorRT-LLM][INFO] Orchestrator recv thread exiting
Executor shutdown.
[TensorRT-LLM][INFO] Leader sendThread exiting
[TensorRT-LLM][INFO] Leader recvReq thread exiting
```

> [!WARNING] Some statistics are not reported.
> There are some statistics that are not reported in the summary (typically as 0.0). These statistics
> are not available currently.
