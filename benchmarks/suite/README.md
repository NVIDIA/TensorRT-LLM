# TensorRT-LLM Benchmarking

> [!WARNING] Work in Progress
> This benchmarking suite is a current work in progress and is prone to large changes.

This package is the official benchmarking suite for TensorRT-LLM. This benchmark will be updated
as development of TensorRT-LLM continues.

## Installation

From this folder, run `pip install -r requirements.txt` to install the extra dependencies required for this tool.

### Available Build and Benchmark Options

The following model options are available for benchmarking models.

| Option | Required | Default | Description |
| :- | :-: | :-: | :- |
| `--model` | Y | - | The name of the model to benchmark. |
| `--dtype` | N | `float16` | The datatype of the weights. |
| `--max-batch-size` | Y | - | The batch size to build the engine with for the benchmark. |
| `--kv-dtype` | N | `float16` | The datatype to store the KV Cache in. |
| `--kv-cache-free-gpu-mem-fraction` | N | `0.98` | The percentage of free memory that the KV cache is allowed to occupy. |
| `--quantization` | N | `None` |The quantization algorithm to be used when benchmarking. See the [documentation](https://nvidia.github.io/TensorRT-LLM/precision.html) for more information|
| `--workspace` | N | `/tmp` | The directory to store benchmarking intermediate files. |
| `--tensor-parallel-size` | N | `1` | Number of tensor parallel shards to run the benchmark with. |
| `--pipeline-parallel-size` | N | `1` | Number of pipeline parallel shards to run the benchmark with. |

#### Supported Networks for Benchmarking

- [`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b)
- [`tiiuae/falcon-40b`](https://huggingface.co/tiiuae/falcon-40b)
- [`tiiuae/falcon-180B`](https://huggingface.co/tiiuae/falcon-180B)
- [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [`meta-llama/Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [`meta-llama/Llama-2-70b-hf`](https://huggingface.co/meta-llama/Llama-2-70b-hf)
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

## Static Benchmarking a Network

In order to benchmark a static batch for a network, run a command like the following:

```shell
cd tensorrt_llm_bench/
python benchmark.py --model tiiuae/falcon-7b static --isl 128 --osl 128 --max-batch-size 1
```

This command line will build a unique engine for the configuration and run the benchmark using
the `gptSessionBenchmark` binary. You need to build the TensorRT-LLM wheel with the `--benchmarks` flag for this binary to be compiled:

```shell
python3 ./scripts/build_wheel.py --benchmarks <other options>
```

If you've already compiled the wheel without benchmarks, you can build the benchmarking binaries with the following after the fact:

```shell
pushd cpp/build/
make -j benchmarks
popd
```

The complete list of arguments for static benchmarking are as follows:
| Option | Required | Default | Description |
| :- | :-: | :-: | :- |
| `--isl` | Y | - | The input sequence length to pass in during benchmark. |
| `--osl` | Y | - | The output sequence length to generate in the benchmark. |
| `--gpt-session-path` | N | `../../cpp/build/benchmarks/gptSessionBenchmark` | The path to the built gptSessionBenchmark binary. |
| `--warm-up-runs` | N | `2` | The number of warm up runs to run before benchmarking actual results. |
| `--num-runs` | N | `10` | The number runs to generate benchmarking results from.  |
| `--duration` | N | `60` | The minimum iteration time, in seconds, to measure.  |

> [!WARNING]
> `gptSession` will be deprecated for the 1.0 release of TensorRT-LLM. This command line will change in order to match and update benchmarks accordingly.


## Inflight Benchmarking with a Dataset

This section covers how to benchmark TensorRT-LLM using inflight batching.

### Workflow

The workflow for inflight batching is slightly different than the [static scenario](#static-benchmarking-a-network) as it requires a workload of requests instead of a single static batch. The following is the workflow for benchmarking using inflight batching:

1. Prepare a dataset to drive the inflight batching benchmark.
2. Run the `inflight` benchmarking subcommand and provide the dataset from step 1.

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
benchmarks/cpp/prepare_dataset.py --tokenizer meta-llama/Llama-2-7b-hf token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1000 --stdout
```

You can pipe the above command to a file to reuse the same dataset, or simply pipe its output to the
benchmark script (example below).

### Running a Dataset with the Benchmarker

Once you've generated a dataset (see [above](#preparing-a-dataset)), you can run the benchmarker
in one of two ways:

```shell
benchmarks/suite/tensorrt_llm_bench/benchmark.py --model $HF_MODEL_NAME --max-batch-size $BATCH_SIZE < $DATASET_PATH
```

> [!INFO] Alternative to piping.
> There is also a `--dataset` option for `benchmark.py` that can be used instead of piping a file.

or

```shell
benchmarks/cpp/prepare_dataset.py --tokenizer $HF_MODEL_NAME --input-mean $ISL --output-mean $OSL --num-requests $NUM_REQUESTS --stdout | benchmarks/suite/tensorrt_llm_bench/benchmark.py --model $HF_MODEL_NAME --max-batch-size $BATCH_SIZE --request-rate $REQUEST_RATE
```

#### How the Benchmarker Works

The benchmarker will read in a data file or standard input (stdin) as a stream where a single line contains
a complete JSON request entry. The process that the benchmarker is as follows:

1. Iterate over all input requests. If `logits` is specified, construct the request using the specified
list of logits. Otherwise, tokenize the `prompt` with as specified by `--model $HF_MODEL_NAME`.
2. Build the TensorRT-LLM engine.
3. Submit the dataset to the TensorRT-LLM `Executor` API at the request rate specified by `--request-rate $REQUEST_RATE`
4. Wait for all requests to return, compute statistics, then report out results.

When the benchmark runs successfully, you will see a report out of the run similar to the following:

```
[RANK 0] Submitting requests...
[RANK 0] Completed request submission.
[RANK 0] Calculating results.
[RANK 0] Reporting...
[RANK 0] JSON: {'benchmark_cmd': '', 'binary': '', 'build_cmd': 'trtllm-build --output_dir /tmp/meta-llama/llama-2-7b-hf --model_config /tmp/generated_config.json --workers 1 --max_batch_size 1024 --max_input_len 128 --max_output_len 128 --max_num_tokens 8000 --context_fmha enable --gpt_attention_plugin float16 --paged_kv_cache enable --multiple_profiles enable --gemm_plugin float16', 'first_token_latency': 0.0, 'inflight_batching': True, 'kv_mem_fraction': 0.98, 'latency_units': 'ms', 'max_batch_size': 1024, 'max_tokens': 8000, 'model': 'meta-llama/Llama-2-7b-hf', 'peak_gpu_mem_units': 'GB', 'peak_gpu_mem': 0.0, 'scheduler': 'Max Utilization', 'throughput_units': 'tokens/second', 'throughput': 17634.422523488243, 'time_per_output_token': 0.0, 'total_input_tokens': 128000, 'total_latency': 7.258530855178833, 'total_output_tokens': 128000}
===========================================================
= METADATA
===========================================================
Model:                  meta-llama/Llama-2-7b-hf
TP Size:                1
PP Size:                1
Scheduling Policy:      Max Utilization
In-flight Batcher?:     True
Dtype:                  float16
KV Cache Dtype:         FP8
Quantization:           FP8
KV Memory Percentage:   98.0%

===========================================================
= ENGINE DETAILS
===========================================================
Engine Directory:       /tmp/meta-llama/llama-2-7b-hf
Max Batch Size:         1024
Total Input Length:     128000
Total Output Length:    128000
Max Tokens:             8000

===========================================================
= STATISTICS
===========================================================
Throughput (tokens/second):     17634.422523488243
Total Latency (ms):             7258.5309
First Token Latency (ms):       0.0
Token-to-token Latency (ms):    0.0
Peak GPU Memory Usage (GB):     0.0

===========================================================
= COMMANDS
===========================================================
Build: trtllm-build --output_dir /tmp/meta-llama/llama-2-7b-hf --model_config /tmp/generated_config.json --workers 1 --max_batch_size 1024 --max_input_len 128 --max_output_len 128 --max_num_tokens 8000 --context_fmha enable --gpt_attention_plugin float16 --paged_kv_cache enable --multiple_profiles enable --gemm_plugin float16
Benchmark:

[RANK 0] Terminating.
```

> [!WARNING] Some statistics are not reported.
> There are some statistics that are not reported in the summary (typically as 0.0). These statistics
> are not available currently.


That's it! -- you've successfully benchmarked TensorRT-LLM!
