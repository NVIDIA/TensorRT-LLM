# TensorRT-LLM Benchmarking

**WORK IN PROGRESS**

This package is the official benchmarking suite for TensorRT-LLM. This benchmark will be updated
as development of TensorRT-LLM continues.

## Installation

From this folder, run `pip install -r requirements.txt` to install the extra dependencies required for this tool.

### Available Model Options

The following model options are available for benchmarking models.

| Option | Required | Default | Description |
| :- | :-: | :-: | :- |
| `--model` | Y | - | The name of the model to benchmark. |
| `--dtype` | N | `float16` | The datatype of the weights. |
| `--kv-dtype` | N | `float16` | The datatype to store the KV Cache in. |
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

TensorRT-LLM supports a number of quanization modes. For more information about quantization, see the [documentation](https://nvidia.github.io/TensorRT-LLM/precision.html).

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
python benchmark.py --model tiiuae/falcon-7b static --isl 128 --osl 128 --batch 1
```

This command line will build a unique engine for the configuration and run the benchmark using
the `gptSessionBenchmark` binary. You need to build the TensorRT-LLM wheel with the `--benchmarks` flag for this binary to be compiled:

```shell
python3 ./scripts/build_wheel.py --benchmarks <other options>
```

The complete list of arguments are given here:
| Option | Required | Default | Description |
| :- | :-: | :-: | :- |
| `--batch` | Y | - | The batch size to benchmark. |
| `--isl` | Y | - | The input sequence length to pass in during benchmark. |
| `--osl` | Y | - | The output sequence length to generate in the benchmark. |
| `--gpt-session-path` | N | `../../cpp/build/benchmarks/gptSessionBenchmark` | The path to the built gptSessionBenchmark binary. |
| `--max-tokens-in-kv-cache` | N | `None` | The maximum number of tokens to store in the KV Cache during benchmarking. |
| `--kv-cache-mem-percent` | N | `0.9` | The percentage of free memory that the KV cache is allowed to occupy. |
| `--warm-up-runs` | N | `2` | The number of warm up runs to run before benchmarking actual results. |
| `--num-runs` | N | `10` | The number runs to generate benchmarking results from.  |
| `--duration` | N | `60` | The minimum iteration time, in seconds, to measure.  |

> [!WARNING]
> `gptSession` will be deprecated for the 1.0 release of TensorRT-LLM. This command line will change in order to match and update benchmarks accordingly.
