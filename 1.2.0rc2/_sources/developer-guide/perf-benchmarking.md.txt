(perf-benchmarking)=

# TensorRT LLM Benchmarking

```{important}
This benchmarking suite is a work in progress.
Expect breaking API changes.
```

TensorRT LLM provides the `trtllm-bench` CLI, a packaged benchmarking utility that aims to make it
easier for users to reproduce our officially published [performance overview](./perf-overview.md#throughput-measurements). `trtllm-bench` provides the follows:

- A streamlined way to build tuned engines for benchmarking for a variety of models and platforms.
- An entirely Python workflow for benchmarking.
- Ability to benchmark various flows and features within TensorRT LLM.

`trtllm-bench` executes all benchmarks using `in-flight batching` -- for more information see
the [in-flight batching section](../features/attention.md#inflight-batching) that describes the concept
in further detail.

To benchmark the OpenAI-compatible `trtllm-serve`, please refer to the [run benchmarking with `trtllm-serve`](../commands/trtllm-serve/run-benchmark-with-trtllm-serve.md) section.

## Before Benchmarking

For rigorous benchmarking where consistent and reproducible results are critical, proper GPU configuration is essential. These settings help maximize GPU utilization, eliminate performance variability, and ensure optimal conditions for accurate measurements. While not strictly required for normal operation, we recommend applying these configurations when conducting performance comparisons or publishing benchmark results.

### Persistence mode

Ensure persistence mode is enabled to maintain consistent GPU state:
```shell
sudo nvidia-smi -pm 1
```

### GPU Clock Management

Allow the GPU to dynamically adjust its clock speeds based on workload and temperature. While locking clocks at maximum frequency might seem beneficial, it can sometimes lead to thermal throttling and reduced performance. Reset GPU clocks using:
```shell
sudo nvidia-smi -rgc
```

### Set power limits

First query the maximum power limit:
```shell
nvidia-smi -q -d POWER
```
Then configure the GPU to operate at its maximum power limit for consistent performance:
```shell
sudo nvidia-smi -pl <max_power_limit>
```

### Boost settings

Potentially a GPU may support boost levels. First query available boost levels:
```shell
sudo nvidia-smi boost-slider -l
```
If supported, enable the boost slider using one of the available levels for maximum performance:
```shell
sudo nvidia-smi boost-slider --vboost <max_boost_slider>
```


## Throughput Benchmarking

### Limitations and Caveats

#### Validated Networks for Benchmarking

While `trtllm-bench` should be able to run any network that TensorRT LLM supports, the following are the list
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

For more information about quantization, refer to [](../features/quantization.md) and
the [support matrix](../features/quantization.md#model-supported-matrix) of the supported quantization methods for each network.

```{tip}
Although TensorRT LLM supports more quantization modes than listed above, `trtllm-bench` currently only configures for
a smaller subset.
```

### Preparing a Dataset

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
python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer meta-llama/Llama-3.1-8B token-norm-dist --input-mean 128 --output-mean 128 --input-stdev 0 --output-stdev 0 --num-requests 1000 > /tmp/synthetic_128_128.txt
```

### Running with the PyTorch Workflow

To benchmark the PyTorch backend (`tensorrt_llm._torch`), use the following command with [dataset](#preparing-a-dataset) generated from previous steps. The `throughput` benchmark initializes the backend by tuning against the dataset provided via `--dataset` (or the other build mode settings described above).

Note that CUDA graph is enabled by default. You can add additional pytorch config with `--extra_llm_api_options` followed by the path to a YAML file. For more details, please refer to the help text by running the command with `--help`.

```{tip}
The command below specifies the `--model_path` option. The model path is optional and used only when you want to run a locally
stored checkpoint. When using `--model_path`, the `--model` is still required for reporting reasons and in order to look up parameters
for build heuristics.
```

```shell
trtllm-bench --model meta-llama/Llama-3.1-8B \
  --model_path /Ckpt/Path/To/Llama-3.1-8B \
  throughput \
  --dataset /tmp/synthetic_128_128.txt \
  --backend pytorch

# Example output
<snip verbose logging>
===========================================================
= PyTorch backend
===========================================================
Model:                  meta-llama/Llama-3.1-8B
Model Path:             /Ckpt/Path/To/Llama-3.1-8B
TensorRT LLM Version:   0.17.0
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

When enabling streaming, time to first token (TTFT) and inter-token latency (ITL) metrics will also be recorded.
```shell
trtllm-bench --model meta-llama/Llama-3.1-8B \
  --model_path /Ckpt/Path/To/Llama-3.1-8B \
  throughput \
  --dataset /tmp/synthetic_128_128.txt \
  --backend pytorch
```

Alternatively, users can benchmark the low latency mode:
```shell
trtllm-bench --model meta-llama/Llama-3.1-8B \
  --model_path /Ckpt/Path/To/Llama-3.1-8B \
  latency \
  --dataset /tmp/synthetic_128_128.txt \
  --backend pytorch
```

#### Benchmarking with LoRA Adapters in PyTorch workflow

The PyTorch workflow supports benchmarking with LoRA (Low-Rank Adaptation) adapters. This requires preparing a dataset with LoRA metadata and configuring the LoRA settings.

**Preparing LoRA Dataset**

Use `prepare_dataset.py` with LoRA-specific options to generate requests with LoRA metadata:

```shell
python3 benchmarks/cpp/prepare_dataset.py \
  --stdout \
  --rand-task-id 0 1 \
  --tokenizer /path/to/tokenizer \
  --lora-dir /path/to/loras \
  token-norm-dist \
  --num-requests 100 \
  --input-mean 128 \
  --output-mean 128 \
  --input-stdev 16 \
  --output-stdev 24 \
  > synthetic_lora_data.json
```

Key LoRA options:
- `--lora-dir`: Parent directory containing LoRA adapter subdirectories named by their task IDs (e.g., `0/`, `1/`, etc.)
- `--rand-task-id`: Range of LoRA task IDs to randomly assign to requests
- `--task-id`: Fixed LoRA task ID for all requests (alternative to `--rand-task-id`)

The generated dataset will include LoRA request metadata. Below is an example of a single such request data entry:

```json
{
  "task_id": 0,
  "input_ids": [3452, 88226, 102415, ...],
  "output_tokens": 152,
  "lora_request": {
    "lora_name": "lora_0",
    "lora_int_id": 0,
    "lora_path": "/path/to/loras/0"
  }
}
```

**LoRA Configuration**

Create an `extra-llm-api-options.yaml` file with LoRA configuration:

```yaml
lora_config:
  lora_dir:
    - /path/to/loras/0
    - /path/to/loras/1
  max_lora_rank: 64
  lora_target_modules:
    - attn_q
    - attn_k
    - attn_v
  trtllm_modules_to_hf_modules:
    attn_q: q_proj
    attn_k: k_proj
    attn_v: v_proj
```

**Running LoRA Benchmark**

```shell
trtllm-bench --model /path/to/base/model \
  throughput \
  --dataset synthetic_lora_data.json \
  --backend pytorch \
  --extra_llm_api_options extra-llm-api-options.yaml
```

```{note}
The LoRA directory structure should have task-specific subdirectories named by their task IDs (e.g., `loras/0/`, `loras/1/`).
Each subdirectory should contain the LoRA adapter files for that specific task.
```

#### Running multi-modal models in the PyTorch Workflow

To benchmark multi-modal models with PyTorch workflow, you can follow the similar approach as above.

First, prepare the dataset:
```python
python ./benchmarks/cpp/prepare_dataset.py \
  --tokenizer Qwen/Qwen2-VL-2B-Instruct \
  --stdout \
  dataset \
  --dataset-name lmms-lab/MMMU \
  --dataset-split test \
  --dataset-image-key image \
  --dataset-prompt-key question \
  --num-requests 10 \
  --output-len-dist 128,5 > mm_data.jsonl
```
It will download the media files to `/tmp` directory and prepare the dataset with their paths. Note that the `prompt` fields are texts and not tokenized ids. This is due to the fact that
the `prompt` and the media (image/video) are processed by a preprocessor for multimodal files.

Sample dataset for multimodal:
```
{"task_id":0,"prompt":"Brahma Industries sells vinyl replacement windows to home improvement retailers nationwide. The national sales manager believes that if they invest an additional $25,000 in advertising, they would increase sales volume by 10,000 units. <image 1> What is the total contribution margin?","media_paths":["/tmp/tmp9so41y3r.jpg"],"output_tokens":126}
{"task_id":1,"prompt":"Let us compute for the missing amounts under work in process inventory, what is the cost of goods manufactured? <image 1>","media_paths":["/tmp/tmpowsrb_f4.jpg"],"output_tokens":119}
{"task_id":2,"prompt":"Tsuji is reviewing the price of a 3-month Japanese yen/U.S. dollar currency futures contract, using the currency and interest rate data shown below. Because the 3-month Japanese interest rate has just increased to .50%, Itsuji recognizes that an arbitrage opportunity exists nd decides to borrow $1 million U.S. dollars to purchase Japanese yen. Calculate the yen arbitrage profit from Itsuji's strategy, using the following data: <image 1> ","media_paths":["/tmp/tmpxhdvasex.jpg"],"output_tokens":126}
...
```

Run the benchmark:
```python
trtllm-bench --model Qwen/Qwen2-VL-2B-Instruct \
  throughput \
  --dataset mm_data.jsonl \
  --backend pytorch \
  --num_requests 10 \
  --max_batch_size 4 \
  --modality image
```


Sample output:
```
===========================================================
= REQUEST DETAILS
===========================================================
Number of requests:             10
Number of concurrent requests:  5.3019
Average Input Length (tokens):  411.6000
Average Output Length (tokens): 128.7000
===========================================================
= WORLD + RUNTIME INFORMATION
===========================================================
TP Size:                1
PP Size:                1
EP Size:                None
Max Runtime Batch Size: 4
Max Runtime Tokens:     12288
Scheduling Policy:      GUARANTEED_NO_EVICT
KV Memory Percentage:   90.00%
Issue Rate (req/sec):   1.4117E+17

===========================================================
= PERFORMANCE OVERVIEW
===========================================================
Request Throughput (req/sec):                     1.4439
Total Output Throughput (tokens/sec):             185.8351
Per User Output Throughput (tokens/sec/user):     38.1959
Per GPU Output Throughput (tokens/sec/gpu):       185.8351
Total Token Throughput (tokens/sec):              780.1607
Total Latency (ms):                               6925.4963
Average request latency (ms):                     3671.8441

-- Request Latency Breakdown (ms) -----------------------

[Latency] P50    : 3936.3022
[Latency] P90    : 5514.4701
[Latency] P95    : 5514.4701
[Latency] P99    : 5514.4701
[Latency] MINIMUM: 2397.1047
[Latency] MAXIMUM: 5514.4701
[Latency] AVERAGE: 3671.8441

===========================================================
= DATASET DETAILS
===========================================================
Dataset Path:         /workspaces/tensorrt_llm/mm_data.jsonl
Number of Sequences:  10

-- Percentiles statistics ---------------------------------

        Input              Output           Seq. Length
-----------------------------------------------------------
MIN:   167.0000           119.0000           300.0000
MAX:  1059.0000           137.0000          1178.0000
AVG:   411.6000           128.7000           540.3000
P50:   299.0000           128.0000           427.0000
P90:  1059.0000           137.0000          1178.0000
P95:  1059.0000           137.0000          1178.0000
P99:  1059.0000           137.0000          1178.0000
===========================================================
```

**Notes and Limitations**:
- Only image datasets are supported for now.
- `--output-len-dist` is a required argument for multimodal datasets.
- Tokenizer is unused during the prepare step but it is still a required argument.
- Since the images are converted to tokens when the model is run, `trtllm-bench` uses a default large value for the maximum input sequence length when setting up the execution settings.
  You can also modify the behavior by specifying a different value with the flag `--max_input_len` that suits your use-case.

#### Quantization in the PyTorch Flow

To run a quantized benchmark with `trtllm-bench` utilizing the PyTorch flow, you will need to use a pre-quantized
checkpoint. For the Llama-3.1 models, TensorRT LLM provides the following checkpoints via HuggingFace:

- [`nvidia/Llama-3.1-8B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)
- [`nvidia/Llama-3.1-70B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8)
- [`nvidia/Llama-3.1-405B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8)

To understand more about how to quantize your own checkpoints, refer to ModelOpt [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/deployment/1_tensorrt_llm.html).

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

If you would like to force the KV cache quantization, you can specify the following in the YAML file to force the precision
when the checkpoint precision is `null`:

```yaml
kv_cache_config:
  dtype: fp8
```

```{tip}
The two valid values for `kv_cache_config.dtype` are `auto` and `fp8`.
```
