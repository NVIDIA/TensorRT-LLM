# DeepSeek-V4

This guide walks you through the examples to run DeepSeek-V4 models using NVIDIA TensorRT LLM with
the PyTorch backend.

DeepSeek-V4 uses the `DeepseekV4ForCausalLM` architecture in TensorRT LLM. Compared with
DeepSeek-V3/R1/V3.2, it has a separate model implementation and sparse attention path. Use the
commands in this guide as starting points and tune the parallelism and memory settings for your
checkpoint and workload.

Please refer to [this guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html)
for how to build TensorRT LLM from source and start a TRT-LLM Docker container.

> [!NOTE]
> This guide assumes that you replace placeholder values such as `<YOUR_MODEL_DIR>` with the
> appropriate paths. Commands in this guide target the PyTorch backend.


## Table of Contents

- [DeepSeek-V4](#deepseek-v4)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Downloading the Model Weights](#downloading-the-model-weights)
  - [Quick Start](#quick-start)
    - [Run a single inference](#run-a-single-inference)
    - [Run chat-style prompts](#run-chat-style-prompts)
    - [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
  - [Benchmarking](#benchmarking)
  - [Evaluation](#evaluation)
  - [Serving](#serving)
    - [trtllm-serve](#trtllm-serve)
    - [OpenAI-compatible request](#openai-compatible-request)
  - [Advanced Configuration](#advanced-configuration)
    - [Parallelism](#parallelism)
    - [Sparse attention](#sparse-attention)
    - [KV cache](#kv-cache)
    - [Quantized checkpoints](#quantized-checkpoints)
  - [DSpark confidence scheduling (prototype)](#dspark-confidence-scheduling-prototype)
  - [Notes and Troubleshooting](#notes-and-troubleshooting)


## Hardware Requirements

DeepSeek-V4 is supported on Hopper (`SM90`) and Blackwell (`SM100+`) GPUs in the PyTorch backend.

DeepSeek-V4 has two model scales, and each scale provides Base and Instruct checkpoints. The table
below follows the model list published on the
[DeepSeek-V4 Hugging Face model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro):

| Checkpoint | Total Params | Activated Params | Context Length | Precision |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-Base | 284B | 13B | 1M | FP8 Mixed |
| DeepSeek-V4-Flash | 284B | 13B | 1M | FP4 + FP8 Mixed |
| DeepSeek-V4-Pro-Base | 1.6T | 49B | 1M | FP8 Mixed |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M | FP4 + FP8 Mixed |

The minimum number of GPUs depends on the model scale, checkpoint precision, KV cache budget,
maximum sequence length, and runtime batch size. For initial bring-up, an 8xH200 or 8xB200 node is
enough for Flash checkpoints. An 8xB200 node is also enough for the FP4 + FP8 mixed DeepSeek-V4-Pro
checkpoint. DeepSeek-V4-Pro-Base is larger because it uses FP8 mixed precision; if you want to keep
the deployment on a single node, use an 8xB300 node. Multi-node deployments are still recommended
for larger KV cache budgets, longer context windows, or higher throughput targets. Tune
`--tp_size`, `--ep_size`, `--max_num_tokens`, and the KV cache memory fraction for your deployment
target.

Hopper (`SM90`) and Blackwell/Rubin (`SM100`, `SM103`, `SM107`) support 128- or 256-token blocks. Hopper uses the
`fp8_ds_mla` cache layout; on `SM120`/`SM121`, this layout requires 256-token blocks.


## Downloading the Model Weights

Choose one of the DeepSeek-V4 checkpoint IDs:

| Checkpoint | Hugging Face model ID | Prompt format |
| --- | --- | --- |
| DeepSeek-V4-Flash-Base | `deepseek-ai/DeepSeek-V4-Flash-Base` | Raw completion |
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash` | Chat/Instruct |
| DeepSeek-V4-Pro-Base | `deepseek-ai/DeepSeek-V4-Pro-Base` | Raw completion |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` | Chat/Instruct |

Then download the weights:

```bash
git lfs install
MODEL_ID=deepseek-ai/DeepSeek-V4-Flash
git clone https://huggingface.co/${MODEL_ID} <YOUR_MODEL_DIR>
```

At minimum, the checkpoint config should identify the architecture as DeepSeek-V4:

```json
{
  "architectures": ["DeepseekV4ForCausalLM"],
  "model_type": "deepseek_v4"
}
```

Do not replace the full checkpoint config with this minimal snippet. TensorRT LLM also reads
DeepSeek-V4-specific sparse attention fields such as `compress_ratios`, `window_size` or
`sliding_window`, and indexer settings from the checkpoint config unless you provide a complete
override through `sparse_attention_config`.


## Quick Start

### Run a single inference

To quickly run DeepSeek-V4, use [examples/llm-api/quickstart_advanced.py](../../../llm-api/quickstart_advanced.py):

```bash
cd examples/llm-api
python quickstart_advanced.py \
  --model_dir <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --moe_ep_size 8 \
  --tokens_per_block 128 \
  --max_num_tokens 8192 \
  --max_seq_len 4096 \
  --kv_cache_fraction 0.5
```

The command above assumes one 8-GPU node. If you use a different number of GPUs, adjust `--tp_size`
and `--moe_ep_size` so that the requested parallelism matches your available world size. DeepSeek-V4
checkpoints advertise a 1M-token context window; for bring-up, set `--max_seq_len` and the KV cache
memory fraction explicitly, then increase them according to your memory budget.

### Run chat-style prompts

DeepSeek-V4 Instruct checkpoints (`DeepSeek-V4-Flash` and `DeepSeek-V4-Pro`) use the checkpoint
reference chat/message format. TensorRT LLM provides a `deepseek_v4` tokenizer wrapper for this
format. Use `custom_tokenizer="deepseek_v4"` only with Instruct checkpoints and chat-style prompts.

Base checkpoints (`DeepSeek-V4-Flash-Base` and `DeepSeek-V4-Pro-Base`) are completion models. For
Base checkpoints, do not apply a chat template and do not pass `custom_tokenizer="deepseek_v4"`;
send raw text prompts instead.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

def main():
    llm = LLM(
        model="<YOUR_MODEL_DIR>",
        backend="pytorch",
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        custom_tokenizer="deepseek_v4",
        kv_cache_config=KvCacheConfig(
            tokens_per_block=128,
            free_gpu_memory_fraction=0.5,
        ),
        max_seq_len=4096,
        max_num_tokens=8192,
    )

    messages = [{"role": "user", "content": "Explain TensorRT LLM in one paragraph."}]
    prompt = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = llm.generate([prompt], SamplingParams(max_tokens=128))
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

### Multi-Token Prediction (MTP)

If the checkpoint contains MTP layers, run MTP speculative decoding with the one-model flow:

```bash
cd examples/llm-api
python quickstart_advanced.py \
  --model_dir <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --moe_ep_size 8 \
  --tokens_per_block 128 \
  --max_num_tokens 8192 \
  --max_seq_len 4096 \
  --kv_cache_fraction 0.5 \
  --spec_decode_algo MTP \
  --spec_decode_max_draft_len N
```

`N` is the number of draft tokens to predict. Start with `N=1` for bring-up, then increase it after
validating accuracy and latency for your workload.


## Benchmarking

The following example prepares a synthetic dataset and runs `trtllm-bench` throughput on one 8-GPU
Blackwell node:

```bash
trtllm-bench --model <MODEL_ID> \
  --model_path <YOUR_MODEL_DIR> \
  prepare-dataset \
  --output /tmp/deepseek_v4_1k1k.txt \
  token-norm-dist \
  --input-mean 1024 \
  --output-mean 1024 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 256

cat > /tmp/deepseek_v4_config.yml <<EOF
enable_attention_dp: true
attention_dp_config:
  batching_wait_iters: 0
  enable_balance: true
  timeout_iters: 60
kv_cache_config:
  tokens_per_block: 128
  dtype: fp8
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
moe_config:
  backend: TRTLLM
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 1
EOF

trtllm-bench --model <MODEL_ID> \
  --model_path <YOUR_MODEL_DIR> \
  throughput \
  --tp 8 \
  --ep 8 \
  --dataset /tmp/deepseek_v4_1k1k.txt \
  --max_batch_size 256 \
  --max_num_tokens 8192 \
  --concurrency 2048 \
  --num_requests 6144 \
  --kv_cache_free_gpu_mem_fraction 0.9 \
  --config /tmp/deepseek_v4_config.yml
```

The example enables attention DP because it is typically beneficial for high-throughput, large-batch
workloads. It also uses FP8 KV cache (`kv_cache_config.dtype: fp8`), which is the recommended
starting point for benchmarking DeepSeek-V4 throughput. For checkpoints with MTP layers, enable MTP
for benchmarking as well: use `num_nextn_predict_layers: 1` for throughput-oriented runs, and use
`num_nextn_predict_layers: 3` for low-latency runs. When `enable_attention_dp` is enabled,
`--max_batch_size` is the maximum batch size per local rank; use `--concurrency` high enough to
saturate all ranks. Tune `--max_batch_size`, `--max_num_tokens`, `--concurrency`, MTP depth, and the
KV cache memory fraction for the target ISL/OSL distribution.


## Evaluation

Evaluate model accuracy using `trtllm-eval`. The following commands are for Instruct checkpoints and
apply the DeepSeek-V4 chat template through `--custom_tokenizer deepseek_v4` and
`--apply_chat_template`. For Base checkpoints, remove both flags because Base models expect raw
completion prompts. `--custom_tokenizer` is a top-level `trtllm-eval` option, so keep it before the
dataset subcommand such as `mmlu`, `gsm8k`, or `gpqa_diamond`.

1. Prepare a configuration file:

```bash
cat > ./deepseek_v4_config.yml <<EOF
kv_cache_config:
  tokens_per_block: 128
  free_gpu_memory_fraction: 0.5
moe_config:
  backend: TRTLLM
EOF
```

2. Evaluate MMLU with an Instruct checkpoint:

```bash
trtllm-eval --model <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --ep_size 8 \
  --max_batch_size 16 \
  --max_num_tokens 8192 \
  --max_seq_len 4096 \
  --custom_tokenizer deepseek_v4 \
  --config ./deepseek_v4_config.yml \
  mmlu \
  --apply_chat_template
```

3. Evaluate GSM8K with an Instruct checkpoint:

```bash
trtllm-eval --model <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --ep_size 8 \
  --max_batch_size 16 \
  --max_num_tokens 8192 \
  --max_seq_len 4096 \
  --custom_tokenizer deepseek_v4 \
  --config ./deepseek_v4_config.yml \
  gsm8k \
  --apply_chat_template \
  --system_prompt "Solve the problem carefully. End your response with a final line exactly in the form #### <answer>, using the simplest numeric form without units or trailing zeros."
```

The `--system_prompt` constrains the answer format so that the lm-eval `strict-match`
regex (which expects a final `#### <answer>` line) can pick up the model's answer.
Without it, DeepSeek-V4 Instruct checkpoints often return the correct value in a
free-form sentence, which `flexible-extract` recovers but `strict-match` does not.

4. Evaluate GPQA Diamond with an Instruct checkpoint:

```bash
trtllm-eval --model <YOUR_MODEL_DIR> \
  --tp_size 8 \
  --ep_size 8 \
  --max_batch_size 16 \
  --max_num_tokens 8192 \
  --max_seq_len 4096 \
  --custom_tokenizer deepseek_v4 \
  --config ./deepseek_v4_config.yml \
  gpqa_diamond \
  --apply_chat_template
```


## Serving

### trtllm-serve

Create a serving config:

```bash
cat > ./deepseek_v4_serve.yml <<EOF
kv_cache_config:
  tokens_per_block: 128
  free_gpu_memory_fraction: 0.5
enable_attention_dp: true
attention_dp_config:
  batching_wait_iters: 0
  enable_balance: true
  timeout_iters: 60
cuda_graph_config:
  enable_padding: true
max_batch_size: 16
max_num_tokens: 8192
stream_interval: 10
EOF
```

Launch the OpenAI-compatible API server for an Instruct checkpoint:

```bash
trtllm-serve <YOUR_MODEL_DIR> \
  --backend pytorch \
  --host 0.0.0.0 \
  --port 8000 \
  --tp_size 8 \
  --ep_size 8 \
  --max_seq_len 4096 \
  --custom_tokenizer deepseek_v4 \
  --config ./deepseek_v4_serve.yml
```

The `/v1/chat/completions` API applies chat formatting on the server side, so clients should send
OpenAI-style `messages` rather than preformatted prompt strings. For Base checkpoints, use the same
command but remove `--custom_tokenizer deepseek_v4`. Increase `max_seq_len`, `max_batch_size`, and
the KV cache memory fraction after validating the memory budget for your target deployment.

### OpenAI-compatible request

For Instruct checkpoints, send a chat-completions request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_ID>",
    "messages": [
      {
        "role": "user",
        "content": "Write a short summary of TensorRT LLM."
      }
    ],
    "stream": true,
    "max_tokens": 128
  }'
```

For Base checkpoints, use the text completions API with a raw prompt:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_ID>",
    "prompt": "TensorRT LLM is",
    "stream": true,
    "max_tokens": 128
  }'
```


## Advanced Configuration

### Parallelism

DeepSeek-V4 supports the same main PyTorch backend parallelism knobs used by other large MoE models:

- Tensor parallelism (`--tp_size` or `tensor_parallel_size`) shards attention and dense weights.
- Pipeline parallelism (`--pp_size` or `pipeline_parallel_size`) distributes model layers across
  pipeline stages, which can help fit larger checkpoints or larger KV cache budgets across more
  GPUs.
- Expert parallelism (`--ep_size` or `moe_expert_parallel_size`) distributes routed experts.
- Attention DP (`enable_attention_dp: true`) keeps attention data-parallel across ranks and is
  commonly used for high-throughput, large-batch serving.

For latency-oriented tests, start without attention DP. For throughput-oriented tests, enable
attention DP in YAML:

```yaml
enable_attention_dp: true
attention_dp_config:
  batching_wait_iters: 0
  enable_balance: true
  timeout_iters: 60
```

When attention DP is enabled, remember that `max_batch_size` is local-rank batch size. Increase
`concurrency` and `num_requests` accordingly when benchmarking.

### Sparse attention

If `sparse_attention_config` is not provided, TensorRT LLM configures DeepSeek-V4 sparse attention
from the model config. It reads fields such as `compress_ratios`, `window_size` or `sliding_window`,
and indexer settings, then constructs the corresponding `DeepSeekV4SparseAttentionConfig`.

If `sparse_attention_config` is provided, user values override the corresponding sparse attention
settings, subject to the current implementation constraints: `window_size` must be `128`, and
`compress_ratios` must use supported ratios (`1`, `4`, or `128`). If checkpoint `compress_ratios`
are present and longer than the user-provided list, TensorRT LLM keeps the checkpoint list to avoid
silently changing the sparse attention layout.

Example YAML override:

```yaml
sparse_attention_config:
  algorithm: deepseek_v4
  window_size: 128
  index_topk: 512
```

### KV cache

DeepSeek-V4 uses `DeepseekV4CacheManager`, a `KvCacheManagerV2` subclass. This manager can describe
different cache layer types per model layer, so DeepSeek-V4 can map sliding-window, compressed,
indexer, and compressor-state caches according to the sparse attention layout from the model config
or user-provided `sparse_attention_config`.

DeepSeek-V4 KV cache requires:

- Hopper: `dtype=fp8_ds_mla` with `tokens_per_block` set to `128` or `256`.
- Blackwell: `tokens_per_block` set to `128` or `256`.
- `max_beam_width=1`.
- Hopper (`SM90`) or Blackwell (`SM100+`) GPUs.

Use a lower `free_gpu_memory_fraction`, `max_batch_size`, or `max_num_tokens` if the workload runs
out of memory during initialization or prefill.

### Quantized checkpoints

TensorRT LLM detects supported quantization metadata from the checkpoint directory, including
`hf_quant_config.json`, `quantization_config`, or `dtypes.json`. For DeepSeek-V4 checkpoints with
MXFP4 routed MoE expert weights, TensorRT LLM automatically applies the routed-expert quantization
configuration.

On Rubin (SM107), the MXFP4 routed experts can also run as NVFP4_PLUS (E2M1 values with one UE5M3
scale per 32 values and no global scale): the MXFP4 weights load unchanged and the expert
activations are quantized to FP4 as well. Set `OVERRIDE_QUANT_ALGO=NVFP4_PLUS` and use the CuteDSL
MoE backend (`moe_config.backend: CUTEDSL`, which `AUTO` selects under this override). See
[NVFP4_PLUS MoE from MXFP4 Checkpoints](../../../../docs/source/features/quantization.md#nvfp4_plus-moe-from-mxfp4-checkpoints).


## DSpark confidence scheduling (prototype)

DSpark always drafts a block of `K` tokens. With confidence scheduling on, a trained confidence head
scores each drafted position, and in decode-only steps the runtime may verify fewer than all
`G * (K + 1)` rows (`G` = requests per rank, `V` = verified rows). It picks `V` from a **cost table**
of measured step times, and only when the predicted accepted tokens per millisecond beat full
verification by `confidence_min_predicted_gain`. Target verification and the acceptance rule are
unchanged; mixed prefill/decode steps always verify every row. The feature is off by default.

### Enable it

```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [16, 32, 64, 128]        # rank-local G; the cost table must cover every entry
speculative_config:
  decoding_type: DSpark
  max_draft_len: 5
  enable_confidence_scheduling: true
  confidence_cost_table_path: /path/to/dspark-costs.json
  confidence_calibration_path: /path/to/calibration.json   # optional
  confidence_min_predicted_gain: 0.04                      # default
```

A static baseline can reuse the same file with `enable_confidence_scheduling: false`; the other
confidence fields are then ignored. `max_draft_len: 1` has no smaller tier and always runs static.

### Get a cost table

Step costs depend on the GPU, parallel layout and kernels, so generate the table on the system
and with the runtime YAML you serve with. One file can hold several `K`; the runtime uses the entry
for `max_draft_len`:

```bash
python -m tensorrt_llm.tools.dspark_profile_costs \
  --model <YOUR_MODEL_DIR> --config runtime.yaml --k 3 5 --output dspark-costs.json
```

The tool loads the model once per `K`. For every captured `G` it runs a short wave of requests
(another one while a budget has too few samples), cycles the full and four compact budgets through
steady decode steps, and records the median whole-step time (draft and verify included). Running it
again adds or replaces the profiled `K` and `G` rows in the same file, so `--k 4` or
`--graph-batch-sizes 512` extends an existing table. `--graph-batch-sizes`, `--tier-fractions` and
`--dataset` override the defaults; see `--help`.

At startup the runtime checks the table against the engine and fails with the command above when:

- the file is missing, or has no entry for `max_draft_len`;
- a captured `cuda_graph_config.batch_sizes` entry has no row (extra rows are ignored);
- it was measured on another GPU model, world size, TP/EP size or attention-DP setting.

A table from another TensorRT LLM version only warns. Tables in the earlier single-`K` format
(`schema_version: 2`) still load.

### Calibration

`confidence_calibration_path` points to per-position temperatures (`{"sts_temperatures": [...]}`).
A vector longer than `K` is truncated, so one file fitted at the checkpoint's block size serves
every `K`. Without it the raw sigmoid confidence is used.

### Device re-ranking

The host chooses the verifier budget from a copy of the previous step's confidence, so it never
waits for the GPU. The GPU then re-ranks which drafted positions to verify, using the confidence of
the block being verified. This is on by default with confidence scheduling; it costs compact steps
a full input rebuild and a small extra GPU pass. It needs the overlap scheduler, so with
`disable_overlap_scheduler: true` ranking stays on the host. To rank on the host only:

```yaml
speculative_config:
  enable_confidence_scheduling: true
  enable_fused_confidence_scheduler: false   # host ranking only
```

On DeepSeek-V4-Pro (8x B300) with attention DP, re-ranking raised throughput over host ranking in
13 of 15 dataset and batch-size settings, by up to 5.5%, without changing GSM8K accuracy; the larger
of its two losses (-2.2%) measured +0.4% when repeated on a later main. With TP8 attention and no
attention DP it raised throughput in 14 of 16 settings at batch sizes 8 to 64, by up to 6.4%.

### Limitations

Confidence scheduling requires the embedded DeepSeek-V4 DSpark drafter, padded CUDA graphs,
PP=1 and CP=1. Guided decoding, LoRA, per-request logits post-processors, per-layer dynamic EPLB,
heuristic sparse top-k, the acceptance-rate gate (`acceptance_rate_window_size` /
`acceptance_rate_threshold`) and non-greedy rejection draft sampling are rejected.

## Notes and Troubleshooting

- `DeepSeek-V4 on Hopper requires kv_cache_config.dtype='fp8_ds_mla'`: remove an explicit BF16/FP8
  KV cache override or set `dtype: fp8_ds_mla` in the YAML config.
- `DeepseekV4CacheManager requires tokens_per_block in [128, 256]`: pass
  `--tokens_per_block 128` in `quickstart_advanced.py` or set
  `kv_cache_config.tokens_per_block: 128` in YAML.
- Out-of-memory during initialization or prefill: reduce `max_batch_size`, `max_num_tokens`, or
  `kv_cache_config.free_gpu_memory_fraction`. For bring-up on 8xB200, set `max_seq_len` explicitly
  instead of using the checkpoint's 1M-token context length.
- Chat formatting issues with `trtllm-serve` or `trtllm-eval` on Instruct checkpoints: pass
  `--custom_tokenizer deepseek_v4`. Do not use this tokenizer wrapper for Base checkpoints.
- Tool-call chat formatting is not supported by the DeepSeek-V4 tokenizer wrapper yet.
