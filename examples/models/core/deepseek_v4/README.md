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


## DSpark confidence verification (prototype)

Confidence-guided verification is **default-off and experimental**. It proposes the full physical
draft block `K`, then may select a smaller, measured verifier-token budget. It does not reduce the
physical drafter length. This draft integration is not a production deployment recipe: receipt
generation and workload qualification are external, and the composed source still needs native
build, GPU/full-model, accuracy and performance validation. Optional mixed scheduling has the
separate evidence and route requirements below; no composed-tip GPU qualification is implied.

### Requested configuration versus actual execution

`DSparkDecodingConfig.enable_confidence_scheduling=True` is a request, not evidence that the
runtime enabled it. For `K > 1`, `TorchLlmArgs` checks workload admission before constructing
engines and workers:

| Startup result | Resolved execution |
| --- | --- |
| Missing, unreadable, stale or non-positive admission evidence | Both confidence flags become false and all confidence artifact paths/digests are cleared; ordinary feature-off static `K` runs. A warning explains the fallback. |
| Physical `K=1` | Feature-off static `K1`; there is no smaller verifier tier. Confidence inputs are ignored and cleared. |
| Matching receipt with positive aggregate value | Confidence remains requested; the existing environment, graph, checkpoint and runtime checks must still pass. |

After successful admission, an **enabled-native** policy decision is different from startup
feature-off fallback: it retains the enabled control path while verifying the native layout.
An enabled compact decision uses a positive measured budget. Optional fused-scheduler failure
falls back to the tensor scheduler for the affected graph size; that is a third, separate fallback.

Before reporting an "enabled" experiment, inspect the resolved
`TorchLlmArgs.speculative_config.enable_confidence_scheduling` and record both the requested and
resolved configuration. Require observed native/compact routing and completed outputs, rather
than inferring execution from an input YAML flag or a successful server start. Do not count
startup feature-off fallback as an enabled-native run. With confidence disabled, omit all
confidence-only fields; explicitly supplying them is a configuration error.

### Admission evidence contract

For a deployment requesting confidence, provide all four prototype inputs:

- `confidence_sps_table_path`: schema-v2 measured costs with exact physical `K`, rank-local graph
  sizes `G`, native `V=0` cells and supported positive verifier budgets.
- `confidence_sps_live_fingerprint_path`: an independently supplied fingerprint for that runtime
  and graph ladder, not a copy relabeled from an older cost artifact.
- `confidence_admission_receipt_path`: the schema-v1 workload receipt.
- `confidence_admission_receipt_sha256`: the expected lowercase SHA256 of the **receipt file
  bytes**, obtained from a trusted, separately reviewed deployment package.

There is no supported public receipt-generation command in this prototype. Do not manufacture an
admitting receipt from the unit-test fixtures. Its producer must preserve the measured cost,
workload, calibration, selector-replay and runtime evidence and review their applicability to the
actual deployment. A source, calibration, physical-`K` or workload change requires reassessment;
historical scores or timing results cannot be relabeled as new-source qualification.

The receipt envelope contains exactly `schema_version: 1`, `admission` and `admission_sha256`.
The last field hashes the canonical admission-body JSON
(`json.dumps(body, sort_keys=True, separators=(",", ":")).encode()`).
The body contains:

- `physical_k`, `source_head`, `source_diff_sha256`, `runtime_snapshot`;
- `sps_cost_table_sha256`, `live_engine_fingerprint_sha256`, `selector_identity_sha256`;
- `workload_identity_sha256`, `calibration_result_sha256`, `selector_replay_sha256`;
- `policy_steps`, `compact_choices`, `gross_compact_value_ms_lower_bound`,
  `fixed_confidence_path_overhead_ms_upper_bound`, `safety_margin_ms`, `admitted`.

The selector identity must match `EXACT_SPS_SELECTOR_IDENTITY_SHA256` in
`tensorrt_llm/_torch/speculative/dspark_planner.py`. Replay the existing marginal selector without
allocating fixed overhead back to candidates through a `policy_steps / compact_choices` ratio.
Admission recomputes:

```text
net_value_lower_bound_ms =
    gross_compact_value_ms_lower_bound
    - policy_steps * fixed_confidence_path_overhead_ms_upper_bound
    - safety_margin_ms

admitted = compact_choices > 0 and net_value_lower_bound_ms > 0
```

Native decisions still pay the enabled confidence-path overhead, so **all** policy steps count.
Admission does not replace per-decision selection: exact measured tiers and the existing
iteration/drain guard still determine whether a compact choice is allowed.

Hashes provide integrity and cross-artifact consistency checks, **not signatures or proof of
origin**. The runtime checks supplied receipt, cost and fingerprint values; it does not independently
derive the running checkpoint/workload identity, re-execute the calibration or selector replay,
or verify their referenced result files. The external producer/deployment package remains the
trust boundary. Hashing an arbitrary receipt and passing that same digest does not establish
trust or qualify an enabled experiment.

Contract regressions are in `test_dspark_workload_admission.py`,
`test_dspark_admission_resolution.py` (both under
`tests/unittest/_torch/speculative/hw_agnostic/`) and
`tests/unittest/llmapi/test_dspark_workload_admission_config.py`. These fixtures are synthetic
validator tests, not deployment evidence.

### Optional mixed context/generation scheduling

`confidence_mixed_evidence` is an optional nested object. Its default is `None`, which leaves
mixed steps at native verification width and does not change admitted decode-only scheduling.
Mixed costs are not decode `T(G,V)` costs: context chunks and their positions, generation history,
bootstrap requests, ordered ranks and the shared body capacity all affect the mixed cost key.

The object contains `costs_path`, `costs_sha256`, `admission_path`, `admission_sha256`,
`protocol_sha256`, `live_fingerprint_path` and `live_fingerprint_sha256`. Cost, receipt and live
fingerprint digests pin exact file bytes; the protocol digest identifies the measured protocol.
These pins come from a trusted deployment package. The live fingerprint must be observed independently
for the current source, runtime, model, GPU class, topology and selector, not copied from the cost
table. The cost file uses `dspark-mixed-breakable-costs-v1`; its separate receipt uses
`dspark-mixed-breakable-admission-v1`. Decode schema-v2 inputs and admission remain unchanged.

The supported mixed route is embedded DSpark with physical `K=3`, TP8, PP1, CP1, attention DP,
greedy sampling and captured BREAKABLE body buckets `[96, 128, 192, 256, 384, 512]`. Standalone
draft models, unsupported routes and uncovered mixed geometry retain native mixed verification.
Missing or invalid mixed evidence does not disable an already admitted decode policy.
First context chunks also remain native because KV preparation may still resolve a prefix-cache
hit and change their geometry. Later chunks are eligible only when their full-width native rows
fit the configured token budget, so resource preparation does not trim a quoted context shape.

When a peer rank prefills, all ranks use the existing shared BREAKABLE padding path rather than
replaying a rank-local full decode graph. Only authenticated generation owners may receive shorter
verification prefixes; context rows, bootstrap requests and graph-padding requests retain their
required rows. The runner publishes those lengths as host-authoritative before attention metadata
preparation, without changing the attention backend, sampler or acceptance rule. Stale or unready
confidence owners retain full physical `K`.

Selection requires an exact measured geometry/action cell and retains the minimum 1% predicted
goodput gate. The current pricing field is `rank0_forward_pricing_proxy`, not a measurement of the
whole distributed step. Workload admission must charge the opt-in path's overhead on **all** policy
steps, including all-decode and native choices: the optional path adds CPU curve preparation,
producer-stamp transfer and a fixed trailer to the existing policy allgather. Policy equivalence
alone is not evidence of zero overhead or preserved throughput.

There is no public mixed receipt-generation command in this change. Supplying valid-shaped JSON
or hashing test fixtures does not qualify a deployment. A separately reviewed producer must supply
matching measured costs, runtime/accuracy results, selector replay and positive workload economics.
Exact-tip GPU/full-model validation and production mixed evidence remain required before making a
deployment or performance claim. The CPU contracts are in `test_dspark_mixed_evidence.py` and
`test_dspark_mixed_runtime.py` under `tests/unittest/_torch/speculative/hw_agnostic/`.

### Drafter fusion scope

The BF16 CuTe attention and RMSNorm–RoPE path is eligible for physical `K=1..8` under the existing
SM100/SM103, geometry and layout guards; unsupported inputs retain the reference path. This
dispatch is independent of the confidence flag, so it also affects static DSpark. The eight-row
storage bound is unchanged; no arbitrary-`K` kernel extension is claimed.

The source-matched extension passed 98 GPU unit cases on B300/SM103. That is not full-model or
performance qualification of this composed integration. SM100 and exact-integration GPU/model
validation remain pending; keep this scope experimental until those gates and review complete.

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
