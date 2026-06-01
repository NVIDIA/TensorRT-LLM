# Sparse Attention

- [Overview](#overview)
- [Algorithms](#algorithms)
  - [RocketKV](#rocketkv)
  - [DeepSeek Sparse Attention (DSA)](#deepseek-sparse-attention-dsa)
  - [Skip Softmax Attention](#skip-softmax-attention)
- [Algorithm Comparison](#algorithm-comparison)
- [Further Reading](#further-reading)

## Overview

Sparse attention reduces the cost of long-context inference by skipping
work on KV entries that contribute little to the attention output. In
TensorRT LLM, sparse attention is enabled by passing a
`sparse_attention_config` object to the `LLM` API (or its YAML
equivalent for `trtllm-serve` / `trtllm-bench` / `trtllm-eval`). The
config object is a discriminated union — each algorithm has its own
subclass of `BaseSparseAttentionConfig` selected via the `algorithm`
field.

This page focuses on the **user-facing API**: how to construct and pass
the config for each supported algorithm, in both Python and YAML form.
For framework design details, see the [tech blog][tech-blog]. For
developers adding a new sparse attention algorithm, see the
[development guide](../developer-guide/sparse-attention-development.md).

[tech-blog]: ../blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md

## Algorithms

| `algorithm` | Config class | Reference |
|---|---|---|
| `rocket` | `RocketSparseAttentionConfig` | [RocketKV paper](https://arxiv.org/pdf/2502.14051) |
| `dsa` | `DeepSeekSparseAttentionConfig` | [DeepSeek V3.2 paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) |
| `skip_softmax` | `SkipSoftmaxAttentionConfig` | [BLASST paper](https://arxiv.org/pdf/2512.12087) |

For per-field semantics, refer to the docstring on each config class in
`tensorrt_llm/llmapi/llm_args.py`.

YAML configs shown below are consumed via the standard
`--extra_llm_api_options` / `--config` flag:

```bash
trtllm-serve --model <model_path> --config extra_config.yaml ...
trtllm-bench --model <model_path> --config extra_config.yaml ...
trtllm-eval --model <model_path> --config extra_config.yaml longbench_v2 --max_output_length 1024
```

### RocketKV

A training-free, two-stage algorithm: permanent KV cache eviction in
the context phase, followed by dynamic Top-K token selection in the
generation phase. Some framework-level algorithms (including RocketKV)
currently require disabling KV cache block reuse.

**Python API**

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import RocketSparseAttentionConfig, KvCacheConfig

sparse_attention_config = RocketSparseAttentionConfig(
    prompt_budget=2048,
    kt_cache_dtype="float8_e5m2",
)
kv_cache_config = KvCacheConfig(enable_block_reuse=False)

llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=sparse_attention_config,
    kv_cache_config=kv_cache_config,
)
outputs = llm.generate(["To be or not to be..."], SamplingParams(max_tokens=128))
```

**YAML**

```yaml
sparse_attention_config:
  algorithm: rocket
  prompt_budget: 2048
  kt_cache_dtype: float8_e5m2
kv_cache_config:
  enable_block_reuse: false
enable_chunked_prefill: false
```

### DeepSeek Sparse Attention (DSA)

A model-native sparse attention mechanism introduced with DeepSeek
V3.2: a lightweight learned indexer scores all KV entries and only the
top-`index_topk` are attended to.

**Python API**

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import DeepSeekSparseAttentionConfig

sparse_attention_config = DeepSeekSparseAttentionConfig(index_topk=64)

llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=sparse_attention_config,
)
```

**YAML**

```yaml
sparse_attention_config:
  algorithm: dsa
  index_topk: 64
```

**Optional: Guess-Verify-Refine Top-K.**
On Blackwell (SM 100+), set `enable_heuristic_topk=True` to use the
Guess-Verify-Refine (GVR) Top-K. GVR is currently supported only for
`index_topk=2048`; other values fall back to the production
insertion/radix Top-K. `TRTLLM_HEURISTIC_NMIN` overrides the
small-batch lower bound and `TRTLLM_SCHEMEX_DEBUG=1` prints the
dispatcher decision.

```python
sparse_attention_config = DeepSeekSparseAttentionConfig(
    index_topk=2048,
    enable_heuristic_topk=True,
)
```

```yaml
sparse_attention_config:
  algorithm: dsa
  index_topk: 2048
  enable_heuristic_topk: true
```

### Skip Softmax Attention

A kernel-level method (BLASST) that dynamically skips computation in a
FlashAttention-style kernel. It is a plug-and-play way to accelerate
existing full-attention models. For algorithm details and end-to-end
results, see [Tech Blog 16][blog16].

[blog16]: ../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md

The value actually consumed by the kernel is
**`threshold_scale_factor`** — the kernel combines it with the sequence
length to compute the skip threshold at runtime. Everything else is
just a way to produce that scalar.

Two configuration paths:

- **Set `threshold_scale_factor` directly.** The value flows straight
  to the kernel. Use this to turn on skip-softmax without a calibrated
  checkpoint ready.

- **Set `target_sparsity` ∈ [0, 1].** The runtime maps it to
  `threshold_scale_factor` via a calibration formula shipped with the
  checkpoint. The formula lives in `config.json` under
  `sparse_attention_config.threshold_scale_factor`:

  ```json
  {
    "sparse_attention_config": {
      "threshold_scale_factor": {
        "formula": "a * exp(b * target_sparsity)",
        "prefill": {"a": 100.0, "b": 5.0},
        "decode":  {"a": 0.05,  "b": 10.0}
      }
    }
  }
  ```

  - `formula` — an **arbitrary**
    [numexpr](https://numexpr.readthedocs.io/)-evaluable expression of
    `target_sparsity` and one or more named coefficients. Standard
    math functions (`exp`, `log`, `sqrt`, `pow`, `**`, …) are
    available. The runtime parses and evaluates it directly, so
    calibration is not locked to a fixed functional form.
  - `prefill` / `decode` — per-phase coefficient dictionaries; each
    must cover every name `formula` references (excluding
    `target_sparsity`).

  This calibration block will be supported by
  [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).
  If the checkpoint's `config.json` has no such
  `sparse_attention_config` block, `target_sparsity` is not usable and
  the runtime raises a clear error.

Both fields take either a scalar (applied to both phases) or a
`{"prefill": ..., "decode": ...}` dict. `threshold_scale_factor` and
`target_sparsity` are alternatives; if a config happens to carry both
(for example, a user override on top of a checkpoint default),
`threshold_scale_factor` wins and the calibration formula is ignored.

**Python API**

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import SkipSoftmaxAttentionConfig

# Direct threshold (single value applied to both phases):
sparse_attention_config = SkipSoftmaxAttentionConfig(threshold_scale_factor=1000.0)

# Direct threshold, per-phase:
sparse_attention_config = SkipSoftmaxAttentionConfig(
    threshold_scale_factor={"prefill": 1000.0, "decode": 500.0},
)

# Target sparsity (requires the checkpoint to carry a calibration formula):
sparse_attention_config = SkipSoftmaxAttentionConfig(target_sparsity=0.5)

# Target sparsity, per-phase:
sparse_attention_config = SkipSoftmaxAttentionConfig(
    target_sparsity={"prefill": 0.5, "decode": 0.3},
)

llm = LLM(model="<path_or_hf_id>", sparse_attention_config=sparse_attention_config)
```

Skip Softmax only works with the **TRTLLM** attention backend (the
default `attn_backend`). Other backends silently bypass skip-softmax.

**YAML**

```yaml
# Direct threshold:
sparse_attention_config:
  algorithm: skip_softmax
  threshold_scale_factor:
    prefill: 1000.0
    decode: 500.0
```

```yaml
# Target sparsity (requires a calibrated checkpoint):
sparse_attention_config:
  algorithm: skip_softmax
  target_sparsity:
    prefill: 0.5
    decode: 0.3
```


## Algorithm Comparison

| Aspect | RocketKV | DSA | Skip Softmax |
|---|---|---|---|
| Prefill acceleration | No | Yes | Yes |
| Decode acceleration | Yes | Yes | Yes |
| KV cache reduction | Yes | No | No |
| Framework-level support required | Yes | Yes | No |
| Model-native | No | Yes | No |

## Further Reading

- [Sparse Attention in TensorRT LLM (tech blog)][tech-blog] — framework
  design, per-algorithm implementation details, evaluation results.
- [Skip Softmax Attention tech blog](../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md)
  — algorithm details, kernel internals, end-to-end benchmarks.
- [Sparse Attention Development Guide](../developer-guide/sparse-attention-development.md)
  — how to add a new sparse attention algorithm (config class,
  prediction module, auxiliary memory, registration).
