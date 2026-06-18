# Sparse Attention

- [Overview](#overview)
  - [Algorithms](#algorithms)
- [RocketKV](#rocketkv)
- [DeepSeek Sparse Attention (DSA)](#deepseek-sparse-attention-dsa)
- [Skip Softmax Attention](#skip-softmax-attention)
- [Algorithm Comparison](#algorithm-comparison)
- [Further Reading](#further-reading)

## Overview

Sparse attention reduces the cost of long-context inference by skipping work on KV entries that contribute little to the attention output. In TensorRT LLM, sparse attention is enabled by passing a `sparse_attention_config` object to the `LLM` API, or its YAML equivalent for `trtllm-serve`, `trtllm-bench`, or `trtllm-eval`. The config object is a discriminated union: each algorithm has its own subclass of `BaseSparseAttentionConfig` selected via the `algorithm` field.

This page focuses on the **user-facing API**: how to construct and pass the config for each supported algorithm, in both Python and YAML form. For framework design details, see [Blog 17: Sparse Attention in TensorRT-LLM](../blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md). For developers adding a new sparse attention algorithm, see the [Sparse Attention Development Guide](../developer-guide/sparse-attention-development-guide.md).

### Algorithms

| `algorithm` | Config class | Reference |
|---|---|---|
| `rocket` | `RocketSparseAttentionConfig` | [RocketKV paper](https://arxiv.org/pdf/2502.14051) |
| `dsa` | `DeepSeekSparseAttentionConfig` | [DeepSeek V3.2 paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) |
| `skip_softmax` | `SkipSoftmaxAttentionConfig` | [BLASST paper](https://arxiv.org/pdf/2512.12087) |

For per-field semantics, refer to the docstring on each config class in `tensorrt_llm/llmapi/llm_args.py`.

YAML configs shown below are consumed via the standard `--extra_llm_api_options` / `--config` flag:

```bash
trtllm-serve --model <model_path> --config extra_config.yaml ...
trtllm-bench --model <model_path> --config extra_config.yaml ...
trtllm-eval --model <model_path> --config extra_config.yaml longbench_v2 --max_output_length 1024
```

## RocketKV

RocketKV is a training-free, two-stage algorithm. It applies permanent KV cache eviction in the context phase, followed by dynamic Top-K token selection in the generation phase. Some framework-level algorithms, including RocketKV, currently require disabling KV cache block reuse.

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

## DeepSeek Sparse Attention (DSA)

DSA is a model-native sparse attention mechanism introduced with DeepSeek V3.2. A lightweight learned indexer scores all KV entries, and only the top-`index_topk` entries are attended to.

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

**Optional: Guess-Verify-Refine Top-K.** On Blackwell (SM 100+), set `enable_heuristic_topk=True` to use the Guess-Verify-Refine (GVR) Top-K. GVR is currently supported only for `index_topk=2048`; other values fall back to the production insertion/radix Top-K. `TRTLLM_HEURISTIC_NMIN` overrides the small-batch lower bound, and `TRTLLM_SCHEMEX_DEBUG=1` prints the dispatcher decision.

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

## Skip Softmax Attention

Skip Softmax Attention is a kernel-level method, also known as BLASST, that dynamically skips computation in a FlashAttention-style kernel. It can accelerate existing full-attention models without changing the model architecture.

The value actually consumed by the kernel is **`threshold_scale_factor`**. The kernel combines it with the **sequence length** to compute the **threshold** at runtime. Other configuration paths resolve to that scalar before the attention backend is constructed.

### Checkpoint Config

[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt) can perform calibration and store metadata for Skip Softmax Attention in the model checkpoint's `config.json`. The checkpoint config provides the formula that maps `target_sparsity` to `threshold_scale_factor`.

This checkpoint config is **optional**. It is only required when using `target_sparsity`, which is a [0, 1] scalar that is more intuitive than directly choosing the kernel-facing `threshold_scale_factor`. But please note that `target_sparsity` only serves as a guidance, the actual **achieved** sparsity in the kernel would vary.

Example checkpoint config:

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "algorithm": "skip_softmax",
        "threshold_scale_factor": {
          "formula": "a * exp(b * target_sparsity)",
          "prefill": {"a": 100.0, "b": 5.0},
          "decode":  {"a": 0.05,  "b": 10.0}
        },
        "target_sparsity": {
          "prefill": 0.5,
          "decode": 0.3
        },
        "ignore": [
          "model.layers.0.self_attn",
          "model.layers.1.self_attn"
        ]
      }
    }
  }
}
```

The checkpoint config may contain multiple `config_groups` for different sparse attention algorithms. At most one group may configure Skip Softmax Attention. Multiple groups whose `algorithm` is `skip_softmax` are invalid.

- `formula` — an **arbitrary** [numexpr](https://numexpr.readthedocs.io/) expression of `threshold_scale_factor` using `target_sparsity` and one or more named coefficients. Standard math functions such as `exp`, `log`, `sqrt`, `pow`, and `**` are available. The runtime parses and evaluates it directly, so calibration is not locked to a fixed functional form. It can be configured separately for prefill and decode; otherwise both phases use the same config.
- `target_sparsity` — optional checkpoint-provided target values. It can be configured separately for prefill and decode; otherwise both phases use the same config.
- `ignore` — optional fnmatch layer patterns where the calibrated Skip Softmax Attention config should not apply.

### User Configuration

User configuration is supplied through Python or YAML and controls how the checkpoint metadata is consumed:

- Set `threshold_scale_factor` directly to pass a concrete threshold to the kernel. This does not require checkpoint config.
- Set `target_sparsity` to request a sparsity target. The runtime resolves it to `threshold_scale_factor` using the checkpoint calibration formula. If the checkpoint does not provide the required Skip Softmax Attention metadata, the runtime raises an error.

Both `threshold_scale_factor` and `target_sparsity` take either a scalar, applied to both prefill and decode, or a `{"prefill": ..., "decode": ...}` dict. `threshold_scale_factor` and `target_sparsity` are alternatives: if both are present, `threshold_scale_factor` takes precedence and the calibration formula is not used. User-provided `target_sparsity` overrides checkpoint-default `target_sparsity`. Checkpoint `ignore` patterns always disable Skip Softmax Attention for matching layers.

#### Python API

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

Skip Softmax Attention only works with the **TRTLLM** attention backend, which is the default attention backend. Other backends silently bypass Skip Softmax Attention.

#### YAML

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

| Aspect | RocketKV | DSA | Skip Softmax Attention |
|---|---|---|---|
| Prefill acceleration | No | Yes | Yes |
| Decode acceleration | Yes | Yes | Yes |
| KV cache reduction | Yes | No | No |
| Framework-level support required | Yes | Yes | No |
| Model-native | No | Yes | No |

## Further Reading

- [Blog 17: Sparse Attention in TensorRT-LLM](../blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md) — framework design, per-algorithm implementation details, evaluation results.
- [Blog 16: Accelerating Long Context Inference with Skip Softmax Attention](../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md) — algorithm details, kernel internals, end-to-end benchmarks.
- [Sparse Attention Development Guide](../developer-guide/sparse-attention-development-guide.md) — how to add a new sparse attention algorithm, including config classes, prediction modules, auxiliary memory, and registration.
