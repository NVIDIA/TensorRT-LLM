# Sparse Attention

- [Overview](#overview)
- [Supported Algorithms](#supported-algorithms)
- [Sparse MQA/GQA Kernel Support](#sparse-mqagqa-kernel-support)
- [Configure Sparse Attention](#configure-sparse-attention)
- [Algorithm Details](#algorithm-details)
- [Further Reading](#further-reading)

## Overview

Sparse attention reduces long-context inference cost by avoiding attention work on
KV entries that an algorithm considers unimportant. TensorRT LLM separates two
parts of that process:

1. An algorithm selects tokens or blocks, or decides which kernel tiles can be
   skipped.
2. An attention implementation consumes that sparse pattern and computes the
   output.

This distinction matters for support. A kernel that can compute sparse MQA/GQA
does not by itself define how a model selects tokens, and therefore is not a
standalone user-facing algorithm.

The user-facing `sparse_attention_config` API is currently prototype and is
supported by the PyTorch execution backend. Each public algorithm has a config
class selected by its `algorithm` field. Model-native algorithms usually read
their geometry from the checkpoint; avoid overriding those values unless the
model-specific guide says they are tunable.

## Supported Algorithms

| `algorithm` | Config class | Sparse mechanism | Attention implementation | Typical use |
|---|---|---|---|---|
| `rocket` | `RocketSparseAttentionConfig` | Prompt KV eviction, then page-level Top-K selection during decode | TRTLLM or Vanilla | Training-free sparsity for standard attention models |
| `dsa` | `DeepSeekSparseAttentionConfig` | Learned token-level indexer followed by sparse MLA | TRTLLM | DeepSeek V3.2 and compatible model-native DSA architectures |
| `deepseek_v4` | `DeepSeekV4SparseAttentionConfig` | Sliding-window attention plus compressed sparse or compressed dense history | TRTLLM | DeepSeek-V4 hybrid attention |
| `minimax_m3` | `MiniMaxM3SparseAttentionConfig` | Learned block selection followed by sparse GQA | Dedicated Triton or MSA implementation | MiniMax-M3 sparse layers |
| `skip_softmax` | `SkipSoftmaxAttentionConfig` | Dynamically skips eligible softmax work inside the FMHA kernel | TRTLLM | Existing full-attention models with calibrated or direct thresholds |

All five configs select the PyTorch execution backend. The "attention
implementation" column refers to the attention kernel/backend used inside that
execution backend.

### Capability Comparison

| Capability | RocketKV | DSA | DeepSeek-V4 | MiniMax-M3 | Skip Softmax |
|---|---:|---:|---:|---:|---:|
| Sparse prefill computation | No | Yes | Yes | Yes | Yes |
| Sparse decode computation | Yes | Yes | Yes | Yes | Yes |
| Reduces retained main KV history | Yes | No | Yes, through model-native compression | No | No |
| Requires a model-trained selector | No | Yes | Yes | Yes | No |
| Selection granularity | Token eviction and pages | Tokens | Compressed entries | Blocks | Kernel tiles |

"No" for RocketKV prefill means that prompt attention is still computed
densely. RocketKV selects which prompt KV entries to retain, so it reduces cache
size and later decode work.

## Sparse MQA/GQA Kernel Support

TensorRT LLM contains an internal TRTLLM-Gen kernel for token-sparse
multi-query attention (MQA) and grouped-query attention (GQA). It accepts a
precomputed token-index list for each KV head and query token. Query heads in
the same KV group share the KV head's index list.

This is a kernel capability, not a public `SparseAttentionConfig` algorithm.
There is no supported `algorithm: mqa_gqa` value for `LLM` or YAML. A sparse
algorithm must provide the selector, metadata, cache management, and attention
backend integration before applications can use this kernel through the public
API.

### Support Matrix

| Parameter | Supported | Not currently supported or not established |
|---|---|---|
| GPU architecture | SM100 and SM103 | Pre-Blackwell GPUs; SM120 and SM121 |
| Attention type | MQA (`num_kv_heads == 1`) and GQA | Arbitrary head mappings |
| Head relationship | `num_q_heads % num_kv_heads == 0`; at most 32 query heads per KV head | Non-divisible Q/KV head counts; MQA/GQA groups larger than 32 |
| Q/K/V dimensions | Equal QK and V head dimensions | Unequal QK/V dimensions (MLA uses a separate sparse path) |
| Head dimension | `64`, `80`, `128`, or `256` | `512` and other head dimensions |
| Q/K/V and output dtype | BF16 or FP16 | Quantized Q/output combinations are not covered by this primitive's regression tests |
| KV cache | Paged KV cache, with the KV dtype matching the input dtype; page size is a power of two and at least 8 tokens | Contiguous KV cache; non-power-of-two pages; pages smaller than 8 tokens |
| Sparse indices | `int32`, token-granular, one list per KV head and query token | A public built-in selector for generic MQA/GQA |
| Sparse Top-K | Positive multiple of 4; shorter sequences may pad unused entries with `-1` | Top-K values not divisible by 4 |
| Inference phase | Fresh context and single-token generation | Mixed context/generation batches are not covered by the regression tests |
| Beam width | `1` | Beam search |
| Attention mask/window | Causal self-attention with a fixed cache window | ALiBi, arbitrary custom masks, StreamingLLM/sink tokens, and variable cyclic windows |

The current main branch JIT-compiles this path with NVRTC. Its support is
therefore defined by the current TRTLLM-Gen source checks, not by the set of
precompiled cubins that was present when the feature was introduced.

The regression tests cover:

- MQA and GQA ratios of 2:1, 4:1, and 8:1;
- the maximum supported query-head group size of 32 for both MQA and GQA;
- variable batch and sequence lengths;
- context KV compaction, context sparse computation, and decode sparse
  computation;
- Top-K values `4`, `64`, and `128`, including Top-K larger than a request's
  current KV length;
- backing KV-cache page sizes `32` and `64`;
- all supported equal head dimensions in both BF16 and FP16.

The shared TRTLLM-Gen option validator also admits head dimension `512`, but
the sparse MQA/GQA path aborts before launch for that configuration on current
main. It is therefore intentionally excluded from the supported matrix and
regression tests.

Backend developers can use
[`test_sparse_attention.py`](../../../tests/unittest/_torch/attention/sparse/test_sparse_attention.py)
as a minimal integration example. `MockSparseParams` and
`TestSparseAttention` deliberately supply fixed sparse predictions so that the
test isolates the cache/index layout and kernel computation. They are not
public application APIs.

## Configure Sparse Attention

Pass a config object to `LLM` in Python, or use the equivalent discriminated
YAML object with `trtllm-serve`, `trtllm-bench`, or `trtllm-eval`.

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import RocketSparseAttentionConfig

llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=RocketSparseAttentionConfig(),
)
```

```yaml
sparse_attention_config:
  algorithm: rocket
```

For example:

```bash
trtllm-serve <path_or_hf_id> --config config.yaml
trtllm-bench --model <path_or_hf_id> throughput --dataset <dataset> --config config.yaml
```

The following sections list algorithm-specific settings and constraints.

## Algorithm Details

### RocketKV

[RocketKV](https://arxiv.org/pdf/2502.14051) is a training-free, two-stage
algorithm for standard attention architectures. During prefill, it computes
dense attention and permanently evicts prompt KV entries beyond a prompt
budget. During decode, it scores retained pages and attends to the selected
Top-K pages.

RocketKV currently requires CUDA compute capability 10.0 or newer. KV-cache
block reuse and chunked prefill must be disabled, and disaggregated serving is
not supported.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, RocketSparseAttentionConfig

llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=RocketSparseAttentionConfig(
        prompt_budget=2048,
        kt_cache_dtype="float8_e5m2",
    ),
    kv_cache_config=KvCacheConfig(enable_block_reuse=False),
    enable_chunked_prefill=False,
)
outputs = llm.generate(
    ["To be or not to be..."],
    SamplingParams(max_tokens=128),
)
```

```yaml
sparse_attention_config:
  algorithm: rocket
  prompt_budget: 2048
  kt_cache_dtype: float8_e5m2
kv_cache_config:
  enable_block_reuse: false
enable_chunked_prefill: false
```

The TRTLLM and Vanilla attention implementations support RocketKV. The
Vanilla implementation requires a BF16 KT cache.

### DeepSeek Sparse Attention

DeepSeek Sparse Attention (DSA) is a model-native mechanism introduced by
DeepSeek V3.2. A learned MQA indexer scores the KV history, Top-K selects token
indices, and sparse MLA consumes them. Checkpoint fields define the indexer
head count, index head dimension, and Top-K; the safest configuration is to let
TensorRT LLM load them from the model.

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import DeepSeekSparseAttentionConfig

llm = LLM(
    model="deepseek-ai/DeepSeek-V3.2",
    sparse_attention_config=DeepSeekSparseAttentionConfig(),
)
```

```yaml
sparse_attention_config:
  algorithm: dsa
```

On supported Blackwell configurations, Guess-Verify-Refine (GVR) can replace
the regular decode Top-K dispatcher. The current implementation accepts
`index_topk` values `512`, `1024`, and `2048`, and indexer compression ratios
`1` and `4`. Unsupported combinations fall back to the production
insertion/radix Top-K path.

```yaml
sparse_attention_config:
  algorithm: dsa
  index_topk: 2048
  enable_heuristic_topk: true
```

See the
[DeepSeek V3/V3.2 example](../../../examples/models/core/deepseek_v3/README.md)
for model precision, hardware, parallelism, MTP, chunked-prefill, cache-reuse,
and disaggregated-serving support.

### DeepSeek-V4 Hybrid Sparse Attention

DeepSeek-V4 interleaves three model-native attention modes:

- sliding-window attention over recent raw tokens;
- compressed sparse attention over 4x-compressed history selected by an
  indexer;
- compressed dense attention over 128x-compressed history.

TensorRT LLM normally constructs `DeepSeekV4SparseAttentionConfig` from the
checkpoint. An explicit config overrides matching fields; it must preserve the
model's attention layout. The current implementation requires
`window_size=128`, compression ratios from `{1, 4, 128}`, data-center Blackwell
GPUs, KV-cache blocks of `128` or `256` tokens, and beam width `1`.

```yaml
sparse_attention_config:
  algorithm: deepseek_v4
  window_size: 128
  index_topk: 512
```

See the
[DeepSeek-V4 example](../../../examples/models/core/deepseek_v4/README.md) for
checkpoint-derived configuration and deployment constraints.

### MiniMax-M3 Block-Sparse GQA

MiniMax-M3 uses model-native block-sparse GQA in its sparse layers. An index
branch scores main KV-cache blocks, forces configured initial/local blocks into
the selection, and chooses the remaining Top-K blocks before sparse GQA.
Defaults such as four index heads, index dimension `128`, block size `128`, and
16 selected blocks come from the checkpoint-compatible config.

```yaml
sparse_attention_config:
  algorithm: minimax_m3
```

Two implementations are available:

- `triton` is the default reference implementation.
- `msa` uses `fmha_sm100` kernels and requires an SM100-family GPU (SM100 or
  SM103), the `fmha_sm100` package, and `sparse_block_size=128`.

```yaml
sparse_attention_config:
  algorithm: minimax_m3
  implementation: msa
```

The sparse path currently has no dense fallback and does not support KV-cache
reuse or MTP. See the
[MiniMax-M3 deployment guide](../deployment-guide/deployment-guide-for-minimax-m3-on-trtllm.md)
for supported checkpoints and parallel deployment settings.

### Skip Softmax Attention

Skip Softmax Attention, also known as BLASST, dynamically skips eligible work
inside a FlashAttention-style kernel. It does not select tokens, alter the
model architecture, or reduce KV-cache storage.

The kernel consumes `threshold_scale_factor` and combines it with sequence
length at runtime. You can provide that value directly:

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import SkipSoftmaxAttentionConfig

llm = LLM(
    model="<path_or_hf_id>",
    sparse_attention_config=SkipSoftmaxAttentionConfig(
        threshold_scale_factor={"prefill": 1000.0, "decode": 500.0},
    ),
)
```

```yaml
sparse_attention_config:
  algorithm: skip_softmax
  threshold_scale_factor:
    prefill: 1000.0
    decode: 500.0
```

Alternatively, provide `target_sparsity`. This path requires the checkpoint to
contain a calibration formula that maps the requested target to the kernel's
threshold scale factor.

```yaml
sparse_attention_config:
  algorithm: skip_softmax
  target_sparsity:
    prefill: 0.5
    decode: 0.3
```

Both fields accept a scalar for both phases or a dictionary with `prefill` and
`decode` values. If both are present, `threshold_scale_factor` takes
precedence. User-provided `target_sparsity` overrides a checkpoint default.

Model Optimizer can store calibration metadata in the checkpoint's
`config.json`:

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "algorithm": "skip_softmax",
        "threshold_scale_factor": {
          "formula": "a * exp(b * target_sparsity)",
          "prefill": {"a": 100.0, "b": 5.0},
          "decode": {"a": 0.05, "b": 10.0}
        },
        "target_sparsity": {"prefill": 0.5, "decode": 0.3},
        "ignore": ["model.layers.0.self_attn"]
      }
    }
  }
}
```

The formula is a [numexpr](https://numexpr.readthedocs.io/) expression over
`target_sparsity` and named coefficients. The optional `ignore` list uses
fnmatch layer patterns. At most one checkpoint config group may use the
`skip_softmax` algorithm.

Skip Softmax Attention requires the TRTLLM attention backend. Other attention
backends do not apply it.

## Further Reading

- [Sparse Attention in TensorRT LLM](../blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md)
  describes the framework and algorithm implementations.
- [Accelerating Long Context Inference with Skip Softmax Attention](../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md)
  covers BLASST kernel behavior and evaluation.
- [Sparse Attention Development Guide](../developer-guide/sparse-attention-development-guide.md)
  explains how to add an algorithm, selector, cache manager, and backend
  integration.
