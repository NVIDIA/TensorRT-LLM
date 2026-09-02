# Sparse Attention

- [Overview](#overview)
- [Supported Sparse Attentions](#supported-sparse-attentions)
  - [Sparse MLA](#sparse-mla)
  - [Sparse MQA/GQA](#sparse-mqagqa)
  - [Sparse MHA](#sparse-mha)
- [Supported Algorithms](#supported-algorithms)
  - [Capability Comparison](#capability-comparison)
  - [Algorithm Details](#algorithm-details)
- [Usage with trtllm-bench and trtllm-serve](#usage-with-trtllm-bench-and-trtllm-serve)
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

## Supported Sparse Attentions

TensorRT LLM supports sparse computation for MLA, MQA/GQA, and MHA. This
section describes the attention and kernel contracts independently of the
algorithm that produces the sparse pattern. The public algorithms that connect
selectors, cache management, and these attention implementations are listed in
[Supported Algorithms](#supported-algorithms).

### Sparse MLA

Sparse MLA consumes token-level selections against a latent KV cache. It is
used by model-native algorithms such as DeepSeek Sparse Attention and
DeepSeek-V4 hybrid attention. Both prefill and generation are supported,
including mixed batches.

| Parameter | Support |
|---|---|
| GPU architecture | SM90, SM100, SM103, SM120, and SM121 through architecture-specific sparse MLA implementations |
| Sparse compute phase | Packed prefill, generation, and mixed context/generation batches |
| Attention type | MLA with a shared latent KV representation |
| Model geometry | Checkpoint-native geometry; tests cover DeepSeek-V3.2 (`qk_head_dim=192`, `v_head_dim=128`) and DeepSeek-V4 (`qk_head_dim=512`, `v_head_dim=512`) |
| Model input dtype | BF16 |
| KV-cache dtype | BF16 and the FP8 modes supported by the selected model and GPU architecture |
| Sparse indices | `int32` token indices; one selection per query token |
| Attention semantics | Causal self-attention |

[`test_sparse_mla_forward.py`](../../../tests/unittest/_torch/attention/sparse/test_sparse_mla_forward.py)
covers pure prefill, pure generation, mixed batches, BF16/FP8 KV-cache modes,
and the direct FlashMLA sparse-forward contract.

### Sparse MQA/GQA

TensorRT LLM provides two sparse MQA/GQA compute paths. The token-sparse path
accepts a precomputed token list for each KV head and query token. Query heads
in the same KV group share the KV head's list. The block-sparse path accepts
request-local selections of 128-token KV blocks from a paged HND cache.

These are attention capabilities, not standalone public
`SparseAttentionConfig` algorithms. A user-facing algorithm must also provide
the selector, metadata, cache management, and backend integration.

| Parameter | Token-sparse | Block-sparse |
|---|---|---|
| Sparse block size | 1 token | 128 tokens |
| GPU architecture | SM100 and SM103 | SM100 and SM103 |
| Sparse compute phase | Packed prefill, single-token generation, and linear draft-token generation; query lengths `1` and `4` are tested | Packed prefill, single-token generation, linear multi-query compute, and mixed batches |
| Attention type | MQA and GQA; Q heads must be divisible by KV heads | MQA and GQA; Q heads must be divisible by KV heads |
| Query heads per KV head | At most 32; tests cover `2`, `3`, `4`, `8`, `16`, `24`, `31`, and `32` | `2`, `4`, `8`, or `16`; all are tested |
| Q/KV head counts | No additional discrete kernel limit; tests cover Q heads `{6, 8, 16, 32, 48, 62, 64}` and KV heads `{1, 2, 4, 8}` | No additional discrete kernel limit; tests cover Q heads `{4, 8, 16, 32}` and KV heads `{1, 2}` |
| Model Q/K/V input dtype | BF16 or FP16 | BF16 or E4M3 FP8 |
| Model Q/K/V input layout | Fused QKV | Q `[tokens, q_heads, 128]`; paged K/V `[pages, kv_heads, 128, 128]` |
| Output dtype | BF16 or FP16 for every supported head dimension; E4M3 FP8 for head dimensions `64`, `128`, and `256` | BF16 |
| KV-cache dtype | BF16 or FP16 for every supported head dimension; E4M3 FP8 for head dimensions `64`, `128`, and `256` | BF16 or E4M3 FP8 |
| Q/K/V head dimension | Equal dimensions of `64`, `80`, `128`, or `256` | Equal dimension of `128` |
| KV-cache layout | Paged cache; page size is a power of two and at least 8 tokens; tests cover `8`, `16`, `32`, `64`, `128`, `256`, and `512` | Paged HND cache with page size `128`; shuffled physical pages and strided outer-page storage are tested |
| Sparse indices | `int32` physical token indices per KV head and query token | `int32` request-local block indices per KV head and query token; per-token lists, `-1` padding, and physical remapping are tested |
| Sparse Top-K | Positive multiple of 4; tests cover `4`, `32`, `64`, and `128` | `4`, `8`, `16`, or `32` selected blocks; all are tested |
| Attention semantics | Causal self-attention | Causal self-attention with bottom-right or explicit per-request query offsets |

The token-sparse path is JIT-compiled with NVRTC. Its support is defined by the
current source checks rather than by the precompiled cubins that were present
when the feature was introduced. Linear draft-token generation is verified
with one target token and three draft tokens. Each query has its own causal
sparse list, including K/V written earlier in the same speculative forward.
Tree-shaped speculative masks are not applied by this path.

For an FP8 KV cache, token-sparse Q is quantized to E4M3 during QKV
preprocessing while the model input remains BF16 or FP16. Tests cover both
BF16 output with an FP8 KV cache and the E4M3 FP8-output kernel. The shared
kernel validator also admits head dimension `512`, but the sparse path aborts
before launch for that configuration, so it is excluded from this matrix.

Backend developers can use
[`test_sparse_mqa_gqa.py`](../../../tests/unittest/_torch/attention/sparse/test_sparse_mqa_gqa.py)
as an integration example. Its static selectors isolate index/cache layout and
attention computation without presenting a public application API.

### Sparse MHA

The shared page-sparse MHA path consumes block indices and per-request offsets
produced by a sparse selector. Sparse MHA computation starts during generation;
prefill attention computation remains dense. An algorithm can still compact
the retained KV cache after prefill to reduce cache size and later decode work.

#### Support Matrix

| Parameter | Support |
|---|---|
| GPU architecture | SM100 is runtime-tested; SM103 is source-supported and enabled by the tests |
| Sparse compute phase | Single-token and linear draft-token generation (`qSeqLen=4` is tested) |
| Attention type | MHA (`num_q_heads == num_kv_heads`) |
| Query heads per KV head | `1` |
| Number of MHA heads | No additional discrete source restriction beyond `num_q_heads == num_kv_heads > 0`; tests cover `1`, `2`, `3`, `4`, `8`, `16`, `24`, `32`, `48`, `64`, `96`, and `128` |
| Model Q/K/V input dtype | BF16 or FP16 |
| Model Q/K/V input layout | Fused QKV |
| Output dtype | Model dtype for head dimensions `64`, `80`, `128`, and `256`; E4M3 FP8 for head dimensions `64`, `128`, and `256` with an FP8 KV cache |
| KV-cache dtype | Model dtype for head dimensions `64`, `80`, `128`, and `256`; E4M3 FP8 for head dimensions `64`, `128`, and `256` |
| Q/K/V dimensions | Equal head dimensions of `64`, `80`, `128`, or `256` |
| KV-cache layout | Paged KV cache; page sizes `8`, `16`, `32`, `64`, `128`, `256`, and `512` are tested |
| Selection granularity | Block indices expanded to KV-cache pages |
| Sparse indices | `int32` block indices with `int32` per-request offsets; per-head patterns, unordered indices, and variable request offsets are tested |
| Sparse index block size | Blocks may cross KV-page boundaries; sizes `1`, `2`, `3`, `4`, `5`, `8`, `16`, `24`, `32`, and `48` are tested |
| Attention semantics | Causal self-attention |

Backend developers can use
[`test_sparse_mha.py`](../../../tests/unittest/_torch/attention/sparse/test_sparse_mha.py)
as an architecture-level integration example. It supplies static page
selections, invokes `TrtllmAttention.forward`, and compares the result with an
equivalent token-level PyTorch reference. RocketKV selector, metadata, and KT
cache tests remain under the `rocketkv/` subdirectory.

## Supported Algorithms

The public `sparse_attention_config` API connects a sparse algorithm to its
selector, runtime metadata, cache management, and attention implementation.

| `algorithm` | Config class | Sparse mechanism | Attention implementation | Typical use |
|---|---|---|---|---|
| `rocket` | `RocketSparseAttentionConfig` | Prompt KV eviction, then page-level Top-K selection during decode | TRTLLM or Vanilla | Training-free sparsity for MHA/MQA/GQA models |
| `dsa` | `DeepSeekSparseAttentionConfig` | Learned token-level indexer followed by sparse MLA | TRTLLM | DeepSeek-V3.2 and compatible model-native DSA architectures |
| `deepseek_v4` | `DeepSeekV4SparseAttentionConfig` | Sliding-window attention plus compressed sparse or compressed dense history | TRTLLM | DeepSeek-V4 hybrid attention |
| `minimax_m3` | `MiniMaxM3SparseAttentionConfig` | Learned block selection followed by sparse GQA | Dedicated Triton or packaged block-sparse implementation | MiniMax-M3 sparse layers |
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

### Algorithm Details

#### RocketKV

[RocketKV](https://arxiv.org/pdf/2502.14051) is a training-free, two-stage
algorithm for MHA, MQA, and GQA architectures. During prefill, it computes dense
attention and permanently evicts prompt KV entries beyond a prompt budget.
During decode, it scores retained pages and attends to the selected Top-K
pages.

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

#### DeepSeek Sparse Attention

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

#### DeepSeek-V4 Hybrid Sparse Attention

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

#### MiniMax-M3 Block-Sparse GQA

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

#### Skip Softmax Attention

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

## Usage with trtllm-bench and trtllm-serve

Sparse attention is configured through `sparse_attention_config` on the
PyTorch backend. DeepSeek-V3.2 provides a mature end-to-end example: its
checkpoint defines the DSA indexer geometry and Top-K, so the minimal YAML only
needs to select the `dsa` algorithm.

```yaml
# config.yml
sparse_attention_config:
  algorithm: dsa
```

Start an OpenAI-compatible server with the same config file used for other
PyTorch backend options:

```bash
trtllm-serve deepseek-ai/DeepSeek-V3.2 \
  --backend pytorch \
  --tp_size 8 \
  --ep_size 8 \
  --custom_tokenizer deepseek_v32 \
  --config ./config.yml
```

For a throughput benchmark, first prepare or supply a tokenized dataset, then
pass the same config to `trtllm-bench`:

```bash
trtllm-bench --model deepseek-ai/DeepSeek-V3.2 \
  prepare-dataset \
  --output ./deepseek-v3.2-dataset.json \
  token-norm-dist \
  --input-mean 4096 \
  --output-mean 512 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 16

trtllm-bench --model deepseek-ai/DeepSeek-V3.2 throughput \
  --backend pytorch \
  --tp 8 \
  --ep 8 \
  --dataset ./deepseek-v3.2-dataset.json \
  --max_batch_size 16 \
  --max_num_tokens 8192 \
  --config ./config.yml
```

Use a local checkpoint path in place of the Hugging Face model ID when needed.
Other sparse algorithms use the same YAML entry point with their own
`algorithm` discriminator and settings. See the
[DeepSeek V3/V3.2 example](../../../examples/models/core/deepseek_v3/README.md)
for model precision, hardware, parallelism, MTP, chunked-prefill, cache-reuse,
and disaggregated-serving configurations.

## Further Reading

- [Sparse Attention in TensorRT LLM](../blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.md)
  describes the framework and algorithm implementations.
- [Accelerating Long Context Inference with Skip Softmax Attention](../blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md)
  covers BLASST kernel behavior and evaluation.
- [Sparse Attention Development Guide](../developer-guide/sparse-attention-development-guide.md)
  explains how to add an algorithm, selector, cache manager, and backend
  integration.
