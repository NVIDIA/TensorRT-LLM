<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->
<!-- Copyright (c) 2026 by FlashInfer team. -->

# Task-Scheduled MLA Decode

This directory contains the CuTe DSL task-scheduled (TS) Multi-head Latent
Attention (MLA) decode kernels used by FlashInfer's experimental Blackwell
paged-cache APIs. The implementation accepts the post-matrix-absorption MLA
layout: every query/cache row contains a 512-element latent component followed
by a 64-element RoPE component, while output contains the 512 latent values.

FlashInfer selects the 1-CTA throughput/latency family or the 2-CTA throughput
family automatically. Query grouping, persistent scheduling, split-KV, and
local versus separate reduction are implementation decisions. Unsupported
shape, dtype, or mask combinations raise an error rather than falling back to
another backend.

When persistent dispatch is selected, cluster launch control (CLC) assigns
work to resident CTAs.

## Public APIs

Import these entry points from `tensorrt_llm._torch.attention_backend.prims_ts`:

| API | Use |
| --- | --- |
| `BatchMLADecodePagedTSWrapper` | Reusable `plan()`/`run()` interface with owned scratch. |
| `batch_decode_mla_with_paged_kv_cache` | One-shot convenience interface. |
| `get_prims_ts_batch_decode_mla_workspace_size` | Size caller-owned standalone scratch. |
| `prims_ts_batch_decode_with_kv_cache_mla` | Standalone launch with caller-owned scratch. |

FlashInfer's optional `fi_trace` integration is not vendored into TensorRT-LLM.
The unbound `wrapper.run.fi_trace(...)` form is rejected because it cannot
carry the wrapper's plan-owned fixed-versus-packed query mode.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Query/cache dimensions | `kv_lora_rank=512`, `qk_rope_head_dim=64` (576 input elements) |
| Output dimension | 512 |
| Q/cache dtype | Matching `torch.bfloat16` or `torch.float8_e4m3fn` |
| Output dtype | `torch.bfloat16` only |
| Q length | Fixed or packed variable length; static maximum must be positive |
| K/V length | Positive and at most `2**31 - 32768`; the reserve keeps the largest padded split-KV coordinate span in signed `int32` |
| Q heads | Validated at 8, 16, 32, 64, and 128; other positive counts are accepted only when automatic selection reports an implementation |
| K/V cache | Paged, one logical KV head; `[num_pages, page_size, 576]` or `[num_pages, 1, page_size, 576]` compact storage |
| Metadata/cache index extents | Flattened query-head capacity, block-table elements, and physical page count must fit signed `int32` |
| Page size | 16, 32, 64, or 128 tokens |
| Mask | Dense or bottom-right causal |
| Scheduling | Automatic nonpersistent, static-persistent, or CLC-persistent selection by kernel family and work shape |
| Accumulation | FP32 QK/PV and softmax state |

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but remains to be qualified.

Some head and Q-length combinations outside the validated power-of-two matrix
do not have a generated implementation and are rejected.

Query, cache, and output tensors must be compact, 16-byte-aligned CUDA tensors
on the metadata device. `block_tables` and `seq_lens` are compact,
16-byte-aligned CUDA `torch.int32`; packed `qo_indptr` is contiguous CUDA
`torch.int32`. A caller-provided `out` must not overlap query, cache,
`block_tables`, `seq_lens`, packed `qo_indptr`, or caller-owned workspace.
The launch conservatively rejects overlapping storage spans. The API returns O
only. FP32 LSE is internal workspace and is not exposed as an output.

## Tensor and metadata layouts

- Fixed query: `[B, SQ, H, 576]`.
- Packed query: `[total_q, H, 576]`, with contiguous CUDA `int32[B + 1]`
  `qo_indptr`.
- Fixed output: `[B, SQ, H, 512]`.
- Packed output: `[total_q, H, 512]`.
- Cache storage has one logical KV head and may omit or retain that singleton
  axis: `[num_pages, page_size, 576]` or
  `[num_pages, 1, page_size, 576]`. "Compact" means adjacent elements follow
  those row-major shapes without padding in the inner page, token, or feature
  dimensions.
- Dense page table: contiguous CUDA `int32[B, max_num_pages]`
  `block_tables`.
- Runtime lengths: contiguous CUDA `int32[B]` `seq_lens`. Every length is
  positive and no larger than the static K/V bound.

For causal request `b`, query row `i` can attend through
`seq_lens[b] - query_length[b] + i`. `bmm1_scale` and `bmm2_scale` default to
1 and must be finite, positive Python scalars representable as positive
`float32` values.

Each `block_tables` row must contain at least
`ceil(max_kv_len / page_size)` columns, and every page ID used by a runtime
length must index the physical cache. Packed offsets start at zero, increase
strictly, end at `total_q`, and have every per-request delta no greater than
`max_seq_len_q`.

The wrapper retains `block_tables`, `seq_lens`, and packed `qo_indptr` as live
device inputs. Their storage must remain valid. Values may be changed in-place
only while page IDs remain valid, K/V lengths stay positive and within the
planned bound, packed-Q deltas stay positive and within their bound, and the
final packed offset remains equal to the planned query/output extent. Causal
metadata must also preserve `q_len[b] <= seq_lens[b]` for every request.
For a packed wrapper plan, omitting `max_seq_len_q` makes `plan()` read the
offsets once and use their largest delta as the bound. Every wrapper plan also
reads `seq_lens` once, rejects nonpositive rows, and checks every row against
the K/V bound. The standalone packed API requires an explicit bound and trusts
the device-side values on each launch. Wrapper and standalone hot paths do not
synchronize live device metadata to the host or fully value- and bounds-check
it. Invalid live page IDs, lengths, or packed offsets violate the contract and
may cause incorrect results or out-of-bounds access. A wrapper owns mutable
scratch and supports only one in-flight run or captured-graph replay; use
separate wrapper instances for concurrent execution.

## Dataflow and source map

```text
Q(latent | RoPE) + paged cache(latent | RoPE)
    -> staged Q/K/V + page IDs
    -> QK MMA -> masked online softmax -> P
    -> PV MMA -> corrected latent O + internal log-normalizer state
    -> direct O, local split reduction, or partials -> reducer -> O
```

The 1-CTA throughput/latency and 2-CTA throughput families select their launch
and reduction topology automatically. CLC-persistent scheduling is used when
the logical work benefits from reusing resident CTAs; callers do not select a
scheduler or kernel family through the public wrappers.

The BF16 2-CTA path enables CLC only when logical work exceeds one resident
wave. Inactive Q groups, pruned split slots, and zero-visible-K tiles skip
their Q/K/V and TMEM data work and skip the active-tile throttle edge
symmetrically. Every participating task still advances the work queue, so all
tasks retire the same work-tile sequence. This matched progression is the
synchronization contract for CLC replay and runtime pruning.

For BF16 2-CTA paged K/V, the TMA warp holds one 32-page-ID register window.
It refreshes the window after `32 / pages_per_k_tile` logical K tiles, when the
next 32 page-table entries are needed. This period follows directly from page
and tile geometry and is internal to the task schedule.

| Source | Responsibility |
| --- | --- |
| [`../../mla_decode.py`](../../mla_decode.py) | Public validation, automatic family selection, workspace binding, JIT caching, and launch adaptation |
| [`kernel_policy.py`](kernel_policy.py) | Eligibility and automatic kernel-family selection |
| [`throughput_latency_1cta/`](throughput_latency_1cta/) | Latency-oriented 1-CTA configs, resources, tasks, kernels, and reducers |
| [`throughput_2cta/`](throughput_2cta/) | M128 2-CTA configs, resources, tasks, kernels, and reducers |
| [`parallel_reduction_topology.py`](parallel_reduction_topology.py) | Shared split-reduction topology decisions |
| [`helpers/`](helpers/) | Layout, mask, schedule, math, and tile helpers |

## Example

```python
import torch
from tensorrt_llm._torch.attention_backend.prims_ts import BatchMLADecodePagedTSWrapper

device = "cuda"
B, H = 2, 16
latent_dim, rope_dim = 512, 64
page_size, pages_per_request = 32, 4
num_pages = B * pages_per_request

query = torch.randn(
    B, 1, H, latent_dim + rope_dim,
    device=device,
    dtype=torch.bfloat16,
)
kv_cache = torch.randn(
    num_pages, page_size, latent_dim + rope_dim,
    device=device,
    dtype=torch.bfloat16,
)
block_tables = torch.arange(
    num_pages, device=device, dtype=torch.int32
).view(B, pages_per_request)
seq_lens = torch.full(
    (B,), pages_per_request * page_size, device=device, dtype=torch.int32
)

wrapper = BatchMLADecodePagedTSWrapper()
wrapper.plan(
    block_tables,
    seq_lens,
    H,
    latent_dim,
    rope_dim,
    page_size,
    seq_len_q=1,
    q_data_type=query.dtype,
    kv_data_type=kv_cache.dtype,
    o_data_type=torch.bfloat16,
    mask_type="causal",
    max_kv_len=pages_per_request * page_size,
)
out = wrapper.run(query, kv_cache)
assert out.shape == (B, 1, H, latent_dim)

# Packed Q uses cumulative per-request offsets and compact token-major rows.
q_lens = (1, 3)
qo_indptr = torch.tensor((0, 1, 4), device=device, dtype=torch.int32)
packed_query = torch.randn(
    sum(q_lens), H, latent_dim + rope_dim,
    device=device,
    dtype=torch.bfloat16,
)
packed_wrapper = BatchMLADecodePagedTSWrapper()
packed_wrapper.plan(
    block_tables,
    seq_lens,
    H,
    latent_dim,
    rope_dim,
    page_size,
    qo_indptr=qo_indptr,
    q_data_type=packed_query.dtype,
    kv_data_type=kv_cache.dtype,
    o_data_type=torch.bfloat16,
    mask_type="causal",
    max_kv_len=pages_per_request * page_size,
)
packed_out = packed_wrapper.run(packed_query, kv_cache)
assert packed_out.shape == (sum(q_lens), H, latent_dim)
```

For the standalone API, call
`get_prims_ts_batch_decode_mla_workspace_size()` with the same shape, dtype,
mask, and Q-bound arguments as the launch. Allocate at least that many bytes as
a contiguous, 32-byte-aligned CUDA `torch.int8` or `torch.uint8` tensor. The
buffer includes internal FP32 LSE storage, does not require initialization,
and is exclusive to one in-flight launch or captured graph. It must not overlap
query, K/V cache, metadata, or output storage. The standalone hot path does not
copy metadata to the host; callers must maintain all page, length, and
packed-offset preconditions for every launch. These live values are not fully
value-checked by the launch.

For CUDA graph capture, compile and warm the planned configuration first,
retain metadata and workspace storage at stable addresses, and provide a
compact, 16-byte-aligned `out` tensor to avoid allocation.

## Limitations

- The latent/RoPE dimensions are fixed at 512/64.
- Output is `torch.bfloat16` even when query/cache input is
  `torch.float8_e4m3fn`.
- Only dense paged MLA cache addressing is exposed; sparse/indexed MLA,
  sliding windows, attention sinks, and custom masks are not public features.
- Query and cache dtypes must match.
- Automatic kernel selection is intentionally not user-selectable through the
  public wrapper.

## Validation

### TensorRT-LLM checkout

Run the local adapter and GPU backend unit coverage from the TensorRT-LLM
repository root:

```bash
pytest -q tests/unittest/_torch/attention/test_prims_ts_fmha.py -k mla
pytest -q tests/unittest/_torch/attention/test_prims_ts_attention_backend.py -k deepseek_v3_lite_mla_generation
```

The model-level MLA decode end-to-end coverage requires a Blackwell GPU and the
model weights under `LLM_MODELS_ROOT`:

```bash
LLM_MODELS_ROOT=/path/to/models pytest -q \
  tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::test_prims_ts_bfloat16
```

### Upstream FlashInfer checkout

The upstream public accuracy suite covers fixed and packed Q,
`torch.bfloat16` and `torch.float8_e4m3fn` input, dense and causal masks, all
four page sizes, both automatic kernel families, runtime K pruning, split-KV
reduction, output/workspace contracts, and CUDA graphs. These paths do not
exist in TensorRT-LLM; run them only from the FlashInfer checkout pinned by the
top-level provenance section:

```bash
pytest -q tests/attention/test_attention_ts_mla_decode.py
pytest -q tests/trace/test_fi_trace_template_consistency.py
```
