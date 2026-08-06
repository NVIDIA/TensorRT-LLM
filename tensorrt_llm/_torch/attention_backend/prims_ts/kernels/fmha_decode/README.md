<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->
<!-- Copyright (c) 2026 by FlashInfer team. -->

# Task-Scheduled FMHA Decode

This directory contains the CuTe DSL task-scheduled (TS) FMHA kernel used by
FlashInfer's experimental paged decode APIs on NVIDIA Blackwell GPUs. It
supports token-at-a-time decode, small fixed speculative-query batches, and
packed variable-length queries over a paged K/V cache.

The public API describes attention semantics and cache metadata. Tile shapes
and launch policy are selected internally for the problem and GPU. Fixed-Q
plans may use direct, persistent, or split-KV execution. Packed-Q and
sliding-window plans remain nonsplit, but may use direct or CLC-persistent
execution. There is no public scheduler or tuning knob and no fallback to
another attention backend.

For eligible nonsplit grids with more than one resident wave, cluster launch
control (CLC) assigns work to resident CTAs. Underfilled fixed-Q grids may
instead split the K/V sequence and reduce partial outputs; other grids use the
direct static launch.

## Public APIs

Import these entry points from `tensorrt_llm._torch.attention_backend.prims_ts`:

| API | Use |
| --- | --- |
| `BatchDecodePagedTSWrapper` | Reusable `plan()`/`run()` interface; owns compiled callables and scratch. |
| `batch_decode_with_paged_kv_cache` | One-shot convenience interface. |
| `get_prims_ts_batch_decode_workspace_size` | Size caller-owned scratch for the standalone launch. |
| `prims_ts_batch_decode_with_kv_cache` | Standalone launch with caller-owned scratch and explicit `seq_lens`. |

FlashInfer's optional `fi_trace` integration is not vendored into TensorRT-LLM.
The unbound `wrapper.run.fi_trace(...)` form is rejected because it cannot
carry the wrapper's plan-owned query mode and output dtype.

Prefer the reusable wrapper when a cache geometry is used repeatedly. Planning
always snapshots and host-validates the derived K/V lengths once; `run()`
performs no device-to-host metadata read. An explicit `max_kv_len` is a bound
that must cover every planned row. Fixed-length scheduling is selected
automatically only when every row equals that bound; it is not a public knob.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Head dimension | 64, 128, or 256 |
| Fixed Q length | Any positive integer representable by the metadata and tensor extents |
| Packed Q | Positive per-request lengths no greater than a positive static maximum |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv` and `1 <= Hq/Hkv <= 32` |
| Q/K/V dtype | Q and K/V must match: `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Output dtype | `torch.float16` for `torch.float16` input; `torch.bfloat16` for `torch.bfloat16` input; `torch.float16` or `torch.float8_e4m3fn` for `torch.float8_e4m3fn` input |
| K/V layout | HND paged cache, combined or separate K/V tensors |
| Page size | 16, 32, 64, or 128 tokens |
| Maximum K/V length | No fixed model limit; the static bound and metadata must fit signed `int32` |
| Mask | Dense or bottom-right causal |
| Sliding window | Causal left window; `window_left=-1` disables it and non-negative values include the current token |
| Scheduling | Automatic direct or CLC-persistent launch; eligible underfilled fixed-Q grids may use split-KV. Packed-Q and sliding-window grids remain nonsplit. No public tuning knob. |
| Accumulation | FP32 QK/PV and softmax state |

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but remains to be qualified.

The public paths require compact, 16-byte-aligned Q and output storage. K/V
pages must have compact HND inner strides; a padded outer page stride is
allowed when pages do not overlap and both the tensor base and outer stride
are 16-byte aligned. All query, cache, metadata, output, and workspace tensors
must be on one CUDA device. Metadata is contiguous CUDA `torch.int32` with
4-byte alignment. A caller-provided `out` must not overlap Q, K/V page
storage, retained metadata, or caller-owned workspace. The launch
conservatively rejects overlapping storage spans. The API returns O only; LSE
and split-KV statistics are internal scratch.

## Tensor and metadata layouts

- SQ=1 fixed Q/O: `[B, Hq, D]`.
- Fixed SQ>1 Q/O: `[B, SQ, Hq, D]`.
- Packed Q/O: `[total_q, Hq, D]`, with contiguous `int32[B + 1]`
  `qo_indptr`. Offsets start at zero, increase strictly, and end at
  `total_q`.
- The planned fixed-capacity Q/head extent, `B * max_seq_len_q * Hq`, must fit
  in signed `int32`. This also bounds every packed `total_q * Hq` extent.
- Combined K/V cache: `[num_pages, 2, Hkv, page_size, D]`.
- Separate K/V cache: a `(K, V)` tuple whose members are
  `[num_pages, Hkv, page_size, D]`.
- Wrapper/one-shot page metadata uses FlashInfer CSR:
  `paged_kv_indptr[B + 1]`, `paged_kv_indices[num_used_pages]`, and
  `paged_kv_last_page_len[B]`, all contiguous CUDA `int32` tensors.
- The standalone launch uses the same indptr/indices plus explicit
  `seq_lens[B]` and a static `max_seq_len` upper bound.

Valid CSR metadata starts `paged_kv_indptr` at zero, increases it strictly,
and ends it at the number of used page-index entries. Every request owns at
least one page, every page ID indexes the physical cache, and wrapper last-page
lengths are in `[1, page_size]`. For every standalone request `b`, the live
metadata must also satisfy
`ceil(seq_lens[b] / page_size) <= paged_kv_indptr[b + 1] - paged_kv_indptr[b]`.
Query offsets start at zero, increase strictly, end at the packed Q extent,
and have every delta no larger than the planned `max_seq_len_q`. Causal
attention additionally requires each fixed or packed per-request Q length to
be no greater than the corresponding K/V length.

For request `b`, bottom-right causal row `i` can see through
`seq_len_k[b] - seq_len_q[b] + i`. A causal left window further retains the
current key and at most `window_left` preceding keys. `bmm1_scale` defaults to
`1 / sqrt(D)` and `bmm2_scale` defaults to 1; supplied scales must be finite,
positive Python scalars representable as positive `float32` values.

## Dataflow and source map

```text
Q + paged K/V
    -> staged Q/K/V
    -> QK MMA -> masked online softmax -> P
    -> PV MMA -> corrected O + internal log-normalizer state
    -> direct O, or split-KV partials -> reduction -> O
```

Eligible nonsplit work that exceeds one resident SM wave uses CLC-persistent
scheduling. A scheduler warp discovers each schedule token once and broadcasts it to
the worker tasks. Underfilled fixed-Q grids may instead split the K/V sequence
and reduce partial outputs. Packed-Q and sliding-window work remains nonsplit:
it uses CLC above one resident wave and the direct static path otherwise.

Page IDs are loaded from the live CSR metadata on every run and graph replay.
The `paged_kv_indices` storage and entry count are part of the plan, but valid
IDs may be remapped in-place without recompiling.

| Source | Responsibility |
| --- | --- |
| [`../../decode.py`](../../decode.py) | Public validation, planning, workspace binding, JIT caching, and launch adaptation |
| [`fmha_decode_config.py`](fmha_decode_config.py) | Kernel configuration and automatic launch selection |
| [`fmha_decode_kernel.py`](fmha_decode_kernel.py) | TS kernel construction and launch |
| [`fmha_decode_tasks.py`](fmha_decode_tasks.py) | Ordered load, MMA, softmax, correction, store, and scheduler work |
| [`fmha_decode_resources/`](fmha_decode_resources/) | GMEM/SMEM/TMEM resources and pipeline state |
| [`reduction.py`](reduction.py) | Separate split-KV reduction |

## Example

```python
import torch
from tensorrt_llm._torch.attention_backend.prims_ts import (
    BatchDecodePagedTSWrapper,
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
)

device = "cuda"
B, Hq, Hkv, D = 2, 32, 4, 128
page_size, pages_per_request = 32, 4
num_pages = B * pages_per_request

q = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
kv = torch.randn(
    num_pages, 2, Hkv, page_size, D,
    device=device,
    dtype=torch.float16,
)
paged_kv_indptr = torch.arange(
    0, num_pages + 1, pages_per_request, device=device, dtype=torch.int32
)
paged_kv_indices = torch.arange(num_pages, device=device, dtype=torch.int32)
last_page_len = torch.full(
    (B,), page_size, device=device, dtype=torch.int32
)

wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
wrapper.plan(
    paged_kv_indptr,
    paged_kv_indices,
    last_page_len,
    Hq,
    Hkv,
    D,
    page_size,
    q_data_type=q.dtype,
    mask_type="causal",
    max_kv_len=pages_per_request * page_size,
)
out = wrapper.run(q, kv)
assert out.shape == q.shape

# The standalone API uses caller-owned scratch and explicit K/V lengths.
max_seq_len = pages_per_request * page_size
workspace_bytes = get_prims_ts_batch_decode_workspace_size(
    B,
    Hq,
    Hkv,
    D,
    page_size,
    max_seq_len,
    q_dtype=q.dtype,
    mask_type="causal",
    device=q.device,
)
workspace = torch.zeros(workspace_bytes, device=device, dtype=torch.int8)
seq_lens = torch.full((B,), max_seq_len, device=device, dtype=torch.int32)
standalone_out = prims_ts_batch_decode_with_kv_cache(
    q,
    kv,
    workspace,
    paged_kv_indptr,
    paged_kv_indices,
    seq_lens,
    max_seq_len,
    mask_type="causal",
)
assert standalone_out.shape == q.shape
```

The wrapper snapshots K/V lengths derived from `paged_kv_indptr` and
`paged_kv_last_page_len`. Both tensors' values must remain unchanged until the
next successful plan. The `paged_kv_indices` storage and entry count must stay
fixed, but valid page IDs may be remapped in-place between completed runs or
graph replays because each execution reloads them. Packed `qo_indptr` storage
also stays fixed; interior offsets may change between completed executions only
while preserving positive deltas within the planned bound and the same final
packed extent. Do not mutate any retained metadata concurrently with a run or
graph replay that reads it. The hot path does not synchronize live device
metadata to the host or fully value- and bounds-check it. Invalid live page IDs
or packed offsets violate the contract and may cause incorrect results or
out-of-bounds access. A wrapper owns mutable scratch and supports only one
in-flight run or captured-graph replay; use separate wrapper instances for
concurrent execution.

For the standalone workflow, call
`get_prims_ts_batch_decode_workspace_size()` with the same shape, dtype, mask,
window, and Q-layout arguments as the launch. Allocate at least that many
bytes as a contiguous, 32-byte-aligned CUDA `torch.int8` or `torch.uint8`
tensor. Zero it before first use, and do not share it between concurrent
launches or captured graphs. It must not overlap Q, K/V cache, metadata, or
output storage. The standalone hot path trusts CSR, `seq_lens`, and packed-Q
values: keep lengths positive and within their static bounds, keep enough page
entries in every CSR row for its live length, and keep all page IDs valid. CSR
offsets, sequence lengths, page IDs, and packed-Q offsets may change between
completed launches or graph replays while preserving those contracts and
stable captured storage. Do not mutate them concurrently with an execution
that reads them. These live values are not host-synchronized or fully
value-checked at launch; invalid lengths or IDs may cause incorrect results or
out-of-bounds access.

For CUDA graph capture, compile and warm the planned configuration first,
retain all metadata and workspace storage at stable addresses, and pass a
preallocated compact, 16-byte-aligned `out` tensor.

## Limitations

- Only HND paged K/V is supported; contiguous K/V and NHD caches are outside
  this API.
- Attention sinks and custom masks are not exposed.
- Q, K, and V cannot use mixed dtypes.
- Runtime K lengths must be positive and no greater than the static plan bound.
- Packed offsets are validated during wrapper planning. The standalone hot
  path intentionally trusts their values to preserve synchronization-free
  launch behavior. Live causal metadata must preserve `q_len[b] <= kv_len[b]`.

## Validation

### TensorRT-LLM checkout

Run the local adapter and GPU backend unit coverage from the TensorRT-LLM
repository root:

```bash
pytest -q tests/unittest/_torch/attention/test_prims_ts_fmha.py
pytest -q tests/unittest/_torch/attention/test_prims_ts_attention_backend.py -k qwen2_gqa
```

The model-level decode end-to-end coverage requires a Blackwell GPU and the
model weights under `LLM_MODELS_ROOT`:

```bash
LLM_MODELS_ROOT=/path/to/models pytest -q \
  tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestQwen2_7BInstruct::test_prims_ts_bfloat16
```

### Upstream FlashInfer checkout

The upstream public accuracy, layout, mask, variable-Q, page-size, dtype,
CUDA-graph, split-KV, and resource-safety coverage lives in the commands below.
These paths do not exist in TensorRT-LLM; run them only from the FlashInfer
checkout pinned by the top-level provenance section:

```bash
pytest -q tests/attention/test_attention_ts_decode.py
pytest -q tests/trace/test_fi_trace_template_consistency.py
```
