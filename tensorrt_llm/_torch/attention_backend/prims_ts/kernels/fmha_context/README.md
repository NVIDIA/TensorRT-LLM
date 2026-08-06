<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->
<!-- Copyright (c) 2026 by FlashInfer team. -->

# Task-Scheduled FMHA Context

This directory contains the CuTe DSL task-scheduled (TS) FMHA context/prefill
kernel used by FlashInfer's experimental Blackwell APIs. One implementation
serves fixed contiguous, packed-ragged contiguous, and packed-query paged-KV
attention with MHA or GQA.

The public API exposes attention semantics, not scheduling controls. Contiguous
and paged plans select a nonpersistent, static-persistent, or CLC-persistent
launch from logical work, task topology, live-metadata requirements, causal
domain structure, and GPU capacity. Paired, live-ragged, and zero-offset
triangular contiguous domains use CLC. Immutable single-instance
bottom-right-offset domains launch directly within one resident wave and use
static persistence above one wave. Single-instance uniform causal paged plans
use static persistence: zero-offset triangular domains run a heavy-first
raster, while bottom-right-offset or windowed domains keep sequence-local
order. A positive causal left window selects an internal head-paired GQA
mapping; other cases use the query-paired mapping.

## Public APIs

Import these entry points from `tensorrt_llm._torch.attention_backend.prims_ts`:

| API | Use |
| --- | --- |
| `BatchPrefillTSWrapper` | Reusable fixed or packed-ragged contiguous Q/K/V plan. |
| `batch_prefill` | One-shot fixed or packed-ragged contiguous attention. |
| `BatchPrefillPagedTSWrapper` | Reusable packed-Q, paged-K/V plan. |
| `batch_prefill_with_paged_kv_cache` | One-shot packed-Q, paged-K/V attention. |

Planning validates static geometry, reads cumulative metadata when needed, and
may compile. The `run()` host path does not copy metadata values to the host or
synchronize. Packed contiguous plans retain `qo_indptr` and `kv_indptr` as
live inputs; general ragged kernels reload their values on every run, while a
uniform packed plan may compile its fixed offsets into the specialization.
General ragged paged kernels reload the retained `qo_indptr`; uniform paged
plans may compile those fixed offsets into the specialization. Both execute
against the K/V lengths and page table translated and snapshotted by `plan()`.

## Supported contract

| Feature | Support |
| --- | --- |
| GPU | SM100a/B200 (qualified); SM103a/B300 (architecture-gated, not yet signoff-qualified) |
| Head dimension | 128 or 256 |
| Head mapping | MHA/GQA; `Hq` must be divisible by `Hkv` |
| Q/K/V dtype | Matching `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Output dtype | `torch.float16`, `torch.bfloat16`, or `torch.float8_e4m3fn` |
| Contiguous storage | Fixed BSHD or packed-ragged THD |
| Paged storage | Packed Q plus separate compact HND K/V page pools |
| Page size | 16, 32, 64, or 128 tokens |
| Mask | Dense or bottom-right causal |
| Sliding window | Positive causal left window; `window_left=-1` disables it |
| Scheduling | Automatic nonpersistent, static-persistent, or CLC-persistent selection; no public tuning knob |
| Accumulation | FP32 QK/PV and softmax state |

Current accuracy and performance signoff is on SM100a/B200. SM103a/B300 is
admitted by the runtime architecture guard but remains to be qualified.

A positive left window requires GQA with an even `Hq/Hkv` ratio greater than
one. Causal attention requires `Sq <= Sk` for every request, both when the plan
is created and after any live cumulative-offset update. All tensor extents and
packed request lengths must be positive. Total logical Q and K extents—
`B*Sq`/`B*Sk` for fixed storage and `total_q`/`total_k` for packed storage—must
be at most `2**31 - 256`; this coordinate-representation limit reserves 255
values for the padded tail of the largest supported 256-row query work tile.

Q, K, V, and `out` must be compact, 16-byte-aligned CUDA tensors on one
device. Metadata must be compact CUDA `torch.int32` on that device and at
least 4-byte aligned. A caller-provided `out` must not overlap Q, K, V, or any
metadata retained by the plan. The launch conservatively rejects overlapping
storage spans. The API returns O only; rowwise LSE and other softmax state
remain internal to the kernel.

## Tensor and metadata layouts

Contiguous inputs:

- Fixed Q/O: `[B, Sq, Hq, D]`; K/V: `[B, Sk, Hkv, D]`.
- Packed Q/O: `[total_q, Hq, D]`; K/V:
  `[total_kv, Hkv, D]`.
- Packed metadata: compact CUDA `int32[B + 1]` `qo_indptr` and `kv_indptr`.
  Both start at zero, increase strictly, and end at the corresponding packed
  tensor extent.

Paged inputs:

- Q/O: `[total_q, Hq, D]`.
- Separate K and V pools: `[num_pages, Hkv, page_size, D]`.
- FlashInfer CSR metadata: `qo_indptr[B + 1]`,
  `paged_kv_indptr[B + 1]`, `paged_kv_indices[num_used_pages]`, and
  `paged_kv_last_page_len[B]`, all compact CUDA `int32` tensors.
- Physical page IDs may be arbitrary, repeated, and nonidentity ordered.

Every cumulative-offset vector starts at zero and increases strictly.
`qo_indptr[-1]` equals `total_q`, `paged_kv_indptr[-1]` equals the number of
page-index entries, every page ID indexes the physical cache, and each last
page length is in `[1, page_size]`.

For request `b`, bottom-right causal row `i` can see through
`Sk[b] - Sq[b] + i`. With `window_left=W>0`, the row retains that key and at
most `W` preceding keys. `sm_scale` defaults to `1 / sqrt(D)` and
`output_scale` defaults to 1; supplied scales must be finite, positive, and
representable as positive `float32` values.

The host reads cumulative metadata once during planning to establish the
static geometry and maximum Q/K capacities. For packed contiguous attention,
the plan keeps `qo_indptr` and `kv_indptr` as live device inputs; their storage
must remain valid and stable. Their values may change between runs while
preserving the planned batch, zero starting offsets, final packed extents,
strictly positive deltas, and these per-request capacity bounds. Each capacity
is the corresponding global plan maximum,
`max_b(Sq_plan[b])` or `max_b(Sk_plan[b])`, and applies independently to every
runtime request:

```text
0 < Sq[b] <= planned max_seq_len_q
0 < Sk[b] <= planned max_seq_len_k
```

Every causal replay must additionally satisfy `Sq[b] <= Sk[b]`. The
request-local bottom-right offset `Sk[b] - Sq[b]` may change; it is derived
from the live offsets. Fixed totals plus the per-request capacity bounds force
plan-time uniform Q or K lengths to remain unchanged. In particular, when a
dense plan compiles away request-local K-tail masking because every K length
equals the same 128-row-aligned maximum, the replay conditions preserve that
specialization.

Paged planning keeps `qo_indptr` as a live device input, but snapshots and
translates `paged_kv_indptr`, `paged_kv_indices`, and
`paged_kv_last_page_len`. Live Q offsets may change while preserving the
planned batch, a zero starting offset, the final packed-Q extent, strictly
positive deltas, and `Sq[b] <= planned max_seq_len_q`. For a causal plan, every
live `Sq[b]` must also be no greater than that request's snapshotted `Sk[b]`.
The kernel derives the request-local causal offset from those live Q and
snapshotted K lengths; there is no separate replay restriction on the maximum
offset. Changing any paged K/V metadata value requires another `plan()` call.
The `run()` host path trusts live offset values; violating this replay contract
can produce incorrect results or out-of-bounds access.

## Dataflow and source map

```text
Q + contiguous or paged K/V
    -> staged Q and streamed K/V
    -> QK MMA -> masked online softmax -> P + row statistics
    -> PV MMA -> online-softmax correction
    -> staged O -> output
```

The TS graph assigns load, MMA, softmax, correction, epilogue, page-offset,
and scheduling work to cooperating tasks. Resources own the corresponding
SMEM/TMEM buffers and pipeline state.

Paged D256 uses topology-derived page-ID staging. For a dense static domain
that is divisible by the complete staged window and whose exact SMEM footprint
fits the K/V cadence, each of the 32 producer lanes loads one page ID for each
of the two head-dimension stages, so one handoff covers 64 page IDs. Other
dtype footprints, short or partial domains, and causal domains retain the
natural 32-lane window or the ordinary per-tile path. This is an internal
consequence of the task topology, static geometry, and resource capacity; it
is not a user-selectable tuning parameter.

| Source | Responsibility |
| --- | --- |
| [`../../context.py`](../../context.py) | Public validation, metadata translation, automatic scheduling, JIT caching, and launch adaptation |
| [`fmha_kernel.py`](fmha_kernel.py) | Unified TS kernel and task graph construction |
| [`fmha_tasks.py`](fmha_tasks.py) | Load, MMA, softmax, correction, epilogue, page-offset, and scheduler work |
| [`fmha_resources.py`](fmha_resources.py) | GMEM/SMEM/TMEM resources and pipelines |
| [`helpers.py`](helpers.py) | Contiguous coordinates, masking, and schedule helpers |
| [`helpers_paged.py`](helpers_paged.py) | Paged-KV addressing and page-ID staging |

## Examples

Fixed contiguous causal attention:

```python
import torch
from tensorrt_llm._torch.attention_backend.prims_ts import BatchPrefillTSWrapper

device = "cuda"
B, Sq, Sk, Hq, Hkv, D = 2, 256, 512, 8, 2, 128
q = torch.randn(B, Sq, Hq, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, Sk, Hkv, D, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)

wrapper = BatchPrefillTSWrapper()
wrapper.plan(q, k, v, mask_type="causal")
out = wrapper.run(q, k, v)
assert out.shape == q.shape
```

Packed Q with a paged K/V cache:

```python
import torch
from tensorrt_llm._torch.attention_backend.prims_ts import batch_prefill_with_paged_kv_cache

device = "cuda"
B, Hq, Hkv, D, page_size = 2, 8, 2, 128, 32
q_lens, kv_pages = (32, 48), (2, 3)
num_pages = sum(kv_pages)

q = torch.randn(sum(q_lens), Hq, D, device=device, dtype=torch.float16)
k_cache = torch.randn(
    num_pages, Hkv, page_size, D, device=device, dtype=torch.float16
)
v_cache = torch.randn_like(k_cache)
qo_indptr = torch.tensor((0, 32, 80), device=device, dtype=torch.int32)
paged_kv_indptr = torch.tensor((0, 2, 5), device=device, dtype=torch.int32)
paged_kv_indices = torch.arange(num_pages, device=device, dtype=torch.int32)
last_page_len = torch.tensor((32, 16), device=device, dtype=torch.int32)

out = batch_prefill_with_paged_kv_cache(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    last_page_len,
    page_size=page_size,
    mask_type="causal",
)
assert out.shape == q.shape
```

For CUDA graph capture, call `plan()` and perform a warm-up `run()` first, keep
all planned tensors at stable addresses, and pass a preallocated,
non-overlapping `out`.

## Limitations

- Paged context accepts separate compact HND K/V pools with page size 16, 32,
  64, or 128.
- `window_left=0` is unsupported; use `-1` to disable the window or a positive
  value to enable it.
- Positive windows are restricted to even-ratio GQA because the kernel pairs
  query heads that share a K/V head.
- Attention sinks, custom masks, and mixed Q/K/V dtypes are not exposed.
- Re-plan after changing paged K/V metadata values. Live packed offsets may be
  updated only within their plan-time Q/K capacities, packed extents, and
  per-request causal contract.

## Validation

### TensorRT-LLM checkout

Run the local adapter and GPU backend unit coverage from the TensorRT-LLM
repository root:

```bash
pytest -q tests/unittest/_torch/attention/test_prims_ts_fmha.py
pytest -q tests/unittest/_torch/attention/test_prims_ts_attention_backend.py -k qwen2_gqa
```

The model-level context end-to-end coverage requires a Blackwell GPU and the
model weights under `LLM_MODELS_ROOT`:

```bash
LLM_MODELS_ROOT=/path/to/models pytest -q \
  tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestQwen2_7BInstruct::test_prims_ts_bfloat16
```

### Upstream FlashInfer checkout

The upstream public suite covers fixed, ragged, and paged layouts; MHA/GQA;
both head dimensions; `torch.float16`, `torch.bfloat16`, and
`torch.float8_e4m3fn` inputs; dense, causal, and left-window masks;
nonidentity pages; scheduler safety; CUDA graphs; and reference accuracy.
Explicit input-to-output dtype conversion coverage spans all nine pairings of
FP16, BF16, and FP8 input and output state. These paths do not exist in
TensorRT-LLM; run them only from the FlashInfer checkout pinned by the
top-level provenance section.

```bash
pytest -q tests/attention/test_attention_ts_context.py
pytest -q tests/attention/test_attention_ts_mask.py
pytest -q tests/trace/test_fi_trace_template_consistency.py
```
