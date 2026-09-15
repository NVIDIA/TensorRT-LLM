---
receipts:
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 39}
---

<!--
`tests:` is filled from the observed pytest count of the certifying run, never
from arithmetic.
-->

# mhc_gemm_sqrsum_fma

**Wraps** `torch.ops.trtllm.mhc_gemm_sqrsum_fma` (one call).

## Semantics

One kernel produces both halves of what the Hyper-Connection coefficient split
needs next — the projection and the statistic its normalization will use:

```
y[m, n] = dot(x[m, :], w[n, :])     # the UNNORMALIZED mix projection
r[m]    = sum_k x[m, k]^2           # the RAW square sum over the flat stream
```

**The fusion boundary.** The call owns the matmul and the square sum, and
**nothing else**. It does not divide `r` by `K`, does not take its reciprocal
square root, and does not scale `y` — `rstd = rsqrt(r / K + rms_eps)` and its
application belong to `activation/mhc_split_sinkhorn`. That split is the point:
it is what lets one `r` feed a normalization the caller schedules later, which
is the same delayed-schedule property that made the split entry preferable to
the large fusions.

The square sum is fused into the **first N-tile only** (`do_sqr = (n_start == 0)`
in `mhcKernels.cu`), so `r` is written exactly once per row no matter how many
column blocks the tactic creates.

Accumulation is fp32 `fmaf` throughout, over `x` converted from bf16 by
zero-extension (exact). A fp32 torch reference is therefore **not** a truth to
compare against but a second approximation in a different summation order —
measured on sm_103 at `M = 4096`, the fp32 reference sits 2.7e-05 from fp64
while this kernel sits 2.4e-06 from it. The GPU test uses float64.

## Signature

```python
def mhc_gemm_sqrsum_fma(x, w, tile_n=0) -> tuple[Tensor, Tensor]
```

| name | shape | dtype | layout / device | notes |
|---|---|---|---|---|
| `x` | `[M, K]` | bfloat16 | contiguous, CUDA | the flattened residual stream |
| `w` | `[N, K]` | float32 | contiguous, CUDA | already transposed: row `n` is output column `n` |
| `tile_n` | scalar | Python int | — | output columns per block; `<= 0` means the heuristic |
| **returns** | `[M, N]`, `[M]` | float32 | contiguous, CUDA | `(y, r)`, freshly allocated |

`M`, `N` and `K` are **derived** — `m, k = x.shape` and `n, _ = w.shape` — and
the outputs are allocated here. The op takes all three as arguments and indexes
whatever buffers it is given with them, so deriving removes the mismatch rather
than guarding it. It also removes the "`x` has more rows than `M`" case
outright: driven raw, the trailing rows are never read and the result is
bit-equal.

**Those two unpacks are the rank contract, not a guard**, and the distinction is
load-bearing. An earlier revision asserted `x.dim() == 2 and w.dim() == 2`;
driven raw on sm_103, storage-equivalent **rank-3** `x` and `w` are accepted and
**correct**, so that assert rejected harmless calls. The documented interface is
still 2-D and a wrong-rank caller still fails — as
`ValueError: too many values to unpack (expected 2)`, an interface mismatch,
rather than as an assertion claiming the op mishandles the input.

**`tile_m` is not in this signature.** The launcher discards it with
`(void) tile_m; // reserved for future multi-row kernel variant`. Driven on
sm_103 with 1, 7, -3 and 4096, every result is bit-equal to `tile_m = 0`.
Exposing an argument that provably does nothing would be exactly the hidden
surprise a contract reader should not have to discover.

## Certified coverage

What the receipt covers. Every row is one or more cases in
`tests/unittest/_torch/staircase/gemm/test_staircase_mhc_gemm_sqrsum_fma.py`.

| axis | certified values | cases |
|---|---|---|
| **V4.1 column x rows** | `N=24`, `K=20480`, x 14 row buckets | 14 |
| accuracy row buckets | 1, 2, 3, 8, 31, 32, 33, 64, 127, 128, 129, 256, 1024, 4096 | — |
| **`M`, as a proved interval** | `M ∈ [0, 8192]` — see "The certified `M` interval" below | 1 |
| zero logical rows | `M = 0` returns `[0,24]` and `[0]` with no queued CUDA error | 1 |
| **`tile_n`, every instantiated value** | 0, -1, 1, 2, 3, 4, 6, 8, 12, 24 — all **bit-equal** | 10 |
| `tile_m` | 0, 1, 7, -3, 4096 — all bit-equal; the launcher discards it | 1 |
| `r` is the raw square sum | no `/K`, no `rsqrt`; asserted against fp64 and by scale | 1 |
| `K` below the vector step | `K = 1023` (not a multiple of 4) accepted and correct | 1 |
| loud rejection | non-divisor `tile_n` (5, 7, 16); `x` fp32; `w` bf16 | 3 |
| wrapper guards (metadata only) | 3 branches, each raw-driven to accepted/finite/wrong first | 3 |
| guard: misaligned `K` | accepted at the call, faults at a later sync, then rejected — in a child process | 1 |
| guards that must NOT fire | extra `x` rows, bit-equal to correct | 1 |
| guards that must NOT fire | storage-equivalent rank-3 `x` / `w`, correct at the raw op | 1 |
| CUDA-graph replay | the **wrapper** captured and replayed twice with different inputs | 1 |
| total | | **39**, all passing; `tests: 39` in the receipt |

The tolerance is `torch.testing.assert_close`'s fp32 default, **unchanged**,
because the reference is float64. Measured against it: `y` max-abs 2.4e-06
(rel 2e-07) and `r` max-abs 3.9e-03 against values of order `K = 20480`
(rel 1.9e-07). Both sit well inside `rtol=1.3e-6, atol=1e-5`, so nothing is
widened and no discrimination study is owed.

### The certified `M` interval

*The rule.* A row's result does not depend on `M`. Each block computes the full
`K` for its own columns with the same thread mapping and the same reduction
tree, the tail is an early `row >= M` return, and every row offset is formed in
64 bits.

*What makes this non-obvious, and why it is the load-bearing measurement.*
`selectFmaTileN` returns **1 when `M <= 32` and 8 otherwise**, so the default
`tile_n = 0` silently selects a *different kernel instantiation* according to
how many rows were submitted. Had the instantiation changed a row's value, a
row's result would depend on the batch it rode in with, and no interval of `M`
could be certified at all — only a sampled list. So the tactic invariance is
asserted first, bit for bit, across all ten accepted `tile_n` values.

*The evidence.* `test_row_results_are_independent_of_m` checks one `M = 8192`
run against the float64 reference, then requires every `M` in a dense sweep of
0..129 plus the neighbourhoods of 256, 512, 1024, 4096 and 8192 to reproduce
that run's first `M` rows **bit-exactly**. 141 values of `M` in all, and the
sweep deliberately contains 31, 32 and 33 — the threshold itself.

So **any `M` in `[0, 8192]` is certified**, including zero logical rows (AC4)
and every small decode batch. `M > 8192` is not claimed.

Uncertified and therefore not claimed: `M > 8192`, `N` other than 24 (every
`tile_n` here is a divisor of 24, so the launcher's `default:` arm is
unreachable at this geometry and a different `N` could reach it), `K` other than
20480 and the 1023/2048/4096 controls, non-bf16 `x`, non-fp32 `w`, and any arch
other than sm_103.

## CUDA-graph replay

The wrapper is capture-safe and the target calls it normally inside a captured
region. Its two `torch.empty` calls are made *inside* `torch.cuda.graph`, so
they come from the graph's private memory pool and the tensors the capture call
returned are the same memory every replay rewrites.

`test_cuda_graph_replay_through_the_wrapper_matches_eager` replays **twice with
different inputs**, checking both against the float64 reference. One replay
would not distinguish a working graph from one that silently reproduced its
captured input.

## Metadata consumed

None. Stateless — no runtime, attention metadata, or workspace.

## Preconditions

Every statement below was driven on **sm_103 / trtllm 1.3.0rc26**.

### Rejected loudly by the op — the wrapper does not repeat any of these

| violation | observed message |
|---|---|
| `x` not bfloat16 | `RuntimeError: expected scalar type BFloat16 but found Float` |
| `w` not float32 | `RuntimeError: expected scalar type Float but found BFloat16` |
| `tile_n` not a divisor of `N` | `RuntimeError: ... mhcGemmSqrsumFmaLaunch: N=24 not divisible by tile_n=5` |

### Accepted silently AND guarded by the wrapper — four metadata violations

A wrapper assert is authorized only for a **pure tensor-metadata** violation
that the op was observed to accept silently. All four below satisfy both halves,
and each is raw-driven in the GPU test before the wrapper's rejection is
asserted.

| violation | outcome |
|---|---|
| `x` non-contiguous (a column slice of a wider buffer) | accepted, finite, wrong |
| `w` non-contiguous | accepted, finite, wrong |
| `w`'s `K` disagreeing with `x`'s | accepted, finite, wrong — the kernel strides `x`'s rows by the wrong length |
| `K % 4 != 0` while `K >= 1024` | **accepted at the call**, then a later unrelated sync raises `CUDA error: misaligned address` |

**The alignment one is the interesting case and the reason it is guarded.** The
vectorized main loop reads 4 bf16 as one 8-byte `ld.global.cs.v2.b32` and 4 fp32
as one 16-byte `ld.global.L1::evict_last.v4.f32`, both of which require `K` to be
a multiple of 4 once row offsets are in play. The fault is **asynchronous**: the
op call returns without raising, and an unrelated later line takes the blame —
so a caller has nothing to catch at the point of the mistake. Worse, the CUDA
context is unusable afterwards, so everything downstream in the process fails
too. That is why the GPU test drives this case in a **child process**: run
in-process it would fail every test after it.

**`K < 1024` is exempt and is NOT rejected.** The main loop runs only while
`k_base + K_STEP <= K`, so below 1024 every element goes through the scalar
2-byte tail, which is always aligned. `K = 1023` is accepted and correct, and
`test_short_k_below_the_vector_step_is_accepted_and_not_guarded` exists to stop
the guard being widened into one that rejects it.

### Accepted silently and NOT guarded — the `tile_n` scalar domain

`tile_n` is a **tactic value, not metadata**, so the metadata-only rule leaves it
to this section however it misbehaves. The launcher's test is `tile_n > 0`, so
**every non-positive value silently means "use the heuristic"** — `-1` is
accepted and bit-equal to `0` rather than being an error. A caller who passed a
negative expecting a diagnostic gets none.

### Harmless and therefore not guarded

* **`x` with more rows than `M`**: the trailing rows are never read and the
  result is bit-equal. Deriving `M` from `x.shape[0]` means this cannot be
  expressed through the wrapper at all.
* **Storage-equivalent rank-3 `x` or `w`** — same bytes, same order, one extra
  length-1 axis. Accepted and correct at the raw op, because it reads
  `data_ptr()` and is told `M`, `N` and `K` explicitly. This is why there is no
  rank assert: see the Signature section.

## Notes

* **A measurement artifact worth recording, because it nearly became a
  contract claim.** A first pass of the domain probe drove `K` in the order
  1023, 1025, 1026, 2048 and reported `K = 2048` as *misaligned*. 2048 is a
  multiple of 4 and perfectly legal; it was inheriting `K = 1025`'s fault,
  because a misaligned-address error **poisons the CUDA context** and every
  later launch in that process reports it whatever its own shapes are. Re-driven
  with the legal values first and the illegal one last, `K = 2048`, `4096` and
  `20480` are all accepted and correct. Any measurement taken after a
  misaligned launch in the same process is worthless.
* The kernel is instantiated for `tile_n` in {1, 2, 3, 4, 6, 8, 12, 24} and the
  launcher's `switch` has a `default:` arm that silently falls back to 24. At
  `N = 24` every divisor has its own instantiation, so that arm is unreachable
  here — it is named because a different `N` could reach it, and a silent
  fallback to a different tile width is not something a caller would see.
