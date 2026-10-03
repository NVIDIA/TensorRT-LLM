---
receipts: {}
---

# k3_ctm_gemv

**Wraps** `torch.ops.trtllm.k3_ctm_gemv` (one call).

## Semantics

The decode GEMV of a Kimi K3 projection whose reduction dimension fits in shared memory: at most 8 bf16 tokens
times a bf16 weight in `nn.Linear` layout (the bias-free `F.linear(x, weight)`), as one CuTe DSL kernel on the
tensor cores.

```
y[m, n] = bf16( sum_k x[m, k] * weight[n, k] )      0 <= m < M <= 8,  0 <= n < N,  0 <= k < K
```

The bf16 products are accumulated in fp32 by the tensor cores (tcgen05 MMAs of 128 weight rows x 8 token columns
x 16 k, into tensor memory) and the sum is rounded to bf16 once, round-to-nearest-even. The summation order is
fixed by `K` and `split`:

- `K` is cut into 128-column k-tiles. Each block of 128 output rows is one CTA (`split=1`) or one cluster of
  `split` CTAs.
- With `split=1` the CTA accumulates all k-tiles in ascending order.
- With `split=2` or `4`, rank `r` of the cluster accumulates k-tiles `r, r + split, r + 2 * split, ...` in
  ascending order into its own fp32 partial (when `split` does not divide the k-tiles, the first
  `k_tiles % split` ranks take one more). The rank that owns an output row adds the `split` partials in rank order
  `0, 1, ...` in fp32 and rounds that sum once.

`push` only chooses how partials travel to the owning rank: DSMEM stores plus a cluster-scope release arrive
(`False`), or 16-byte `st.async` stores that complete the owner's barrier by bytes (`True`). The sums and their
order are the same either way, and `push` has no effect at `split=1`. Token rows are independent: rows past `M`
enter the MMA as zeros and are not stored, so a token's output row has the same bits whatever `M` is.

Fusion boundary: the single call computes the GEMV only. There is no bias, activation, residual, all-reduce
(under tensor parallelism the per-rank partial is returned as is) or quantization; the caller owns those.

## Signature

```python
def k3_ctm_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    trigger_early: bool = True,
    split: int = 1,
    push: bool = False,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, K]`, `1 <= M <= 8` | bf16 | contiguous, 16-byte-aligned start | CUDA |
| `weight` | `[N, K]` (`nn.Linear` layout) | bf16 | contiguous, 16-byte-aligned start | CUDA, `x`'s device |
| `trigger_early` | scalar | bool: let PDL dependents launch early | — | — |
| `split` | scalar | int: 1, 2 or 4 CTAs per 128 output rows | — | — |
| `push` | scalar | bool: split-K transport | — | — |
| returns | `[M, N]` | bf16 | newly allocated, contiguous | `x`'s device |

Every element of the returned tensor is written; the inputs are not mutated.

### Certified arguments

The Kimi K3 TP16 per-rank shapes with the `split` and `push` their call sites pass, `trigger_early=True`, at
every `M` in 1..8:

| Cell | `N` | `K` | `split` | `push` |
|---|---|---|---|---|
| MLA o_proj | 7168 | 768 | 1 | False |
| MLA o_proj | 7168 | 768 | 2 | False |
| drafter o_proj | 7168 | 384 | 1 | True |
| drafter o_proj, synthetic drafter | 7168 | 256 | 1 | True |

Inputs are `x ~ N(0, 1)` and `weight ~ 0.03 * N(0, 1)`, rounded to bf16. The gate per cell:
`max|y - ref| / max|ref| <= 8e-3` against the fp64 product of the same bf16 inputs, identical bits on a repeated
call, and each `M`'s rows bit-identical to the same rows of the 8-row call. Also certified, at the MLA o_proj:

- with `split=2`, `push=True` and `trigger_early=False` return the bits of the `push=False`,
  `trigger_early=True` call, at every `M`;
- with `split` 1 and 2, `M` 1 and 8: calls captured in a CUDA graph after an eager call per key, replayed with
  `x` rewritten in place, return the bits of eager calls on the new `x`.

`split=4` and other shapes the preconditions admit are accepted but not certified.

## Metadata consumed

None. The op reads no attention metadata, KV cache or module state; every input is an argument, and the kernel
runs on the current CUDA stream of `x`'s device.

One process-global cache sits behind it, and it is result-neutral: the compiled kernel per key
`(N, K, split, trigger_early, push, PDL on/off)`. The first call with a new key compiles the kernel with the CuTe
DSL, which costs host time; `M` is a runtime argument and never recompiles. That first call refuses to run under
CUDA-graph capture: it raises `RuntimeError` ("run once per shape outside CUDA-graph capture first") before
launching anything. Call every key once eagerly, then capture; captured and later calls reuse the compiled
kernel. The PDL setting is read from `TRTLLM_ENABLE_PDL` on every call, so changing it within a process adds a
key.

## Preconditions

The op checks these and raises `ValueError` ("k3_ctm_gemv: unsupported call ...") when one fails:

- `x` is a 2-D contiguous CUDA bf16 tensor with `1 <= M <= 8` rows.
- `weight` is a 2-D contiguous bf16 tensor with `weight.shape[1] == x.shape[1]`.
- `N % 128 == 0` and `K % 128 == 0`.
- `split` is 1, 2 or 4, `split <= K / 128`, and no rank holds more than 6 k-tiles:
  `ceil(K / 128 / split) <= 6`, i.e. `K <= 768` at `split=1`, `K <= 1536` at 2, `K <= 3072` at 4.

Certified refusals: 0 and 9 rows, fp16 `x`, a row-strided `x`, `N = 7104`, `K = 704`, `split=3`, and `K = 896`
(7 k-tiles) at `split=1`.

Not checked by the op:

- `x` and `weight` start at 16-byte-aligned addresses. The op passes both to the kernel with a declared 16-byte
  alignment and loads them by TMA; a view whose start is not a multiple of 8 elements past an aligned allocation
  is outside the contract. Row slices `t[a:b]` of these shapes are aligned.
- `weight` is on `x`'s device; the op checks only `x.is_cuda`.
- The GPU has compute capability 10.x: the kernel uses tcgen05 MMA, tensor memory and, at `split > 1`,
  thread-block clusters.
- The CuTe DSL (`cutlass`) and `cuda-python` (`cuda.bindings`) are importable. The op's check imports the kernel
  module, so without them a call raises `ImportError`, not `ValueError`.

## Notes

- Programmatic dependent launch (PDL). With `TRTLLM_ENABLE_PDL` unset or `1` the kernel is launched with PDL; any
  other value launches it without. Each CTA issues its whole weight read (TMA, L2 evict-first) without waiting
  for the grid dependency; only the read of `x` follows `griddepcontrol.wait`. So `weight` must not be written by
  work still running ahead of this call on the stream, while `x` may be. With `trigger_early=True` each CTA executes
  `griddepcontrol.launch_dependents` right after issuing its weight loads: the next kernel on the stream, if
  launched with PDL, can start while this one runs, and it must execute `griddepcontrol.wait`
  (`cudaGridDependencySynchronize`) before it reads `y`. Kernels launched without PDL, torch's included, start
  after this one completes as usual. With `trigger_early=False` dependents launch when this grid completes.
  None of this changes results.
- Two identical calls return identical bits. The bits depend on `split`, whose summation orders can differ in the
  last bf16 place, but not on `push`, `trigger_early`, the PDL setting or `M`.
- The result is not bit-identical to cuBLAS: `F.linear` sums in another order. The two agree within the
  tolerance above.
- Grid: `(N / 128) * split` CTAs of 256 threads, in clusters of `split` when `split > 1`.
- Only sm_100 has been measured.
