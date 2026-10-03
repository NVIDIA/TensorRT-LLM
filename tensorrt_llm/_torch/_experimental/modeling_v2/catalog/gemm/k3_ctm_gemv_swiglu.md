---
receipts: {}
---

# k3_ctm_gemv_swiglu

**Wraps** `torch.ops.trtllm.k3_ctm_gemv_swiglu` (one call).

## Semantics

The down projection of a SwiGLU MLP with the activation folded into it: the `k3_ctm_gemv` GEMV whose input is
`silu(gate) * up` of a gate_up output, computed inside the kernel and never written to memory.

```
K = weight.shape[1];   g = gu[m, k],  u = gu[m, K + k]                     (gate columns first)
act[m, k] = bf16( (g * sigmoid(g)) * u )          in fp32; sigmoid(g) = 1 / (1 + exp(-g))
y[m, n]   = bf16( sum_k act[m, k] * weight[n, k] )                         0 <= m < M <= 8
```

The activation is evaluated in fp32 in that order, with IEEE division and the full-precision `exp`, and rounded
to bf16 once, as `silu_and_mul` rounds its output. The GEMV is `k3_ctm_gemv`'s: the products are accumulated in
fp32 by the tensor cores (tcgen05 MMA into tensor memory) and the sum is rounded to bf16 once, round-to-nearest-even.
Summation order: `K` is cut into 128-column k-tiles; each block of 128 output rows runs on `split` CTAs (a
cluster when `split > 1`), rank `r` accumulating k-tiles `r, r + split, ...` in ascending order into its own fp32
partial (when `split` does not divide the k-tiles, the first `k_tiles % split` ranks take one more); the rank that
owns an output row adds the `split` partials in rank order in fp32 and rounds once. `push` only chooses how
partials travel to the owner (DSMEM stores plus a release arrive, or `st.async` stores completing the owner's
barrier by bytes); the sums and their order are the same. Rows past `M` are zeros and are not stored, so a token's
output row has the same bits whatever `M` is.

The activation is the formula above, not a call of another kernel. torch's
`(F.silu(gate.float()) * up.float()).bfloat16()` computes the same function, but not necessarily with the same fp32
operations, so individual activation values can differ from it in the last bf16 place.

Fusion boundary: the single call computes the activation and the GEMV. The caller owns the gate_up projection
that produces `gu`. There is no bias, all-reduce (under tensor parallelism the per-rank partial is returned as is)
or quantization.

## Signature

```python
def k3_ctm_gemv_swiglu(
    gu: torch.Tensor,
    weight: torch.Tensor,
    trigger_early: bool = True,
    split: int = 2,
    push: bool = False,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `gu` | `[M, 2 * K]`, gate columns first, `1 <= M <= 8` | bf16 | contiguous, 16-byte-aligned start | CUDA |
| `weight` | `[N, K]` (`nn.Linear` layout) | bf16 | contiguous, 16-byte-aligned start | CUDA, `gu`'s device |
| `trigger_early` | scalar | bool: let PDL dependents launch early | — | — |
| `split` | scalar | int: 1, 2 or 4 CTAs per 128 output rows | — | — |
| `push` | scalar | bool: split-K transport | — | — |
| returns | `[M, N]` | bf16 | newly allocated, contiguous | `gu`'s device |

Every element of the returned tensor is written; the inputs are not mutated.

### Certified arguments

The Kimi K3 TP16 per-rank drafter down projection with the `split` and `push` its call site passes,
`trigger_early=True`, at every `M` in 1..8:

| Cell | `N` | `K` | `split` | `push` |
|---|---|---|---|---|
| drafter down | 7168 | 896 | 2 | True |
| drafter down, synthetic drafter | 7168 | 768 | 2 | True |

`K = 896` is 7 k-tiles, so the two ranks hold 4 and 3. Inputs are `gu ~ 2 * N(0, 1)` and
`weight ~ 0.02 * N(0, 1)`, rounded to bf16. The gate per cell: `max|y - ref| / max|ref| <= 8e-3` against the fp64
product of `weight` and torch's bf16 activation `(F.silu(gate.float()) * up.float()).bfloat16()`, identical bits on
a repeated call, and each `M`'s rows bit-identical to the same rows of the 8-row call. Also certified, at the
drafter down:

- `push=False` and `trigger_early=False` return the call site's bits, at every `M`;
- at `M` 1 and 8, calls captured in a CUDA graph after an eager call per key, replayed with `gu` rewritten in
  place, return the bits of eager calls on the new `gu`.

`split` 1 and 4 and other shapes the preconditions admit are accepted but not certified.

## Metadata consumed

None. The op reads no attention metadata, KV cache or module state; every input is an argument, and the kernel
runs on the current CUDA stream of `gu`'s device.

One process-global cache sits behind it, and it is result-neutral: the compiled kernel per key
`(N, K, split, trigger_early, push, PDL on/off)`. The first call with a new key compiles the kernel with the CuTe
DSL, which costs host time; `M` is a runtime argument and never recompiles. That first call refuses to run under
CUDA-graph capture: it raises `RuntimeError` ("run once per shape outside CUDA-graph capture first") before
launching anything. Call every key once eagerly, then capture; captured and later calls reuse the compiled
kernel. The PDL setting is read from `TRTLLM_ENABLE_PDL` on every call, so changing it within a process adds a
key.

## Preconditions

The op checks these and raises `ValueError` ("k3_ctm_gemv_swiglu: unsupported call ...") when one fails:

- `gu` is a 2-D contiguous CUDA bf16 tensor with `1 <= M <= 8` rows and `gu.shape[1] == 2 * weight.shape[1]`.
- `weight` is a 2-D contiguous bf16 tensor.
- `N % 128 == 0` and `K % 128 == 0`.
- `split` is 1, 2 or 4, `split <= K / 128`, and no rank holds more than 6 k-tiles:
  `ceil(K / 128 / split) <= 6`, i.e. `K <= 768` at `split=1`, `K <= 1536` at 2, `K <= 3072` at 4.

Certified refusals: 0 and 9 rows, fp16 `gu`, a row-strided `gu`, a `gu` of width `1664 != 2 * 896`, `N = 7104`,
`split=3`, and `K = 896` (7 k-tiles) at `split=1`.

Not checked by the op:

- `gu` and `weight` start at 16-byte-aligned addresses. The op passes both to the kernel with a declared 16-byte
  alignment and loads them by TMA; a view whose start is not a multiple of 8 elements past an aligned allocation
  is outside the contract. Row slices `t[a:b]` of these shapes are aligned.
- `weight` is on `gu`'s device; the op checks only `gu.is_cuda`.
- The GPU has compute capability 10.x: the kernel uses tcgen05 MMA, tensor memory and, at `split > 1`,
  thread-block clusters.
- The CuTe DSL (`cutlass`) and `cuda-python` (`cuda.bindings`) are importable. The op's check imports the kernel
  module, so without them a call raises `ImportError`, not `ValueError`.

## Notes

- Programmatic dependent launch (PDL). With `TRTLLM_ENABLE_PDL` unset or `1` the kernel is launched with PDL; any
  other value launches it without. Each CTA issues its whole weight read (TMA, L2 evict-first) without waiting
  for the grid dependency; only the read of `gu` follows `griddepcontrol.wait`. So `weight` must not be written by
  work still running ahead of this call on the stream, while `gu` may be. With `trigger_early=True` each CTA
  executes `griddepcontrol.launch_dependents` right after issuing its weight loads: the next kernel on the stream,
  if launched with PDL, can start while this one runs, and it must execute `griddepcontrol.wait`
  (`cudaGridDependencySynchronize`) before it reads `y`. Kernels launched without PDL, torch's included, start
  after this one completes as usual. With `trigger_early=False` dependents launch when this grid completes. None
  of this changes results.
- Two identical calls return identical bits. The bits depend on `split`, whose summation orders can differ in the
  last bf16 place, but not on `push`, `trigger_early`, the PDL setting or `M`.
- The result is not bit-identical to `silu_and_mul` followed by cuBLAS: the activation can differ in the last bf16
  place (see Semantics) and `F.linear` sums in another order. They agree within the tolerance above.
- Grid: `(N / 128) * split` CTAs of 256 threads, in clusters of `split` when `split > 1`.
- Only sm_100 has been measured.
