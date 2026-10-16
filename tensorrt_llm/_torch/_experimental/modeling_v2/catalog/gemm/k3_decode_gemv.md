---
receipts:
  sm_100: {status: passed, tests: 4}
---

# k3_decode_gemv

**Wraps** `torch.ops.trtllm.k3_decode_gemv` (one call).

## Semantics

The product of a decode step's activation rows with a projection weight, for at most 8 rows, in bf16:

```
y[m, n] = bf16(acc[m, n])          acc[m, n] = sum over k of x[m, k] * weight[n, k], accumulated in fp32
                                   m < M <= 8, n < N, k < K
```

The products are accumulated in fp32 by tcgen05 MMAs (M 128 x N 8 x K 16; the weight is the 128-row operand, the
activation the 8 token columns) into a TMEM accumulator, and each sum is rounded once, to nearest, to bf16. The op
picks one of two kernels from the weight's shape:

- **Short K** (`N % 128 == 0`, `K % 64 == 0`, `K <= 768`): one CTA per 128-row weight tile accumulates the whole of
  `K`, in k order, into one fp32 accumulator. Its whole weight slice is resident in shared memory, one stage per
  128-column k-tile.
- **Split K** (`K % 512 == 0`, `K > 768`, any `N`): a cluster of 4 CTAs per 128-row weight tile. Rank `r`
  accumulates k-tiles `r, r + 4, r + 8, ...` in fp32; rank `r` also owns rows `32 r .. 32 r + 31` of the tile, the
  other ranks push their fp32 partials of those rows into its shared memory, and it adds the four partials in rank
  order, `((p0 + p1) + p2) + p3`, before the one bf16 rounding. A last tile that runs past `N` computes zero-filled
  rows there and does not store them.

The kernel always computes 8 token columns (rows of `x` past `M` arrive as zeros from the TMA), so row `m` of the
result depends on row `m` of `x` only: the result of an `M`-row call is bit-identical to the same rows of an 8-row
call (certified). The summation order is fixed, so the result is bit-identical from run to run (certified). It is not
bit-identical to cuBLAS (`F.linear`), whose accumulation order differs.

Fusion boundary. Inside: the product and its bf16 rounding. Outside: any bias, activation, scaling or quantization,
the tensor-parallel reduction of a row-parallel projection's partial outputs, and flattening leading dims into `M`.
`x` and `weight` are read only; the result is a new tensor.

## Signature

```python
def k3_decode_gemv(x: torch.Tensor, weight: torch.Tensor, trigger_early: bool = True) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, K]`, `1 <= M <= 8` | bf16 | contiguous, 16-byte aligned | CUDA |
| `weight` | `[N, K]`, short K or split K (*Semantics*) | bf16 | contiguous, 16-byte aligned | CUDA, `x`'s device |
| `trigger_early` | scalar | Python bool | — | — |
| returns | `[M, N]` | bf16 | contiguous, newly allocated | `x`'s device |

`trigger_early`: under programmatic dependent launch (PDL), release the next kernel on the stream as soon as every CTA
has issued its weight loads (short K: its whole slice; split K: the first 6 of its k-tiles, or all if it has fewer),
rather than when this grid ends. Scheduling only (*Notes*).

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, K]`, `M` = 1, 2, ..., 8 | bf16 | contiguous | CUDA |
| `weight` | `[7168, 768]` (short K) and `[3208, 7168]` (split K) | bf16 | contiguous | CUDA |
| `trigger_early` | `True` and `False` | bool | — | — |
| returns | `[M, N]` | bf16 | contiguous | CUDA |

The weights are Kimi K3's TP16 per-rank KDA o_proj slice (`[7168, 768]`) and KDA input projection slice
(`[3208, 7168]`, whose 26th tile has 8 rows). Every cell: the error is at most 8e-3 of `max |ref|` against an fp64
torch product; a rerun is bit-identical; the rows are bit-identical to the same rows of the 8-row call;
`trigger_early=False` is bit-identical to `True`. Also certified: a CUDA graph of `M` 1 and `M` 8 calls on each
weight, captured after eager calls and replayed with `x` rewritten in place, bit-identical to eager calls on the same
rows; and the refusals under *Preconditions*.

## Metadata consumed

None besides the arguments and the current CUDA stream (the kernel is launched there), plus two pieces of process
state:

- A per-process compile cache keyed by `(N, K, trigger_early, PDL)`. `M` is a runtime argument, so one compiled
  kernel serves every `M`. The first call for a key compiles the kernel (seconds) and must be made eagerly: under
  CUDA-graph capture it raises `RuntimeError` ("must run once per shape outside CUDA-graph capture first"). The cache
  is result-neutral.
- `TRTLLM_ENABLE_PDL` (default `"1"`), read on every call: whether the kernel is launched with PDL. It is part of the
  cache key and changes scheduling, not results. The test runs with the default.

## Preconditions

- `x` and `weight` bf16, 2-D and contiguous; `x.shape[1] == weight.shape[1]`; `1 <= M <= 8`; `x` on a CUDA device.
- `(N, K)` taken by one of the kernels: short K (`N % 128 == 0`, `K % 64 == 0`, `ceil(K / 128) <= 6`) or split K
  (`K % 128 == 0`, `K / 128 > 6` and a multiple of 4, `N >= 1`).
- A call outside these raises `ValueError` ("k3_decode_gemv: unsupported call ...") before launching anything.
  Certified: `M` = 0 and 9, `[128, 1152]` (9 k-tiles: too many for short K, not a multiple of 4 for split K) and
  `[200, 768]` (short K needs whole 128-row tiles, split K more than 6 k-tiles).
- Not checked by the op: `weight` on `x`'s device; 16-byte-aligned data pointers (the op declares that alignment to
  the kernel; row slices `x[a:b]` of these shapes are aligned); an SM 10.x GPU (the kernels use tcgen05); the CuTe DSL
  (`cutlass`) and `cuda-python` (`cuda.bindings`), which the call imports, so without them it raises `ImportError`.
- Under PDL each CTA loads its weight before it waits for the kernel it follows on the stream, so `weight` must not
  be written by that kernel; a projection weight is constant during inference. `x` is read, and `y` written, after
  the wait.

## Notes

- PDL: with `TRTLLM_ENABLE_PDL` = `"1"` the kernel is launched with programmatic dependent launch. Each short-K CTA
  issues the loads of its whole weight slice at launch; each split-K CTA loads up to 6 of its k-tiles into its
  shared-memory ring and prefetches the rest into L2. Only the activation load waits for the preceding kernel
  (`griddepcontrol.wait`), so behind a long predecessor the call pays for the activation load, the MMAs and the store.
  A dependent released early by `trigger_early` must itself wait for this grid before reading `y`, as every PDL kernel
  waits for its predecessor before reading its output.
- The weight's shared-memory loads carry an L2 EVICT_FIRST hint: a projection's weight streams once per step without
  evicting the rest of L2.
- Grid: short K, `N / 128` CTAs of 256 threads; split K, `4 x ceil(N / 128)` CTAs in clusters of 4.
- `trtllm::k3_decode_gemv_tail` (the row-parallel MoE tail on the same kernel, with an RMS-scaled latent part) is a
  separate op and not this entry.
