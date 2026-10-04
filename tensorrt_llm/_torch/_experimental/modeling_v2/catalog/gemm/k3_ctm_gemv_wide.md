---
receipts:
  sm_100: {status: passed, tests: 4}
---

# k3_ctm_gemv_wide

**Wraps** `torch.ops.trtllm.k3_ctm_gemv_wide` (one call).

## Semantics

The `k3_ctm_gemv_long` GEMV for up to 64 bf16 tokens (a wide decode step, e.g. speculative verification of several
requests): all `M` token columns go through one tensor-core MMA of 16, 32 or 64 columns, the smallest that holds
`M`, on the long kernel's split-K clusters and weight ring. Split and ring are not arguments; the op picks them
from the shape and the device. Output columns from `sig_col0` on can hold the sigmoid of the product, or the whole
output can be fp32.

```
acc[m, n] = sum_k x[m, k] * weight[n, k]       fp32 accumulation, 0 <= m < M <= 64, 0 <= n < N
out_fp32=False:  y[m, n] = bf16(acc[m, n])                  n < sig_col0, or every n when sig_col0 < 0
                 y[m, n] = bf16(sigmoid(bf16(acc[m, n])))   n >= sig_col0 >= 0; sigmoid in fp32
out_fp32=True:   y[m, n] = acc[m, n]                        fp32, not rounded
```

The bf16 products are accumulated in fp32 by the tensor cores (tcgen05 MMA into tensor memory). `split` and `ring`
come from `wide_config(N, K, token tile, SM count)` in the op module: the largest split in (8, 7, 6, 5, 4, 2) whose
`ceil(N / 128) * split` CTAs fit one wave of the device's SMs (with `ceil(K / 128) >= split`), then the deepest
weight ring that fits shared memory beside an activation ring of up to 3 stages. The summation order is
`k3_ctm_gemv_long`'s at that split: `K` in 128-column k-tiles (the last may be a 64-column half, read as zeros past
`K`), rank `r` of a 128-row block's cluster accumulating k-tiles `r, r + split, ...` in ascending order into its
own fp32 partial (the first `k_tiles % split` ranks one more when `split` does not divide them), and the `split`
partials of a token added in rank order in fp32 by the rank that reduces that token, then rounded once,
round-to-nearest-even. So a token's row is bit-identical to the row `k3_ctm_gemv_long` returns for that token at
the same split, with any ring and either transport, and it does not depend on `M`. A sigmoid column rounds the sum
to bf16, takes the sigmoid in fp32 (IEEE division, full-precision `exp`) and rounds again: its bits are exactly
`torch.sigmoid` of the bf16 value the same call returns with `sig_col0=-1`.

The split follows the device's SM count, so the bits can differ between devices with different SM counts. On
148- and 152-SM devices the certified shapes run at the splits listed under *Certified arguments*.

Fusion boundary: the single call computes the GEMV and either the sigmoid from `sig_col0` on or an fp32 output.
There is no bias, other activation, residual, all-reduce (under tensor parallelism the per-rank partial is
returned as is) or quantization; the caller owns those.

## Signature

```python
def k3_ctm_gemv_wide(
    x: torch.Tensor,
    weight: torch.Tensor,
    sig_col0: int = -1,
    out_fp32: bool = False,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, K]`, `1 <= M <= 64` | bf16 | contiguous, 16-byte-aligned start | CUDA |
| `weight` | `[N, K]` (`nn.Linear` layout) | bf16 | contiguous, 16-byte-aligned start | CUDA, `x`'s device |
| `sig_col0` | scalar | int: first sigmoid column, `< N`; `< 0`: none | — | — |
| `out_fp32` | scalar | bool: fp32 output, no rounding (needs `sig_col0 < 0`) | — | — |
| returns | `[M, N]` | bf16, or fp32 with `out_fp32` | newly allocated, contiguous | `x`'s device |

Every element of the returned tensor is written; the inputs are not mutated.

### Certified arguments

The Kimi K3 TP16 per-rank shapes of the projections of a wide decode step, at every `M` in 1..64 (all three token
tiles):

| Cell | `N` | `K` | `sig_col0` | `out_fp32` | split at 148 / 152 SMs |
|---|---|---|---|---|---|
| KDA q/k/v/g/f_a/b (last block 8 rows) | 3208 | 7168 | -1 | False | 5 |
| MLA `[W_a; W_g]`, gate columns as sigmoid | 2880 | 7168 | 2112 | False | 6 |
| KDA / MLA o_proj | 7168 | 768 | -1 | False | 2 |
| MoE head (latent down slice; router rows) | 280 | 7168 | -1 | True | 8 |
| MoE tail (5 k-tiles) | 7168 | 640 | -1 | False | 2 |
| shared expert gate_up | 768 | 7168 | -1 | False | 8 |
| dense MLP gate_up | 4224 | 7168 | -1 | False | 4 |
| dense MLP down (last k-tile a half) | 7168 | 2112 | -1 | False | 2 |
| drafter qkv | 512 | 7168 | -1 | False | 8 |
| drafter gate_up | 1792 | 7168 | -1 | False | 8 |
| drafter o_proj | 7168 | 384 | -1 | False | 2 |

Inputs are `x ~ N(0, 1)` and `weight ~ 0.02 * N(0, 1)`, rounded to bf16. The gate per cell: the `sig_col0=-1`
output within `max|y - ref| / max|ref| <= 8e-3` (bf16 output) or `<= 1e-4` (fp32 output) of the fp64 product of
the same bf16 inputs; with `sig_col0 >= 0`, the columns before it bit-identical to that output and the columns
from it bit-identical to `torch.sigmoid` of it; identical bits on a repeated call; and each `M`'s rows
bit-identical to the same rows of the 64-row call. Also certified:

- at the five shapes that `k3_ctm_gemv_long` runs at up to 8 tokens (MLA `[W_a; W_g]` with split 6, ring 6,
  `push=True`; dense MLP gate_up 4, 5, False; dense MLP down 2, 6, False; drafter qkv and drafter gate_up 8, 6,
  True), at `M` 1..8 and 16, 24, ..., 64: every token's row is bit-identical to `k3_ctm_gemv_long`'s for that
  token, in calls of up to 8 tokens, wherever the device's split equals that call's (all five at 148 or 152 SMs);
- at the MLA `[W_a; W_g]` and MoE head cells, `M` 1 and 64: calls captured in a CUDA graph after an eager call per
  key, replayed with `x` rewritten in place, return the bits of eager calls on the new `x`.

Other shapes the preconditions admit are accepted but not certified.

## Metadata consumed

None. The op reads no attention metadata, KV cache or module state; every input is an argument, and the kernel
runs on the current CUDA stream of `x`'s device. The device's SM count enters through `wide_config`.

One process-global cache sits behind it, and it is result-neutral: the compiled kernel per key
`(N, K, split, ring, activation ring, token tile, out_fp32, PDL on/off)`, where split and the rings follow from
`(N, K, token tile)` and the SM count. On one device that is one compile per `(N, K, out_fp32)` and token tile
(16 for `M <= 16`, 32 for `M <= 32`, 64 above). The first call with a new key compiles the kernel with the CuTe
DSL, which costs host time; `M` within a token tile and `sig_col0` are runtime arguments and never recompile. That
first call refuses to run under CUDA-graph capture: it raises `RuntimeError` ("run once per shape outside
CUDA-graph capture first") before launching anything. Call every key once eagerly, then capture; captured and
later calls reuse the compiled kernel. The PDL setting is read from `TRTLLM_ENABLE_PDL` on every call, so changing
it within a process adds a key.

## Preconditions

The op checks these and raises `ValueError` ("k3_ctm_gemv_wide: unsupported call ...") when one fails:

- `x` is a 2-D contiguous CUDA bf16 tensor with `1 <= M <= 64` rows whose start address is 16-byte aligned.
- `weight` is a 2-D contiguous bf16 tensor with `weight.shape[1] == x.shape[1]` whose start address is 16-byte
  aligned.
- `sig_col0 < N`, and `sig_col0 < 0` when `out_fp32`.
- `K % 64 == 0` and `N % 8 == 0`.
- A split exists: `2 * ceil(N / 128) <= SM count` and `ceil(K / 128) >= 2`. On a 148-SM device that is
  `N <= 9472`.

Certified refusals, at `N = 3208`, `K = 7168`: 0 and 65 rows, an `x` starting 2 bytes past a 16-byte boundary,
`sig_col0 = N`, `sig_col0 = 100` with `out_fp32`, `N = 3204`, `K = 7136`, and fp16 `x`.

Not checked by the op:

- `weight` is on `x`'s device; the op checks only `x.is_cuda`.
- The GPU has compute capability 10.x: the kernel uses tcgen05 MMA, tensor memory and thread-block clusters.
- The CuTe DSL (`cutlass`) and `cuda-python` (`cuda.bindings`) are importable. The op's check imports the kernel
  module, so without them a call raises `ImportError`, not `ValueError`.

## Notes

- Programmatic dependent launch (PDL). With `TRTLLM_ENABLE_PDL` unset or `1` the kernel is launched with PDL; any
  other value launches it without. Each CTA fills its weight ring (TMA, L2 evict-first) and prefetches the rest of
  its weight k-tiles into L2 without waiting for the grid dependency; only the reads of `x` follow
  `griddepcontrol.wait`. So `weight` must not be written by work still running ahead of this call on the stream,
  while `x` may be. There is no `trigger_early` argument: each CTA always executes
  `griddepcontrol.launch_dependents` right after issuing that first weight traffic, as `k3_ctm_gemv_long` does
  with `trigger_early=True`. The next kernel on the stream, if launched with PDL, can start while this one runs,
  and it must execute `griddepcontrol.wait` (`cudaGridDependencySynchronize`) before it reads `y`. Kernels
  launched without PDL, torch's included, start after this one completes as usual. None of this changes results.
- There is no `push` argument either: the split-K partials always travel by 16-byte `st.async` stores, and each
  rank reduces the partials of its own share of the tokens.
- Two identical calls return identical bits. The bits depend on the split, so on the device's SM count, but not
  on `M` or the PDL setting.
- The result is not bit-identical to cuBLAS: `F.linear` sums in another order. The two agree within the
  tolerances above.
- Grid: `ceil(N / 128) * split` CTAs of 256 threads in clusters of `split`; the choice of split keeps that at
  most the SM count.
- Only sm_100 has been measured.
