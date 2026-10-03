---
receipts: {}
---

# k3_ctm_gemv_long

**Wraps** `torch.ops.trtllm.k3_ctm_gemv_long` (one call).

## Semantics

The decode GEMV of a Kimi K3 projection with a long reduction dimension (the hidden size 7168, or any `K` too long
to hold in shared memory): at most 8 bf16 tokens times a bf16 weight in `nn.Linear` layout, as one CuTe DSL kernel
that splits each 128-row block of the weight over a cluster of `split` CTAs and streams it through a `ring` of
shared-memory stages. Output columns from `sig_col0` on can hold the sigmoid of the product instead.

```
acc[m, n] = sum_k x[m, k] * weight[n, k]          fp32 accumulation, 0 <= m < M <= 8, 0 <= n < N
y[m, n]   = bf16(acc[m, n])                       n < sig_col0, or every n when sig_col0 < 0
y[m, n]   = bf16(sigmoid(bf16(acc[m, n])))        n >= sig_col0 >= 0; sigmoid(v) = 1 / (1 + exp(-v)) in fp32
```

The bf16 products are accumulated in fp32 by the tensor cores (tcgen05 MMAs of 128 weight rows x 8 token columns
x 16 k, into tensor memory). Summation order: `K` is cut into 128-column k-tiles; the last one may be a 64-column
half, for which the TMA reads zeros past `K` in both operands. Rank `r` of a block's cluster accumulates k-tiles
`r, r + split, r + 2 * split, ...` in ascending order into its own fp32 partial (when `split` does not divide the
k-tiles, the first `k_tiles % split` ranks take one more). Each 32-row quarter `q` of the block has an owning rank
(`q`, or `q // 2` at `split=2`), which adds the `split` partials of its rows in rank order `0, 1, ...` in fp32 and
rounds the sum once, round-to-nearest-even. A sigmoid column rounds that sum to bf16, takes the sigmoid of the
bf16 value in fp32 (IEEE division, full-precision `exp`) and rounds again: its bits are exactly `torch.sigmoid` of
the bf16 value the same call returns with `sig_col0=-1`. That is the form of the MLA output gate
`bf16(sigmoid(g))` in the fused `[W_a; W_g]` projection.

`ring` (the weight stages each CTA keeps in shared memory) and `push` (how partials travel to the owning rank:
DSMEM stores plus a cluster-scope release arrive, or 16-byte `st.async` stores completing the owner's barrier by
bytes) change neither the sums nor their order. Rows past `M` enter the MMA as zeros and are not stored, and rows
past `N` are not stored, so a token's output row has the same bits whatever `M` is.

Fusion boundary: the single call computes the GEMV and, from `sig_col0` on, the sigmoid. There is no bias, other
activation, residual, all-reduce (under tensor parallelism the per-rank partial is returned as is) or
quantization; the caller owns those.

## Signature

```python
def k3_ctm_gemv_long(
    x: torch.Tensor,
    weight: torch.Tensor,
    sig_col0: int = -1,
    split: int = 6,
    ring: int = 5,
    trigger_early: bool = True,
    push: bool = False,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, K]`, `1 <= M <= 8` | bf16 | contiguous, 16-byte-aligned start | CUDA |
| `weight` | `[N, K]` (`nn.Linear` layout) | bf16 | contiguous, 16-byte-aligned start | CUDA, `x`'s device |
| `sig_col0` | scalar | int: first sigmoid column; `< 0` or `>= N`: none | — | — |
| `split` | scalar | int: 2, 4, 5, 6, 7 or 8 CTAs per 128 output rows | — | — |
| `ring` | scalar | int: weight k-tiles staged per CTA (see Preconditions) | — | — |
| `trigger_early` | scalar | bool: let PDL dependents launch early | — | — |
| `push` | scalar | bool: split-K transport | — | — |
| returns | `[M, N]` | bf16 | newly allocated, contiguous | `x`'s device |

Every element of the returned tensor is written; the inputs are not mutated.

### Certified arguments

The Kimi K3 TP16 per-rank shapes with the `sig_col0`, `split`, `ring` and `push` their call sites pass,
`trigger_early=True`, at every `M` in 1..8:

| Cell | `N` | `K` | `sig_col0` | `split` | `ring` | `push` |
|---|---|---|---|---|---|---|
| MLA `[W_a; W_g]`, gate columns as sigmoid | 2880 | 7168 | 2112 | 6 | 6 | True |
| dense MLP gate_up | 4224 | 7168 | -1 | 4 | 5 | False |
| dense MLP down (last k-tile a half) | 7168 | 2112 | -1 | 2 | 6 | False |
| drafter qkv | 512 | 7168 | -1 | 8 | 6 | True |
| drafter gate_up | 1792 | 7168 | -1 | 8 | 6 | True |
| drafter gate_up, synthetic drafter | 1536 | 7168 | -1 | 8 | 6 | True |
| KDA q/k/v/g/f_a/b (last block 8 rows) | 3208 | 7168 | -1 | 5 | 6 | True |

Inputs are `x ~ N(0, 1)` and `weight ~ 0.02 * N(0, 1)`, rounded to bf16. The gate per cell: the `sig_col0=-1`
output within `max|y - ref| / max|ref| <= 8e-3` of the fp64 product of the same bf16 inputs; with `sig_col0 >= 0`,
the columns before it bit-identical to that output and the columns from it bit-identical to `torch.sigmoid` of it;
identical bits on a repeated call; and each `M`'s rows bit-identical to the same rows of the 8-row call. Also
certified:

- one flag changed from the call site's values returns the call site's bits, at every `M`: `push=True` and
  `ring=3` at the dense MLP down, `push=True` and `trigger_early=False` at the dense MLP gate_up;
- at the MLA `[W_a; W_g]` and dense MLP down cells, `M` 1 and 8: calls captured in a CUDA graph after an eager
  call per key, replayed with `x` rewritten in place, return the bits of eager calls on the new `x`.

Other values the preconditions admit are accepted but not certified.

## Metadata consumed

None. The op reads no attention metadata, KV cache or module state; every input is an argument, and the kernel
runs on the current CUDA stream of `x`'s device.

One process-global cache sits behind it, and it is result-neutral: the compiled kernel per key
`(N, K, split, ring, trigger_early, push, PDL on/off)`. The first call with a new key compiles the kernel with the
CuTe DSL, which costs host time; `M` and `sig_col0` are runtime arguments and never recompile. That first call
refuses to run under CUDA-graph capture: it raises `RuntimeError` ("run once per shape outside CUDA-graph capture
first") before launching anything. Call every key once eagerly, then capture; captured and later calls reuse the
compiled kernel. The PDL setting is read from `TRTLLM_ENABLE_PDL` on every call, so changing it within a process
adds a key.

## Preconditions

The op checks these and raises `ValueError` ("k3_ctm_gemv_long: unsupported call ...") when one fails:

- `x` is a 2-D contiguous CUDA bf16 tensor with `1 <= M <= 8` rows.
- `weight` is a 2-D contiguous bf16 tensor with `weight.shape[1] == x.shape[1]` and `N >= 1`. `N` needs no
  divisibility: the last 128-row block may be partial.
- `K % 64 == 0`.
- `split` is 2, 4, 5, 6, 7 or 8.
- `1 <= ring <= ceil(K / 128) // split` (so `ceil(K / 128) >= split`), and the ring and the resident activation
  fit 216 KiB of shared memory: `ring * 32 KiB + (ceil(K / 128) // split + 1) * 2 KiB <= 216 KiB`. That caps
  `ring` at 6, and at 5 for `K = 7168` with `split=4`.

Certified refusals, at `N = 7168`, `K = 2112` (17 k-tiles): 0 and 9 rows, fp16 `x`, a row-strided `x`,
`K = 2080`, `split` 3 and 9, `ring=0`, `ring=3` at `split=8` (2 k-tiles per rank) and `ring=7` at `split=2`
(past shared memory).

Not checked by the op:

- `x` and `weight` start at 16-byte-aligned addresses. The op passes both to the kernel with a declared 16-byte
  alignment and loads them by TMA; a view whose start is not a multiple of 8 elements past an aligned allocation
  is outside the contract. Row slices `t[a:b]` are aligned, since `K % 64 == 0`.
- `weight` is on `x`'s device; the op checks only `x.is_cuda`.
- The GPU has compute capability 10.x: the kernel uses tcgen05 MMA, tensor memory and thread-block clusters.
- The CuTe DSL (`cutlass`) and `cuda-python` (`cuda.bindings`) are importable. The op's check imports the kernel
  module, so without them a call raises `ImportError`, not `ValueError`.

## Notes

- Programmatic dependent launch (PDL). With `TRTLLM_ENABLE_PDL` unset or `1` the kernel is launched with PDL; any
  other value launches it without. Each CTA fills its weight ring (TMA, L2 evict-first) and prefetches the rest of
  its weight k-tiles into L2 without waiting for the grid dependency; only the read of `x` follows
  `griddepcontrol.wait`. So `weight` must not be written by work still running ahead of this call on the stream,
  while `x` may be. With `trigger_early=True` each CTA executes `griddepcontrol.launch_dependents` right after
  issuing that first weight traffic: the next kernel on the stream, if launched with PDL, can start while this one
  runs, and it must execute `griddepcontrol.wait` (`cudaGridDependencySynchronize`) before it reads `y`. Kernels
  launched without PDL, torch's included, start after this one completes as usual. With `trigger_early=False`
  dependents launch when this grid completes. None of this changes results.
- Two identical calls return identical bits. The bits depend on `split`, whose summation orders can differ in the
  last bf16 place, but not on `ring`, `push`, `trigger_early`, the PDL setting or `M`. `k3_ctm_gemv_wide` at the
  same split returns the same bits for each token.
- Grid: `ceil(N / 128) * split` CTAs of 256 threads in clusters of `split`; the op does not limit it to one
  wave.
- The result is not bit-identical to cuBLAS: `F.linear` sums in another order. The two agree within the
  tolerance above.
- Only sm_100 has been measured.
