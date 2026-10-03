---
receipts:
  sm_100: {status: passed, tests: 5}
---

# k3_embed_norm

**Wraps** `torch.ops.trtllm.k3_embed_norm` (one call).

## Semantics

A decode step's embedding rows and the first layer's input RMSNorm, in one launch. For `N <= 64` token ids, a bf16
table `[V, H]` and a bf16 norm weight `[H]`:

```
row[t]    = table[ids[t], :]  if 0 <= ids[t] < V,  else a zero row                 t < N
raw[t, :] = row[t]                                                                 (written into the caller's raw)
ss[t]     = sum over h of fp32(row[t, h])^2                                        fp32
rstd[t]   = rsqrt(ss[t] / H + eps)                                                 fp32, approximate (fast-math) rsqrt
out[t, h] = bf16((fp32(row[t, h]) * rstd[t]) * fp32(weight[h]))                    one bf16 rounding
```

`raw` receives the rows bit for bit. One 128-thread CTA computes one token's row: each thread squares and sums its
`H / 128` values, the sum is reduced across the warp's lanes, then across the 4 warps through shared memory, in a
fixed order; so the result is bit-identical from run to run (certified), and a row's result depends only on its id,
not on `N` or the other ids (certified: every row bit-identical to the same id's row of a 64-id call). An id outside
`[0, V)` reads nothing from the table: its `raw` row and its output row are zeros (certified).

The op states that `out` is bit-identical to the gather (`trtllm::k3_embed`) followed by `flashinfer.norm.rmsnorm` on
flashinfer's CuTe DSL RMSNorm kernel, the kernel the catalog's `norm/flashinfer_rmsnorm` runs unless
`FLASHINFER_USE_CUDA_NORM=1`: it keeps that kernel's geometry for `6144 < H <= 16384` (one 128-thread CTA per row)
and its arithmetic order. The op's own test checks that bit for bit. This entry's test checks `out` against a native
fp64 torch RMSNorm, within 8e-3 of each row's largest magnitude, and `raw` bit for bit against a torch gather.

Fusion boundary. Inside: the gather (zero rows for ids outside `[0, V)`), the copy of the rows into `raw`, the RMSNorm
and the weight scaling. Outside: producing the ids; allocating `raw` (the Kimi K3 model passes slot 0 of its
attention-residual snapshot bank `[S, N, H]`, so the rows become layer 0's first snapshot); any residual add,
`(1 + weight)` scaling or quantization. `ids`, `table` and `weight` are read only; every element of `raw` is written;
the result is a new tensor.

## Signature

```python
def k3_embed_norm(
    ids: torch.Tensor,
    table: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    raw: torch.Tensor,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `ids` | `[N]`, `1 <= N <= 64` | int32 or int64 | 1-D (the op copies a strided one) | CUDA |
| `table` | `[V, H]`, `6144 < H <= 16384`, `H % 1024 == 0` | bf16 | contiguous, 16-byte aligned | CUDA, `ids`' device |
| `weight` | `[H]` | bf16 | contiguous, 16-byte aligned | CUDA, `ids`' device |
| `eps` | scalar | Python float | — | — |
| `raw` | `[N, H]`, overwritten | bf16 | contiguous, 16-byte aligned | CUDA, `ids`' device |
| returns | `[N, H]` | bf16 | contiguous, newly allocated | `ids`' device |

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `ids` | `[N]`, `N` = 1, 2, ..., 8 and 16, 24, ..., 64 | int32 (every `N`), int64 (`N` 1, 8, 64) | contiguous | CUDA |
| `table` | `[163840, 7168]` | bf16 | contiguous | CUDA |
| `weight` | `[7168]` | bf16 | contiguous | CUDA |
| `eps` | `1e-6` (every cell) and `1e-5` | float | — | — |
| `raw` | `[N, 7168]`: slot 0 (every cell) or slot 1 of a `[4, N, 7168]` bank | bf16 | contiguous | CUDA |
| returns | `[N, 7168]` | bf16 | contiguous | CUDA |

The `N` are the token counts of Kimi K3's decode steps: up to 8 tokens, and DSpark verify steps of up to 8 requests of
8 tokens. The table has Kimi K3's shape (a replicated vocabulary of 163840), with rows at log-uniform scales from 1e-4
to 1e2 and one all-zero row; the norm weight is around 1 with every 101st element negative. The ids are random in
`[0, V)` with 0, `V - 1`, the zero row, the smallest-scale row and repeats; and, at `N` = 8, ids outside `[0, V)`
(-1, `V`, `V + 7`, `-V`, `2^31 - 1`, `-2^31`, and for int64 also `2^32`, `2^32 + 5`, `3 - 2^32`, `2^40`).

Every cell: `raw` bit-identical to the torch gather (zero rows outside `[0, V)`) and the bank's other slots
untouched; `out` within 8e-3 of each row's `max |ref|` against an fp64 torch RMSNorm, and exactly zero on zero rows;
a rerun bit-identical; every row bit-identical to the same id's row of the 64-id call; int64 ids bit-identical to
int32 ids. Also certified: a CUDA graph of an `N` = 8 call replayed with the ids rewritten in place, bit-identical to
eager calls; and the refusals under *Preconditions*.

## Metadata consumed

None besides the arguments and the current CUDA stream (the kernel is launched there), plus two pieces of process
state:

- A per-process compile cache keyed by `(N, V, H, ids dtype, PDL)`: `N`, `V` and `H` are compiled into the kernel,
  `eps` and the tensors' addresses are runtime arguments. The first call for a key compiles the kernel (seconds) and
  must be made eagerly: under CUDA-graph capture it raises `RuntimeError` ("must run once per shape outside CUDA-graph
  capture first"). The cache is result-neutral.
- `TRTLLM_ENABLE_PDL` (default `"1"`), read on every call: whether the kernel is launched with PDL. It is part of the
  cache key and changes scheduling, not results. The test runs with the default.

## Preconditions

- `ids` 1-D, int32 or int64, `1 <= N <= 64`, on a CUDA device.
- `table` bf16, 2-D, contiguous, 16-byte-aligned data pointer, `6144 < H <= 16384` and `H % 1024 == 0`.
- `weight` bf16 of shape `(H,)`, contiguous, 16-byte aligned; `raw` bf16 of shape `(N, H)`, contiguous, 16-byte
  aligned.
- A call outside these raises `ValueError` ("k3_embed_norm: unsupported call ...") before launching anything, leaving
  `raw` untouched. Certified: `N` = 0 and 65, int16 ids, `raw` `[64, H]` for 8 ids, `raw` 2 bytes past a 16-byte
  boundary, `weight` `[H / 2]`, and tables with `H` = 6144 and 7680.
- Not checked by the op: `table`, `weight` and `raw` on `ids`' device; `raw` not overlapping `ids`, `table` or
  `weight`; a GPU with programmatic dependent launch (SM 9.0 or newer: the kernel uses `griddepcontrol`; the receipts
  name the architectures it is certified on); the CuTe DSL (`cutlass`) and `cuda-python` (`cuda.bindings`), which the
  call imports, so without them it raises `ImportError`.

## Notes

- PDL: with `TRTLLM_ENABLE_PDL` = `"1"` the kernel is launched with programmatic dependent launch. It releases its
  dependents at entry and waits for the kernel it follows (`griddepcontrol.wait`) before reading anything, so the ids
  may be the predecessor's output. A dependent must itself wait for this grid before reading `raw` or the result, as
  every PDL kernel waits for its predecessor before reading its output.
- Grid: `N` CTAs of 128 threads; no CTA reads another's writes.
- The Kimi K3 table is replicated (no vocabulary shard), so the model never passes an id outside `[0, V)`; the zero
  rows define the op on every input. Where `weight` is negative, a zero row's output holds `-0.0`.
- `trtllm::k3_embed` (the gather alone, into a new tensor) is a separate op and not this entry.
