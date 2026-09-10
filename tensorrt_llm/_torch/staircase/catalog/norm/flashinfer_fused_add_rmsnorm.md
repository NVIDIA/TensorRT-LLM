---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 5}
---

# flashinfer_fused_add_rmsnorm

**Wraps** `torch.ops.trtllm.flashinfer_fused_add_rmsnorm` (one call).

## Semantics

Fused residual add + RMS normalization, in-place on both tensor arguments.
One call computes, per row:

```
h           = fp32(x) + fp32(residual)          # fp32 accumulation
residual    = cast(h, dtype)                     # overwritten in place
x           = cast(h / sqrt(mean(h^2) + eps) * weight, dtype)  # overwritten
```

The normalization reads the fp32 `h`, not the rounded `residual` output:
the add, the squared-mean reduction, and the weight scaling all happen in
fp32 inside the kernel before the final cast back to the input dtype.

Fusion boundary: residual add + rmsnorm + weight scaling, nothing else.
No `(1 + weight)` gemma-style scaling (see
`flashinfer_gemma_fused_add_rmsnorm`) and no quantization (see
`flashinfer_fused_add_rmsnorm_quant`). The caller owns everything before
(producing `x`, e.g. an attention/MLP output) and after (consuming the
normed `x` and the updated `residual`).

## Signature

```python
def flashinfer_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[num_tokens, hidden]` | fp16 / bf16 / fp32 | last-dim stride must be 1; row stride may exceed `hidden` | CUDA |
| `residual` | `[num_tokens, hidden]` | same as `x` | same constraint as `x` | CUDA (same device) |
| `weight` | `[hidden]` | same as `x` | contiguous | CUDA (same device) |
| `eps` | scalar | Python float | — | — |
| returns | — | — | — | — |

Returns `None`. Both `x` and `residual` are mutated in place: on return,
`residual` holds the pre-norm sum `x + residual` (the residual stream for
the next layer) and `x` holds the normalized, weight-scaled output.

## Metadata consumed

None. Stateless.

## Preconditions

- `x` and `residual` are 2D on the same CUDA device, same shape, same
  dtype. The kernel is compiled for `[M, H]`; do not pass 3D tensors (it
  would read `M = shape[0]`), but the guard is **loud, not silent**: the
  CuTe compiled-kernel argument check raises `ValueError: Mismatched
  Tensor on argument #0 ... expected ndim=2` before launch. Nothing is
  silently skipped (measured 2026-07-28); the precondition stands, its
  earlier justification did not.
- `weight.shape == (hidden,)`, `weight.dtype == x.dtype`, contiguous.
- Dtype is one of fp16, bf16, fp32. `float64`, `int8`, `uint8` and
  `float8_e5m2` raise `KeyError`, but **`float8_e4m3fn` does not** — the
  CuTe DSL path's dtype table maps it and returns a plausible fp8
  rmsnorm, so a caller who forgets to dequantize gets a result instead of
  an error (measured on the certified path, 2026-07-28). The uncertified
  CUDA-JIT path (`FLASHINFER_USE_CUDA_NORM=1`) does reject it — this
  bullet described that path.
- `x.stride(-1) == 1` and `residual.stride(-1) == 1`. If either tensor is
  non-contiguous (row stride > `hidden`), a strided kernel variant is
  selected; its symbolic row stride is declared divisible by the kernel
  vector size (up to 8 elements for bf16/fp16, 4 for fp32) and data
  pointers are assumed 16-byte aligned, so arbitrary odd row strides or
  misaligned slices are outside the contract.
- `hidden` has no divisibility requirement: sizes not divisible by the
  128-bit vector width (111, 1152) were verified correct on this machine.
- The caller must not need the pre-call contents of `x` or `residual`
  afterward — both are destroyed.

## Notes

- The op is registered only when flashinfer is importable
  (`IS_FLASHINFER_AVAILABLE`); this pinned install ships
  flashinfer-python 0.6.14, which routes to the CuTe DSL kernel
  (`fused_add_rmsnorm_cute`); a CUDA JIT fallback exists behind
  `FLASHINFER_USE_CUDA_NORM=1` but is not what these receipts certify.
- Programmatic dependent launch (PDL) is controlled by the env var
  `TRTLLM_ENABLE_PDL` (default enabled) inside the trtllm custom op;
  it affects scheduling only, not results.
- The kernel uses fast-math `rsqrt`; results matched the fp32 torch
  reference at default `assert_close` tolerances for all three dtypes.
- TRT-LLM's own `RMSNorm` module routes to this op only for fp16/bf16
  inputs, but fp32 works and passed the test on sm_100.
- Contiguous inputs with `num_tokens * hidden > 2^31 - 1` are
  transparently routed to the strided (int64-offset) kernel variant.
