---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 6}
---

# flashinfer_rmsnorm

**Wraps** `torch.ops.trtllm.flashinfer_rmsnorm` (one call).

## Semantics

Root-mean-square normalization over the last dimension, with elementwise
weight scaling:

```
out[..., i] = x[..., i] / sqrt(mean(x[..., :]^2) + eps) * weight[i]
```

The squared-mean reduction and normalization are accumulated in fp32
inside the kernel; the result is cast back to the input dtype.

Fusion boundary: the single call computes normalization and weight scaling
only. There is no residual add (see `flashinfer_fused_add_rmsnorm` for
that), no `(1 + weight)` gemma-style scaling (see `flashinfer_gemma_rmsnorm`),
and no quantization. The caller owns everything else.

## Signature

```python
def flashinfer_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[num_tokens, hidden]` or `[batch, num_heads, head_dim]` | fp16 / bf16 / fp32 | last-dim stride must be 1; non-contiguous row strides are handled | CUDA |
| `weight` | `[hidden]` (or `[head_dim]` for 3D input) | same as `x` | contiguous | CUDA (same device as `x`) |
| `eps` | scalar | Python float | — | — |
| returns | same shape as `x` | same as `x` | newly allocated | CUDA |

The output is a new tensor (`x` is not mutated).

## Metadata consumed

None. Stateless.

## Preconditions

- `x` is 2D or 3D on a CUDA device; the normalized dim is the last one.
- `weight.shape == (x.shape[-1],)` and `weight.dtype == x.dtype`.
- `x.stride(-1) == 1` (rows may be strided, e.g. a column slice of a wider
  buffer; the kernel selects a strided code path in that case).
- Dtype is one of fp16, bf16, fp32. `float64`, `int8`, `uint8` and
  `float8_e5m2` raise `KeyError`, but **`float8_e4m3fn` does not** — the
  CuTe DSL path's dtype table maps it and returns a plausible fp8
  rmsnorm, so a caller who forgets to dequantize gets a result instead of
  an error (measured on the certified path, 2026-07-28). The uncertified
  CUDA-JIT path (`FLASHINFER_USE_CUDA_NORM=1`) does reject it — this
  bullet described that path.
- `hidden` (last dim) has no divisibility requirement: sizes not divisible
  by the 128-bit vector width (e.g. 111, 1152) fall back to a smaller
  vector size and were verified correct on this machine.

## Notes

- The op is registered only when flashinfer is importable
  (`IS_FLASHINFER_AVAILABLE`); this pinned install ships
  flashinfer-python 0.6.14.
- Programmatic dependent launch (PDL) is controlled by the env var
  `TRTLLM_ENABLE_PDL` (default enabled) inside the trtllm custom op;
  it affects scheduling only, not results.
- TRT-LLM's own `RMSNorm` module routes to this op only for fp16/bf16
  inputs, but fp32 works and passed the test on sm_100.
- Weight scaling is plain `w * normed(x)`, not the gemma `(1 + w)` form.
