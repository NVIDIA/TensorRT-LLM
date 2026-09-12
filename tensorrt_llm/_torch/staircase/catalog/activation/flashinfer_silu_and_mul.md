---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 4}
---

# flashinfer_silu_and_mul

**Wraps** `torch.ops.trtllm.flashinfer_silu_and_mul` (one call).

## Semantics

SiLU-gated elementwise multiply (the SwiGLU MLP activation), fused in one
kernel. The last dim of the input holds the gate half followed by the up
half:

```
d = x.shape[-1] // 2
out[..., i] = silu(x[..., i]) * x[..., d + i]      for i in [0, d)
silu(v) = v / (1 + exp(-v))
```

Both halves are loaded and the silu/multiply are computed in fp32 inside
the kernel; the result is cast back to the input dtype.

Fusion boundary: the single call computes activation and gating multiply
only. The caller owns the gate/up projection that produces `x` (whether as
one fused matmul or two) and the down projection that consumes the output.
There is no quantization of the output (a separate
`torch.ops.trtllm.silu_and_mul` op exists with an optional quant scale) and
no gelu variant (a separate `torch.ops.trtllm.flashinfer_gelu_tanh_and_mul`
op exists).

## Signature

```python
def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[..., 2 * d]`, e.g. `[num_tokens, 2 * intermediate]` | fp16 / bf16 | contiguous | CUDA |
| returns | `[..., d]` | same as `x` | newly allocated, contiguous | CUDA (same device as `x`) |

The output is a new tensor (`x` is not mutated).

## Metadata consumed

None. Stateless.

## Preconditions

- `x` is on a CUDA device and fully contiguous: the kernel indexes rows as
  `token * 2d` on the raw pointer, so any non-contiguity silently corrupts
  results.
- `x.shape[-1]` is even and a multiple of 16 elements for fp16/bf16
  (equivalently `d * itemsize % 16 == 0`). This is stricter than the
  op's own runtime check — see Notes.
- `x.shape[-1] >= 16` for fp16/bf16 (`d` must hold at least one 16-byte
  vector, or the computed block size is 0 and the launch is invalid).
- Dtype is fp16 or bf16. Other dtypes (including fp32) are rejected by the
  kernel's dtype dispatch.
- No upper bound on `d` beyond memory: `d` larger than
  `1024 * (16 / itemsize)` (e.g. > 8192 for bf16) takes a scalar remainder
  loop, verified correct on this machine.

## Notes

- The op is registered only when flashinfer is importable
  (`IS_FLASHINFER_AVAILABLE`); this pinned install ships
  flashinfer-python 0.6.14.
- Alignment trap: the op's Python-side check only validates
  `x.shape[-1] * itemsize % 16 == 0` (raising `ValueError`), but the
  vectorized load of the up half starts at element offset `d`, so `d`
  itself must be 16-byte aligned. Inputs that pass the check with
  `d * itemsize % 16 != 0` (e.g. bf16 with last dim 24 or 2728) crash with
  CUDA `misaligned address` — observed on sm_100. Keep the last dim a
  multiple of 16 elements.
- The misaligned-address failure is sticky: it poisons the CUDA context
  for the rest of the process.
- Programmatic dependent launch (PDL) is controlled by the env var
  `TRTLLM_ENABLE_PDL` (default enabled) inside the trtllm custom op; it
  affects scheduling only, not results.
- silu uses the fast-math `__expf`; observed error vs an fp32 torch
  reference stays within default `assert_close` tolerances for fp16/bf16.
