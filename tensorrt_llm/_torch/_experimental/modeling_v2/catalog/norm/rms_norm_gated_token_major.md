---
receipts:
  sm_100: {status: passed, tests: 6}
---

# rms_norm_gated_token_major

**Wraps** `torch.ops.trtllm.rms_norm_gated_token_major` (one call).

## Semantics

Gated RMS norm of per-head rows, norm first, then the gate. `x` holds one row
per (token, head) pair, token-major (`r = t * heads + h`); the gate of row `r`
is `z[t, h, :]`. Per row, in fp32:

```
y   = x[r] / sqrt(mean(x[r]^2) + eps) * weight
g   = sigmoid(z[t, h])                    # gate_activation == "sigmoid" (KDA)
g   = z[t, h] * sigmoid(z[t, h])          # gate_activation == "silu"    (GDN)
out = (y * g) cast to x.dtype
```

With `fp8_scale` (a scalar fp32 tensor, the downstream static input scale)
the output is quantized in the same kernel the way a separate norm and
static-quantize pair would do it: `y * g` is first rounded to `x.dtype`, then
multiplied by the correctly rounded reciprocal of the scale and rounded to
float8_e4m3fn.

Kimi K3's KDA layers call it on the chunked-delta-rule output `o`
(`[tokens * heads, 128]`) with the full-rank gate projection as `z`, a column
slice of a wider per-token buffer, `gate_activation="sigmoid"`, the `o_norm`
weight and `eps = 1e-5`.

Fusion boundary: the norm, the weight and the gate only. The projections
that produce `x` and `z` and the output projection are the caller's.

## Signature

```python
def rms_norm_gated_token_major(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_scale: Optional[torch.Tensor] = None,
    gate_activation: str = "silu",
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[tokens * heads, N]` | bf16 / fp16 / fp32 | last dim dense | CUDA |
| `z` | `[tokens, heads, N]` | as `x` | see Preconditions | CUDA, same device |
| `weight` | `[N]` | as `x` (read as fp32) | any (made contiguous) | CUDA, same device |
| `eps` | scalar | Python float | — | — |
| `fp8_scale` | scalar tensor or `None` | fp32 | — | CUDA, same device |
| `gate_activation` | `"silu"` or `"sigmoid"` | Python str | — | — |
| returns | `[tokens * heads, N]` | `x.dtype`, or float8_e4m3fn with `fp8_scale` | newly allocated, contiguous | same device |

No input is mutated.

## Metadata consumed

None. Stateless. Triton compiles on the first call per specialization, so the
first call must not be inside a CUDA-graph capture; after that the op
captures and replays with rewritten inputs bit for bit. The multi-row kernel
launches with programmatic dependent launch (waiting on its predecessor
before the first load, releasing its dependents after the last store) when
`TRTLLM_ENABLE_PDL` is unset or `"1"` (any other value turns it off), the GPU
is SM 9.0 or newer, and its grid has fewer CTAs than the GPU has SMs.

## Preconditions

- `x` is 2-D and `z` 3-D, otherwise `ValueError` (unpacking their shapes);
  `z.shape[2] == x.shape[1]` and `z.shape[0] * z.shape[1] == x.shape[0]`,
  otherwise `AssertionError`.
- `gate_activation` is `"silu"` or `"sigmoid"`; otherwise `ValueError`.
- Kernel choice, all with the same result:
  - **multi-row kernel over tokens** (4 rows per CTA, the gate read in place)
    when `x.stride(-1) == 1`, `z.stride(2) == 1`, `z.stride(1) == N` (each
    token's `(heads, N)` block dense; the token stride is free) and `N` is a
    power of two `<= 256`;
  - otherwise on `z.reshape(tokens * heads, N)` (a view when the strides
    allow it, else a copy, one extra kernel), where `x` and that gate need a
    unit last stride (`AssertionError`): the multi-row kernel one head per
    row when `N` is a power of two `<= 256`, the **generic kernel** when it
    is not.

## Notes

- **Certified surface** (sm_100): Kimi K3's KDA output gate (`N = 128`,
  heads 6 / 12 / 24 / 96, tokens 1 to 8192, `z` a column slice, sigmoid,
  bf16); `N` 32 / 64 / 128 / 256 with the silu and sigmoid gates and
  `heads = 1`; the generic kernel at `N` 96 and 512; a gate whose heads are
  strided (the multi-row kernel one head per row); fp8 output at scales
  0.5 / 1 / 3; CUDA-graph replay.
- **Numerics.** Against an fp32 torch reference the output is within one ulp
  of `x.dtype` (the test gates at two, `atol = 1e-5`); with `fp8_scale` it is
  within one e4m3 step, the size of a one-ulp difference before
  quantization.
- `trtllm::rms_norm_gated_token_major` is the token-major form of the
  module-level gated norm in `tensorrt_llm._torch.modules.mamba.layernorm_gated`
  (`RMSNorm` with a gate); that module's own call takes `z` row-aligned with
  `x` instead.
