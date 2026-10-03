---
receipts: {}
---

# situ_and_mul

**Wraps** `torch.ops.trtllm.situ_and_mul` (one call).

## Semantics

SiTU-gated elementwise multiply, Kimi K3's MLP activation (`hidden_act:
"situ"`), fused in one Triton kernel. The last dim of the input holds the
gate half followed by the up half:

```
d = x.shape[-1] // 2
g, u = x[:, :d], x[:, d:]                                 # loaded, then fp32
a    = beta * tanh(g / beta) * sigmoid(g)                 # soft-capped SiLU-like gate
u    = linear_beta * tanh(u / linear_beta)                # only if linear_beta is not None
out  = (a * u) cast to x.dtype                            # one final rounding
```

Everything between the loads and the store is fp32; `tanh` is libdevice's.
`beta` caps the gate at `beta` (as `g -> +inf`, `a -> beta`; as
`g -> -inf`, `a -> 0`), `linear_beta` caps the up half at `+-linear_beta`.
With `linear_beta=None` the up half is used unchanged. Kimi K3 calls it with
`beta = 4.0` (`activation_situ_beta`) and `linear_beta = 25.0`
(`activation_situ_linear_beta`).

Fusion boundary: activation and gating multiply only. The gate / up
projection that produces `x` and the down projection that consumes the
output are the caller's; there is no quantization of the output. Kimi K3's
routed experts apply the same activation inside the MXFP8 x MXFP4 MoE runner
(`act_type = 3`, see that contract); this entry is the one its dense layer
and shared experts call.

## Signature

```python
def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, 2 * d]` (2-D only) | bf16 / fp16 / fp32 | last dim dense (`stride(-1) == 1`); any row stride | CUDA |
| `beta` | scalar, `> 0` | Python float | — | — |
| `linear_beta` | scalar `> 0`, or `None` | Python float / None | — | — |
| returns | `[M, d]` | same as `x` | newly allocated, contiguous | same device as `x` |

`x` is not mutated.

## Metadata consumed

None. Stateless. Triton compiles the kernel on the first call per
specialization (`x.dtype`, whether `linear_beta` is `None`, and Triton's
divisibility specializations of `d`, the row strides and the pointers), so the
first call must not be inside a CUDA-graph capture; after that the op captures and replays with
rewritten inputs bit for bit.

## Preconditions

- `x` is 2-D: a 1-D or 3-D input raises `ValueError` (the op unpacks
  `x.shape` into two values). Flatten leading dims first.
- `x.shape[-1]` is even: an odd width raises `AssertionError`.
- `x.stride(-1) == 1`, else `ValueError: situ_and_mul requires a contiguous
  last dimension`. Rows may be strided (a column slice of a wider buffer is
  read correctly).
- `beta > 0` and `linear_beta > 0` (or `None`). Neither is checked: a cap of
  `0` zeroes its factor (`0 * tanh(+-inf)`), except for `NaN` where that half
  of the row is exactly zero (`0 / 0`). A negative cap computes the same
  values as its magnitude (`c * tanh(v / c)` is even in `c`).

## Notes

- **Certified surface** (sm_100): Kimi K3's per-rank widths 768, 3072, 4224
  and 16896 (shared experts and dense layer, TP16 and TP4) at `M` in
  `{1, 8, 64, 2048}` with `(beta, linear_beta) = (4.0, 25.0)`; width 6144
  with `(2.5, 7.0)`, `(1.0, None)`, `(4.0, None)`; bf16, fp16 and fp32
  inputs; saturated inputs (`|x| = 1e4`); a row-strided input; CUDA-graph
  replay.
- **Numerics.** Against an fp32 torch reference (`torch.tanh`,
  `torch.sigmoid`) the output is within one ulp of a 16-bit dtype (the test
  gates at two, `atol = 1e-5`) and within `1e-5` relative for fp32 output:
  the two `tanh` implementations differ only in the last fp32 bits.
- The module form, `tensorrt_llm._torch.modules.situ.SituAndMul`, calls this
  op when built with `use_fused_activation=True` on CUDA inputs and otherwise
  evaluates the same formula in eager torch.
