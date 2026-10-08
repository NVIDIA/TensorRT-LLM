---
receipts:
  sm_100: {status: passed, tests: 6}
---

# kimi_k3_noaux_tc_mxfp8_quant

**Wraps** `torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant` (one call).

## Semantics

Kimi K3's routed-MoE input for a decode-sized batch, in one launch: the
`noaux_tc` routing of `moe/noaux_tc_op` at Kimi K3's fixed configuration and
the MXFP8 quantization of `quantization/mxfp8_quantize` of the latent hidden
row that the routed experts read. The grid has `2 * M` CTAs: CTAs `[0, M)`
route one token each, CTAs `[M, 2M)` quantize one hidden row each. The two
halves share no data.

Routing, per token, is `noaux_tc_op(router_logits, bias, n_group=1,
topk_group=1, topk=16, routed_scaling_factor)` over 896 experts, computed by
the same device routine (see that contract for the sigmoid's tanh form, the
bias entering selection only, the stable descending order with ties toward
the lower expert index, and the fp64 renormalization):

```
s     = 0.5 * tanh(0.5 * L) + 0.5                  # fp32
ids   = 16 largest (s + bias), stable descending   # int32
w     = s[ids]
w     = fp64(w) * routed_scaling_factor / (fp64(sum_fp32(w)) + 1e-20)
w     = round_to_nearest_even_bf16(w)              # ONE rounding, fp64 -> bf16
```

The weights are rounded **once, from fp64 to bf16**, where `noaux_tc_op`
called with fp32 logits rounds fp64 to fp32. Rounding that fp32 result to
bf16 therefore differs from this op by one bf16 ulp at about half of the fp32
values that sit exactly on a bf16 rounding midpoint, those whose fp64 value
lies on the other side of the midpoint from its even neighbour (a
double-rounding difference, expected about once in 2^17 weights).

Quantization, per row, is `mxfp8_quantize(hidden_states,
swizzled_layout=False, alignment=512)`: one UE8M0 scale per 32 contiguous
elements (the smallest power of two at or above `amax / 448`), the elements
rounded to e4m3, scales in the linear layout. 3584 is a multiple of 512, so
nothing is padded. The scale buffer comes back as a 2-D
`[M, 112]` view of the linear buffer `mxfp8_quantize` returns 1-D.

**The tuple order is ids first, weights second** — the opposite of
`noaux_tc_op`, which returns weights first.

Fusion boundary: the router GEMM that produces `router_logits` and
everything downstream (the MXFP8 x MXFP4 experts, the combine) are the
caller's. The four outputs are the `topk_ids` / `topk_weights` /
`hidden_states` / `hidden_states_scale` that
`moe/mxe4m3_mxe2m1_block_scale_moe_runner` takes in its pre-routed form.

## Signature

```python
def kimi_k3_noaux_tc_mxfp8_quant(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    hidden_states: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

The wrapper names the op's first schema argument (`scores`) `router_logits`,
as `noaux_tc_op`'s does; arguments are passed positionally and unchanged.

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `router_logits` | `[M, 896]`, `1 <= M <= 64` | fp32 | contiguous | CUDA |
| `bias` | `[896]` | fp32 | contiguous | CUDA, same device |
| `hidden_states` | `[M, 3584]` (same `M`) | bf16 | contiguous | CUDA, same device |
| `routed_scaling_factor` | scalar | Python float | — | — |
| returns `[0]` (`topk_ids`) | `[M, 16]` | int32 | contiguous, newly allocated | same device |
| returns `[1]` (`topk_weights`) | `[M, 16]` | bf16 | contiguous, newly allocated | same device |
| returns `[2]` (`data`) | `[M, 3584]` | float8_e4m3fn | contiguous, newly allocated | same device |
| returns `[3]` (`scales`) | `[M, 112]` | uint8 (UE8M0) | contiguous, newly allocated; linear | same device |

No input is mutated or retained.

## Metadata consumed

None. Stateless — no workspace, no cache, no tuner state. The call runs on
the ambient stream and is safe to capture in a CUDA graph; replaying with
rewritten inputs gives the eager result bit for bit. It launches with
programmatic dependent launch when `TRTLLM_ENABLE_PDL` is set (the default),
which affects scheduling only.

## Preconditions

All of these are checked by the op, and a violation raises `RuntimeError`:

- An SM 10.x GPU (`requires an SM10x architecture`).
- `router_logits` and `bias` fp32, `hidden_states` bf16; all three CUDA
  tensors on one device and contiguous. A strided view raises rather than
  being read as dense (unlike `noaux_tc_op`).
- `router_logits` 2-D `[M, 896]`, `bias` `[896]`, `hidden_states` 2-D
  `[M, 3584]` with the same `M`.
- `1 <= M <= 64`. `M == 0` raises (where `noaux_tc_op` returns empty tensors);
  a batch above 64 tokens is routed by `noaux_tc_op` and quantized by
  `mxfp8_quantize` instead.

The expert count, top-k, grouping (one group) and hidden width are fixed;
there is no argument for them.

## Notes

- **Certified surface** (sm_100): `M` in `{1, 2, 5, 8, 16, 33, 63, 64}`,
  `routed_scaling_factor` in `{1.0, 2.5}` (1.0 is Kimi K3's), random logits at
  scale 2.5, tied logits (four levels, zero bias), saturated logits (scale
  40), hidden rows scaled by `1e-3` / `1` / `1e4` with an all-zero block;
  CUDA-graph capture and replay at `M` 1 / 8 / 64; an alternate stream.
- **Numerics.** Expert ids are bit-identical to `noaux_tc_op`'s. The
  quantized data and scales are bit-identical to `mxfp8_quantize`'s. The
  weights match `noaux_tc_op`'s fp32 weights rounded to bf16 within one bf16
  ulp, and bit for bit except for the double-rounding cases above; the test
  gates at one ulp plus a budget of one mismatch per 1000 weights.
- The `moe/noaux_tc_op` and `quantization/mxfp8_quantize` contracts carry the
  rest of the behaviour this op inherits: the saturating tanh-form sigmoid,
  `-inf` / `NaN` logits, non-finite or denormal hidden blocks.
- Kimi K3's model code calls this op on its decode layout for batches of at
  most 64 tokens, and `noaux_tc_op` plus `mxfp8_quantize(alignment=512)`
  above that.
