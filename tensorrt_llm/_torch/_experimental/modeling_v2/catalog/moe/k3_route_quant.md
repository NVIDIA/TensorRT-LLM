---
receipts:
  sm_100: {status: pending, tests: 18}
---

# k3_route_quant

**Wraps** `torch.ops.trtllm.k3_route_quant` (one call).

## Semantics

Kimi K3's top-16 routing and the MXFP8 quantization of the routed latent, for a decode batch of up to 64 tokens, in
one kernel: the CuTe DSL form of `trtllm::kimi_k3_noaux_tc_mxfp8_quant`, whose four outputs it returns bit for bit
(certified, below). Per token `t`, as the kernel module states it:

```
s       = 0.5 * tanhf(0.5 * router_logits[t]) + 0.5                          # fp32, [896]
ids     = the 16 experts with the largest s + e_score_correction_bias, descending, ties to the lower id
weights = bf16(s[ids] * routed_scaling_factor / (sum of the 16 s[ids] + 1e-20))
          # the division in fp64; the sum in fp32, in the order of a 16-lane xor-butterfly warp reduction
x_fp8, x_sf = MXFP8(latent[t]): one UE8M0 scale per 32 columns, rounded up from amax / 448
          (the recipe of cvt_warp_fp16_to_mxfp8), and the e4m3 codes of the scaled values
```

It returns `(topk_ids, topk_weights, quantized, scales)`: the global expert ids (int32 `[M, 16]`, in selection
order), their routing weights (bf16 `[M, 16]`), the latent's e4m3 codes (`[M, 3584]`) and its scales (uint8 `[M,
112]`, linear: byte `t * 112 + b` scales columns `[32 b, 32 b + 32)` of token `t`). The bias enters the selection
only; the weights are the unbiased sigmoids, renormalized over the 16 and scaled. The kernel selects differently from
the C++ one (one warp per token, each lane sorting its 28 keys, 16 rounds of warp max and min reductions; the
kernel's statement); the results are the same bits.

Certified, bit for bit against `trtllm::kimi_k3_noaux_tc_mxfp8_quant`, with and without the early trigger: at `M`
1-8, 16, 33 and 64 with random logits, and at `M` 1, 3 and 8 for 40 tied selection keys (ties to the lower id), equal
logits, huge logits (saturated sigmoids) and zero, large and denormal latent rows. Run-to-run bit identical, and each
`M`'s outputs the bits of the same rows of the 64-token call.

Fusion boundary. Inside: the sigmoid, the bias-corrected top-16 selection, the renormalized weights, the MXFP8
quantization of the latent. Outside: the router GEMM that produces the logits and the latent-down GEMM that produces
the latent (`moe/k3_moe_front` fuses both, the head all-gather and this op's device code into one kernel); the routed
experts (`moe/k3_moe`, which consumes these outputs).

## Signature

```python
def k3_route_quant(
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    latent: torch.Tensor,
    routed_scaling_factor: float,
    early_trigger: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

The op's schema has the same arguments (`scores`, `bias`, `hidden_states`, `routed_scaling_factor`,
`early_trigger`); `mutates_args=()`: it writes only its new outputs.

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `router_logits` | `[M, 896]`, `M` 1-64 | fp32 | contiguous | CUDA |
| `e_score_correction_bias` | `[896]` | fp32 | contiguous | CUDA, the same device |
| `latent` | `[M, 3584]`, the same `M` | bf16 | contiguous | CUDA, the same device |
| `routed_scaling_factor` | scalar (2.827 certified) | Python float | — | — |
| `early_trigger` | `False`, `True` | Python bool | — | — |
| returns | `topk_ids [M, 16]` int32, `topk_weights [M, 16]` bf16, `quantized [M, 3584]` float8_e4m3fn, `scales [M, 112]` uint8 | — | contiguous, newly allocated | `router_logits.device` |

`early_trigger` lets the next kernel on the stream, launched as a programmatic dependent, start as soon as every CTA
of this one has passed its own grid-dependency wait, before any output is written; without it, each CTA lets it start
once it has written its outputs, behind a GPU-scope fence (the kernel's code). A dependent launched early must wait
for this grid (`griddepcontrol.wait`) before it reads the outputs, as `moe/k3_moe` does: a caller sets it when the
next kernel is `k3_moe` launched as a programmatic dependent (a state with `use_pdl`). The outputs are the same bits
either way (certified).

## Metadata consumed

Stateless: no state object, nothing kept from one call to the next. Process state:

- A cache of compiled kernels keyed by (early trigger, PDL). The first call of a key compiles (seconds) and must be
  eager: under capture it raises `RuntimeError` ("must run once outside CUDA-graph capture first") instead of
  compiling, certified for both early-trigger builds with the cache cold. Result-neutral.
- `TRTLLM_ENABLE_PDL` (default on), read on every call and part of the key: launch the kernel as a programmatic
  dependent of the kernel before it. It changes scheduling, not results (the op's statement).

## Preconditions

- CUDA tensors on one device; fp32 logits and bias, bf16 latent; all contiguous; `router_logits` `[M, 896]`, the bias
  896 elements, `latent` `[M, 3584]` with the same `M`; 1 <= `M` <= 64. Anything else raises `ValueError` before any
  launch (the op's check). Certified: `M` 0 and 65, a strided view of the logits, bf16 logits, an fp16 latent, 895
  biases and a latent with another `M` raise `ValueError`, and the next call is correct.
- Before its own grid-dependency wait the kernel reads only `e_score_correction_bias`, a weight (the kernel's code):
  the kernel before it must not be writing the bias when this one launches early.
- The first call of each build eagerly (*Metadata consumed*). Calls may be captured: certified with a captured
  sequence at `M` 8, 3 and 64 (early trigger on, off, on) replayed 4 times with rewritten inputs, an eager call of
  another `M` between replays, every replayed and eager call the bits of the same call made alone.
- Certified on sm_100 only; the stock op it is compared with requires an SM 10.x device.

## Notes

- Certified path: one GB200 GPU (sm_100), test
  `tests/unittest/_torch/modeling_v2/moe/test_modeling_v2_k3_route_quant.py`. Reference: the stock
  `trtllm::kimi_k3_noaux_tc_mxfp8_quant`, bit for bit. The kernel test
  (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_route_quant.py`) also compares with the unfused chain
  (`trtllm::noaux_tc_op`, then `trtllm::mxfp8_quantize`) and reports a stable PyTorch sort of sigmoid + bias.
- Users: `moe/k3_moe` consumes the four outputs directly; `moe/k3_moe_front` runs this op's device code on the
  gathered head (and so returns the same bits for the same logits and latent).
- The compile cache is a module-level dict (result-neutral, above).
