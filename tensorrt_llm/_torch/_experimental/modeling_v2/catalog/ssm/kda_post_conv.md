---
receipts:
  sm_100: {status: passed, tests: 5}
---

# kda_post_conv

**Wraps** `tensorrt_llm._torch.modules.kimi_kda._kda_kernels.fused_kda_post_conv`
(one call; a Python function launching one Triton kernel, not a registered
torch op).

## Semantics

The step between KDA prefill's short convolution (`ssm/causal_conv1d_fwd`)
and its chunked delta rule (`ssm/kda_prefill`): the conv output arrives
channel-major as one packed `[3 * H * D, T]` tensor (`q | k | v` sections of
`H * D` channels each), and the delta rule wants token-major, per-head q / k / v
with q and k L2-normalized. Per token `t` and head `h`, in fp32:

```
q[0, t, h, :] = q_in[h, :, t] / sqrt(sum(q_in[h, :, t]^2) + l2_norm_eps)   # cast to packed.dtype
k[0, t, h, :] = k_in[h, :, t] / sqrt(sum(k_in[h, :, t]^2) + l2_norm_eps)
v[0, t, h, :] = v_in[h, :, t]                                               # copied, not computed
```

The epsilon sits inside the square root, so an all-zero head stays zero.

Fusion boundary: the normalization and the relayout only. The convolution
before it and the delta rule after it are separate entries.

## Signature

```python
def kda_post_conv(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `packed` | `[3 * num_heads * head_dim, tokens]` | bf16 / fp16 / fp32 | contiguous (channel-major) | CUDA |
| `num_heads`, `head_dim` | scalars | Python int | — | — |
| `l2_norm_eps` | scalar | Python float | — | — |
| returns `q`, `k`, `v` | `[1, tokens, num_heads, head_dim]` each | `packed.dtype` | newly allocated, contiguous | same device |

`packed` is not mutated.

## Metadata consumed

None. Stateless. Triton compiles on the first call per shape class; `tokens`
is not a specialization key.

## Preconditions

- `packed` is 2-D with `3 * num_heads * head_dim` rows, else `ValueError`.
- `packed` is contiguous, else `ValueError` (a channel-last view is refused,
  not read as dense).
- `tokens == 0` returns three empty `[1, 0, H, D]` tensors without a launch.

## Notes

- **Certified surface** (sm_100): Kimi K3's KDA prefill (`head_dim` 128,
  heads 6 and 24, bf16) at 1, 15, 16, 300 and 4096 tokens; fp16 and fp32 at
  4 heads; all-zero and tiny heads; zero tokens; the rejected inputs above.
- **Numerics.** q and k match an fp32 torch reference rounded once to
  `packed.dtype` within `rtol = 2 * eps(dtype)` (`1e-5` for fp32) and
  `atol = 1e-6` (the kernel's reduction order differs); v is bit-exact.
- Kimi K3's layer calls it with its default `l2_norm_eps`.
