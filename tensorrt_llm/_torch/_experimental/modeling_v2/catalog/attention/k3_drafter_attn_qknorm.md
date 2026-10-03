---
receipts:
  sm_100: {status: passed, tests: 8}
---

# k3_drafter_attn_qknorm

**Wraps** `torch.ops.trtllm.k3_drafter_attn_qknorm` (one call), the CuTe DSL kernel of `attention/k3_drafter_attn`
built with the q / k normalization in front.

## Semantics

`attention/k3_drafter_attn` on the raw output of the drafter's QKV projection. Before the attention, the kernel
applies to every q head and every k head of `qkv` (the block's own `k`, not the cached rows):

```
x = x / sqrt(mean(x^2) + eps) * w            # w = q_w for q heads, k_w for k heads; per head over its 64 dims
x = NeoX RoPE(x, positions[row], rope_base)  # rotary dim 64: the halves (x[:32], x[32:]) rotate together
```

with `attention/fused_qk_norm_rope`'s arithmetic (`is_neox = True`, full rotary dim, no YaRN), then attends exactly
as `k3_drafter_attn`. The normalized values exist only inside the kernel: `qkv` is read only (certified bit for
bit) and `out` is written.

Fusion boundary. Inside: the q / k RMSNorm, the RoPE and the whole of `k3_drafter_attn`. Outside: the projection
that produced `qkv`, and storing the roped `k` / `v` in the cache, if the caller keeps them.

## Signature

```python
def k3_drafter_attn_qknorm(
    qkv: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    rope_base: float,
    cache: torch.Tensor,
    page_table: torch.Tensor,
    ctx_len: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    out: torch.Tensor,
) -> None
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `qkv` | as `k3_drafter_attn`, unnormalized | bf16 | contiguous | CUDA |
| `q_w`, `k_w` | `[64]` | bf16 | contiguous | CUDA |
| `positions` | `[>= M]`, one per row of `qkv` | int32 or int64 (bit-identical results, certified) | — | CUDA |
| `eps`, `rope_base` | 1e-5 and 10000.0 certified (Kimi K3's drafter) | Python float | — | — |
| `cache`, `page_table`, `ctx_len`, `out` | as `k3_drafter_attn` | | | |
| `num_heads`, `num_kv_heads` | 6 / 1 certified | Python int | — | — |

Certified splits `R x T`: 1x1, 1x8, 2x4, 4x2, 8x1, 3x1, 2x8, 8x8, context lengths 0 to 2041.

## Metadata consumed

None; the compile cache and its capture rule are `k3_drafter_attn`'s (the normalization is part of the compile
key). A captured call replays with `positions` rewritten in place (certified with `k3_drafter_attn`'s replays).

## Preconditions

`k3_drafter_attn`'s, plus: `q_w` / `k_w` bf16 `[64]` contiguous and `positions` int32 / int64 with at least `M`
entries, else `ValueError`.

## Notes

- Certified on GB200 (sm_100) against fp64 RMSNorm + NeoX RoPE rounded to bf16 (as `fused_qk_norm_rope` stores
  them), then the fp64 attention, within 1e-2 of the largest output magnitude.
- The op's own test also checks it against `fused_qk_norm_rope` followed by `k3_drafter_attn`.
