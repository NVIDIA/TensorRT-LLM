---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 9}
---

# fused_qk_norm_rope

**Wraps** `torch.ops.trtllm.fused_qk_norm_rope` (one call).

## Semantics

In-place, on a packed QKV activation `qkv` of shape
`[num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]` whose
rows lay out `num_heads_q` query heads, then `num_heads_k` key heads, then
`num_heads_v` value heads, each `head_dim` wide. One call computes, per
token, for every **q and k head** `x` (fp32 internally, one final cast back
to bf16):

```
# 1. per-head RMS norm over the full head_dim (skipped if is_qk_norm=False)
h = x / sqrt(mean(x^2) + eps) * w          # w = q_weight or k_weight
                                           # use_gemma: * (1 + w) instead
# 2. RoPE on the first rotary_dim dims of h; h[rotary_dim:] passes through
inv_freq[j] = base^(-2j/rotary_dim)                    for j in [0, rotary_dim/2)
angle[j]    = position_id * inv_freq[j]
cos, sin    = cos(angle) * attention_factor, sin(angle) * attention_factor
# is_neox=True  — half-split pairs (x1, x2) = (h[j], h[j + rotary_dim/2])
# is_neox=False — interleaved pairs (x1, x2) = (h[2j], h[2j+1])
(x1, x2)   -> (x1*cos - x2*sin, x1*sin + x2*cos)
```

Value heads are passed through bit-exactly untouched.

YaRN scaling (`factor`, `low`, `high`, `attention_factor`) replaces
`inv_freq` with the blend

```
ramp[j]     = clamp((j - low) / (high - low), 0, 1)   # high==low: high += 0.001
inv_freq[j] = (1 - ramp[j]) * base^(-2j/rotary_dim)
            + ramp[j] * base^(-2j/rotary_dim) / factor
```

which reduces to plain RoPE at `factor=1.0` (the `low`/`high` values are
then irrelevant). `attention_factor` scales cos/sin — but **not at
`factor == 1.0` exactly**, where any `attention_factor != 1.0` raises
`Assertion failed: attention_factor == 1.0f`
(`fusedQKNormRopeKernel.cu:322`). `factor = 1.0000001` accepts it.
Measured 2026-07-28; the gate is on `factor` alone and is not otherwise
documented.

Interleaved mRoPE (`use_mrope=True`) takes 3 position rows
(temporal/height/width) per token. Frequency index `j` reads its position
from row 1 when `j % 3 == 1 and j < 3*mrope_section1`, from row 2 when
`j % 3 == 2 and j < 3*mrope_section2`, and from row 0 otherwise. Only this
interleaved layout is implemented — not the contiguous-section mRoPE
variant.

Fusion boundary: q-norm + k-norm + RoPE, nothing else. The caller owns the
QKV projection before the call and everything after it (KV-cache append,
attention, output projection). A sibling op `fused_dit_qk_norm_rope`
exists for the DiT-style variant of this fusion.

## Signature

```python
def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
    is_qk_norm: bool = True,
    use_gemma: bool = False,
    use_mrope: bool = False,
    mrope_section1: int = 0,
    mrope_section2: int = 0,
) -> None
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `qkv` | `[num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]` (2D only) | bf16 | contiguous | CUDA |
| `num_heads_q/k/v` | scalar | Python int | — | — |
| `head_dim` | scalar, one of 64 / 128 / 256 | Python int | — | — |
| `rotary_dim` | scalar, even, `<= head_dim` | Python int | — | — |
| `eps` | scalar | Python float | — | — |
| `q_weight`, `k_weight` | `[head_dim]` | bf16 | contiguous | CUDA (same device) |
| `base` | scalar (RoPE theta) | Python float | — | — |
| `is_neox` | scalar | Python bool | — | — |
| `position_ids` | `[num_tokens]`, or `[3, num_tokens]` row-major when `use_mrope` | int32 | contiguous | CUDA (same device) |
| `factor`, `low`, `high`, `attention_factor` | scalar (YaRN; defaults = plain RoPE) | Python float | — | — |
| `is_qk_norm` | scalar (False: RoPE only, weights ignored) | Python bool | — | — |
| `use_gemma` | scalar (True: scale by `1 + w`) | Python bool | — | — |
| `use_mrope`, `mrope_section1`, `mrope_section2` | scalar (interleaved mRoPE) | Python bool / int | — | — |
| returns | — | — | — | — |

Returns `None`. `qkv` is mutated in place: q and k head slices are
overwritten with the normed + rotated values; v head slices are untouched.

## Metadata consumed

None. Stateless — all RoPE frequencies are derived on the fly from the
scalar arguments; no cos/sin cache and no attention metadata are read.
`position_ids` are the absolute positions of the `num_tokens` tokens, in
row order of `qkv` (batch layout is the caller's concern; the kernel is
purely per-token).

## Preconditions

- `qkv` is 2D, contiguous, bf16, on CUDA, with
  `qkv.shape[1] == (num_heads_q + num_heads_k + num_heads_v) * head_dim`.
  bf16 is the only accepted dtype (fp16/fp32 raise).
- `head_dim` is one of 64, 128, 256. Others (32, 80, 96, 192 verified)
  raise `Unsupported head dimension for fusedQKNormRope`.
- `q_weight` and `k_weight` are contiguous bf16 `[head_dim]` tensors on
  the same device; they must be passed (and valid) even when
  `is_qk_norm=False`, though their values are then unused.
- `position_ids` is contiguous int32 on the same device: flat
  `[num_tokens]` normally, `[3, num_tokens]` row-major when
  `use_mrope=True`. int64 raises. A 3D upstream view must be reshaped and
  made contiguous by the caller.
- `rotary_dim` is even and at most `head_dim`; dims beyond `rotary_dim`
  are normed (if `is_qk_norm`) but not rotated. **Only the evenness half
  is enforced** (`Assertion failed: rotary_dim must be even`);
  `rotary_dim > head_dim` is accepted silently and rewrites the q and k
  head columns — measured at 160 and 256 against `head_dim = 128`,
  2026-07-28. This is the caller's obligation, not the op's.
- Everything else above is validated by the op with a clear error, and in
  every raising case `qkv` comes back bit-identical to its pre-call
  contents. The two exceptions are the `rotary_dim` bound just named and
  the `attention_factor` gate under *Semantics*.

## Notes

- The kernel checks dtype/shape/contiguity eagerly (thop-level asserts),
  so contract violations fail loudly rather than corrupting `qkv`.
- Accuracy: internal math is fp32 with a single bf16 round at the end.
  Because RoPE rotates a bf16-rounded pair, the per-element error is
  absolute in the pair magnitude — with unit-scale inputs, max abs error
  observed on sm_100 was 0.031, at most 78% of the test's
  `rtol=1.6e-2, atol=2e-2` allowance (compare with `atol` sized to the q/k
  magnitude, not the default bf16 `atol=1e-5`). The ceiling sits on the
  neox-prefill and Gemma-norm cases; most of the file sits at 0.008-0.016.
- `low`/`high` are float half-dim indices in `[0, rotary_dim/2)`
  (fractional values legal, matching YaRN's `truncate=False`).
- TRT-LLM's own caller derives `factor/low/high/attention_factor` from a
  YaRN config and uses `rotary_dim = head_dim * partial_rotary_factor`;
  this entry exposes them raw.
