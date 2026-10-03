---
receipts: {}
---

# causal_conv1d_fwd

**Wraps** `torch.ops.trtllm.causal_conv1d_fwd` (one call).

## Semantics

Causal depthwise 1-D convolution over a batch of sequences (the short
convolution of Mamba-style and KDA layers in prefill), each sequence started
from its slot's conv state, which the call then advances. Per channel `c`
and sequence `b` of `L_b` tokens, with `W = weight.shape[-1]` taps and
`hist_b` the `W - 1` previous inputs (the slot's state if
`has_initial_state[b]`, zeros otherwise):

```
full  = concat(hist_b, x[c, tokens of b])                       # W - 1 + L_b inputs
y[t]  = sum_{j=0}^{W-1} weight[c, j] * full[t + j] + bias[c]   # fp32, oldest tap first
y[t]  = y[t] * sigmoid(y[t])                                    # only if silu_activation
out[c, tokens of b] = y cast to x.dtype
conv_states[cache_indices[b], c, :] = full[-(W - 1):]          # the last W - 1 inputs, raw
```

The new state is a copy of raw inputs (or of kept history when `L_b < W - 1`),
so it is bit-exact; the outputs are fp32 sums rounded once.

A sequence whose `cache_indices[b] == pad_slot_id` (a CUDA-graph padding
row) is skipped entirely: its columns of `x` / `out` and every slot are left
as they were.

`x` holds either a varlen batch (`query_start_loc` given: `x` is
`[dim, total_tokens]`, sequences concatenated along the token axis) or an
equal-length batch (`x` is `[batch, dim, seqlen]`). Its layout picks the
kernel:

- **channel-major** (tokens contiguous, `x.stride(-1) == 1`): the result is
  written back into `x` unless `out` is given (same layout as `x`);
- **channel-last** (channels contiguous, a token-major activation passed as a
  transposed view): `out` must be given, channel-last, and must not overlap
  `x`; `x` is left untouched.

Kimi K3's KDA layer calls it in prefill on its packed `q | k | v` projection
(`[3 * 128 * heads, tokens]`, channel-major, in place), `W = 4`, SiLU, no
bias, with the cache manager's conv pool.

Fusion boundary: the convolution, bias and SiLU only. The projection that
produces `x` and everything after it are the caller's.

## Signature

```python
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
    out: Optional[torch.Tensor] = None,
) -> None
```

The wrapper mirrors the op schema argument for argument (schema name
`bias_` for `bias`).

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | varlen `[dim, total_tokens]`, or `[batch, dim, seqlen]` | bf16 / fp16 / fp32 | channel-major or channel-last (see above) | CUDA |
| `weight` | `[dim, W]`, `2 <= W <= 4` | as `x` (bf16 / fp16 / fp32) | contiguous | CUDA |
| `bias` | `[dim]` or None | `weight.dtype` | last dim dense | CUDA |
| `conv_states` | `[slots, dim, >= W - 1]` or None | `x.dtype` | the width axis dense (the channel-major kernel reads it with unit stride); any slot and channel strides (the slot stride may exceed `dim * (W - 1)`) | CUDA |
| `query_start_loc` | `[batch + 1]` cumulative token offsets, or None (equal-length batch) | int32 | — | CUDA |
| `cache_indices` | `[batch]` slots | int32 | — | CUDA |
| `has_initial_state` | `[batch]` | bool | — | CUDA |
| `silu_activation` | scalar | Python bool | — | — |
| `pad_slot_id` | scalar (`PAD_SLOT_ID = -1` in this repo) | Python int | — | — |
| `out` | `x.shape` or None | `x.dtype` | see above | same device as `x` |

Returns None. Mutates `x` (channel-major, no `out`) or `out`, and
`conv_states` at the slots the batch names.

## State

**Object.** None of its own (stateful kind P): the op advances the caller's
conv pool in place, one slot per sequence. Nothing persists in the op
between calls.

**The pool.** The cache manager's per-layer conv states (the V2 hybrid
manager's conv pool for a KDA layer, bf16), its slots possibly strided wider
than one state. A call writes exactly the slots in `cache_indices` (other
slots and the slot padding stay bit-unchanged, certified).

**Call-order invariant.** A slot carries one sequence's conv history: chunked
prefill of a sequence must call chunk after chunk, each with
`has_initial_state = True` after the first, and no other sequence may use
the slot in between. Then the chunks give the bits of one call over the
whole sequence (certified at 64 + 1 + 116 tokens).

**What a wrong order does.** Chunks in the wrong order raise nothing and
return the wrong numbers (the negative control): the second call convolves
across a history it should not see.

## Metadata consumed

`query_start_loc`, `cache_indices`, `has_initial_state` (built by the
caller from its batch). No attention metadata.

## Preconditions

Each is the op's check (`RuntimeError`) unless noted:

- `x`, `weight` each bf16 / fp16 / fp32; `bias` of `weight.dtype` with a
  dense last dim; `conv_states` of `x.dtype`; `query_start_loc` and
  `cache_indices` int32; `has_initial_state` bool; all on CUDA.
- The table's shapes for `x`, `weight` (exactly `[dim, W]`), `bias`,
  `has_initial_state`, `cache_indices` and `out`.
- Channel-last `x` needs a channel-last `out` that does not overlap it.
- Not checked:
  - `weight.dtype == x.dtype` (the kernels read `weight` as `x.dtype`);
  - `2 <= W <= 4` (the kernels are instantiated for these widths);
  - `conv_states`'s shape and its dense width axis, and slots inside the
    pool; `query_start_loc`'s shape;
  - a `conv_states` given whenever some `has_initial_state[b]` is True
    (without one the kernel dereferences a null pointer);
  - an `x` in neither layout, which runs the channel-major kernel.

## Notes

- **Certified surface** (sm_100): Kimi K3's widths `dim` 2304 and 9216
  (TP16 / TP4), `W = 4`, SiLU, bf16, varlen batches mixing fresh sequences
  and continuations with lengths 1, 2, 3, 17, 64, 300 over a padded pool;
  chunked continuation and its negative control; a `PAD_SLOT_ID` sequence;
  bias with and without SiLU at `dim` 768; the channel-last path with an
  explicit `out`; CUDA-graph capture and replay; six of the checks above,
  each raising.
- **Numerics.** Outputs within one bf16 ulp of an fp32 reference (the test
  gates at `rtol = 2^-7`, `atol = 1e-2`); states bit-exact.
- `tensorrt_llm._torch.modules.mamba.causal_conv1d.causal_conv1d_fn` is the
  Python form; it allocates a channel-last `out` when `x` is channel-last.
