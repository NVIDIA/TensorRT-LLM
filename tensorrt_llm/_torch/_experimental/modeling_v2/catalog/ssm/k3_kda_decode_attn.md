---
receipts:
  sm_100: {status: passed, tests: 14}
---

# k3_kda_decode_attn

**Wraps** `torch.ops.trtllm.k3_kda_decode_attn` (one call), over a caller-owned `K3KdaBuffers`
(`catalog/ssm/k3_kda_buffers.py`), the same object `ssm/k3_kda_attn` takes.

Kimi K3's KDA layer for plain (non-speculative) decode, at the TP16 rank slice: the fused input projection of one
token of each of R <= 8 requests and the one-token KDA decode of `ssm/kda_decode` on it, in one launch.

## Semantics

`x` bf16 `[R, 7168]`, one token of each of R requests. One launch computes:

```
# 1. projection: k3_kda_attn's stream (26 clusters of 4 CTAs, three Lamport phases), rows >= R read as zero
y = x @ w^T              # q, k, f_a, v, og, b rounded to bf16 as in ssm/k3_kda_attn
# 2. per request r, on slot s = slots[r] (six head clusters of 4 CTAs, one V quarter per CTA):
g = bf16(f_a @ w_fb^T)
q, k, v = SiLU(conv4(conv[s] window, new raw))   # q, k L2-normalized, q *= scale
conv[s] = the window shifted by one: its last two raw inputs, then the new one
beta = sigmoid(b);  decay = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)))
S = ssm[s] * decay (per key);  S += beta (v - S k) k^T;  ssm[s] = S;  o = S q
out[r] = o * rsqrt(mean(o^2) + eps) * onorm_w * sigmoid(og)
```

and returns `out` bf16 `[R, 6, 128]`, the gated-norm core output. The decode is `ssm/kda_decode`'s arithmetic, row and
key layout. Certified against the model's unfused path (the projection stream alone, f_b as a bf16 `F.linear`, then
`ssm/kda_decode` on a copy of the pools) and a float64 decode, both at fp32 tolerance for the output (2e-2 relative,
bf16 outputs) and the state rows (1e-3); the conv windows bit for bit (raw bf16 inputs). Repeated runs are
bit-identical.

## Signature

```python
def k3_kda_decode_attn(
    x: torch.Tensor,
    w: torch.Tensor,
    w_fb: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_w: torch.Tensor,
    conv: torch.Tensor,
    ssm: torch.Tensor,
    slots: torch.Tensor,
    buffers: K3KdaBuffers,
    lower_bound: float,
    scale: float,
    eps: float,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[R, 7168]`, 1 <= R <= 8 | bf16 | contiguous | CUDA |
| `w` | `[3208, 7168]` (as `ssm/k3_kda_attn`) | bf16 | contiguous | CUDA |
| `w_fb` | `[768, 128]` | bf16 | contiguous | CUDA |
| `w_q`, `w_k`, `w_v` | `[768, 4]` conv taps, oldest input first | fp32 | dense | CUDA |
| `a_log` | `[6]` | fp32 | dense | CUDA |
| `dt_bias` | `[768]` | fp32 | dense | CUDA |
| `onorm_w` | `[128]` | fp32 | dense | CUDA |
| `conv` | `[slots, 2304, 3]`: q, k, v channels, the last three raw inputs oldest first | bf16 | each slot dense; slot stride a multiple of 8 elements; 16-byte aligned base | CUDA |
| `ssm` | `[slots, 6, 128, 128]` (V rows, K contiguous) | fp32 | each slot dense; slot stride a multiple of 4; 16-byte aligned base | CUDA |
| `slots` | `[R]` | int32 | contiguous | CUDA |
| `buffers` | `K3KdaBuffers` made with `ctas=FUSED_CTAS` (128) | — | — | CUDA |
| `lower_bound`, `scale`, `eps` | scalars (Kimi K3: -5.0, 128^-0.5, 1e-5) | Python float | — | — |
| returns | `[R, 6, 128]` | bf16 | contiguous, fresh | CUDA |

`mutates_args`: `conv`, `ssm`, and the set's `p1`, `part`, `epoch`.

## State

**Object.** `K3KdaBuffers`, shared with `ssm/k3_kda_attn`: one per device for both ops and every KDA layer. Its
contents, creation, re-arming, the 128-CTA requirement and the call-order invariant (launches on one set run one at a
time, in stream order; each launch moves every CTA's index once) are in `k3_kda_attn.md`'s `## State`. Certified here:
plain-decode launches and `ssm/k3_kda_attn` verify launches interleaved on one set give the bits of the same launches
on sets of their own; layers x steps on one set, a set per layer and two sets alternating agree bit for bit; the index
preset to 2^31 - 2 gives the bits of a fresh set and stays in 0..2; a 104-CTA set raises `ValueError` before anything
is written; `create` under capture raises `RuntimeError`.

**The pools.** The cache manager's plain-decode states, not part of the object: with the V2 hybrid manager
(`conv_state_layout="q_k_v"`, bf16 conv states, fp32 SSM states), layer `l`'s `get_conv_states(l)` (`conv`) and
`get_ssm_states(l)` (`ssm`), their slots strided by the manager's per-slot coalescing; `slots` from
`get_state_indices`. A launch on layer `l` writes only layer `l`'s views, at its slots: certified on a real manager,
with the other slots of the layer and every slot of the other layers bit-unchanged. A batch names each slot once.

**The PDL rule.** As `ssm/k3_kda_attn`: the head CTAs read the slots' pools (the conv windows, 32 state rows per CTA)
before their grid-dependency wait, so a launch must not directly follow another launch on the same pools in the
stream. A kernel that waits, or a non-PDL kernel, sits between them in the model; one launch of this op (or of
`ssm/k3_kda_attn`) on other pools is also enough, since its dependents launch only after its own wait.

**Why the test drives call sequences.** Each step's state and conv window are the next step's input, so the test
runs layers x steps of a real manager, and captures one step of every layer once and replays it with rewritten
inputs: the replays give the bits of the same steps run eagerly on a copy of the pools.

**What a wrong order does.** Two steps of one request in swapped order (the negative control): nothing raises, and the
second step's output and the final state are those of a different history. Measured on sm_100: the second step's output is off by 0.81 and the state by
0.89 (the largest absolute difference over the in-order result's largest magnitude).

## Metadata consumed

`slots`, read before the grid-dependency wait: under CUDA graphs it must be written before the graph runs.

## Preconditions

- sm_100: the kernel uses tcgen05, TMA and clusters.
- The op checks `x` (rank 2, 1..8 rows, 7168 columns, bf16, contiguous), `w`, `w_fb`, `conv` (bf16, `[*, 2304, 3]`,
  strides `(*, 3, 1)`, slot stride a multiple of 8, 16-byte base), `ssm` (fp32, `[*, 6, 128, 128]`, dense slots, slot
  stride a multiple of 4, 16-byte base), `slots` (R int32 elements, contiguous), the set's sizes (`epoch` of 128) and
  the fp32 dtypes, and raises `ValueError` before a launch. The fp32 weights' shapes and the slots' range are the
  caller's obligation. The V2 hybrid manager's per-layer views meet the layout (certified on a real manager).
- The first call of each configuration (`lower_bound`, `scale`, `eps` and whether PDL is on) compiles the kernel and
  must run outside CUDA-graph capture; under capture it raises `RuntimeError`. PDL follows `TRTLLM_ENABLE_PDL`
  (default on), read at every call.

## Notes

- The pools are addressed at 64-bit slot offsets, so pools past 2 GiB are fine (the op's own test,
  `test_k3_kda_pools_past_2g.py`). `slots` may be a slice of a longer index tensor starting at any element; the op's
  own test certifies offsets 0..3.
- Within the fp32 tolerances above, not bitwise, against `ssm/kda_decode`: the two read the same rows but sum the
  decode in their own orders.
