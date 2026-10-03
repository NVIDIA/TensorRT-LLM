---
receipts: {}
---

# kda_decode

**Wraps** `torch.ops.trtllm.kda_decode` (one call).

One-token KDA (Kimi Delta Attention) decode of B requests: the causal depthwise conv over each request's conv window,
the gated delta rule on its recurrent state, and the gated RMS norm, with the state (and optionally the conv windows)
updated in place at each request's slot. Certified here at Kimi K3's TP16 rank slice (6 heads, K = V = 128, conv
width 4) on a real cache manager's pools.

## Semantics

For request b (slot `s = ssm_state_indices[b]`), head h, with `x_q`, `x_k`, `x_v` the new raw q, k, v inputs:

```
q, k, v = SiLU(conv4(window[s], new raw) + bias)   # taps w_*_t[j], oldest input first; bias_* added before SiLU
q = q / sqrt(sum(q^2) + 1e-6) * scale;  k = k / sqrt(sum(k^2) + 1e-6)
beta = sigmoid(beta)
decay = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)))
S = state[s] * decay (per key);  S += beta (v - S k) k^T;  state[s] = S;  o = S q
output = o * rsqrt(mean(o^2) + onorm_eps) * onorm_weight * sigmoid(onorm_g)
```

`apply_onorm`, `use_lower_bound` and `apply_beta_sigmoid` must all be True: the op supports only that combination
(*Preconditions*), which is the one above and Kimi K3's.

With `update_conv_cache` the conv windows at the slot shift by one (the last two raw inputs, then the new one). The
float64 reference of the test is this arithmetic; the op matches it within fp32 tolerance (outputs 2e-2 relative,
bf16; state rows 1e-3), and the conv windows bit for bit (raw bf16 inputs).

**Kernels.** On sm_100 and sm_103 the dispatcher picks by the workload B x H: up to 32 the four-CTA cluster kernel, up
to 144 the legacy compact-heads kernel, above that per-architecture choices among the optimized single-CTA and bulk
kernels and the legacy many-heads kernel. At Kimi K3's 6 heads: B <= 5 runs the cluster kernel and 6 <= B <= 24 the
legacy compact-heads kernel (whose block reduction #19830 orders with `__syncwarp`). The test runs B = 1..8, so both.
Other architectures run the legacy kernels.

## Signature

```python
def kda_decode(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_q_t: torch.Tensor,
    w_k_t: torch.Tensor,
    w_v_t: torch.Tensor,
    bias_q: torch.Tensor,
    bias_k: torch.Tensor,
    bias_v: torch.Tensor,
    conv_state_q: torch.Tensor,
    conv_state_k: torch.Tensor,
    conv_state_v: torch.Tensor,
    a_log: torch.Tensor,
    g: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    ssm_state_indices: Optional[torch.Tensor],
    state: torch.Tensor,
    apply_onorm: bool,
    update_conv_cache: bool,
    use_lower_bound: bool,
    apply_beta_sigmoid: bool,
    lower_bound: float,
    scale: float,
    onorm_eps: float,
    output: torch.Tensor,
) -> None
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x_q`, `x_k`, `x_v` | `[1, B, H, 128]` | bf16 | head and channel axes packed; any row stride | CUDA |
| `w_q_t`, `w_k_t`, `w_v_t` | `[4, H * 128]` conv taps, oldest input first | bf16 | contiguous | CUDA |
| `bias_q`, `bias_k`, `bias_v` | `[H * 128]` (zeros: no bias) | bf16 | contiguous | CUDA |
| `conv_state_q/k/v` | with `update_conv_cache`: `[slots, H * 128, 3]` section views of one packed `[slots, 3 H 128, 3]` pool (`q | k | v`); else `[B, H * 128, 3]` | bf16 | packed: equal slot strides >= `3 H 128 * 3`, strides `(*, 3, 1)`; else contiguous | CUDA |
| `a_log` | `[H]` | fp32 | contiguous | CUDA |
| `g` | `[1, B, H, 128]` (the gate before `dt_bias`) | bf16 | head and channel axes packed | CUDA |
| `dt_bias` | `[H * 128]` | fp32 | contiguous | CUDA |
| `beta` | `[1, B, H]` | bf16 | head axis stride 1 | CUDA |
| `onorm_g` | `[1, B, H, 128]` (the output gate) | bf16 | head and channel axes packed | CUDA |
| `onorm_weight` | `[128]` | fp32 | contiguous | CUDA |
| `ssm_state_indices` | `[B]` slots, or None (state rows 0..B-1) | int32 | contiguous | CUDA |
| `state` | `[slots >= B, H, 128, 128]` (V rows, K contiguous) | fp32 | each slot dense; slot stride a multiple of 4; 16-byte aligned base | CUDA |
| `apply_onorm`, `update_conv_cache`, `use_lower_bound`, `apply_beta_sigmoid` | scalars (Kimi K3: all True) | Python bool | — | — |
| `lower_bound`, `scale`, `onorm_eps` | scalars (Kimi K3: -5.0, 128^-0.5, 1e-5) | Python float | — | — |
| `output` | `[B, 1, H, 128]` | bf16 | contiguous | CUDA |

H is one of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96 and the same for q / k and v. Returns None; writes `output`.
Schema mutations (`Tensor(a!)`): `conv_state_q`, `conv_state_k`, `conv_state_v`, `state`, `output`.

## State

**Object.** None of its own (stateful kind P): the op updates the caller's pools in place, the recurrent `state` at
each request's slot and, with `update_conv_cache`, the conv windows there. Nothing persists in the op between calls.

**The pools.** The cache manager's: with the V2 hybrid manager (`conv_state_layout="q_k_v"`, bf16 conv states, fp32
SSM states), layer `l`'s `get_conv_states(l)` split into its q, k, v sections and `get_ssm_states(l)`, their slots
strided by the manager's per-slot coalescing; `ssm_state_indices` from `get_state_indices`. A call on layer `l` writes
only layer `l`'s views, at its slots: certified on a real manager, with the other slots of the layer and every slot of
the other layers bit-unchanged.

**Call-order invariant.** One call per layer per decode step, in step order; a batch names each slot once. A call
reads the state and conv window that the previous step's call on that layer left at the slot.

**Why the test drives call sequences.** Each step's state is the next step's input, so the test runs layers x steps
on a real manager and captures a step of every layer once, replaying it with rewritten inputs: the replays give the
bits of the same steps run eagerly on a copy of the pools.

**What a wrong order does.** Two steps of one request in swapped order (the negative control): nothing raises, and the
second step's output and the final state are those of a different history. Measured at Kimi K3's shape on sm_100: the second step's output is off by 1.38 and the state
by 1.27 (the largest absolute difference over the in-order result's largest magnitude).

## Metadata consumed

`ssm_state_indices` (the requests' slots). None of the attention metadata.

## Preconditions

- Every check in the table is the op's (`TORCH_CHECK`): a violation raises `RuntimeError` before the launch, with the
  pools unchanged. Among them: the state base must be 16-byte aligned and its slot stride a multiple of 4 floats (the
  kernels move state with 16-byte accesses at `slot * stride(0)`); `ssm_state_indices` must be int32.
- `apply_onorm`, `use_lower_bound` and `apply_beta_sigmoid` must all be True, on every architecture and batch size:
  the op checks them first and otherwise raises `RuntimeError` ("KDA decode only supports apply_onorm=true,
  use_lower_bound=true, and apply_beta_sigmoid=true") before the launch. Certified with the lower bound off at B = 2,
  H = 6, the pools unchanged.
- The conv inputs, gates and outputs are bf16 and the state fp32; K = V = 128 and conv width 4 only.

## Notes

- Kimi K3's KDA layer calls this op through `run_kda_decode_fusion_cuda`
  (`_torch/modules/kimi_kda/_kda_decode.py`), which fills the optional bias, gate and norm arguments with cached
  dummies; this entry exposes the schema raw.
- `ssm/k3_kda_decode_attn` computes the same decode fused with the layer's projection, for up to 8 requests.
