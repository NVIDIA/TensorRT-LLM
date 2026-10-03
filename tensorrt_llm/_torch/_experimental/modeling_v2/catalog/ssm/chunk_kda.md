---
receipts: {}
---

# chunk_kda

**Wraps** `fla.ops.kda.chunk_kda` (one call; flash-linear-attention's Triton
implementation, imported on the first call).

## Semantics

Chunked KDA (Kimi Delta Attention: a gated delta rule with a per-key decay)
prefill of a batch of sequences, each started from its row of
`initial_state`; with `output_final_state=True` it also returns each
sequence's state after its last token. It computes the recurrence of
`ssm/kda_prefill`: per sequence and head, with `S` the `[V, K]` state
(`state_v_first=True`), `q`, `k` already L2-normalized by the caller, and
Kimi K3's configuration (`use_gate_in_kernel=True`, `safe_gate=True`,
`use_beta_sigmoid_in_kernel=True`):

```
S     = initial_state[b, h]                                  # or zeros when initial_state is None
for t in the sequence:
    decay = exp(lower_bound * sigmoid(exp(A_log[h]) * (g[t, h] + dt_bias[h])))   # per key
    S     = S * decay                                        # column k scaled by decay[k]
    S     = S + sigmoid(beta[t, h]) * (v[t, h] - S k[t, h]) k[t, h]^T
    o[t, h] = S (scale * q[t, h])
final_state[b, h] = S
```

The kernels evaluate this in chunks of 64 tokens with bf16 operands and fp32
accumulation, carrying the state across chunks in fp32 (the copies the output
reads are bf16); `o` comes back in `q.dtype`, `final_state` in fp32.

Kimi K3's KDA layer runs this entry for a prefill batch of fewer than four
64-token chunks in all (its dispatch sends larger batches to
`ssm/kda_prefill`); it gathers the batch's pool rows into a dense fp32
`initial_state` (fresh rows zeroed, or `None` when no sequence continues) and
writes `final_state` back into the pool.

Fusion boundary: the gate activation, the beta sigmoid and the chunked delta
rule. The projections, the conv, the q / k normalization (`ssm/kda_post_conv`)
before it, the pool gather / scatter around it, and the gated output norm
after it are the caller's.

## Signature

```python
def chunk_kda(
    q, k, v, g, beta,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `q`, `k` | `[1, T, H, 128]` (varlen) | bf16, L2-normalized per head | contiguous | CUDA |
| `v` | as `q` | bf16 | contiguous | CUDA |
| `g` | as `q` (the raw gate) | bf16 | contiguous | CUDA |
| `beta` | `[1, T, H]` (raw) | fp32 | contiguous | CUDA |
| `scale` | scalar (Kimi K3: `128 ** -0.5`) | Python float | — | — |
| `initial_state` | `[num_sequences, H, 128, 128]` (V-first) or None | fp32 | dense | CUDA |
| `output_final_state` | Kimi K3: True | Python bool | — | — |
| `use_qk_l2norm_in_kernel` | Kimi K3: False | Python bool | — | — |
| `use_gate_in_kernel`, `use_beta_sigmoid_in_kernel`, `safe_gate` | Kimi K3: True | Python bool | — | — |
| `lower_bound` | Kimi K3: -5.0 | Python float | — | — |
| `state_v_first` | Kimi K3: True | Python bool | — | — |
| `cu_seqlens` | `[num_sequences + 1]` | int64 (the layer's `query_start_loc_long`) | — | CUDA |
| `A_log` | `[H]` | fp32 | contiguous | CUDA |
| `dt_bias` | `[H * 128]` | fp32 | contiguous | CUDA |
| returns `o` | `v.shape` | `q.dtype` | new tensor | same device |
| returns `final_state` | as `initial_state` (None unless `output_final_state`) | fp32 | new tensor | same device |

`initial_state` is not written (certified).

The wrapper passes the inference arguments of `fla.ops.kda.chunk_kda` through
by name. It does not expose FLA's training and context-parallel arguments
(`disable_recompute`, `cp_context`, `cu_seqlens_cpu`), its intermediate-state
return (`return_intermediate_states`), `allow_neg_eigval` or `chunk_size` (64).

## Metadata consumed

`cu_seqlens` only.

## Preconditions

- Checked by FLA (`ValueError`): a batch of one row when `cu_seqlens` is
  given; one `initial_state` row per sequence; with `safe_gate` and
  `use_gate_in_kernel`, a `lower_bound` in `[-5, 0)`.
- Checked by FLA (`AssertionError`): an fp32 `initial_state`; `A_log` given
  when `use_gate_in_kernel`; `q` / `k` / `g` / `beta` shapes; `K <= 256`.
- flash-linear-attention installed (`requirements-dev.txt` pins it; Kimi K3's
  KDA module imports `fla` when it loads). The wrapper imports it on its first
  call, so importing the catalog does not need it.
- Run under `torch.inference_mode()` (as the layer is): the call is an autograd
  function and would otherwise keep its activations for a backward pass.

## Notes

- **Certified surface** (sm_100): heads 6 (TP16) and 24 (TP4); varlen
  batches below four chunks (lengths 100 + 37, 150, 1, 17 + 5 + 64) mixing
  fresh (zero) and continuing states; a 128 + 64 continuation against one
  192-token call and its negative control; agreement with `ssm/kda_prefill`
  on a four-chunk batch either path can take; `initial_state=None` against
  zero states; a repeated call; four of FLA's `ValueError` checks above and
  its fp32 `initial_state` assertion.
- **Numerics.** Against an fp64 token-by-token reference, `o` is within
  `2e-2` of each sequence's largest output and `final_state` within `1e-2` of
  its largest entry (measured: at most 6.5e-3 and 4.3e-3 over sequences of 1
  to 191 tokens, 6 and 24 heads). Against `ssm/kda_prefill` on the same
  four-chunk batch: within the same bounds (measured 6.4e-3 and 4.7e-3; at
  most 8.2e-3 and 5.0e-3 over three batches). The 128 + 64 continuation is
  within `2e-3` / `1e-2` of the one call (measured identical).
- FLA's Triton kernels autotune once per process. Within a process a repeated
  call is bit-identical (certified); across processes the picks can differ,
  which is why Kimi K3's layer stages a bf16 pool through fp32 rows for
  `ssm/kda_prefill` rather than falling back here (its own note).
