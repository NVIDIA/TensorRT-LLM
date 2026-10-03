---
receipts:
  sm_100: {status: passed, tests: 7}
---

# k3_kda_verify

**Wraps** `torch.ops.trtllm.k3_kda_verify` (one call).

Kimi K3's KDA speculative verify of N requests of 1 + num_spec tokens (a golden token and its drafts), from the fused
projection rows to the gated-norm core output, committing the state after every verify token so that the next round
starts from the drafts the sampler accepted instead of replaying them.

## Semantics

`proj` bf16 `[N (1 + num_spec), cols]` holds each request's 1 + num_spec rows of the fused projection
`[q | k | v | og | f_a | b | pad]` (H K, H K, H K, H K, K, H columns, then padding). Grid (H, N, 8): a cluster of 8
CTAs per (head, request), each owning 16 V rows. For request n on slot `s = slots[n]`, with `P = pending[s]`:

```
S = ssm[s] if P == 0 else state_tok[s, P - 1]                 # the state after the last accepted token
window = conv caches cs_*[s] columns P..P+2                    # raw inputs at positions -3..-1
per token t = 0..num_spec:
  g = bf16(f_a[t] @ w_fb^T)     (or g_ext[t]: the unfused f_b output)
  q, k, v = SiLU(conv4(window, raw[t])); q, k L2-normalized (q *= scale)
  beta = sigmoid(b[t]);  decay = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)))
  S *= decay (per key);  S += beta (v - S k) k^T;  o = S q
  out[t] = o * rsqrt(mean(o^2) + eps) * onorm_w * sigmoid(og[t])
ssm[s] = the state after t = 0 (the golden token);  state_tok[s, t - 1] = the state after draft t
cs_*[s] = the raw inputs at positions -2..num_spec around the golden token
```

Returns `out` bf16 `[N (1 + num_spec), H, 128]`. The recurrence is `kda_mtp_decode`'s V-split arithmetic over the
verify tokens, unrolled: with `g_ext` the op is bit-exact against `kda_mtp_decode` fed the same gate and replaying the
same accepted drafts (the op's own test). Certified here against a float64 verify over each request's committed
history: every round's outputs within 2e-2 relative (bf16) and the committed states within 1e-3, the per-draft states
and conv caches through the next round, which starts from them. Repeated runs, and graph replays against eager runs,
are bit-identical.

## Signature

```python
def k3_kda_verify(
    proj: torch.Tensor,
    w_fb: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_w: torch.Tensor,
    cs_q: torch.Tensor,
    cs_k: torch.Tensor,
    cs_v: torch.Tensor,
    ssm: torch.Tensor,
    state_tok: torch.Tensor,
    slots: torch.Tensor,
    pending: torch.Tensor,
    num_spec: int,
    lower_bound: float,
    scale: float,
    eps: float,
    g_ext: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `proj` | `[N (1 + num_spec), cols]`, cols a multiple of 8 and >= 4 H K + K + H | bf16 | contiguous | CUDA |
| `w_fb` | `[H K, K]` (f_b, out x in) | bf16 | contiguous | CUDA |
| `w_q`, `w_k`, `w_v` | `[H K, 4]` conv taps, oldest input first | fp32 | dense | CUDA |
| `a_log` | `[H]` | fp32 | dense | CUDA |
| `dt_bias` | `[H K]` | fp32 | dense | CUDA |
| `onorm_w` | `[K]` | fp32 | dense | CUDA |
| `cs_q`, `cs_k`, `cs_v` | `[pool, H K, 3 + num_spec]` | fp32 | channel stride 1 (dim-contiguous) | CUDA |
| `ssm` | `[pool, H, K, K]` (V rows, K contiguous) | fp32 | each slot dense, slots at any stride | CUDA |
| `state_tok` | `[pool, num_spec, H, K, K]` | fp32 | contiguous | CUDA |
| `slots` | `[N]` | int32 | any element offset | CUDA |
| `pending` | `[pool]`: drafts the sampler accepted last round, per slot | int32 | any element offset | CUDA |
| `num_spec` | drafts per request | Python int | — | — |
| `lower_bound`, `scale`, `eps` | scalars (Kimi K3: -5.0, 128^-0.5, 1e-5) | Python float | — | — |
| `g_ext` | `[N (1 + num_spec), H K]` or None (fold f_b into the kernel) | bf16 | contiguous | CUDA |
| returns | `[N (1 + num_spec), H, K]` | bf16 | contiguous, fresh | CUDA |

K = V = 128, conv width 4. Kimi K3's TP16 rank slice is H = 6, num_spec = 7. `mutates_args`: `cs_q`, `cs_k`, `cs_v`,
`ssm`, `state_tok`.

## State

**Object.** None of its own (stateful kinds P and R): the op updates the caller's pools in place and reads the
record the sampler wrote after the previous round. No buffer persists in the op.

**The pools and the record.** The cache manager's: with the V2 hybrid manager built with the KDA replay caches and
`kda_token_states` (`kda_replay_num_spec` = num_spec), layer `l`'s `mamba_layer_cache(l).kda_conv_q` / `kda_conv_k`
/ `kda_conv_v` (`cs_*`), `get_ssm_states(l)` (`ssm`, its slots strided by the manager's per-slot coalescing) and
`mamba_layer_cache(l).kda_state_tok`. `pending` is the manager's `prev_num_accepted_tokens`: one record shared by
every layer, written between steps by the sampler's acceptance. A call on layer `l` writes only layer `l`'s views, at
its slots: certified on a real manager, the other slots bit-unchanged.

**Call-order invariant.** Per layer, one call per verify round, in round order, with `pending` updated between rounds
(after the sampler accepts) and not during one. A call reads what the previous round's call on that layer left at the
slot: the state after the last accepted token (`ssm` if none was accepted, else `state_tok[s, P - 1]`) and the conv
window starting at column P. The per-draft states of a round are read only by the next round.

**The PDL rule.** The kernel reads the slot, P, the starting state and the conv window before its grid-dependency
wait, and lets its dependents launch in its prologue, before that wait. So a launch must not follow another launch on
the same pools without a kernel that waits (or a non-PDL kernel) in between, and the launch in between must not be
one that lets its dependents launch before its own wait either: one `k3_kda_verify` launch on other pools is not
enough. In the model, a layer's other kernels sit between its launches.

**Why the test drives call sequences.** The record and the per-draft states make each round depend on the previous
one's acceptance, which no single call can check. The test therefore runs 6 rounds of every layer with a pending count
per request that changes every round (every count 0..num_spec over the rounds), against the float64 history, and
captures a round of every layer once, replaying it with rewritten rows and records: the replays give the bits of the
same rounds run eagerly, and the schedule run twice gives the same bits.

**What a wrong order does.** Two rounds of one request in swapped order (the negative control): nothing raises, and
the second round's outputs and the slot's state are those of a different history. Measured on sm_100: the second round's outputs are off by 0.98 and the
state by 0.98 (the largest absolute difference over the in-order result's largest magnitude).

## Metadata consumed

`slots` and `pending`, both read before the grid-dependency wait: under CUDA graphs they must be written before the
graph runs (the step's preparation and the previous step's acceptance).

## Preconditions

- sm_100: the kernel uses tcgen05, TMA and clusters.
- The op checks `proj` (bf16, rank 2, N (1 + num_spec) rows, columns), `w_fb`, V = K = 128, `ssm` / `state_tok`
  (shapes, fp32, `ssm.stride()[1:] == (K K, K, 1)`, `state_tok` contiguous), `cs_q`'s window width, the int32 index
  tensors and the fp32 dtypes, and raises `ValueError` before a launch. A conv cache that is not dense raises
  `ValueError` too. The fp32 weights' shapes, `cs_k` / `cs_v`, `pending` covering the slots and the slots' range are
  the caller's obligation.
- The first call of each configuration (H, num_spec, `lower_bound`, `scale`, `eps`, with or without `g_ext`, and
  whether PDL is on) compiles the kernel and must run outside CUDA-graph capture; under capture it raises
  `RuntimeError`. PDL follows `TRTLLM_ENABLE_PDL` (default on), read at every call.
- A call names each slot once.

## Notes

- `slots` and `pending` may be slices of longer index tensors starting at any element (the mixer passes
  `state_indices[num_prefills:]`); the op's own test certifies the bits at offsets 0..3.
- The pools are addressed at 64-bit slot offsets, so pools past 2 GiB are fine (the op's own test,
  `test_k3_kda_pools_past_2g.py`).
- `ssm/k3_kda_attn` computes this verify fused with the layer's projection, for one request of 8 tokens.
