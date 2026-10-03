---
receipts: {}
---

# k3_kda_attn

**Wraps** `torch.ops.trtllm.k3_kda_attn` (one call), over a caller-owned `K3KdaBuffers`
(`catalog/ssm/k3_kda_buffers.py`). Sibling in this contract: `k3_kda_qkvg`, which wraps
`torch.ops.trtllm.k3_kda_qkvg` (the projection stream alone, one call).

Kimi K3's KDA layer for one speculative-decoding request, at the TP16 rank slice: the fused input projection of the
request's golden token and its 7 drafts and the KDA verify of `ssm/k3_kda_verify` on it, in one launch.

## Semantics

`x` bf16 `[8, 7168]` is one request's golden token and 7 drafts. One launch computes:

```
# 1. projection, streamed by 26 clusters of 4 CTAs (w = [q | k | v | og | f_a | b | pad], 3208 rows)
y = x @ w^T
#    q, k, f_a: bf16 of the four split-K partials of a cluster, summed in rank order
#    v, og, b:  bf16(half_0 + half_1), the fp32 sums of the two K halves
# 2. verify (exactly ssm/k3_kda_verify on y with the gate folded in, g_ext = None):
#    starting state S = ssm[slot] if P == 0 else state_tok[slot, P - 1],   P = pending[slot]
#    per token t = 0..7: g = bf16(f_a @ w_fb^T); q, k, v = SiLU(conv4(window, new raw)); q, k L2-normalized,
#    q *= scale; beta = sigmoid(b); decay = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)));
#    S *= decay (per key); S += beta (v - S k) k^T; o = S q;
#    out[t] = o * rsqrt(mean(o^2) + eps) * onorm_w * sigmoid(og)
```

and returns `out` bf16 `[8, 6, 128]`, the gated-norm core output. In place, at the request's slot:

- `ssm[slot]` = the state after the golden token (t = 0);
- `state_tok[slot, t - 1]` = the state after draft t, t = 1..7;
- `cs_q` / `cs_k` / `cs_v[slot]` = the raw inputs at positions -2..7 around the golden token (the next round's
  window starts at column P).

The result is bit for bit that of `k3_kda_qkvg` followed by `ssm/k3_kda_verify` on the decoded rows, on copies of the
same pools (certified, every pool word). Repeated runs are bit-identical.

**`k3_kda_qkvg`** runs phase 1 alone on `x` bf16 `[T <= 8, 7168]` and publishes the rows: buffer
`e = buffers.epoch[0]` (before the call) holds q, k and f_a as bf16 bits in `p1`, and v, og and b as the fp32 bits of
the two K-half partials in `part`. The projection is bf16 of their sum, the consumer's job. Measured against a float64
`x @ w^T`: the rows are within 2.5e-3 of the projection's largest magnitude at T = 1, and 2.4e-3 at T = 3 and 8.

## Signature

```python
def k3_kda_attn(
    x: torch.Tensor,
    w: torch.Tensor,
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
    buffers: K3KdaBuffers,
    num_spec: int,
    lower_bound: float,
    scale: float,
    eps: float,
) -> torch.Tensor

def k3_kda_qkvg(x: torch.Tensor, w: torch.Tensor, buffers: K3KdaBuffers) -> None
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[8, 7168]` (`k3_kda_qkvg`: `[T <= 8, 7168]`) | bf16 | contiguous | CUDA |
| `w` | `[3208, 7168]`: q, k, v, og (768 rows each), f_a (128), b (6), pad (2) | bf16 | contiguous | CUDA |
| `w_fb` | `[768, 128]` (f_b, out x in) | bf16 | contiguous | CUDA |
| `w_q`, `w_k`, `w_v` | `[768, 4]` conv taps, oldest input first | fp32 | dense | CUDA |
| `a_log` | `[6]` | fp32 | dense | CUDA |
| `dt_bias` | `[768]` | fp32 | dense | CUDA |
| `onorm_w` | `[128]` | fp32 | dense | CUDA |
| `cs_q`, `cs_k`, `cs_v` | `[pool, 768, 3 + num_spec]` | fp32 | channel stride 1 (dim-contiguous) | CUDA |
| `ssm` | `[pool, 6, 128, 128]` (V rows, K contiguous) | fp32 | each slot dense, slots at any stride | CUDA |
| `state_tok` | `[pool, num_spec, 6, 128, 128]` | fp32 | contiguous | CUDA |
| `slots` | `[1]`: the request's slot | int32 | any element offset | CUDA |
| `pending` | `[pool]`: drafts the sampler accepted last round, per slot | int32 | any element offset | CUDA |
| `buffers` | `K3KdaBuffers` made with `ctas=FUSED_CTAS` (128) | — | — | CUDA |
| `num_spec` | 7 | Python int | — | — |
| `lower_bound`, `scale`, `eps` | scalars (Kimi K3: -5.0, 128^-0.5, 1e-5) | Python float | — | — |
| returns | `[8, 6, 128]` | bf16 | contiguous, fresh | CUDA |

`mutates_args`: `cs_q`, `cs_k`, `cs_v`, `ssm`, `state_tok`, and the set's `p1`, `part`, `epoch`. `k3_kda_qkvg` takes a
set made with `ctas=CTAS` (104) and mutates its `p1`, `part`, `epoch`; it returns None.

## State

**Object.** `K3KdaBuffers` (`catalog/ssm/k3_kda_buffers.py`), one per device, owned by the caller.

**Contents and size.** The projection's three Lamport buffers and each CTA's buffer index:

- `p1` int16 `[3 * 8 * 1664]`: per buffer, 8 token rows of the q, k and f_a columns (79,872 bytes);
- `part` int32 `[3 * 3 * 2 * 8 * 768]`: per buffer, the fp32 bits of the two K-half partials of v, og and b
  (442,368 bytes);
- `epoch` int32 `[128]` (`[104]` for `k3_kda_qkvg`): each CTA's launch count mod 3.

Every buffer word holds the sentinel (all ones, a NaN no finite GEMV produces; a computed all-ones word is stored as
the canonical NaN instead) until the launch's producer CTAs write it. The consumer CTAs poll the words they need until
none is the sentinel: the data words are the flags, with no fence or counter. 8 token rows are published per launch,
whatever the batch.

**Who creates it, and when.** The target, in `post_load_weights`, with `K3KdaBuffers.create(device)`:

- eager: it allocates, so it refuses to run under CUDA-graph capture;
- it arms every buffer word to the sentinel and every index to 0;
- `ctas` is `FUSED_CTAS` (128) for this op and `ssm/k3_kda_decode_attn`, or `CTAS` (104) for `k3_kda_qkvg`; any other
  count raises. No environment variable is read.

**Which ops may share one object.** `k3_kda_attn` and `ssm/k3_kda_decode_attn`, of every KDA layer on the device:
one set per device serves them all. Both run the same stream role, publish all 8 token rows, re-arm the same words
and move every CTA's index once per launch, so either may follow the other in any order. Certified: plain-decode and
verify launches interleaved on one set give the bits of the same launches on sets of their own; layers x steps on one
set, a set per layer, and two sets alternating between launches agree bit for bit. `k3_kda_qkvg` needs a set of its
own (104 CTAs); a 104-CTA set passed here raises `ValueError` before anything is written.

**Call-order invariant.** Launches on one set run one at a time, in stream order: never on two streams at once. Each
launch reads `e = epoch[cta]` after its grid-dependency wait, writes buffer e, stores the sentinel into the same words
of buffer (e + 1) % 3 (the next launch's, which the launch before last read) and leaves `epoch[cta] = (e + 1) % 3`.
After any launch every CTA's index is equal.

**What a later launch reads.** `epoch`, as the previous launch left it, and its own buffer's words, which the previous
launch re-armed to the sentinel.

**How it is re-armed.** By the previous launch (above); the first launch on a new set writes buffer 0, which `create`
armed. The index stays in 0..2: a raw launch count would turn negative after 2^31 launches and its signed remainder
would index before the buffers. Certified with every index preset to 2^31 - 2: four launches give the bits of a fresh
set's and leave every index in 0..2 (the op's own test also checks with guard bands that nothing is written outside the
buffers).

**The pools.** Not part of the object: they belong to the cache manager. With the V2 hybrid manager built with the
KDA replay caches and `kda_token_states`, layer `l`'s are `mamba_layer_cache(l).kda_conv_q` / `kda_conv_k` /
`kda_conv_v` (`cs_*`), `get_ssm_states(l)` (`ssm`, its slots strided by the manager's per-slot coalescing) and
`mamba_layer_cache(l).kda_state_tok`; `pending` is `prev_num_accepted_tokens`, one record shared by every layer and
written by the sampler's acceptance between steps. A launch on layer `l` writes only layer `l`'s views, at its slot.

**The PDL rule.** The head CTAs read the slot's pools (state, records, conv window) before their grid-dependency wait.
A launch must therefore not directly follow another launch on the same pools in the stream: a kernel that waits, or a
non-PDL kernel, must sit between them, as the model's other layers do. This op lets its dependents launch only once
its head CTAs have passed their own wait, i.e. once the launch before it has completed, so one launch of this op (or of
`ssm/k3_kda_decode_attn`) on other pools in between is enough. A kernel that lets its dependents launch before its
wait (`ssm/k3_kda_verify` does, in its prologue) is not.

**Why the test drives call sequences.** The state is carried by the pools from round to round: `pending` selects the
next round's starting state and conv window, and a round's per-draft states are only read by the next one. The test
therefore runs two requests one after the other, each for 9 rounds with pending running through 0..7, on every layer
of a real manager, against the unfused path bit for bit and layer 0 also against a float64 verify over the request's
committed history; then layers x rounds on one shared set against a set per layer, and a round of every layer captured
once and replayed with rewritten inputs and records.

**What a wrong order does.** Two rounds of one request in swapped order (the test's negative control): nothing raises,
and the second round's output and the slot's state are those of a different history. Measured on sm_100: the second round's output is off by 1.00 and
the state by 1.01 (the largest absolute difference over the in-order result's largest magnitude).

## Metadata consumed

`slots` and `pending`. Both are read before the grid-dependency wait, so under CUDA graphs they must be written before
the graph runs (by the step's preparation and the previous step's acceptance), never by a kernel of the same step
ahead of this launch.

## Preconditions

- sm_100: the kernel uses tcgen05, TMA and clusters.
- The op checks `x`, `w`, `w_fb` (shape, dtype, contiguity), `num_spec` = 7, `ssm` / `state_tok` (shape, fp32,
  `ssm.stride()[1:] == (16384, 128, 1)`, `state_tok` contiguous), `cs_q`'s window width, the fp32 dtypes, `slots`
  (one int32 element), `pending` (int32) and the set's sizes (`epoch` of 128), and raises `ValueError` before a launch.
  A conv cache that is not dense raises `ValueError` too. The other shapes in the table (the fp32 weights, `cs_k` /
  `cs_v`, `pending` covering the slot, the slot inside the pool) are the caller's obligation.
- Every tensor the kernel addresses is 16-byte aligned except `slots` / `pending` (see *Notes*); the DSL checks the
  assumed alignment at the call.
- The first call of each configuration (`lower_bound`, `scale`, `eps` and whether PDL is on) compiles the kernel and
  must run outside CUDA-graph capture; under capture it raises `RuntimeError`. PDL follows `TRTLLM_ENABLE_PDL` (default
  on), read at every call.
- The slot is not in use by another launch on the same pools (one request per launch; a batch is a sequence of
  launches).

## Notes

- `slots` and `pending` may be slices of longer index tensors starting at any element (the mixer passes
  `state_indices[num_prefills:]`): the kernel reads them at their element's alignment. The op's own test certifies
  the bits at offsets 0..3.
- The pools are addressed at 64-bit slot offsets, so pools past 2 GiB are fine (the op's own test,
  `test_k3_kda_pools_past_2g.py`).
- Repeated runs, and graph replays against eager runs, are bit-identical.
