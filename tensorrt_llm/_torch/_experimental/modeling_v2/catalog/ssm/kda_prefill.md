---
receipts:
  sm_100: {status: passed, tests: 5}
---

# kda_prefill

**Wraps** `torch.ops.trtllm.kda_prefill` (one call).

## Semantics

Chunked KDA (Kimi Delta Attention: a gated delta rule with a per-key decay)
prefill of a batch of sequences, each started from its slot's recurrent
state, which the call then replaces with the state after the sequence's last
token. Per sequence and head, with `S` the `[V, K]` state (V-first), `q`, `k`
already L2-normalized by the caller, and Kimi K3's configuration
(`use_gate_in_kernel=True`, `safe_gate=True`, `use_beta_sigmoid_in_kernel=True`):

```
S     = state_pool[state_indices[b], h]                     # read
for t in the sequence:
    decay = exp(lower_bound * sigmoid(exp(A_log[h]) * (g[t, h] + dt_bias[h])))   # per key
    S     = S * decay                                       # column k scaled by decay[k]
    S     = S + sigmoid(beta[t, h]) * (v[t, h] - S k[t, h]) k[t, h]^T
    o[t, h] = S (scale * q[t, h])
state_pool[state_indices[b], h] = S                         # written
```

The kernels evaluate this in chunks of `chunk_size` (64) tokens with bf16
operands and fp32 accumulation, not token by token; `o` comes back in
`v.dtype`.

Batch forms:

- **varlen** (`cu_seqlens` given): `q / k / v / g` are `[1, T, H, K]`,
  `beta` `[1, T, H]`, sequence `b` the tokens `cu_seqlens[b]` to
  `cu_seqlens[b + 1]`. The caller passes `chunk_indices`
  (`prepare_chunk_indices(cu_seqlens, chunk_size)`). A single varlen
  sequence must be zero-padded to a chunk multiple by the caller while
  `cu_seqlens` keeps its real length (Kimi K3's layer does this).
- **equal-length** (`cu_seqlens=None`): `[B, T, H, K]`, one sequence per row.

The op launches any chunk count. Kimi K3's layer sends it batches of 4 or more
64-token chunks in all and runs smaller ones on FLA (`ssm/chunk_kda`); this
entry is certified from 4 chunks up.

Fresh sequences start from whatever their slot holds: the caller zeroes those
rows first (Kimi K3's layer calls `reset_recurrent_state_rows`).

Fusion boundary: the gate activation, the beta sigmoid, the chunked delta rule
and the state write-back. The projections, the conv and the q / k
normalization before it, and the gated output norm after it
(`norm/rms_norm_gated_token_major`) are the caller's.

## Signature

```python
def kda_prefill(
    q, k, v, g, beta,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    varlen_is_aligned: Optional[bool] = None,
    single_sequence_length: Optional[int] = None,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `q`, `k` | `[1, T, H, 128]` (varlen) or `[B, T, H, 128]` | bf16, L2-normalized per head | contiguous | CUDA |
| `v` | as `q` | bf16 | contiguous | CUDA |
| `g` | as `q` (the raw gate) | bf16 | contiguous | CUDA |
| `beta` | `[1, T, H]` or `[B, T, H]` (raw) | fp32 | contiguous | CUDA |
| `state_pool` | `[slots, H, 128, 128]` (V-first) | fp32 | each slot dense; slot stride a multiple of 4 floats and at least `H * 128 * 128`; 16-byte aligned base | CUDA |
| `state_indices` | `[num_sequences]` | int32 / int64 | contiguous, 16-byte aligned | CUDA |
| `scale` | scalar (Kimi K3: `128 ** -0.5`) | Python float | — | — |
| `cu_seqlens` | `[num_sequences + 1]` or None | int64 (Kimi K3's layer passes the metadata's `query_start_loc_long`) or int32 | — | CUDA |
| `chunk_indices` | `prepare_chunk_indices(cu_seqlens, chunk_size)` or None | as `cu_seqlens` | — | CUDA |
| `chunk_size` | 64 | Python int | — | — |
| `safe_gate`, `lower_bound` | Kimi K3: True, -5.0 | Python bool / float | — | — |
| `use_gate_in_kernel`, `use_beta_sigmoid_in_kernel` | Kimi K3: True, True | Python bool | — | — |
| `A_log` | `[H]` | fp32 | contiguous | CUDA |
| `dt_bias` | `[H * 128]` | fp32 | contiguous | CUDA |
| `varlen_is_aligned`, `single_sequence_length` | optional host metadata (Kimi K3's layer passes both: every length a chunk multiple; the length of a lone sequence) | Python bool / int | — | — |
| returns `o` | `v.shape` | `v.dtype` | newly allocated | same device |

Schema mutation: `state_pool` (`mutates_args=("state_pool",)`).

## State

**Object.** None of its own (stateful kind P): the op replaces the caller's
recurrent state rows in place. The runner keeps compiled kernels and
intermediate scratch per batch shape in a process-wide cache (an A1 gap,
noted below); the returned `o` is not part of it: every call allocates a new
one, which later calls leave alone (pinned by `test_output_is_a_new_tensor`).

**The pool.** The cache manager's per-layer SSM states (the V2 hybrid
manager's fp32 V-first pool for a KDA layer), slots possibly strided wider
than one state. A call writes exactly the slots in `state_indices` (other
slots and the slot padding stay bit-unchanged, certified).

**Call-order invariant.** A slot carries one sequence's recurrent state:
chunked prefill of a sequence calls chunk after chunk on the same slot, and
then matches one call over the whole sequence within tolerance (certified at
256 + 256 tokens against 512, the outputs within `2e-3` of the one call's
largest).

**What a wrong order does.** The halves in the wrong order raise nothing and
leave a different state (the negative control).

## Metadata consumed

`cu_seqlens`, `chunk_indices`, `state_indices` and the optional host metadata
(`varlen_is_aligned`, `single_sequence_length`), all built by the caller once
per batch. No attention metadata.

## Preconditions

- Checked by the op (`ValueError`, before any launch): `state_pool` is a
  rank-4 fp32 tensor `[slots, H, V, K]` matching `q` / `v`, with dense inner
  strides, a non-overlapping 16-byte-aligned slot stride and a 16-byte-aligned
  base; `state_indices` is 1-D, one entry per sequence, int32 / int64,
  contiguous and 16-byte aligned; `q`, `v` and `state_indices` on the pool's
  device; `chunk_size == 64`; `A_log` given.
- Not checked by the op: the zero padding of a single varlen sequence,
  `K == V == 128`, the other inputs' devices.
- Requires the CuTe DSL and FlashInfer (the op registers only then) and an
  SM 10.x GPU.

## Notes

- **Certified surface** (sm_100): heads 6 (TP16) and 24 (TP4); varlen
  batches of lengths 100, 64, 300, 37 mixing fresh sequences (zeroed rows) and
  continuations (random states) over scattered slots of a padded pool; a
  256 + 256 continuation against one 512-token call and its negative
  control; an equal-length batch of two 256-token sequences; a new output on
  every call; five of the pool and index checks above, each raising.
  Varlen calls pass int64 `cu_seqlens` and `chunk_indices` and the host
  metadata, as the layer does, plus one int32 batch.
- **Numerics.** Against an fp64 token-by-token reference, `o` is within
  `2e-2` of each sequence's largest output and the final state within `1e-2`
  of its largest entry (measured: at most 5.5e-3 for `o` and 5e-3 for the
  state, over sequences of 37 to 1024 tokens).
- The A1 gap: the runner's compiled kernels and intermediate scratch live in
  a module-level cache, not a caller-owned object.
- Kimi K3's layer runs a batch of fewer than 4 chunks on FLA's `chunk_kda`
  instead (`ssm/chunk_kda`, which agrees with this entry on a batch either can
  take); a bf16 pool it stages through dense fp32 rows and keeps here.
