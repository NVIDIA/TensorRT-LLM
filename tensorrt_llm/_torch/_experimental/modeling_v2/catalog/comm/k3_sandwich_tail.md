---
receipts:
  sm_100: {status: passed, world_size: 4}
---

# k3_sandwich_tail

**Wraps** `torch.ops.trtllm.k3_sandwich_tail` (one call).

## Semantics

Kimi K3's pre-attention step for a decode batch of at most 8 tokens, in one kernel: the row-parallel MoE tail (this
rank's slice of the latent up projection of the normed latent, and its slice of the shared experts' down
projection), the TP all-reduce of its output, and the next layer's residual update. Every rank of the workspace's TP
group calls with the whole reduced latent `latent` `[M, 3584]` (the same on every rank), its slice of the
shared-expert activation `act` `[M, 384]`, its `tail_weight` `[7168, 640]` and its slice's first latent column `lo`;
every rank gets back the same two tensors. Per token, with `W` ranks:

```
scale     = 1 / sqrt(mean over the 3584 columns of latent^2 + lat_eps)   # fp32, the whole reduced latent row
partial_r = bf16(scale * (latent[:, lo_r : lo_r + 224] @ tail_weight_r[:, 0:224]^T)
                 + act_r @ tail_weight_r[:, 256:640]^T)                  # fp32 accumulators, one rounding
updated   = bf16(prefix + bf16(sum over the W ranks of partial_r))        # the sum alone when prefix is None
normed    = RMSNorm(attn_res(block_residual[0], ..., block_residual[S-1], updated); output_rms_weight, output_rms_eps)
```

and returns `(normed, updated)`: `[rmsnorm(latent)[:, lo:lo+224] | act] @ tail_weight^T` with the latent's RMS
applied to the fp32 latent accumulator rather than to the latent, then the epilogue of `comm/k3_sandwich_oproj` —
the tail's partial followed by `trtllm::mnnvl_allreduce_attn_res` (the op's statement). The latent RMSNorm carries
no weight here: a norm weight belongs folded into the latent columns of `tail_weight` (Kimi K3's model folds it).
The kernel multiplies latent columns `[lo, lo + 256)` by `tail_weight[:, 0:256]`; columns 224-255 of `tail_weight`
are zero padding, so the next slice's columns (or, past the end of the row, zeros) add nothing. The result is bitwise
identical on every rank (certified, every call of the test).

Two optional outputs. With `tap` the kernel also stores into it the pre-norm attention-residual mixture — the bf16
`attn_res(...)` that the RMSNorm normalizes, a DSpark capture layer's tap — or, with `tap_updated`, `updated`. With
`updated_out` it stores `updated` there instead of into a new tensor (e.g. the next row of the snapshot bank), and the
wrapper returns that tensor as `updated` (the op itself returns an empty `[0, 7168]` in its place). The options move
outputs, not values: certified, the same inputs give bit-identical `normed` and `updated` with each option and
without.

Fusion boundary. Inside: the latent's RMS, the tail projection, the all-reduce, the residual add, the selection, the
RMSNorm, and the tap. Outside: the latent all-reduce that produced `latent`, the shared experts' gate_up and
activation that produced `act`, folding the latent norm's weight into `tail_weight`, keeping the snapshot bank,
chaining `prefix`, and the attention that consumes `normed` (its post-attention step is `comm/k3_sandwich_oproj`, on
the same workspace).

## Signature

```python
def k3_sandwich_tail(
    latent: torch.Tensor,
    act: torch.Tensor,
    tail_weight: torch.Tensor,
    lo: int,
    lat_eps: float,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    workspace: K3SandwichWorkspace,
    tap: Optional[torch.Tensor] = None,
    tap_updated: bool = False,
    updated_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `latent` | `[M, 3584]`, `M` 1-8 (below) | bf16 | contiguous, 16-byte aligned | CUDA, this rank's device |
| `act` | `[M, 384]` (this rank's slice) | bf16 | contiguous | CUDA |
| `tail_weight` | `[7168, 640]` (below) | bf16 | contiguous | CUDA |
| `lo` | `0, 224, ..., 3360` (below) | Python int | — | — |
| `lat_eps`, `rms_eps`, `output_rms_eps` | scalar (certified at 1e-6) | Python float | — | — |
| `prefix` | `None`, or `[M, 7168]` | bf16 | contiguous | CUDA |
| `block_residual` | `[S, M, 7168]`, `S` 0-8 | bf16 | contiguous | CUDA |
| `res_weight`, `rms_weight`, `output_rms_weight` | `[7168]` | bf16 | contiguous | CUDA |
| `workspace` | a `K3SandwichWorkspace` of this rank's TP group (see *State*) | — | — | — |
| `tap` | `None`, or `[M, 7168]` (below) | bf16 | unit column stride | CUDA |
| `tap_updated` | `False` (the mixture) or `True` (`updated`) | bool | — | — |
| `updated_out` | `None`, or `[M, 7168]` | bf16 | contiguous, 16-byte aligned | CUDA |
| returns | `(normed, updated)`, each `[M, 7168]` | bf16 | contiguous | = `latent.device` |

- `latent`: the whole reduced latent row, the same on every rank (the latent all-reduce's output).
- `tail_weight`: columns 0-223 this rank's slice of the latent up projection (the latent norm's weight folded in),
  224-255 zero, 256-639 its slice of the shared experts' down projection.
- `lo`: the slice's first latent column, 224 x the slice index in Kimi K3. All 16 slices are certified (every rank
  takes each of them across the single calls), including the last, whose padding columns run past the row.
- `tap`: rows a multiple of 8 elements apart, 16-byte aligned; certified as a column slice of an `[M, 5 x 7168]`
  capture buffer, nothing outside the slice written.
- `updated_out`: certified as a row of a `[3, M, 7168]` bank, the other rows untouched; `updated` is then that
  tensor. Otherwise both outputs are newly allocated.
- Single calls are certified at every `M` x `S` in {0, 1, 4, 8} x with and without `prefix`, and the options at every
  `M`; the call sequences below take `S` 0-8. The inputs are read only.

Inert (not exposed by the wrapper): `x_slab`, `slab_buf`, `src_slab`, `src_buf` — publishing `normed` into a
Lamport slab and polling the reduced latent from its producer's slab, cross-call state of their own as for
`comm/k3_sandwich_oproj`; and `lat_uc`, `lat_flags` — the latent all-reduce folded into this op over a second state
object, a `K3SandwichLatentExchange` (see *Notes*).

## State

**Object.** `K3SandwichWorkspace`, the object `comm/k3_sandwich_oproj` takes, defined in
`tensorrt_llm/_torch/cute_dsl_kernels/k3_sandwich/op.py` and re-exported by this entry's wrapper too; one per TP
group, owned by the caller.

**Contents and size.** As in `k3_sandwich_oproj.md`: two alternating halves of `[8 tokens][W ranks][7168]` bf16 per
rank, as int32 words (`0x80000000` = empty), behind one multicast mapping (`uc`, `mc`) — `2 x 8 x W x 7168 x 2`
bytes, 0.92 MB at `W` = 4 and 3.67 MB at `W` = 16 — and `flags`, int32 `[64]`, the call counts of the kernel's 56
CTAs, with the handle that owns the memory and the communicator. This op's partial rows go into the same slots as
`k3_sandwich_oproj`'s. Certified after `create`: sized for the group, every word empty, every counter zero.

**Who creates it, and when.** The target, in `post_load_weights`, with
`K3SandwichWorkspace.create(mapping, fabric_handle=None)`:

- collective over `mapping`'s TP group only: every rank of the group, and no other rank of the session, calls it
  at the same point. Under MPI its communicator is made from the group's ranks alone (`MPI_Comm_create_group`;
  certified in `comm/k3_latent_reduce`'s matrix: with the job split into two TP groups of `W / 2`, one group
  makes its communicator while the other's ranks do not call); a rank that calls it while its group's peers do
  not waits for them;
- failure model:
  - before allocating, the ranks agree that each of them can (not capturing, the buffer within that rank's free
    device memory). If one cannot, every rank raises `RuntimeError`, none allocates, and under MPI each frees the
    communicator made for the call (certified: one rank capturing while its peers call it eagerly, every rank
    raises, the capturing rank naming the capture and its peers another rank; no rank reaches the allocation,
    every rank frees that communicator, and the next call is correct);
  - a failure that returns from the allocation is agreed and handled the same way;
  - a rank that fails inside the allocation's handle exchange can leave its peers waiting in that exchange; this
    is not turned into an error on the other ranks;
- eager: it allocates and exchanges handles, so it refuses to run under CUDA-graph capture (certified: every rank
  capturing, every rank raises; no stream is left capturing);
- it empties every word and zeroes every counter, and returns only once every rank has (the agreement after the
  allocation), so no peer can push into a buffer its owner has not emptied yet;
- `fabric_handle`: share the memory by fabric handle (required across nodes) or POSIX file descriptor; default
  `mapping.is_multi_node()`. No environment variable is read.

**Which ops may share one object.** The three sandwich entries — `comm/k3_sandwich_oproj`, this one and
`comm/k3_sandwich_plain` — take the same object, and in the model the target's and the drafter's calls run on one
workspace per TP group, one call sequence. That sharing is certified by `comm/k3_sandwich_oproj`'s matrix: decode
steps of target layers (`k3_sandwich_oproj`, then this op) with drafter layers interleaved, eager with a random rank
late and captured with eager calls between replays. The folded latent exchange (`K3SandwichLatentExchange`) is a
separate object with its own counter. Two workspaces keep independent counters: calls of this op alternating between
two in an irregular pattern are all correct (certified, 20 calls).

**Call-order invariant.** As in `k3_sandwich_oproj.md`: every rank makes the same sequence of sandwich calls on one
workspace (the same number of calls, the `k`-th with the same op and `M`) across layers, decode steps, eager calls
and graph replays; the same order of calls across objects on one stream; and no kernel between two calls that skips
its grid-dependency wait.

**What a later launch reads.** `flags`: every call adds one to every CTA's counter (certified for this op: all 56 by
exactly one per call, the spare words untouched), and the counter's parity before the call selects the half this
call pushes into and polls; and that half's words, which must be empty except for this call's pushes.

**How it is re-armed.** By its readers, as for `k3_sandwich_oproj`: a thread empties every word it read right after
reading it, so no separate clear and no record of the previous call's size is needed (the kernel's statement).

**Why the test drives call sequences.** See `k3_sandwich_oproj.md`. This entry's test runs 11 decode steps of 12
chained layers at `M` = 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8, a random rank 5 ms late at every call, each call against
the reference.

**What a wrong order does.** Certified by the test's negative control: rank 0 issues two same-shaped calls (no
prefix) on one workspace in swapped order. Nothing raises and nothing hangs, but every rank's two results are wrong,
far outside the 8e-3 tolerance below: more than half the `updated` elements are off by more than it, the largest by
over 10 times it. A plain call right after is correct again.

## Metadata consumed

Besides `workspace`, the op module's process-wide compile cache, keyed by (`W`, publish, input source, tap kind,
PDL): a call without a tap, one tapping the mixture and one tapping `updated` run three kernels, each compiled
(seconds) on its first call, which must be eager — under capture the op raises "must run once per configuration
outside CUDA-graph capture first". `updated_out` is not part of the key. The cache is result-neutral. PDL
(`TRTLLM_ENABLE_PDL`, read at every call, default on) changes scheduling, not results.

## Preconditions

- bf16; `latent` `[M, 3584]`, `act` `[M, 384]` and `tail_weight` `[7168, 640]` contiguous; `M` 1-8; at most 8
  snapshots: the op's `supports_tail`. `latent`'s rows 16-byte aligned (the kernel bulk-copies them); `tap` with unit
  column stride, a row stride that is a multiple of 8 elements and a 16-byte aligned start; `updated_out` contiguous
  and 16-byte aligned. Each violation raises `ValueError` on every rank before any launch — certified for `M` = 9,
  a latent 2 bytes off alignment and a tap whose rows are 7172 elements apart (every counter unchanged) — and the
  next call is correct.
- Not checked by the op, so the caller's: `lo` a slice start (a multiple of 224 from 0 to 3360 in Kimi K3), the
  zero columns 224-255 of `tail_weight`, and `prefix`, `block_residual` and the weights being bf16 `[M, 7168]`,
  `[S, M, 7168]` and `[7168]` (certified contiguous; the op flattens them with `reshape`, which copies a
  non-contiguous view).
- `updated_out` is not a tensor the call reads (the op's example: the next row of the snapshot bank, which this call
  does not read).
- Every rank calls with the same `M`, `S` and `prefix` presence, and the same `latent`, `prefix`, `block_residual`
  and weights (the replicated residual stream); the call order is the *State* invariant.
- `workspace` was created, and each kernel compiled (one eager call per key), before any capture. Calls may be
  captured: certified with a captured step of 12 chained calls at `M` = 8 — one tapping the mixture, one tapping
  `updated`, one storing `updated` into a bank row — replayed 8 times with rewritten inputs, an eager call of
  another `M` on the same workspace between replays, every replayed and eager call against the reference.
- Under PDL the kernel launches its dependents once every CTA has pushed, before its outputs are written (the
  kernel's statement): a kernel launched after it reads the outputs only after its own grid-dependency wait.

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, POSIX-fd handles, Kimi K3 TP16 per-rank
  shapes. Test: `tests/unittest/_torch/modeling_v2/comm/_k3_sandwich_tail_op_matrix.py`, helpers in
  `_k3_sandwich_common.py`. The reference is native torch, in fp64 for the partials: `latent`, `act` and
  `tail_weight` are small multiples of 1/8 and 1/16, so both accumulators are exact, but the RMS scale is the
  kernel's fp32 rsqrt, not torch's, so a partial element can round to the neighbouring bf16. `updated` is compared
  within 8e-3 of its largest magnitude (about one bf16 ulp of its largest elements); `normed` and the tapped mixture
  within 2e-2 of an fp32 reference; the tapped `updated` bit for bit with the returned one; every output bitwise
  across the ranks. The negative control's wrong pairing puts more than half the elements outside the 8e-3 bound,
  the largest error over 10 times it.
- World sizes: as for `k3_sandwich_oproj`; this entry's 16-rank receipt is pending. The op's kernel test
  (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_sandwich.py`) passed every case at 16 ranks on four
  trays in a recorded run: the kernel's record, not this entry's receipt.
- State: `mutates_args` names every buffer the op can write: `ws_uc`, `ws_mc`, `ws_flags`, `x_slab`, `lat_uc`,
  `lat_flags`, `tap` and `updated_out`. The compile cache is the documented process-wide cache above.
- Not certified (an op option the wrapper does not expose): the folded latent all-reduce. With `lat_uc` / `lat_flags`
  of a `K3SandwichLatentExchange` — its own collective `create(mapping, fabric_handle=None)`; two halves of
  `[8 tokens][W ranks][3584]` bf16 per rank, a call count mod 6 and a latent-scale slab — `k3_moe` pushes its
  routed latent partial into every rank's exchange and exits, and this op sums the ranks' partials itself, in the
  one-shot's order, instead of reading a reduced `latent`, which then gives only the shape; every push-only `k3_moe`
  call must be followed by exactly one such tail call on the same exchange (the op's statement). That is a second
  state object with its own call-order rule, outside this entry.
