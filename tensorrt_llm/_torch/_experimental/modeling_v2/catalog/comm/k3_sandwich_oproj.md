---
receipts:
  sm_100: {status: pending, world_size: 4}
---

# k3_sandwich_oproj

**Wraps** `torch.ops.trtllm.k3_sandwich_oproj` (one call).

## Semantics

Kimi K3's post-attention step for a decode batch of at most 8 tokens, in one kernel: the row-parallel attention
output projection, the TP all-reduce of its output, and the residual update. Every rank of the workspace's TP group
calls with its own attention output `core` `[M, 768]` and its slice of the output projection `o_weight`
`[7168, 768]`; every rank gets back the same two tensors. Per token, with `W` ranks:

```
partial_r = bf16(core_r @ o_weight_r^T)                            # this rank's [M, 7168] share, fp32 accumulator
updated   = bf16(prefix + bf16(sum over the W ranks of partial_r))  # the sum alone when prefix is None
normed    = RMSNorm(attn_res(block_residual[0], ..., block_residual[S-1], updated); output_rms_weight, output_rms_eps)
```

and returns `(normed, updated)`. `attn_res` is the attention-residual selection of `comm/mnnvl_allreduce_attn_res`
(candidates scored by `rmsnorm(v) . (rms_weight * res_weight)` with `rms_eps`, softmax, mix); the RMSNorm is Kimi
K3's (normalize in fp32, round to bf16, apply the weight). The kernel follows `oneshotAllreduceAttnResKernel`'s
arithmetic — ranks summed in chunks of 8 in rank order, the same statistics, summation orders and roundings — so
its outputs are bit-identical to `o_proj` followed by `trtllm::mnnvl_allreduce_attn_res` (its kernel's statement).
The result is bitwise identical on every rank (certified, every call of the test).

Fusion boundary. Inside: the projection, the all-reduce, the residual add, the selection, the RMSNorm. Outside: the
attention that produced `core`, keeping the snapshot bank, chaining `prefix` from layer to layer, and the MoE that
consumes `normed`. The next pre-attention step (the MoE tail, its all-reduce and the next layer's residual update)
is `comm/k3_sandwich_tail`, on the same workspace.

## Signature

```python
def k3_sandwich_oproj(
    core: torch.Tensor,
    o_weight: torch.Tensor,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    workspace: K3SandwichWorkspace,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `core` | `[M, 768]`, `M` 1-8 (TP16's per-rank shape, whatever `W`) | bf16 | contiguous | CUDA, this rank's device |
| `o_weight` | `[7168, 768]` (this rank's slice) | bf16 | contiguous | CUDA |
| `prefix` | `None`, or `[M, 7168]` | bf16 | contiguous | CUDA |
| `block_residual` | `[S, M, 7168]`, `S` 0-8 | bf16 | contiguous | CUDA |
| `res_weight`, `rms_weight`, `output_rms_weight` | `[7168]` | bf16 | contiguous | CUDA |
| `rms_eps`, `output_rms_eps` | scalar (certified at 1e-6) | Python float | — | — |
| `workspace` | a `K3SandwichWorkspace` of this rank's TP group (see *State*) | — | — | — |
| returns | `(normed, updated)`, each `[M, 7168]` | bf16 | contiguous, newly allocated | = `core.device` |

Single calls are certified at every `M` x `S` in {0, 1, 4, 8} x with and without `prefix`; the call sequences below
take `S` 0-8. The inputs are read only.

Inert (not exposed by the wrapper): `x_slab=None, slab_buf=0, src_slab=None, src_buf=0` — the op can also publish
`normed` into a Lamport slab the next kernel polls, and poll `core` from its producer's slab. Each slab is cross-call
state of its own (three sentinel-armed buffers the caller rotates by the call's ordinal), not part of the workspace;
see *Notes*.

## State

**Object.** `K3SandwichWorkspace`, defined beside the buffer layout it allocates
(`tensorrt_llm/_torch/cute_dsl_kernels/k3_sandwich/op.py`) and re-exported by this entry's wrapper; one per TP
group, owned by the caller.

**Contents and size.** The all-reduce buffer: two alternating halves of `[8 tokens][W ranks][7168]` bf16 per rank,
as int32 words (`0x80000000` = empty; a push writes bf16 `-0.0` as `+0.0`, so a pushed word never reads as empty),
in one multicast allocation (`uc`: this rank's words; `mc`: the same words through the multicast mapping, where the
peers push) — `2 x 8 x W x 7168 x 2` bytes, 0.92 MB at `W` = 4 and 3.67 MB at `W` = 16; `flags`, int32 `[64]`, the
call count of each of the kernel's 56 CTAs (the other 8 words unused); `rank` and `world_size`; the `McastGPUBuffer`
handle that owns the memory (the workspace is valid while the object lives); and the communicator the handles were
exchanged over. The size depends on `W` only, not on `M`: every call fits. Certified after `create`: these sizes,
every word empty, every counter zero.

**Who creates it, and when.** The target, in `post_load_weights`, with
`K3SandwichWorkspace.create(mapping, fabric_handle=None)`:

- collective over `mapping`'s TP group: every rank calls it at the same point;
- failure model:
  - before allocating, the ranks agree that each of them can (not capturing, the buffer within that rank's free
    device memory). If one cannot, every rank raises `RuntimeError` and none allocates (certified: one rank
    capturing while its peers call it eagerly, every rank raises, the capturing rank naming the capture and its
    peers another rank; no rank reaches the allocation, and the next call is correct);
  - a failure that returns from the allocation is agreed the same way;
  - a rank that fails inside the allocation's handle exchange can leave its peers waiting in that exchange; this
    is not turned into an error on the other ranks;
- eager: it allocates and exchanges handles, so it refuses to run under CUDA-graph capture (certified: every rank
  capturing, every rank raises; no stream is left capturing);
- it empties every word and zeroes every counter, and returns only once every rank has (the agreement after the
  allocation), so no peer can push into a buffer its owner has not emptied yet;
- `fabric_handle`: share the memory by fabric handle (required across nodes) or POSIX file descriptor; default
  `mapping.is_multi_node()`. No environment variable is read.

**Which ops may share one object.** The three sandwich entries take the same object — `comm/k3_sandwich_oproj`,
`comm/k3_sandwich_tail` and the drafter's `comm/k3_sandwich_plain` (their wrappers export the one type) — and in the
model they run on one workspace per TP group, target layers and drafter layers alike, so they form one call
sequence. Certified: decode steps of three target layers (this op, then `k3_sandwich_tail`, one of the tails storing
`updated` into a bank row) with two drafter layers (`k3_sandwich_plain`, then its SwiGLU form) interleaved, at
(target `M`, drafter `M`) = (8, 3), (2, 8), (7, 7), a random rank late at every call, every call against its
reference; then that step captured at (8, 4) and replayed 6 times with rewritten inputs, one or two eager calls of
the three ops at other token counts between replays. The sandwich buffer is not the MNNVL workspace: it keeps its own
counters, and a sandwich call does not advance an `MnnvlWorkspace`. Two objects keep independent counters: calls
alternating between two workspaces in an irregular pattern, so that their counters differ, are all correct
(certified, 20 calls).

**Call-order invariant.** Every rank of the group makes the same sequence of sandwich calls on one workspace — the
same number of calls, the `k`-th with the same op and `M` — across layers and decode steps, eager calls and graph
replays alike; and on one stream the same order of calls across objects: each call spins until its peers' rows of
the same call arrive, so two ranks issuing calls on two objects in different orders on one stream deadlock (measured
for `comm/mnnvl_allreduce_attn_res`, see its contract; this kernel waits the same way). On each rank the
calls on one workspace run one after another: a call reads the counters after its grid-dependency wait, which covers
the previous call because every kernel between two calls waits for its predecessor (the kernel's statement) — a
kernel launched under PDL that skips that wait must not sit between two calls.

**What a later launch reads.** `flags`: each CTA reads its counter after its grid-dependency wait and stores it plus
one after its push, so every call adds one to every CTA's counter (certified: all 56 advanced by exactly one per
call, the 8 spare words untouched), and the counter's parity before the call selects the half this call pushes into
and polls. And that half's words, which must be empty except for this call's pushes.

**How it is re-armed.** By its readers: a thread empties every word it read right after reading it. The next push
into that half comes from a peer's call after next, which starts only after this call has ended on this rank (the
kernel's statement), so no separate clear and no record of the previous call's size is needed — the
`k3_spec_accept` failure (a re-arm sized by the current call) cannot occur. The test still drives the sequence that
exposed it (below).

**Why the test drives call sequences.** See `mnnvl_allreduce_attn_res.md` (*State*): Kimi K3's `k3_spec_accept` once
re-armed its Lamport buffer for the current call's rows only, every single-call test passed, and a call sequence
whose row count dipped and grew back caught it; in serving it hung the ranks. This entry's test runs 11 decode steps
of 12 chained layers at `M` = 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8, a random rank 5 ms late at every call, each call
against the reference.

**What a wrong order does.** Certified by the test's negative control: rank 0 issues two same-shaped calls on one
workspace in swapped order. Nothing raises and nothing hangs — the counters still agree — but every rank's two
results are wrong (each call paired with the peers' call at the same position; more than half the `updated` elements
differ on every rank). A plain call right after is correct again. A rank making one call more or fewer than its peers
was not exercised: its later calls would no longer meet their peers' calls of the same position.

## Metadata consumed

Besides `workspace` (an explicit argument), one process-wide cache inside the op module: compiled kernels keyed by
(`W`, publish, input source, PDL), `W` read off the workspace's size. The first call for a key compiles (seconds) and
must be made eagerly — under capture the op raises "must run once per configuration outside CUDA-graph capture
first". The cache is result-neutral. PDL (`TRTLLM_ENABLE_PDL`, read at every call, default on) is part of the key;
it changes scheduling, not results.

## Preconditions

- bf16, contiguous; `core` `[M, 768]` with `M` 1-8, `o_weight` `[7168, 768]`, at most 8 snapshots: the op's
  `supports`. Anything else (e.g. `M` = 9) raises `ValueError` on every rank before any launch: certified (every
  counter unchanged), and the next call is correct. The op does not check the other tensors: `prefix`,
  `block_residual` and the weights must be bf16 `[M, 7168]`, `[S, M, 7168]` and `[7168]` (certified contiguous; the
  op flattens them with `reshape`, which copies a non-contiguous view such as a slice of the snapshot bank).
- Every rank calls with the same `M`, `S` and `prefix` presence, and the same `prefix`, `block_residual` and weights
  (the replicated residual stream) for the same result on every rank; the call order is the *State* invariant.
- `workspace` was created, and the kernel compiled (one eager call per key), before any capture. Calls may be
  captured: certified with a captured step of 12 chained calls at `M` = 8 replayed 8 times with rewritten inputs, an
  eager call of another `M` on the same workspace between replays, every replayed and eager call against the
  reference; and with the shared step above.
- Under PDL the kernel launches its dependents once every CTA has pushed, before its outputs are written (the
  kernel's statement): a kernel launched after it reads `normed` / `updated` only after its own grid-dependency wait.

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, POSIX-fd handles, Kimi K3 TP16 per-rank
  shapes. Test: `tests/unittest/_torch/modeling_v2/comm/_k3_sandwich_oproj_op_matrix.py`, with the helpers of the
  three sandwich entries in `_k3_sandwich_common.py` beside it. The reference is native torch: `core` and `o_weight`
  are small multiples of 1/8 and 1/16, so every partial sum is exact in fp32 and `updated` is compared bit for bit;
  `normed` against an fp32 reference within 2e-2 of its largest magnitude.
- World sizes: the matrix takes `--world-size` and `--launcher` (`mpirun` on one tray, `srun` across trays). The
  kernel sums ranks in chunks of 8, so a run at `W` <= 8 exercises one chunk; Kimi K3 runs `W` = 16 over four trays.
  This entry's 16-rank receipt is pending. The op's kernel test
  (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_sandwich.py`; this op bit for bit against `o_proj` and
  the MNNVL one-shot) passed every case at 16 ranks on four trays in a recorded run: the kernel's record, not this
  entry's receipt.
- State: `mutates_args` names `ws_uc`, `ws_mc` and `ws_flags` — every call pushes through `ws_mc` into every rank's
  `ws_uc`, empties the words it read in `ws_uc` and advances `ws_flags` — and `x_slab`. The op module keeps no
  workspace registry: the caller passes the object it created. The compile cache is the documented process-wide cache
  above. The published / polled slabs (`x_slab`, `src_slab`) are cross-call state without a state object, so they
  stay inert here: proposed, a `K3Slab` state type (three buffers, sentinel-armed) whose rotation index the object
  owns instead of the caller's `slab_buf` ordinal, certified in its own entry.
