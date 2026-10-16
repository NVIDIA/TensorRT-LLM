---
receipts:
  sm_100: {status: passed, world_size: 4}
---

# mnnvl_allreduce_attn_res

**Wraps** `torch.ops.trtllm.mnnvl_allreduce_attn_res` (one call).

A stateful entry: its correctness depends on a caller-owned state object, `MnnvlWorkspace`, passed as the last
argument and described under *State*.

## Semantics

A one-shot all-reduce over a TP group's MNNVL multicast workspace with Kimi K3's residual update as its epilogue.
Every rank of the group calls with its own rows `input` `[T, H]`; every rank gets back the same two tensors. Per
token, with `W` ranks:

```
r       = bf16(sum over the W ranks of input)              # every rank sums every rank's rows, fixed order
updated = bf16(prefix_sum + r)                             # r alone when prefix_sum is None
v       = [block_residual[0], ..., block_residual[S-1], updated]                  # S + 1 candidates
score_c = sum_h rmsnorm(v_c)[h] * rms_weight[h] * res_weight[h]    # rmsnorm: v_c / sqrt(mean(v_c^2) + rms_eps)
p       = softmax over the candidates of score
normed  = RMSNorm(sum_c p_c v_c; output_rms_weight, output_rms_eps)
```

and returns `(normed, updated)`. The selection is HF's `modeling_kimi._apply_attn_res`; the op rounds like the
unfused all-reduce followed by `trtllm::attn_res_add_rmsnorm_fwd` (its kernel's statement). The reduction order is
fixed: the result is deterministic and bitwise identical on every rank (certified, every call of the test).

Fusion boundary. Inside: the all-reduce, the residual add, the attention-residual selection, the RMSNorm. Outside:
whatever produced `input` (a row-parallel projection), keeping the snapshot bank `block_residual` (which layers push
a snapshot, and when), and `prefix_sum`'s chaining from layer to layer.

## Signature

```python
def mnnvl_allreduce_attn_res(
    input: torch.Tensor,
    prefix_sum: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    workspace: MnnvlWorkspace,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `input` | `[T, H]`; `H` = 7168 certified (the op takes a multiple of 1024 up to 8192); `T` 1-8 and 16 certified, at most `workspace.max_one_shot_tokens(H)` | bf16 | contiguous | CUDA, this rank's device |
| `prefix_sum` | `None`, or `[T, H]` | bf16 | contiguous | CUDA |
| `block_residual` | `[S, T, H]`, `S` = 0..11 (all certified) | bf16 | contiguous | CUDA |
| `res_weight`, `rms_weight`, `output_rms_weight` | `[H]` | bf16 | contiguous | CUDA |
| `rms_eps`, `output_rms_eps` | scalar | Python float | — | — |
| `workspace` | an `MnnvlWorkspace` of this rank's TP group (see *State*) | — | — | — |
| returns | `(normed, updated)`, each `[T, H]` | bf16 | contiguous, newly allocated | = `input.device` |

`input`, `prefix_sum` and `block_residual` are read only. The op writes the workspace's buffers and flag words; its
schema declares both mutable.

## State

**Object.** `MnnvlWorkspace` (`catalog/comm/mnnvl_workspace.py`), one per TP group, owned by the caller. A state
type, not an entry: it launches nothing per call.

**Contents and size.** Three Lamport buffers of `buffer_bytes` each in one multicast allocation (this rank's
unicast view as `lamport`; every word `-0.0` when armed), the flag words `buffer_flags` (uint32 `[9]`: current
buffer, dirty buffer, bytes per buffer, dirty stages, bytes to clear x 4, access count), the `McastGPUBuffer`
handle that owns the memory, and the communicator the handles were exchanged over. A call of `T` tokens pushes
`T x H x W x 2` bytes into one buffer, so `buffer_bytes` must cover the largest call: Kimi K3 uses 4 MiB, i.e.
`T` <= 73 at `W` = 4 and `T` <= 18 at `W` = 16 for `H` = 7168 (`max_one_shot_tokens`). The workspace is never grown;
a call over one buffer raises (below).

**Who creates it, and when.** The target, in `post_load_weights`, with
`MnnvlWorkspace.create(mapping, buffer_bytes, fabric_handle=None)`:

- collective over `mapping`'s TP group only: every rank of the group calls it at the same point, and ranks outside
  the group take no part (certified: under TP W/2 x PP 2, one group creates a workspace while the other makes no
  MNNVL call);
- failure model:
  - before allocating, the ranks agree that each of them can (not capturing, a valid `buffer_bytes`, the three
    buffers within that rank's free device memory). If one cannot, every rank raises `RuntimeError`, none
    allocates, and under MPI each frees the communicator it made for the call (certified: one rank short of
    memory, every rank raises and frees that communicator, the next call is correct);
  - a failure that returns from the allocation is agreed and handled the same way;
  - a rank that fails inside the allocation's handle exchange can leave its peers waiting in that exchange; this
    is not turned into an error on the other ranks;
- eager: it allocates and exchanges handles, so it refuses to run under CUDA-graph capture (every rank raises);
- it arms every buffer word and the flags, and returns only once every rank has armed its buffers, so no peer can
  push into memory a rank has not armed;
- `fabric_handle`: share the memory by fabric handle (required across nodes) or POSIX file descriptor; default
  `mapping.is_multi_node()`. No environment variable is read.

**Which ops may share one object.** Every MNNVL one-shot op that takes the pair (`comm_buffer`, `buffer_flags`):
this entry and `trtllm::mnnvl_fusion_allreduce`. Ops that share one object share one rotation: their calls form one
sequence, interleaved in one order on every rank. Two objects are two independent rotations: calls alternating
between two workspaces in an irregular pattern, so that the two positions differ, are all correct (certified, 20
calls). They are not independent *orders*, though: each call spins until its peers' rows of the same call arrive and
the calls of one stream run one after the other, so ranks that issue calls on two workspaces in different relative
orders on one stream deadlock (measured at `W` = 4: rank 0 issued a pair B-then-A, the others A-then-B; every GPU
spun at 100 % until the job was cancelled). The order of all of a stream's collectives must agree across ranks,
whatever object each belongs to.

**Call-order invariant.** Every rank of the group makes the same sequence of calls on one workspace — the same
number of calls, the `k`-th call on every rank with the same `T` — across layers and decode steps, eager calls and
graph replays alike; and on one stream, the same order of calls across workspaces (above). Each call takes the next
buffer (the current one plus one, mod 3), pushes its rows into that buffer on every rank, and polls its own copy
until every rank's rows of *this* call are there.

**What a later launch reads.** `buffer_flags`, which the previous call left: the current buffer, the dirty buffer
(the previous call's) and the bytes the previous call wrote into it; and the Lamport words of its own buffer, which
must all be `-0.0` except for this call's pushes.

**How it is re-armed.** Each call clears the previous call's buffer by the bytes the previous call recorded, not by
its own size, and records its own (`cpp/tensorrt_llm/common/lamportUtils.cuh`, `LamportFlags`). So after a call with
fewer tokens than an earlier one, every word the earlier call pushed is cleared before that buffer comes round again.
Certified by the sequence below.

**Why the test drives call sequences.** A re-arm sized by the current call instead passes every single-call test:
it fails only after a smaller call, when an older, larger call's words stay in the buffer and a later larger call
reads them as fresh rows whenever a peer has not pushed yet. In serving, that makes ranks disagree, then hang. This
entry's test therefore runs that shape of sequence: 12 decode steps of 12 chained layers at `T` = 8, 8, 8, 2, 7, 8,
1, 1, 8, 16, 3, 8, a random rank 5 ms late at every call, each call against the reference.

**What a wrong order does.** Measured at `W` = 4 (the test's negative control): rank 0 issues two same-shaped calls on
one workspace in swapped order. Nothing raises and nothing hangs — the rotation positions still agree — but every
rank's two results are wrong (each call paired with the peers' call at the same position; more than half the
`updated` elements differ on every rank). A plain call right after is correct again: a swapped pair realigns the
positions. A rank making one call more or fewer than its peers was not exercised; its positions never realign.

## Metadata consumed

None besides `workspace`, which is an explicit argument. The op keeps no process cache and compiles nothing.

## Preconditions

- bf16, contiguous, `input` 2-D; `H` a multiple of 1024 and at most 8192; `W` in {2, 4, 8, 16}; `S` <= 11.
- `T x H x W x 2 <= workspace.buffer_bytes`. A call over one buffer raises `RuntimeError` ("the one-shot footprint
  ... exceeds one Lamport buffer") on every rank before it touches the workspace: certified, and the next call is
  correct.
- Every rank calls with the same `T`, `S` and `prefix_sum` presence; the call order is the *State* invariant.
- `workspace` was created before any capture. Calls may be captured: certified with a captured step of 12 chained
  calls at `T` = 8 replayed 8 times with rewritten inputs, an eager call of another `T` on the same workspace between
  replays, every replayed and eager call against the reference.

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, POSIX-fd handles, `H` 7168. Test:
  `tests/unittest/_torch/modeling_v2/comm/_mnnvl_allreduce_attn_res_op_matrix.py` (the reference is native torch:
  the sum's inputs are multiples of 1/16, so `updated` is exact whatever the summation order and is compared bit for
  bit; `normed` against an fp32 reference within 2e-2 of its largest magnitude).
- World size: the matrix takes `--world-size` and `--launcher` (`mpirun` on one node, `srun` across nodes). Kimi K3
  runs the op over 16 ranks on four GB200 trays, where the kernel sums ranks in chunks of 8, which a 4-rank run never
  reaches. At 16 ranks (four trays, fabric handles) the same kernel passed a recorded check of 110 cases (`T` 1-8,
  16, 32, 64; 0, 1, 3, 8 and 11 snapshots; with and without `prefix_sum`) against the unfused all-reduce followed by
  `attn_res_add_rmsnorm_fwd`; this matrix itself has not run at 16 ranks.
- The op finds its multicast mapping by looking `comm_buffer`'s data pointer up in a process registry of multicast
  buffers, which the workspace's handle keeps registered, rather than taking the handle as an argument (as
  `mnnvl_fusion_allreduce` does).
- `MNNVLAllReduce` keeps its own workspaces in a dict keyed by `Mapping`, grown on demand by the first eager call
  that needs more. This entry takes the explicit object instead; its size is the caller's decision, made at
  construction.
