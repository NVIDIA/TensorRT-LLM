---
receipts:
  sm_100: {status: pending, world_size: 4}
---

# k3_latent_reduce

**Wraps** `torch.ops.trtllm.k3_latent_reduce` (one call).

A stateful entry: its correctness depends on a caller-owned state object, `K3LatentExchange`, passed as the last
argument and described under *State*. The op is the consumer half of a collective; the producers that push into the
exchange are not part of this entry (see *Notes*).

## Semantics

Kimi K3's latent all-reduce at decode size (at most 8 tokens), split in two. On every rank of a TP group of `W`
ranks, the routed experts' push-only kernel (the producer) stores this rank's routed partial rows into every rank's
exchange through a multicast mapping, and exits; this op then waits until every rank's rows of the same call are in
its own copy and sums them. Every rank gets back the same rows. Per call of `M` tokens:

```
p_r = rank r's routed partial [M, 3584] bf16, as r's producer pushed it (-0.0 stored as +0.0)
c_k = fp32 sum of p_r over r = 8k .. min(8k + 8, W) - 1, in rank order, from +0        # chunks of 8 ranks
out = bf16(c_0 + c_1 + ..., in order, from +0)                    # round to nearest even; [M, 3584]
```

This is the order of the MNNVL one-shot all-reduce (`reduceOneshotLamport`,
`cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu`), so `out` equals the one-shot all-reduce of
the partials bit for bit (the kernel's statement). Certified at every `M` from 1 to 8: `out` equals
`comm/mnnvl_fusion_allreduce` sent one-shot over an `MnnvlWorkspace`, and a torch reference in that order, bit for
bit, including for partials whose sum another order changes in at least a quarter of the elements. The result is
bitwise identical on every rank (certified for every call of the test outside its negative controls).

Fusion boundary. Inside: waiting for every rank's rows of this call, the sum, the output, emptying the words it read
and advancing the exchange's call count. Outside: computing the routed partial and pushing it (the producer), the
shared experts, and whatever consumes `out` (the routed latent's up-projection). Steps of more than 8 tokens need
another all-reduce of the partial.

## Signature

```python
def k3_latent_reduce(num_tokens: int, exchange: K3LatentExchange) -> torch.Tensor
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `num_tokens` | `M`, 1-8 (all certified) | Python int | — | host |
| `exchange` | a `K3LatentExchange` of this rank's TP group, `W` = 4 certified (see *State*) | — | — | — |
| returns | `[M, 3584]` | bf16 | contiguous, newly allocated | = `exchange.uc.device` |

The summands are not arguments: they are rows `0..M-1` of every rank's slot in this call's half of the exchange,
written by the producers. The op runs on the current stream of `exchange.uc`'s device. Inert (not exposed by the
wrapper): the op's `ctas_per_token` = 0, i.e. 4 CTAs per token row up to 8 ranks and 14 at 16. The op also takes 4,
14 or 28, which its own test (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_latent_reduce.py`) runs; only
the default is certified here.

## State

**Object.** `K3LatentExchange`, defined beside the buffer layout it allocates
(`cute_dsl_kernels/k3_fused_moe/latent_op.py`) and re-exported by this entry's wrapper; one per TP group, owned by
the caller. A state type, not an entry: it launches nothing per call.

**Contents and size.**

- `uc`: int32 `[2 x 8 x W x 1792]`, this rank's buffer: two halves of 8 token rows of `W` rank slots of 3584 bf16,
  stored in pairs as int32 words (the even column in the low 16 bits). The word `0x80000000` is empty (not written).
  Every word is empty after `create` and again after every call (certified).
- `mc`: the same words through the multicast mapping; a store through `mc` lands in every rank's `uc`.
- `flags`: int32 `[4]`, this rank's own: `[0]` the op's call count, whose parity is the half the next push and
  reduce use; `[2]` the arrivals of the running reduce's CTAs, 0 between calls; `[1]` and `[3]` unused, 0. After
  `create` all four are 0, and after every call they read `[count, 0, 0, 0]` (certified).
- `handle`, the `McastGPUBuffer` that owns the memory (the exchange is valid while it lives); `comm`, the TP-group
  communicator the handles were exchanged over; `rank`, `world_size`.

The views are `2 x 8 x W x 1792 x 4` bytes per rank (certified): 448 KiB at `W` = 4, 896 KiB at 8, 1.75 MiB at 16;
the `McastGPUBuffer` rounds the allocation up to its multicast granularity. The size depends on `W` only: every call
of up to 8 tokens fits.

**Who creates it, and when.** The target, in `post_load_weights`, with
`K3LatentExchange.create(mapping, fabric_handle=None)`:

- collective over `mapping`'s TP group: every rank calls it at the same point. Every rank first joins the split of
  the TP group's communicator, so a rank that calls it while its peers do not waits for them there;
- failure model (`k3_fused_moe.op.create_mcast_state`, the same as `MnnvlWorkspace.create`'s):
  - before allocating, the ranks agree that each of them can (not capturing, the buffer within that rank's free
    device memory). If one cannot, every rank raises `RuntimeError`, none allocates, and under MPI each frees the
    communicator it split for the call (certified: every rank capturing, and one rank capturing while its peers
    call it eagerly at the same point; every rank raises at that agreement and frees its split, the capturing
    rank's message naming the capture, and the next call is correct);
  - a failure that returns from the allocation is agreed and handled the same way;
  - a rank that fails inside the allocation's handle exchange can leave its peers waiting in that exchange; this
    is not turned into an error on the other ranks;
- eager: it allocates and exchanges handles, so it refuses to run under CUDA-graph capture (every rank raises);
- it empties every word and zeroes `flags`, and returns only once every rank has done so (the agreement after the
  allocation is the barrier), so no producer can push into a buffer before its rank has armed it; armed and sized on
  every rank right after `create` is certified;
- `fabric_handle`: share the memory by fabric handle (required across nodes) or POSIX file descriptor; default
  `mapping.is_multi_node()`. No environment variable is read.

`create` raises `ValueError` for a TP size other than 4, 8 or 16, on every rank and before any collective step (the
size is the same on every rank of the group; code).

**Which ops may share one object.** One call's producers and this op. The producers are the push-only builds of the
routed experts (`k3_moe_m1` / `k3_moe_m2` push, `trtllm::k3_fused_moe_push`, `trtllm::k3_fused_moe_front_push`;
not in this tree yet): they take `exchange.push_args()` (`uc`, `mc`, `flags`, `rank`), read `flags[0]`, store
through `mc` and write no flag. One exchange serves every MoE layer of a TP group: all the layers' calls form one
sequence. It is not the MNNVL workspace (`MnnvlWorkspace`), the sandwich workspace, or the sandwich tail's latent
exchange (`K3SandwichLatentExchange`: the same buffer layout, but its own flags); a push must be reduced from the
exchange it went into. Two exchanges keep independent counts and buffers: calls alternating between two exchanges
in an irregular pattern, so that the halves they use differ from call to call, are all correct, and each exchange
ends clean at its own count (certified, 20 calls). They are not independent orders: a reduce spins until every
rank's push of the same call has landed, and a stream runs its kernels one after the other, so ranks that issue
calls on two exchanges (or this op and another polling collective) in different relative orders on one stream
deadlock (the kernel's behaviour; not run).

**Call-order invariant.** On one exchange every rank makes the same sequence of calls, across layers and decode
steps, eager calls and graph replays alike. A call is one push of `M` rows by the rank's producer followed by exactly
one reduce of the same `M`, before the next push; the `k`-th call has the same `M` on every rank. A push never waits
for the peers; only the reduce does. A producer reads the half from `flags[0]` after its grid-dependency wait,
and the op reads it before its own (its CTA 0 advances the count at the very end, once every CTA has read it). So
every kernel launched between a reduce and the next push on a stream must end only after its predecessor has ended:
it calls `griddepcontrol.wait`, or it launches without programmatic dependent launch. Otherwise a push could read
the count before that reduce has advanced it and write into the half the reduce is still reading (the op's
statement, `latent_op.py`).

**How far ranks run apart.** Only the reduce waits, so a rank's host may enqueue any number of calls ahead of its
peers, but its GPU runs at most one call ahead of the slowest peer: its push of a call lands only after its reduce of
the previous call has ended, which waited for every rank's push of that call, and each of those landed after that
rank's reduce of the call before. So a push lands only in a half that every rank has finished reading. Certified
with the test's pushes, which are copies issued after the previous reduce on the same stream: one rank enqueues a
whole step of 12 calls (`M` dipping and growing back) and waits until its first push has completed, while its peers
are held at a host barrier. Before they issue anything, every peer finds that push in its buffer and the other half
empty (the rank's second push is held behind its first reduce). Then the converse, every rank but one a whole step
ahead of it. Every result is correct and the exchange ends clean. The random late rank of the sequence below adds
per-call skew on top. Not certified here: a producer launched as a programmatic dependent starts before the previous
reduce has ended and stores only after its grid-dependency wait; that ordering, and the condition above that it
relies on, are not exercised by the copies.

**What a later launch reads.** Before its grid-dependency wait the op reads `flags[0]` (its half) and counts each
CTA into `flags[2]`. After the wait it reads rows `0..M-1` of every rank's slot of that half, polling until none of
the words is empty. At its very end CTA 0 waits until all `M x CTAs` CTAs have counted in, then sets `flags[2]` back
to 0 and `flags[0]` to the count plus one. Rows `M..7` are neither summed nor emptied (certified by the second
negative control below). The next producer reads the advanced `flags[0]`. The count is int32 and only its parity is
read: it passes from 2^31 - 1 to -2^31 without consequence (certified: four calls across the wrap from a preset
count).

**How it is re-armed.** By the op: each thread empties every word it read (its 16-byte vector of every rank's slot)
after storing its output. Nobody pushes into that half again before every rank's reduce of it has ended: the next
push into it is two calls later, and on each rank it follows that rank's reduce of the call in between, which waits
for every rank's push of that call, each launched after that rank's reduce of this call (the kernel's statement; it
relies on the condition on programmatic dependent launch above). So no separate clear and no record of an earlier
call's size is needed, as long as every push is reduced with its own `M`: rows a reduce does not sum stay as they
are (*What a wrong order does*). Certified by the sequences below: after every call of the single-call check and
after every decode step, both halves are empty on every rank.

**Why the test drives call sequences.** A re-arm that depends on the call's size passes every single-call test: it
fails only after a smaller call, when an older, larger call's words are still in the buffer and a later larger call
reads them as fresh rows. This entry's test therefore runs 11 decode steps of 12 layers at `M` = 8, 8, 8, 2, 7, 8, 1,
1, 8, 3, 8, each step's calls queued without host synchronization, a random rank 5 ms late before every call (its
push lands while the others' reduces poll), every call against the reference.

**What a wrong order does.** Two negative controls at `W` = 4, both certified:

- Swapped calls. Rank 0 makes two same-shaped calls in swapped order (it pushes its partial of the second call
  first). Nothing raises and nothing hangs, since the counts still agree, but every rank's two results are wrong:
  the `k`-th reduce sums every rank's `k`-th push, so each pairs rank 0's partial of one call with the others' of
  the other (bit for bit), and more than half of the elements differ from the intended call's on every rank. The
  exchange is clean afterwards and a plain call right after is correct.
- A token count that differs from the push. Every rank pushes 8 rows and rank 0 reduces 4: its 4 rows are right,
  but rows 4-7 of that half stay full in its buffer. Two calls later every rank pushes 4 rows into that half and rank
  0 reduces 8: it does not wait for rows 4-7, which are already full, and returns the older call's sums for them,
  bit for bit. Nothing raises or hangs; that reduce empties all 8 rows, the exchange is clean again and the next call
  is correct.

Not run, because each waits forever (the op polls until no word it reads is empty): a reduce of rows nobody pushed
(with nothing stale there), a push into the other half, and a rank making one call more or fewer than its peers
(from then on its count's parity differs from theirs).

## Metadata consumed

Besides `exchange` (an explicit argument), one process-wide cache inside the op: compiled kernels keyed by (`W`,
CTAs per token, PDL). `M` is a runtime argument, so one compile serves every `M` (the op's code). The first call for
a key compiles (seconds) and must be eager: under capture it raises `RuntimeError` ("must run once outside
CUDA-graph capture first") before launching anything (certified: the test's first call is made under capture on
every rank, and the exchange is untouched). The cache is result-neutral. `TRTLLM_ENABLE_PDL` (default `1`), read at
every call, is part of the key: it decides whether the kernel launches as a programmatic dependent, which changes
scheduling, not results (the op's statement; the test runs the default).

## Preconditions

- `M` 1-8 and `W` 4, 8 or 16 (the op's `supports`). `M` = 0 or 9 raises `ValueError` on every rank before the op
  touches the exchange: certified, with the count unchanged, every word still empty and the next call correct.
- Every rank calls with the same `M`, the token count its producer pushed in this call; the call order is the
  *State* invariant, including its condition on programmatic dependent launch.
- The producers push rows `0..M-1` of this rank's partial into slot `[rank]` of half `flags[0] & 1` of every rank's
  buffer through `mc`, as bf16 pairs in int32 words with -0.0 stored as +0.0. A word `0x80000000` (-0.0 in the odd
  column, +0.0 in the even one) reads as not written, so a producer that stored one would make every rank's reduce
  wait forever (the kernel's code). The test's emulated pushes store exactly this, -0.0 entries included; columns
  that are -0.0 on every rank sum to +0.0 (certified).
- `exchange` was created, and the kernel compiled by one eager call, before any capture. Calls may be captured:
  certified with two captured steps on one exchange, 12 push + reduce pairs at `M` = 8 and at `M` = 3, replayed
  alternately 4 times each with rewritten partials and a random rank late, two eager calls of other `M` between
  replays, every replayed and eager call against the reference.

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, POSIX-fd handles, 4 CTAs per token row.
  Test: `tests/unittest/_torch/modeling_v2/comm/_k3_latent_reduce_op_matrix.py`, collected by
  `test_modeling_v2_k3_latent_reduce_op_matrix.py`. The reference is native torch in the one-shot's order. Most
  checks use partials that are multiples of 1/16, whose sum is exact in any order; the bit-identity check adds
  partials with a +B / -B pair per element, whose sum depends on the order (summing the ranks in reverse, or at `W` >
  8 without chunks, must change at least a quarter of the elements), and normal-distributed ones. Every comparison is
  bit for bit, and every output is also compared across the ranks.
- The pushes are emulated. The producers are not in this tree yet, so the test stores each rank's partial as they
  do, with a tensor copy through `mc` into the call's half. Not reached by the emulation: a producer reading the half
  on the device (the test takes it from its own count of the calls, checked against `flags[0]` each time it checks
  the exchange clean), and a producer still running under programmatic dependent launch when the reduce starts (the
  copies launch without it). The emulated pushes also fix their half when captured, so each captured step holds an
  even number of calls and an even number of eager calls runs between replays; a producer reads the half at replay
  time.
- World size: the matrix takes `--world-size` and `--launcher` (`mpirun` on one node, `srun` across nodes). Kimi K3
  runs the op over 16 ranks on four trays, where the kernel sums the ranks in two chunks of 8 and uses 14 CTAs per
  token row; a 4-rank run reaches neither. The 16-rank receipt is pending; `W` = 8 is not run.
- The op writes `lat_uc` (it empties words) and `lat_flags`, and its schema declares both mutable; the producers'
  writes through `mc` are theirs to declare. The compile cache is a module-level dict (result-neutral, above).
