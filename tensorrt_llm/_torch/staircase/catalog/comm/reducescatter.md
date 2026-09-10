---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21, world_size: 4}
  sm_103: {status: passed, trtllm: 1.3.0rc26, world_size: 4}
---

# reducescatter

**Wraps** `torch.ops.trtllm.reducescatter` (one call).

## Semantics

Every rank in `group` calls with a tensor of the same full shape. The op sums
them elementwise, splits the sum along **dim 0**, and gives each rank the
slice belonging to **its position in `group`, in ascending rank order**:

```
total  = input_r0 + input_r1 + ... + input_r(G-1)     # elementwise
out_i  = total[offset_i : offset_i + n_i]             # rank at group position i
```

It is **not** in place — the result is a freshly allocated tensor, and
`input` is left untouched, so clobbering `input` right after the call cannot
reach the result.

Only dim 0 is split. Every other dimension is carried through untouched:
`[G*n, 2560]` scatters to `[n, 2560]`, `[G*n, 2, 1280]` to `[n, 2, 1280]`,
and a 1-D `[G*n]` to `[n]`.

The `sizes` argument picks between two splits:

**`sizes = None`** — the split is even. `input.shape[0]` must be a multiple of
`len(group)`; rank at position `i` gets rows `[i*n, (i+1)*n)` where
`n = input.shape[0] / len(group)`.

**`sizes = [n_0, ..., n_(G-1)]`** — rank at position `i` gets `n_i` rows
starting at `sum(sizes[:i])`. The list is in ascending rank order, must be
identical on every rank, and `sum(sizes)` must equal `input.shape[0]`.
Entries may be `0`: a rank that keeps no rows still has to call, and the
reduction is still correct. This is the form attention data parallelism needs,
where per-rank token counts differ by construction.

Both forms are certified. They are **not the same code path**, and NCCL's own
trace shows why: `sizes = None` issues **one `ncclReduceScatter`** whose count
is the elements one rank keeps (rows kept times the trailing dims), while a
`sizes` vector issues **one grouped `ncclReduce` per rank**, rooted at that
rank (for `sizes = [1, 5, 9, 13]` at hidden 2560: counts 2560 / 12800 / 23040 /
33280 rooted at 0 / 1 / 2 / 3, in that one call). An even explicit `sizes`
vector returns bits identical to `sizes = None` and costs the same, while a
genuinely uneven split costs about 1.5x more at 128 rows and above (see
*Notes*). Either way the whole call is bracketed by one
`ncclGroupStart`/`ncclGroupEnd` pair, so one call is **one** position on the
communicator whichever form it takes, and the ragged form's `G` reduces pair
across ranks atomically rather than one root at a time — measured, and matching
the disassembly of `ReducescatterOp::run_list` in `libth_common.so`. That is
what makes the call-order rule in *Preconditions* form-independent.

**It is the inverse of the sibling all-gather, in layout.** The rows this op
hands back to rank `i` are exactly the rows rank `i` would have contributed to
a gather with the same `sizes` and `group` — certified end to end, including
`reducescatter(allgather(x)) == G * x`. What it is *not* is a byte-exact
inverse: the gather moves bytes, this one computes a sum, and the sum is taken
in the input dtype rather than in fp32. Read *Numerics* before relying on the
result of a chain.

**Fusion boundary.** Inside the call: the reduction, the split, and the output
allocation, nothing else. Outside: everything that produced `input` (under
attention DP, this rank's expert-window output over the whole gathered token
set), any residual add, any scaling by routing weights, any quantization, any
padding of row counts to a uniform value, and any reshaping — including the
reshape a caller needs to split along an axis other than dim 0, which this op
cannot do.

**Group membership.** `group` names **ranks in trtllm's MPI session
communicator** (`MPI_COMM_WORLD` under `mpirun`), not device ordinals or
`torch.distributed` ranks. A subset is legal: with `group = [2, 3]` at world
size 4, ranks 2 and 3 reduce only between themselves — rank 2 gets slice 0 and
rank 3 gets slice 1, i.e. the **position in the group**, not the MPI rank
(certified; a subset excluding rank 0 is what pins this). Ranks 0 and 1 must
not call. The list is treated as a **set** — passing `[3, 2, 1, 0]` produces
the identical result to `[0, 1, 2, 3]` — so `group` cannot be used to permute
the split.

## Signature

```python
def reducescatter(
    input: torch.Tensor,
    sizes: Optional[List[int]],
    group: List[int],
) -> torch.Tensor
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `input` | `[rows, ...]`, at least 1-D; `[T, 2560]`, `[T, 2, 1280]` and `[T]` tested | bf16; also fp16, fp32, int32, int64, uint8, int8 | **contiguous** | CUDA |
| `sizes` | `None`, or a list of `len(group)` non-negative ints in ascending rank order summing to `input.shape[0]` | Python `int` | — | host |
| `group` | list of MPI session ranks | Python `int` | — | host |
| returns | `[input.shape[0] // len(group), *input.shape[1:]]` for `sizes = None`, `[sizes[my position], *input.shape[1:]]` otherwise | = `input.dtype` | contiguous, newly allocated | = `input.device` |

There are no other arguments — no reduction operator (the reduction is always
a sum), no strategy, no workspace, no output buffer, no autotuner. Unlike the
sibling all-reduce there is no transport to select.

Every float dtype in the table reduces as an arithmetic sum in **that dtype**
(see *Numerics*). The integer dtypes reduce as integer sums and wrap on
overflow in the narrow ones: four ranks of `int8` 100 came back as `-112`
(measured, outside the test, which keeps its integer payloads inside range).

Dtypes that are **not** in the table divide into two kinds, and both are
traps:

- `torch.float8_e4m3fn` is **accepted and silently wrong**. The op sums the
  e4m3 *bit patterns* as unsigned bytes, wrapping at 256: four ranks each
  holding `1.0` (byte `0x38`) return byte `0xE0`, i.e. `-32.0`, not `4.0`.
  The wrapper rejects the dtype for this reason. Note that the sibling
  all-gather moves fp8 correctly — it only copies — so a post-quantization
  dispatch that works on the way out does **not** work on the way back.
  `torch.bool` is the same shape of trap: the underlying bytes are summed
  (four ranks of `True` leave byte value 4), which torch then reads as `True`,
  so it behaves like an OR. Neither is certified.
- `torch.float64`, `torch.float8_e5m2`, `torch.float8_e4m3fnuz`,
  `torch.float8_e5m2fnuz` and `torch.float8_e8m0fnu` **raise**
  `RuntimeError: unsupported data type (../tensorrt_llm/runtime/torchUtils.h:113)`
  — and that raise is terminal for the whole process. See *Preconditions*.

## Metadata consumed

No attention metadata, no KV cache, no registered layer, no workspace.
Two pieces of process state stand behind the call:

1. **trtllm's MPI session communicator.** The op resolves `group` against it
   and finds this rank's position in it. Under `mpirun` the session is
   `MPI_COMM_WORLD`; a trtllm engine's worker ranks are already inside one, so
   a target's forward needs no setup. World size 1 was not exercised; this
   entry's receipt is world size 4. The sibling op
   `torch.ops.trtllm.reducescatter_pg` takes a `c10d.ProcessGroup` instead,
   for the `TLLM_DISABLE_MPI=1` path; `torch.ops.trtllm.reducescatter_list`
   reduce-scatters a list of tensors in one call. Both are different ops and
   neither is this entry.
2. **The NCCL communicator cache**, keyed by the rank set and shared with the
   other collectives. Built on first use for a given `group`, which makes the
   first call for a group much slower than the rest — and means that first
   call **cannot be inside a CUDA-graph capture** (see *Preconditions*).

## Preconditions

- **Every rank in `group` calls, and their arguments agree**: same dtype, same
  trailing shape, same `sizes` list, and the same `input.shape[0]`.
  Disagreement is **not survivable**: it hangs rather than raising. Measured at
  world size 4, each in its own process — ranks passing different row counts
  with `sizes = None` (three ranks returned a plausible-looking tensor, one
  never came back), and ranks passing different `sizes` vectors (no rank came
  back). Both jobs had to be killed. This is why this entry's test carries an
  external deadline. Do not read the hang as a guarantee, though: both of those
  disagreements change how many bytes a rank moves, and a disagreement that
  leaves the byte counts equal on every rank is silently wrong instead — see
  the call-order bullet below, where that is measured.
- **A rank outside `group` must not call.** Measured: with `group = [0, 1]` at
  world size 4, ranks 0 and 1 returned correctly and ranks 2 and 3 **never
  returned** — they block inside the communicator bootstrap without ever
  reaching the op's own membership check. The job had to be killed.
- `input` is **contiguous**. The op reads `input.numel()` elements from
  `input.data_ptr()` and ignores strides, so a strided view is reduced from the
  wrong bytes and returns a plausible-looking wrong answer with no error. The
  wrapper asserts this.
- `input.dim() >= 1`. A 0-d tensor **segfaults** inside
  `ReducescatterOp::run_list` and kills every rank in the job with no
  diagnostic beyond the fault handler's stack. The wrapper asserts this.
- **The split must cover the input exactly**, and nothing checks it but the
  wrapper:
  - `sizes is None` requires `input.shape[0] % len(group) == 0`. Otherwise the
    op takes `shape[0] // len(group)` rows per rank and the remainder is
    **silently dropped** — 9 rows over 4 ranks returns 2 rows per rank, and the
    9th row is never reduced (measured; the 8 rows that are returned are
    correct).
  - `sizes` given requires `sum(sizes) == input.shape[0]`. A sum smaller than
    the input silently ignores the tail rows (measured: the rows the split does
    cover come back correct). A sum larger than the input makes the op read
    **past the end of the buffer** — observed once, during this entry's test
    bring-up, to return a correctly shaped tensor full of `NaN`.
  - `len(sizes) == len(group)`. One entry short raises
    `IndexError: vector::_M_range_check: __n (which is 3) >= this->size()
    (which is 3)` on the highest rank while the others sit in the collective —
    a raise on one rank and a wedge everywhere else, measured, not recoverable
    inside the job. One entry too long is the out-of-bounds read above.
- `input.dtype` must be one the op supports, and this is a check the caller
  has to make **before** calling rather than by catching. The unsupported
  dtypes raise `RuntimeError: unsupported data type`, but the raise happens
  after the op has opened an NCCL group and before it closes it, so the
  group is left unbalanced and **every collective the process issues
  afterwards silently returns garbage** — measured for this op (99.6% of
  elements wrong on the very next call), and measured for
  `torch.ops.trtllm.allgather` and `torch.ops.trtllm.allreduce` on the same
  group too. It is the process that is finished, not just this entry's
  communicator. Every other failure in this section that returns at all leaves
  the group usable — the test checks that explicitly after the float8,
  non-contiguous and uncovered-split calls.
- `input.device` is the device this rank set with `torch.cuda.set_device`, and
  one rank owns one device.
- **A `group`'s first call must not be made inside a CUDA-graph capture.** It
  is where the group's NCCL communicator gets built, and that build raises
  under capture — see the *CUDA graphs* note for the error text and the
  one-call fix. The cache is shared across collectives, so a group whose
  communicator another op already built is warm for this one.
- **Every rank issues the same sequence of calls on `group`, in the same
  order**, graph replays included. What pairs one rank's call with another's is
  its **position** on the communicator — not the arguments, not the op, not the
  stream — so call order is the caller's whole responsibility, and disagreeing
  about it is *worse* than disagreeing about the arguments, because it does not
  hang. Certified at world size 4 in the configuration of *Notes*:

  | how the ranks disagreed | what happened |
  |---|---|
  | two same-shaped calls issued in swapped order on one rank | **silently wrong on every rank.** No error, no hang, right shape, finite values. Each position returned the sum of whatever payloads met there, matched **bitwise** against a locally computed mix. Both `sizes` forms behave identically. A swapped *pair* puts the positions back, so a plain call straight afterwards is bitwise correct. In this entry's test. |
  | one rank issuing one call more than the others | **silently wrong from the extra call onward, and it does not heal.** Every position after it stays paired one off; there is no resynchronization point inside a stream of collectives. Certified over five consecutive positions, every one of them bitwise equal to its own mix, and alignment came back only once the call counts were equalized. In this entry's test — where the other ranks issue one catch-up call at the end, which is what lets it terminate. What a rank left *permanently* ahead does was not probed: its last call has no partner, so the test would not have returned. |
  | this op swapped against `torch.ops.trtllm.allgather` of the **same** element count on one rank | **silently wrong on every rank**, both calls returning at the shapes their own arguments imply. An even reduce-scatter of `[G*n, H]` and an all-gather of `[n, H]` both move `n*H` elements as NCCL counts them. The pair realigns. In this entry's test. |
  | the same swap where the two calls carry **different** element counts | **wedged.** All four ranks entered the pair, none came out of it, and the job had to be ended from outside. Certified by the test's own second job, which is the only place it can live: a job that wedges never reports. |
  | ranks disagreeing about the **stream** | **correct.** See the next bullet. |

  **How wrong** follows from the fact that this op computes rather than moves.
  A collective that only copies spoils just the block the disagreeing rank
  contributed; here that rank's payload is an addend of **every** element of
  **every** rank's slice, so a single rank out of step makes every rank wrong
  nearly everywhere — measured 0.982-0.993 of elements differing from the
  intended result, on every rank, for every divergence above that returned.
  The residue is elements where the two payloads happened to agree.

  **A mispaired result is deterministic, not noise**, which is the trap. It
  carries no `NaN`, no `Inf` and no shape anomaly, it is bitwise identical
  across repeats, and it is bitwise the **ring-order sum of the addends that
  met** — the accumulation order of *Numerics*, applied to the wrong operands.
  Certified on payloads with no exactness property, where a changed
  accumulation order would show, and against a reference computed locally from
  seeds, which is what makes it reproducible in any process that mispairs the
  same way rather than only in this one (the cross-op mispairing was also
  checked directly, returning identical bits in two separate processes). So a
  divergence cannot be found by re-running the step and looking for
  instability, and under attention data parallelism — where every rank pads to
  `max(all_rank_num_tokens)` and every call therefore moves the same number of
  bytes — it surfaces as a stable accuracy loss and nothing else. **Do not wait
  for a hang to tell you the ranks have diverged.**
- **The stream is the engine's to choose and the ranks need not agree on it.**
  The call runs on `torch.cuda.current_stream()`, and a serving engine moves
  that stream under the model — the same forward runs on torch's graph-capture
  stream while a decode graph is being captured and on the serving stream
  otherwise, so a target cannot pin it. Stream identity plays no part in
  pairing the calls, and none in the reduction order either: certified with
  every rank on a side stream, with one rank on a side stream while the others
  stayed on the default, and with ranks alternating in opposite patterns so
  they disagreed at every call index — every result bitwise correct, including
  against the ring-order chain of *Numerics* on payloads where a changed
  accumulation order would show. Certified with the side stream joined to the
  current one on both ends, which is what the engine and a target's forward
  both do; two calls on one `group` running *concurrently* on two streams of
  the same rank was not exercised.

## Notes

The certified path: `mpirun`-launched ranks whose session communicator is
`MPI_COMM_WORLD`, one rank per B200 (sm_100), world size 4, group
`[0, 1, 2, 3]` (subsets `[0, 1]` and `[2, 3]` for the membership claims),
bf16 hidden 2560 unless the dtype table says otherwise, NCCL as the only
transport. There is no strategy, workspace or autotuner state here, so the
only axis with two execution paths is `sizes`, and both are certified.

**Who else is on the communicator, inside a serving engine.** Nobody — so the
only calls whose order has to line up are the ones the caller makes. Measured
on this machine by tracing a live 4-rank trtllm serving engine under attention
data parallelism with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL`: one NCCL
communicator per rank, 14,036 collectives on it over 242 forwards (7,018
all-gathers and 7,018 reduce-scatters, all issued by the model, none by the
runtime), and per-rank sequences bit-identical on (op kind, element count,
stream index) position by position. The runtime's own cross-rank step under
attention DP — agreeing `all_rank_num_tokens` once per forward — is a
**host-side MPI** collective on a communicator it builds itself, congruent to
the session communicator but never identical to it, and never NCCL. That trace
also showed the engine issuing a model's collectives on **two** different
streams (torch's graph-capture stream during capture, the serving stream
otherwise), which is why the stream bullet in *Preconditions* matters. This
paragraph is measured evidence from this repo's own runs, not something this
entry's test carries — the test certifies the pairing rule itself, in isolation.

**Numerics — this op computes, so the result is not the exact sum.** All of
the following is measured at world size 4, bf16, on payloads with no exactness
property (standard normal cast to bf16):

- It **is deterministic**. Eight identical eager calls return bitwise
  identical tensors; eight replays of a captured call return bitwise identical
  tensors and agree bitwise with the eager result; and — in probing, outside
  the test — three separate 4-rank processes returned byte-identical results
  for the same payloads at 1, 16, 256 and 2048 rows and for a ragged split. A
  target's accuracy gate can therefore expect run-to-run reproducibility.
- It is **not** the correctly rounded exact sum. About a third of the elements
  differ from an fp32 reduction rounded once to bf16 (measured 0.33 at 1, 8,
  256 and 2048 rows).
- It **is a sequential summation in the input dtype**, in ring order starting
  at the destination's successor: for the rank at group position `i` the order
  is `x_{i+1} + x_{i+2} + ... + x_{i+G-1} + x_i` (indices mod `G`), each
  partial sum rounded to the input dtype. Matched bitwise on every element at
  1, 8 and 256 rows with hidden 2560, and at hidden 64. A consequence worth
  stating plainly: **two ranks holding identical rows can get different
  answers**, because their orders differ. With rank 0 holding `1.0` and the
  other three holding `2^-9`, rank 0 came back with `1.0078125` (its order
  accumulates the three small terms first) and rank 2 with `1.0` (its order
  starts from the large one and the small terms round away).
- Its distance from the exact sum stays inside the classical bound for a
  sequential sum of `G` terms, `(G-1) * u * sum_r |x_r|` elementwise, with
  `u = 2^-8` for bf16, `2^-11` for fp16 and `2^-24` for fp32. Largest observed
  ratio to that bound: 0.89 for bf16 (over 1, 8, 64, 256 and 2048 rows), 0.82
  for fp16, 0.80 for fp32.
- The ring order is NCCL's algorithm choice for this topology, not a promise
  of the op's interface. The entry's test asserts it, so a change shows up as
  a test failure rather than as drifting accuracy.

Because of the exactness of small dyadic values, everything else in this
entry's test is asserted **bitwise** (`rtol=0, atol=0`): its payloads are
multiples of 1/8 bounded so that every partial sum, in any order, is exactly
representable. That is a tightening of the default tolerances, not a
loosening — but it is only available to a test that controls its inputs. A
caller's real activations get the bullets above.

**CUDA graphs — the surface a decode step actually needs.** Certified at world
size 4, group `[0, 1, 2, 3]`, bf16, under torch's default
`capture_error_mode="global"`, with every rank capturing and replaying the
same graph:

| form | under capture |
|---|---|
| `sizes = None` (even) | certified at **all 35 per-rank row counts `1..32, 64, 128, 256`** |
| `sizes = [...]` (uneven) | certified — but the split is frozen at capture, see below |

A trtllm engine instantiates one decode graph per configured batch size —
with `cuda_graph_config.max_batch_size = 256` and no explicit `batch_sizes`
list, 35 of them at `1..32, 64, 128, 256` — keeps all of them alive in one
memory pool, and replays them interleaved as the served batch size moves.
Under attention DP a decode call's per-rank row count *is* that batch size, so
this op's captured input is `G` times it. The whole set is certified: all 35
captured into one shared pool, none released, then replayed in four orders —
ascending, descending, shuffled, and largest-jump-first (`256, 1, 128, 2, 64,
3, ...`) — each order twice, once one replay at a time with a synchronize and
a full check between graphs, and once with all 35 issued back to back and
nothing synchronizing between them. Every replay's payload is one no earlier
call used, and every result is bitwise equal to the arithmetic reference. The
same is certified with **29 independent reduce-scatters inside each of the 35
graphs** — 1015 captured collectives in one pool, the shape this checkpoint's
decode graph has (30 layers, layer 0 dense, so 29 expert-parallel MoE calls
each followed by one reduce-scatter).

A replay re-runs the collective over whatever the input buffer holds at replay
time and writes into the same tensor the capture returned — keep it and read
it after each `replay()`. Replays also stay correct with eager traffic in
between: an eager even reduce-scatter at 1500, 2048 or 8192 rows per rank and
an eager uneven one between every pair of replays, all bitwise correct, which
is the shape a server has when a prefill runs between two decode steps.

**`sizes` is frozen into a graph; row counts must be padded to capture one.**
`sizes` is a host-side argument, so a capture bakes in the split it was given
and a replay re-runs *that* split. Certified by capturing two graphs with
different sizes vectors into one pool and replaying them in both orders across
two payload rounds: each keeps its own row split and its own output length.
The consequence for an attention-DP target is concrete — a graph-captured
decode step must pad every rank to the same row count and pass `sizes = None`,
because the uneven split cannot vary per replay; the uneven form belongs to
the eager (non-captured) steps.

**A `group`'s first-ever call cannot be captured**, because it is where the
NCCL communicator is built. Four ranks capturing their first call each raised,
out of the op:

```
RuntimeError: Failed, NCCL error ../tensorrt_llm/common/opUtils.cpp:175
'unhandled cuda error (run with NCCL_DEBUG=INFO for details)'
```

which invalidates the capture, so the `with torch.cuda.graph(...)` block then
raises

```
AcceleratorError: CUDA error: operation failed due to a previous error
during capture
```

(`cudaErrorStreamCaptureInvalidated`). It is survivable, and the fix is one
line: make one eager call per `group` before any capture. After the failure
the same group reduce-scatters correctly eagerly, and a capture taken after
that replays correctly — both certified. One caveat for anyone catching it:
torch's graph context manager ends the capture before it restores the stream,
so a failed capture leaves its own stream current — put the resting one back
with `torch.cuda.set_stream`.

**Cost: the even split is the cheap one.** Measured on the certified path
(world size 4, bf16, hidden 2560, 200 timed iterations after 20 warm-up calls,
CUDA events, eager), per-rank row count `n`, all times per call:

| `n` | `sizes = None` | even `sizes` vector | uneven `sizes` vector |
|---|---|---|---|
| 1 | 23.9 us | 24.1 us | 10.2 us (degenerate: `[4, 0, 0, 0]`) |
| 8 | 24.5 us | 24.3 us | 28.1 us |
| 128 | 31.0 us | 30.7 us | 47.2 us |
| 2048 | 163.4 us | 163.6 us | 248.7 us |

`sizes = None` and an even explicit vector are indistinguishable (ratio
1.00-1.01) and return identical bits, so a caller with even counts loses
nothing by passing them. A genuinely uneven split costs about 1.5x from 128
rows up — so unlike the sibling all-gather, where the two forms measured
identical, here padding to an even split is a small win on cost as well as
being the only form a graph can replay. Below ~128 rows the call is
launch-bound (24-25 us across a 128x message-size range), which is the regime
a decode step runs in.

**Splitting along another axis is the caller's problem.** This op only splits
dim 0. trtllm's module-level helper reaches other axes by reshaping around
this call — `chunk`/`split` along the target dim, `reshape`, `cat`, then a
`view` of the result — which is Python-level composition and would have to be
written from catalog entries, not hidden inside a wrapper. For the attention-DP
use (return each rank its own token rows after an expert-parallel MoE call)
dim 0 is already the token axis and no reshape is needed.

**The forward op exists.** `torch.ops.trtllm.allgather` has the same
`(input, sizes, group)` signature and is the other half of the round trip —
it is a different op and is not this entry.
