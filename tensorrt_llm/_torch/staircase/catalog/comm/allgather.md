---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21, world_size: 4}
  sm_103: {status: passed, trtllm: 1.3.0rc26, world_size: 4}
---

# allgather

**Wraps** `torch.ops.trtllm.allgather` (one call).

## Semantics

Every rank in `group` calls with its own slice of a tensor; every rank gets
back the whole thing, concatenated along **dim 0 in ascending rank order**:

```
out = cat([input_r0, input_r1, ..., input_r(G-1)], dim=0)
```

It is **not** in place — the result is a freshly allocated tensor, and
`input` is left untouched, so clobbering `input` right after the call
cannot reach the result.

Only dim 0 is a gather axis. Every other dimension is carried through
untouched: `[n, 2560]` gathers to `[G*n, 2560]`, `[n, 2, 1280]` to
`[G*n, 2, 1280]`, and a 1-D `[n]` to `[G*n]`. Nothing is computed — the
call moves bytes, so the result is bit-for-bit the inputs' concatenation
in every dtype below.

The `sizes` argument picks between two forms:

**`sizes = None`** — every rank must hold the same number of rows `n`, and
`out.shape[0] = n * len(group)`.

**`sizes = [n_0, ..., n_(G-1)]`** — rank `i` contributes `n_i` rows and
`out.shape[0] = sum(sizes)`. The list is in ascending rank order, must be
identical on every rank, and `sizes[my_rank]` must equal this rank's
`input.shape[0]`. Entries may be `0`: a rank with no rows contributes
nothing and the gather is still correct. This is the form attention data
parallelism needs, where per-rank token counts differ by construction.

Both forms are certified, and they are certified separately because they are
not the same call underneath — an all-gather primitive takes one element
count. Read off NCCL's own trace of this entry's test: `sizes = None`
issues **one `ncclAllGather`** whose count is this rank's element count,
and the ragged form issues **one grouped `ncclBroadcast` per rank**, each
rooted at that rank with that rank's element count (for
`sizes = [1, 5, 9, 13]` at hidden 2560: counts 2560, 12800, 23040, 33280 at
roots 0, 1, 2, 3). So one ragged call enqueues `len(group)` NCCL operations
where a uniform call enqueues one — worth knowing when reasoning about the
call-order precondition, though the two forms were never mixed across ranks
in a probe, and the `sizes` list already has to be identical everywhere.
The two cost the same (see *Notes*), so the choice between them is about
what the caller can guarantee, not about speed.

**Fusion boundary.** Inside the call: the collective and the output
allocation, nothing else. Outside: everything that produced `input`
(under attention DP, this rank's own attention output), any padding of
row counts to a uniform value, any quantization, and any reshaping —
including the reshape a caller needs to gather along an axis other than
dim 0, which this op cannot do.

**Group membership.** `group` names **ranks in trtllm's MPI session
communicator** (`MPI_COMM_WORLD` under `mpirun`), not device ordinals or
`torch.distributed` ranks. A subset is legal: with `group = [0, 1]` at
world size 4, ranks 0 and 1 gather only between themselves and ranks 2
and 3 must not call at all. The list is treated as a **set** — passing
`[3, 2, 1, 0]` produces the identical result to `[0, 1, 2, 3]`, still in
ascending rank order — so `group` cannot be used to permute the output.

## Signature

```python
def allgather(
    input: torch.Tensor,
    sizes: Optional[List[int]],
    group: List[int],
) -> torch.Tensor
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `input` | `[rows, ...]`, at least 1-D; `[T, 2560]`, `[T, 2, 1280]` and `[T]` tested | bf16; also fp16, fp32, int32, uint8, float8_e4m3fn | **contiguous** | CUDA |
| `sizes` | `None`, or a list of `len(group)` non-negative ints in ascending rank order | Python `int` | — | host |
| `group` | list of MPI session ranks | Python `int` | — | host |
| returns | `[n * len(group), *input.shape[1:]]` for `sizes = None`, `[sum(sizes), *input.shape[1:]]` otherwise | = `input.dtype` | contiguous, newly allocated | = `input.device` |

There are no other arguments — no strategy, no workspace, no output
buffer, no autotuner. Nothing selects a transport, so unlike the sibling
all-reduce there is no strategy axis to certify.

Every dtype in the table was certified in both forms (uniform and ragged)
as a bitwise move: bf16 for hidden states, uint8 for packed NVFP4 bytes
and their scale factors, float8_e4m3fn for fp8 activations, int32 for
expert ids, fp32/fp16 for routing weights and non-bf16 activations. int64
and bool were also observed to move correctly in probing and are not
certified. There is no arithmetic here for a dtype to be wrong about —
this op does not reinterpret bytes the way a summing collective does.

## Metadata consumed

No attention metadata, no KV cache, no registered layer, no workspace.
Two pieces of process state stand behind the call:

1. **trtllm's MPI session communicator.** The op resolves `group` against
   it. Under `mpirun` the session is `MPI_COMM_WORLD`; a trtllm engine's
   worker ranks are already inside one, so a target's forward needs no
   setup. World size 1 was not exercised; this entry's receipt is world
   size 4. The sibling op `torch.ops.trtllm.allgather_pg` takes a
   `c10d.ProcessGroup` instead, for the `TLLM_DISABLE_MPI=1` path;
   `torch.ops.trtllm.allgather_list` gathers a list of tensors in one
   call. Both are different ops and neither is this entry.
2. **The NCCL communicator cache**, keyed by the rank set. Built on first
   use for a given `group`, which makes the first call for a group much
   slower than the rest — and means that first call **cannot be inside a
   CUDA-graph capture** (see *Preconditions*). Inside a trtllm serving
   engine this communicator is the **caller's alone**: the runtime issues
   nothing on it, so the only calls whose order has to line up are the
   ones the caller makes. Measured — see *Notes*, "Inside a live serving
   engine under attention data parallelism".

## Preconditions

- **Every rank in `group` calls, and their arguments agree**: same dtype,
  same trailing shape, same `sizes` list, and — when `sizes is None` —
  the same row count. Disagreement is **not survivable**: it hangs rather
  than raising. Measured at world size 4, three ways, each in its own
  process: ranks passing different row counts with `sizes = None`; a rank
  holding more rows than its `sizes` entry; a rank holding fewer. In each
  case some ranks returned a plausible-looking tensor and the rest never
  came back, and the job had to be killed. This is why this entry's test
  carries an external deadline. Do not read the hang as a guarantee,
  though: every one of those three cases changes how many bytes a rank
  moves, and a disagreement that leaves the byte counts equal on every rank
  is silently wrong instead — see the call-order bullet below, where that
  is measured.
- `input` is **contiguous**. The op reads `input.numel()` elements from
  `input.data_ptr()` and ignores strides, so a strided view is gathered
  from the wrong bytes and returns a plausible-looking wrong answer with
  no error. The wrapper asserts this.
- `input.dim() >= 1`. A 0-d tensor **segfaults** inside
  `AllgatherOp::run_list` and kills every rank in the job with no
  diagnostic beyond the fault handler's stack. The wrapper asserts this.
- `len(sizes) == len(group)` when `sizes` is given. The op sizes its
  output from `sizes` alone and does not cross-check it against `group`:
  a list one entry short silently drops the highest rank's rows (measured
  — the result is a correct gather of the remaining ranks, at the shorter
  length), and a list one entry too long returns
  `sum(sizes)` rows whose tail is uninitialized memory (measured once, in
  probing; indexing past the group is undefined behaviour and not
  something to rely on). The wrapper asserts this.
- `sizes[my_rank] == input.shape[0]`, in ascending rank order. The op
  does not check it and cannot be made to — it takes no rank argument —
  and a mismatch hangs, as above. The wrapper cannot check it either, for
  the same reason: this one is the caller's.
- `input.device` is the device this rank set with
  `torch.cuda.set_device`, and one rank owns one device.
- **A `group`'s first call must not be made inside a CUDA-graph capture.**
  It is where the group's NCCL communicator gets built, and that build
  raises under capture — see the *CUDA graphs* note for the error text and
  the one-call fix.
- **Every rank issues the same sequence of calls on `group`, in the same
  order**, graph replays included. Position on the communicator is what
  pairs one rank's call with another's, so order is the caller's whole
  responsibility — and disagreeing on it is *worse* than disagreeing on the
  arguments, because it usually does not hang. Measured at world size 4 in
  the certified configuration:

  | how the ranks disagreed | what happened |
  |---|---|
  | two same-shaped gathers issued in swapped order on one rank | **silently wrong on every rank.** No error, no hang. The swapping rank had 3/4 of its elements wrong in both results, every other rank exactly 1/4 — the swapping rank's block. A swapped *pair* puts the positions back, so a plain gather straight afterwards is bitwise correct. In this entry's test. |
  | one rank issuing one gather more than the others, each iteration | **silently wrong on every rank, permanently**: the positions never realign, so every later call stays paired one off. Probing, outside the test, for that reason. |
  | a gather swapped against a `torch.ops.trtllm.reducescatter` of the **same** byte count on one rank | **silently wrong on every rank** — the gather came back holding reduce-scatter payload and vice versa. Probing, outside the test. |
  | the same swap where the two calls carry **different** byte counts | **wedged.** No rank returned; the job had to be killed. Probing, outside the test, for that reason. |

  So an ordering divergence surfaces as an accuracy loss whenever the
  mispaired calls happen to move the same number of bytes, and as a hang
  only when they do not. A caller must not wait for a hang to tell it that
  its ranks have diverged.
- **The stream is the engine's to choose and the ranks need not agree on
  it.** The call runs on `torch.cuda.current_stream()`, and a serving
  engine moves that stream under the model — the same forward runs on
  torch's graph-capture stream while a decode graph is being captured and
  on the serving stream otherwise, so a target cannot pin it. Stream
  identity plays no part in pairing the calls: certified with every rank on
  a side stream, with one rank on a side stream while the others stayed on
  the default, and with ranks alternating in opposite patterns so they
  disagreed at every call index — all bitwise correct. Certified with the
  side stream joined to the current one on both ends, which is what the
  engine and a target's forward both do; two calls on one `group` running
  *concurrently* on two streams of the same rank was not exercised.
- `group` entries must be ranks of the MPI session. With
  `group = [0, 1, 2, 3, 4]` at world size 4, **rank 0 raised**:

  ```
  RuntimeError: [TensorRT-LLM][ERROR] Assertion failed: Failed: MPI error
  ../tensorrt_llm/runtime/utils/mpiUtils.cpp:277 '6'
  (../tensorrt_llm/runtime/utils/mpiUtils.cpp:277)
  ```

  and ranks 1, 2 and 3 never returned from the call — the job had to be
  killed, twice out of twice. So this is a raise on one rank and a wedge
  everywhere else, not a survivable error: measured in probing, outside
  the test, for exactly that reason.
- `rows = 0` on **every** rank (an empty `[0, H]` input with
  `sizes = None`) is accepted and returns an empty `[0, H]` tensor
  (observed in probing, outside the test). A single rank at zero rows is
  certified through the `sizes` form.

## Notes

The certified path: `mpirun`-launched ranks whose session communicator is
`MPI_COMM_WORLD`, one rank per B200 (sm_100), world size 4, group
`[0, 1, 2, 3]` and `[0, 1]`, bf16 hidden 2560 unless the dtype table says
otherwise. The op has no strategy, workspace or autotuner state, so there
is no second execution path here for a precondition to be quietly true
of — the only axis with two paths is `sizes`, and both are certified.

**Inside a live serving engine under attention data parallelism.** The
question this answers is whether the runtime is a second party on the
communicator — whether its own cross-rank work can interleave with a
model's calls and pair against them. It is not. Measured by running a
4-rank trtllm serving engine (attention DP on, MoE expert-parallel over the
same four ranks) whose model issues 29 of these gathers and 29
reduce-scatters per forward, with NCCL's own collective trace on for the
whole process — engine warm-up, 68 decode-graph captures and the served
requests:

- **The runtime issues nothing on this communicator.** NCCL built exactly
  **one** communicator per rank in the whole process, and all 14,036
  collectives logged on it were the model's own — 7,018 `AllGather` and
  7,018 `ReduceScatter` per rank, over 242 forwards. The engine contributed
  none.
- **What the engine does instead is host-side MPI on a different
  communicator.** Under attention DP it agrees the per-rank token counts
  once per step — that is where `attn_metadata.all_rank_num_tokens` comes
  from — with `MPIDist.tp_allgather`, an `MPI_Allgather` / `MPI_Allgatherv`
  on a sub-communicator it builds with `MPI_Comm_create_group`: congruent
  to the session communicator, never identical to it, and never NCCL.
  Graph-capture consensus, batch-size consensus, request broadcast and
  response gathering all go the same way. So a model's gather cannot be
  paired against an engine collective — there is none on this communicator
  to pair it against. Certified in this entry's test by driving the engine's
  own `MPIDist` object between batches of gathers.
- **All four ranks' call sequences agreed exactly**: the 14,036 trace
  entries matched across ranks in op kind, element count and stream index,
  position by position.
- **The engine issues a model's collective on two different streams.** The
  trace splits into 137 runs: one of 580 calls on the serving stream, then
  68 pairs of [58 calls on torch's graph-capture stream][116 on the serving
  stream] — one forward captured, two run eagerly as the runner's warm-up.
  Which stream a call lands on is therefore the engine's choice, and the
  *Preconditions* entry above records that this is harmless: the ranks
  agreed on it here, and they do not have to.

What those runs do **not** establish. Ten runs of that engine were made;
eight finished and two stopped with every rank spinning in
`cuLaunchKernelEx` (`sched_yield`) — the launch queue full because the
device was not retiring work. One stopped in engine warm-up, all four ranks
in `cudaDeviceSynchronize` after the same warm-up forward; one stopped in a
served forward, all four ranks past the gather and inside the
non-collective work behind it. Nothing measured attributes either to this
call: the ranks' enqueued sequences agreed everywhere they were traced, and
the same collective pattern driven without an engine — the engine's real
host-side step sync plus 29 gathers and 29 reduce-scatters per step at
ragged attention-DP row counts up to 2048, nothing synchronising — ran
2000 steps (58,000 gathers per rank, more collectives than a whole
benchmark run of that engine) clean on those same shared GPUs. Both
stalls landed on the four GPUs that were sharing SMs roughly 50/50 with an
unrelated process; the four exclusive GPUs took four of the ten runs and
stalled on none. That is a correlation on a small sample, not a mechanism.
Note also that stock trtllm does **not** take this path at that
configuration: its MoE communication factory selected `DeepEPLowLatency`
there (measured), so a stock run is not a control for this op.

**CUDA graphs — the surface a decode step actually needs.** Certified at
world size 4, group `[0, 1, 2, 3]`, bf16, under torch's default
`capture_error_mode="global"`, with every rank capturing and replaying the
same graph:

| form | under capture |
|---|---|
| `sizes = None` (uniform) | certified at **all 35 of `1..32, 64, 128, 256` rows** |
| `sizes = [...]` (ragged) | certified — but the split is frozen at capture, see below |

A trtllm engine instantiates one decode graph per configured batch size —
with `cuda_graph_config.max_batch_size = 256` and no explicit
`batch_sizes` list, 35 of them at `1..32, 64, 128, 256` — keeps all of
them alive in one memory pool, and replays them interleaved as the served
batch size moves. Under attention DP a decode call's per-rank row count
*is* that batch size, so the whole set is certified: all 35 captured into
one shared pool, none released, then replayed in four orders — ascending,
descending, shuffled, and largest-jump-first (`256, 1, 128, 2, 64, 3, ...`)
— each order twice, once one replay at a time with a synchronize and a
full check between graphs, and once with all 35 issued back to back and
nothing synchronizing between them. Every replay's payload is one no
earlier call used, and every result is bitwise equal to the concatenation
reference. The same is certified with **29 independent gathers inside each
of the 35 graphs** — 1015 captured collectives in one pool, the shape this
checkpoint's decode graph has (30 layers, layer 0 dense, so 29
expert-parallel MoE calls each preceded by one gather).

A replay re-runs the collective over whatever the input buffer holds at
replay time and writes into the same tensor the capture returned — keep it
and read it after each `replay()`. Replays also stay correct with eager
traffic in between: an eager uniform gather at 1500, 2048 or 8192 rows and
an eager ragged gather between every pair of replays, all bitwise correct,
which is the shape a server has when a prefill runs between two decode
steps.

**`sizes` is frozen into a graph; row counts must be padded to capture
one.** `sizes` is a host-side argument, so a capture bakes in the split it
was given and a replay re-runs *that* split. Certified by capturing two
graphs with different sizes vectors into one pool and replaying them in
both orders across two payload rounds: each keeps its own row split and
its own output length. The consequence for an attention-DP target is
concrete — a graph-captured decode step must pad every rank to the same
row count and pass `sizes = None`, because the ragged split cannot vary
per replay; the ragged form belongs to the eager (non-captured) steps.

**A `group`'s first-ever call cannot be captured**, because it is where the
NCCL communicator is built. Four ranks capturing their first call each
raised, out of `tensorrt_llm::_v1::getComm`:

```
RuntimeError: Failed, NCCL error ../tensorrt_llm/common/opUtils.cpp:175
'unhandled cuda error (run with NCCL_DEBUG=INFO for details)'
```

which invalidates the capture, so the `with torch.cuda.graph(...)` block
itself then raises

```
AcceleratorError: CUDA error: operation failed due to a previous error
during capture
```

(`cudaErrorStreamCaptureInvalidated`). It is survivable, and the fix is one
line: make one eager call per `group` before any capture. After the failure
the same group gathers correctly eagerly, and a capture taken after that
replays correctly — both certified. The same failure was seen in probing
for a subgroup whose communicator did not yet exist in a process that
already had the full group's. One caveat for anyone catching it: torch's
graph context manager ends the capture before it restores the stream, so a
failed capture leaves its own stream current — put the resting one back
with `torch.cuda.set_stream`.

**Numerics.** There are none. The op copies bytes, so every assertion in
this entry's test is bitwise (`rtol=0, atol=0`) rather than
tolerance-based, in every dtype, in both forms, eager and captured. A
caller does not have to reason about accumulation order the way the
sibling all-reduce forces it to.

**Gathering along another axis is the caller's problem.** This op only
concatenates dim 0. trtllm's module-level helper reaches other axes by
reshaping around this call — `view` to 2-D before, `chunk`/`split` and
`cat` after — which is Python-level composition and would have to be
written from catalog entries, not hidden inside a wrapper. For the
attention-DP use (gather token rows before an expert-parallel MoE call)
dim 0 is already the token axis and no reshape is needed. Note also that
"dim 0" is the buffer's first dimension, not necessarily the token axis:
a scale-factor buffer in a swizzled layout has no token rows to gather.

**Cost: the two forms are indistinguishable.** Measured on the certified
path (world size 4, bf16, hidden 2560, 200 timed iterations after 20 warm-up
calls, CUDA events, eager): `sizes = None` against an explicit uniform
`sizes` vector of the same total size came out at 12.5 vs 12.4 us at 1 row,
13.0 vs 13.2 at 8, 20.2 vs 20.1 at 128 and 80.1 vs 80.2 at 2048 — ratio
1.00-1.01 throughout. So a caller that already has the per-rank counts loses
nothing by passing them, and the reason a captured decode step pads to
uniform rows is the frozen split above, not the cost of the ragged form.
Below ~128 rows the call is launch-bound (12-13 us for a message 20x apart
in size), which is the regime a decode step runs in.

**The inverse op exists.** `torch.ops.trtllm.reducescatter` has the same
`(input, sizes, group)` signature and undoes this one — it is a different
op and is not this entry.
