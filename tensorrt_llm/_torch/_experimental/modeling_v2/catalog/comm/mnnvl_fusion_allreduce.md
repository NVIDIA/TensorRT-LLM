---
receipts:
  sm_100: {status: pending, world_size: 4}
---

# mnnvl_fusion_allreduce

**Wraps** `torch.ops.trtllm.mnnvl_fusion_allreduce` (one call).

## Semantics

The MNNVL all-reduce of a TP group over the group's caller-owned `MnnvlWorkspace`, optionally followed by a residual
add and an RMSNorm. Every rank of the workspace's group calls with its own rows `input` `[T, H]`; every rank gets back
the same result. With `W` ranks, per token:

```
s       = sum over the ranks r = 0 .. W-1 of input_r       # fp32, a fixed rank order, then one rounding to bf16
plain:    returns s                                         # residual, norm_weight, eps all None
fused:    updated = bf16(s + residual)                      # AllReduceFusionOp.RESIDUAL_RMS_NORM
          rcp     = rsqrt(sum_h bf16(updated_h * updated_h) / H + eps)        # fp32; each square rounded to bf16
          normed  = bf16(updated * rcp * norm_weight)                         # fp32 products
          returns (normed, updated)
```

An input `-0.0` travels as `+0.0` (the Lamport buffers' empty word is `-0.0`), so a sum is never `-0.0`: certified on
the plain sum (every rank's input holds `-0.0` in the same columns of every call of the test, and the plain result
there is `+0.0`, bit for bit). The squares inside the RMSNorm are rounded to bf16 before they are summed (both
kernels' code; the one-shot kernel marks it `FIXME: Use float square if accuracy issue`).

**Two paths, chosen per call.** With `one_shot = T x H x W x 2` bytes, the call is sent **one-shot** when
`one_shot <= one_shot_max_bytes` and **two-shot** otherwise. Certified at the exact boundary for every certified
shape: `one_shot_max_bytes` equal to the footprint goes one-shot, one byte less goes two-shot, as the stage count the
call leaves in the workspace's flags shows (*State*).

- One-shot: one kernel. Each rank writes its rows into every rank's buffer through the multicast mapping, waits for
  all `W` rows of every token in its own copy and sums them; the residual add and the RMSNorm run in the same kernel.
- Two-shot: each rank writes token `t`'s row into rank `t mod W`'s buffer, which sums the `W` rows of its tokens and
  writes the bf16 sums into every rank's buffer through the multicast mapping. Plain, the same kernel then waits for
  every token's sum and copies it out; fused, a second kernel does that and adds the residual and normalizes.

Both paths return the same plain sum and `updated` on the test's inputs (certified: every certified shape is sent
one-shot and then two-shot with the same inputs, each bit for bit against the exact reference). Beyond exact inputs
(the kernels' code): up to 8 ranks both paths add the ranks in ascending order with the same fp32 operations, so the
sum and `updated` agree bit for bit; at 16 ranks the one-shot adds two partial sums of 8 ranks while the two-shot adds
all 16 in sequence, so an inexact sum can differ in the last bit between the paths. `normed` can differ in the last
bit at any `W`: the two RMSNorm kernels form `updated * rcp * norm_weight` in different orders (`x * rcp * g`
one-shot, `g * x * rcp` two-shot); the test bounds both against fp32. Whatever the path, the result is bitwise the
same on every rank (certified, every call of the test).

Fusion boundary. Inside: the exchange, the sum, and with `residual` the add and the RMSNorm. Outside: whatever
produced `input` (a row-parallel projection's partial output), the choice of `one_shot_max_bytes` (the caller's, per
call), the workspace (the target's).

Inert (not exposed by the wrapper): the op's quantizing epilogues (`fusion_op` `RESIDUAL_RMS_NORM_QUANT_FP8`,
`_OUT_QUANT_FP8`, `_QUANT_NVFP4`, `_OUT_QUANT_NVFP4`, with `scale`). The wrapper passes `scale=None` and `fusion_op`
`NONE` or `RESIDUAL_RMS_NORM` only.

## Signature

```python
def mnnvl_fusion_allreduce(
    input: torch.Tensor,
    workspace: MnnvlWorkspace,
    one_shot_max_bytes: int,
    residual: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

def required_buffer_bytes(
    num_tokens: int, hidden: int, world_size: int, dtype: torch.dtype, one_shot_max_bytes: int
) -> int
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `input` | `[T, H]`; `H` 3584 and 7168; `T` 1-8, 16, 32, 64 on both paths (and 65 two-shot, above 2 ranks) | bf16 | contiguous | CUDA, this rank's device |
| `workspace` | an `MnnvlWorkspace` of this rank's TP group whose buffers hold the call (see *State*) | — | — | — |
| `one_shot_max_bytes` | int >= 0; for every certified `(T, H)` its one-shot footprint `T x H x W x 2` and one less; Kimi K3's 4 MiB and 1 MiB (*Notes*) | Python int | — | — |
| `residual` | `None` (plain), or `[T, H]` | bf16 | contiguous | CUDA |
| `norm_weight` | `None`, or `[H]`; with `residual` | bf16 | contiguous | CUDA |
| `eps` | `None`, or a float with `residual`; 1e-5 certified | Python float | — | — |
| returns | plain: the sum `[T, H]`; fused: `(normed, updated)`, each `[T, H]` | bf16 | contiguous, newly allocated | = `input.device` |

`input`, `residual` and `norm_weight` are read only. `required_buffer_bytes` is the space one Lamport buffer must
have for the call: `T x H x W x 2` one-shot, `2 x ceil(T / W) x W x H x 2` two-shot (two stages); certified equal to
the space the call's stages take, every call of the shape grid.

## State

**Object.** `MnnvlWorkspace` (`catalog/comm/mnnvl_workspace.py`), one per TP group, owned by the caller; the object's
own contract is the *State* section of `mnnvl_allreduce_attn_res.md`. This section states what this op does with it.

**Contents and size.** Three Lamport buffers of `buffer_bytes` each behind one multicast mapping (every word `-0.0`
when armed) and the flag words `buffer_flags` (uint32 `[9]`: current buffer, dirty buffer, bytes per buffer, dirty
stage count, bytes to clear x 4, arrival count). A one-shot call writes `T x H x W x 2` bytes from the start of one
buffer. A two-shot call splits the buffer into two stages of `buffer_bytes / 2`: the scatter stage (`ceil(T / W) x W
x H x 2` bytes from the start) and the broadcast stage (`T x H x 2` bytes from the middle). A call that needs more
than `buffer_bytes` (`required_buffer_bytes`) is refused by the wrapper with `ValueError` on every rank before it
touches the workspace (certified: `T` = 65 at `H` = 7168 one-shot; the flags do not move and the next call is
correct; the op itself does not check, see *Notes*). Two-shot needs about `2 / W` of the one-shot space, so a call
too large for one buffer one-shot can fit two-shot (certified: that `T` = 65 call sent two-shot is correct, above 2
ranks). The workspace is never grown. With `one_shot_max_bytes <= buffer_bytes` every one-shot call fits; the
two-shot calls Kimi K3 makes (at most 64 tokens of 7168) take at most 1.75 MiB, so a 4 MiB buffer holds all its
calls of this op (arithmetic, not a test). The test's buffer is the one-shot footprint of `[64, 7168]`, 3.5 MiB at
`W` = 4.

**Who creates it, and when.** The target, in `post_load_weights`, with
`MnnvlWorkspace.create(mapping, buffer_bytes, fabric_handle=None)`: collective over the TP group, eager, every word
and flag armed before any rank returns (see `mnnvl_allreduce_attn_res.md`). It refuses CUDA-graph capture: certified
with every rank capturing, each raising `RuntimeError`. The check runs before any communication, so a rank that is
not capturing while its peers are would go on into the communicator split and wait for them (code).

**Which ops may share one object.** Every MNNVL op of the group takes the same `comm_buffer` / `buffer_flags`:
`comm/mnnvl_allreduce_attn_res`, this entry on either path, and `comm/mnnvl_allgather_split`. Their calls form one
sequence and each takes exactly one turn of the one rotation, whatever the op and path (certified: after every eager
call of the test the flags equal this test's model of one turn per call). Certified on one workspace in Kimi K3's
order, a random rank late at every call: decode steps of [attention-residual all-reduce, head all-gather, latent
all-reduce `[T, 3584]`, fused all-reduce `[T, 7168]`] and wide steps of [all-reduce `[T, 7168]`, head all-gather,
latent all-reduce] at Kimi K3's ceilings, `T` = 8, 2, 16, 64, 1, 7, 32, 8, 3, 16, two layers each. Two objects are
two independent rotations: 20 calls of mixed shapes and paths alternating irregularly between two workspaces are all
correct, and each workspace's flags move with its own calls only (certified). They are not independent orders: on one
stream every rank must issue its collectives in the same order, whatever object each belongs to (measured for
`comm/mnnvl_allreduce_attn_res`: ranks issuing B-then-A against A-then-B deadlocked; this op waits for its peers
the same way).

**Call-order invariant.** Every rank of the group makes the same sequence of calls on one workspace — the same
number, the `k`-th with the same op, `T`, `H`, fusion and path — across layers and decode steps, eager calls and
graph replays alike; and on one stream the same order of calls across workspaces. The path is part of the sequence:
the two paths write and wait for different words, so the ranks' `one_shot_max_bytes` must pick the same path (they do
when they pass the same value).

**What a later launch reads.** `buffer_flags`, which every call leaves as: current = its own buffer plus one, mod 3;
dirty = its own buffer; bytes per buffer unchanged; dirty stage count 1 (one-shot) or 2 (two-shot); bytes to clear
`(T x H x W x 2, 0, 0, 0)` one-shot or `(ceil(T / W) x W x H x 2, T x H x 2, 0, 0)` two-shot; arrival count 0
(certified after every eager call of the test). The next call, of any MNNVL op, takes the current buffer, whose words
must all be `-0.0` but for its own pushes, and clears the dirty one by those sizes. Each call's first kernel waits
for the previous kernel on the stream before it reads the flags (the kernels' code).

**How it is re-armed.** Each call clears the previous call's buffer stage by stage, by the bytes the previous call
recorded and in the previous call's stage layout (`cpp/tensorrt_llm/common/lamportUtils.cuh`,
`LamportFlags::clearDirtyLamportBuf`): after a two-shot call both stages, after a one-shot call, an all-gather or an
attention-residual all-reduce the first stage, whichever path the clearing call itself takes. Certified by the shape
grid (every shape one-shot and then two-shot back to back on one workspace, then the next shape) and by the
sequences below.

**Why the test drives call sequences.** See `mnnvl_allreduce_attn_res.md` (*State*): Kimi K3's `k3_spec_accept` once
re-armed its Lamport buffer for the current call's rows only; every single-call test passed, and a sequence whose row
count dipped and grew back caught it; in serving it made the ranks disagree and hang. This entry's test runs 16 decode
steps of 8 layers, each layer the latent all-reduce `[T, 3584]` and the fused all-reduce `[T, 7168]` chained through
`updated`, at `T` = 8, 8, 8, 2, 7, 8, 1, 1, 64, 3, 32, 8, 16, 1, 64, 8 with Kimi K3's ceilings, so the path changes
inside the sequence (at `W` = 4 `[64, 3584]` and `[32 or 64, 7168]` go two-shot, at `W` = 16 every step above 8
tokens), a random rank 5 ms late at every call, each call against the reference and its flags against the model.

**What a wrong order does.** Certified (the test's negative control, one-shot and two-shot): rank 0 issues two
same-shaped calls on one workspace in swapped order. Nothing raises and nothing hangs — the two calls write and wait
for the same words of the same buffers — but every rank's two results are wrong, and are exactly the
position-paired sums: the `k`-th call on every rank adds what every rank sent at position `k` (rank 0's second call
with the others' first; more than half the elements differ from the intended sums). A plain call right after is
correct again: a swapped pair realigns the positions. Two calls that write different words (another `T` or `H`, or
the other path) cannot pair like that: a rank would wait for words its peers do not write at that position, and hang
(the protocol, not exercised). A rank making one call more or fewer than its peers was not exercised; its positions
never realign.

## Metadata consumed

None besides `workspace` and `one_shot_max_bytes`, both explicit. The op keeps no cache and compiles nothing
(precompiled kernels). It finds the multicast mapping by looking `comm_buffer`'s address up in a process registry of
multicast buffers, which the workspace's handle keeps registered. `TRTLLM_ENABLE_PDL` (read once per process,
default on at SM 90 and newer) launches the kernels as programmatic dependents: the one-shot kernel releases its own
dependents as it starts, the two-shot one after its scatter; consumers of the outputs wait for the grid (the kernels'
statement); results do not depend on it. The launch shape (CTAs per token, cluster size) follows `T`, `H` and the
device's SM count, and in the fused form it sets the order in which the squares are summed.

## Preconditions

- bf16, contiguous, `input` 2-D (the op flattens all but the last dimension of an N-D input; not certified; it also
  takes fp16 and fp32, not certified). `H` a multiple of 8; otherwise the op raises `RuntimeError` before it touches
  the workspace (certified at `H` = 3580, on every rank, the flags unmoved, the next call correct). The one-shot
  kernel's launch check names 65536 as its largest `H` (bf16).
- `W` in {2, 4, 8, 16, 32, 64} (the kernels' dispatch).
- `required_buffer_bytes(T, H, W, bf16, one_shot_max_bytes) <= workspace.buffer_bytes` (*State*).
- `residual`, `norm_weight` and `eps` together or not at all; otherwise the wrapper raises `ValueError` on every rank
  before it touches the workspace (certified).
- Every rank calls with the same `T`, `H`, fusion and `one_shot_max_bytes` decision; the call order is the *State*
  invariant.
- `workspace` was created before any capture. Calls may be captured: certified with a captured step of six calls on
  one workspace — the attention-residual all-reduce, this op plain and fused one-shot at `T` = 8, the head all-gather,
  a plain `[32, 7168]` and a fused `[16, 7168]` sent two-shot (the fused two-shot call is two kernels, both in the
  graph) — replayed 8 times with rewritten inputs and an eager call of another shape or op on the same workspace
  between replays, every replayed and eager result against the reference and the flags after each.
- SM 90 or newer (the kernels' check).

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, POSIX-fd handles, PDL on (the default).
  Test: `tests/unittest/_torch/modeling_v2/comm/_mnnvl_fusion_allreduce_op_matrix.py`. The reference is native
  torch: the inputs are multiples of 1/16 with at most 4/16 per rank, so every sum over the ranks is exact in fp32 and
  bf16 and the residual add is one bf16 rounding of an exact fp32 value in the op and in the reference; the sum and
  `updated` are compared bit for bit, `normed` against the fp32 RMSNorm of `updated` within 1e-2 of its largest
  magnitude (the bf16 squares cost at most 2^-9 on `rcp`, the bf16 output 2^-8).
- State and test design: a typed state object built by an explicit, collective, eager `create()`; a test that drives
  call sequences on real state (layers x steps, capture + replay, two objects interleaved) plus a negative control;
  every written buffer named in the schema (the op falls short there, see the gaps below); the matrix takes
  `--world-size` and `--launcher` (`mpirun` on one node, `srun` across trays) and CI runs it at 4 ranks on one GB200
  tray; one caller-owned `MnnvlWorkspace` shared by every MNNVL entry of the TP group, and
  `one_shot_max_bytes` per call.
- At `W` = 16 the one-shot kernel adds the ranks in two chunks of 8, a branch a 4-rank run never reaches. The 16-rank
  receipt is pending.
- Kimi K3's calls (its decode path, not this test): the model sets every `MNNVLAllReduce` of the target, its
  LM head and a drafter to `one_shot_max_bytes` = 4 MiB (`DECODE_AR_ONE_SHOT_MAX_BYTES`, against main's 1 MiB), and
  a wide decode step (9 to 64 tokens) passes 1 MiB per call (`WIDE_AR_ONE_SHOT_MAX_BYTES`). Plain: the routed-latent
  all-reduce `[T, 3584]` (decode steps where the latent exchange push is not used; wide steps) and a wide step's
  attention all-reduces `[T, 7168]`. Fused: the DSpark drafter's residual + RMSNorm all-reduces `[T, 7168]` where
  `k3_sandwich_plain` does not take the call. At 4 MiB a `[T, 7168]` call goes one-shot up to `T` = 18 at `W` = 16
  (73 at `W` = 4) and a `[T, 3584]` one up to 36 (146); at 1 MiB `[T, 7168]` up to 4 (18) and `[T, 3584]` up to 9
  (36).
- Gaps (the op is unchanged by this entry): the schema marks `comm_buffer` mutable `(a!)` but not `buffer_flags`,
  which every call advances; the op does not check that the call fits `comm_buffer` (the
  attention-residual and all-gather ops do), so a direct op call over one buffer writes past it — the wrapper's
  `required_buffer_bytes` check is the guard; `MnnvlWorkspace.create` accepts any multiple of 16 bytes, but the
  two-shot broadcast stage starts at `buffer_bytes / 2` and is accessed in 16-byte vectors, so a two-shot call needs
  `buffer_bytes` to be a multiple of 32 (code; every buffer in the test is); the schema's default
  `one_shot_max_bytes=1048576` applies to a direct op call (the wrapper always passes one).
- In the model today the workspace is `MNNVLAllReduce`'s (a dict keyed by `Mapping`, grown on demand by the first
  eager call that needs more, in 8 MiB steps) and the one-shot ceiling is a module attribute with a per-call override.
  This entry takes both explicitly: the workspace sized at construction, the ceiling per call.
