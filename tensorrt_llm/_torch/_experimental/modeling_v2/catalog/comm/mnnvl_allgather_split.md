---
receipts:
  sm_100: {status: pending, world_size: 4}
---

# mnnvl_allgather_split

**Wraps** `torch.ops.trtllm.mnnvl_allgather_split` (one call).

## Semantics

A one-shot all-gather over a TP group's caller-owned `MnnvlWorkspace` of fp32 rows whose leading columns travel, and
are gathered, as bf16. Every rank of the workspace's group calls with its own rows `input` `[T, B + F]` (fp32) and the
same `bf16_columns` = `B`; every rank gets back the same two tensors, the ranks' slices in rank order. With `W` ranks:

```
bf16_out[t, r * B + j] = bf16(input_r[t, j])          j < B     # round to nearest, ties to even
fp32_out[t, r * F + j] = input_r[t, B + j]            j < F     # the fp32 word, unchanged
```

and `-0.0` arrives as `+0.0` in both outputs: the Lamport buffers' empty word is the fp32 `-0.0`, so the kernel
replaces a bf16 `-0.0` half and an fp32 `-0.0` word by `+0.0` before sending. Nothing else is computed. Certified bit
for bit against torch (`input[:, :B].bfloat16()` and the fp32 columns copied, in rank order, `-0.0` made `+0.0`) on
rows that exercise each rule: normal values over exponents `2^-16` to `2^16` in the bf16 columns (inexact in bf16, so
rounded), exact midpoints between two bf16 values in every fourth of them (ties to even decide), `-0.0` in every
fourth column of each part, and in the fp32 part infinities, a NaN, signed denormals and the largest float, which
arrive unchanged. The result is bitwise the same on every rank (certified, every call of the test).

Kimi K3's use: the row-sharded MoE head of a wide decode step (9 to 64 tokens), and of a
decode step of at most 8 tokens where the fused MoE front kernel does not run. Rank `r`'s GEMV gives fp32
`[T, 3584/W + 896/W]`: the latent down projection's columns `[r * 3584/W, (r+1) * 3584/W)`, then the router logits of
experts `[r * 896/W, (r+1) * 896/W)`. This call assembles the bf16 latent `[T, 3584]` (`bf16_out`) and the fp32
logits `[T, 896]` (`fp32_out`) on every rank in one exchange, on the workspace of the routed experts' all-reduce.

Fusion boundary. Inside: the bf16 rounding of the leading columns and the exchange. Outside: the GEMV that produced
`input`, and the routing and quantization that consume the outputs (`trtllm::k3_route_quant`;
`k3_fused_moe/k3_route_quant_ag.py` fuses this exchange with them and states it is bit for bit this op followed by
`k3_route_quant`).

The kernel releases its programmatic dependents as soon as it starts; its outputs are complete only when its grid is,
so a kernel launched as its programmatic dependent must wait for the grid before reading them (the kernel's
statement).

## Signature

```python
def mnnvl_allgather_split(
    input: torch.Tensor,
    bf16_columns: int,
    workspace: MnnvlWorkspace,
) -> Tuple[torch.Tensor, torch.Tensor]

def required_buffer_bytes(num_tokens: int, bf16_columns: int, fp32_columns: int, world_size: int) -> int
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `input` | `[T, B + F]`; `(B, F)` = (1792, 448), (896, 224), (448, 112), (224, 56) — Kimi K3's per-rank head at `W` = 2, 4, 8, 16, each certified at the world size of the run — and (8, 4); `T` 1-8, 16, 32, 64 | fp32 | contiguous, 16-byte aligned | CUDA, this rank's device |
| `bf16_columns` | `B`, a multiple of 8, with `F` = `input.shape[1] - B` a multiple of 4 | Python int | — | — |
| `workspace` | an `MnnvlWorkspace` of this rank's TP group whose buffers hold the call (see *State*) | — | — | — |
| returns | `(bf16_out, fp32_out)`: `[T, W x B]` and `[T, W x F]` | bf16, fp32 | contiguous, newly allocated | = `input.device` |

`input` is read only. `required_buffer_bytes` = `T x W x (2B + 4F)`, the bytes the call writes into one Lamport
buffer (certified equal to what the call records, every call of the split grid).

## State

**Object.** `MnnvlWorkspace` (`catalog/comm/mnnvl_workspace.py`), one per TP group, owned by the caller; the object's
own contract is the *State* section of `mnnvl_allreduce_attn_res.md`. This section states what this op does with it.

**Contents and size.** Three Lamport buffers of `buffer_bytes` each behind one multicast mapping (every word `-0.0`
when armed) and the flag words `buffer_flags` (uint32 `[9]`: current buffer, dirty buffer, bytes per buffer, dirty
stage count, bytes to clear x 4, arrival count). A call writes, from the start of one buffer, one slot per token and
rank: `B / 8` vectors of 8 bf16 values, then `F / 4` vectors of 4 fp32 words, `T x W x (2B + 4F)` bytes in all —
688128 bytes at Kimi K3's split for 64 tokens, at any `W`. A call that needs more than `buffer_bytes` is refused by
the wrapper with `ValueError` on every rank before it touches the workspace (certified at the first `T` over one
buffer; the flags do not move and the next call is correct; the op has the same check of its own, raising
`RuntimeError`). The workspace is never grown. The test's buffer is the largest of its calls, 1.75 MiB at `W` = 4 (a
two-shot `[64, 7168]` all-reduce of its shared-workspace sequence).

**Who creates it, and when.** The target, in `post_load_weights`, with
`MnnvlWorkspace.create(mapping, buffer_bytes, fabric_handle=None)`: collective over the TP group, eager, every word
and flag armed before any rank returns (see `mnnvl_allreduce_attn_res.md`). It refuses CUDA-graph capture: certified
with every rank capturing, each raising `RuntimeError`. The check runs before any communication, so a rank that is
not capturing while its peers are would go on into the communicator split and wait for them (code).

**Which ops may share one object.** Every MNNVL op of the group takes the same `comm_buffer` / `buffer_flags`:
`comm/mnnvl_allreduce_attn_res`, `comm/mnnvl_fusion_allreduce` on either path, and this entry. Their calls form one
sequence and each takes exactly one turn of the one rotation, whatever the op (certified: after every eager call of
the test the flags equal this test's model of one turn per call). Certified on one workspace in Kimi K3's order, a
random rank late at every call: decode steps of [attention-residual all-reduce, head all-gather, latent all-reduce
`[T, 3584]`, fused all-reduce `[T, 7168]`] and wide steps of [all-reduce `[T, 7168]`, head all-gather, latent
all-reduce] at Kimi K3's one-shot ceilings, `T` = 8, 2, 16, 64, 1, 7, 32, 8, 3, 16, two layers each, so all-gathers
follow one-shot and two-shot all-reduces. Two objects are two independent rotations: 20 all-gathers of mixed token
counts and splits alternating irregularly between two workspaces are all correct, and each workspace's flags move
with its own calls only (certified). They are not independent orders: on one stream every rank must issue its
collectives in the same order, whatever object each belongs to (measured for `comm/mnnvl_allreduce_attn_res`:
ranks issuing B-then-A against A-then-B deadlocked; this op waits for its peers the same way).

**Call-order invariant.** Every rank of the group makes the same sequence of calls on one workspace — the same
number, the `k`-th with the same op, `T`, `B` and `F` — across layers and decode steps, eager calls and graph replays
alike; and on one stream the same order of calls across workspaces.

**What a later launch reads.** `buffer_flags`, which every call leaves as: current = its own buffer plus one, mod 3;
dirty = its own buffer; bytes per buffer unchanged; dirty stage count 1; bytes to clear `(T x W x (2B + 4F), 0, 0,
0)`; arrival count 0 (certified after every eager call of the test). The next call, of any MNNVL op, takes the current
buffer, whose words must all be `-0.0` but for its own pushes, and clears the dirty one by that size. The kernel waits
for the previous kernel on the stream before it reads the flags (its code).

**How it is re-armed.** Each call clears the previous call's buffer stage by stage, by the bytes the previous call
recorded and in the previous call's stage layout (`cpp/tensorrt_llm/common/lamportUtils.cuh`,
`LamportFlags::clearDirtyLamportBuf`): after a two-shot all-reduce both of its stages, after any other call the first.
Certified by the split grid (55 calls of different sizes back to back on one workspace) and by the sequences below.

**Why the test drives call sequences.** See `mnnvl_allreduce_attn_res.md` (*State*): Kimi K3's `k3_spec_accept` once
re-armed its Lamport buffer for the current call's rows only; every single-call test passed, and a sequence whose row
count dipped and grew back caught it. This entry's test runs 16 decode steps of 6 layers, one head all-gather per
layer at Kimi K3's split, at `T` = 8, 8, 8, 2, 7, 8, 1, 1, 64, 3, 32, 8, 16, 1, 64, 8, a random rank 5 ms late at
every call, each call against the reference and its flags against the model.

**What a wrong order does.** Certified (the test's negative control): rank 0 issues two same-shaped all-gathers on
one workspace in swapped order. Nothing raises and nothing hangs — the two calls write and wait for the same words of
the same buffers — but every rank's two results are wrong in rank 0's columns, which hold rank 0's rows of the other
call, while every other rank's columns are right: the `k`-th call on every rank gathers what every rank sent at
position `k` (bit for bit). A plain call right after is correct again: a swapped pair realigns the positions. Two
calls that write different words (another `T`, `B` or `F`) cannot pair like that: a rank would wait for words its
peers do not write at that position, and hang (the protocol, not exercised). A rank making one call more or fewer
than its peers was not exercised; its positions never realign.

## Metadata consumed

None besides `workspace`, an explicit argument. The op keeps no cache and compiles nothing (a precompiled kernel). It
finds the multicast mapping by looking `comm_buffer`'s address up in a process registry of multicast buffers, which
the workspace's handle keeps registered. `TRTLLM_ENABLE_PDL` (read once per process, default on at SM 90 and newer)
launches the kernel as a programmatic dependent (see *Semantics* for what that asks of its consumers); results do not
depend on it.

## Preconditions

- `input` fp32, contiguous, 2-D, 16-byte aligned; `B` a multiple of 8 and `F` a multiple of 4; `T` at least 1.
  Otherwise the op raises `RuntimeError` on every rank before it touches the workspace (certified: `B` = 12, `F` = 2,
  a bf16 input, `T` = 0; the flags do not move and the next call is correct).
- `required_buffer_bytes(T, B, F, W) <= workspace.buffer_bytes` (*State*).
- Every rank calls with the same `T`, `B` and `F`; the call order is the *State* invariant.
- `workspace` was created before any capture. Calls may be captured: certified with a captured step of five calls on
  one workspace — the head all-gather at `T` = 8, the latent all-reduce one-shot, the all-gather again, a `[32, 3584]`
  all-reduce sent two-shot and the all-gather at `T` = 32 — replayed 8 times with rewritten rows and an eager
  all-gather of another `T` or split, or an all-reduce, on the same workspace between replays, every replayed and
  eager result against the reference and the flags after each.
- The kernel takes any `W`; the all-reduces sharing the workspace take `W` in {2, 4, 8, 16, 32, 64}.

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, POSIX-fd handles, PDL on (the default).
  Test: `tests/unittest/_torch/modeling_v2/comm/_mnnvl_allgather_split_op_matrix.py`. The reference is native torch
  and exact; this op's outputs are compared bit for bit. The all-reduce and attention-residual calls of its sequences
  are checked as in their own matrices (sums bit for bit, normed outputs within a tolerance).
- State and test design: a typed state object built by an explicit, collective, eager `create()`; a test that drives
  call sequences on real state (layers x steps, capture + replay, two objects interleaved) plus a negative control;
  every written buffer named in the schema (the op falls short there, see the gaps below); the matrix takes
  `--world-size` and `--launcher` (`mpirun` on one node, `srun` across trays) and CI runs it at 4 ranks on one GB200
  tray; one `MnnvlWorkspace` shared by every MNNVL entry of the TP group. The 16-rank receipt is pending.
- Not exercised: `B` = 0 or `F` = 0 (the op accepts both), denormal and non-finite values in the bf16 columns, an
  accepted call of more than 64 tokens.
- Gaps (the op is unchanged by this entry): the schema marks `comm_buffer` mutable `(a!)` but not `buffer_flags`,
  which every call advances; the op has no `register_fake`
  (`tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py` registers one for the other two MNNVL ops), so fake-tensor
  tracing, e.g. `torch.compile`, cannot run it.
- In the model today the call is `MNNVLAllReduce.allgather_split(input, bf16_columns)` on `MNNVLAllReduce`'s workspace
  (a dict keyed by `Mapping`, grown to the call's footprint by the first eager call that needs more). This entry takes
  the explicit object instead, sized at construction.
