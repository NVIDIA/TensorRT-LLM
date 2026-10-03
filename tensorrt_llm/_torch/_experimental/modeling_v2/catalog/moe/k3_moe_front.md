---
receipts:
  sm_100: {status: pending, world_size: 4}
---

# k3_moe_front

**Wraps** `torch.ops.trtllm.k3_moe_front` (one call), over a caller-owned `K3MoeHeadWorkspace`.

## Semantics

Kimi K3's MoE front for a decode batch of at most 8 tokens, in one kernel: the row-sharded MoE head GEMV (this rank's
slice of the latent-down projection and of the router), the all-gather of every rank's head slice over the TP group's
head workspace, the top-16 routing and the MXFP8 quantization of the gathered latent, and this rank's slice of the
shared experts' gate_up GEMV with its SiTU-and-mul. Every rank of the workspace's TP group (`W` ranks) calls with the
same MoE input `x` `[M, K]` (`K` = 7168) and its own front weight. With `WL = 3584 / W` and `WE = 896 / W`:

```
head_r  = x @ head_weight_r^T              # fp32 [M, WL + WE] on rank r: latent columns [r WL, (r + 1) WL),
                                           # then the logits of experts [r WE, (r + 1) WE)
latent  = bf16(head_0[:, :WL] | head_1[:, :WL] | ... | head_{W-1}[:, :WL])     # [M, 3584]; -0.0 becomes +0.0
logits  = head_0[:, WL:] | head_1[:, WL:] | ... | head_{W-1}[:, WL:]           # fp32 [M, 896]
(topk_ids, topk_weights, quantized, scales) = k3_route_quant(logits, e_score_correction_bias, latent, rsf)
g, u    = bf16(x @ gate^T), bf16(x @ up^T)                                     # this rank's shared_cols columns
shared  = bf16(gate_cap * tanh(g / gate_cap) * sigmoid(g) * linear_cap * tanh(u / linear_cap))
```

`head_weight_r` is rank `r`'s `[WL + WE, K]` head slice and `gate`, `up` this rank's `[shared_cols, K]` shared rows,
all inside the ranks' `w_front`. `k3_route_quant` is the routing and quantization of `moe/k3_moe` (top-16 of the
sigmoid plus the bias, ties to the lower id, weights renormalized times `routed_scaling_factor`; MXFP8 with one UE8M0
scale per 32 columns): the front selects with all warps of a CTA but returns `top16_warp`'s experts, order and weight
bits, and quantizes with the same device code (the kernel's statement). Every head and shared value is an fp32 sum
over `K` split across the 8 CTAs of a cluster, the 8 partials added in cluster-rank order from +0.0 (the kernel's
statement): deterministic, but not the summation order of another GEMM.

Certified at every `M` 1-8 and for every front call the matrix makes alone, with payloads whose head and shared sums
are exact in fp32 in any order (`x` a multiple of 1/8 in [-1/4, 1/4], the latent-down and shared rows multiples of
1/16 in [-1/8, 1/8], the router rows multiples of 1/8 in [-1/4, 1/4]):

- `topk_ids`, `topk_weights`, `quantized` and `scales` bit for bit those of `trtllm::kimi_k3_noaux_tc_mxfp8_quant`
  on the exactly gathered head (logits in fp32, latent rounded to bf16), and bit for bit the same on every rank;
- `shared` within 2e-2 (of its largest magnitude) of the fp32 SiTU of the bf16-rounded exact sums: the kernel's tanh
  and sigmoid are fast approximations;
- every output run-to-run bit identical, and each `M`'s outputs bit for bit the same rows of the 8-token call.

With general inputs the head sums round in the kernel's own order, so the logits and the latent can differ from
another GEMM's in the last bit; the kernel test (`test_k3_moe_front.py`) bounds the effect: other experts only where
the 16th and 17th selection keys are within 1e-4, more than 99.9 % of the MXFP8 codes and scales equal.

Fusion boundary. Inside: the head GEMV, the head all-gather, the routing, the MXFP8 latent, the shared gate_up and
its SiTU. Outside: the producer of the MoE input `x`; the routed experts (`moe/k3_moe`'s `k3_moe_fused_front` runs
`k3_moe` on this op's outputs within the same call); the shared experts' down projection and its all-reduce; the
routed partials' all-reduce and the latent-up projection.

## Signature

```python
def k3_moe_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    workspace: K3MoeHeadWorkspace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

The wrapper module also exports the load-time helpers `front_weight(head_weight, gate_up_weight)`, which packs
`w_front`, and `weight_supported(world, shared_cols, k_in, device)`.

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, 7168]`, `M` 1-8; the same values on every rank | bf16 | contiguous | CUDA, this rank's device |
| `w_front` | `front_weight(head_weight, gate_up_weight)`: `[head_tiles(W) x 128 + 2 shared_cols, 7168]`, i.e. `[1920, 7168]` at `W` 4 | bf16 | contiguous, as `front_weight` returns it | CUDA |
| `head_weight` (to `front_weight`) | this rank's `[WL + WE, 7168]`: its latent-down rows, then its router rows (1120 rows at `W` 4; 280 at `W` 16) | bf16 | contiguous | CUDA |
| `gate_up_weight` (to `front_weight`) | `[2 shared_cols, 7168]`: gate rows, then up rows | bf16 | contiguous | CUDA |
| `e_score_correction_bias` | `[896]` | fp32 | contiguous | CUDA |
| `routed_scaling_factor` | scalar (2.827 certified) | Python float | — | — |
| `shared_cols` | 384 (Kimi K3 TP16's per-rank width: two shared experts of 3072 over 16 ranks) | Python int | — | — |
| `gate_cap`, `linear_cap` | 4.0, 25.0 (Kimi K3's SiTU caps) | Python float | — | — |
| `workspace` | the `K3MoeHeadWorkspace` of this rank's TP group, `W` = 4 certified (see *State*) | — | — | — |
| returns | `topk_ids [M, 16]` int32, `topk_weights [M, 16]` bf16, `quantized [M, 3584]` float8_e4m3fn, `scales [M, 112]` uint8 (linear: one byte per 32 columns), `shared [M, shared_cols]` bf16 | — | contiguous, newly allocated | `x.device` |

Inert (not exposed by the wrapper): `ring` (4, the weight ring's stages) and `ag_ready` (`None`: the standalone front
publishes no ready words; `K3MoeLayer.front` on a head_flags state passes the workspace's `ready`, see *State*).
`gate_cap` and `linear_cap` are compile-time constants of the kernel: each pair compiles once (*Metadata consumed*).

## State

**Object.** `K3MoeHeadWorkspace` (`cute_dsl_kernels/k3_fused_moe/op.py`, re-exported by `catalog/moe/k3_moe_front.py`
and `catalog/moe/k3_moe.py`), one per TP group, owned by the caller.

**Contents and size.** One multicast allocation of `workspace_words(W)` int32 words per rank, 157,696 words (616 KiB)
at `W` 4, 8 and 16 (certified at 4):

- `uc`: this rank's words, every one 0x80000000 (empty) when armed. First the two alternating Lamport buffers
  `[buffer 2][token 8][rank W][slot]` (43,008 words), a slot holding `3584 / W / 8` latent vectors of 8 bf16 (4 words
  each), then `896 / W / 4` logit vectors; the front pushes each rank's latent rows into the latent vectors, and the
  logit vectors (the layout of `k3_route_quant_ag.py`) stay empty. Then the router partials,
  `[buffer 2][token 8][rank W][cluster rank 8][896 / W]` fp32 words (114,688 words): each CTA of a GEMV cluster
  pushes its split-K partial of its rank's router rows there.
- `mc`: the same words through the multicast mapping, where every rank pushes.
- `flags`, int32 `[4]`: `[0]` the buffer of the next call; `[1]` unused (zero); `[2]` the ready words' epoch,
  advanced only by a head_flags build of `k3_moe` (`moe/k3_moe`); `[3]` the sign-ins of the CTAs that read `[0]` in
  the current call.
- `ready`, int32 `[32]`: `[t]` token `t`'s routing and `[8 + t]` its MXFP8 row, released as `epoch + 1` by a
  publishing front (`K3MoeLayer.front` on a head_flags state); `[16, 32)` unused.
- `rank`, `world_size`; `handle`, the `McastGPUBuffer` that owns the memory (the workspace is valid while this object
  lives); `comm`, the TP-group communicator the handles were exchanged over.

The size depends on `W` only, not on `M`: every call fits.

**Who creates it, and when.** The target, in `post_load_weights`, with `K3MoeHeadWorkspace.create(mapping,
fabric_handle=None)`: collective over `mapping`'s TP group (every rank calls it at the same point; it returns on every
rank or raises on every rank, the agreement also being the barrier that keeps any rank from pushing into a peer's
buffer before the peer has emptied it); eager: under CUDA-graph capture it raises `RuntimeError` before entering the
collective (certified, on every rank). It empties every word and zeroes `flags` and `ready` (certified, both of the
matrix's workspaces). `fabric_handle`: share the memory by fabric handle (required across nodes) or POSIX file
descriptor; default `mapping.is_multi_node()`. No environment variable is read.

**Which ops may share one object.** Every MoE front call of the TP group: this entry and `moe/k3_moe`'s
`k3_moe_fused_front`, on a plain or a head_flags `K3MoeState`. They form one sequence on the workspace: mixed in one
step (the fused front on each state, then the front alone) they all stay correct (certified, below). Separate from the
MNNVL all-reduce workspace and the sandwich workspace: a front call advances neither. Two workspaces are two
independent rotations and two independent epochs: 20 calls alternating between two workspaces in an irregular
pattern, kinds mixed, each return the bits of the same call made alone, and each workspace's epoch advances by its
own head_flags calls only (certified). They are not independent orders: each call spins until its peers' pushes of
the same call arrive and the calls of one stream run one after the other, so ranks that order calls on two
workspaces differently on one stream would deadlock (the kernel's design; not exercised).

**Call-order invariant.** Every rank of the group makes the same sequence of front calls on one workspace (the same
number of calls, the `k`-th with the same `M`), eager calls and graph replays alike, with the same `x`; and on one
stream the same order of calls across workspaces and other collectives. Each call reads `flags[0]` (its buffer),
pushes its latent slice and router partials into that buffer on every rank, polls its own copy until every rank's
pushes of this call are there, and empties what it read. Every CTA that reads `flags[0]` signs in on `flags[3]` right
after the read; the CTA that flips `flags[0]` for the next call waits for all of them and zeroes `flags[3]` (the
kernel's statement; certified: `flags[0]` flips once per call, and `flags[3]` is zero after each single call and
each sequence).

**What a later launch reads.** `flags[0]`, and its buffer's words, which must be empty except for this call's pushes;
`flags[3]` at zero. A publishing front also reads the epoch `flags[2]`, before it lets its dependent launch (the
kernel's statement), and releases its ready words as `epoch + 1`; `k3_moe`, not the front, reads the ready words.

**How it is re-armed.** By its readers: each CTA writes the empty word back over every word it read, which are the
words of the same call's tokens. The next push into a buffer comes from a call two calls later, which starts only
after this call has ended on this rank (the kernel's statement, from the stream order and the grid-dependency waits).
So no separate clear and no record of the previous call's size is needed, and a call after a smaller one finds no
word of an older, larger one. Certified: every word of this rank's buffers empty and `flags[1]`, `flags[3]` zero
after each single call and after each sequence below; decode steps of three layers (the fused front on the plain
state, the fused front on the head_flags state, the front alone) at `M` 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8 with new
inputs every call, back to back, a random rank 5 ms late before every call: every call the bits of the same call made
alone (itself checked against the references).

The ready words are re-armed by the head_flags `k3_moe` (`moe/k3_moe`, *State*): at its last tile claim it writes
`epoch + 1` into the words of the tokens past its `M` and into `flags[2]`, so after every head_flags call `flags[2]`
and all 16 words hold one value and the next call waits for a value no word holds. Certified: before every head_flags
call made alone no word it polls already holds `epoch + 1`, and after it the epoch advanced by one with all 16 words
at it; after every back-to-back sequence and every replayed step, the epoch advanced by the number of head_flags
calls with all 16 words at it; across -1 -> 0 (from zeroed words and epoch 0, calls at `M` 1, 1, then the epoch
preset to -2 and calls at `M` 1, 8, 3, 8: the `M` 8 call at epoch -1 waits for 0, the value a word past an earlier
call's tokens would still hold without that re-arm) and across 2^31 - 1 -> -2^31 (epoch preset to 2^31 - 2, calls at
`M` 8, 1, 8), each call's outputs the plain build's bits. The standalone front leaves `flags[2]` and the ready words
untouched (certified). Nothing else may write them: a caller that resets `flags[2]` while the words hold
`epoch + 1` would let the next head_flags `k3_moe` read the front's outputs before the front writes them (not
exercised).

**What a wrong order does.** Certified at `W` = 4 (the negative control): rank 0 issues two same-shaped front calls
on one workspace in swapped order. Nothing raises and nothing hangs, the rotation positions still agree, but each call
pairs with the peers' call at the same position: every rank's gathered head mixes rank 0's slice of its input with
the peers' slices of theirs. Every rank returns the same wrong routing and latent, bit for bit those of the reference
of the mixed inputs; the MXFP8 latent's columns `[0, 3584 / W)` and their scales are bit for bit those of rank 0's
input's call and the other columns those of the peers' input's call, so each rank's latent codes differ from those of
the call it made in more than half of the mixed-in columns. The shared activation, local to each rank, is right for
each rank's own input, bit for bit. A plain call right after is correct. Ranks calling with different `M` at one
position, or one rank making a call more or fewer, were not exercised; a rank would then poll for words its peers
never push.

## Metadata consumed

Besides `workspace` (an explicit argument):

- A process-wide cache of compiled kernels keyed by (`W`, `shared_cols`, tiles, clusters, `K`, ring, `gate_cap`,
  `linear_cap`, publish, PDL, half-tile head). The first call of a key compiles (seconds) and must be eager: under
  capture it raises `RuntimeError` ("must run once per configuration outside CUDA-graph capture first") before any
  launch, the workspace untouched (certified with other SiTU caps). The cache is result-neutral.
- `TRTLLM_ENABLE_PDL` (default on), read on every call and part of the key; it changes scheduling, not results (the
  op's statement).
- The device's cluster capacity (`max_clusters`, cached per device), which sizes the grid and picks the head's
  geometry: 128-row tiles, or one round of 64-row half-tiles when they fit next to the shared tiles
  (`half_geometry`; the op's statement: TP16). At `W` 4 the head (1120 rows per rank, 18 half-tiles) does not fit in
  one round, so the 4-rank run uses 128-row tiles.

## Preconditions

- sm_100 (tcgen05, clusters of 8 CTAs, one per SM) and `weight_supported(W, shared_cols, K, device)`: `W` in {4, 8,
  16}, `K` a multiple of 1024, `shared_cols` a multiple of 64, and the tiles fitting the device's clusters (true for
  `W` 4, 384 columns, `K` 7168 on the certified device). Any unsupported call (`M` outside 1-8, a `w_front` of
  another shape or dtype, another `W`) raises `ValueError` before it touches the workspace (the op's check; certified
  for `M` 0 and 9 on every rank, the workspace's words, flags and ready words unchanged, the next call correct). A
  workspace smaller than `workspace_words(W)` raises `ValueError` too.
- Every rank calls with the same `x` and `M`, its own head slice in `w_front`; the call order is the *State*
  invariant. Nothing checks that `x` agrees across ranks: a rank with another `x` mixes its slice into every rank's
  result (the negative control shows that effect).
- `workspace` was created, and the kernel compiled (one eager call per configuration), before any capture. Calls may
  be captured: certified with a captured step of three calls at `M` 8 (the fused front on the plain and on the
  head_flags state, the front alone) replayed 6 times with rewritten inputs, an eager call of another `M` on the same
  workspace between replays, every replayed and eager call the bits of the same call made alone, the epoch advanced
  once per head_flags call, replayed or eager.

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, the head sharded over those 4 ranks (1120
  rows per rank, 128-row tiles), the shared activation at TP16's per-rank width (384). Kimi K3 TP16 shards the head
  over 16 ranks on four trays (280 rows per rank), which runs the half-tile geometry and which only a 16-rank run
  reaches; the matrix takes `--world-size` and `--launcher`, and that receipt is pending.
- Test: `tests/unittest/_torch/modeling_v2/comm/_k3_moe_front_op_matrix.py` (rank body), collected by
  `tests/unittest/_torch/modeling_v2/moe/test_modeling_v2_k3_moe_front_op_matrix.py`. It also certifies
  `moe/k3_moe`'s `k3_moe_fused_front` cells.
- Reference: native torch for the head (every rank's slice in fp64, exact for the payloads, then fp32; the latent
  columns rounded to bf16) and the shared gate_up (fp64, exact, rounded to bf16) with SiTU in fp32; the stock
  `trtllm::kimi_k3_noaux_tc_mxfp8_quant` for the routing and quantization of the gathered head. Every rank draws every
  rank's head slice from one seed, so each holds the whole reference. The routed experts of the fused cells are
  random checkpoint-format MXFP4 (224 per rank, rank `r` at global ids `[224 (r % 4), 224 (r % 4) + 224)`), the
  reference for `y` the stock TRTLLM-Gen runner. The kernel test
  (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_moe_front.py`) covers Gaussian payloads and races the
  publishing front's epoch read against `k3_moe`'s epoch advance (`check_publish_order`); neither is repeated here.
- `mutates_args` names every buffer the op writes (`ag_uc`, `ag_mc`, `ag_flags`, `ag_ready`). Gaps (the op is
  unchanged by this entry): the compile cache is a module-level dict (result-neutral, documented above); the slots'
  logit vectors are dead space for this kernel; `ring` is not exposed.
- Before this entry the head workspace was `head_workspace(mapping)`, a module dict created on the first eager call;
  this entry takes the explicit object instead.
