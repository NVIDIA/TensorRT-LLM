---
receipts:
  sm_100: {status: passed, tests: 62}
---

# k3_moe

**Wraps** `torch.ops.trtllm.k3_moe` (one call), on caller-owned state: a `K3MoeState` (up to 8 tokens) or
`K3MoeWideState` (up to 64) and one `K3MoeLayer` per MoE layer; for a head_flags state, also the TP group's
`K3MoeHeadWorkspace`; for the push form (`k3_moe_push`), also the TP group's `K3LatentExchange`.

## Semantics

Kimi K3's routed experts at decode size: this rank's routed partial, i.e. for each token the sum over its top-16
experts that this rank holds of the expert's output times its routing weight, from the persistent CuTe DSL kernel
`k3_moe`. Its inputs are the routing and MXFP8 latent of `moe/k3_route_quant` or `moe/k3_moe_front`: `topk_ids`,
`topk_weights`, and `x_fp8` with its UE8M0 scales `x_sf` (`x[t] = x_fp8[t] * 2^(x_sf[t] - 127)` per 32 columns). Per
token `t`:

```
y[t] = bf16( sum over the slots j with offset <= topk_ids[t, j] < offset + num_local of
             topk_weights[t, j] * FC2_e(q8(SiTU(FC1_e(x[t])))) ),    e = topk_ids[t, j] - offset
FC1_e(x) = (x @ W_up[e]^T, x @ W_gate[e]^T)                     # [i_tp] each, fp32 accumulation
SiTU     = 4 tanh(gate / 4) sigmoid(gate) * 25 tanh(up / 25)    # caps 4.0 / 25.0
q8       = MXFP8 per 32 intermediate columns, scale 2^ceil(log2(amax / 448)), e4m3 round to nearest even
FC2_e(a) = a @ W_down[e]^T                                      # [3584], fp32
```

`W_up[e]`, `W_gate[e]` (`[i_tp, 3584]`) and `W_down[e]` (`[3584, i_tp]`) are the values of the layer's MXFP4 expert
`e` (E2M1 codes times `2^(E8M0 - 127)` per 32 K elements), read in place from the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers
(*Preconditions*). `offset` is `local_expert_offset`, `num_local` the state's. The SiTU caps are constants of the
kernel (Kimi K3's `activation_situ_beta` 4.0 and `activation_situ_linear_beta` 25.0), not arguments. The combine adds
each token's expert terms in fp32 in an order fixed by the call's grouping.

Numerics, certified with `moe/k3_route_quant` as the producer, for the M <= 8 build at every `M` 1-8 and for the wide
build at `M` 1, 2, 7, 8, 9, 16, 33, 40 and 64, in three routing cases each: random logits and no local expert for
both; 16 local experts per token, none shared (128 groups at `M` 8, the M <= 8 build's group capacity), for the M <= 8
build; 100 local experts with 9 of 64 tokens and 124 with one (324 groups at `M` 64, the wide build's capacity), for
the wide build:

- within the op-catalog gates (8 bf16 ulp of the token row's largest magnitude per element, 4 ulp relative RMS) of an
  fp64 reference of the formula above over the dequantized experts, and of the stock path
  (`kimi_k3_noaux_tc_mxfp8_quant`, then `mxe4m3_mxe2m1_block_scale_moe_runner` pre-routed with SiTU);
- run-to-run bit identical;
- a token's row may round differently when other tokens share the call: FC2 adds a token's expert terms in slices
  whose bounds follow the call's group count. At most 1 bf16 ulp: each `M`'s rows against the same rows of the
  8-token call, and the wide build's rows at `M` <= 8 against the M <= 8 build's;
- a token with no expert on this rank gets a zero row.

With `moe/k3_moe_front` as the producer (certified in that entry's 4-rank matrix): `y` within the op-catalog gates of
the stock runner on the front's own routing and latent, and on a head_flags state bit for bit the plain state's.

**Push form.** `k3_moe_push` (`M` <= 8, a `K3MoeState`) computes the same partial and, instead of returning it,
stores token `t`'s row into slot `slot` (default `exchange.rank`) of half `exchange.flags[0] & 1` of every rank's
`K3LatentExchange` (int32 `[2][8][slots][1792]`: bf16 pairs, `0x80000000` empty, -0.0 stored as +0.0, zero rows when
nothing is routed here) through its multicast mapping: the kernel's fused all-reduce in its push-only mode. It reads
the half after its grid-dependency wait and writes no flags word; `comm/k3_latent_reduce` sums the slots in the MNNVL
one-shot's order, empties the half it read and advances the count. Certified at 4 ranks, `M` 1, 3 and 8, after
`moe/k3_route_quant` and after `moe/k3_moe_front`: a push and its reduce equal `MNNVLAllReduce`'s one-shot of the
plain partials bit for bit, on every rank and run to run, and leave the exchange empty with its count advanced; the
same into a 16-slot exchange filled 4 slots per rank (the one-shot's 16-slot order), and with the exchange's call
count across the int32 wrap.

Fusion boundary. Inside: the grouping of (expert, token) pairs, FC1, SiTU, the MXFP8 intermediate, FC2, the
routing-weighted combine. Outside: the routing and the latent's MXFP8 quantization (`moe/k3_route_quant` or
`moe/k3_moe_front`); the sum of the routed partials over the ranks that hold the other experts and intermediate
slices (the routed-latent all-reduce, or the push form plus `comm/k3_latent_reduce`); the latent-up projection; the
shared experts; the residual.

## Signature

```python
def k3_moe(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeLayer,
    head: Optional[K3MoeHeadWorkspace] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor

def k3_moe_push(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeLayer,
    exchange: K3LatentExchange,
    slot: Optional[int] = None,
    head: Optional[K3MoeHeadWorkspace] = None,
) -> None
```

The entry passes the layer's weight buffers and counters and its state's scratch and build options to the op:
`trtllm::k3_moe(x_fp8, x_sf, topk_ids, topk_weights, w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale,
c, cs, part, counters, local_expert_offset, num_local, num_ctas, m_max, use_pdl, head_ready=None, head_flags=None,
out=None, exchange_uc=None, exchange_mc=None, exchange_flags=None, exchange_slot=0)`, with `mutates_args = (c, cs,
part, counters, head_ready, head_flags, out, exchange_uc, exchange_mc)`: every buffer the kernel writes. The push form
passes the exchange's `uc`, `mc` and `flags` and the slot (`K3MoeLayer.push` makes the same call). The wrapper module re-exports `K3MoeState`, `K3MoeWideState`, `K3MoeLayer`, `K3MoeHeadWorkspace` and
`is_supported`.

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x_fp8` | `[M, 3584]`; `M` 1-8 on a `K3MoeState`, 1-64 on a `K3MoeWideState` | float8_e4m3fn | contiguous | CUDA, the state's device |
| `x_sf` | `[M, 112]` (`M * 112` bytes, linear: one byte per 32 columns) | uint8 (UE8M0) | contiguous | CUDA |
| `topk_ids` | `[M, 16]`, global expert ids | int32 | contiguous | CUDA |
| `topk_weights` | `[M, 16]` | bf16 | contiguous | CUDA |
| `local_expert_offset` | scalar: the global id of the layer's expert 0 (224 certified: experts `[224, 448)`) | Python int | — | — |
| `layer` | a `K3MoeLayer` of a plain or head_flags `K3MoeState`, or of a `K3MoeWideState` | — | — | — |
| `head` | `None`; for and only for a head_flags state's layers, the TP group's `K3MoeHeadWorkspace` (certified in `moe/k3_moe_front`'s matrix) | — | — | — |
| `out` | `None`, or `[>= M, 3584]`: the call writes `out[:M]` and returns an empty `[0, 3584]`; rows past `M` untouched (certified at `M` 3, 8 and 9, 64 on the two builds) | bf16 | contiguous | CUDA |
| `exchange` (push form) | the TP group's `K3LatentExchange` (4 ranks certified, and a 16-slot exchange) | — | — | — |
| `slot` (push form) | `None` (this rank's) or a slot of the exchange (certified: each of 16 slots, 4 per rank) | Python int | — | — |
| returns | `y [M, 3584]`, or `[0, 3584]` with `out`; `None` for the push form | bf16 | contiguous, newly allocated | the inputs' device |

The four inputs are `moe/k3_route_quant`'s outputs for `M` tokens (certified) or `moe/k3_moe_front`'s (certified in
its matrix). State construction certified: `K3MoeState(device, 768, 224)` (`head_flags` False or True; `use_pdl` and
`num_ctas` at their defaults), `K3MoeWideState(device, 768, 224)` (`use_pdl` True), and `state.layer(w3_w1_weight,
w3_w1_weight_scale, w2_weight, w2_weight_scale)` over this rank's 224 experts in the TRTLLM-Gen W4A8_MXFP4_MXFP8
layout at intermediate 768: `[224, 1536, 1792]`, `[224, 1536, 112]`, `[224, 3584, 384]`, `[224, 3584, 24]`, uint8.

## State

**Objects.** Per rank and per device, owned by the caller (`cute_dsl_kernels/k3_fused_moe/op.py`, re-exported by
`catalog/moe/k3_moe.py`): a build's scratch, `K3MoeState` (`m_max` 8) or `K3MoeWideState` (`m_max` 64), and one
`K3MoeLayer` per MoE layer from `state.layer(...)`. A head_flags `K3MoeState`'s calls also read and write the TP
group's `K3MoeHeadWorkspace` (collective; its *State* is in `moe/k3_moe_front`).

**Contents and size.** At the certified layout (`i_tp` 768, `num_local` 224), certified right after construction:

| Object | Tensor | Shape and dtype | Bytes | Between calls |
|---|---|---|---|---|
| `K3MoeState` | `c`: the FC1 -> FC2 intermediate slab | int8 `[G, 8, i_tp]`, `G = min(num_local, 128)` = 128 | 786,432 | armed: every byte 0x80 (FP8 -0.0) |
| | `cs`: its E8M0 scales | int8 `[G, 8, i_tp / 8]` | 98,304 | armed: bytes 0-3 of every 16-byte group 0xFF (E8M0 NaN), bytes 4-15 zero |
| | `part`: the FC2 partial rows | fp32 `[8 G, 3584]` | 14,680,064 | zero when built; then what the last call left |
| `K3MoeWideState` | `c`, `cs` | int8 `[G, 8, i_tp]`, `[G, 8, i_tp / 8]`, `G = 224 + (1024 - 224) / 8` = 324 | 1,990,656 + 248,832 | armed, as above |
| | `part` | fp32 `[1024, 3584]` | 14,680,064 | zero when built; then what the last call left |
| `K3MoeLayer` | `counters` | int32 `[32 + 2 G]`: `[288]` on a `K3MoeState`, `[680]` on a `K3MoeWideState` | 1,152 / 2,720 | zero |

`G` is the build's group capacity (`group_capacity` in the kernel module): an (expert, up to 8 tokens) group per local
expert a call routes to, at most `min(num_local, 8 x 16)` for `M` <= 8; the wide build gives an expert with `t`
tokens `ceil(t / 8)` groups. A state also holds its build's options (`m_max`, `num_ctas`: one CTA per SM by default,
`use_pdl`, `head_flags`; certified), its kernel module and `compiled` (whether its build is in the compile cache). A
layer holds its four weight tensors themselves (no copy; certified) and its counters; it does not hold
`local_expert_offset`.

**Who creates it, and when.** The target, in `post_load_weights`, after the expert weights are final (a layer reads
the buffers it was built over; see *What a wrong order does*): `K3MoeState(device, i_tp, num_local,
head_flags=False, use_pdl=None, num_ctas=None)` and `K3MoeWideState(device, i_tp, num_local, use_pdl=True)` once per
device, `state.layer(w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)` once per MoE layer. Not
collective. Eager: the four constructors raise `RuntimeError` under CUDA-graph capture (certified). Each build
compiles on its first call on a device (seconds; one process-wide cache keyed by device and build options), which must
be eager: under capture that call raises `RuntimeError` before the `k3_moe` launch and the build stays uncompiled
(certified for both builds, with the cache cold).

**Which ops may share one object.** All layers of a state share its slab and partial rows; each layer has its own
counters. Layers of one state may be built over the same weight buffers (two counter sets) or over different ones
(certified, both). Separate states share nothing: with calls on two `K3MoeState`s and a `K3MoeWideState` interleaved
in an irregular pattern (26 calls, back to back), each returns the bits of the same call made alone (certified). The
entry passes `head` for, and only for, a head_flags state's layers: both mismatches raise `ValueError` before any
launch (certified). The op itself picks the head_flags build by whether `head_ready` / `head_flags` are passed; this
check, and the same one in `K3MoeLayer`, is what ties the build to the state. A head_flags state's calls pair with the
front's on one `K3MoeHeadWorkspace`: the front call before each must publish the workspace's ready words (the
`moe/k3_moe_front` entry with `publish=True`), and each publishing front call must be followed by exactly one such
`k3_moe` call on that workspace (*Preconditions*). The push form shares the state with the plain calls. Its exchange
is a separate, collective object (`comm/k3_latent_reduce`'s *State*): each push of `M` tokens is followed by one
reduce of `M` tokens on that exchange before the next push, in the same order on every rank.

**Call-order invariant.** The calls on all layers of one state run one after the other in one stream order. Every
call needs the slab armed and its layer's counters at zero, which only the end of the previous call on the state
guarantees, and a call claims its first tile on its layer's counters before its grid-dependency wait (the kernel's
statement; *Preconditions*). Calls on one state from two streams at once are not exercised: nothing in the state keeps
two concurrent calls' groups apart. The state is per rank: no cross-rank order, except through the head workspace on
a head_flags state (`moe/k3_moe_front`).

**What a later launch reads.** The slab armed: FC2 treats a group's intermediate as written only once its FP8 -0.0
and E8M0 NaN sentinels are gone. The layer's counters at zero: the tile-queue cursor, the FC2 m-tile arrivals, the
per-group FC1 and FC2 counts. The partial rows: the M <= 8 build's combine also loads the rows of the token slots past
`M`, whose sums it drops; they are zero in a new state, so those loads never read unwritten memory (the op module's
statement). On a head_flags state: the workspace's epoch `flags[2]` and its ready words (below).

**How it is re-armed.** By `k3_moe` itself, on every call: the last FC2 task that reads a group's intermediate puts
its sentinels back, and each counter is reset by its last user (the cursor by the grid's last claim). Nothing is
cleared between calls and nothing records a call's size, so a call after a smaller one reads nothing an older,
larger call left. Certified: the slab armed and every counter zero after every single call and every sequence of
the test; decode steps of three layers at `M` 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8 and of two wide layers at `M` 64, 64,
16, 1, 40, 64, 8, 23, 64, 9, 64, new inputs every call, back to back, each call the bits of the same call made alone
(itself within the gates of the stock path); a captured step (three M <= 8 calls and one wide call, each routed by
`k3_route_quant` inside the step) replayed 6 times with rewritten inputs and eager calls of other `M` on the same
states between replays, every replayed and eager call the bits of the same call alone.

On a head_flags state, the ready-word handoff (the kernel's statement). The front releases `ready[t]` (token `t`'s
ids and weights) and `ready[8 + t]` (its MXFP8 row) as `epoch + 1`, with `epoch = head.flags[2]`; `k3_moe` reads the
epoch, acquires those words for its `M` tokens instead of waiting for the front's grid, and at its last tile claim
writes `epoch + 1` into the ready words of the tokens past `M` and into `flags[2]`. So after every head_flags call
`flags[2]` and all 16 words hold one value, and the next call waits for a value no word holds, across the int32
wraps too. Certified in `moe/k3_moe_front`'s matrix: before every head_flags call made alone no polled word already
holds `epoch + 1`, and after it the epoch advanced by one with all 16 words at it, across -1 -> 0 and
2^31 - 1 -> -2^31 too; after every back-to-back sequence and every replayed step, the epoch advanced by the number of
head_flags calls with all 16 words at it.

**What a wrong order does.** The negative control (certified): a layer reads the weight buffers it was built over.
Layers of both builds are built over one set of experts whose weights are then "reloaded" by rebinding to new tensors
(the experts rolled by one), as a loader that replaces its parameters would. Nothing raises, and each layer keeps
returning the old experts' partial bit for bit, outside the op-catalog gates of the new experts' stock path:
silently stale. Copying the new weights into the old buffers in place is seen by the next call (the new experts'
partial, bit for bit), and layers built over the new tensors are correct. So build the layers once the weights are
final, and reload weights in place. Not exercised: one state on two streams at once (a race, no deterministic
control); breaking the head_flags pairing. Without a publishing front before it, a head_flags `k3_moe` waits for
ready words nobody releases; after a publishing front not followed by one, the epoch does not advance, and the next
head_flags call finds its words already at `epoch + 1` and can read the routing before the front writes it (the
matrix checks that precondition before every head_flags call made alone).

## Metadata consumed

Besides the state objects (explicit arguments):

- The launch's programmatic dependent launch is the state's `use_pdl`: a `K3MoeState` built with `use_pdl=None` (the
  default) reads `TRTLLM_ENABLE_PDL` (default on) at construction; a `K3MoeWideState` defaults to `True`. The op
  itself reads no environment. PDL changes scheduling, not results (the op's statement); the tests run with the
  default.
- Process-wide caches, result-neutral: the compiled kernels keyed by (device, build options) (`op._compiled`) and the
  kernel modules keyed by build options (`op._modules`). A state's `compiled` reads the former.

## Preconditions

- sm_100, with the CuTe DSL package (`is_supported`): `state.layer()` and the op raise `ValueError` otherwise.
- The weights are the four contiguous uint8 buffers W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod writes (rows `[up; gate]`
  interleaved and shuffled in 32-row blocks, scales block-interleaved 128 x 4): `w3_w1_weight [E, 2 i_tp, 1792]`,
  `w3_w1_weight_scale [E, 2 i_tp, 112]`, `w2_weight [E, 3584, i_tp / 2]`, `w2_weight_scale [E, 3584, i_tp / 32]`,
  with `E` the state's `num_local` (at most 896) and `i_tp` the state's, a multiple of 128. Anything else raises
  `ValueError` in `state.layer()`.
- The four inputs as *Certified arguments* (the op checks dtypes, shapes and contiguity: `ValueError`, certified for
  int64 ids); `M` 1-8 (`K3MoeState`) or 1-64 (`K3MoeWideState`): `M` 0 and 9, and 0 and 65, raise `ValueError` before
  any launch. Certified: every state's slab, partial rows and counters keep their bits through the refused calls, and
  the next call returns the bits of the same call made before. `out` with fewer than `M` rows raises `ValueError`
  (certified).
- `local_expert_offset` is the global id of the expert the layer's buffers start with. The layer does not record it:
  another value computes other global ids with these weights, without an error.
- Inputs on the state's device, with that device current (not checked).
- On a head_flags state: `head` is the TP group's `K3MoeHeadWorkspace` whose ready words the front call just before
  published (`moe/k3_moe_front` with `publish=True`), and every publishing front call is followed by exactly one such
  call (*State*).
- Before its grid-dependency wait (with `use_pdl`, `k3_moe` launches as a programmatic dependent of the kernel before
  it), the kernel touches, from its code:
  - M <= 8 and wide builds without head_flags: only the layer's `counters` word 0, the tile-queue cursor, one atomic
    add per CTA. It reads its inputs, the weights and the scratch, and writes `part` and the output, only after the
    wait. So the kernel before it must not write the layer's counters (no producer does), and the previous call on
    the same layer must have ended by the time that kernel lets `k3_moe` launch. On one stream that holds when that
    kernel triggers its dependents only after its own grid-dependency wait, as `k3_route_quant` (its early trigger
    comes right after that wait) and `k3_moe_front` (its role CTAs trigger after theirs) do, and every kernel since
    the previous call waited for its predecessor before finishing.
  - head_flags build: it does not wait for the front's grid until its FC2 phase. Before that it reads
    `head.flags[2]`, acquires `head.ready[t]` and `head.ready[8 + t]` for its tokens and then reads the routing and
    the MXFP8 rows they release, streams the weights, writes the slab and the layer's counters, and at its last claim
    writes `head.ready` past `M` and `head.flags[2]`. It waits for the front's grid before writing `part` and the
    output, and before it exits. So the front must write each token's routing and MXFP8 row before releasing their
    ready words (it fences, then releases), and must not write the weights, the scratch, the counters or
    `head.flags[2]` (it does not).
  - Either build lets its own dependents launch early (plain builds right after the wait, the head_flags build at
    launch): a consumer of `y` must wait for `k3_moe`'s grid (`griddepcontrol.wait`, or plain stream order) before
    reading it.
- Push form: `M` <= 8 on a `K3MoeState`, no `out`; the exchange's words are int32 `[2][8][slots][1792]` (this
  rank's and multicast) with int32 flags, and `slot` is one of its slots (`ValueError` otherwise, before any launch).
  The exchange belongs to this rank's TP group, and one reduce of `M` tokens on it follows each push before the next,
  on every rank in the same order. The push build compiles on its first push, which must be eager like the plain
  build's.
- Calls may be captured once the build's first call has run eagerly: certified with the captured step above, and in
  `moe/k3_moe_front`'s matrix on both K3MoeStates.

## Notes

- Certified path: one GB200 GPU (sm_100), test `tests/unittest/_torch/modeling_v2/moe/test_modeling_v2_k3_moe.py`,
  at the Kimi K3 TP16 deployment's routed-expert rank layout (experts TP4 x EP4: 224 local experts of 896,
  intermediate 768), with random checkpoint-format MXFP4 experts put through TRT-LLM's own loader, routed by
  `moe/k3_route_quant`. The head_flags build and the front as producer: 4 ranks of one GB200 tray in
  `tests/unittest/_torch/modeling_v2/comm/_k3_moe_front_op_matrix.py` (entry point
  `moe/test_modeling_v2_k3_moe_front_op_matrix.py`); its 16-rank receipt is pending with `moe/k3_moe_front`'s.
- The push form: 4 ranks of one GB200 tray in `tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_moe_push.py`
  (`k3_moe_push`: `K3MoeLayer.push` into a 4-slot exchange, this entry's `k3_moe_push` into a 16-slot one); its
  16-rank receipt is pending.
- References: an fp64 reference over the dequantized experts (from the checkpoint-format tensors) and the stock path.
  The kernel tests (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_fused_moe.py`, `test_k3_moe_wide.py`)
  remain the exhaustive numerics; this entry's test copies their references.
- Gaps (the op is unchanged by this entry): a layer does not record `local_expert_offset`; nothing ties a state to a
  stream or checks the inputs' device; the head_flags pairing with the front is the caller's (the entry checks only
  that `head` matches the state); the compile cache is a module-level dict (result-neutral).
