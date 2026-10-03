---
receipts:
  sm_100: {status: pending, world_size: 4}
---

# k3_moe

**Wraps** three calls of the caller-owned layer objects in `cute_dsl_kernels/k3_fused_moe/op.py`, one each. The
expert kernel `k3_moe` is a CuTe DSL kernel launched through its compiled function, not a torch op.

| Function | Runs | Tokens `M` |
|---|---|---|
| `k3_moe` | `K3MoeLayer.__call__`: `torch.ops.trtllm.k3_route_quant`, then `k3_moe` as its programmatic dependent | 1-8 |
| `k3_moe_fused_front` | `K3MoeLayer.front`: `torch.ops.trtllm.k3_moe_front` (entry `moe/k3_moe_front`), then `k3_moe` | 1-8 |
| `k3_moe_wide` | `torch.ops.trtllm.k3_route_quant(early_trigger=True)`, then `K3MoeWideLayer.__call__`: the m_max 64 build of `k3_moe` | 1-64 |

## Semantics

Kimi K3's routed experts at decode size: this rank's routed partial, i.e. for each token the sum over its top-16
experts that this rank holds of the expert's output times its routing weight. Two launches on the current stream, no
host synchronization (the op module's statement).

Routing and MXFP8 quantization of the latent, `k3_route_quant` (inside `k3_moe` and `k3_moe_wide`): the CuTe DSL
form of `trtllm::kimi_k3_noaux_tc_mxfp8_quant`, whose four outputs it returns bit for bit (certified at `M` 1-8, 16,
33 and 64, with and without the early dependent trigger). Per token, as the kernel module states it:

```
s       = 0.5 * tanh(0.5 * router_logits) + 0.5                      # fp32, [896]
ids     = the 16 experts with the largest s + e_score_correction_bias, descending, ties to the lower id
weights = bf16(s[ids] * routed_scaling_factor / (sum of the 16 s[ids] + 1e-20))       # in fp64
xq, xs  = MXFP8(latent): e4m3 codes, one UE8M0 scale per 32 columns, scale 2^ceil(log2(amax / 448))
```

The experts, `k3_moe`, per token `t`:

```
y[t] = bf16( sum over the slots j with offset <= ids[t, j] < offset + num_local of
             weights[t, j] * FC2_e(q8(SiTU(FC1_e(x_t)))) ),    e = ids[t, j] - offset
x_t      = xq[t] * 2^(xs[t] - 127)                              # the dequantized latent row, [3584]
FC1_e(x) = (x @ W_up[e]^T, x @ W_gate[e]^T)                     # [i_tp] each, fp32 accumulation
SiTU     = 4 tanh(gate / 4) sigmoid(gate) * 25 tanh(up / 25)    # caps 4.0 / 25.0
q8       = MXFP8 per 32 intermediate columns, scale 2^ceil(log2(amax / 448)), e4m3 round to nearest even
FC2_e(a) = a @ W_down[e]^T                                      # [3584], fp32
```

`W_up[e]`, `W_gate[e]` (`[i_tp, 3584]`) and `W_down[e]` (`[3584, i_tp]`) are the values of the layer's MXFP4 expert
`e` (E2M1 codes times `2^(E8M0 - 127)` per 32 K elements), read in place from the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers
(*Preconditions*). `offset` is `local_expert_offset`, `num_local` the state's. The SiTU caps are constants of the
`k3_moe` build (Kimi K3's `activation_situ_beta` 4.0 and `activation_situ_linear_beta` 25.0), not arguments; the
`gate_cap` / `linear_cap` arguments of `k3_moe_fused_front` are the shared activation's (the front's).

Numerics, certified for `k3_moe` at every `M` 1-8 and for `k3_moe_wide` at `M` 1, 2, 7, 8, 9, 16, 33, 40 and 64, in
three routing cases each: random logits and no local expert for both; 16 local experts per token, none shared (128
groups at `M` 8, the M <= 8 build's group capacity) for `k3_moe`; 100 local experts with 9 of 64 tokens and 124 with
one (324 groups at `M` 64, the wide build's capacity) for `k3_moe_wide`:

- within the op-catalog gates (8 bf16 ulp of the token row's largest magnitude per element, 4 ulp relative RMS) of an
  fp64 reference of the formula above over the dequantized experts, and of the stock path
  (`kimi_k3_noaux_tc_mxfp8_quant`, then `mxe4m3_mxe2m1_block_scale_moe_runner` pre-routed with SiTU);
- run-to-run bit identical;
- a token's row may round differently when other tokens share the call: FC2 adds a token's expert terms in slices
  whose bounds follow the call's group count. At most 1 bf16 ulp: each `M`'s rows against the same rows of the
  8-token call, and `k3_moe_wide`'s rows at `M` <= 8 against `k3_moe`'s;
- a token with no expert on this rank gets a zero row.

`k3_moe_fused_front` returns `(y, shared)`: `y` is `k3_moe` applied to the front's routing and MXFP8 latent, `shared`
the front's shared activation (`moe/k3_moe_front`). Certified in that entry's 4-rank matrix: `y` within the op-catalog
gates of the stock runner on the front's own routing and latent; `shared` bit for bit the front's; on a head_flags
state, `y` and `shared` bit for bit the plain state's.

Fusion boundary. Inside: the routing and the MXFP8 latent (`k3_moe`, `k3_moe_wide`) or the whole MoE front
(`k3_moe_fused_front`); the grouping of (expert, token) pairs; FC1, SiTU, the MXFP8 intermediate, FC2; the
routing-weighted combine. Outside: the router and latent-down GEMMs that produce `router_logits` and `latent`
(`k3_moe`, `k3_moe_wide`); the sum of the routed partials over the ranks that hold the other experts and
intermediate slices (the routed-latent all-reduce); the latent-up projection; the shared experts' down projection
(and, except in `k3_moe_fused_front`, their gate_up and activation); the residual.

## Signature

```python
def k3_moe(
    latent: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    layer: K3MoeLayer,
) -> torch.Tensor

def k3_moe_fused_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    head: K3MoeHeadWorkspace,
    layer: K3MoeLayer,
) -> Tuple[torch.Tensor, torch.Tensor]

def k3_moe_wide(
    latent: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    layer: K3MoeWideLayer,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

The wrapper module re-exports the state types (`K3MoeState`, `K3MoeLayer`, `K3MoeWideState`, `K3MoeWideLayer`,
`K3MoeHeadWorkspace`) and `is_supported`.

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `latent` | `[M, 3584]`; `M` 1-8 (`k3_moe`), 1-64 (`k3_moe_wide`) | bf16 | contiguous | CUDA, the state's device |
| `router_logits` | `[M, 896]` | fp32 | as `latent` | CUDA |
| `e_score_correction_bias` | `[896]` | fp32 | contiguous | CUDA |
| `local_expert_offset` | scalar: the global id of the layer's expert 0 (224 certified: experts `[224, 448)`) | Python int | — | — |
| `routed_scaling_factor` | scalar (2.827 certified) | Python float | — | — |
| `layer` | a `K3MoeLayer` of a plain `K3MoeState` (`k3_moe`); of a plain or head_flags one (`k3_moe_fused_front`); a `K3MoeWideLayer` (`k3_moe_wide`) | — | — | — |
| `out` (`k3_moe_wide`) | `None`, or `[>= M, 3584]`: the result is `out[:M]` (the same storage), rows past `M` untouched (certified at `M` 9 and 64) | bf16 | contiguous | CUDA |
| `x`, `w_front`, `shared_cols`, `gate_cap`, `linear_cap`, `head` (`k3_moe_fused_front`) | as `moe/k3_moe_front`; `head` that entry's `K3MoeHeadWorkspace` | — | — | — |
| returns | `y [M, 3584]` (`k3_moe_fused_front`: `(y, shared [M, shared_cols])`) | bf16 | contiguous, newly allocated (or `out[:M]`) | the inputs' device |

The state objects' certified construction: `K3MoeState(device, 768, 224)` (`head_flags` False or True, `config` None),
`K3MoeWideState(device, 768, 224)` (`use_pdl` True), and `state.layer(w3_w1_weight, w3_w1_weight_scale, w2_weight,
w2_weight_scale)` over this rank's 224 experts in the TRTLLM-Gen W4A8_MXFP4_MXFP8 layout at intermediate 768:
`[224, 1536, 1792]`, `[224, 1536, 112]`, `[224, 3584, 384]`, `[224, 3584, 24]`, all uint8.

## State

**Objects.** Per rank and per device, owned by the caller (`cute_dsl_kernels/k3_fused_moe/op.py`, re-exported by
`catalog/moe/k3_moe.py`):

- `K3MoeState` (`M` <= 8) and one `K3MoeLayer` per MoE layer from `state.layer(...)`;
- `K3MoeWideState` (`M` <= 64) and one `K3MoeWideLayer` per MoE layer;
- `k3_moe_fused_front` also takes the TP group's `K3MoeHeadWorkspace` (collective; its *State* is in
  `moe/k3_moe_front`).

**Contents and size.** At the certified layout (`i_tp` 768, `num_local` 224), certified right after construction:

| Object | Tensor | Shape and dtype | Bytes | Between calls |
|---|---|---|---|---|
| `K3MoeState` | `c`: the FC1 -> FC2 intermediate slab | int8 `[G, 8, i_tp]`, `G = min(num_local, 128)` = 128 | 786,432 | armed: every byte 0x80 (FP8 -0.0) |
| | `cs`: its E8M0 scales | int8 `[G, 8, i_tp / 8]` | 98,304 | armed: bytes 0-3 of every 16-byte group 0xFF (E8M0 NaN), bytes 4-15 zero |
| | `part`: the FC2 partial rows | fp32 `[8 G, 3584]` | 14,680,064 | zero when built; then what the last call left |
| `K3MoeLayer` | `counters` | int32 `[32 + 2 G]` = `[288]` | 1,152 | zero |
| `K3MoeWideState` | `c`, `cs` | int8 `[G, 8, i_tp]`, `[G, 8, i_tp / 8]`, `G = 224 + (1024 - 224) / 8` = 324 | 1,990,656 + 248,832 | armed, as above |
| | `part` | fp32 `[1024, 3584]` | 14,680,064 | not initialized |
| `K3MoeWideLayer` | `counters` | int32 `[32 + 2 G]` = `[680]` | 2,720 | zero |

`G` is the build's group capacity (`group_capacity` in the kernel module): an (expert, up to 8 tokens) group per local
expert a call routes to, at most `min(num_local, 8 x 16)` for `M` <= 8; the wide build gives an expert with `t` tokens
`ceil(t / 8)` groups. Each state also holds `mod` (its configuration's kernel module, from a process-wide cache keyed
by the configuration), `compiled` (the compiled `k3_moe`, built by the state's first call) and, for the M <= 8 build,
`head_flags`. A layer holds views of its four weight buffers (no copy) and its counters; it does not hold
`local_expert_offset`.

**Who creates it, and when.** The target, in `post_load_weights`, after the expert weights are final (a layer reads
the buffers it was built over; see *What a wrong order does*): `K3MoeState(device, i_tp, num_local,
head_flags=False)` once per device and `state.layer(w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)`
once per MoE layer; likewise `K3MoeWideState(device, i_tp, num_local)` and its layers. Not collective. Eager: the
four constructors (`K3MoeState()`, `K3MoeState.layer()`, `K3MoeWideState()`, `K3MoeWideState.layer()`) raise
`RuntimeError` under CUDA-graph capture (certified). The kernel compiles on the first call of each
state object (seconds), which must be eager: under capture that call raises `RuntimeError` before the `k3_moe` launch
and leaves the state uncompiled (certified for both builds). `config` (kernel options for tests and A/B runs) stays
`None`. No environment variable selects anything here except PDL (*Metadata consumed*).

**Which ops may share one object.** All layers of a state share its slab and partial rows; each layer has its own
counters. Layers of one state may be built over the same weight buffers (two counter sets) or over different ones
(certified, both). On a plain `K3MoeState`, `k3_moe` and `k3_moe_fused_front` calls may be mixed (one build). A
head_flags `K3MoeState` serves `k3_moe_fused_front` only: `k3_moe` on its layers raises `ValueError` before any
launch (certified). `K3MoeWideState` serves `k3_moe_wide` only. Separate states share nothing: with calls on two
`K3MoeState`s and a `K3MoeWideState` interleaved in an irregular pattern (26 calls, back to back), each returns the
bits of the same call made alone (certified).

**Call-order invariant.** The calls on all layers of one state run one after the other in one stream order. Every
call needs the slab armed and its layer's counters at zero, which only the end of the previous call on the state
guarantees; and (the kernel's statement) a call issues its first tile claim on its layer's counters before its grid
dependency wait, relying on the previous call of that layer having completed before the producer kernel launched this
one, which holds on one stream. Calls on one state from two streams at once are not exercised: nothing in the state
keeps two concurrent calls' groups apart. The state is per rank: there is no cross-rank order, except through the
head workspace for `k3_moe_fused_front` (`moe/k3_moe_front`).

**What a later launch reads.** The slab armed: FC2 treats a group's intermediate as written only once its FP8 -0.0
and E8M0 NaN sentinels are gone. The layer's counters at zero: the tile-queue cursor, the FC2 m-tile arrivals, the
per-group FC1 and FC2 counts. The partial rows: the M <= 8 build's combine also loads the rows of the token slots past
`M`, whose sums it drops; they are zero in a new state, so those loads never read unwritten memory (the op module's
statement). With head_flags: the head workspace's epoch `flags[2]` and its ready words (below).

**How it is re-armed.** By `k3_moe` itself, on every call: the last FC2 task that reads a group's intermediate puts
its sentinels back, and each counter is reset by its last user (the cursor by the grid's last claim). Nothing is
cleared between calls and nothing records a call's size, so a call after a smaller one reads nothing an older,
larger call left. Certified: the slab armed and every counter zero after every single call and every sequence of
the test; decode steps of three layers at `M` 8, 8, 8, 2, 7, 8, 1, 1, 8, 3, 8 and of two wide layers at `M` 64, 64,
16, 1, 40, 64, 8, 23, 64, 9, 64, new inputs every call, back to back, each call the bits of the same call made alone
(itself within the gates of the stock path); a captured step (three `k3_moe` and one `k3_moe_wide` call) replayed 6
times with rewritten inputs and eager calls of other `M` on the same states between replays, every replayed and
eager call the bits of the same call alone.

With head_flags (`k3_moe_fused_front`), the ready-word handoff. The front releases `ready[t]` (token `t`'s ids and
weights) and `ready[8 + t]` (its MXFP8 row) as `epoch + 1`, with `epoch = head.flags[2]`; the head_flags `k3_moe`
reads the epoch, acquires those words for its `M` tokens instead of waiting for the front's grid, waits for the grid
only before its FC2 phase (which writes the output, memory it does not own), and at its last tile claim writes
`epoch + 1` into the ready words of the tokens past `M` and into `flags[2]` (the kernel's statement). So after every
head_flags call `flags[2]` and all 16 words hold one value, and the next call waits for a value no word holds, across
the int32 wraps too. Certified in `moe/k3_moe_front`'s matrix: before every head_flags call made alone no polled word
already holds `epoch + 1`, and after it the epoch advanced by one with all 16 words at it, across -1 -> 0 and
2^31 - 1 -> -2^31 too; after every back-to-back sequence and every replayed step, the epoch advanced by the number of
head_flags calls with all 16 words at it.

**What a wrong order does.** The negative control (certified): a layer reads the weight buffers it was built over.
Layers of both builds are built over one set of experts whose weights are then "reloaded" by rebinding to new tensors
(the experts rolled by one), as a loader that replaces its parameters would. Nothing raises, and each layer keeps
returning the old experts' partial bit for bit, outside the op-catalog gates of the new experts' stock path:
silently stale. Copying the new weights into the old buffers in place is seen by the next call (the new experts'
partial, bit for bit), and layers built over the new tensors are correct. So build the layers once the weights are
final, and reload weights in place. Not exercised: one state on two streams at once (a race, no deterministic
control); a head_flags call whose polled ready words already hold `epoch + 1`, e.g. after a caller resets
`flags[2]` (`k3_moe` would read the front's output buffers before the front writes them; the matrix checks this
precondition before every head_flags call instead).

## Metadata consumed

Besides the state objects (explicit arguments):

- `TRTLLM_ENABLE_PDL` (default on), read by `k3_route_quant` on every call (part of its compile key), by
  `k3_moe_front` on every call, and by the M <= 8 build's kernel module when a configuration is first loaded (the
  wide build takes `use_pdl` from its constructor). It changes scheduling, not results (the ops' statement); the
  tests run with the default.
- Process-wide caches, result-neutral: the kernel modules keyed by configuration (`op._modules`), `k3_route_quant`'s
  compiled kernels keyed by (early trigger, PDL), `k3_moe_front`'s keyed by its configuration. The compiled `k3_moe`
  is per state object (`state.compiled`): every new state compiles on its first call.

## Preconditions

- sm_100, with the CuTe DSL package (`is_supported`): the layer constructor raises `ValueError` otherwise.
- The weights are the four contiguous uint8 buffers W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod writes (rows `[up; gate]`
  interleaved and shuffled in 32-row blocks, scales block-interleaved 128 x 4): `w3_w1_weight [E, 2 i_tp, 1792]`,
  `w3_w1_weight_scale [E, 2 i_tp, 112]`, `w2_weight [E, 3584, i_tp / 2]`, `w2_weight_scale [E, 3584, i_tp / 32]`,
  with `E` the state's `num_local` (at most 896) and `i_tp` the state's, a multiple of 128. Anything else raises
  `ValueError` in `state.layer()`.
- `M` 1-8 (`k3_moe`, `k3_moe_fused_front`) or 1-64 (`k3_moe_wide`): `M` 0 and 9, and 0 and 65, raise `ValueError`
  before any launch, every state's slab, partial rows and counters keep their bits, and the next call returns the
  bits of the same call made before (certified).
- `local_expert_offset` is the global id of the expert the layer's buffers start with. The layer does not record it:
  another value computes other global ids with these weights, without an error.
- `k3_moe_wide`'s `latent`, `router_logits` and `e_score_correction_bias`, and `k3_moe`'s bias, contiguous:
  `k3_route_quant` raises `ValueError` otherwise (its check). `k3_moe` makes its `latent` and `router_logits`
  contiguous itself; only contiguous inputs are certified.
- Inputs on the state's device, with that device current (not checked).
- Calls may be captured once the state's first call has run eagerly: certified with the captured step above, and in
  `moe/k3_moe_front`'s matrix for `k3_moe_fused_front` on both states.

## Notes

- Certified path: one GB200 GPU (sm_100) for `k3_moe` and `k3_moe_wide`, test
  `tests/unittest/_torch/modeling_v2/moe/test_modeling_v2_k3_moe.py`, at the Kimi K3 TP16 deployment's routed-expert
  rank layout (experts TP4 x EP4: 224 local experts of 896, intermediate 768), with random checkpoint-format MXFP4
  experts put through TRT-LLM's own loader. `k3_moe_fused_front`: 4 ranks of one GB200 tray in
  `tests/unittest/_torch/modeling_v2/comm/_k3_moe_front_op_matrix.py` (entry point
  `moe/test_modeling_v2_k3_moe_front_op_matrix.py`); its 16-rank receipt is pending with `moe/k3_moe_front`'s.
- References: an fp64 reference over the dequantized experts (from the checkpoint-format tensors) and the stock path.
  The kernel tests (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_fused_moe.py`, `test_k3_moe_wide.py`,
  `test_k3_route_quant.py`) remain the exhaustive numerics; this entry's test copies their references.
- Gaps (the kernels are unchanged by this entry):
  - The `k3_moe` launch is not a torch op, so nothing declares what it writes: the state's slab and partial rows, the
    layer's counters, and with head_flags the head workspace's `flags[2]` and ready words (a torch op would name them
    in `mutates_args`). `trtllm::k3_route_quant` writes only its new outputs (`mutates_args=()`).
  - A layer does not record `local_expert_offset`, and nothing ties a state to a stream or checks the inputs' device.
  - The compiled kernel is per state object, not per configuration: a second state of the same configuration
    compiles again.
