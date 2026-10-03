---
receipts:
  sm_100: {status: passed, tests: 7}
---

# k3_moe_m2

**Wraps** `torch.ops.trtllm.k3_moe_m2` through `K3MoeM2Layer.__call__` and `K3MoeM2Layer.push`
(`tensorrt_llm/_torch/cute_dsl_kernels/k3_fused_moe/op.py`; one CuTe DSL kernel, `k3_moe_m2_kernel.py`, launched
with programmatic dependent launch).

A stateful entry: its correctness depends on a caller-owned state object, `K3MoeM2State`, which the `layer` argument
carries; see *State*.

## Semantics

Kimi K3's routed experts for two decode tokens when every expert is on this rank and a rank's intermediate is at
most 256 (the routed experts at moe TP16 x EP1: all 896 experts, a 192-wide slice of each intermediate). It computes
the same function as `moe/k3_moe_m1` at M = 2, with a different schedule: every distinct local expert of the two
tokens takes one slot in ascending id, each slot's intermediate is computed for both tokens (the MMA columns are
independent, so a token's bits do not depend on the other token), and FC2 starts per group of 4 slots as soon as
that group's FC1 rows are complete. Per token `t`, over its 16 routed experts `e` whose
global id is in `[local_expert_offset, local_expert_offset + num_local)`:

```
x       = x_fp8[t] * 2^(x_sf[t, k // 32] - 127)                  # the MXFP8 latent, [3584]
gate    = x @ W1[e].T ; up = x @ W3[e].T                        # MXFP4 weights, fp32 accumulation, [i_tp]
act     = 4 tanh(gate / 4) sigmoid(gate) * 25 tanh(up / 25)     # SiTU
h       = MXFP8(act)                                            # per 32 columns, round-up E8M0 scale
out[t]  = bf16(sum over e of topk_weights[t, e] * (h @ W2[e].T)) # [3584]
```

The experts' weights are the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in place, with the loader's zero padding of
the intermediate to `i_pad` (192 -> 256): the kernel streams the `i_tp` real values only. The combine sums a token's
experts in `trtllm::k3_moe`'s order (ascending local id, in up to 5 slices; products and sums rounded on their
own), so the result has k3_moe's bits except where FC1's two partial sums round an intermediate value differently
(one bf16 ulp of the row's max at most in the op tests). The order is fixed: the result is deterministic (certified,
run to run).

**Push form.** `k3_moe_m2_push` computes the same partials and, instead of returning them, stores each token's row
into slot `rank` of half `exchange.flags[0] & 1` of every rank's latent exchange (`K3LatentExchange`, int32
`[2][8][world][1792]`: bf16 pairs, `0x80000000` empty, -0.0 stored as +0.0) through its multicast mapping. It reads
the half after its grid-dependency wait and writes no flags word. `trtllm::k3_latent_reduce` sums the slots in the
MNNVL one-shot's order, empties the half it read and advances the count. A push and its reduce equal
`MNNVLAllReduce`'s one-shot of the plain partials bit for bit (certified at 4 ranks).

Fusion boundary. Inside: FC1, SiTU, the MXFP8 intermediate, FC2 and the routing-weighted combine of this rank's
experts. Outside: the routing and input quantization (their producer), the sum over ranks (MNNVL all-reduce, or the
push form plus `trtllm::k3_latent_reduce`), the shared expert.

## Signature

```python
def k3_moe_m2(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeM2Layer,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor

def k3_moe_m2_push(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeM2Layer,
    exchange: K3LatentExchange,
) -> None
```

`layer = state.layer(w3_w1_weight, w3_w1_weight_scale, w2_weight, w2_weight_scale)`: one MoE layer's buffers on a
state (it checks them and keeps them; it allocates nothing).

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x_fp8` | `[2, 3584]` | float8_e4m3fn | contiguous | CUDA, the state's device |
| `x_sf` | `2 x 112` E8M0 bytes, any shape | uint8 | contiguous | CUDA |
| `topk_ids` | `[2, 16]`, global expert ids | int32 | contiguous | CUDA |
| `topk_weights` | `[2, 16]` | bf16 | contiguous | CUDA |
| `local_expert_offset` | the first global id of the layer's experts (0 at TP16) | Python int | — | — |
| `layer` weights | `w3_w1_weight [E, 2 i_pad, 1792]`, `w3_w1_weight_scale [E, 2 i_pad, 112]`, `w2_weight [E, 3584, i_pad / 2]`, `w2_weight_scale [E, 3584, i_pad / 32]`; `E` = 896, `i_tp` = 192, `i_pad` = 256 certified | uint8 | contiguous (TRTLLM-Gen layout) | CUDA |
| `out` | `None`, or `[2, 3584]` | bf16 | contiguous | CUDA |
| `exchange` (push) | a `K3LatentExchange` of this rank's TP group, 4 ranks certified | — | — | — |
| returns | `[2, 3584]` (`out` when given); `None` for the push form | bf16 | contiguous | the state's device |

The inputs are read only. A call writes the state's workspace; the push form also writes every rank's exchange.
The op's `mutates_args` names them: `hbuf`, `counts`, `epochs`, `exchange_mc` (the push form) and `out`.

## State

**Object.** `K3MoeM2State` (`k3_fused_moe/op.py`), one per device, owned by the caller. Every layer's handle
(`K3MoeM2Layer`) points at it.

**Contents and size.** The workspace its layers share: `hbuf`, the intermediate rows of one call (int8
`[32 x 2 x 208]`, 13 KB); `counts`, the FC1 -> FC2 counters of the 8 groups, one 128-byte line each, in two sets by
epoch parity (int32 `[2, 8, 32]`); `epochs`, one word per CTA (int32 `[SMs]`); The compiled builds (the plain build and one push build per (exchange slots, copies)) live in the module's cache,
keyed by device and configuration.

**Who creates it, and when.** The target, in `post_load_weights`, with
`K3MoeM2State.create(device, i_tp, i_pad, num_local, push=((tp_size, 1),))`:

- eager: it allocates the workspace (all zeros) and compiles the plain build and each listed push build without
  launching anything, so it refuses to run under CUDA-graph capture;
- not collective: the state is this device's alone (the exchange is a separate, collective object);
- a build it did not compile compiles on its first call, which must also come before capture.

**Which ops may share one object.** Every layer of the model and both forms (plain and push) share one state per
device. A `K3MoeM1State` is a separate workspace. Two states are independent: calls alternating between two states in
an irregular order are all correct (certified).

**Call-order invariant.** The calls on one state run one after another in one stream order: eager calls and graph
replays alike, whichever layer they belong to. Each call is a complete kernel: its CTAs use the counter set of their
epoch's parity, CTA 0 zeroes the other set (the next call's), and every CTA advances its epoch. With programmatic
dependent launch the kernel's `griddepcontrol.wait` precedes every read of the producer's outputs and every global
write, so a call may start early beside its predecessor but touches the workspace only after it.

**What a later launch reads.** The CTAs' epochs (their parity picks the set) and the set the previous call zeroed.
The intermediate rows are written before they are read within one call; no call reads another call's rows.

**How it is re-armed.** Every call zeroes the set the next call uses and advances the epochs, so the workspace never
needs a reset. Only the parity of an epoch matters, so the int32 wrap after 2^31 calls is harmless (certified:
epochs preset at 2^31 - 2 and 2^31 - 1).

**Why the test drives call sequences.** The re-arm and the parity act only across calls: a call that left the next
set armed wrongly, or a parity that broke at the wrap, passes every single-call test. The test runs 12 steps of 3
layers on one state with the tokens changing every step, a captured step replayed with rewritten inputs and eager
calls between replays, two states interleaved, and the wrap; every call against the same call on a reference state,
bit for bit, and the epochs and the next set checked after each call.

**What a wrong order does.** Not exercised on the state: two calls that overlap on one state (two streams without
ordering) are outside the contract. The push form's order across ranks is the exchange's (`trtllm::k3_latent_reduce`:
each push followed by one reduce of the same token count, in the same order on every rank).

## Metadata consumed

None besides `layer` (its state and weights) and `exchange`. The kernel modules and their compiled builds are
cached per configuration in the process (code only, no state).

## Preconditions

- sm_100 and the CuTe DSL (`nvidia-cutlass-dsl`); at least 112 SMs.
- `op.m2_supported(...)` holds for the layer's buffers: `op.m1_supported`'s conditions and a padded intermediate of
  at most 256. The layer raises `ValueError` otherwise, and when its padded intermediate differs from the state's.
- Two tokens, inputs contiguous; a call raises `ValueError` otherwise.
- The state was created before any capture; calls may be captured (certified: 3 layers captured, replayed 4 times).
- Push form: the exchange belongs to this rank's TP group, and one reduce of 2 tokens on it follows each push before
  the next, on every rank in the same order.

## Notes

- Certified path: one GB200 GPU (sm_100), TP16 shapes (896 experts, `i_tp` 192 in buffers of 256). Tests:
  `tests/unittest/_torch/modeling_v2/moe/test_modeling_v2_k3_moe_m2.py` (the reference is fp32 over the dequantized
  checkpoint slices: 8 ulp of the row's max per element, 4 ulp relative RMS);
  `tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_moe_m2.py` (also against `trtllm::k3_moe`, two
  tokens routed to the same and to disjoint experts); the push form at 4 ranks in `test_k3_moe_push.py` (bit-exact
  against `MNNVLAllReduce`, a 16-slot exchange filled 4 slots per rank, the exchange's count across the int32 wrap).
- Kimi K3 runs the push form over 16 ranks; a 4-rank run fills a 16-slot exchange only by repeating each rank's
  partial. At 16 ranks (four GB200 trays of one rack, one distinct partial per slot) `test_k3_moe_push.py` passes in
  a recorded run.
