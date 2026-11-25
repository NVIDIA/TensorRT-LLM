# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the wide decode expert replicas (MTP3 / MTP5 decode rows 65..128) through the
target's own routed-experts methods.

On a wide decode step `_moe_topk` runs the wide replica top-k for a trunk layer with hot ids
(`_moe_topk_wide_rep`) and `_moe_experts` runs its maps on the context replicas' `64 + R` deep
prefill pair. A rank then computes the hot experts the assignment gave it from replica slots
`64 + h`, and none from their home slots. This drives `_moe_begin_layer` / `_routed` on a core
holding only the MoE state (as the wide decode capture test does) and checks:

* the four EP ranks emulated on one GPU (the expert offset per call): the sum of their outputs
  -- what the post-MoE all-reduce adds -- within the experts test's band of the fp32 reference
  of the whole layer, as the home mapping's sum is; each rank's output within the band of the
  fp32 reference of its own maps and replica stacks;
* a CUDA graph of two layers replayed on fresh inputs against the same calls run eagerly;
* the gate: only a trunk layer with hot ids on a wide decode step of up to 128 rows takes the
  replicas; wider steps keep the wide top-k's bytes.

Every global expert `g` holds the base stack's expert `g % 64` with its own per-expert scales
(`alpha1` / `alpha2` of `g`), so a replica slot holding the wrong expert, or a selection
computed on two ranks or none, moves the sum far outside the band.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL MoE kernels are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import (  # noqa: E402, E501
    fp4_quantize,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    modeling as _modeling,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.moe import (  # noqa: E402, E501
    moe_fc1_kblock as _fc1_kblock,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_copy as _copy_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_nvfp4_experts as _experts_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_router as _router_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_routing as _routing_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing as _topk_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing_rep as _rep_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing_wide_rep as _wrep_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization import (  # noqa: E402, E501
    cute_dsl_moe_quantize_nvfp4 as _quant_op,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, INTER, EXPERTS, LOCAL = 6144, 2048, 256, 64
TOP_K = _routing_op.TOP_K
TILE = _routing_op.TILE
NREP = 8
#: The favoured window (EP rank 1): a hot expert every token takes (past 64 rows, two tiles
#: past 128), experts no token takes, tied pairs.
WIN = 64
HOT = WIN + 1
EMPTY = tuple(range(WIN + 10, WIN + 14))
TIED = tuple((WIN + 20 + 2 * p, WIN + 21 + 2 * p) for p in range(16))
#: Each layer's replica slots: the hot expert, an expert no token takes, five experts of the
#: favoured window and the other windows, and an unused slot.
HOT_IDS = (HOT, EMPTY[0], WIN + 2, WIN + 3, 3, 130, 200, -1)
LAYERS = (3, 4)
ROUTED_SCALE = 2.5
FC1_KBLK = _fc1_kblock.enabled()
FC2_KBLK = _fc1_kblock.fc2_enabled()
#: The rows the target runs with replicas (`_WIDE_REP_MAX_ROWS`, the wide top-k's 65..128 range):
#: MTP3 verify / draft-step-0 rows at padded batches 17..32 (and MTP5's at 11..21) and the edges.
ODD_ROWS = (65, 68, 96, 126, 127, 128)
#: Wider decode steps (MTP5 at batch 22..32): no replicas, the wide top-k.
WIDE_ROWS = (129, 144, 150, 191, 192, 200, 256)
#: Replay against eager on the FC2 output (the wide decode capture test's band: bf16 atomics
#: add in a run-dependent order); upstream bitwise.
REPLAY_ELEMENT_ULP, REPLAY_RMS_ULP = 16.0, 2.0

_STATE: dict = {}


def _seed(tag: str) -> int:
    return zlib.crc32(tag.encode()) % 2**31


def _sf_atoms(s: torch.Tensor) -> torch.Tensor:
    n, sf_k = s.shape
    return (
        s.reshape(n // 128, 4, 32, sf_k // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(n // 128, sf_k // 4, 512)
    )


def _nvfp4(w: torch.Tensor):
    packed, sf = fp4_quantize.reference(w, torch.ones(1, device=w.device), 16, False, False)
    return packed, sf.view(w.shape[0], -1)


def _stacks():
    """The base 64-deep NVFP4 stack, then `64 + NREP` deep: slot `64 + h` holds
    `HOT_IDS[h] % 64`'s weights (zero for an unused slot)."""
    g = torch.Generator(device="cuda").manual_seed(_seed("wide replicas experts"))
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    depth = LOCAL + NREP
    fc1_w = torch.zeros(depth, 2 * INTER, HIDDEN // 2, **u8)
    fc1_s = torch.zeros(depth, 2 * INTER // 128, HIDDEN // 64, 512, **u8)
    fc2_w = torch.zeros(depth, HIDDEN, INTER // 2, **u8)
    fc2_s = torch.zeros(depth, HIDDEN // 128, INTER // 64, 512, **u8)
    for e in range(LOCAL):
        p, s = _nvfp4(torch.randn(2 * INTER, HIDDEN, generator=g, device="cuda") * 0.02)
        fc1_w[e], fc1_s[e] = p, _sf_atoms(s)
        p, s = _nvfp4(torch.randn(HIDDEN, INTER, generator=g, device="cuda") * 0.03)
        fc2_w[e], fc2_s[e] = p, _sf_atoms(s)
    for h, e in enumerate(HOT_IDS):
        if e >= 0:
            for t in (fc1_w, fc1_s, fc2_w, fc2_s):
                t[LOCAL + h] = t[e % LOCAL]
    if FC1_KBLK:
        fc1_w = _fc1_kblock.to_kblock(fc1_w)
    if FC2_KBLK:
        fc2_w = _fc1_kblock.to_kblock(fc2_w)
    e4 = torch.float8_e4m3fn
    return fc1_w, fc1_s.view(e4), fc2_w, fc2_s.view(e4)


def _layer(layer: int, stacks) -> SimpleNamespace:
    """A layer's router (skewed, as the capture test's), and per-global-expert scales; the
    per-rank operands `_rank_weights` hands the core."""
    g = torch.Generator(device="cuda").manual_seed(_seed(f"wide replicas layer {layer}"))
    router = torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.2
    bias[WIN : WIN + LOCAL] += 0.3
    bias[HOT] = 4.0
    bias[list(EMPTY)] = -4.0
    for a, b in TIED:
        router[b] = router[a]
        bias[b] = bias[a]
    return SimpleNamespace(
        router=router.to(torch.bfloat16).contiguous(),
        router_bias=bias.contiguous(),
        stacks=stacks,
        a1=torch.rand(EXPERTS, generator=g, device="cuda") + 0.5,
        a2=torch.rand(EXPERTS, generator=g, device="cuda") + 0.5,
        ranks={},
    )


def _rank_weights(lw: SimpleNamespace, offset: int) -> SimpleNamespace:
    """`_derive_moe`'s namespace for EP rank `offset // 64`: the shared stacks, the window's
    scales then the replica slots' (an unused slot's are 1)."""
    if offset not in lw.ranks:
        hot = torch.tensor([max(e, 0) for e in HOT_IDS], device="cuda")
        unused = torch.tensor([e < 0 for e in HOT_IDS], device="cuda")
        a1 = torch.cat([lw.a1[offset : offset + LOCAL], torch.where(unused, 1.0, lw.a1[hot])])
        a2 = torch.cat([lw.a2[offset : offset + LOCAL], torch.where(unused, 1.0, lw.a2[hot])])
        fc1_w, fc1_s, fc2_w, fc2_s = lw.stacks
        one = torch.ones(1, dtype=torch.float32, device="cuda")
        lw.ranks[offset] = SimpleNamespace(
            router=lw.router, router_bias=lw.router_bias, fc1_w=fc1_w, fc1_s=fc1_s,
            fc2_w=fc2_w, fc2_s=fc2_s, alpha1=a1.contiguous(), alpha2=a2.contiguous(),
            norm_const=one.clone(), g1=one.clone(),
        )  # fmt: skip
    return lw.ranks[offset]


def _slot(dev) -> SimpleNamespace:
    rows = _modeling._MOE_MAX_T
    counts, barrier = _routing_op.allocate_workspace(dev)
    return SimpleNamespace(
        out=torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=dev),
        x=torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=dev),
        w=torch.zeros((rows, TOP_K), dtype=torch.float32, device=dev),
        sel=torch.empty(rows * TOP_K, dtype=torch.int32, device=dev),
        maps=_routing_op.allocate_maps(rows, dev, replicas=NREP),
        counts=counts,
        barrier=barrier,
        logits=torch.zeros((rows, 1, EXPERTS), dtype=torch.float32, device=dev),
        topk_scratch=_topk_op.allocate_prefill_scratch(dev),
        sfc=_experts_op.allocate_sfc(rows, INTER, dev, NREP),
        pc=_experts_op.allocate_pair_c(rows, INTER, dev, NREP),
        router_wide_out=torch.zeros(
            _router_op.output_shape(_router_op.WIDE_MAX_ROWS, wide=True),
            dtype=torch.float32,
            device=dev,
        ),
    )


def state() -> SimpleNamespace:
    """A core holding the MoE state an MTP3 decode step above 64 rows reads, with the context
    replicas' pair and hot ids, and the wide replica top-k `_build_moe_wide_rep` builds."""
    if "state" in _STATE:
        return _STATE["state"]
    dev = torch.cuda.current_device()
    core = _modeling.GlmMoeDsaTep4ModelingV2.__new__(_modeling.GlmMoeDsaTep4ModelingV2)
    torch.nn.Module.__init__(core)
    core.route_hits, core.route_misses = {}, {}
    core.expert_offset = WIN
    core.routed_scale = ROUTED_SCALE
    core.max_draft_len = 3
    core.fc1_kblk, core.fc2_kblk = FC1_KBLK, FC2_KBLK
    core.ctx_replicas = NREP
    core.wide_dec_replicas = True
    core._rows = SimpleNamespace(
        norm=None, add=None, gather=None, topk=None, swiglu=None, copy=_copy_op.compile(),
        argmax=None,
    )  # fmt: skip
    core._side_stream = torch.cuda.Stream(device=dev)
    core._moe_slot = 0
    core._moe_slots = [_slot(dev) for _ in range(2)]
    rows = _modeling._MOE_MAX_T
    core._moe_xq = torch.zeros((rows, HIDDEN // 2), dtype=torch.uint8, device=dev)
    core._moe_xsf = torch.zeros((rows, HIDDEN // 16), dtype=torch.uint8, device=dev)
    core._moe_rep_maps = None
    hot = torch.tensor(HOT_IDS, dtype=torch.int32, device=dev)
    core._ctx_hot_ids = {i: hot.clone() for i in LAYERS}
    core._dec_hot_ids = {}
    core._pf = None
    k = SimpleNamespace(
        routing=None,
        pair=_experts_op.build_prefill_pair(
            HIDDEN, INTER, LOCAL, dev, fc1_kblk=FC1_KBLK, fc2_kblk=FC2_KBLK
        ),
        pair_rep=_experts_op.build_prefill_pair(
            HIDDEN, INTER, LOCAL + NREP, dev, fc1_kblk=FC1_KBLK, fc2_kblk=FC2_KBLK
        ),
        topk_pre_rep=None,
        router_pre=None,
        topk_pre=_topk_op.build_prefill(dev),
        router_wide=_router_op.build_wide(dev),
        topk_wide={
            r: _topk_op.build_wide(r[1], _router_op.SPLIT_K, dev) for r in _topk_op.WIDE_RANGES
        },
        quant_pre=_quant_op.build_prefill(dev),
        zero_pre=None,
        merge=None,
        experts=None,
        topk={},
        topk_rep={},
        quant={},
        zero={},
        router={},
    )
    core._build_moe_wide_rep(k, dev)
    assert sorted(k.topk_wide_rep) == [(65, 128)], (
        f"wide replicas built for {sorted(k.topk_wide_rep)}"
    )
    core._moe_k = k
    stacks = _stacks()
    st = SimpleNamespace(core=core, layers=[_layer(i, stacks) for i in LAYERS])
    _STATE["state"] = st
    return st


def moe_layer(core, x, partial, m, rows: int, layer: int):
    """`_mlp_and_close`'s MoE half on an MTP3 decode step's exact rows (prequantized off: the
    routed experts quantize): the shared expert's partial copied on the side stream into the
    routed output, FC2 accumulating onto it. Returns `(out, slot)`."""
    exact = core._moe_decode_exact(rows)
    assert exact
    acc = core._moe_begin_layer(rows, exact)
    slot = core._moe_slot
    main = torch.cuda.current_stream()
    side = core._side_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        core._copy_into(acc, partial)
    routed = core._routed(x, m, rows, False, side, True, True, layer, exact)
    return routed, slot


def _snapshot(core, results, rows: int) -> dict:
    snap = {}
    for j, (out, slot) in enumerate(results):
        s = core._moe_slots[slot]
        snap[f"out{j}"] = out.clone()
        snap[f"partials{j}"] = s.router_wide_out[:rows].clone()
        snap[f"w{j}"] = s.w[:rows].clone()
        snap[f"ids{j}"] = s.sel[: rows * TOP_K].clone()
        for k, t in _rep_op.canonical_slot_maps(s.maps, rows * TOP_K, LOCAL + NREP).items():
            snap[f"maps{j}.{k}"] = t
    snap["xq"] = core._moe_xq[:rows].clone()
    snap["xsf"] = core._moe_xsf[:rows].clone()
    return snap


def _reference(core, snap: dict, j: int, partial, m, maps_key: str = "maps") -> torch.Tensor:
    """fp32 grouped GEMMs on the maps, weights and NVFP4 input the step wrote, onto the
    shared partial, on the replica stacks."""
    maps = {k: snap[f"{maps_key}{j}.{k}"] for k in _routing_op.MAP_KEYS}
    return _experts_op.cute_dsl_moe_nvfp4_experts.reference(
        snap["xq"], snap["xsf"], maps, snap[f"w{j}"], m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s,
        m.norm_const, m.alpha1, m.alpha2, partial, None, None, None, core._moe_k.pair_rep,
    )  # fmt: skip


def _inputs(rows: int, tag: str):
    g = torch.Generator(device="cuda").manual_seed(_seed(tag))
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    partials = [
        (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        for _ in range(4)
    ]
    return x, partials


@pytest.mark.parametrize("rows", ODD_ROWS)
def test_four_ranks_sum_to_the_whole_layer(rows: int) -> None:
    """The four ranks' outputs on one layer (each onto its own shared partial), with the wide
    replicas and with the home mapping: each rank within the experts test's band (32 ulp, 2
    RMS) of the fp32 reference of its own maps; the four-rank sums -- what the all-reduce adds
    -- each within that band of the fp32 reference of the whole layer; selection and weights
    identical in both. The draw is not degenerate: the hot expert takes every row and is
    computed on one rank only, away from its home; an inactive hot expert and the unused slot
    on none."""
    st = state()
    core = st.core
    lw = st.layers[0]
    layer = LAYERS[0]
    x, partials = _inputs(rows, f"sum {rows}")
    hot = core._ctx_hot_ids[layer]
    sums, refs, ids_by_arm, hot_tiles = {}, [], {}, {}
    try:
        for arm in ("rep", "home"):
            core._ctx_hot_ids = {layer: hot} if arm == "rep" else {}
            hits = dict(core.route_hits)
            total = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
            for r, off in enumerate((0, 64, 128, 192)):
                core.expert_offset = off
                m = _rank_weights(lw, off)
                xin = x.clone()
                with validating(_wrep_op.cute_dsl_moe_topk_routing_wide_rep):
                    out, slot = moe_layer(core, xin, partials[r], m, rows, layer)
                torch.cuda.synchronize()
                snap = _snapshot(core, [(out, slot)], rows)
                ref = _reference(core, snap, 0, partials[r], m)
                assert_within_ulp(out, ref, element_ulp=32.0, rms_ulp=2.0)
                total += out.float()
                if arm == "home":
                    # This rank's routed part alone (onto zeros), for the whole-layer sum.
                    zero = torch.zeros_like(partials[r])
                    refs.append(_reference(core, snap, 0, zero, m).float())
                ids_by_arm.setdefault(arm, []).append((snap["ids0"].clone(), snap["w0"].clone()))
                tiles = snap["maps0.tile_idx_to_expert_idx"].tolist()
                hot_tiles.setdefault(arm, []).append([tiles.count(LOCAL + h) for h in range(NREP)])
                if arm == "rep":
                    assert (HOT - WIN) not in tiles or off != WIN, "the hot expert ran at home"
            took = core.route_hits.get("moe_wide_replica", 0) - hits.get("moe_wide_replica", 0)
            assert took == (4 if arm == "rep" else 0), f"{arm}: {took} replica top-k calls"
            sums[arm] = total
    finally:
        core._ctx_hot_ids = {i: hot for i in LAYERS}
        core.expert_offset = WIN
    for (ia, wa), (ib, wb) in zip(ids_by_arm["rep"], ids_by_arm["home"]):
        assert torch.equal(ia, ib) and torch.equal(wa, wb), "selection differs between arms"
    # The whole layer: every selection once (the home maps of the four ranks), plus the four
    # shared partials.
    whole = sum(refs) + sum(p.float() for p in partials)
    whole = whole.to(torch.bfloat16)
    for arm in ("rep", "home"):
        assert_within_ulp(sums[arm].to(torch.bfloat16), whole, element_ulp=32.0, rms_ulp=2.0)
    # Not degenerate: the hot expert's slot runs on exactly one rank, with all its rows; the
    # empty expert's slot and the unused one on none.
    per_slot = [sum(t[h] for t in hot_tiles["rep"]) for h in range(NREP)]
    assert per_slot[0] == (rows + TILE - 1) // TILE, f"hot expert tiles over ranks: {per_slot}"
    assert sum(1 for t in hot_tiles["rep"] if t[0]) == 1, "the hot expert ran on two ranks"
    assert hot_tiles["rep"][WIN // 64][0] == 0, "the hot expert stayed on its busiest home rank"
    assert per_slot[1] == 0 and per_slot[7] == 0, f"inactive / unused slot ran: {per_slot}"
    assert all(sum(t) == 0 for t in hot_tiles["home"]), "the home mapping used a replica slot"
    d = (sums["rep"] - sums["home"]).float()
    rel = float(d.pow(2).mean().sqrt() / sums["home"].float().pow(2).mean().sqrt())
    assert rel < 2.0**-7, f"replicas vs home: relative RMS {rel:.2e}"


@pytest.mark.parametrize("rows", (65, 96, 128))
def test_capture_replays_match_eager(rows: int) -> None:
    """Two layers with replicas on EP rank 1 captured once, then three times: fresh inputs,
    replay, snapshot, the same calls eagerly, snapshot, compare -- everything upstream of FC2
    bit for bit, FC2's output within the replay band -- and the eager output within the band
    of the fp32 reference. Every layer takes the wide replica top-k, none the wide top-k."""
    st = state()
    core = st.core
    ms = [_rank_weights(lw, WIN) for lw in st.layers]
    bf = {"dtype": torch.bfloat16, "device": "cuda"}
    inputs = [(torch.empty(rows, HIDDEN, **bf), torch.empty(rows, HIDDEN, **bf)) for _ in LAYERS]

    def fill(tag):
        g = torch.Generator(device="cuda").manual_seed(_seed(tag))
        for x, p in inputs:
            x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
            p.copy_(
                (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
            )

    def step():
        return [moe_layer(core, x, p, m, rows, i) for (x, p), m, i in zip(inputs, ms, LAYERS)]

    fill(f"warm {rows}")
    hits = dict(core.route_hits)
    step()
    torch.cuda.synchronize()
    assert core.route_hits.get("moe_wide_replica", 0) - hits.get("moe_wide_replica", 0) == 2
    assert core.route_hits.get("moe_topk_wide", 0) == hits.get("moe_topk_wide", 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = step()
    for rep in range(3):
        fill(f"replay {rows} {rep}")
        graph.replay()
        torch.cuda.synchronize()
        got = _snapshot(core, results, rows)
        eager = step()
        torch.cuda.synchronize()
        want = _snapshot(core, eager, rows)
        for k in want:
            if k.startswith("out"):
                assert_within_ulp(got[k], want[k], REPLAY_ELEMENT_ULP, REPLAY_RMS_ULP)
            else:
                assert torch.equal(got[k], want[k]), f"{k}: replay differs from eager"
        j = len(LAYERS) - 1
        ref = _reference(core, want, j, inputs[-1][1], ms[-1])
        assert_within_ulp(want[f"out{j}"], ref, element_ulp=32.0, rms_ulp=2.0)


@pytest.mark.parametrize("rows", WIDE_ROWS)
def test_wider_steps_keep_the_wide_top_k(rows: int) -> None:
    """Past 128 rows a trunk layer with hot ids takes the wide top-k and the 64-deep pair: the
    same bytes upstream of FC2 as with replicas switched off, the output within the replay
    band (FC2's bf16 atomics add in a run-dependent order)."""
    st = state()
    core = st.core
    layer, m = LAYERS[0], _rank_weights(st.layers[0], WIN)
    x, partials = _inputs(rows, f"wider {rows}")
    hot = core._ctx_hot_ids[layer]
    snaps = {}
    try:
        for arm in ("on", "off"):
            core._ctx_hot_ids = {layer: hot} if arm == "on" else {}
            hits = dict(core.route_hits)
            out, slot = moe_layer(core, x.clone(), partials[0], m, rows, layer)
            torch.cuda.synchronize()
            assert core.route_hits.get("moe_wide_replica", 0) == hits.get("moe_wide_replica", 0)
            assert core.route_hits.get("moe_topk_wide", 0) == hits.get("moe_topk_wide", 0) + 1
            snaps[arm] = _snapshot(core, [(out, slot)], rows)
    finally:
        core._ctx_hot_ids = {i: hot for i in LAYERS}
    for k in snaps["off"]:
        if k.startswith("out"):
            assert_within_ulp(snaps["on"][k], snaps["off"][k], REPLAY_ELEMENT_ULP, REPLAY_RMS_ULP)
        else:
            assert torch.equal(snaps["on"][k], snaps["off"][k]), f"{k} differs"
    assert all(t < LOCAL for t in snaps["on"]["maps0.tile_idx_to_expert_idx"].tolist())


def test_only_trunk_layers_with_hot_ids_on_wide_steps_take_replicas() -> None:
    """`_moe_topk_wide_rep` answers only for a trunk layer with hot ids (not the MTP layer,
    which has none) and a built handle; `_build_moe_wide_rep` builds nothing without the wide
    top-k (MTP1, no speculation), the context replicas' pair, `ctx_replicas` or the switch.
    MTP1 and no speculation never run a wide decode step (`_moe_decode_exact`)."""
    core = state().core
    k = core._moe_k
    assert core._moe_topk_wide_rep(128, LAYERS[0]) is not None
    assert core._moe_topk_wide_rep(65, LAYERS[1]) is not None
    assert core._moe_topk_wide_rep(128, 78) is None, "the MTP layer has no hot ids"
    for rows in WIDE_ROWS:
        assert core._moe_topk_wide_rep(rows, LAYERS[0]) is None, f"replicas at {rows} rows"
    dev = torch.cuda.current_device()
    base = dict(
        topk_wide=k.topk_wide, pair_rep=k.pair_rep, ctx=core.ctx_replicas, sw=True,
        hot=core._ctx_hot_ids,
    )  # fmt: skip
    cases = (
        dict(base, topk_wide={}),
        dict(base, pair_rep=None),
        dict(base, ctx=0),
        dict(base, sw=False),
        dict(base, hot={}),
    )
    try:
        for c in cases:
            kk = SimpleNamespace(topk_wide=c["topk_wide"], pair_rep=c["pair_rep"])
            core.ctx_replicas, core.wide_dec_replicas = c["ctx"], c["sw"]
            core._ctx_hot_ids = c["hot"]
            core._build_moe_wide_rep(kk, dev)
            assert kk.topk_wide_rep == {}, c
        for draft_len in (0, 1):
            core.max_draft_len = draft_len
            assert not any(core._moe_decode_exact(r) for r in range(1, 257))
    finally:
        core.ctx_replicas, core.wide_dec_replicas = NREP, True
        core._ctx_hot_ids = base["hot"]
        core.max_draft_len = 3
