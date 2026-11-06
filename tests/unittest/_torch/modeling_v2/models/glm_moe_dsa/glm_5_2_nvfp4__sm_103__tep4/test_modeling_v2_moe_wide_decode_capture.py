# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the trunk's routed MoE layer at a decode row count above 64, captured in a CUDA
graph and replayed.

MTP3 / MTP5 verify steps and draft step 0 carry 32 x (D + 1) = 128 / 192 rows (fewer at a
smaller padded batch), and the engine runs them inside its captured decode graph. Above 64
rows the routed experts take the context kernels -- prefill router, prefill top-k, the
routing kernel on padded rows, the segmented quantize, the byte clear, the merge and the
prefill grouped GEMM pair -- which a context step only ever runs eagerly. This drives the
target's own methods (`_moe_begin_layer`, `_routed` and the merge of `_mlp_and_close`) on a
core holding only the MoE state, captures two consecutive layers (the two operand slots),
replays them on fresh inputs, and compares with the same calls run eagerly.

Two paths. `pow2`: a context step's, and every decode step's above 64 rows before MTP3 /
MTP5 decode ran exact: the rows padded to a power of two below 513, the top-k's maps
discarded for the routing kernel's, the output cleared and the shared expert merged after.
`exact`: the decode target's at MTP3 / MTP5 (`_moe_decode_exact`): the row count unpadded,
the wide decode top-k's maps used as written, the shared expert accumulated by FC2;
`exact_pre` the same on the prefill top-k (the wide one not built). `exact_prequant`: `exact`
with the routed experts' NVFP4 input written ahead of the layer, as the pre-MoE collective
writes it on those steps (`_moe_prequant`): the routed experts then neither stage nor
quantize. One GPU has no collective, so the routed experts' own prefill quantize stands in
for it; the 4-GPU comm matrix proves the collective's bytes equal that quantize's bit for bit.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL MoE kernels are built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (  # noqa: E402, E501
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4 import (  # noqa: E402, E501
    modeling as _modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.moe import (  # noqa: E402, E501
    moe_fc1_kblock as _fc1_kblock,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_copy as _copy_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_merge as _merge_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_nvfp4_experts as _experts_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_router as _router_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_routing as _routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_topk_routing as _topk_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_zero as _zero_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization import (  # noqa: E402, E501
    cute_dsl_moe_quantize_nvfp4 as _quant_op,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, INTER, EXPERTS, LOCAL = 6144, 2048, 256, 64
TOP_K = _routing_op.TOP_K
TILE = _routing_op.TILE
#: The rank under test: EP rank 1's window.
OFFSET = 64
#: Decode row counts above 64: MTP3's 128 and MTP5's 192 verify rows, and the counts on
#: either side of the routing schedule's and the top-k's boundaries.
ROWS = (65, 127, 128, 129, 191, 192, 200)
#: Two trunk layer slots, so the capture alternates both operand slots.
LAYERS = (3, 4)
#: Hot expert (every token selects it: > 64 rows from 65 rows up, two tiles past 128), the
#: never-selected ones, and the 16 tied pairs (identical router rows and bias, so their
#: scores are bitwise equal and selection breaks the tie toward the lower id).
HOT = OFFSET + 1
EMPTY = tuple(range(OFFSET + 10, OFFSET + 14))
TIED = tuple((OFFSET + 20 + 2 * p, OFFSET + 21 + 2 * p) for p in range(16))
ROUTED_SCALE = 2.5
#: The routed experts' FC1 / FC2 weight layouts, read as the target reads them
#: (`moe_fc1_kblock`): the stack, the compiled pair and the reference all follow them.
FC1_KBLK = _fc1_kblock.enabled()
FC2_KBLK = _fc1_kblock.fc2_enabled()

_STATE: dict = {}


def _seed(tag: str) -> int:
    return zlib.crc32(tag.encode()) % 2**31


def _sf_atoms(s: torch.Tensor) -> torch.Tensor:
    """Row-major `[N, K/16]` scale bytes -> `[N/128, K/64, 512]` atoms, the weight loader's
    layout (restated so the test does not import the loader)."""
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


def _experts():
    """One NVFP4 expert stack for both layers, scaled so each GEMM's sums are O(1)."""
    g = torch.Generator(device="cuda").manual_seed(_seed("wide decode experts"))
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    fc1_w = torch.empty(LOCAL, 2 * INTER, HIDDEN // 2, **u8)
    fc1_s = torch.empty(LOCAL, 2 * INTER // 128, HIDDEN // 64, 512, **u8)
    fc2_w = torch.empty(LOCAL, HIDDEN, INTER // 2, **u8)
    fc2_s = torch.empty(LOCAL, HIDDEN // 128, INTER // 64, 512, **u8)
    for e in range(LOCAL):
        p, s = _nvfp4(torch.randn(2 * INTER, HIDDEN, generator=g, device="cuda") * 0.02)
        fc1_w[e], fc1_s[e] = p, _sf_atoms(s)
        p, s = _nvfp4(torch.randn(HIDDEN, INTER, generator=g, device="cuda") * 0.03)
        fc2_w[e], fc2_s[e] = p, _sf_atoms(s)
    e4 = torch.float8_e4m3fn
    if FC1_KBLK:
        fc1_w = _fc1_kblock.to_kblock(fc1_w)
    if FC2_KBLK:
        fc2_w = _fc1_kblock.to_kblock(fc2_w)
    return fc1_w, fc1_s.view(e4), fc2_w, fc2_s.view(e4)


def _layer_weights(layer: int, stacks) -> SimpleNamespace:
    """A layer's operands as `_derive_moe` builds them (g1 and norm_const 1, as on this
    checkpoint). The router is skewed so the reference is not degenerate: a hot expert
    every token takes, experts no token takes, tied pairs, and the window favoured so
    about half of each row's winners are this rank's."""
    g = torch.Generator(device="cuda").manual_seed(_seed(f"wide decode layer {layer}"))
    router = torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.2
    bias[OFFSET : OFFSET + LOCAL] += 0.3
    bias[HOT] = 4.0
    bias[list(EMPTY)] = -4.0
    for a, b in TIED:
        router[b] = router[a]
        bias[b] = bias[a]
    fc1_w, fc1_s, fc2_w, fc2_s = stacks
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    return SimpleNamespace(
        router=router.to(torch.bfloat16).contiguous(),
        router_bias=bias.contiguous(),
        fc1_w=fc1_w,
        fc1_s=fc1_s,
        fc2_w=fc2_w,
        fc2_s=fc2_s,
        alpha1=torch.rand(LOCAL, generator=g, device="cuda") + 0.5,
        alpha2=torch.rand(LOCAL, generator=g, device="cuda") + 0.5,
        norm_const=one.clone(),
        g1=one.clone(),
    )


def _slot(dev) -> SimpleNamespace:
    """The fields of `_build_moe`'s trunk slot that a row count above 64 reads."""
    rows = _modeling._MOE_MAX_T
    counts, barrier = _routing_op.allocate_workspace(dev)
    ns = SimpleNamespace(
        out=torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=dev),
        x=torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=dev),
        w=torch.zeros((rows, TOP_K), dtype=torch.float32, device=dev),
        sel=torch.empty(rows * TOP_K, dtype=torch.int32, device=dev),
        maps=_routing_op.allocate_maps(rows, dev),
        counts=counts,
        barrier=barrier,
        logits=torch.zeros((rows, 1, EXPERTS), dtype=torch.float32, device=dev),
        topk_scratch=_topk_op.allocate_prefill_scratch(dev),
        sfc=_experts_op.allocate_sfc(rows, INTER, dev),
        pc=_experts_op.allocate_pair_c(rows, INTER, dev),
        router_wide_out=torch.zeros(
            _router_op.output_shape(_router_op.WIDE_MAX_ROWS, wide=True),
            dtype=torch.float32,
            device=dev,
        ),
    )
    return ns


def state() -> SimpleNamespace:
    """A target core holding only what the routed experts read above 64 rows (built once a
    session): the context kernels `_build_moe` builds, the two slots, and two layers."""
    if "state" in _STATE:
        return _STATE["state"]
    dev = torch.cuda.current_device()
    core = _modeling.GlmMoeDsaTep4ModelingV2.__new__(_modeling.GlmMoeDsaTep4ModelingV2)
    torch.nn.Module.__init__(core)
    core.route_hits, core.route_misses = {}, {}
    core.expert_offset = OFFSET
    core.routed_scale = ROUTED_SCALE
    # MTP3: decode steps above 64 rows run unpadded (`_moe_decode_exact`).
    core.max_draft_len = 3
    core.fc1_kblk = FC1_KBLK
    core.fc2_kblk = FC2_KBLK
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
    core._ctx_hot_ids = {}
    core._dec_hot_ids = {}
    core._pf = None
    core._moe_k = SimpleNamespace(
        # The padded counts ROWS reach: 128 and 256 (the routing kernel's other buckets
        # are minutes of compile and unreachable here).
        routing=_routing_op.build((128, 256), dev),
        pair=_experts_op.build_prefill_pair(
            HIDDEN, INTER, LOCAL, dev, fc1_kblk=FC1_KBLK, fc2_kblk=FC2_KBLK
        ),
        pair_rep=None,
        topk_pre_rep=None,
        router_pre=_router_op.build_prefill(dev),
        topk_pre=_topk_op.build_prefill(dev),
        # As `_build_moe` builds them at MTP3 / MTP5 (no L2 prefetch here: no counts); the
        # one-plane top-k is what it builds when the wide router does not build.
        router_wide=_router_op.build_wide(dev),
        topk_wide={
            r: _topk_op.build_wide(r[1], _router_op.SPLIT_K, dev) for r in _topk_op.WIDE_RANGES
        },
        topk_wide1={r: _topk_op.build_wide(r[1], 1, dev) for r in _topk_op.WIDE_RANGES},
        quant_pre=_quant_op.build_prefill(dev),
        zero_pre=_zero_op.build_prefill(dev),
        merge=_merge_op.build(dev),
        experts=None,
        topk={},
        topk_rep={},
        quant={},
        zero={},
        router={},
    )
    stacks = _experts()
    st = SimpleNamespace(core=core, layers=[_layer_weights(i, stacks) for i in LAYERS])
    _STATE["state"] = st
    return st


def prequantize(core, x, m, rows: int) -> None:
    """Stand in for the pre-MoE collective's NVFP4 epilogue on one GPU: the routed experts'
    prefill quantize of `x` into `xq` / `xsf`, the bytes the collective writes there."""
    k = core._moe_k
    _quant_op.cute_dsl_moe_quantize_nvfp4(
        x, m.g1, core._moe_xq[:rows], core._moe_xsf[:rows], k.quant_pre
    )


def moe_layer(
    core, x, partial, m, rows: int, layer: int, decode: bool = False, prequant: bool = False
):
    """`_mlp_and_close`'s MoE half from the pre-MoE collective's output `x` on, at a row count
    above 64 (no decode L2 prefetch), called by the decode target (`decode`) or the prefill
    target. `prequant`: the collective also wrote the routed experts' NVFP4 input (an exact
    decode step, `_moe_prequant`); otherwise it wrote none. The shared expert's result is
    `partial`, copied on the side stream into the buffer its GEMM writes: the routed output
    when `_moe_begin_layer` hands one out, else a fresh tensor that is merged after.
    Returns `(mlp_out, slot)`."""
    exact = decode and core._moe_decode_exact(rows)
    began = rows <= _modeling._MOE_MAX_T
    acc = core._moe_begin_layer(rows, exact) if began else None
    slot = core._moe_slot
    main = torch.cuda.current_stream()
    side = core._side_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        shared = acc if acc is not None else torch.empty_like(partial)
        core._copy_into(shared, partial)
    routed = core._routed(
        x, m, rows, prequant, side if acc is not None else None, began, acc is not None, layer,
        exact,
    )  # fmt: skip
    if acc is not None:
        return routed, slot
    main.wait_stream(side)
    merge = core._moe_k.merge
    assert routed.shape == shared.shape and routed.is_contiguous() and shared.is_contiguous()
    return _merge_op.cute_dsl_moe_merge(routed, shared, merge), slot


def fill(inputs, rows: int, tag: str) -> None:
    """Fresh activations and shared partials into the static input buffers."""
    g = torch.Generator(device="cuda").manual_seed(_seed(tag))
    for x, p in inputs:
        x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
        p.copy_((torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16))


def _snapshot(core, results, rows: int, exact: bool) -> dict:
    """Everything the step wrote that the next step overwrites: per layer its output, router
    logits, top-k weights and ids (as FC2 read them: padded rows included), and maps in
    canonical form; the NVFP4 input of the last layer (one buffer for both)."""
    padded = core._moe_padded_rows(rows, exact)
    snap = {}
    for j, (out, slot) in enumerate(results):
        s = core._moe_slots[slot]
        snap[f"out{j}"] = out.clone()
        snap[f"logits{j}"] = s.logits[:rows].clone()
        snap[f"partials{j}"] = s.router_wide_out[:rows].clone()
        snap[f"w{j}"] = s.w[:padded].clone()
        snap[f"ids{j}"] = s.sel[: padded * TOP_K].clone()
        for k, t in _routing_op.canonical_maps(s.maps, padded * TOP_K).items():
            snap[f"maps{j}.{k}"] = t
    snap["xq"] = core._moe_xq[:padded].clone()
    snap["xsf"] = core._moe_xsf[:padded].clone()
    return snap


#: Replay against eager on the FC2 output. FC2 adds each token's (up to eight) expert
#: contributions onto the shared partial with bf16 atomics in an order that varies run to
#: run, so two eager runs already differ there; everything upstream of it is compared
#: bitwise. Each run is within ~4.5 ulp of the row scale of the fp32 sum at its worst
#: element and ~0.8 ulp relative RMS (the experts test's derivation), so two runs within 9
#: and ~1.1; the band is 16 and 2. A token missing an expert is ~35 % relative RMS.
REPLAY_ELEMENT_ULP, REPLAY_RMS_ULP = 16.0, 2.0


def _compare(got: dict, want: dict) -> None:
    for k in want:
        if k.startswith("out"):
            assert_within_ulp(got[k], want[k], REPLAY_ELEMENT_ULP, REPLAY_RMS_ULP)
        else:
            assert torch.equal(got[k], want[k]), f"{k}: replay differs from eager"


def _check_reference(core, snap: dict, partial, m, rows: int, exact: bool) -> None:
    """The last layer's eager output against the fp32 reference of the grouped GEMMs on the
    maps, weights and NVFP4 input the step wrote, onto the shared partial (accumulated by
    FC2, or merged after a padded launch). The experts test's band: 32 ulp, 2 RMS."""
    padded = core._moe_padded_rows(rows, exact)
    j = len(LAYERS) - 1
    maps = {k: snap[f"maps{j}.{k}"] for k in _routing_op.MAP_KEYS}
    xq, xsf, w = snap["xq"], snap["xsf"], snap[f"w{j}"]
    if padded == rows:
        initial = partial
    else:
        initial = torch.zeros(padded, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ref = _experts_op.cute_dsl_moe_nvfp4_experts.reference(
        xq, xsf, maps, w, m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s, m.norm_const, m.alpha1,
        m.alpha2, initial, None, None, None, core._moe_k.pair,
    )  # fmt: skip
    if padded != rows:
        ref = (ref[:rows].float() + partial.float()).to(torch.bfloat16)
    assert_within_ulp(snap[f"out{j}"], ref, element_ulp=32.0, rms_ulp=2.0)


def _assert_not_degenerate(snap: dict, rows: int) -> None:
    """The draw exercises what the maps can get wrong: an expert past 64 rows (two tiles past
    128), experts with no rows, and ties broken at the selection edge (one of a tied pair
    taken without the other)."""
    straddles = 0
    for j in range(len(LAYERS)):
        expert = snap[f"maps{j}.tile_idx_to_expert_idx"].long()
        limit = snap[f"maps{j}.tile_idx_to_mn_limit"].long()
        start = torch.arange(expert.numel(), device=expert.device) * TILE
        counts = torch.zeros(LOCAL, dtype=torch.long, device=expert.device)
        counts.index_add_(0, expert, limit - start)
        assert int(counts[HOT - OFFSET]) == rows, "the hot expert does not take every row"
        assert int((expert == HOT - OFFSET).sum()) == (rows + TILE - 1) // TILE
        assert all(int(counts[e - OFFSET]) == 0 for e in EMPTY), "an empty expert got rows"
        ids = snap[f"ids{j}"][: rows * TOP_K].view(rows, TOP_K)
        for a, b in TIED:
            ha, hb = (ids == a).any(1), (ids == b).any(1)
            assert not bool((hb & ~ha).any()), f"tie {a}/{b} broken toward the higher id"
            straddles += int((ha & ~hb).sum())
    assert straddles > 0, "no tied pair straddled the selection edge in this draw"


def _refuse(*args, **kwargs):
    raise AssertionError("an unpadded decode step launched the routing kernel or a clear")


#: `(decode, wide front end built, prequantized)` per path: `full` the wide router and the wide
#: top-k on its partials, `topk1` the wide top-k on the prefill router's logits (the wide
#: router failed to build), `none` the prefill router and top-k (neither built).
PATHS = {
    "pow2": (False, "full", False),
    "exact_pre": (True, "none", False),
    "exact_topk1": (True, "topk1", False),
    "exact": (True, "full", False),
    # The step's NVFP4 input written ahead of each layer (`_moe_prequant`).
    "exact_prequant": (True, "full", True),
}


def _refuse_quant(*args, **kwargs):
    raise AssertionError("the routed experts quantized a step the collective prequantized")


def _refuse_stage(core, monkeypatch) -> None:
    """Make a copy into a slot's staged activation (`s.x`) raise: a prequantized step has
    nothing to stage. Every other copy goes through."""
    staged = {s.x.data_ptr() for s in core._moe_slots}
    copy = core._copy_into

    def guarded(dst, src):
        assert dst.data_ptr() not in staged, "a prequantized step staged its activation"
        return copy(dst, src)

    monkeypatch.setattr(core, "_copy_into", guarded)


def layer_inputs(rows: int) -> list:
    bf = {"dtype": torch.bfloat16, "device": "cuda"}
    return [(torch.empty(rows, HIDDEN, **bf), torch.empty(rows, HIDDEN, **bf)) for _ in LAYERS]


def run_step(st, inputs, rows: int, decode: bool, prequant: bool) -> list:
    """The two layers of one step; with `prequant` each layer's NVFP4 input is written first,
    as the collective that opens it would."""
    results = []
    for (x, p), m, i in zip(inputs, st.layers, LAYERS):
        if prequant:
            prequantize(st.core, x, m, rows)
        results.append(moe_layer(st.core, x, p, m, rows, i, decode, prequant))
    return results


@pytest.mark.parametrize("path", tuple(PATHS))
@pytest.mark.parametrize("rows", ROWS)
def test_capture_replays_match_eager(rows: int, path: str, monkeypatch) -> None:
    """Capture two layers, then three times: fresh inputs, replay, snapshot, the same calls
    eagerly, snapshot, compare. The first eager run goes through every op's guard. On the
    exact paths neither the routing kernel nor a clear may run (the merge cannot: the shared
    expert writes the routed output), and the front end is the wide one as far as built. On
    the prequantized path the routed experts neither stage nor quantize."""
    st = state()
    core = st.core
    decode, built, prequant = PATHS[path]
    exact = decode and core._moe_decode_exact(rows)
    assert exact == decode, "every row count here is a wide decode width at MTP3"
    # `exact` without the prequant is the target's path for exact rows past
    # `_MOE_PREQUANT_WIDE_MAX` (257..512), and what MTP3 / MTP5 ran before it widened.
    assert not prequant or core._moe_prequant(rows, exact), "the gate refuses this path"
    if exact:
        monkeypatch.setattr(_modeling, "cute_dsl_moe_routing", _refuse)
        monkeypatch.setattr(_modeling, "cute_dsl_moe_zero", _refuse)
    if prequant:
        monkeypatch.setattr(_modeling, "cute_dsl_moe_quantize_nvfp4", _refuse_quant)
        _refuse_stage(core, monkeypatch)
    if built != "full":
        monkeypatch.setattr(core._moe_k, "router_wide", None)
        topk = core._moe_k.topk_wide1 if built == "topk1" else {}
        monkeypatch.setattr(core._moe_k, "topk_wide", topk)
    hits = dict(core.route_hits)
    inputs = layer_inputs(rows)

    def step():
        return run_step(st, inputs, rows, decode, prequant)

    fill(inputs, rows, f"warm {rows}")
    with validating(
        _router_op.cute_dsl_moe_router,
        _topk_op.cute_dsl_moe_topk_routing,
        _routing_op.cute_dsl_moe_routing,
        _quant_op.cute_dsl_moe_quantize_nvfp4,
        _zero_op.cute_dsl_moe_zero,
        _merge_op.cute_dsl_moe_merge,
        _experts_op.cute_dsl_moe_nvfp4_experts,
        _copy_op.cute_dsl_copy,
    ):
        step()
    torch.cuda.synchronize()

    def took(route: str) -> int:
        return core.route_hits.get(route, 0) - hits.get(route, 0)

    assert took("moe_topk_wide") == (len(LAYERS) if exact and built != "none" else 0)
    assert took("moe_router_wide") == (len(LAYERS) if exact and built == "full" else 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = step()
    for rep in range(3):
        fill(inputs, rows, f"replay {rows} {rep}")
        graph.replay()
        torch.cuda.synchronize()
        got = _snapshot(core, results, rows, exact)
        eager = step()
        torch.cuda.synchronize()
        want = _snapshot(core, eager, rows, exact)
        assert [s for _, s in eager] == [s for _, s in results], "slot parity drifted"
        _compare(got, want)
        _assert_not_degenerate(want, rows)
        _check_reference(core, want, inputs[-1][1], st.layers[-1], rows, exact)


@pytest.mark.parametrize("rows", ROWS)
def test_prequant_matches_the_quantizing_path(rows: int, monkeypatch) -> None:
    """The same two layers on the same inputs, once as an MTP3 / MTP5 decode step ran before
    its collective prequantized (the routed experts stage the activation and quantize it),
    once prequantized (they do neither): every byte upstream of FC2 -- router partials, top-k
    weights and ids, the canonical maps, the NVFP4 input FC1 reads -- equal bit for bit, and
    FC2's output within the replay band (its bf16 atomics add in a run-dependent order)."""
    st = state()
    core = st.core
    inputs = layer_inputs(rows)
    fill(inputs, rows, f"prequant {rows}")
    want = _snapshot(core, run_step(st, inputs, rows, True, False), rows, True)
    torch.cuda.synchronize()
    monkeypatch.setattr(_modeling, "cute_dsl_moe_quantize_nvfp4", _refuse_quant)
    _refuse_stage(core, monkeypatch)
    got = _snapshot(core, run_step(st, inputs, rows, True, True), rows, True)
    torch.cuda.synchronize()
    _compare(got, want)
    _assert_not_degenerate(want, rows)


def test_prequant_only_on_decode_widths_and_wide_decode_steps() -> None:
    """The collective writes the routed experts' NVFP4 input at 1..64 rows in every
    configuration, as before; above 64 only on an MTP3 / MTP5 decode step's exact rows up to
    256 (the call `_mlp_and_close` makes). MTP1, no speculation and every context step keep
    the plain collective above 64 rows."""
    core = state().core
    try:
        for draft_len in (0, 1, 3, 5):
            core.max_draft_len = draft_len
            for decode in (False, True):
                got = [
                    r
                    for r in range(1, _modeling._MOE_MAX_T + 1)
                    if core._moe_prequant(r, decode and core._moe_decode_exact(r))
                ]
                wide = list(range(65, 257)) if decode and draft_len > 1 else []
                assert got == list(range(1, 65)) + wide, (draft_len, decode)
    finally:
        core.max_draft_len = 3


def test_reduce_norm_quant_writes_the_routed_input_where_the_gate_says(monkeypatch) -> None:
    """`_reduce_norm_quant` calls the quant entry exactly where `_moe_prequant` admits the row
    count, into the first `rows` rows of the routed experts' `xq` / `xsf` (the buffers FC1
    reads), and the plain entry everywhere else. The entries are recorded, not run: one
    GPU has no collective."""
    core = state().core
    calls = []

    def quant(partial, residual, weight, g1, out, xq, xsf, workspace):
        calls.append(("quant", xq.data_ptr(), tuple(xq.shape), xsf.data_ptr(), tuple(xsf.shape)))
        return out

    def plain(partial, residual, weight, out, workspace, x8=None):
        calls.append(("plain",))
        return out

    monkeypatch.setattr(_modeling, "cute_dsl_allreduce_add_rmsnorm_quant_nvfp4", quant)
    monkeypatch.setattr(_modeling, "cute_dsl_allreduce_add_rmsnorm", plain)
    lamport = SimpleNamespace(max_rows=_modeling._MOE_MAX_T, fused_twoshot_x8=None)
    monkeypatch.setattr(core, "_lamport", lamport, raising=False)
    bf = {"dtype": torch.bfloat16, "device": "cuda"}
    weight = torch.ones(HIDDEN, **bf)
    g1 = torch.ones(1, dtype=torch.float32, device="cuda")
    cases = (
        (1, False, True), (64, False, True), (64, True, True), (65, False, False),
        (65, True, True), (128, True, True), (192, True, True), (256, True, True),
        (257, True, False), (192, False, False),
    )  # fmt: skip
    for rows, exact, want in cases:
        calls.clear()
        partial, residual = torch.zeros(rows, HIDDEN, **bf), torch.zeros(rows, HIDDEN, **bf)
        _, prequantized = core._reduce_norm_quant(partial, residual, weight, g1, rows, exact)
        assert prequantized == want, (rows, exact)
        if not want:
            assert calls == [("plain",)], (rows, exact, calls)
            continue
        assert calls == [
            (
                "quant", core._moe_xq.data_ptr(), (rows, HIDDEN // 2), core._moe_xsf.data_ptr(),
                (rows, HIDDEN // 16),
            )
        ], (rows, exact, calls)  # fmt: skip


def test_only_wide_decode_steps_run_exact() -> None:
    """MTP1 and no speculation never run a decode row count unpadded (their decode steps stay
    within 64 rows, and the gate refuses anyway); MTP3 / MTP5 do from 65 up to 512 rows,
    from where every count is exact already."""
    core = state().core
    try:
        for draft_len in (0, 1):
            core.max_draft_len = draft_len
            assert not any(core._moe_decode_exact(r) for r in range(1, 1025))
        for draft_len in (3, 5):
            core.max_draft_len = draft_len
            exact = [r for r in range(1, 1025) if core._moe_decode_exact(r)]
            assert exact == list(range(65, 513))
            assert all(core._moe_padded_rows(r, True) == r for r in exact)
            assert core._moe_padded_rows(192) == 256, "a context step still pads below 513"
    finally:
        core.max_draft_len = 3


@pytest.mark.parametrize("decode", (False, True), ids=("pow2", "exact"))
@pytest.mark.parametrize("rows", ROWS)
def test_each_selection_is_routed_on_exactly_one_rank(rows: int, decode: bool) -> None:
    """The four EP ranks' maps for one step cover every `(token, k)` selection exactly once,
    padding rows none: the router and top-k on each rank's window, and the maps FC1 reads
    (the top-k's own at an unpadded count, else the routing kernel's on the padded ids)."""
    st = state()
    core = st.core
    m = st.layers[0]
    g = torch.Generator(device="cuda").manual_seed(_seed(f"coverage {rows}"))
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    exact = decode and core._moe_decode_exact(rows)
    padded = core._moe_padded_rows(rows, exact)
    seen = torch.zeros(padded * TOP_K, dtype=torch.long, device="cuda")
    try:
        for offset in (0, 64, 128, 192):
            core.expert_offset = offset
            slot = core._moe_slot
            logits = core._moe_router(x, m, slot, exact)
            w, ids, maps = core._moe_topk(logits, True, m, slot, LAYERS[0], exact)
            if padded != rows:
                pad = torch.full(
                    (padded - rows, TOP_K), _routing_op.pad_expert_id(offset), dtype=ids.dtype,
                    device="cuda",
                )  # fmt: skip
                s = core._moe_slots[slot]
                sel = s.sel[: padded * TOP_K]
                sel.copy_(torch.cat([ids, pad]).reshape(-1))
                maps = _routing_op.cute_dsl_moe_routing(
                    sel, offset, s.maps, s.counts, s.barrier, core._moe_k.routing
                )
                ids = sel.view(padded, TOP_K)
            torch.cuda.synchronize()
            _routing_op.check_maps(maps, ids, offset)
            tiles = int(maps["num_non_exiting_tiles"][0])
            limit = maps["tile_idx_to_mn_limit"][:tiles].tolist()
            p2e = maps["permuted_idx_to_expanded_idx"]
            for t, hi in enumerate(limit):
                seen.index_add_(0, p2e[t * TILE : hi].long(), torch.ones(hi - t * TILE, **_i64()))
    finally:
        core.expert_offset = OFFSET
    assert bool((seen[: rows * TOP_K] == 1).all()), "a selection is routed on 0 or 2+ ranks"
    assert bool((seen[rows * TOP_K :] == 0).all()), "a padding row is routed"


def _i64() -> dict:
    return {"dtype": torch.long, "device": "cuda"}
