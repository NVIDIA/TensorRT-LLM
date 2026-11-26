# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the trunk's routed MoE layer at a decode width up to 64 that routing pads to a
power of two, with the shared expert accumulated by FC2 (`_moe_acc_padded`).

MTP5 at 1-3 requests carries 6 / 12 / 18 rows per decode step (MTP3 at 3: 12). Routing pads
them to 8 / 16 / 32: the padding tokens route to a non-local expert at weight 0, the routing
kernel rebuilds the maps, and before this path the output was cleared and the shared expert
added after FC2 in a separate launch. Now `_moe_begin_layer` hands the slot's first `rows`
rows to the shared expert there too, as at the power-of-two widths, and FC2 adds onto them.

This drives the target's own methods (`_moe_begin_layer`, `_routed`, and the add of
`_mlp_and_close` on the old path) on a core holding only the decode MoE state, prequantized
as the pre-MoE collective leaves a decode width (the routed experts' quantize at the exact
rows stands in for it on one GPU), and checks:

* each path against the fp32 reference of the grouped GEMMs on the maps, weights and NVFP4
  input the step wrote, onto the shared partial; and the two paths against each other;
* the padding rows of the slot are neither cleared nor read: both slots are filled with NaN
  before the step, and the real rows come out finite while the padding rows stay NaN;
* a CUDA graph of two layers (both operand slots) replayed on fresh inputs against the same
  calls run eagerly;
* the gate: which widths accumulate, with the switch on and off, and the route counter.
"""

import math
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
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_hidden_add as _add_op,
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
    cute_dsl_moe_zero as _zero_op,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization import (  # noqa: E402, E501
    cute_dsl_moe_quantize_nvfp4 as _quant_op,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

HIDDEN, INTER, EXPERTS, LOCAL = 6144, 2048, 256, 64
TOP_K = _routing_op.TOP_K
#: The rank under test: EP rank 1's window.
OFFSET = 64
#: MTP5 at 1-3 requests (6 / 12 / 18 rows; MTP3 at 3 is 12), a count below the router's
#: split-K partials (3), one in the 64-row bucket (40), and 16 as the power-of-two control.
ROWS = (3, 6, 12, 16, 18, 40)
#: Two trunk layer slots, so the capture alternates both operand slots.
LAYERS = (3, 4)
#: Hot expert (every token selects it, so every real row gets a routed contribution) and
#: the window favoured so about half of each row's winners are this rank's.
HOT = OFFSET + 1
ROUTED_SCALE = 2.5
FC1_KBLK = _fc1_kblock.enabled()
FC2_KBLK = _fc1_kblock.fc2_enabled()

_STATE: dict = {}


def _seed(tag: str) -> int:
    return zlib.crc32(tag.encode()) % 2**31


def _pow2(rows: int) -> int:
    return 1 << max(0, (rows - 1).bit_length())


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
    g = torch.Generator(device="cuda").manual_seed(_seed("acc padded experts"))
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
    checkpoint), the router skewed toward this rank's window with one hot expert."""
    g = torch.Generator(device="cuda").manual_seed(_seed(f"acc padded layer {layer}"))
    router = torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.2
    bias[OFFSET : OFFSET + LOCAL] += 0.3
    bias[HOT] = 4.0
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
    """The fields of `_build_moe`'s trunk slot that a decode width reads."""
    rows = _modeling._MOE_MAX_T
    counts, barrier = _routing_op.allocate_workspace(dev)
    f32 = {"dtype": torch.float32, "device": dev}
    return SimpleNamespace(
        out=torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=dev),
        x=torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=dev),
        w=torch.zeros((rows, TOP_K), **f32),
        sel=torch.empty(rows * TOP_K, dtype=torch.int32, device=dev),
        maps=_routing_op.allocate_maps(rows, dev),
        counts=counts,
        barrier=barrier,
        router_out=torch.zeros((_modeling._DECODE_MAX_ROWS, EXPERTS), **f32),
        router_partials=torch.zeros(
            (_modeling._DECODE_MAX_ROWS, _router_op.SPLIT_K, EXPERTS), **f32
        ),
        router_x=torch.zeros(
            (_modeling._DECODE_MAX_ROWS, HIDDEN), dtype=torch.bfloat16, device=dev
        ),
        topk_out={w: _topk_op.allocate_decode_outputs(w, dev) for w in ROWS},
        c=_experts_op.allocate_decode_c(rows, INTER, dev),
        sfc=_experts_op.allocate_sfc(rows, INTER, dev),
    )


def state() -> SimpleNamespace:
    """A target core holding only what the routed experts read at a decode width (built once
    a session): the decode kernels `_build_moe` builds at ROWS and their padded widths, the
    two slots, and two layers."""
    if "state" in _STATE:
        return _STATE["state"]
    dev = torch.cuda.current_device()
    core = _modeling.GlmMoeDsaTep4ModelingV2.__new__(_modeling.GlmMoeDsaTep4ModelingV2)
    torch.nn.Module.__init__(core)
    core.route_hits, core.route_misses = {}, {}
    core.expert_offset = OFFSET
    core.routed_scale = ROUTED_SCALE
    core.max_draft_len = 5
    core.fc1_kblk = FC1_KBLK
    core.fc2_kblk = FC2_KBLK
    core._moe_acc_padded = True
    core._rows = SimpleNamespace(
        norm=None, add=_add_op.compile(), gather=None, topk=None, swiglu=None,
        copy=_copy_op.compile(), argmax=None,
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
    s0 = core._moe_slots[0]
    padded = sorted({_pow2(r) for r in ROWS})
    core._moe_k = SimpleNamespace(
        routing=_routing_op.build(padded, dev),
        experts=_experts_op.build_decode(
            c=s0.c,
            sfc=s0.sfc,
            maps=s0.maps,
            xq=core._moe_xq,
            xsf=core._moe_xsf,
            out=s0.out,
            hidden=HIDDEN,
            inter=INTER,
            experts=LOCAL,
            max_rows=rows,
            fc1_kblk=FC1_KBLK,
            fc2_kblk=FC2_KBLK,
            dyn_slots=max(LAYERS) + 1,
        ),
        # As `_build_moe`: the fused top-k releases FC1 early only at a power of two.
        topk={
            w: _topk_op.build_decode(
                w, _router_op.decode_splits(w), dev, signal_gg1=(w & (w - 1)) == 0
            )
            for w in ROWS
        },
        topk_rep={},
        router={w: _router_op.build(w, dev) for w in ROWS},
        # The collective's NVFP4 epilogue stand-in at the exact rows; the clears per width.
        quant={w: _quant_op.build_decode(w, dev) for w in ROWS},
        zero={w: _zero_op.build_decode(w, dev) for w in padded},
        pair=None,
        pair_rep=None,
        topk_pre=None,
        topk_pre_rep=None,
        quant_pre=None,
        zero_pre=None,
        merge=None,
    )
    stacks = _experts()
    st = SimpleNamespace(core=core, layers=[_layer_weights(i, stacks) for i in LAYERS])
    _STATE["state"] = st
    return st


def moe_layer(core, x, partial, m, rows: int, layer: int):
    """`_mlp_and_close`'s MoE half at a decode width from the pre-MoE collective's output on:
    the routed experts' NVFP4 input already written (`_moe_prequant` holds at 1..64 rows).
    The shared expert's result is `partial`, copied on the side stream into the buffer its
    GEMM writes: the routed output when `_moe_begin_layer` hands one out, else a fresh tensor
    that is added after. Returns `(mlp_out, slot)`."""
    _quant_op.cute_dsl_moe_quantize_nvfp4(
        x, m.g1, core._moe_xq[:rows], core._moe_xsf[:rows], core._moe_k.quant[rows]
    )
    acc = core._moe_begin_layer(rows)
    slot = core._moe_slot
    main = torch.cuda.current_stream()
    side = core._side_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        shared = acc if acc is not None else torch.empty_like(partial)
        core._copy_into(shared, partial)
    routed = core._routed(
        x, m, rows, True, side if acc is not None else None, True, acc is not None, layer
    )
    if acc is not None:
        return routed, slot
    main.wait_stream(side)
    return core._add(routed, shared), slot


def layer_inputs(rows: int) -> list:
    bf = {"dtype": torch.bfloat16, "device": "cuda"}
    return [(torch.empty(rows, HIDDEN, **bf), torch.empty(rows, HIDDEN, **bf)) for _ in LAYERS]


def fill(inputs, rows: int, tag: str) -> None:
    """Fresh activations and shared partials into the static input buffers."""
    g = torch.Generator(device="cuda").manual_seed(_seed(tag))
    for x, p in inputs:
        x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
        p.copy_((torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16))


def poison(core) -> None:
    """NaN in both slots' outputs: a padding row that were read, or a real row that were not
    written before FC2 added onto it, would carry a NaN out."""
    for s in core._moe_slots:
        s.out[: _modeling._DECODE_MAX_ROWS].fill_(float("nan"))


def run_step(st, inputs, rows: int) -> list:
    return [moe_layer(st.core, x, p, m, rows, i) for (x, p), m, i in zip(inputs, st.layers, LAYERS)]


def _snapshot(core, results, rows: int) -> dict:
    """Per layer its output, the padded slot rows past it, top-k weights and ids as FC2 read
    them (padding included), and the maps in canonical form; the NVFP4 input. At a power of
    two FC2 reads the top-k's own outputs; padded, the staged ids and weights and the
    routing kernel's maps."""
    padded = _pow2(rows)
    snap = {}
    for j, (out, slot) in enumerate(results):
        s = core._moe_slots[slot]
        if padded == rows:
            t = s.topk_out[rows]
            w, ids, maps = t["topk_weights"].view(rows, TOP_K), t["topk_indices"], t
        else:
            w, ids, maps = s.w[:padded], s.sel[: padded * TOP_K], s.maps
        snap[f"out{j}"] = out.clone()
        snap[f"pad{j}"] = s.out[rows:padded].clone()
        snap[f"w{j}"] = w.clone()
        snap[f"ids{j}"] = ids.clone()
        for k, t in _routing_op.canonical_maps(maps, padded * TOP_K).items():
            snap[f"maps{j}.{k}"] = t
    snap["xq"] = core._moe_xq[:rows].clone()
    snap["xsf"] = core._moe_xsf[:rows].clone()
    return snap


#: FC2 adds each token's (up to eight) expert contributions with bf16 atomics in an order
#: that varies run to run (onto the shared partial on the new path, onto zeros on the old
#: one, which then adds the partial in one more rounding). Each run is within ~4.5 ulp of the
#: row scale of the fp32 sum at its worst element and ~0.8 ulp relative RMS (the experts
#: test's derivation), so two runs within 9 and ~1.1; the band is 16 and 2. A token missing
#: an expert is ~35 % relative RMS.
PAIR_ELEMENT_ULP, PAIR_RMS_ULP = 16.0, 2.0


def _reference(core, snap: dict, partial, m, rows: int, j: int) -> torch.Tensor:
    """Layer `j`'s fp32 reference of the grouped GEMMs on the maps, weights and NVFP4 input
    the step wrote, onto the shared partial, rounded once to bf16."""
    padded = _pow2(rows)
    maps = {k: snap[f"maps{j}.{k}"] for k in _routing_op.MAP_KEYS}
    xq = torch.zeros(padded, HIDDEN // 2, dtype=torch.uint8, device="cuda")
    xsf = torch.zeros(padded, HIDDEN // 16, dtype=torch.uint8, device="cuda")
    xq[:rows], xsf[:rows] = snap["xq"], snap["xsf"]
    initial = torch.zeros(padded, HIDDEN, dtype=torch.bfloat16, device="cuda")
    initial[:rows] = partial
    ref = _experts_op.cute_dsl_moe_nvfp4_experts.reference(
        xq, xsf, maps, snap[f"w{j}"], m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s, m.norm_const,
        m.alpha1, m.alpha2, initial, None, None, None, core._moe_k.experts,
    )  # fmt: skip
    return ref[:rows]


def _assert_not_degenerate(ref, partial, snap: dict, rows: int, j: int) -> None:
    """Every real row takes the hot expert, so its routed part is far from zero; the padding
    rows are routed nowhere."""
    routed = ref.float() - partial.float()
    per_row = routed.pow(2).mean(1).sqrt() / partial.float().pow(2).mean(1).sqrt()
    assert bool((per_row > 0.05).all()), f"a real row has no routed contribution: {per_row}"
    ids = snap[f"ids{j}"].view(-1, TOP_K)
    assert bool((ids[:rows] == HOT).any(1).all()), "a real row missed the hot expert"
    pad_id = _routing_op.pad_expert_id(OFFSET)
    assert bool((ids[rows:] == pad_id).all()), "a padding row selected a real expert"


def _stats(got, want) -> tuple[float, float, float]:
    """`(max abs, max abs over the largest |want|, relative RMS)`."""
    d = got.float() - want.float()
    scale = want.float().abs().amax().clamp_min(1e-9)
    rms = d.pow(2).mean().sqrt() / want.float().pow(2).mean().sqrt().clamp_min(1e-9)
    return float(d.abs().amax()), float(d.abs().amax() / scale), float(rms)


def _took(core, before: dict, route: str) -> int:
    return core.route_hits.get(route, 0) - before.get(route, 0)


@pytest.mark.parametrize("rows", ROWS)
def test_accumulate_matches_the_add_and_the_reference(rows: int, monkeypatch) -> None:
    """Same inputs through the old path (clear, FC2 onto zeros, add the shared partial) and
    the new one (FC2 onto the partial): each within the experts test's band of the fp32
    reference, and the two within the pair band. Both slots are NaN-poisoned first: the new
    path's real rows come out finite, its padding rows stay NaN (not cleared, not written,
    not read), and the old path clears them."""
    st = state()
    core = st.core
    padded = _pow2(rows)
    inputs = layer_inputs(rows)
    fill(inputs, rows, f"pair {rows}")
    results = {}
    for acc_padded in (False, True):
        monkeypatch.setattr(core, "_moe_acc_padded", acc_padded)
        before = dict(core.route_hits)
        missed = dict(core.route_misses)
        poison(core)
        with validating(
            _router_op.cute_dsl_moe_router,
            _topk_op.cute_dsl_moe_topk_routing,
            _routing_op.cute_dsl_moe_routing,
            _quant_op.cute_dsl_moe_quantize_nvfp4,
            _zero_op.cute_dsl_moe_zero,
            _experts_op.cute_dsl_moe_nvfp4_experts,
            _copy_op.cute_dsl_copy,
            _add_op.cute_dsl_hidden_add,
        ):
            snap = _snapshot(core, run_step(st, inputs, rows), rows)
        torch.cuda.synchronize()
        accumulated = acc_padded or padded == rows
        assert _took(core, before, "moe_acc_padded") == (
            len(LAYERS) if acc_padded and padded != rows else 0
        )
        adds = (
            _took(core, before, "hidden_add")
            + core.route_misses.get("hidden_add", 0)
            - missed.get("hidden_add", 0)
        )
        assert adds == (0 if accumulated else len(LAYERS))
        for j in range(len(LAYERS)):
            assert bool(torch.isfinite(snap[f"out{j}"]).all()), "a real row read NaN"
            if padded != rows:
                pad = snap[f"pad{j}"]
                if accumulated:
                    assert bool(torch.isnan(pad).all()), "a padding row was written"
                else:
                    assert bool((pad == 0).all()), "the old path left a padding row uncleared"
        results[acc_padded] = snap
    for k in results[True]:
        # Everything upstream of FC2 is the same bytes on both paths.
        if k.startswith(("w", "ids", "maps", "xq", "xsf")):
            assert torch.equal(results[True][k], results[False][k]), k
    for j, ((x, partial), m) in enumerate(zip(inputs, st.layers)):
        # The NVFP4 input is one buffer, holding the last layer's after the step: requantize
        # this layer's (the same kernel, so the bytes FC1 read).
        _quant_op.cute_dsl_moe_quantize_nvfp4(
            x, m.g1, core._moe_xq[:rows], core._moe_xsf[:rows], core._moe_k.quant[rows]
        )
        xq, xsf = core._moe_xq[:rows].clone(), core._moe_xsf[:rows].clone()
        snap = dict(results[True], xq=xq, xsf=xsf)
        ref = _reference(core, snap, partial, m, rows, j)
        _assert_not_degenerate(ref, partial, results[True], rows, j)
        new, old = results[True][f"out{j}"], results[False][f"out{j}"]
        assert_within_ulp(new, ref, element_ulp=32.0, rms_ulp=2.0)
        assert_within_ulp(old, ref, element_ulp=32.0, rms_ulp=2.0)
        assert_within_ulp(new, old, PAIR_ELEMENT_ULP, PAIR_RMS_ULP)
        n_abs, n_rel, n_rms = _stats(new, ref)
        o_abs, o_rel, o_rms = _stats(old, ref)
        d_abs, d_rel, d_rms = _stats(new, old)
        print(
            f"ACC_PADDED rows={rows} layer={j} new-vs-ref max_abs={n_abs:.3e} "
            f"max_rel={n_rel:.3e} rms={n_rms:.3e} | old-vs-ref max_abs={o_abs:.3e} "
            f"max_rel={o_rel:.3e} rms={o_rms:.3e} | new-vs-old max_abs={d_abs:.3e} "
            f"max_rel={d_rel:.3e} rms={d_rms:.3e} bitwise={torch.equal(new, old)}"
        )
        assert math.isfinite(d_abs)


@pytest.mark.parametrize("rows", ROWS)
def test_capture_replays_match_eager(rows: int) -> None:
    """Capture two layers on the accumulate path, then three times: fresh inputs, NaN in the
    slots, replay, snapshot, the same calls eagerly, snapshot, compare (everything upstream
    of FC2 bitwise; FC2's output within the pair band; the padding rows untouched by both),
    and the eager output against the reference."""
    st = state()
    core = st.core
    core._moe_acc_padded = True
    padded = _pow2(rows)
    inputs = layer_inputs(rows)
    fill(inputs, rows, f"warm {rows}")
    run_step(st, inputs, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = run_step(st, inputs, rows)
    for rep in range(3):
        fill(inputs, rows, f"replay {rows} {rep}")
        poison(core)
        graph.replay()
        torch.cuda.synchronize()
        got = _snapshot(core, results, rows)
        poison(core)
        eager = run_step(st, inputs, rows)
        torch.cuda.synchronize()
        want = _snapshot(core, eager, rows)
        assert [s for _, s in eager] == [s for _, s in results], "slot parity drifted"
        for k in want:
            if k.startswith("out"):
                assert bool(torch.isfinite(got[k]).all()), f"{k}: a real row read NaN"
                assert_within_ulp(got[k], want[k], PAIR_ELEMENT_ULP, PAIR_RMS_ULP)
            elif k.startswith("pad"):
                assert bool(torch.isnan(got[k]).all()) == bool(torch.isnan(want[k]).all())
                assert padded == rows or bool(torch.isnan(got[k]).all()), k
            else:
                assert torch.equal(got[k], want[k]), f"{k}: replay differs from eager"
        j = len(LAYERS) - 1
        ref = _reference(core, want, inputs[j][1], st.layers[j], rows, j)
        _assert_not_degenerate(ref, inputs[j][1], want, rows, j)
        assert_within_ulp(want[f"out{j}"], ref, element_ulp=32.0, rms_ulp=2.0)


def test_which_widths_accumulate(monkeypatch) -> None:
    """With the switch on, every decode width 1..64 hands out the slot's first `rows` rows
    (counted as `moe_acc_padded` where routing pads); off, only the powers of two. Above 64
    a padded count (no prefill pair here) never does. Each call advances the slot once."""
    core = state().core
    pow2 = {1, 2, 4, 8, 16, 32, 64}
    try:
        for on in (True, False):
            monkeypatch.setattr(core, "_moe_acc_padded", on)
            for rows in range(1, _modeling._DECODE_MAX_ROWS + 1):
                before = dict(core.route_hits)
                slot = core._moe_slot
                acc = core._moe_begin_layer(rows)
                assert core._moe_slot == slot ^ 1
                if on or rows in pow2:
                    s = core._moe_slots[core._moe_slot]
                    assert acc is not None and acc.data_ptr() == s.out.data_ptr(), rows
                    assert tuple(acc.shape) == (rows, HIDDEN)
                else:
                    assert acc is None, rows
                counted = _took(core, before, "moe_acc_padded")
                assert counted == (1 if on and rows not in pow2 else 0), rows
            for rows in (65, 100, 200):
                assert core._moe_begin_layer(rows) is None, rows
    finally:
        core._moe_slot = 0
