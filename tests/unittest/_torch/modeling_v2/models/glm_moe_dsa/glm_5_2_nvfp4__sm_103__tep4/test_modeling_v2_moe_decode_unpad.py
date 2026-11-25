# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the trunk's routed MoE layer at a decode width up to 64 that is not a power of
two, run unpadded on the fused top-k's own maps (`_moe_dec_unpadded`,
`TRTLLM_GLM52_MOE_DEC_UNPAD`).

MTP5 at 1-3 requests carries 6 / 12 / 18 rows per decode step. Before, `_moe_experts` padded
them to 8 / 16 / 32: two fills and two cats for the padding ids and weights, two staging
copies and the routing kernel rebuilding the maps the top-k had already written, seven
launches between the top-k and FC1 (8.3 us of each MoE layer's critical chain at 12 rows on
B300). Unpadded, FC1 takes the top-k's maps and weights directly; on an MTP3 / MTP5 engine
the step widths (multiples of `max_draft_len + 1`) also select on a top-k handle that
releases FC1 early (`k.topk_sig`), as the power-of-two widths do.

This drives the target's own methods (`_moe_begin_layer`, `_routed`) on a core holding only
the decode MoE state (as `test_modeling_v2_moe_decode_acc_padded.py`), prequantized as the
pre-MoE collective leaves a decode width, and checks:

* unpadded against padded on the same inputs: the same selection, weights and maps (the
  padded maps restricted to the real rows), both within the experts band of the fp32
  reference on those maps and within the pair band of each other; real rows finite and the
  slot's padding rows untouched (NaN-poisoned) on both paths; the route counters (no
  routing launch unpadded, the early-release handle at the step widths);
* a CUDA graph of two layers (both operand slots) on the unpadded path, early-release
  handles included, replayed on fresh inputs against the same calls run eagerly;
* the MTP layer (quantizes and clears itself) keeps the padded path and the plain handle;
* the fail-fast guard: an early-release top-k followed by the padded path raises.
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
#: EP rank 2's window.
OFFSET = 128
#: MTP5 at 1-3 requests (6 / 12 / 18: the early-release widths of `max_draft_len` 5), odd
#: counts below the router's split-K partials (3) and in the 64-row bucket (41, 63), and 16
#: as the power-of-two control (unchanged by the switch).
ROWS = (3, 6, 12, 16, 18, 41, 63)
MAX_DRAFT_LEN = 5
SIG_ROWS = tuple(
    w for w in range(MAX_DRAFT_LEN + 1, 65, MAX_DRAFT_LEN + 1) if w & (w - 1) and w in ROWS
)
#: Two trunk layer slots, so a capture alternates both operand slots; the MTP layer's slot.
LAYERS = (3, 4)
NUM_LAYERS = 78
HOT = OFFSET + 5
ROUTED_SCALE = 2.5
FC1_KBLK = _fc1_kblock.enabled()
FC2_KBLK = _fc1_kblock.fc2_enabled()
#: The experts test's bands (see `test_modeling_v2_moe_decode_acc_padded.py`): each run
#: within 32 / 2 ulp of the fp32 reference, two runs (atomic order) within 16 / 2.
REF_ELEMENT_ULP, REF_RMS_ULP = 32.0, 2.0
PAIR_ELEMENT_ULP, PAIR_RMS_ULP = 16.0, 2.0

_STATE: dict = {}


def _seed(tag: str) -> int:
    return zlib.crc32(tag.encode()) % 2**31


def _pow2(rows: int) -> int:
    return 1 << max(0, (rows - 1).bit_length())


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


def _experts():
    g = torch.Generator(device="cuda").manual_seed(_seed("unpad experts"))
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
    g = torch.Generator(device="cuda").manual_seed(_seed(f"unpad layer {layer}"))
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
    """A core holding only the decode MoE state, with `_build_moe`'s handles at ROWS: the
    plain top-k (early release at the powers of two) and the early-release handles at the
    step widths (`k.topk_sig`)."""
    if "state" in _STATE:
        return _STATE["state"]
    dev = torch.cuda.current_device()
    core = _modeling.GlmMoeDsaTep4ModelingV2.__new__(_modeling.GlmMoeDsaTep4ModelingV2)
    torch.nn.Module.__init__(core)
    core.route_hits, core.route_misses = {}, {}
    core.expert_offset = OFFSET
    core.routed_scale = ROUTED_SCALE
    core.max_draft_len = MAX_DRAFT_LEN
    core.num_layers = NUM_LAYERS
    core.fc1_kblk = FC1_KBLK
    core.fc2_kblk = FC2_KBLK
    core._moe_acc_padded = True
    core._moe_dec_unpad = True
    core._moe_dec_unpad_signal = True
    core._moe_topk_signaled = False
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
            dyn_slots=NUM_LAYERS + 1,
        ),
        topk={
            w: _topk_op.build_decode(
                w, _router_op.decode_splits(w), dev, signal_gg1=(w & (w - 1)) == 0
            )
            for w in ROWS
        },
        topk_sig={
            w: _topk_op.build_decode(w, _router_op.decode_splits(w), dev, signal_gg1=True)
            for w in SIG_ROWS
        },
        topk_rep={},
        router={w: _router_op.build(w, dev) for w in ROWS},
        # The collective's NVFP4 epilogue stand-in at the exact rows; the MTP layer's own
        # quantize and clear at the padded widths.
        quant={w: _quant_op.build_decode(w, dev) for w in sorted(set(ROWS) | set(padded))},
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
    """`_mlp_and_close`'s MoE half at a decode width, from the pre-MoE collective's output on
    (its NVFP4 epilogue stood in for by the quantize at the exact rows); the shared expert's
    result `partial` copied on the side stream into the buffer `_moe_begin_layer` hands out.
    Returns `(mlp_out, slot)`."""
    _quant_op.cute_dsl_moe_quantize_nvfp4(
        x, m.g1, core._moe_xq[:rows], core._moe_xsf[:rows], core._moe_k.quant[rows]
    )
    acc = core._moe_begin_layer(rows)
    assert acc is not None, "a decode width always accumulates here"
    slot = core._moe_slot
    main = torch.cuda.current_stream()
    side = core._side_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        core._copy_into(acc, partial)
    routed = core._routed(x, m, rows, True, side, True, True, layer)
    return routed, slot


def layer_inputs(rows: int) -> list:
    bf = {"dtype": torch.bfloat16, "device": "cuda"}
    return [(torch.empty(rows, HIDDEN, **bf), torch.empty(rows, HIDDEN, **bf)) for _ in LAYERS]


def fill(inputs, rows: int, tag: str) -> None:
    g = torch.Generator(device="cuda").manual_seed(_seed(tag))
    for x, p in inputs:
        x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
        p.copy_((torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16))


def poison(core) -> None:
    for s in core._moe_slots:
        s.out[: _modeling._DECODE_MAX_ROWS].fill_(float("nan"))


def run_step(st, inputs, rows: int) -> list:
    return [moe_layer(st.core, x, p, m, rows, i) for (x, p), m, i in zip(inputs, st.layers, LAYERS)]


def _snapshot(core, results, rows: int, unpadded: bool) -> dict:
    """Per layer its output, the slot's rows past it up to the padded width, and the weights,
    ids and canonical maps FC2 read: the top-k's own outputs unpadded or at a power of two,
    else the staged weights and ids and the routing kernel's maps (all over the padded
    width). The NVFP4 input of the last layer."""
    padded = _pow2(rows)
    snap = {}
    for j, (out, slot) in enumerate(results):
        s = core._moe_slots[slot]
        if unpadded or padded == rows:
            t = s.topk_out[rows]
            w, ids, maps, n = t["topk_weights"].view(rows, TOP_K), t["topk_indices"], t, rows
        else:
            w, ids, maps, n = s.w[:padded], s.sel[: padded * TOP_K], s.maps, padded
        snap[f"out{j}"] = out.clone()
        snap[f"pad{j}"] = s.out[rows:padded].clone()
        snap[f"w{j}"] = w.reshape(n, TOP_K).clone()
        snap[f"ids{j}"] = ids.reshape(-1)[: n * TOP_K].view(n, TOP_K).clone()
        for k, v in _routing_op.canonical_maps(maps, n * TOP_K).items():
            snap[f"maps{j}.{k}"] = v
    return snap


def _reference(core, snap: dict, x, partial, m, rows: int, j: int) -> torch.Tensor:
    """Layer `j`'s fp32 reference of the grouped GEMMs on the maps and weights the step
    wrote (over the real rows), onto the shared partial, rounded once to bf16."""
    _quant_op.cute_dsl_moe_quantize_nvfp4(
        x, m.g1, core._moe_xq[:rows], core._moe_xsf[:rows], core._moe_k.quant[rows]
    )
    xq, xsf = core._moe_xq[:rows].clone(), core._moe_xsf[:rows].clone()
    n = snap[f"w{j}"].shape[0]
    maps = {k: snap[f"maps{j}.{k}"] for k in _routing_op.MAP_KEYS}
    if n != rows:
        maps["expanded_idx_to_permuted_idx"] = maps["expanded_idx_to_permuted_idx"][: rows * TOP_K]
    ref = _experts_op.cute_dsl_moe_nvfp4_experts.reference(
        xq, xsf, maps, snap[f"w{j}"][:rows].contiguous(), m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s,
        m.norm_const, m.alpha1, m.alpha2, partial.clone(), None, None, None,
        core._moe_k.experts,
    )  # fmt: skip
    return ref[:rows]


def _assert_not_degenerate(ref, partial, ids, rows: int) -> None:
    routed = ref.float() - partial.float()
    per_row = routed.pow(2).mean(1).sqrt() / partial.float().pow(2).mean(1).sqrt()
    assert bool((per_row > 0.05).all()), f"a real row has no routed contribution: {per_row}"
    assert bool((ids[:rows] == HOT).any(1).all()), "a real row missed the hot expert"


def _took(core, before: dict, route: str) -> int:
    return core.route_hits.get(route, 0) - before.get(route, 0)


def _same_maps(a: dict, ja: int, b: dict, jb: int, rows: int) -> None:
    """Two canonical map sets over the same real rows (either may cover padding rows past
    them, which route nowhere): same tiles, same rows in them."""
    for k in _routing_op.MAP_KEYS:
        x, y = a[f"maps{ja}.{k}"], b[f"maps{jb}.{k}"]
        if k == "expanded_idx_to_permuted_idx":
            n = rows * TOP_K
            assert torch.equal(x[:n], y[:n]), k
            assert bool((x[n:] == -1).all()) and bool((y[n:] == -1).all()), f"{k}: padding mapped"
        else:
            assert torch.equal(x, y), k


@pytest.mark.parametrize("rows", ROWS)
def test_unpadded_matches_padded_and_the_reference(rows: int, monkeypatch) -> None:
    st = state()
    core = st.core
    padded = _pow2(rows)
    inputs = layer_inputs(rows)
    fill(inputs, rows, f"pair {rows}")
    results = {}
    for unpad in (False, True):
        monkeypatch.setattr(core, "_moe_dec_unpad", unpad)
        before = dict(core.route_hits)
        poison(core)
        with validating(
            _router_op.cute_dsl_moe_router,
            _topk_op.cute_dsl_moe_topk_routing,
            _routing_op.cute_dsl_moe_routing,
            _quant_op.cute_dsl_moe_quantize_nvfp4,
            _experts_op.cute_dsl_moe_nvfp4_experts,
            _copy_op.cute_dsl_copy,
        ):
            snap = _snapshot(core, run_step(st, inputs, rows), rows, unpad)
        torch.cuda.synchronize()
        moved = unpad and padded != rows
        assert _took(core, before, "moe_dec_unpad") == (len(LAYERS) if moved else 0)
        assert _took(core, before, "moe_dec_unpad_signal") == (
            len(LAYERS) if moved and rows in SIG_ROWS else 0
        )
        assert _took(core, before, "moe_acc_padded") == (
            0 if unpad or padded == rows else len(LAYERS)
        )
        for j in range(len(LAYERS)):
            assert bool(torch.isfinite(snap[f"out{j}"]).all()), "a real row read NaN"
            if padded != rows:
                assert bool(torch.isnan(snap[f"pad{j}"]).all()), "a padding row was written"
        results[unpad] = snap
    new, old = results[True], results[False]
    for j in range(len(LAYERS)):
        # Upstream of FC1 the same selection on both paths.
        assert torch.equal(new[f"w{j}"], old[f"w{j}"][:rows]), "weights differ"
        assert torch.equal(new[f"ids{j}"], old[f"ids{j}"][:rows]), "ids differ"
        _same_maps(new, j, old, j, rows)
    for j, ((x, partial), m) in enumerate(zip(inputs, st.layers)):
        ref = _reference(core, new, x, partial, m, rows, j)
        _assert_not_degenerate(ref, partial, new[f"ids{j}"], rows)
        assert_within_ulp(new[f"out{j}"], ref, element_ulp=REF_ELEMENT_ULP, rms_ulp=REF_RMS_ULP)
        assert_within_ulp(old[f"out{j}"], ref, element_ulp=REF_ELEMENT_ULP, rms_ulp=REF_RMS_ULP)
        assert_within_ulp(new[f"out{j}"], old[f"out{j}"], PAIR_ELEMENT_ULP, PAIR_RMS_ULP)
        d = (new[f"out{j}"].float() - old[f"out{j}"].float()).abs().amax()
        print(
            f"UNPAD rows={rows} layer={j} sig={rows in SIG_ROWS} new-vs-old max_abs={float(d):.3e} "
            f"bitwise={torch.equal(new[f'out{j}'], old[f'out{j}'])}"
        )


@pytest.mark.parametrize("rows", ROWS)
def test_capture_replays_match_eager(rows: int) -> None:
    """Two layers on the unpadded path (early-release handles at the step widths) in one CUDA
    graph, replayed three times on fresh inputs against the same calls eagerly: selection
    and maps bitwise, outputs within the pair band, the eager output against the reference."""
    st = state()
    core = st.core
    core._moe_dec_unpad = True
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
        got = _snapshot(core, results, rows, True)
        poison(core)
        eager = run_step(st, inputs, rows)
        torch.cuda.synchronize()
        want = _snapshot(core, eager, rows, True)
        assert [s for _, s in eager] == [s for _, s in results], "slot parity drifted"
        for k in want:
            if k.startswith("out"):
                assert bool(torch.isfinite(got[k]).all()), f"{k}: a real row read NaN"
                assert_within_ulp(got[k], want[k], PAIR_ELEMENT_ULP, PAIR_RMS_ULP)
            elif k.startswith("pad"):
                assert bool(torch.isnan(got[k]).all()), f"{k}: a padding row was written"
            else:
                assert torch.equal(got[k], want[k]), f"{k}: replay differs from eager"
        j = len(LAYERS) - 1
        x, partial = inputs[j]
        ref = _reference(core, want, x, partial, st.layers[j], rows, j)
        _assert_not_degenerate(ref, partial, want[f"ids{j}"], rows)
        assert_within_ulp(want[f"out{j}"], ref, element_ulp=REF_ELEMENT_ULP, rms_ulp=REF_RMS_ULP)


@pytest.mark.parametrize("rows", [r for r in ROWS if r & (r - 1)])
def test_mtp_layer_keeps_the_padded_path(rows: int) -> None:
    """The MTP layer's call (`_mtp_close`: not prequantized, no shared partial, layer slot
    `num_layers`) at a width that is not a power of two: the plain top-k handle, the padded
    path with its own quantize and clear, and a result within the band of the reference."""
    st = state()
    core = st.core
    core._moe_dec_unpad = True
    m = st.layers[0]
    x = layer_inputs(rows)[0][0]
    fill([(x, torch.empty_like(x))], rows, f"mtp {rows}")
    before = dict(core.route_hits)
    poison(core)
    routed = core._routed(x, m, rows, False, None, False, False, NUM_LAYERS)
    torch.cuda.synchronize()
    assert not core._moe_topk_signaled
    assert _took(core, before, "moe_dec_unpad") == 0
    assert _took(core, before, "moe_dec_unpad_signal") == 0
    assert bool(torch.isfinite(routed).all())
    slot = core._moe_slot ^ 1  # `_routed` advanced it after (began False)
    snap = _snapshot(core, [(routed, slot)], rows, False)
    ref = _reference(core, snap, x, torch.zeros_like(x), m, rows, 0)
    assert_within_ulp(routed, ref, element_ulp=REF_ELEMENT_ULP, rms_ulp=REF_RMS_ULP)


def test_gate_and_fail_fast(monkeypatch) -> None:
    """`_moe_dec_unpadded` holds for every decode width with a top-k handle and only with the
    switch on; an early-release top-k followed by the padded path raises instead of
    launching the routing kernel on ids still being written."""
    st = state()
    core = st.core
    for on in (True, False):
        monkeypatch.setattr(core, "_moe_dec_unpad", on)
        for rows in ROWS:
            assert core._moe_dec_unpadded(rows) == on, rows
        assert not core._moe_dec_unpadded(65)
    monkeypatch.setattr(core, "_moe_dec_unpad", False)
    rows = 12
    m = st.layers[0]
    x = layer_inputs(rows)[0][0]
    fill([(x, torch.empty_like(x))], rows, "fail fast")
    monkeypatch.setattr(core, "_moe_topk_signaled", True)
    s = core._moe_slots[core._moe_slot]
    t = s.topk_out[rows]
    with pytest.raises(RuntimeError, match="early-release top-k"):
        core._moe_experts(
            x, t["topk_indices"].view(rows, TOP_K), t["topk_weights"].view(rows, TOP_K),
            {k: t[k] for k in _routing_op.MAP_KEYS}, m, rows, False, None, False, False,
            NUM_LAYERS,
        )  # fmt: skip
    assert core._moe_topk_signaled is False
