# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the unpadded trunk decode MoE (`_moe_dec_unpadded`) with decode expert replicas,
all four EP ranks simulated on one GPU.

Unpadded, a decode width that is not a power of two runs on the replica top-k's maps (hot
experts moved whole to the least loaded rank, local slots `64 + h`), where the padded path
rebuilt home maps with the routing kernel. Each rank's FC2 output is a partial the EP
all-reduce sums, so the check is on the sum over the four ranks: the unpadded replica path
against the padded home path on the same inputs, both against the fp32 reference (the sum
of every rank's grouped-GEMM reference on its own maps), with the replica slots actually
taking tiles (a hot expert every token selects) and every (token, k) pair mapped on exactly
one rank.
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
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_hidden_add as _add_op,
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
    cute_dsl_moe_topk_routing_rep as _topk_rep_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.moe import (  # noqa: E402, E501
    cute_dsl_moe_zero as _zero_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization import (  # noqa: E402, E501
    cute_dsl_moe_quantize_nvfp4 as _quant_op,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402 — needs the path declared above

HIDDEN, INTER, EXPERTS, LOCAL = 6144, 2048, 256, 64
TOP_K = _routing_op.TOP_K
NREP = 8
#: MTP5 at 2 / 3 requests, and an odd count in the 64-row bucket; all on the replica top-k.
ROWS = (12, 18, 41)
SIG_ROWS = (12, 18)
LAYER = 5
NUM_LAYERS = 78
OFFSETS = (0, 64, 128, 192)
#: Two hot experts every token selects (bias), on different home ranks.
FORCED = (70, 200)
ROUTED_SCALE = 2.5
FC1_KBLK = _fc1_kblock.enabled()
FC2_KBLK = _fc1_kblock.fc2_enabled()

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


def _global_experts():
    """All 256 experts' NVFP4 weights (row-major) and per-expert alphas, one seed each."""
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    w = SimpleNamespace(
        fc1_w=torch.empty(EXPERTS, 2 * INTER, HIDDEN // 2, **u8),
        fc1_s=torch.empty(EXPERTS, 2 * INTER // 128, HIDDEN // 64, 512, **u8),
        fc2_w=torch.empty(EXPERTS, HIDDEN, INTER // 2, **u8),
        fc2_s=torch.empty(EXPERTS, HIDDEN // 128, INTER // 64, 512, **u8),
    )
    for e in range(EXPERTS):
        g = torch.Generator(device="cuda").manual_seed(_seed(f"rep expert {e}"))
        p, s = _nvfp4(torch.randn(2 * INTER, HIDDEN, generator=g, device="cuda") * 0.02)
        w.fc1_w[e], w.fc1_s[e] = p, _sf_atoms(s)
        p, s = _nvfp4(torch.randn(HIDDEN, INTER, generator=g, device="cuda") * 0.03)
        w.fc2_w[e], w.fc2_s[e] = p, _sf_atoms(s)
    g = torch.Generator(device="cuda").manual_seed(_seed("rep alphas"))
    w.alpha1 = torch.rand(EXPERTS, generator=g, device="cuda") + 0.5
    w.alpha2 = torch.rand(EXPERTS, generator=g, device="cuda") + 0.5
    return w


def _rank_weights(gw, router, bias, off: int, hot: list[int]) -> SimpleNamespace:
    """Rank `off`'s stacks as the loader lays them out: its 64 home experts, then the hot
    experts' copies in slots 64.. (`weights.DEC_HOT`)."""
    ids = list(range(off, off + LOCAL)) + list(hot)
    idx = torch.tensor(ids, device="cuda")
    e4 = torch.float8_e4m3fn
    fc1_w, fc2_w = gw.fc1_w[idx].contiguous(), gw.fc2_w[idx].contiguous()
    if FC1_KBLK:
        fc1_w = _fc1_kblock.to_kblock(fc1_w)
    if FC2_KBLK:
        fc2_w = _fc1_kblock.to_kblock(fc2_w)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    return SimpleNamespace(
        router=router,
        router_bias=bias,
        fc1_w=fc1_w,
        fc1_s=gw.fc1_s[idx].contiguous().view(e4),
        fc2_w=fc2_w,
        fc2_s=gw.fc2_s[idx].contiguous().view(e4),
        alpha1=gw.alpha1[idx].contiguous(),
        alpha2=gw.alpha2[idx].contiguous(),
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
        topk_out={w: _topk_rep_op.allocate_outputs(w, NREP, dev) for w in ROWS},
        c=_experts_op.allocate_decode_c(rows, INTER, dev),
        sfc=_experts_op.allocate_sfc(rows, INTER, dev),
    )


def state() -> SimpleNamespace:
    if "state" in _STATE:
        return _STATE["state"]
    dev = torch.cuda.current_device()
    core = _modeling.GlmMoeDsaTep4ModelingV2.__new__(_modeling.GlmMoeDsaTep4ModelingV2)
    torch.nn.Module.__init__(core)
    core.route_hits, core.route_misses = {}, {}
    core.expert_offset = 0
    core.routed_scale = ROUTED_SCALE
    core.max_draft_len = 5
    core.num_layers = NUM_LAYERS
    core.fc1_kblk, core.fc2_kblk = FC1_KBLK, FC2_KBLK
    core._moe_acc_padded = True
    core._moe_dec_unpad = True
    core._moe_dec_unpad_signal = True
    core._moe_topk_signaled = False
    core.dec_replicas = NREP
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
    core._pf = None
    core._dec_hot_none = torch.full((NREP,), -1, dtype=torch.int32, device=dev)
    s0 = core._moe_slots[0]
    splits = {w: _router_op.decode_splits(w) for w in ROWS}
    core._moe_k = SimpleNamespace(
        routing=_routing_op.build(sorted({_pow2(r) for r in ROWS}), dev),
        experts=_experts_op.build_decode(
            c=s0.c, sfc=s0.sfc, maps=s0.maps, xq=core._moe_xq, xsf=core._moe_xsf, out=s0.out,
            hidden=HIDDEN, inter=INTER, experts=LOCAL + NREP, max_rows=rows,
            fc1_kblk=FC1_KBLK, fc2_kblk=FC2_KBLK, dyn_slots=NUM_LAYERS + 1,
        ),  # fmt: skip
        topk={w: _topk_op.build_decode(w, splits[w], dev, signal_gg1=False) for w in ROWS},
        topk_rep={
            w: _topk_rep_op.build_decode(w, splits[w], NREP, dev, signal_gg1=False) for w in ROWS
        },
        topk_sig={
            w: _topk_rep_op.build_decode(w, splits[w], NREP, dev, signal_gg1=True)
            for w in SIG_ROWS
        },
        router={w: _router_op.build(w, dev) for w in ROWS},
        quant={w: _quant_op.build_decode(w, dev) for w in ROWS},
        zero={w: _zero_op.build_decode(w, dev) for w in sorted({_pow2(r) for r in ROWS})},
        pair=None, pair_rep=None, topk_pre=None, topk_pre_rep=None, quant_pre=None,
        zero_pre=None, merge=None,
    )  # fmt: skip
    g = torch.Generator(device="cuda").manual_seed(_seed("rep router"))
    router = (torch.randn(EXPERTS, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(
        torch.bfloat16
    )
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.2
    for e in FORCED:
        bias[e] = 4.0
    st = SimpleNamespace(core=core, gw=_global_experts(), router=router.contiguous(), bias=bias)
    _STATE["state"] = st
    return st


def _hot_ids(st, x) -> list[int]:
    """The batch's most selected experts first (the worst skew), the forced ones included."""
    logits = x.float() @ st.router.float().t()
    scores = torch.sigmoid(logits) + st.bias
    ids = scores.topk(TOP_K, dim=1).indices.reshape(-1)
    counts = torch.bincount(ids, minlength=EXPERTS)
    return [int(e) for e in counts.argsort(descending=True)[:NREP]]


def _rank_step(core, x, partial, m, rows: int):
    """`_mlp_and_close`'s MoE half on one rank (as test_modeling_v2_moe_decode_unpad.py)."""
    _quant_op.cute_dsl_moe_quantize_nvfp4(
        x, m.g1, core._moe_xq[:rows], core._moe_xsf[:rows], core._moe_k.quant[rows]
    )
    acc = core._moe_begin_layer(rows)
    slot = core._moe_slot
    main = torch.cuda.current_stream()
    side = core._side_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        core._copy_into(acc, partial)
    out = core._routed(x, m, rows, True, side, True, True, LAYER)
    torch.cuda.synchronize()
    s = core._moe_slots[slot]
    padded = _pow2(rows)
    if core._moe_dec_unpad or padded == rows:
        t = s.topk_out[rows]
        w, maps, n = t["topk_weights"].view(rows, TOP_K), t, rows
    else:
        w, maps, n = s.w[:padded], s.maps, padded
    snap = {k: maps[k].clone() for k in _routing_op.MAP_KEYS}
    return out.clone(), w[:n].clone(), snap, n


def _reference(core, x, partial, m, rows, w, maps, n):
    _quant_op.cute_dsl_moe_quantize_nvfp4(
        x, m.g1, core._moe_xq[:rows], core._moe_xsf[:rows], core._moe_k.quant[rows]
    )
    xq, xsf = core._moe_xq[:rows].clone(), core._moe_xsf[:rows].clone()
    maps = dict(maps)
    maps["expanded_idx_to_permuted_idx"] = maps["expanded_idx_to_permuted_idx"][: rows * TOP_K]
    return _experts_op.cute_dsl_moe_nvfp4_experts.reference(
        xq, xsf, maps, w[:rows].contiguous(), m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s, m.norm_const,
        m.alpha1, m.alpha2, partial.clone(), None, None, None, core._moe_k.experts,
    )[:rows]  # fmt: skip


def _mapped(maps, n_exp: int) -> torch.Tensor:
    """Per expanded index (token * 8 + k), 1 where this rank's maps hold it."""
    tiles = int(maps["num_non_exiting_tiles"][0])
    mn = maps["tile_idx_to_mn_limit"][:tiles].long()
    hit = torch.zeros(n_exp, dtype=torch.int32, device=mn.device)
    p2e = maps["permuted_idx_to_expanded_idx"].long()
    for t in range(tiles):
        e = p2e[t * 128 : int(mn[t])]
        hit[e[(e >= 0) & (e < n_exp)]] += 1
    return hit


@pytest.mark.parametrize("rows", ROWS)
def test_four_rank_sum_replicas_unpadded_vs_padded(rows: int, monkeypatch) -> None:
    st = state()
    core = st.core
    g = torch.Generator(device="cuda").manual_seed(_seed(f"rep x {rows}"))
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    partial = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    hot = _hot_ids(st, x)
    hot_t = torch.tensor(hot, dtype=torch.int32, device="cuda")
    core._dec_hot_ids = {LAYER: hot_t}
    sums, refs, rep_tiles = {}, {}, 0
    for unpad in (True, False):
        monkeypatch.setattr(core, "_moe_dec_unpad", unpad)
        total = torch.zeros(rows, HIDDEN, dtype=torch.float32, device="cuda")
        ref = torch.zeros_like(total)
        cover = torch.zeros(rows * TOP_K, dtype=torch.int32, device="cuda")
        for r, off in enumerate(OFFSETS):
            core.expert_offset = off
            m = _rank_weights(st.gw, st.router, st.bias, off, hot)
            p = partial if r == 0 else torch.zeros_like(partial)
            out, w, maps, n = _rank_step(core, x, p, m, rows)
            assert bool(torch.isfinite(out).all())
            total += out.float()
            ref += _reference(core, x, p, m, rows, w, maps, n).float()
            cover += _mapped(maps, rows * TOP_K)
            if unpad:
                tiles = int(maps["num_non_exiting_tiles"][0])
                rep_tiles += int((maps["tile_idx_to_expert_idx"][:tiles] >= LOCAL).sum())
            del m
        # every (token, k) pair on exactly one rank
        assert bool((cover == 1).all()), (
            f"unpad={unpad}: pairs mapped {cover.unique().tolist()} times"
        )
        sums[unpad], refs[unpad] = total, ref
    assert rep_tiles > 0, "no replica slot took a tile: the test does not reach the replica maps"
    for unpad in (True, False):
        assert_within_ulp(
            sums[unpad].to(torch.bfloat16), refs[unpad].to(torch.bfloat16),
            element_ulp=64.0, rms_ulp=4.0,
        )  # fmt: skip
    assert_within_ulp(
        sums[True].to(torch.bfloat16), sums[False].to(torch.bfloat16), element_ulp=64.0, rms_ulp=4.0
    )
    d = (sums[True] - sums[False]).abs().amax() / sums[False].abs().amax()
    rms = (sums[True] - sums[False]).pow(2).mean().sqrt() / sums[False].pow(2).mean().sqrt()
    print(
        f"UNPAD_REP rows={rows} hot={hot} rep_tiles={rep_tiles} "
        f"rep-vs-home max_rel={float(d):.3e} rel_rms={float(rms):.3e}"
    )
