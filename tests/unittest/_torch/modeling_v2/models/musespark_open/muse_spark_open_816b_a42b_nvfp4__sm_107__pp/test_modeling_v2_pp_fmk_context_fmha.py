# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The multi-instance context FMHA at a pipeline stage's 128 query / 16 KV heads.

The PP stage target installs the TEP target's ``FmkPureContextFmha`` on a stage's
full-attention layers, which run all of the model's heads (TP1); its own ``stage_library``
plans, checks, warms up and installs it at those heads. Claims checked here:

* (CPU) ``StagePlans`` plans exactly what the planner plans at 128 / 16 with the TEP build's
  instances and key tile, byte for byte, and reuses a plan for the same lengths;
* (CPU) the served call (``_run_fmk``, preprocess and kernel replaced by recorders) hands the
  kernel the stage's plan, and the stage warm-up compiles and plans at 128 / 16;
* (CPU) ``stage_refusal`` refuses exactly what the TEP library's pure context check refuses
  when both serve 32 / 4 (a drift guard over synthetic defects), and at its default heads
  serves 128 / 16 only; ``prepare`` checks every layer and builds before any layer changes,
  and ``commit`` puts the library before each full-attention layer's thop fallback;
* (CPU) at 128 / 16 the plan covers every key tile of every (request, 128-row block, head
  group) exactly once, for all 32 head groups, cuts only before an item's first diagonal
  tile, and stays within ``max_slots``;
* (CPU) executing the plan's work units as the kernel documents them (heads ``4 hg + j`` of
  KV head ``4 hg // 8``, key tiles ``[t0, t1)``, causal bottom-right; cut pieces merged by
  their ``(m, l)``) in float64 gives softmax attention over every row and every one of the
  128 heads: an emulation of the unit contract, not of the kernel;
* (CPU) the TEP library's pure context install still refuses anything but 32 / 4 per rank;
* (sm107 GPU) on pure context steps at 128 / 16 (one 8k chunk at depth 0, 61,696 and 85,504,
  two requests, odd lengths; NaN-filled output buffers): the multi-instance kernel served the
  step and wrote every row and head, the K / V cache is
  bitwise the thop op's, and the rows' relative L2 error against an fp32 reference is within
  15 % of the thop op's own (the bound the TEP target's test uses at 32 / 4); the stage
  warm-up runs at 128 / 16.
"""

import math
import types

import numpy as np
import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__pp.kernels.fmk_pure_context_fmha import (  # noqa: E501
    stage_library as stage,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.fmk_context_fmha import (  # noqa: E501
    attention as fmkc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.attention import (  # noqa: E501
    context_fmha_fmk as fmk,
)
from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionRuntimeFeatures,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import torch_dtype_to_binding
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

#: A stage's per-rank heads (the model's, at TP1) and a TEP4 rank's.
STAGE_HEADS = (128, 16)
TEP4_HEADS = (32, 4)
D, P = 64, 32
CPU = torch.device("cpu")

PLAN_MIXES = [
    [(0, 8192)],
    [(61696, 8192)],
    [(85504, 8192)],
    [(560000, 8192)],
    [(0, 300)],
    [(5306, 1)],
    [(5007, 300), (777, 1), (33019, 4097)],
    [(63, 129), (64, 64), (1000, 255), (12345, 1000), (0, 1)],
    [(35232, 5208), (128, 2976)],
    [(15872, 6240), (83416, 970), (598262, 970)],
]


def _ids(mix):
    return f"{len(mix)}req-{mix[0][0]}x{mix[0][1]}"


def _lengths(mix):
    return [q for _, q in mix], [c + q for c, q in mix]


def _plan_bytes(plan):
    """Every host-visible field of a plan, the device buffers as bytes."""
    tensors = [plan.units, plan.cta_off, plan.merge_meta, plan.merge_slots]
    return (
        [None if t is None else t.cpu().numpy().tobytes() for t in tensors],
        (plan.num_units, plan.num_ctas, plan.num_slots, plan.max_pieces, plan.num_inst),
        (plan.tile_n, plan.makespan, plan.ideal),
    )


def _groups(plan):
    return {int(hg) for hg in plan.units[: plan.num_units, 2].tolist()}


def _stage_plan(q_lens, kv_lens, num_ctas):
    """The planner's plan at the stage's heads, with the TEP build's instances and key tile."""
    return fmk.plan_context_fmha(
        q_lens,
        kv_lens,
        num_ctas=num_ctas,
        num_heads=STAGE_HEADS[0],
        num_kv_heads=STAGE_HEADS[1],
        num_inst=fmkc.NUM_INST,
        tile_n=fmkc.TILE_N,
        device=CPU,
    )


# --------------------------------------------------------------------------- plans (CPU)


@pytest.mark.parametrize("mix", PLAN_MIXES, ids=_ids)
@pytest.mark.parametrize("num_ctas", [148, 212])
def test_stage_plans_are_the_planners_at_the_stage_heads(mix, num_ctas):
    q_lens, kv_lens = (tuple(x) for x in _lengths(mix))
    plans = stage.StagePlans(num_ctas)
    plan = plans.get(q_lens, kv_lens, CPU)
    assert _plan_bytes(plan) == _plan_bytes(_stage_plan(q_lens, kv_lens, num_ctas))
    assert _groups(plan) == set(range(STAGE_HEADS[0] // fmkc.NUM_INST))
    assert plans.get(q_lens, kv_lens, CPU) is plan and plans.planned == 1
    longer = tuple(k + 1 for k in kv_lens)
    assert plans.get(q_lens, longer, CPU) is not plan and plans.planned == 2


@pytest.mark.parametrize("mix", PLAN_MIXES, ids=_ids)
@pytest.mark.parametrize("split", [True, False])
def test_the_plan_covers_every_tile_of_every_head_group_once(mix, split):
    q_lens, kv_lens = _lengths(mix)
    heads, kv_heads = STAGE_HEADS
    plan = fmk.plan_context_fmha(
        q_lens,
        kv_lens,
        num_ctas=212,
        num_heads=heads,
        num_kv_heads=kv_heads,
        split=split,
        device=CPU,
    )
    items = fmk._items(q_lens, kv_lens, heads, kv_heads, fmkc.NUM_INST, fmkc.TILE_N)
    assert {hg for _, _, hg, _, _ in items} == set(range(heads // fmkc.NUM_INST))
    cover = {}
    for b, m, hg, t0, t1, mode, slot, _ in plan.units[: plan.num_units].tolist():
        cover.setdefault((b, m, hg), []).append((t0, t1, mode, slot))
    assert len(cover) == len(items)
    cut = 0
    for b, m, hg, t_end, first_masked in items:
        segs = sorted(cover[(b, m, hg)])
        assert segs[0][0] == 0 and segs[-1][1] == t_end
        assert all(x[1] == y[0] for x, y in zip(segs, segs[1:]))
        if len(segs) == 1:
            assert segs[0][2] == 0
            continue
        cut += 1
        assert all(seg[2] == 1 for seg in segs)
        assert all(seg[1] <= first_masked for seg in segs[:-1])
    offsets = plan.cta_off.tolist()
    assert offsets[0] == 0 and offsets[-1] == plan.num_units and offsets == sorted(offsets)
    assert (plan.merge_meta is None and cut == 0) or plan.merge_meta.shape[0] == cut
    assert plan.num_slots <= fmk.max_slots(212)


def _emulate(plan, q, k, v, q_lens, kv_lens, heads, kv_heads):
    """The plan's work units run as the kernel documents them, in float64.

    ``q`` ``[total_q, heads, D]``; ``k`` / ``v`` per request ``[kv_len, kv_heads, D]``.
    Returns ``[total_q, heads, D]``, NaN where no unit (or merge) wrote.
    """
    ni, tile = plan.num_inst, plan.tile_n
    ratio = heads // kv_heads
    out = torch.full_like(q, float("nan"))
    starts = np.cumsum([0] + list(q_lens)).tolist()
    pieces = {}
    for b, m, hg, t0, t1, mode, slot, _ in plan.units[: plan.num_units].tolist():
        q_len, kv_len = q_lens[b], kv_lens[b]
        rows = torch.arange(m * fmk.M_TILE, min(q_len, (m + 1) * fmk.M_TILE))
        keys = torch.arange(t0 * tile, min(t1 * tile, kv_len))
        allowed = keys[None, :] <= (kv_len - q_len + rows)[:, None]
        kvh = (hg * ni) // ratio
        for j in range(ni):
            head = hg * ni + j
            assert head // ratio == kvh  # the instances of a unit share one KV head
            s = q[starts[b] + rows, head] @ k[b][keys, kvh].T / math.sqrt(D)
            s = s.masked_fill(~allowed, float("-inf"))
            mx = s.max(dim=1).values
            p = torch.where(allowed, torch.exp(s - mx[:, None].clamp(min=-1e300)), 0.0)
            l_sum = p.sum(dim=1)
            o = (p @ v[b][keys, kvh]) / l_sum.clamp(min=1e-300)[:, None]
            if mode == 0:
                assert bool(out[starts[b] + rows, head].isnan().all()), "a row written twice"
                out[starts[b] + rows, head] = o
            else:
                pieces.setdefault((b, m, hg, j), []).append((o, mx, l_sum))
    for (b, m, hg, j), parts in pieces.items():
        rows = torch.arange(m * fmk.M_TILE, min(q_lens[b], (m + 1) * fmk.M_TILE))
        top = torch.stack([torch.where(lsum > 0, mx, -math.inf) for _, mx, lsum in parts]).max(
            dim=0
        )
        weights = [
            torch.where(lsum > 0, lsum * torch.exp(mx - top.values), 0.0) for _, mx, lsum in parts
        ]
        den = sum(weights)
        merged = sum(w[:, None] * o for w, (o, _, _) in zip(weights, parts)) / den[:, None]
        out[starts[b] + rows, hg * ni + j] = merged
    return out


def _attention(q, k, v, q_lens, kv_lens, heads, kv_heads):
    """Causal (bottom-right) softmax attention, float64, every row and head."""
    out = torch.empty_like(q)
    ratio = heads // kv_heads
    row0 = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        rows = torch.arange(q_len)
        allowed = torch.arange(kv_len)[None, :] <= (kv_len - q_len + rows)[:, None]
        for head in range(heads):
            s = q[row0 : row0 + q_len, head] @ k[b][:, head // ratio].T / math.sqrt(D)
            p = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=1)
            out[row0 : row0 + q_len, head] = p @ v[b][:, head // ratio]
        row0 += q_len
    return out


EMULATED = {
    "one_shallow": ([(0, 300)], 212),
    "deep_cut": ([(6000, 100)], 48),
    "three": ([(1700, 300), (63, 129), (4000, 1)], 16),
}


@pytest.mark.parametrize("name", list(EMULATED))
def test_the_units_at_the_stage_heads_compute_attention_over_every_head(name):
    mix, num_ctas = EMULATED[name]
    q_lens, kv_lens = _lengths(mix)
    heads, kv_heads = STAGE_HEADS
    plan = stage.StagePlans(num_ctas).get(tuple(q_lens), tuple(kv_lens), CPU)
    gen = torch.Generator().manual_seed(len(name))
    q = torch.randn(sum(q_lens), heads, D, generator=gen, dtype=torch.float64)
    k = [torch.randn(n, kv_heads, D, generator=gen, dtype=torch.float64) for n in kv_lens]
    v = [torch.randn(n, kv_heads, D, generator=gen, dtype=torch.float64) for n in kv_lens]
    got = _emulate(plan, q, k, v, q_lens, kv_lens, heads, kv_heads)
    assert not bool(got.isnan().any()), "a row / head no unit wrote"
    ref = _attention(q, k, v, q_lens, kv_lens, heads, kv_heads)
    assert torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    if name == "deep_cut":
        assert plan.merge_meta is not None  # the case exercises the merge


# --------------------------------------------------------------------------- call sites (CPU)


class _Unset(types.SimpleNamespace):
    """A namespace whose unset fields read as None (fields the stand-in preprocess ignores)."""

    def __getattr__(self, name):
        return None


@pytest.fixture
def kernel_calls(monkeypatch):
    """The kernel replaced by a recorder: each call's (plan, q / out / pool shapes); out zeroed."""
    calls = []

    def record(q, kv_pool, table, cu_q, kv_lens, plan, s1, s2, out, workspace=None, **kw):
        calls.append((plan, tuple(q.shape), tuple(out.shape), tuple(kv_pool.shape)))
        out.zero_()
        return out

    monkeypatch.setattr(fmk, "paged_context_fmha_fmk", record)
    return calls


def _serve(monkeypatch, plans, heads, q_lens, kv_lens):
    """``_run_fmk`` (the served call) on a layer at ``heads`` with ``plans``, the preprocess
    replaced by a recorder; returns the preprocess's arguments."""
    num_heads, num_kv_heads = heads
    rows = sum(q_lens)
    seen = {}

    def preprocess(**kw):
        seen.update(kw)
        q = torch.zeros(rows * num_heads * D, dtype=torch.uint8)
        pool = torch.zeros(4, num_kv_heads, P, D, dtype=torch.float8_e4m3fn)
        scales = torch.ones(2)
        cu_q = torch.tensor([0, rows], dtype=torch.int32)
        table = torch.zeros(1, 2, 4, dtype=torch.int32)
        return q, pool, table, None, scales, scales, None, cu_q

    monkeypatch.setattr(
        fmkc, "thop", types.SimpleNamespace(trtllm_gen_context_preprocess=preprocess)
    )
    monkeypatch.setattr(fmkc, "get_bmm1_scale", lambda attn: 1.0)
    rope = types.SimpleNamespace(dim=D, theta=1.0, scale_type=0, scale=1.0, max_positions=1)
    attn = _Unset(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=D, rope_params=rope)
    params = _Unset(
        attn=attn,
        meta=_Unset(),
        fwd=_Unset(),
        batch_size=len(q_lens),
        num_tokens=rows,
        tokens_per_block=P,
        sequence_lengths=torch.tensor(kv_lens, dtype=torch.int32),
        output=torch.zeros(rows, num_heads * D, dtype=torch.bfloat16),
    )
    phase = types.SimpleNamespace(_plans=plans, _workspace=fmk.FmkWorkspace())
    fmkc._FmkContextPhase._run_fmk(phase, params, q_lens, kv_lens)
    return seen


def test_the_served_call_hands_the_kernel_the_stage_plan(monkeypatch, kernel_calls):
    """With ``StagePlans``, the TEP library's served call hands the kernel a plan of all 32
    head groups at 128 / 16, the planner's plan at the stage's heads."""
    q_lens, kv_lens = (8192,), (61696 + 8192,)
    seen = _serve(monkeypatch, stage.StagePlans(212), STAGE_HEADS, q_lens, kv_lens)
    assert (seen["num_heads"], seen["num_kv_heads"]) == STAGE_HEADS
    ((plan, q_shape, out_shape, pool_shape),) = kernel_calls
    assert q_shape == out_shape == (q_lens[0], STAGE_HEADS[0], D)
    assert pool_shape[1] == STAGE_HEADS[1]
    assert _groups(plan) == set(range(STAGE_HEADS[0] // fmkc.NUM_INST))
    assert _plan_bytes(plan) == _plan_bytes(_stage_plan(q_lens, kv_lens, 212))


def test_the_stage_warm_up_plans_at_the_stage_heads(kernel_calls):
    """``stage_warm_up`` (the install's compile and check) runs at 128 / 16 and its plan cuts
    (the merge runs)."""
    stage.stage_warm_up(CPU, 212, fmk.FmkWorkspace())
    ((plan, q_shape, out_shape, pool_shape),) = kernel_calls
    assert q_shape == out_shape == (1, STAGE_HEADS[0], D) and pool_shape[1] == STAGE_HEADS[1]
    assert _groups(plan) == set(range(STAGE_HEADS[0] // fmkc.NUM_INST))
    assert plan.merge_meta is not None
    assert _plan_bytes(plan) == _plan_bytes(_stage_plan((1,), (400 * P,), 212))


# --------------------------------------------------------------------------- installers (CPU)


class _Fallback:
    """The thop fallback library (stands in for ``FallbackFmha``)."""


class _PureLibrary:
    """``FmkPureContextFmha``: records what ``commit`` built it with."""

    def __init__(self, attn, fallback, plans, workspace):
        self.attn, self.fallback, self.plans, self.workspace = attn, fallback, plans, workspace


#: The stand-in quant mode of an FP8 KV cache (``_fp8_kv_only`` reads it).
FP8_KV = "fp8"


def _fp8_kv_only(mode):
    return types.SimpleNamespace(has_fp8_kv_cache=lambda: mode == FP8_KV)


def _layer(index, sliding, heads=STAGE_HEADS, head_dim=D):
    manager = types.SimpleNamespace(fmha_libs=[_Fallback()], _cache={"stale": 1})
    attn = types.SimpleNamespace(
        num_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim=head_dim,
        is_mla_enable=False,
        quant_mode=FP8_KV,
        _fmha_manager=manager,
    )
    attention = types.SimpleNamespace(layer_idx=index, is_sliding=sliding, attn=attn)
    return types.SimpleNamespace(self_attn=attention)


#: Synthetic defects of a layer at a TEP4 rank's heads: (SM, a change to the layer, whether the
#: binding has the preprocess). The refusal of each is checked against the TEP library's.
DEFECTS = {
    "served": (107, None, True),
    "sm103": (103, None, True),
    "no_device": (None, None, True),
    "no_fallback": (107, lambda attn: attn._fmha_manager.fmha_libs.clear(), True),
    "mla": (107, lambda attn: setattr(attn, "is_mla_enable", True), True),
    "head_dim_128": (107, lambda attn: setattr(attn, "head_dim", 128), True),
    "kv_heads_not_dividing": (107, lambda attn: setattr(attn, "num_heads", 30), True),
    "six_per_kv_head": (107, lambda attn: setattr(attn, "num_heads", 24), True),
    "four_per_kv_head": (107, lambda attn: setattr(attn, "num_kv_heads", 8), True),
    "tep8_heads": (
        107,
        lambda attn: (setattr(attn, "num_heads", 16), setattr(attn, "num_kv_heads", 2)),
        True,
    ),
    "stage_heads": (
        107,
        lambda attn: (setattr(attn, "num_heads", 128), setattr(attn, "num_kv_heads", 16)),
        True,
    ),
    "kv_not_fp8": (107, lambda attn: setattr(attn, "quant_mode", "bf16"), True),
    "no_preprocess": (107, None, False),
    "sm103_and_mla": (103, lambda attn: setattr(attn, "is_mla_enable", True), True),
    "head_dim_and_kv_not_fp8": (
        107,
        lambda attn: (setattr(attn, "head_dim", 128), setattr(attn, "quant_mode", "bf16")),
        True,
    ),
}


@pytest.fixture
def refusal_stand_ins(monkeypatch):
    """The fallback class, the quant mode and the binding, alike for both libraries; returns a
    setter for whether the binding has the preprocess."""
    for module in (fmkc, stage):
        monkeypatch.setattr(module, "FallbackFmha", _Fallback)
        monkeypatch.setattr(module, "QuantMode", _fp8_kv_only)

    def binding(has_preprocess):
        thop = types.SimpleNamespace()
        if has_preprocess:
            thop.trtllm_gen_context_preprocess = None
        for module in (fmkc, stage):
            monkeypatch.setattr(module, "thop", thop)

    return binding


@pytest.mark.parametrize("name", list(DEFECTS))
def test_the_stage_refusal_at_tep4_heads_matches_the_tep_librarys(refusal_stand_ins, name):
    """Drift guard: given a TEP4 rank's heads, ``stage_refusal`` refuses exactly what the TEP
    library's pure context check refuses, with the same reason up to the served heads."""
    sm, change, has_preprocess = DEFECTS[name]
    refusal_stand_ins(has_preprocess)
    attention = _layer(1, False, TEP4_HEADS).self_attn
    if change is not None:
        change(attention.attn)
    tep = fmkc._refusal(attention, sm, pure=True)
    got = stage.stage_refusal(attention, sm, heads=TEP4_HEADS)
    assert (got is None) == (tep is None), (got, tep)
    if got is None:
        assert name == "served"
        return
    per_rank = "KV heads per rank (the "
    assert (per_rank in got) == (per_rank in tep), (got, tep)
    if per_rank in got:
        attn = attention.attn
        assert (
            got.split(" (the ")[0]
            == tep.split(" (the ")[0]
            == (f"{attn.num_heads} query heads over {attn.num_kv_heads} KV heads per rank")
        )
        assert got.endswith("(the stage install serves 32 over 4)")
    else:
        assert got == tep


def test_the_stage_refusal_serves_the_stage_heads_only(refusal_stand_ins):
    refusal_stand_ins(True)
    assert stage.stage_refusal(_layer(1, False).self_attn, 107) is None
    assert stage.stage_refusal(_layer(1, False, TEP4_HEADS).self_attn, 107) == (
        "32 query heads over 4 KV heads per rank (the stage install serves 128 over 16)"
    )
    # The kernel's own constraint holds at any heads: 4 instances per KV head.
    odd = _layer(1, False, (48, 16)).self_attn
    assert stage.stage_refusal(odd, 107, heads=(48, 16)) == "48 query heads over 16 KV heads"


@pytest.fixture
def stage_host(monkeypatch, refusal_stand_ins):
    """The stage library on the host: device queries, the warm-up and the library classes
    answered by stand-ins; returns the CTA counts the workspace was built for."""
    refusal_stand_ins(True)
    built = []

    def stage_prepare(device, num_ctas):
        built.append(num_ctas)
        return fmk.FmkWorkspace()

    monkeypatch.setattr(fmkc, "FmkPureContextFmha", _PureLibrary)
    monkeypatch.setattr(
        fmkc,
        "pure_context_library",
        lambda attn: next(
            (lib for lib in attn._fmha_manager.fmha_libs if isinstance(lib, _PureLibrary)), None
        ),
    )
    monkeypatch.setattr(fmkc, "context_library", lambda attn: None)
    monkeypatch.setattr(stage, "stage_prepare", stage_prepare)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(multi_processor_count=212),
    )
    return built


def test_the_stage_prepare_changes_no_layer_and_commit_installs_before_the_fallback(stage_host):
    layers = [_layer(index, index % 4 != 1) for index in range(15, 31)]
    prepared = stage.prepare(layers, (10, 7))
    assert stage_host == [212]  # the workspace and the warm-up, once
    assert all(len(layer.self_attn.attn._fmha_manager.fmha_libs) == 1 for layer in layers)
    assert [a.layer_idx for a in prepared.chosen] == [17, 21, 25, 29]
    assert isinstance(prepared.plans, stage.StagePlans) and prepared.plans.num_ctas == 212
    assert prepared.num_ctas == 212
    assert prepared.commit() == 4
    for layer in layers:
        libs = layer.self_attn.attn._fmha_manager.fmha_libs
        if layer.self_attn.is_sliding:
            assert len(libs) == 1
            continue
        assert isinstance(libs[0], _PureLibrary) and isinstance(libs[1], _Fallback)
        assert libs[0].fallback is libs[1] and libs[0].plans is prepared.plans
        assert libs[0].workspace is prepared.workspace
        assert layer.self_attn.attn._fmha_manager._cache == {}
    assert prepared.commit() == 4  # idempotent
    assert all(len(layer.self_attn.attn._fmha_manager.fmha_libs) in (1, 2) for layer in layers)


@pytest.mark.parametrize(
    "heads, capability, message",
    [
        (TEP4_HEADS, (10, 7), r"layer 17: 32 query heads over 4 KV heads per rank \(the stage "),
        (STAGE_HEADS, (10, 3), "layer 17: the kernel is written for sm107, not sm103"),
    ],
    ids=["tep4_heads", "sm103"],
)
def test_the_stage_prepare_refuses_before_it_builds(stage_host, heads, capability, message):
    layers = [_layer(16, True), _layer(17, False, heads)]
    with pytest.raises(ValueError, match=message):
        stage.prepare(layers, capability)
    assert stage_host == []
    assert len(layers[1].self_attn.attn._fmha_manager.fmha_libs) == 1


def test_the_stage_prepare_refuses_a_layer_with_the_mixed_step_library(stage_host, monkeypatch):
    monkeypatch.setattr(fmkc, "context_library", lambda attn: object())
    with pytest.raises(ValueError, match="layer 17: fmk_context_fmha is installed on it"):
        stage.prepare([_layer(17, False)], (10, 7))
    assert stage_host == []


@pytest.fixture
def host(monkeypatch):
    """The TEP library on the host: device queries, the warm-up and the library classes
    answered by stand-ins."""
    prepared = []

    def prepare(device, num_heads, num_kv_heads, num_ctas):
        prepared.append((num_heads, num_kv_heads, num_ctas))
        return fmk.FmkWorkspace()

    monkeypatch.setattr(fmkc, "FallbackFmha", _Fallback)
    monkeypatch.setattr(fmkc, "FmkPureContextFmha", _PureLibrary)
    monkeypatch.setattr(
        fmkc,
        "pure_context_library",
        lambda attn: next(
            (lib for lib in attn._fmha_manager.fmha_libs if isinstance(lib, _PureLibrary)), None
        ),
    )
    monkeypatch.setattr(fmkc, "context_library", lambda attn: None)
    monkeypatch.setattr(fmkc, "_prepare", prepare)
    monkeypatch.setattr(
        fmkc, "QuantMode", lambda mode: types.SimpleNamespace(has_fp8_kv_cache=lambda: True)
    )
    monkeypatch.setattr(fmkc, "thop", types.SimpleNamespace(trtllm_gen_context_preprocess=None))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(multi_processor_count=212),
    )
    return prepared


def test_the_tep_install_still_refuses_anything_but_tep4_heads(host):
    model = types.SimpleNamespace(layers=[_layer(1, False), _layer(2, True)])
    with pytest.raises(ValueError, match="128 query heads over 16 KV heads per rank .*32 over 4"):
        fmkc.install_pure_context(model, (10, 7))
    assert fmkc._refusal(_layer(1, False).self_attn, 107, pure=True) is not None
    assert fmkc._refusal(_layer(1, False, TEP4_HEADS).self_attn, 107, pure=True) is None
    assert host == []


# --------------------------------------------------------------------------- sm107 GPU


def _sm() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


needs_sm107 = pytest.mark.skipif(_sm() != fmkc.SERVED_SM, reason="the kernel is written for sm107")


class _PureStep:
    """A pure context step on a real TrtllmAttention at the stage's heads and a
    ``KVCacheManagerV2`` (layer 0), with an ``FmkPureContextFmha`` on the stage's plans placed
    by ``run``."""

    def __init__(self, contexts, seed):
        dev = torch.device("cuda")
        self.dev = dev
        self.h, self.g = STAGE_HEADS
        self.contexts = contexts
        self.seq_lens = [q for _, q in contexts]
        self.cached = [c for c, _ in contexts]
        self.token_nums = [c + s for c, s in zip(self.cached, self.seq_lens)]
        self.request_ids = list(range(len(contexts)))
        pages_per_seq = max(-(-t // P) for t in self.token_nums)
        self.mgr = KVCacheManagerV2(
            KvCacheConfig(max_tokens=len(contexts) * pages_per_seq * P),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=self.g,
            head_dim=D,
            tokens_per_block=P,
            max_seq_len=pages_per_seq * P,
            max_batch_size=len(contexts),
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=torch_dtype_to_binding(torch.float8_e4m3fn),
        )
        self.mgr.add_dummy_requests(self.request_ids, self.token_nums)
        gen = torch.Generator(device=dev).manual_seed(seed)
        self.cached_k = [
            torch.randn(c, self.g, D, generator=gen, device=dev).to(torch.float8_e4m3fn)
            for c in self.cached
        ]
        self.cached_v = [
            torch.randn(c, self.g, D, generator=gen, device=dev).to(torch.float8_e4m3fn)
            for c in self.cached
        ]
        tokens = sum(self.seq_lens)
        self.qkv = torch.randn(tokens, (self.h + 2 * self.g) * D, generator=gen, device=dev).to(
            torch.bfloat16
        )
        self.buf = self._fill_cache()
        self.attn = create_attention(
            "TRTLLM",
            layer_idx=0,
            num_heads=self.h,
            head_dim=D,
            num_kv_heads=self.g,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
            q_scaling=1.0,
        )
        libs = self.attn._fmha_manager.fmha_libs
        self.fallback = next(f for f in libs if isinstance(f, FallbackFmha))
        num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
        workspace = fmk.FmkWorkspace()
        workspace.reserve(fmk.max_slots(num_ctas), fmkc.NUM_INST, dev)
        self.pure_lib = fmkc.FmkPureContextFmha(
            self.attn, self.fallback, stage.StagePlans(num_ctas), workspace
        )

    def _fill_cache(self):
        """Write each request's cached prefix (``[c, G, D]`` E4M3) into its pages (HND)."""
        buf = self.mgr.get_buffers(0, kv_layout="HND")  # [pages, 2, G, P, D]
        raw = buf.view(torch.uint8)
        raw.zero_()
        blocks = self.mgr.get_batch_cache_indices(list(self.request_ids), 0)
        for i, ids in enumerate(blocks):
            c = self.cached_k[i].shape[0]
            if c == 0:
                continue
            n = -(-c // P)
            pages = torch.tensor([b for b in ids if b != -1][:n], dtype=torch.long, device=self.dev)
            for kv, src in ((0, self.cached_k[i]), (1, self.cached_v[i])):
                pad = torch.zeros(n * P, self.g, D, dtype=torch.uint8, device=self.dev)
                pad[:c] = src.view(torch.uint8)
                raw[pages, kv] = pad.view(n, P, self.g, D).transpose(1, 2)
        return buf

    def cache_tokens(self, page_ids, kv: int, n_pages: int):
        """``[n_pages * P, G, D]`` E4M3 tokens of one request's K (0) or V (1) pages."""
        pages = torch.tensor(page_ids[:n_pages], dtype=torch.long, device=self.dev)
        raw = self.buf.view(torch.uint8)[pages, kv]  # [n, G, P, D]
        return raw.transpose(1, 2).reshape(n_pages * P, self.g, D).view(torch.float8_e4m3fn)

    def reference(self, q_e4m3, k_all, v_all, rows, heads, kv_len, q_len):
        """fp32 attention of the sampled rows / heads of one request (bottom-right causal)."""
        out = torch.empty(len(rows), len(heads), D, dtype=torch.float32, device=self.dev)
        off = kv_len - q_len
        k = k_all[:kv_len].float()
        v = v_all[:kv_len].float()
        ratio = self.h // self.g
        for a, r in enumerate(rows):
            lim = off + r + 1
            for b_, h in enumerate(heads):
                s = (k[:lim, h // ratio] @ q_e4m3[r, h].float()) / math.sqrt(D)
                out[a, b_] = torch.softmax(s, dim=0) @ v[:lim, h // ratio]
        return out

    def metadata(self):
        md = get_attention_backend("TRTLLM").Metadata(
            num_contexts=len(self.contexts),
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=self.cached),
            seq_lens=torch.tensor(self.seq_lens, dtype=torch.int),
            max_num_requests=len(self.seq_lens),
            max_num_tokens=max(8192, sum(self.seq_lens)),
            kv_cache_manager=self.mgr,
            request_ids=self.request_ids,
            prompt_lens=list(self.seq_lens),
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True, cache_reuse=True),
        )
        md.prepare()
        return md

    def run(self, lib):
        """The step with ``lib`` just before the fallback, or with none (the thop op).

        The output buffer is the caller's and NaN-filled, so a head or row the call did not
        write stays NaN (a buffer from the caching allocator could hold the previous run's
        output, which would hide it)."""
        libs = self.attn._fmha_manager.fmha_libs
        libs[:] = [f for f in libs if not isinstance(f, fmkc.FmkPureContextFmha)]
        if lib is not None:
            libs.insert(libs.index(self.fallback), lib)
        self.attn._fmha_manager._cache.clear()
        md = self.metadata()
        assert fmkc.takes_pure_context(md)
        tokens = self.qkv.shape[0]
        output = torch.full(
            (tokens, self.h * D), float("nan"), dtype=torch.bfloat16, device=self.dev
        )
        out = self.attn.forward(
            self.qkv.clone(),
            None,
            None,
            md,
            forward_args=AttentionForwardArgs(
                attention_mask=PredefinedAttentionMask.CAUSAL,
                attention_input_type=AttentionInputType.mixed,
                output=output,
            ),
        )
        out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
        return out[: sum(self.seq_lens)].clone()

    def cache_bytes(self):
        return self.buf.view(torch.uint8).clone()

    def close(self):
        self.mgr.shutdown()


#: Pure context steps of a PP4 context stage: one 8k chunk at served depths, two
#: requests, odd lengths.
STAGE_CASES = {
    "chunk_depth0": [(0, 8192)],
    "chunk_depth61696": [(61696, 8192)],
    "chunk_depth85504": [(85504, 8192)],
    "two": [(35232, 5208), (128, 2976)],
    "odd": [(63, 129), (64, 64), (1000, 255), (12345, 1000)],
}

#: Sampled heads: the first and last of several KV groups, across all 16.
SAMPLED_HEADS = [0, 7, 8, 31, 32, 63, 64, 95, 120, 127]


@needs_sm107
@pytest.mark.parametrize("name", list(STAGE_CASES))
def test_pure_context_rows_at_the_stage_heads(name):
    contexts = STAGE_CASES[name]
    step = _PureStep(contexts, seed=31 + len(name))
    try:
        ctl = step.run(None)
        ctl_cache = step.cache_bytes()
        got = step.run(step.pure_lib)
        got_cache = step.cache_bytes()
        assert step.pure_lib.fmk_calls == 1 and step.pure_lib.declined_calls == 0
        assert torch.equal(ctl_cache, got_cache), "K / V appended differently"
        # Both outputs start NaN: every row and head was written by its call.
        assert bool(torch.isfinite(ctl).all()) and bool(torch.isfinite(got).all())
        rows = sum(q for _, q in contexts)
        h = step.h
        blocks = step.mgr.get_batch_cache_indices(step.request_ids, 0)
        q_e4m3 = step.qkv[:rows, : h * D].view(rows, h, D).to(torch.float8_e4m3fn)
        e_ctl = e_got = r2 = 0.0
        row0 = 0
        for b, (c, q) in enumerate(contexts):
            kv_len = c + q
            n = -(-kv_len // P)
            ids = [x for x in blocks[b] if x != -1]
            k_all = step.cache_tokens(ids, 0, n)
            v_all = step.cache_tokens(ids, 1, n)
            sampled = sorted({0, q // 3, q // 2, q - 1, max(0, q - 2)})
            ref = step.reference(
                q_e4m3[row0 : row0 + q], k_all, v_all, sampled, SAMPLED_HEADS, kv_len, q
            )
            idx = torch.tensor(sampled, device=step.dev) + row0
            a = ctl[idx].view(len(sampled), h, D)[:, SAMPLED_HEADS].float()
            g = got[idx].view(len(sampled), h, D)[:, SAMPLED_HEADS].float()
            e_ctl += ((a - ref) ** 2).sum().item()
            e_got += ((g - ref) ** 2).sum().item()
            r2 += (ref**2).sum().item()
            row0 += q
        rel_ctl, rel_got = math.sqrt(e_ctl / r2), math.sqrt(e_got / r2)
        diff = (got.float() - ctl.float()).abs()
        print(
            f"{name}: rel L2 vs fp32 thop {rel_ctl:.4e} fmk {rel_got:.4e}; "
            f"|fmk - thop| max {diff.max().item():.3e} mean {diff.mean().item():.3e}"
        )
        assert rel_got <= 1.15 * rel_ctl + 1e-3, (rel_got, rel_ctl)
    finally:
        step.close()


@needs_sm107
def test_the_warm_up_runs_at_the_stage_heads():
    dev = torch.device("cuda", torch.cuda.current_device())
    num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
    workspace = stage.stage_prepare(dev, num_ctas)
    assert workspace.nbytes > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
