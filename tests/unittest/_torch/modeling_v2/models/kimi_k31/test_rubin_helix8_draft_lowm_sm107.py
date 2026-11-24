# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter under ``draft_lowm_gemm``, ``draft_conv_norm``, ``draft_conv_pdl``,
``draft_q_a_kv_a_fused``, ``draft_norm_ctas`` and ``draft_window_mla``, on one sm_107 GPU.

``draft_step`` runs the drafter's five layers at helix8's rank shapes (96 heads, block 6, the o_proj and MLP
row-parallel over 8 ranks) with random BF16 weights, once with both levers off (today's kernels) and once with
the levers on; the window, the helix exchange and the all-reduces are deterministic single-GPU stand-ins, the
same in both runs, and every other kernel is the target's:

* ``draft_conv_norm`` with the bit-identical plan (``K31_LOW_M_DRAFT_PLAN_HELIX8``): the drafter's output equals
  today's bit for bit at every helix8 row count (b1 .. b12, k5), eagerly and in CUDA graph replay;
* the history projection (``precompute_context_kv``'s GEMM) equals F.linear bit for bit at the same rows;
* the D-class plan (``K31_LOW_M_DRAFT_PLAN_HELIX8_D``): finite and within a band of today's output (another FP32
  summation order of the same BF16 products, through five layers);
* ``draft_conv_pdl``: today's output bit for bit at every row count, eagerly and in replay;
* ``draft_q_a_kv_a_fused`` (one GEMM over the stacked ``[q_a ; kv_a]``): finite and within the band of today's
  output; on top of it the bit-identical plan + ``draft_conv_norm`` give its own output bit for bit;
* ``draft_norm_ctas`` 7 under ``draft_conv_norm``: finite and within the band (the norm's sum in another order).
* ``draft_window_mla`` alone swaps only the window, and only at the row counts of ``draft_window_mla_bands``:
  with the CuTe DSL window's stand-in equal to the Triton window's, the drafter's output is today's word for word
  at every row count (the GEMM pins and every other kernel unchanged), the CuTe DSL route runs exactly in the
  bands and the window plan exactly outside them.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._utils import get_sm_version

_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix8.modeling"
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the helix8 drafter's low-M plans were measured on VR200 (SM 10.7)",
)

H, Q_LORA, KV_LORA, ROPE, NOPE, V = 7168, 1536, 512, 64, 128, 128
HEADS, LOCAL_HEADS, INTER, LAYERS, BLOCK, TAPS, GROUP = 96, 12, 14336 // 8, 5, 6, 2, 16
ROWS = tuple(6 * b for b in range(1, 13))
#: The D-class plan's band: max |diff| over max |today| of the drafter's output after five layers.
_D_BAND = 2.0**-4
#: draft_q_a_kv_a_bands of the B-class bundle: the rows where cuBLAS's [q_a ; kv_a] GEMM gives the two
#: GEMMs' words on VR200 (6 / 12 rows take another K split).
_B_BANDS = "18-78"


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(_MODULE)


def _weights(mod, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def w(*shape, scale=None):
        k = shape[-1]
        return (torch.randn(*shape, generator=g, device="cuda") * (scale or k**-0.5)).to(
            torch.bfloat16
        )

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g, device="cuda")).to(torch.bfloat16)

    kv_a = w(LAYERS * (KV_LORA + ROPE), H)
    layers = []
    for j in range(LAYERS):
        kv_b = w(HEADS * 2 * NOPE, KV_LORA)
        layers.append(
            mod._DraftLayerW(
                in_norm=norm(H),
                post_norm=norm(H),
                q_a=w(Q_LORA, H),
                q_a_norm=norm(Q_LORA),
                q_b=w(HEADS * (NOPE + ROPE), Q_LORA),
                kv_a=kv_a[j * (KV_LORA + ROPE) : (j + 1) * (KV_LORA + ROPE)],
                kv_a_norm=norm(KV_LORA),
                k_b=kv_b[: HEADS * NOPE].view(HEADS, NOPE, KV_LORA),
                v_b_t=kv_b[HEADS * NOPE :].view(HEADS, -1, KV_LORA).transpose(1, 2),
                o=w(H, LOCAL_HEADS * V),
                gate_up=w(2 * INTER, H),
                down=w(H, INTER),
                attn_conv_base=w(2, TAPS, H, scale=0.3),
                attn_conv_proj=w(2 * TAPS * (H // GROUP), H, scale=0.1 * H**-0.5),
                mlp_conv_base=w(2, TAPS, H, scale=0.3),
                mlp_conv_proj=w(2 * TAPS * (H // GROUP), H, scale=0.1 * H**-0.5),
            )
        )
    return mod._DraftW(fc=None, layers=tuple(layers), final_norm=norm(H)), kv_a


def _fused(mod, DW):
    """``DW`` with each layer's ``[q_a ; kv_a]`` stacked as the table stacks it (q_a a view of it)."""
    layers = []
    for L in DW.layers:
        q_a_kv_a = torch.cat((L.q_a, L.kv_a)).contiguous()
        layers.append(L._replace(q_a=q_a_kv_a[:Q_LORA], q_a_kv_a=q_a_kv_a))
    return DW._replace(layers=tuple(layers))


def _stand_ins(monkeypatch, mod):
    """The window, the helix exchange and the all-reduce on one GPU: pure functions of their inputs."""

    def window(query, pool, block_kv, plan, scale, kv_lora, kv_scale, workspace, stats, **_):
        return (query[..., :kv_lora].float() * 0.5).to(query.dtype).contiguous()

    def combine(partial, stats, mapping, heads, v_dim, zero_kv_mask=None):
        return partial[:, :LOCAL_HEADS].reshape(partial.shape[0], LOCAL_HEADS * v_dim).contiguous()

    monkeypatch.setattr(mod, "rubin_dflash2_window_plan", lambda *a, **k: None)
    monkeypatch.setattr(mod, "rubin_dflash2_window_triton", window)
    monkeypatch.setattr(mod, "window_vec_io_aligned", lambda *a, **k: False)
    monkeypatch.setattr(mod, "rubin_dflash2_helix_combine", combine)
    monkeypatch.setattr(mod, "allreduce_module", lambda x, *a, **k: x.clone())


def _step(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = rows // BLOCK
    num_ctx = torch.randint(1000, 100000, (batch,), generator=g, device="cuda")
    positions = (num_ctx[:, None] + torch.arange(BLOCK, device="cuda")).reshape(-1)
    freqs = torch.polar(
        torch.ones(200_000, ROPE // 2, device="cuda"),
        torch.rand(200_000, ROPE // 2, generator=g, device="cuda") * 6.283,
    )
    d = SimpleNamespace(
        eps=1e-6,
        taps=TAPS,
        groups=H // GROUP,
        group=GROUP,
        freqs=freqs,
        softmax_scale=(NOPE + ROPE) ** -0.5,
        kv_scale=1.0,
        mapping=None,
        cp_size=8,
        cp_rank=0,
        plan=None,
        workspace=None,
        block_kv=torch.zeros(rows, 1, KV_LORA + ROPE, dtype=torch.float8_e4m3fn, device="cuda"),
        block_slots=torch.arange(rows, device="cuda"),
        block_offsets=torch.zeros(rows, dtype=torch.int64, device="cuda"),
        stats=torch.zeros(rows, HEADS, 2, device="cuda"),
        mla={},
    )
    st = SimpleNamespace(
        block=BLOCK,
        rows=rows,
        num_ctx=num_ctx,
        page_table=None,
        table_rows=None,
        page_size=64,
        positions=positions,
        pools=[torch.zeros(1, 1, 1, 64, KV_LORA + ROPE, dtype=torch.float8_e4m3fn, device="cuda")]
        * LAYERS,
    )
    noise = torch.randn(rows, H, generator=g, device="cuda").to(torch.bfloat16)
    return SimpleNamespace(draft=d, all_reduce=None), st, noise


def _run(
    mod,
    monkeypatch,
    DW,
    rows,
    *,
    plan,
    conv_norm,
    conv_pdl=False,
    ctas=0,
    mla=False,
    bands=((1, 1024),),
    seed=7,
):
    rt, st, noise = _step(rows, seed)
    monkeypatch.setattr(mod, "_DRAFT_Q_A_KV_A_BANDS", bands)
    monkeypatch.setattr(mod, "_DRAFT_LOWM_GEMM", plan is not None)
    monkeypatch.setattr(mod, "_DRAFT_LOWM_PLAN", plan)
    monkeypatch.setattr(mod, "_DRAFT_CONV_NORM", conv_norm)
    monkeypatch.setattr(mod, "_DRAFT_CONV_PDL", conv_pdl)
    monkeypatch.setattr(mod, "_DRAFT_NORM_CTAS", ctas)
    monkeypatch.setattr(mod, "_DRAFT_WINDOW_MLA", mla)
    return mod.draft_step(DW, noise, st, rt), (rt, st, noise)


def _band(lever, today):
    return float((lever.float() - today.float()).abs().max() / today.float().abs().max())


def _words(t):
    return t.contiguous().view(torch.int16)


@torch.inference_mode()
def test_b_class_levers_keep_the_drafter_bits(mod, monkeypatch):
    """B plan + draft_conv_norm: the drafter's output is today's word for word at every helix8 row count."""
    _stand_ins(monkeypatch, mod)
    DW, _ = _weights(mod, seed=1)
    plan = mod.K31_LOW_M_DRAFT_PLAN_HELIX8
    for rows in ROWS:
        today, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False)
        lever, _ = _run(mod, monkeypatch, DW, rows, plan=plan, conv_norm=True)
        assert torch.equal(_words(lever), _words(today)), f"{rows} rows"
        conv_only, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=True)
        assert torch.equal(_words(conv_only), _words(today)), f"{rows} rows, draft_conv_norm alone"


@torch.inference_mode()
def test_b_class_levers_replay_in_a_graph(mod, monkeypatch):
    """The lever path captured in a CUDA graph (compiled eagerly first) replays its eager words."""
    _stand_ins(monkeypatch, mod)
    DW, _ = _weights(mod, seed=2)
    plan = mod.K31_LOW_M_DRAFT_PLAN_HELIX8
    for rows in (24, 54):
        eager, (rt, st, noise) = _run(mod, monkeypatch, DW, rows, plan=plan, conv_norm=True)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
            out = mod.draft_step(DW, noise, st, rt)
        torch.cuda.current_stream().wait_stream(s)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_words(out), _words(eager)), f"{rows} rows"


@torch.inference_mode()
def test_b_class_history_projection_keeps_the_bits(mod, monkeypatch):
    """``precompute_context_kv``'s GEMM (``_draft_linear`` over the five layers' stacked kv_a) is F.linear's."""
    _, kv_a = _weights(mod, seed=3)
    monkeypatch.setattr(mod, "_DRAFT_LOWM_PLAN", mod.K31_LOW_M_DRAFT_PLAN_HELIX8)
    g = torch.Generator(device="cuda").manual_seed(4)
    for rows in ROWS:
        for _ in range(3):
            x = torch.randn(rows, H, generator=g, device="cuda").to(torch.bfloat16)
            got = mod._draft_linear(x, kv_a, BLOCK)
            assert torch.equal(_words(got), _words(F.linear(x, kv_a))), rows


@torch.inference_mode()
def test_window_mla_swaps_only_the_window_up_to_its_rows(mod, monkeypatch):
    """``draft_window_mla`` with the GEMM and norm levers off: ``draft_step`` runs the CuTe DSL window (a stand-in
    returning the Triton stand-in's output) on every layer at the row counts of ``draft_window_mla_bands`` and
    plans no Triton window there; at any other row count it plans and runs the Triton window and makes no CuTe DSL
    buffers. The drafter's output is today's word for word at every row count: the route changes the window
    only."""
    _stand_ins(monkeypatch, mod)
    routes, plans, made = [], [], []

    def mla_window(query, pool, table, table_rows, num_ctx, block_kv, block, scale, cp, rank, ws):
        routes.append((query.shape[0], ws.rows))
        rows, heads = query.shape[:2]
        out = (query[..., :KV_LORA].float() * 0.5).to(query.dtype).contiguous()
        stats = torch.zeros(rows, heads, 2, device=query.device)
        return out, stats, torch.zeros(rows, dtype=torch.bool, device=query.device)

    monkeypatch.setattr(mod, "rubin_ull_dflash2_helix_window", mla_window)
    monkeypatch.setattr(mod, "rubin_dflash2_window_plan", lambda *a, **k: plans.append(a[3]))
    monkeypatch.setattr(
        mod,
        "helix_window_workspace",
        lambda rows, *a: made.append(rows) or SimpleNamespace(rows=rows),
    )
    DW, _ = _weights(mod, seed=6)
    bands = mod._DRAFT_WINDOW_MLA_BANDS
    assert any(mod._moe_in_bands(bands, r) for r in ROWS)
    assert not all(mod._moe_in_bands(bands, r) for r in ROWS)
    for rows in ROWS:
        today, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False)
        routes.clear(), plans.clear(), made.clear()
        lever, (rt, _, _) = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False, mla=True)
        assert torch.equal(_words(lever), _words(today)), f"{rows} rows"
        if mod._moe_in_bands(bands, rows):
            assert routes == [(rows, rows)] * LAYERS and plans == [] and made == [rows], rows
            assert sorted(rt.draft.mla) == [rows]
        else:
            assert routes == [] and plans == [BLOCK] and made == [] and rt.draft.mla == {}, rows


@torch.inference_mode()
def test_d_class_plan_stays_in_band(mod, monkeypatch):
    """D plan + draft_conv_norm: finite, and within the band of today's output after five layers."""
    _stand_ins(monkeypatch, mod)
    DW, _ = _weights(mod, seed=5)
    plan = mod.K31_LOW_M_DRAFT_PLAN_HELIX8_D
    for rows in ROWS:
        today, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False)
        lever, _ = _run(mod, monkeypatch, DW, rows, plan=plan, conv_norm=True)
        assert torch.isfinite(lever).all(), rows
        err = (lever.float() - today.float()).abs().max() / today.float().abs().max()
        assert err <= _D_BAND, f"{rows} rows: {float(err):.3e}"


@torch.inference_mode()
def test_conv_pdl_keeps_the_drafter_bits(mod, monkeypatch):
    """draft_conv_pdl: the four convolutions a layer with PDL; today's output word for word, also in replay."""
    _stand_ins(monkeypatch, mod)
    DW, _ = _weights(mod, seed=6)
    for rows in ROWS:
        today, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False)
        lever, (rt, st, noise) = _run(
            mod, monkeypatch, DW, rows, plan=None, conv_norm=False, conv_pdl=True
        )
        assert torch.equal(_words(lever), _words(today)), f"{rows} rows"
        if rows in (24, 54):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
                out = mod.draft_step(DW, noise, st, rt)
            torch.cuda.current_stream().wait_stream(s)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(_words(out), _words(today)), f"{rows} rows, replay"


@torch.inference_mode()
def test_q_a_kv_a_fused_stays_in_band_and_the_b_plan_keeps_its_bits(mod, monkeypatch):
    """draft_q_a_kv_a_fused: finite and within the band of today's output at every row count; the B plan +
    draft_conv_norm on top of it give its own output word for word."""
    _stand_ins(monkeypatch, mod)
    DW, _ = _weights(mod, seed=8)
    QW = _fused(mod, DW)
    plan = mod.K31_LOW_M_DRAFT_PLAN_HELIX8
    for rows in ROWS:
        today, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False)
        fused, _ = _run(mod, monkeypatch, QW, rows, plan=None, conv_norm=False)
        assert torch.isfinite(fused).all(), rows
        assert _band(fused, today) <= _D_BAND, f"{rows} rows: {_band(fused, today):.3e}"
        lever, _ = _run(mod, monkeypatch, QW, rows, plan=plan, conv_norm=True)
        assert torch.equal(_words(lever), _words(fused)), f"{rows} rows, B plan on the stack"


@torch.inference_mode()
def test_the_b_bundle_keeps_the_drafter_bits(mod, monkeypatch):
    """draft_q_a_kv_a_fused at the B-class bands (18-78 rows: where cuBLAS's one GEMM gives the two GEMMs'
    words) + the bit-identical plan + draft_conv_norm: today's output word for word at every helix8 row count."""
    _stand_ins(monkeypatch, mod)
    DW, _ = _weights(mod, seed=10)
    QW = _fused(mod, DW)
    plan = mod.K31_LOW_M_DRAFT_PLAN_HELIX8
    bands = mod._moe_row_bands(_B_BANDS, 1024)
    for rows in ROWS:
        today, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False)
        lever, _ = _run(mod, monkeypatch, QW, rows, plan=plan, conv_norm=True, bands=bands)
        assert torch.equal(_words(lever), _words(today)), f"{rows} rows"


@torch.inference_mode()
def test_norm_ctas_stays_in_band(mod, monkeypatch):
    """draft_norm_ctas 7 under draft_conv_norm: finite and within the band of today's output."""
    _stand_ins(monkeypatch, mod)
    DW, _ = _weights(mod, seed=9)
    for rows in ROWS:
        today, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=False)
        lever, _ = _run(mod, monkeypatch, DW, rows, plan=None, conv_norm=True, ctas=7)
        assert torch.isfinite(lever).all(), rows
        assert _band(lever, today) <= _D_BAND, f"{rows} rows: {_band(lever, today):.3e}"
