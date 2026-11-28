# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's decode-sized routed experts on routing tables (`ops/moe/dsv4_route_tables.py`).

`dsv4_route_nvfp4_quant_tables` must route and quantize bit for bit as the family's `dsv4_route_nvfp4_quant` (the same
launch, with the tables written on the side). `fp4_moe_runner_tables` must give bit for bit the outputs of the
catalog runner routing itself on the same ids and weights: finalized rows equal, and unfinalized the FC2 rows read
through each one's permutation (the table's head on the tables) equal. Both runs use the same tactic, so any
difference is a wrong table, not rounding.

`dsv4_route_tables` (no quantization) must route as `dsv4_route_nvfp4_quant_tables` and its tables must drive the runner
to the same rows. With dynamic expert parallelism (384 experts, both EP ranks' calls made on one GPU) every pick is on
exactly one rank, each rank runs at most ceil(used / 2) experts, and every pick's FC2 row is bit for bit the row of the
runner over every expert; the CuTe DSL FC2 behind the FC1-only run gives the FC2 cubins' rows bit for bit.
The boundary's NVFP4 copy (`fp4_quantize(h, g, 16, False, False)`, the family boundary's quantizer) has the same rows
and scales as the routing launch's own quantization.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # _validating
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "moe"))  # _fp4_moe_test_utils
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.moe.dsv4_route_tables import (  # noqa: E402,E501
    dsv4_route_nvfp4_quant_tables,
    dsv4_route_tables,
    dynamic_ep_applies,
    fp4_moe_runner_tables,
    routing_table_max_ctas,
    table_permutation,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (  # noqa: E402
    dsv4_route_nvfp4_quant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)
H, TOPK, SCALE = 5120, 6, 2.5
TILES = [8, 16, 32]


def _route_inputs(n, experts, gen):
    x = (torch.randn(n, H, generator=gen, device=DEV) * 2).bfloat16()
    logits = torch.randn(n, experts, generator=gen, device=DEV) * 3
    bias = torch.randn(experts, generator=gen, device=DEV) * 0.5
    g = (448.0 * 6.0 / x.float().abs().max()).reshape(1).to(DEV)
    return logits, bias, x, g


@pytest.mark.parametrize(
    "cell",
    dsv4_route_nvfp4_quant_tables.CELLS,
    ids=[c.why[:48] for c in dsv4_route_nvfp4_quant_tables.CELLS],
)
def test_route_tables_cells(cell, certifying):  # noqa: F811
    certifying(dsv4_route_nvfp4_quant_tables)
    s = cell.spec
    logits, bias, x, g = _route_inputs(s["n"], s["experts"], gen_for(cell))
    ids, w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(
        logits, bias, x, g, TOPK, SCALE, s["first"], s["count"], TILES, s.get("dynamic_rank")
    )
    ids0, w0, xq0, xsf0 = dsv4_route_nvfp4_quant(logits, bias, x, g, TOPK, SCALE)
    for a, b in ((ids, ids0), (w, w0), (xq, xq0), (xsf, xsf0)):
        assert torch.equal(a, b)
    assert tables.dtype == torch.int32 and tables.shape[:2] == (len(TILES), 5)
    # and the routing itself is the entry's reference (the family test's gate, here on the picks only)
    rids = dsv4_route_nvfp4_quant_tables.reference(
        logits, bias, x, g, TOPK, SCALE, s["first"], s["count"], TILES
    )[0]
    assert torch.equal(ids.long().sort(-1).values, rids.long().sort(-1).values)


@pytest.mark.parametrize(
    "cell", dsv4_route_tables.CELLS, ids=[c.why[:48] for c in dsv4_route_tables.CELLS]
)
def test_route_only_cells(cell, certifying):  # noqa: F811
    """The route-only launch routes as the quantizing one (same scope), and as the family reference."""
    certifying(dsv4_route_tables)
    s = cell.spec
    logits, bias, x, g = _route_inputs(s["n"], s["experts"], gen_for(cell))
    dyn = s.get("dynamic_rank")
    ids, w, tables = dsv4_route_tables(
        logits, bias, TOPK, SCALE, H, s["first"], s["count"], TILES, dyn
    )
    ids0, w0, _, _, tables0 = dsv4_route_nvfp4_quant_tables(
        logits, bias, x, g, TOPK, SCALE, s["first"], s["count"], TILES, dyn
    )
    assert torch.equal(ids, ids0) and torch.equal(w, w0)
    assert tables.shape == tables0.shape
    # Row 0 of each tile's table maps every pick to its permuted row (-1: another rank): the same picks are local.
    k = s["n"] * TOPK
    assert torch.equal(tables[:, 0, :k] >= 0, tables0[:, 0, :k] >= 0)
    rids, rw, _ = dsv4_route_tables.reference(
        logits, bias, TOPK, SCALE, H, s["first"], s["count"], TILES, dyn
    )
    assert torch.equal(ids.long().sort(-1).values, rids.long().sort(-1).values)
    # the FFN boundary's NVFP4 copy is the routing launch's quantization bit for bit
    _, _, xq, xsf, _ = dsv4_route_nvfp4_quant_tables(
        logits, bias, x, g, TOPK, SCALE, s["first"], s["count"], TILES, dyn
    )
    fq, fsf = torch.ops.trtllm.fp4_quantize(x, g, 16, False, False)
    assert torch.equal(fq.view(torch.uint8), xq.view(torch.uint8))
    assert torch.equal(fsf.reshape(-1).view(torch.uint8), xsf.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("n", [1, 6, 8])
def test_dynamic_split_covers_every_pick_once(n):
    """Both EP ranks' tables of one call: each pick local on exactly one rank, each rank at most ceil(used / 2) experts,
    and a call already balanced runs the static split."""
    gen = torch.Generator(device=DEV).manual_seed(100 + n)
    for _ in range(20):
        logits, bias, _, _ = _route_inputs(n, 384, gen)
        ids, _, t0 = dsv4_route_tables(logits, bias, TOPK, SCALE, H, 0, 384, TILES, 0)
        _, _, t1 = dsv4_route_tables(logits, bias, TOPK, SCALE, H, 0, 384, TILES, 1)
        k = n * TOPK
        on0, on1 = t0[0, 0, :k] >= 0, t1[0, 0, :k] >= 0
        assert torch.equal(on0 ^ on1, torch.ones_like(on0)), "every pick on exactly one rank"
        flat = ids.reshape(-1).long()
        used = flat.unique().numel()
        cap = (used + 1) // 2
        for on in (on0, on1):
            assert flat[on].unique().numel() <= cap
        assert routing_table_max_ctas(n, TOPK, 384, 0) == (min(n * TOPK, 384) + 1) // 2 >= cap
        home0 = flat < 192
        if max(int(flat[home0].unique().numel()), int(flat[~home0].unique().numel())) <= cap:
            assert torch.equal(on0, home0), "a balanced call keeps the static split"


def test_dynamic_ep_rule():
    assert dynamic_ep_applies(1, 6, False, 2, 4)
    assert not dynamic_ep_applies(4, 6, False, 2, 4)  # batch invariance: batch-1 workers only
    assert not dynamic_ep_applies(1, 9, False, 2, 4)  # past the route launch's tables
    assert not dynamic_ep_applies(1, 6, True, 2, 4)  # attention DP
    assert not dynamic_ep_applies(1, 6, False, 4, 2)  # a two-rank rule; MoE TP >= 4 (memory)
    assert not dynamic_ep_applies(1, 6, False, 2, 2)


_EXPERTS, _LOCAL, _OFF, _INTER = 256, 128, 128, 128


@pytest.fixture(scope="module")
def layer():
    from _fp4_moe_test_utils import _build, _scalars

    args, _ = _build(_EXPERTS, H, _INTER, seed=8)
    win = {
        k: args[k][_OFF : _OFF + _LOCAL].contiguous()
        for k in ("gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale")
    }
    return win, _scalars


def _rows_combined(rows, w, perm):
    """sum_k w[t, k] * rows[perm[t, k]] in fp32 (-1: another rank's expert, skipped)."""
    ok = perm >= 0
    picked = rows.float()[perm.clamp(min=0).long()] * (w.float() * ok).unsqueeze(-1)
    return picked.sum(1)


_BIG_EXPERTS, _BIG_INTER = 384, 576  # the target's: 384 experts, MoE TP4 of 2304


@pytest.fixture(scope="module")
def big_layer():
    from _fp4_moe_test_utils import _build, _scalars

    args, _ = _build(_BIG_EXPERTS, H, _BIG_INTER, seed=9)
    full = {
        k: args[k]
        for k in ("gemm1_weights", "gemm1_weights_scale", "gemm2_weights", "gemm2_weights_scale")
    }
    return full, _scalars


_RUNNER_CELLS = [c for c in fp4_moe_runner_tables.CELLS if not c.spec.get("cute_fc2")]
_CUTE_CELLS = [c for c in fp4_moe_runner_tables.CELLS if c.spec.get("cute_fc2")]


@pytest.mark.parametrize("cell", _CUTE_CELLS, ids=[c.why[:48] for c in _CUTE_CELLS])
def test_runner_cute_dynamic_cells(cell, big_layer, certifying, monkeypatch):  # noqa: F811
    """384 experts: the CuTe FC2 behind the FC1-only run, the bounded grid, and the dynamic split."""
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe

    fc2_calls = []
    run_cute_fc2 = decode_moe.run_cute_fc2

    def spy(*a, **k):  # the engine's CuTe FC2 behind the FC1-only run (`decode_moe.run_cute_fc2`)
        fc2_calls.append(1)
        return run_cute_fc2(*a, **k)

    monkeypatch.setattr(decode_moe, "run_cute_fc2", spy)

    certifying(fp4_moe_runner_tables)
    full, scalars = big_layer
    n, dyn = cell.spec["n"], cell.spec.get("dynamic_rank")
    logits, bias, x, g = _route_inputs(n, _BIG_EXPERTS, gen_for(cell))
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, g, 16, False, False)
    if dyn is None:  # the static window of EP rank 1
        off, local = 192, 192
        ops = {k: v[off : off + local] for k, v in full.items()}
    else:
        off, local, ops = 192 * dyn, 192, full
    held = ops["gemm1_weights"].shape[0]
    ids, w, tables = dsv4_route_tables(logits, bias, TOPK, SCALE, H, off, local, TILES, dyn)
    s1, sg, s2 = scalars(held, float(g), 1.0 / 32.0)
    args = (xq, xsf.reshape(-1).view(torch.float8_e4m3fn), ops["gemm1_weights"], ops["gemm1_weights_scale"], None,
            ops["gemm2_weights"], ops["gemm2_weights_scale"], s1, sg, s2, _BIG_EXPERTS, TOPK, _BIG_INTER, off, local,
            1, False, 0, w, ids, tables, TILES)  # fmt: skip
    kw = dict(
        cute_fc2=True,
        routing_max_ctas=routing_table_max_ctas(n, TOPK, _BIG_EXPERTS, dyn),
        dynamic_experts=_BIG_EXPERTS if dyn is not None else 0,
    )
    with autotune():
        fp4_moe_runner_tables(*args, **kw)
    fc2_calls.clear()
    got = fp4_moe_runner_tables(*args, **kw)
    engaged = bool(fc2_calls)
    perm, used = table_permutation(got, tables, TILES, n, TOPK)
    assert used is not None, "the runner's tactic did not run on the tables: nothing was proved"
    assert engaged or used != 8, "a tile-8 tactic of the bitwise FC2 cubins must take the CuTe FC2"
    # the same call with the FC2 cubins, unbounded: bit for bit the same rows
    plain = fp4_moe_runner_tables(*args, dynamic_experts=kw["dynamic_experts"])
    pperm, _ = table_permutation(plain, tables, TILES, n, TOPK)
    ok = perm >= 0
    assert torch.equal(ok, pperm >= 0)
    assert torch.equal(got[0][perm[ok].long()], plain[0][pperm[ok].long()])
    # every local pick's row is the row of the runner routing itself (over every expert when dynamic)
    ref = fp4_moe_runner_tables.reference(*args, **kw)
    rperm, _ = table_permutation(ref, tables, TILES, n, TOPK)
    assert bool((rperm[ok] >= 0).all())
    assert torch.equal(got[0][perm[ok].long()], ref[0][rperm[ok].long()])


# 2: the routing the target tells the runner (the engine's DeepSeek-V4 MoE: DeepSeekV3, n_group = topk_group = 1, its
# routed scaling); 1: Renormalize. The ids and weights are the caller's either way; the runner's own routing (the
# reference here) runs that type's kernel.
@pytest.mark.parametrize("routing", [2, 1])
@pytest.mark.parametrize("cell", _RUNNER_CELLS, ids=[c.why[:48] for c in _RUNNER_CELLS])
def test_runner_tables_cells(cell, layer, certifying, routing):  # noqa: F811
    certifying(fp4_moe_runner_tables)
    win, scalars = layer
    n, finalize = cell.spec["n"], cell.spec["finalize"]
    logits, bias, x, g = _route_inputs(n, _EXPERTS, gen_for(cell))
    ids, w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(
        logits, bias, x, g, TOPK, SCALE, _OFF, _LOCAL, TILES
    )
    s1, sg, s2 = scalars(_LOCAL, float(g), 1.0 / 32.0)
    args = (xq, xsf.reshape(-1).view(torch.float8_e4m3fn), win["gemm1_weights"], win["gemm1_weights_scale"], None,
            win["gemm2_weights"], win["gemm2_weights_scale"], s1, sg, s2, _EXPERTS, TOPK, _INTER, _OFF, _LOCAL,
            routing, finalize, 0, w, ids, tables, TILES)  # fmt: skip
    from tensorrt_llm._torch.autotuner import autotune

    kw = dict(n_group=1, topk_group=1, routed_scaling_factor=SCALE) if routing == 2 else {}
    with autotune():  # the tables serve tuned tactics only (an untuned run routes in the runner), as in the engine
        fp4_moe_runner_tables(*args, **kw)
    got = fp4_moe_runner_tables(*args, **kw)
    ref = fp4_moe_runner_tables.reference(*args, **kw)
    if finalize:
        assert torch.equal(got[0], ref[0])
        return
    perm, used = table_permutation(got, tables, TILES, n, TOPK)
    assert used is not None, "the runner's tactic did not run on the tables: nothing was proved"
    ref_perm, ref_used = table_permutation(ref, tables, TILES, n, TOPK)
    assert ref_used is None
    assert torch.equal(_rows_combined(got[0], w, perm), _rows_combined(ref[0], w, ref_perm))


def test_runner_on_route_only_tables(layer):
    """The runner on `dsv4_route_tables`' tables with the boundary's NVFP4 rows (`fp4_quantize`) gives bit for bit the
    expert rows of the runner on `dsv4_route_nvfp4_quant_tables`' tables and rows (same tactic, on the tables)."""
    from tensorrt_llm._torch.autotuner import autotune

    win, scalars = layer
    n = 6
    logits, bias, x, g = _route_inputs(n, _EXPERTS, torch.Generator(device=DEV).manual_seed(11))
    s1, sg, s2 = scalars(_LOCAL, float(g), 1.0 / 32.0)
    ids, w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(
        logits, bias, x, g, TOPK, SCALE, _OFF, _LOCAL, TILES
    )
    ids2, w2, tables2 = dsv4_route_tables(logits, bias, TOPK, SCALE, H, _OFF, _LOCAL, TILES)
    fq, fsf = torch.ops.trtllm.fp4_quantize(x, g, 16, False, False)

    def run(q, sf, w_, ids_, t):
        args = (q, sf.reshape(-1).view(torch.float8_e4m3fn), win["gemm1_weights"], win["gemm1_weights_scale"], None,
                win["gemm2_weights"], win["gemm2_weights_scale"], s1, sg, s2, _EXPERTS, TOPK, _INTER, _OFF, _LOCAL,
                2, False, 0, w_, ids_, t, TILES)  # fmt: skip
        kw = dict(n_group=1, topk_group=1, routed_scaling_factor=SCALE)
        with autotune():
            fp4_moe_runner_tables(*args, **kw)
        out = fp4_moe_runner_tables(*args, **kw)
        perm, used = table_permutation(out, t, TILES, n, TOPK)
        assert used is not None, "the runner's tactic did not run on the tables: nothing was proved"
        return _rows_combined(out[0], w_, perm)

    assert torch.equal(run(fq, fsf, w2, ids2, tables2), run(xq, xsf, w, ids, tables))


def test_rejections():
    logits, bias, x, g = _route_inputs(6, 384, torch.Generator(device=DEV).manual_seed(1))
    dsv4_route_nvfp4_quant_tables.is_valid(logits, bias, x, g, TOPK, SCALE, 192, 192, TILES)
    big = _route_inputs(9, 384, torch.Generator(device=DEV).manual_seed(1))
    with pytest.raises(AssertionError, match="at most 8 rows"):
        dsv4_route_nvfp4_quant_tables.is_valid(*big, TOPK, SCALE, 192, 192, TILES)
    with pytest.raises(AssertionError):
        dsv4_route_nvfp4_quant_tables.is_valid(logits, bias, x, g, TOPK, SCALE, 192, 192, [64])


@pytest.mark.parametrize("n", range(1, 9))
def test_every_tile_table_row_is_16_byte_aligned(n):
    """An unfinalized run's permutation is row 0 of its tile's table (`table_permutation`), which the next boundary
    kernel reads as a 16-byte aligned tensor: every row of every tile's table starts 16 bytes aligned, for every row
    count the launch takes and both launches (with and without the quantization)."""
    logits, bias, x, g = _route_inputs(n, 384, torch.Generator(device=DEV).manual_seed(n))
    for tables in (
        dsv4_route_nvfp4_quant_tables(logits, bias, x, g, TOPK, SCALE, 192, 192, TILES)[4],
        dsv4_route_tables(logits, bias, TOPK, SCALE, H, 192, 192, TILES)[2],
    ):
        assert tables.is_contiguous() and tables.shape[:2] == (len(TILES), 5)
        for t in range(len(TILES)):
            for r in range(5):
                assert tables[t, r].data_ptr() % 16 == 0, (n, t, r, tables.shape)
        perm = tables[len(TILES) - 1, 0][: n * TOPK].view(n, TOPK)
        assert perm.data_ptr() % 16 == 0
