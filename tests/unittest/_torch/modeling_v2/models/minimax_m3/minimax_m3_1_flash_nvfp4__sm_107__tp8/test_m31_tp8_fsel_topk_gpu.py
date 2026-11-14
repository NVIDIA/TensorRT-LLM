# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSEL top-16 select (family ``minimax_m3/ops/m3_fsel_topk``,
``mv2_minimax_m3::minimax_m3_sm107tp4_fsel_topk``): the same
table as the TP8 target's SelectLongDsl (``m3_select_long_dsl``) and the select contract's reference, as a set and in
order, on every certified cell, the DSL test's mixed / clamped / padding rows, ties (mass ties into the radix
fallback), signed zeros, subnormals and large finite scores, both score layouts, every trigger phase, both launch
variants, PDL on and off, and inside a CUDA graph; the two non-finite classes the op answers differently are pinned;
the TP8 target's TRTLLM_M31_DISABLE_TP8_FSEL kill switch and SELECT_EARLY-linked trigger phase, the fake, the
long-context decode rows 1..8 and the bound + 1, and the engagement line."""

import importlib
import os
import zlib

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import m3_fsel_topk as fsel
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.m3_fsel_topk import TOPK
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.m3_fsel_topk import (
    m3_fsel_topk as op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

_TP8 = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8"

# Valid-block counts on every path: <= 16 (copied), the 6-, 16- and 18-key bodies and their edges, past the row
# (clamped, no local forcing), and zero / negative (CUDA-graph padding rows).
_MIXED_VALID = (
    0,
    5,
    16,
    17,
    200,
    1536,
    1537,
    4096,
    4097,
    4224,
    8192,
    8193,
    9000,
    1,
    300,
    6000,
    129,
    3000,
    -3,
)


def _dsl():
    """The TP8 target's SelectLongDsl entry (the select this op replaces), or None when its stack does not import."""
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.attention.m3_select_long_dsl import (  # noqa: E501
            m3_select_long_dsl,
        )
    # ImportError: the optional CuTe DSL stack; the comparison against the reference still runs
    except ImportError:
        return None
    return m3_select_long_dsl


def _scores(spec, g):
    nb, tq, h = spec["num_blocks"], spec["total_q"], spec["num_kv_heads"]
    s = torch.randn(h, nb, tq, generator=g, device="cuda", dtype=torch.float32)
    kind = spec["scores"]
    if kind == "ties":
        # Even tokens: a flat row with ten winners (thousands tie at the 16th-largest score: pool overflow, the radix
        # fallback's mass-tie path). Odd tokens: a coarse grid (ties broken by block index).
        s[:, :, 0::2] = 0.25
        winners = torch.randint(0, nb, (h, 10, tq), generator=g, device="cuda")
        s[:, :, 0::2].scatter_(1, winners[:, :, 0::2], 3.0)
        s[:, :, 1::2] = (s[:, :, 1::2] * 2).round() / 2
    elif kind == "clustered":
        # The top scores in one contiguous run of blocks (one warp's lanes): the witness pool's worst case.
        start = int(torch.randint(0, max(1, nb - 64), (1,), generator=g, device="cuda"))
        s[:, start : start + 48, :] += 6.0
    elif kind == "edges":
        s[:, 3::40, :] = -0.0
        s[:, 4::40, :] = 0.0
        s[:, 5::41, :] = 1e-40
        s[:, 6::43, :] = -1e-40
        s[:, 9::97, :] = 1e28
        s[:, 10::89, :] = -1e28
    elif kind == "nonfinite":
        s[:, 7::50, :] = float("inf")
        s[:, 11::70, :] = float("nan")
        s[:, :, 1::2] = float("-inf")
        s[:, 0:5, 1::2] = 1.0
    return s


def _valid(spec, g):
    tq, nb = spec["total_q"], spec["num_blocks"]
    if spec["valid"] == "mixed":
        vals = [_MIXED_VALID[i % len(_MIXED_VALID)] for i in range(tq)]
    elif spec["valid"] == "c1_150k":
        # 1 request x 8 query rows at 150,000 tokens (kv_len 149,993 .. 150,000), 128-token blocks.
        vals = [(150000 - tq + t + 1 + 127) // 128 for t in range(tq)]
    else:
        vals = (nb - torch.randint(0, 12, (tq,), generator=g, device="cuda")).tolist()
        vals[0] = nb
    return torch.tensor(vals, dtype=torch.int32, device="cuda")


def _case(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = _scores(spec, g)
    nvb = _valid(spec, g)
    out = torch.empty(
        spec["num_kv_heads"], spec["total_q"], TOPK, dtype=torch.int32, device="cuda"
    ).permute(1, 0, 2)
    return scores, nvb, out, spec["init_blocks"], spec["local_blocks"]


def _check(scores, nvb, out, init_blocks, local_blocks):
    """The op's table equals the reference (set and order: both are ascending ids, -1 tail) and SelectLongDsl's."""
    ref = op.reference(scores, nvb, out, init_blocks, local_blocks)
    assert torch.equal(out, ref), "FSEL table differs from the select contract's reference"
    dsl = _dsl()
    if dsl is not None and scores.shape[1] <= 9216:
        dsl_out = torch.empty_like(out)
        dsl(scores, nvb, dsl_out, init_blocks, local_blocks, False)
        assert torch.equal(out, dsl_out), "FSEL table differs from SelectLongDsl"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    scores, nvb, out, init_blocks, local_blocks = _case(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    op(scores, nvb, out, init_blocks, local_blocks)
    _check(scores, nvb, out, init_blocks, local_blocks)


@pytest.mark.parametrize("kind", ["random", "ties", "clustered", "edges"])
@pytest.mark.parametrize("num_blocks", [129, 1172, 2048, 4500, 8193, 9216])
@pytest.mark.parametrize("seed", range(4))
def test_mixed_rows_match_selectlongdsl(kind, num_blocks, seed) -> None:
    spec = dict(
        num_blocks=num_blocks,
        total_q=len(_MIXED_VALID),
        valid="mixed",
        scores=kind,
        num_kv_heads=1,
        init_blocks=0,
        local_blocks=1,
    )
    scores, nvb, out, init_blocks, local_blocks = _case(spec, seed=seed * 1000 + num_blocks)
    op(scores, nvb, out, init_blocks, local_blocks, True)
    _check(scores, nvb, out, init_blocks, local_blocks)


@pytest.mark.parametrize("layout", ["kv_blocks_q", "kv_q_blocks"])
@pytest.mark.parametrize("valid", [17, 1172, 1537, 4097, 9216])
def test_both_score_layouts(layout, valid) -> None:
    """The scorer's block-major layout (block stride = total_q) and the row-contiguous one (block stride 1)."""
    g = torch.Generator(device="cuda").manual_seed(valid)
    nb = max(valid, 129)
    s = torch.randn(1, nb, 8, generator=g, device="cuda")
    if layout == "kv_q_blocks":
        s = s.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    nvb = torch.tensor([valid - (t % 2) for t in range(8)], dtype=torch.int32, device="cuda")
    out = torch.empty(1, 8, TOPK, dtype=torch.int32, device="cuda").permute(1, 0, 2)
    op(s, nvb, out, 0, 1, True)
    _check(s, nvb, out, 0, 1)


@pytest.mark.parametrize("variant", [5, 6])
@pytest.mark.parametrize("trigger_phase", [0, 1, 2])
@pytest.mark.parametrize("use_pdl", [False, True])
def test_launch_options_select_the_same(variant, trigger_phase, use_pdl) -> None:
    spec = op.CELLS[2].spec
    scores, nvb, out, init_blocks, local_blocks = _case(spec, seed=31)
    op(scores, nvb, out, init_blocks, local_blocks, use_pdl, trigger_phase, variant)
    _check(scores, nvb, out, init_blocks, local_blocks)


def test_graph_replay_selects_the_same() -> None:
    spec = op.CELLS[0].spec
    scores, nvb, out, init_blocks, local_blocks = _case(spec, seed=9)
    ref = op.reference(scores, nvb, out, init_blocks, local_blocks)
    fsel.prepare(scores.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(scores, nvb, out, init_blocks, local_blocks, True, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(scores, nvb, out, init_blocks, local_blocks, True, 0)
    for _ in range(3):
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def test_target_adapter_matches_long_select_blocks() -> None:
    """fsel_select_blocks returns long_select_blocks' head-major backed table with the same ids."""
    spec = op.CELLS[0].spec
    scores, nvb, _, init_blocks, local_blocks = _case(spec, seed=5)
    table = fsel.fsel_select_blocks(
        scores, n_valid_blocks=nvb, init_blocks=init_blocks, local_blocks=local_blocks, use_pdl=True
    )
    assert table is not None and table.shape == (8, 1, TOPK)
    assert table.permute(1, 0, 2).is_contiguous(), (
        "the LC decode reads the [heads, total_q, 16] view contiguously"
    )
    _check(scores, nvb, table, init_blocks, local_blocks)
    assert (
        fsel.fsel_select_blocks(
            torch.zeros(1, 9217, 8, device="cuda"),
            n_valid_blocks=nvb,
            init_blocks=0,
            local_blocks=1,
        )
        is None
    )
    assert (
        fsel.fsel_select_blocks(
            torch.zeros(1, 128, 8, device="cuda"), n_valid_blocks=nvb, init_blocks=0, local_blocks=1
        )
        is None
    )


def test_the_nonfinite_classes_are_the_documented_ones() -> None:
    """-inf picks are emitted as ids (the contract emits -1); rows without them are the same table; is_valid names
    the class."""
    spec = dict(
        num_blocks=600,
        total_q=8,
        valid="mixed",
        scores="nonfinite",
        num_kv_heads=1,
        init_blocks=0,
        local_blocks=1,
    )
    scores, nvb, out, init_blocks, local_blocks = _case(spec, seed=3)
    op(scores, nvb, out, init_blocks, local_blocks)
    ref = op.reference(scores, nvb, out, init_blocks, local_blocks)
    neg_inf_rows = torch.isneginf(scores[0].t()).any(dim=1)  # [total_q]
    same = (out == ref).all(dim=2).all(dim=1)
    assert bool(same[~neg_inf_rows].all()), "rows without -inf scores differ"
    assert not bool(same[neg_inf_rows & (nvb > 16)].all()), (
        "the -inf class no longer differs: update the note"
    )
    with pytest.raises(AssertionError, match="-inf"):
        op.is_valid(scores, nvb, out, init_blocks, local_blocks)


def test_op_refuses_a_row_past_its_width() -> None:
    scores = torch.zeros(1, 9217, 1, device="cuda")
    nvb = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.empty(1, 1, TOPK, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="at most 9216"):
        op(scores, nvb, out, 0, 1)


@pytest.mark.parametrize("value, on", [(None, True), ("0", True), ("1", False), ("true", False)])
def test_tp8_kill_switch(value, on, monkeypatch) -> None:
    """Default ON; TRTLLM_M31_DISABLE_TP8_FSEL=1 turns the FSEL path off (the DSL select runs), read at import."""
    switches = importlib.import_module(f"{_TP8}.m31_switches")
    modeling = importlib.import_module(f"{_TP8}.modeling")
    _orig = os.environ.get("TRTLLM_M31_DISABLE_TP8_FSEL")
    monkeypatch.delenv("TRTLLM_M31_DISABLE_TP8_FSEL", raising=False)
    if value is not None:
        monkeypatch.setenv("TRTLLM_M31_DISABLE_TP8_FSEL", value)
    try:
        importlib.reload(switches)
        m = importlib.reload(modeling)
        assert m.FSEL_TOPK is on
        assert m.FSEL_SELECT.keys == (frozenset({"fsel"}) if on else frozenset())
        assert m.FSEL_SELECT in m.FAST_PATHS
        assert (m.FSEL_TRIGGER_PHASE, m.FSEL_MAX_ROWS, m.FSEL_VARIANT) == (0, 8, 6)
    finally:
        # Back to the value the test started with (not just unset), then reload under it.
        if _orig is None:
            monkeypatch.delenv("TRTLLM_M31_DISABLE_TP8_FSEL", raising=False)
        else:
            monkeypatch.setenv("TRTLLM_M31_DISABLE_TP8_FSEL", _orig)
        importlib.reload(switches)
        importlib.reload(modeling)


def test_select_early_off_moves_the_fsel_release_to_phase_1(monkeypatch) -> None:
    """SELECT_EARLY's kill switch (the early trigger path) also moves FSEL's release to phase 1, so one switch covers
    the release point on either select."""
    switches = importlib.import_module(f"{_TP8}.m31_switches")
    modeling = importlib.import_module(f"{_TP8}.modeling")
    _orig = os.environ.get("TRTLLM_M31_DISABLE_SELECT_EARLY")
    monkeypatch.setenv("TRTLLM_M31_DISABLE_SELECT_EARLY", "1")
    try:
        importlib.reload(switches)
        m = importlib.reload(modeling)
        assert (m.SELECT_EARLY_MAX_ROWS, m.FSEL_TRIGGER_PHASE) == (0, 1)
    finally:
        # Back to the value the test started with (not just unset), then reload under it.
        if _orig is None:
            monkeypatch.delenv("TRTLLM_M31_DISABLE_SELECT_EARLY", raising=False)
        else:
            monkeypatch.setenv("TRTLLM_M31_DISABLE_SELECT_EARLY", _orig)
        importlib.reload(switches)
        importlib.reload(modeling)


def test_fake_registered() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import fake_ops

    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        f"{fake_ops._NS}::minimax_m3_sm107tp4_fsel_topk", "Meta"
    )


@pytest.mark.parametrize("total_q", list(range(1, 10)))
def test_long_context_rows_and_the_bound_plus_one(total_q) -> None:
    """The long-context shape (a 1M-token cache's 8193-block scores, 150K-token rows) at every row count the TP8 step
    dispatches to FSEL (1..8) and at the bound + 1 (9: the target runs the DSL select there; the op itself stays
    exact), at the target's launch (PDL, its trigger phase, variant 6), through the target adapter."""
    modeling = importlib.import_module(f"{_TP8}.modeling")
    spec = dict(
        num_blocks=8193,
        total_q=total_q,
        valid="c1_150k",
        scores="random",
        num_kv_heads=1,
        init_blocks=0,
        local_blocks=1,
    )
    for seed in (11, 12, 13):
        scores, nvb, _, init_blocks, local_blocks = _case(spec, seed=seed)
        table = fsel.fsel_select_blocks(
            scores,
            n_valid_blocks=nvb,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            use_pdl=True,
            trigger_phase=modeling.FSEL_TRIGGER_PHASE,
            variant=modeling.FSEL_VARIANT,
        )
        assert table is not None
        _check(scores, nvb, table, init_blocks, local_blocks)
    assert (total_q <= modeling.FSEL_MAX_ROWS) == (total_q <= 8)


def test_engagement_line(capfd) -> None:
    # capfd, not capsys: tests/unittest/conftest.py's autouse cuda_error_early_quit holds capfd, and pytest refuses both
    fsel._ENGAGED.discard(("test", 0))
    fsel.log_engaged("test", 0, torch.device("cuda"), 8, 0)
    fsel.log_engaged("test", 0, torch.device("cuda"), 8, 0)
    err = capfd.readouterr().err
    # The family op's line (shared with the tp4 targets, where FSEL started): the tag names its origin, not the caller.
    assert (
        err.count("[M31_TP4_FSEL] engaged: target test rank 0") == 1
        and "rows <= 8 trigger phase 0" in err
    )


@pytest.mark.parametrize("num_blocks", [4097, 8192, 8193, 9216])
@pytest.mark.parametrize("trigger_phase", [0, 1, 2])
def test_variant5_rows_past_8192_blocks(num_blocks, trigger_phase) -> None:
    """Regression (variant 5 left the 9000-valid row of an 8193-block call
    unwritten): the 256-thread launch over more than 8192 blocks runs the 512-thread instance; at and below 8192 the
    256-thread body (32 keys) is exact. out is poisoned first, so an unwritten row cannot pass."""
    spec = dict(
        num_blocks=num_blocks,
        total_q=32,
        valid="mixed",
        scores="random",
        num_kv_heads=1,
        init_blocks=0,
        local_blocks=1,
    )
    for seed in (31, 32, 33, 34, 35):
        scores, nvb, out, init_blocks, local_blocks = _case(spec, seed=seed)
        out.fill_(-7)
        op(scores, nvb, out, init_blocks, local_blocks, False, trigger_phase, 5)
        torch.cuda.synchronize()
        _check(scores, nvb, out, init_blocks, local_blocks)
