# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate e on the tp2 target: the per-rank engagement line and work-done counter of every ported item.

* kernels/m3_report.work: the first call logs "[M3 sm107 tp2] rank N: <ID> engaged (<detail>); work 1 (host count:
  ...)", every power of two logs the running count, nothing else is logged, and the exit line holds every item's final
  count; every line says what the count counts (captures + eager calls, replays not counted).
* modeling.taken counts every taken fast path under its path name; the multicast all-reduce's line carries world size,
  kernel (world 2: the 8-CTA cluster kernel) and the per-rank call count.
* the DSpark hooks: a wrapped hook counts only the calls that did the work (and the items riding on it); the S9b op
  counts S9b on every call and S9B-MC only when the multicast form took the operands.
* the target tag line and the eng kit's patterns (eng_tp2_v3) match what the code formats. CPU only.
"""

import re
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    m3_report,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def lines(monkeypatch):
    """Every work / engagement line logged (m3_report's and modeling's), with rank 0 and fresh counters."""
    out = []
    monkeypatch.setattr(m3_report, "_info", out.append)
    monkeypatch.setattr(m3_report, "_rank", lambda: 0)
    monkeypatch.setattr(modeling, "mpi_rank", lambda: 0)
    monkeypatch.setattr(modeling, "logger", SimpleNamespace(info=out.append, warning=out.append))
    m3_report.reset_work()
    modeling.reset_logged()
    yield out
    m3_report.reset_work()
    modeling.reset_logged()


def test_first_call_and_powers_of_two_log_the_count(lines):
    for _ in range(9):
        m3_report.work("UT_ITEM", "unit detail")
    assert lines == [
        f"[M3 sm107 tp2] rank 0: UT_ITEM engaged (unit detail); work 1 ({m3_report.HOST_COUNT})",
        f"[M3 sm107 tp2] rank 0: UT_ITEM work 2 ({m3_report.HOST_COUNT})",
        f"[M3 sm107 tp2] rank 0: UT_ITEM work 4 ({m3_report.HOST_COUNT})",
        f"[M3 sm107 tp2] rank 0: UT_ITEM work 8 ({m3_report.HOST_COUNT})",
    ]
    assert m3_report.work_count("UT_ITEM") == 9
    assert "replays not counted" in m3_report.HOST_COUNT


def test_the_exit_line_holds_every_final_count(lines):
    for item, n in (("B_ITEM", 3), ("A_ITEM", 1)):
        for _ in range(n):
            m3_report.work(item)
    lines.clear()
    m3_report.log_work_totals()
    assert lines == [
        f"[M3 sm107 tp2] rank 0: work-done final ({m3_report.HOST_COUNT}): A_ITEM=1, B_ITEM=3"
    ]


def test_items_count_independently(lines):
    m3_report.work("X1")
    m3_report.work("X2")
    m3_report.work("X1")
    assert m3_report.work_totals() == {"X1": 2, "X2": 1}


def test_taken_counts_the_path(lines):
    for _ in range(3):
        modeling.taken("m3_ut_path", 8, "detail")
    assert m3_report.work_count("m3_ut_path") == 3
    assert "[M3 dispatch] rank 0: m3_ut_path taken at 8 rows (detail)" in lines
    assert any(
        re.match(r"\[M3 sm107 tp2\] rank 0: m3_ut_path engaged \(8 rows: detail\)", s)
        for s in lines
    )


@pytest.mark.parametrize(
    "world,rows,kernel",
    [(2, 1, "8-CTA cluster kernel"), (2, 32, "8-CTA cluster kernel"), (4, 16, "row kernel")],
)
def test_the_mc_allreduce_line_names_world_kernel_and_call(lines, world, rows, kernel):
    mapping = SimpleNamespace(tp_size=world)
    modeling._mc_taken(mapping, rows, "x")
    modeling._mc_taken(mapping, rows, "x")
    taken = [s for s in lines if "mc_allreduce_norm taken" in s]
    assert taken == [
        f"[M3 dispatch] rank 0: mc_allreduce_norm taken at {rows} rows (x, world {world}, {kernel}, call 1)"
    ]
    assert m3_report.work_count(modeling.MC_ALLREDUCE.path) == 2
    eng_x = (
        r"\[M3 dispatch\] rank [0-9]+: mc_allreduce_norm taken at [0-9]+ rows "
        r"\(x.*, world 2, 8-CTA cluster kernel, call [0-9]+\)"
    )
    assert bool(re.search(eng_x, taken[0])) is (world == 2)


def test_the_prepushed_line_matches_the_eng_kit(lines):
    modeling._mc_taken(SimpleNamespace(tp_size=2), 8, "prepushed 2")
    eng = (
        r"\[M3 dispatch\] rank [0-9]+: mc_allreduce_norm taken at [0-9]+ rows "
        r"\(prepushed 2, world 2, 8-CTA cluster kernel, call [0-9]+\)"
    )
    assert any(re.search(eng, s) for s in lines)


@pytest.mark.parametrize(
    "result,counted", [(object(), True), (True, True), (None, False), (False, False)]
)
def test_a_wrapped_hook_counts_only_the_work_it_did(lines, result, counted):
    hook = dspark._counted("UT_HOOK", lambda *a, **k: result, ("UT_RIDER", True), ("UT_OFF", False))
    assert hook(1, x=2) is result
    assert m3_report.work_count("UT_HOOK") == int(counted)
    assert m3_report.work_count("UT_RIDER") == int(counted)
    assert m3_report.work_count("UT_OFF") == 0
    assert dspark._counted("UT_NONE", None) is None


def test_the_s9b_op_counts_s9b_always_and_s9b_mc_when_taken(lines):
    mapping = SimpleNamespace(tp_size=2)
    taken_out = (torch.zeros(2, 4), torch.zeros(2, 4))
    op = dspark._s9b_count_op(mapping, lambda p, r, w, e: taken_out if p.shape[0] <= 8 else None)
    assert op(torch.zeros(8, 4), None, None, 1e-6) is taken_out
    assert op(torch.zeros(16, 4), None, None, 1e-6) is None
    assert dspark._s9b_count_op(mapping, None)(torch.zeros(4, 4), None, None, 1e-6) is None
    assert m3_report.work_count("S9B") == 3 and m3_report.work_count("S9B_MC") == 1
    assert any(
        re.search(
            r"S9B_MC engaged .*world 2, drafter form \(opts 16\) on the 8-CTA cluster kernel", s
        )
        for s in lines
    )


def test_the_target_tag_line_matches_the_eng_kit():
    """The tag line is formatted in the producer's decode branch; the eng kit's ``target`` pattern matches it."""
    with open(modeling.__file__) as f:
        text = f.read()
    assert (
        'f"[M3 sm107 tp2] fast path engaged: m3_producer (decode, rank {mpi_rank()}, {rows} rows'
        in text
    )
    sample = (
        "[M3 sm107 tp2] fast path engaged: m3_producer (decode, rank 1, 8 rows, nvfp4_tp2: ...)"
    )
    assert re.search(r"\[M3 sm107 tp2\] fast path engaged: m3_producer \(decode", sample)


def test_the_tag_is_the_eng_kits():
    assert m3_report.TAG == "[M3 sm107 tp2]"


def test_c6_the_producer_covers_the_tp2_heads():
    """C6: the producer gate holds this target's per-rank heads (32 query, 2 KV, 2 index: the TP2 op's), not the TP4
    heads (16, 1, 1) it was copied with, under which no TP2 sparse layer took ``m3_producer`` (no target tag line)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention import (  # noqa: E501
        m3_producer as op,
    )

    assert modeling.PRODUCER_HEADS == (op.HEADS_Q, op.HEADS_KV, op.HEADS_INDEX) == (32, 2, 2)
    tp2 = SimpleNamespace(main_kv_is_nvfp4=True, heads=32, kv_heads=2, index_heads=2)
    assert modeling._producer_covered(tp2)
    tp4 = SimpleNamespace(main_kv_is_nvfp4=True, heads=16, kv_heads=1, index_heads=1)
    assert not modeling._producer_covered(tp4)
    fp8 = SimpleNamespace(main_kv_is_nvfp4=False, heads=32, kv_heads=2, index_heads=2)
    assert not modeling._producer_covered(fp8)
    with open(modeling.__file__) as f:
        text = f.read()
    # Both the coverage and the call site read the one constant; the TP4 heads are gone.
    assert text.count("(core.heads, core.kv_heads, core.index_heads) == PRODUCER_HEADS") == 2
    assert "(core.heads, core.kv_heads, core.index_heads) == (16, 1, 1)" not in text
