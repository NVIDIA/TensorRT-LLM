# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 16-row one-shot published MoE tail runs late RMS.

``compile_tail``'s ``late_rms`` picks the one-shot WaveMoeTail's late RMS (None:
``RUBIN_WAVE_LATE_RMS``, whose rows stay as they were: 5-8 on Rubin, none elsewhere; never on
tcgen05 or two-shot tails, whose late RMS is ``RUBIN_TWOSHOT_LATE_RMS``'s). ``tail_late_rms``
recompiles only the published mma.sync tail of the rows it is given and keeps the choice in the
recorded compile options (so ``use_heads`` keeps it). WaveMoeTail takes late RMS over two groups
(9-16 rows) only on the one-shot mma.sync grid. On an SM 10.x GPU the 16-row tail compiles with it.
"""

import importlib
from types import SimpleNamespace as NS
from unittest import mock

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
TP, ROWS = 16, 16


def _vendor(name):
    return importlib.import_module(f"{_FAMILY}.kernels.vendor.vllm_mr8").module(name)


def _tail():
    return _vendor("cutedsl.moe.tail_runtime")


class _Stop(Exception):
    pass


def _late_of(tail, batch, rubin, mode="publish", **options):
    """The ``late_rms`` ``compile_tail`` hands WaveMoeTail at ``batch``, without compiling."""
    made = {}

    def record(batch, tp, twoshot, early, umma, poll_ns, direct, late, *rest, **kw):
        made["late"] = late
        raise _Stop

    with (
        mock.patch.object(tail, "rubin", lambda device: rubin),
        mock.patch.object(tail, "WaveMoeTail", record),
        pytest.raises(_Stop),
    ):
        tail.compile_tail.__wrapped__(TP, batch, 0, mode, **options)
    return made["late"]


def test_the_table_rows_are_unchanged_without_late_rms() -> None:
    tail = _tail()
    assert {b: _late_of(tail, b, True) for b in (4, 5, 8, ROWS, 24)} == {
        4: False, 5: True, 8: True, ROWS: False, 24: False,
    }  # fmt: skip
    assert {b: _late_of(tail, b, False) for b in (5, 8, ROWS)} == {5: False, 8: False, ROWS: False}


@pytest.mark.parametrize("rubin", (True, False))
def test_late_rms_sets_the_one_shot_tail(rubin) -> None:
    tail = _tail()
    assert _late_of(tail, ROWS, rubin, late_rms=True) is True
    assert _late_of(tail, 8, rubin, late_rms=False) is False
    # tcgen05 / two-shot tails take late RMS from RUBIN_TWOSHOT_LATE_RMS only.
    assert _late_of(tail, ROWS, rubin, late_rms=True, umma=True) is False


def test_wave_tail_takes_two_group_late_rms_on_the_one_shot_grid_only() -> None:
    wave = _vendor("cutedsl.moe.tail").WaveMoeTail
    for batch in (9, 12, ROWS):
        assert wave(batch, TP, late_rms=True).late_rms
    for batch, options in ((24, {}), (ROWS, {"twoshot": True})):
        with pytest.raises(AssertionError):
            wave(batch, TP, late_rms=True, **options)


def test_tail_late_rms_recompiles_only_the_published_mma_rows_given() -> None:
    modeling = importlib.import_module(f"{_PKG}.modeling")
    assert modeling.TAIL_LATE_RMS_ROWS == (ROWS,)
    compiles = {
        8: ((TP, 8, 0, "publish"), {"early": True, "umma": False}),
        ROWS: ((TP, ROWS, 0, "publish"), {"early": True, "umma": False, "poll_ns": 0}),
        24: ((TP, 24, 0, "publish"), {"early": True, "umma": True}),
        32: ((TP, 32, 0, "twoshot"), {"early": True, "umma": True}),
        1: ((TP, 1, 0, "front"), {"early": False, "umma": False}),
    }
    codes = {b: (f"code{b}", "views", 0, False) for b in compiles}
    weights = NS(compiles=dict(compiles), codes=dict(codes))
    calls = []

    def compile_tail(*args, **options):
        calls.append((args, options))
        return "late"

    with mock.patch.object(_tail(), "compile_tail", compile_tail):
        done = modeling.tail_late_rms(NS(tail_weights=weights), (ROWS, 24, 32, 1, 48))
    assert done == [ROWS]
    late = {"early": True, "umma": False, "poll_ns": 0, "late_rms": True}
    assert calls == [((TP, ROWS, 0, "publish"), late)]
    assert weights.compiles[ROWS][1] == late
    assert weights.codes[ROWS] == ("late", "views", 0, False)
    for b in (8, 24, 32, 1):
        assert weights.compiles[b] == compiles[b] and weights.codes[b] == codes[b]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="compiles the MR8 tail for an SM10x GPU",
)
def test_sixteen_row_tail_compiles_with_late_rms() -> None:
    """The 16-row published tail compiles with and without late RMS, as two kernels (the numerics
    against an FP32 reference are the GB300 gate's)."""
    tail = _tail()
    device = torch.cuda.current_device()
    late = tail.compile_tail.__wrapped__(TP, ROWS, device, "publish", early=True, late_rms=True)
    table = tail.compile_tail.__wrapped__(TP, ROWS, device, "publish", early=True)
    assert late is not None and table is not None and late is not table
