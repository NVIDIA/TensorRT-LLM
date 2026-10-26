# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2WIDE_GEMM: at 33..48 rows the dense MLP's MXFP8 projections leave the GEMV for the pinned FlashInfer GEMM
(MX-TAC), every other projection keeps its GEMV rows, rows <= 32 are unchanged, and ``linear_path`` never names a
GEMM ``_linear`` does not run (the VR_GEMV48 / LINEAR_CUBLASLT tables are empty at TP2).
TRTLLM_M31_DISABLE_TP2WIDE_GEMM=1 restores the GEMV rows and the two tables."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
    modeling,
)

DENSE = (modeling.DENSE_GATE_UP, modeling.DENSE_DOWN)
OTHERS = (
    modeling.QKV_SPARSE,
    modeling.QKV,
    modeling.O_PROJ,
    modeling.SHARED_GATE_UP,
    modeling.SHARED_DOWN,
)
STEP_ROWS = range(1, modeling.DECODE_STEP_MAX + 1)


def _tables(m) -> dict:
    return {
        name: dict(getattr(m, name))
        for name in ("MX_GEMV_ROWS", "GEMV48_ROWS", "LINEAR_CUBLASLT_ROWS")
    }


# The default module's view, recorded at collection before any reload.
_ON = _tables(modeling)


@pytest.fixture
def killed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_TP2WIDE_GEMM", "1")
    importlib.reload(m31_switches)
    try:
        yield importlib.reload(modeling)
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_TP2WIDE_GEMM", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)


def test_dense_mlp_takes_the_gemm_at_33_to_48_rows() -> None:
    m = modeling
    assert m.TP2WIDE_GEMM
    wide = frozenset(range(33, m.DECODE_MAX_TOKENS + 1))
    for nk in DENSE:
        assert m.MX_GEMV_ROWS[nk] == frozenset(range(1, 33)), nk
        for rows in wide:
            assert m.linear_path(*nk, rows, True, False) == m.PREQUANT_GEMM_PATH, (nk, rows)
        for rows in range(1, 33):
            assert m.linear_path(*nk, rows, True, False) == m.MX_GEMV_PATH, (nk, rows)


def test_other_projections_and_small_rows_unchanged(killed) -> None:
    """Every projection but the dense MLP keeps its GEMV rows; rows <= 32 route the same with the item killed."""
    off = _tables(killed)
    for nk in OTHERS:
        assert _ON["MX_GEMV_ROWS"][nk] == off["MX_GEMV_ROWS"][nk], nk
    for nk in DENSE:
        assert _ON["MX_GEMV_ROWS"][nk] == off["MX_GEMV_ROWS"][nk] - frozenset(range(33, 49)), nk


def test_no_dead_wide_path() -> None:
    """``_linear`` has no branch for the GEMV48 / cuBLASLt paths at TP2: no row count may select them."""
    m = modeling
    assert not m.GEMV48_ROWS and not m.LINEAR_CUBLASLT_ROWS
    for nk in DENSE + OTHERS:
        for rows in STEP_ROWS:
            for prequant in (True, False):
                assert m.linear_path(*nk, rows, prequant, False) not in (
                    m.GEMV48_PATH,
                    m.CUBLASLT_GEMM_PATH,
                ), (nk, rows, prequant)


def test_kill_switch_restores_today(killed) -> None:
    m = killed
    assert not m.TP2WIDE_GEMM
    for nk in DENSE:
        assert m.MX_GEMV_ROWS[nk] == frozenset(range(1, m.DECODE_MAX_TOKENS + 1)), nk
    assert bool(m.LINEAR_CUBLASLT_ROWS) == m.LINEAR_CUBLASLT
