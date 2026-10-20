# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_AR_NO_NCCL_SYMM: the NCCL_SYMMETRIC all-reduce strategy is never tuned or run (plain NCCL instead);
inert unless set to 1."""

import pytest
import torch

from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

SYMM = AllReduceStrategy.NCCL_SYMMETRIC.value
NCCL = AllReduceStrategy.NCCL.value


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(tco._Q35_AR_NO_NCCL_SYMM_ENV, raising=False)


def _runner():
    return tco.AllReduceRunner(4, [0, 1, 2, 3], int(AllReduceFusionOp.NONE), 1e-6, False)


def _tactics(rows, hidden=4096):
    return list(
        _runner().get_valid_tactics([torch.empty((rows, hidden), dtype=torch.bfloat16)], None)
    )


@pytest.mark.parametrize("value", [None, "0", "true", ""])
def test_inert_unless_set_to_one(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(tco._Q35_AR_NO_NCCL_SYMM_ENV, value)
    assert tco.q35_ar_strategy_without_nccl_symm(SYMM) == SYMM
    assert SYMM in _tactics(7)
    assert SYMM in _tactics(65536, 1024)


def test_nccl_symmetric_maps_to_nccl(monkeypatch):
    monkeypatch.setenv(tco._Q35_AR_NO_NCCL_SYMM_ENV, "1")
    assert tco.q35_ar_strategy_without_nccl_symm(SYMM) == NCCL
    for other in (
        AllReduceStrategy.NCCL,
        AllReduceStrategy.ONESHOT,
        AllReduceStrategy.TWOSHOT,
        AllReduceStrategy.MNNVL,
        AllReduceStrategy.AUTO,
    ):
        assert tco.q35_ar_strategy_without_nccl_symm(other.value) == other.value


def test_nccl_symmetric_is_never_tuned(monkeypatch):
    stock_small, stock_large = _tactics(7), _tactics(65536, 1024)
    monkeypatch.setenv(tco._Q35_AR_NO_NCCL_SYMM_ENV, "1")
    small, large = _tactics(7), _tactics(65536, 1024)
    assert SYMM not in small and SYMM not in large
    # Everything else the stock list offers stays (NCCL always; ONESHOT / TWOSHOT where valid).
    assert set(small) == set(stock_small) - {SYMM}
    assert set(large) == set(stock_large) - {SYMM}
    assert NCCL in small and NCCL in large
