# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AR64AUTO dispatch (CPU): the generation boundaries past the multicast all-reduce's rows, up to the decode bound,
take the strategy-MNNVL instance; every other row count, and a core without it (kill switch), the configured one."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4 import (
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops import (
    m3_mc_allreduce,
)


@pytest.fixture
def taken(monkeypatch):
    seen = []
    monkeypatch.setattr(
        dispatch, "m3_allreduce", lambda x, comm, params: seen.append(comm) or (x, x)
    )
    return seen


# AllReduceParams(fusion_op=RESIDUAL_RMS_NORM) asserts a residual; the patched m3_allreduce never reads these.
_RESIDUAL = object()
_NORM_W = object()


class _Rows:
    def __init__(self, m: int):
        self.shape = (m, 6144)

    def contiguous(self):
        return self


def _core(wide=True):
    return SimpleNamespace(
        allreduce="configured", allreduce_wide="auto" if wide else None, rms_eps=1e-6
    )


def test_the_wide_rows_take_the_mnnvl_instance(taken) -> None:
    lo, hi = m3_mc_allreduce.max_rows(), dispatch.DECODE_MAX_TOKENS
    assert hi > lo, (
        "the decode bound must exceed the multicast all-reduce's rows for AR64AUTO to apply"
    )
    for m, want in ((1, "configured"), (lo, "configured"), (lo + 1, "auto"), (hi, "auto"), (hi + 1, "configured"),
                    (2048, "configured")):  # fmt: skip
        dispatch._native_allreduce_norm(_core(), _Rows(m), _RESIDUAL, _NORM_W)
        assert taken[-1] == want, (m, taken[-1])


def test_without_the_auto_instance_every_row_count_takes_the_configured_one(taken) -> None:
    for m in (1, m3_mc_allreduce.max_rows() + 1, dispatch.DECODE_MAX_TOKENS):
        dispatch._native_allreduce_norm(_core(wide=False), _Rows(m), _RESIDUAL, _NORM_W)
        assert taken[-1] == "configured", m


def test_a_declined_wide_call_runs_the_configured_instance(monkeypatch) -> None:
    """MNNVL returning None (a shape it does not take) must not fall to AUTO: the configured instance runs."""
    seen = []

    def fake(x, comm, params):
        seen.append(comm)
        return None if comm == "auto" else (x, x)

    monkeypatch.setattr(dispatch, "m3_allreduce", fake)
    dispatch._native_allreduce_norm(
        _core(), _Rows(m3_mc_allreduce.max_rows() + 1), _RESIDUAL, _NORM_W
    )
    assert seen == ["auto", "configured"]


def test_the_kill_switch_is_read_at_import() -> None:
    assert isinstance(dispatch.AR64AUTO, bool)
