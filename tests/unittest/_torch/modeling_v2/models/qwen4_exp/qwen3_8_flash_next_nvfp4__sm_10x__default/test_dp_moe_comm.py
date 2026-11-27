# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE round trip's transport switch (`DP_MOE_COMM_ENV`).

`nccl` (the default) keeps the all-gather / reduce-scatter pair; `lamport` sends every round trip
the Lamport transport `takes` through `dep_lamport_round_trip` and every other one through NCCL,
the same branch on every rank. The collectives are recorders; no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "value, expected", [(None, "nccl"), ("", "nccl"), ("NCCL", "nccl"), (" lamport ", "lamport")]
)
def test_switch_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(M.DP_MOE_COMM_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_MOE_COMM_ENV, value)
    assert M._dp_moe_comm() == expected


def test_switch_refuses_another_value(monkeypatch):
    monkeypatch.setenv(M.DP_MOE_COMM_ENV, "nvls")
    with pytest.raises(ValueError, match="one of nccl, lamport"):
        M._dp_moe_comm()


class _Trip:
    """A stand-in transport that takes up to `max_rows` rows a rank."""

    def __init__(self, max_rows: int):
        self.max_rows = max_rows

    def takes(self, rows: int, x: torch.Tensor) -> bool:
        return 1 <= rows <= self.max_rows


def _round_trip(monkeypatch, comm: str, max_rows: int = 4):
    calls = []
    monkeypatch.setenv(M.DP_MOE_COMM_ENV, comm)
    monkeypatch.setattr(
        M, "allgather", lambda x, *a: calls.append(("nccl gather", x.shape[0])) or x.repeat(4, 1)
    )
    monkeypatch.setattr(
        M,
        "reducescatter",
        lambda x, *a: calls.append(("nccl reduce", x.shape[0])) or x[: x.shape[0] // 4],
    )

    def lamport(x, trip, reduce):
        calls.append(("lamport reduce" if reduce else "lamport gather", x.shape[0]))
        return x[: x.shape[0] // 4] if reduce else x.repeat(4, 1)

    monkeypatch.setattr(M, "dep_lamport_round_trip", lamport)
    core = SimpleNamespace(
        dp_size=4,
        dp_group=[0, 1, 2, 3],
        mapping=SimpleNamespace(tp_group_pg=None),
        _dp_lamport=lambda group: _Trip(max_rows),
    )
    return M.Qwen4ExpDefaultModelingV2._expert_sum_op(core), calls


@pytest.mark.parametrize(
    "comm, rows, pad, expected",
    [
        ("nccl", 3, 1, [("nccl gather", 4), ("nccl reduce", 16)]),
        ("lamport", 3, 1, [("lamport gather", 4), ("lamport reduce", 16)]),
        ("lamport", 5, 0, [("nccl gather", 5), ("nccl reduce", 20)]),  # past the transport's rows
    ],
)
def test_round_trip_transport(monkeypatch, comm, rows, pad, expected):
    expert_sum, calls = _round_trip(monkeypatch, comm)
    x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    out = expert_sum(lambda gathered: gathered, x, pad)
    assert calls == expected
    assert out.shape == x.shape and torch.equal(out, x)
