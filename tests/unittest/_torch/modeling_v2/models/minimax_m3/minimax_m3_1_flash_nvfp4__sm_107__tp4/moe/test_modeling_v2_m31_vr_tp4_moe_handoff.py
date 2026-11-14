# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MOE_HANDOFF on the sm_107 tp4 m3_moe engine: the per-token-count FC1 -> FC2 hand-off tables of op.py (``vpoll``,
``epi_vskip``, ``b2buf``, ``hvpf``, ``lag``) change scheduling only, so the engine built with them and the engine
built with TRTLLM_M31_DISABLE_MOE_HANDOFF (the deployed kernel's path) give the same bits for every token count and
routing case, eager, on repeated calls (the call tags advance) and from a CUDA-graph replay."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)

KNOBS = ("vpoll", "epi_vskip", "b2buf", "hvpf", "lag")


def _state(monkeypatch, num_tokens: int, killed: bool):
    monkeypatch.setattr(moe_op, "HANDOFF", not killed)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return moe_op.M3MoeState(dev, num_tokens=num_tokens, num_ctas=sms - 24, shared=True)


@pytest.mark.parametrize("case", ("random", "concentrated"))
@pytest.mark.parametrize("m", range(1, 9))
def test_handoff_is_bitwise_its_kill_switch(
    monkeypatch: pytest.MonkeyPatch, m: int, case: str
) -> None:
    on = _state(monkeypatch, m, killed=False)
    off = _state(monkeypatch, m, killed=True)
    assert not any(k in off.config for k in KNOBS), "the kill switch leaves the deployed config"
    for k, table in (
        ("vpoll", moe_op.VPOLL_BY_M),
        ("epi_vskip", moe_op.EPI_VSKIP_BY_M),
        ("b2buf", moe_op.B2BUF_BY_M),
        ("hvpf", moe_op.HVPF_BY_M),
    ):
        assert on.config.get(k, 0) == table[m - 1]
    a, b = layer_for(off, case), layer_for(on, case)
    for i in range(3):  # repeated calls on the same states: the tags advance
        hidden = hidden_states(m, seed=zlib.crc32(f"handoff-{m}-{case}-{i}".encode()) % 2**31)
        ref = a(hidden)
        out = b(hidden)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
            f"{m} tokens, {case}, call {i}"
        )
        assert torch.equal(on.route_dbg, off.route_dbg)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = b(hidden)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int16), ref.view(torch.int16)), (
        f"{m} tokens, {case} (graph)"
    )


@pytest.mark.parametrize("m", (4, 8))
def test_handoff_lag_builds_without_the_unsafe_override(
    monkeypatch: pytest.MonkeyPatch, m: int
) -> None:
    """The hand-off lag (3 at 4 tokens, 4 at 8) is inside the guard's admitted range: lag 4 runs on one fc1_done
    barrier per group, so the kernel is not told lag_unsafe."""
    st = _state(monkeypatch, m, killed=False)
    lag = moe_op.LAG_BY_M[m - 1]
    assert st.lag == lag and int(st.mod.LAG) == lag and "lag_unsafe" not in st.config
    assert int(st.mod.FC1D) == moe_op.fc1_done_slots(lag) and (lag <= 3 or int(st.mod.FC1D) >= m)


def test_handoff_argument_overrides_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``handoff=False`` (the reference engine's base state) builds the plain schedule even with HANDOFF on."""
    monkeypatch.setattr(moe_op, "HANDOFF", True)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    st = moe_op.M3MoeState(dev, num_tokens=8, num_ctas=sms - 24, shared=True, handoff=False)
    assert not st.handoff and st.lag == moe_op.M3_MOE_LAG
    assert not any(k in st.config for k in KNOBS)
