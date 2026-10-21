# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: SH_TAKE (the shared-expert CTAs take the extra FC2 output blocks) changes no output bit.

With 188 routed CTAs, 192 FC2 output blocks leave four routed CTAs with two blocks. Under SH_TAKE the last four
shared-expert CTAs run, after their shared work, one of those blocks each as FC2-only CTAs, with the same tiles,
scales and finalize. ``SH_TAKE_BY_M`` turns it on per token count (shipped all off). For every token count the
engine's output under the gated table (on at 6 / 7 / 8 tokens) equals its output under the shipped table, bit for bit,
eager and replayed from a CUDA graph, for spread and concentrated routing.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.m3_moe import (  # noqa: E501
    op as engine_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
    M3MoeState,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.moe.m3_moe_engine import (  # noqa: E501
    m3_moe_engine as run,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_moe_operands import hidden_states, layer_for  # noqa: E402

assert torch.cuda.is_available(), "m3_moe_engine requires a CUDA device"

SHIPPED = engine_op.SH_TAKE_BY_M
GATED = (0, 0, 0, 0, 0, 1, 1, 1)


def _state(monkeypatch, m, on):
    monkeypatch.setattr(engine_op, "SH_TAKE_BY_M", GATED if on else SHIPPED)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return M3MoeState(dev, num_tokens=m, num_ctas=sms - engine_op.SHARED_CTAS, shared=True)


@pytest.mark.parametrize("m", range(1, 9))
def test_tables(monkeypatch, m) -> None:
    assert not any(SHIPPED)
    on = _state(monkeypatch, m, True)
    off = _state(monkeypatch, m, False)
    nheavy = (engine_op.HIDDEN // 32) % on.num_ctas
    expect = bool(GATED[m - 1]) and 0 < nheavy <= engine_op.SHARED_CTAS
    assert bool(on.config.get("sh_take")) == expect
    assert "sh_take" not in off.config


@pytest.mark.parametrize("case", ["random", "concentrated"])
@pytest.mark.parametrize("m", range(1, 9))
def test_bitwise_against_the_shipped_table(monkeypatch, m, case) -> None:
    states = {on: _state(monkeypatch, m, on) for on in (True, False)}
    layers = {on: layer_for(st, case) for on, st in states.items()}
    for i in range(8):
        x = hidden_states(m, 6000 + i)
        a, b = run(x, layers[True]), run(x, layers[False])
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"eager call {i}"
    x = hidden_states(m, 6100)
    outs = {}
    for on in (True, False):
        run(x, layers[on])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = run(x, layers[on])
        graph.replay()
        torch.cuda.synchronize()
        outs[on] = y.clone()
    assert torch.equal(outs[True], outs[False]), "graph replay"
