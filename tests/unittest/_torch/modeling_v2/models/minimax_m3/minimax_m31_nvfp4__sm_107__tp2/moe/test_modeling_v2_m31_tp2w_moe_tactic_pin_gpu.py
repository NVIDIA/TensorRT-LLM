# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LOGITS-route bitwise gate against frozen K3 tactics and CTX's MB references.

Uses the existing loader-shuffled M3.1 operands. Both native finalization and the
production unfinalized-parts/Triton-finalize path are checked, eager and captured.
Local/QA gate, not a performance benchmark or a model-level numerical gate.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.m3_moe_finalize import (  # noqa: E501
    m3_moe_finalize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.moe import (
    m3_moe_tactic_pin as pin,
)
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

__extra_import_path__ = ["."]
from _m3_moe_operands import (  # noqa: E402
    CONCENTRATED,
    MINIMAX2_ROUTING,
    ROUTED_SCALING,
    experts,
    hidden_states,
)


def _inputs(rows, seed, concentrated):
    ops = experts()
    w31, s31, w2, s2, alpha31, scale_c, act_alpha, beta, clamp, alpha2 = ops.tensors
    hidden = hidden_states(rows, seed)
    x, sf = torch.ops.trtllm.fp4_quantize(hidden, ops.fc31_input_scale, 16, False, False)
    logits = torch.nn.functional.linear(hidden.float(), ops.router_weight).contiguous()
    bias = ops.bias.clone()
    if concentrated:
        bias[list(CONCENTRATED)] += 1.0
    return [
        logits,
        bias,
        x,
        sf.view(torch.float8_e4m3fn).flatten(),
        w31,
        s31,
        None,
        act_alpha,
        beta,
        clamp,
        w2,
        s2,
        None,
        scale_c,
        alpha31,
        alpha2,
    ]


def _finish(parts, finalize):
    return parts[0] if finalize else m3_moe_finalize(*parts)


@pytest.mark.parametrize("rows", [96, 104, 112, 120, 127, 232, 256])
@pytest.mark.parametrize("finalize", [False, True])
def test_logits_route_bitwise_eager_and_graph(monkeypatch, rows, finalize):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
    monkeypatch.setattr(pin, "ENABLED", True)
    core = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=2, tp_rank=0),
        hidden=6144,
        moe_inter=1536,
        num_experts=128,
        top_k=4,
        routed_scaling_factor=ROUTED_SCALING,
        max_num_tokens=16384,
    )
    reference = FP4BlockScaleMoERunner(
        128,
        4,
        None,
        None,
        1536,
        0,
        128,
        ROUTED_SCALING,
        MINIMAX2_ROUTING,
        finalize,
        0,
        tune_max_num_tokens=16384,
        use_dp=False,
    )
    # Frozen843057 both-rank cache09a8fe22, buckets64/128/256.
    stored = [16, 196] if rows == 256 else ([16, 236] if rows == 232 else [16, 226])
    ctx = [16, 130] if rows == 256 else ([16, 96] if rows == 232 else [8, 187])
    valid = reference.get_runner().get_valid_configs(4, 6144, 1536, 128, rows)
    assert all(
        t in [list(v) for v in valid] for t in (stored, ctx, list(pin.tactic_for_rows(rows)))
    )

    def candidate(inputs):
        parts = pin.run_pinned(core, inputs, finalize, MINIMAX2_ROUTING, 0)
        assert parts is not None, "pinned path declined"
        return _finish(parts, finalize)

    inputs = _inputs(rows, 1, False)
    candidate(inputs)  # eager initialization before capture (runner and finalize)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = candidate(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (2, 3, 4):
        for concentrated in (False, True):
            fresh = _inputs(rows, seed, concentrated)
            for index in (0, 1, 2, 3):
                inputs[index].copy_(fresh[index])
            eager = candidate(inputs).clone()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))
            for tactic in (stored, ctx):
                expected = _finish(
                    reference.forward([*inputs, None, None], tactic=tactic), finalize
                )
                assert torch.equal(eager.view(torch.int16), expected.view(torch.int16)), (
                    rows,
                    finalize,
                    seed,
                    concentrated,
                    tactic,
                )
