# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest
import torch
from torch._inductor.pattern_matcher import (
    MULTIPLE,
    CallFunction,
    KeywordArg,
    MultiOutputPattern,
    PatternMatcherPass,
)
from torch.fx.experimental.proxy_tensor import make_fx

from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.compilation.patterns import _register_pattern_replacement
from tensorrt_llm.mapping import Mapping


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Undo the environment build_custom_passes writes to.

    The multi-GPU path sets DISABLE_LAMPORT_REDUCE_NORM_FUSION in os.environ, and a
    leaked value makes later tests in the same process compile through a different
    path. patch.dict snapshots os.environ and drops whatever the block added.
    """
    with patch.dict(os.environ, {}, clear=False):
        yield


@pytest.mark.parametrize("enable_userbuffers", [False, True])
def test_multi_gpu_pattern_registration(enable_userbuffers: bool) -> None:
    mapping = Mapping(world_size=4, tp_size=4, rank=0)

    with (
        patch("tensorrt_llm.mpi_world_size", return_value=4),
        patch(
            "tensorrt_llm.bindings.internal.userbuffers.ub_supported",
            return_value=enable_userbuffers,
        ),
    ):
        Backend.build_custom_passes(enable_userbuffers, mapping)


def test_single_gpu_pattern_registration() -> None:
    """Cover the world_size == 1 branch.

    It is the only caller of register_add_norm_quant, so without this the quantized
    add-norm registration is never exercised.
    """
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    with patch("tensorrt_llm.mpi_world_size", return_value=1):
        Backend.build_custom_passes(False, mapping)


@pytest.mark.cpu_only
@pytest.mark.parametrize("shape", [(3, 7), (17, 32)])
@pytest.mark.parametrize("accept_match", [False, True])
def test_explicit_pattern_replacement(shape, accept_match):
    """Match named captures and preserve multiple outputs and custom guards."""
    add = CallFunction(
        torch.ops.aten.add.Tensor, KeywordArg("input"), KeywordArg("residual"), _users=MULTIPLE
    )
    pattern = MultiOutputPattern([CallFunction(torch.ops.aten.mul.Tensor, add, 2), add])

    def replacement(input, residual):
        combined = input + residual
        return combined + combined, combined

    custom_pass = PatternMatcherPass("explicit_pattern_test")
    _register_pattern_replacement(
        pattern, replacement, custom_pass, extra_check=lambda match: accept_match
    )

    def original(input, residual):
        combined = input + residual
        return combined * 2, combined

    inputs = [torch.randn(shape), torch.randn(shape)]
    gm = make_fx(original, tracing_mode="fake")(*inputs)
    assert custom_pass.apply(gm) == int(accept_match)
    gm.graph.lint()
    gm.recompile()
    torch.testing.assert_close(gm(*inputs), original(*inputs))
    multiplications = [node for node in gm.graph.nodes if node.target == torch.ops.aten.mul.Tensor]
    assert len(multiplications) == int(not accept_match)
