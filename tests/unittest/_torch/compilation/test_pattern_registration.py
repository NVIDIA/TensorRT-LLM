# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.compilation.backend import Backend
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
