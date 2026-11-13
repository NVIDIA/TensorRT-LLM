# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The imported DSpark drafter glue (``ops/dspark``) with its native ops killed falls back early, without CUDA work."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tensorrt_llm  # noqa: F401 -- register ops before loading the glue modules

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark"


@pytest.mark.cpu_only
@pytest.mark.parametrize("combined", [False, True])
def test_imported_drafter_inputs_kill_has_no_output_allocation(monkeypatch, combined):
    module = importlib.import_module(f"{PKG}.dspark_drafter_inputs")
    monkeypatch.setattr(module, "_GLUE_CPP_OFF", True)
    allocate = Mock(side_effect=AssertionError("killed path must not allocate outputs"))
    monkeypatch.setattr(module, "_outputs", allocate)
    args = (None, SimpleNamespace(shape=(1,)), None, None, 0, None, None, 8, 8)
    if combined:
        result = module.drafter_inputs_and_refresh(*args, None, 0, 1, 128, None, None)
    else:
        result = module.drafter_inputs(*args)
    assert result is None
    allocate.assert_not_called()


@pytest.mark.cpu_only
def test_imported_block_refresh_kill_needs_no_tensor_access(monkeypatch):
    module = importlib.import_module(f"{PKG}.dspark_block_tables")
    monkeypatch.setattr(module, "_GLUE_CPP_OFF", True)
    assert not module.refresh_block_tables(None, 0, 1, 128, None, None)
