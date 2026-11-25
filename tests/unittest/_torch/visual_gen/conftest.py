# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the VisualGen unit tests."""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig


@pytest.fixture(scope="module")
def disable_cosmos3_guardrails() -> Iterator[None]:
    """Disable Cosmos3 guardrails for the requesting module, leak-free.

    Patches both the environment variable (re-read by
    ``load_standard_components`` on every call) and the pipeline module's
    derived global (assigned by that same function), so teardown restores
    both. Opt in per module via ``pytest.mark.usefixtures``.
    """
    import tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 as pipe_mod

    patcher = pytest.MonkeyPatch()
    patcher.setenv("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "1")
    patcher.setattr(pipe_mod, "TRTLLM_DISABLE_COSMOS3_GUARDRAILS", True)
    yield
    patcher.undo()


@pytest.fixture(autouse=True)
def isolated_cudnn_plans(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """An empty, private cuDNN plan cache for every test, so each process runs the plan
    search itself (no stale plan restored from the user's cache without the finite-output
    check) and the user's on-disk cache is left alone."""
    from tensorrt_llm._torch.visual_gen.models.wan import cudnn_conv

    def reset() -> None:
        cudnn_conv._plan_choices = None
        cudnn_conv._convs.clear()

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    reset()
    yield
    reset()


@pytest.fixture
def deco_recipe() -> Callable[..., "QuantAttentionConfig"]:
    """Return a function that builds the DeCo-Attention recipe: compensated NVFP4 Q/K with V
    in ``v_dtype`` (``fp8`` by default)."""
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    def recipe(v_dtype: str = "fp8") -> QuantAttentionConfig:
        return QuantAttentionConfig(qk_dtype="nvfp4", v_dtype=v_dtype, deco_attention=True)

    return recipe


@pytest.fixture
def require_deco_kernels() -> Callable[..., torch.device]:
    """Return a function that skips the test unless the GPU has a DeCo-Attention kernel for a
    V format (``fp8`` by default; NVFP4 PV: SM120 only), and returns the device otherwise. A
    kernel package that does not import, or a GPU that has kernels but offers fewer V formats
    than it should, fails the test."""
    # the V formats the kernel package offers per GPU capability
    expected_formats = {
        (10, 0): ("fp8",),
        (10, 3): ("fp8",),
        (10, 7): ("fp8",),
        (12, 0): ("fp8", "nvfp4"),
    }

    def require(v_dtype: str = "fp8") -> torch.device:
        from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import deco

        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the DeCo-Attention kernels.")
        if deco._deco_import_error is not None:
            pytest.fail(f"The DeCo-Attention kernels do not import: {deco._deco_import_error}")
        expected = expected_formats.get(torch.cuda.get_device_capability())
        if expected is None:
            pytest.skip("DeCo-Attention kernels need an SM100, SM103, SM107 or SM120 GPU.")
        formats = deco.supported_v_dtypes()
        missing = [f for f in expected if f not in formats]
        if missing:
            pytest.fail(f"DeCo-Attention offers {formats} on this GPU; missing: {missing}")
        if v_dtype not in formats:
            pytest.skip(f"DeCo-Attention with {v_dtype} PV has no kernel on this GPU.")
        return torch.device("cuda:0")

    return require
