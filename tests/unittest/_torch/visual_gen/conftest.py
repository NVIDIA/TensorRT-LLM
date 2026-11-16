# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the VisualGen unit tests."""

from collections.abc import Iterator

import pytest


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
