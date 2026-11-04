# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for the VisualGen unit tests."""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig


@pytest.fixture
def require_sage_recipe() -> Callable[["QuantAttentionConfig"], "AttentionConfig"]:
    """Validate Sage recipes, skipping only the unsupported INT8 hardware case."""
    from tensorrt_llm.visual_gen.args import AttentionConfig

    def validate(quant_config: "QuantAttentionConfig") -> AttentionConfig:
        try:
            return AttentionConfig(backend="TRTLLM", quant_attention_config=quant_config)
        except ValidationError as exc:
            errors = exc.errors()
            if (
                quant_config.qk_dtype == "int8"
                and len(errors) == 1
                and errors[0]["msg"].startswith(
                    "Value error, INT8 SageAttention is unsupported on SM"
                )
            ):
                pytest.skip(errors[0]["msg"])
            raise

    return validate


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
