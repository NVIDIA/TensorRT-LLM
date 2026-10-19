# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEEnvironment
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import build_moe_deployment

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "global_flag,moe_flag,explicit_disable,expected_disable",
    [
        (None, None, False, False),
        ("0", "0", False, False),
        ("1", None, False, True),
        (None, "1", False, True),
        ("1", "0", False, True),
        ("0", "1", False, True),
        ("1", "1", False, True),
        (None, None, True, True),
        ("0", "0", True, True),
        ("true", "true", False, False),
    ],
)
def test_deterministic_flags_reach_moe_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    global_flag: str | None,
    moe_flag: str | None,
    explicit_disable: bool,
    expected_disable: bool,
) -> None:
    """Deterministic flags must reach backend policy without probing GPU hardware."""
    for key, value in (
        ("FORCE_DETERMINISTIC", global_flag),
        ("FORCE_MOE_KERNEL_DETERMINISTIC", moe_flag),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    config = ModelConfig(moe_disable_finalize_fusion=explicit_disable)
    deployment = build_moe_deployment(config, num_experts=8, environment=MoEEnvironment(sm=100))

    assert config.moe_disable_finalize_fusion is expected_disable
    assert deployment.fused_finalize_enabled is not expected_disable
