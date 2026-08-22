# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops as cute_dsl_custom_ops
from tensorrt_llm._torch.autotuner import OptimizationProfile
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
    )


_CUTEDSL_FC2_N_TILE_SIZE_ENV = "TRTLLM_CUTEDSL_FC2_N_TILE_SIZE"

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE,
    reason="Requires CUTLASS DSL",
)


def _make_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> "Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner":
    monkeypatch.setattr(cute_dsl_custom_ops, "get_sm_version", lambda: 100)
    monkeypatch.setattr(
        Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
        "kernel_class",
        SimpleNamespace(can_implement=lambda **kwargs: True),
    )
    return Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner(
        num_experts=256,
        top_k=8,
        num_local_experts=8,
        local_expert_offset=0,
        tile_size=128,
        output_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    ("env_value", "expected_n_tiles"),
    [
        (None, {128, 256}),
        ("128", {128}),
        ("256", {256}),
    ],
)
def test_fc2_n_tile_size_override(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected_n_tiles: set[int],
) -> None:
    if env_value is None:
        monkeypatch.delenv(_CUTEDSL_FC2_N_TILE_SIZE_ENV, raising=False)
    else:
        monkeypatch.setenv(_CUTEDSL_FC2_N_TILE_SIZE_ENV, env_value)

    runner = _make_runner(monkeypatch)
    inputs = [torch.empty(256, 64), torch.empty(8, 256)]

    tactics = runner.get_valid_tactics(inputs, OptimizationProfile())

    assert {tactic[0][1] for tactic in tactics} == expected_n_tiles
    assert runner._get_default_tactic()[0][1] == (int(env_value) if env_value is not None else 128)
    if env_value is None:
        assert "fc2_n_tile_size_override" not in runner.unique_id()
    else:
        assert runner.unique_id()[-2:] == (
            "fc2_n_tile_size_override",
            int(env_value),
        )


@pytest.mark.parametrize("env_value", ["", "64", "128,256", " 128", "invalid"])
def test_fc2_n_tile_size_override_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv(_CUTEDSL_FC2_N_TILE_SIZE_ENV, env_value)

    with pytest.raises(ValueError, match=f"{_CUTEDSL_FC2_N_TILE_SIZE_ENV} must be unset"):
        _make_runner(monkeypatch)
