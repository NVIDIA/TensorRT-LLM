# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""compressed-tensors NVFP4 tensor-name normalization shared by weight mappers."""

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.compressed_tensors import (
    invert_global_scale,
    normalize_compressed_tensors_nvfp4_names,
)

pytestmark = pytest.mark.cpu_only

_NVFP4 = "model.layers.0.mlp.gate_proj"
# A mixed-precision checkpoint stores FP8 modules alongside NVFP4 ones, and both
# carry a ``weight_scale``: per-channel [out, 1] for FP8, per-group [out, in/16]
# for NVFP4. Neither may be renamed.
_FP8 = "model.layers.63.mlp.gate_proj"


def _mixed_checkpoint() -> dict:
    return {
        f"{_NVFP4}.weight_packed": torch.zeros(8, 4, dtype=torch.uint8),
        f"{_NVFP4}.weight_scale": torch.zeros(8, 2, dtype=torch.float8_e4m3fn),
        f"{_NVFP4}.weight_global_scale": torch.tensor([4.0]),
        f"{_NVFP4}.input_global_scale": torch.tensor([8.0]),
        f"{_FP8}.weight": torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
        f"{_FP8}.weight_scale": torch.ones(8, 1, dtype=torch.bfloat16),
    }


def test_renames_only_nvfp4_suffixes() -> None:
    out = normalize_compressed_tensors_nvfp4_names(_mixed_checkpoint())

    assert set(out) == {
        f"{_NVFP4}.weight",
        f"{_NVFP4}.weight_scale",
        f"{_NVFP4}.weight_scale_2",
        f"{_NVFP4}.input_scale",
        f"{_FP8}.weight",
        f"{_FP8}.weight_scale",
    }


def test_per_group_and_per_channel_scales_are_untouched() -> None:
    weights = _mixed_checkpoint()
    out = normalize_compressed_tensors_nvfp4_names(weights)

    # The NVFP4 per-group FP8 scale is byte-identical between producers, and the
    # FP8 module's per-channel scale belongs to a different algorithm entirely.
    assert out[f"{_NVFP4}.weight_scale"] is weights[f"{_NVFP4}.weight_scale"]
    assert out[f"{_FP8}.weight_scale"] is weights[f"{_FP8}.weight_scale"]
    assert out[f"{_FP8}.weight"] is weights[f"{_FP8}.weight"]
    # weight_packed is renamed, not materialized.
    assert out[f"{_NVFP4}.weight"] is weights[f"{_NVFP4}.weight_packed"]


def test_global_scales_are_inverted_to_float32() -> None:
    out = normalize_compressed_tensors_nvfp4_names(_mixed_checkpoint())

    assert out[f"{_NVFP4}.weight_scale_2"].dtype == torch.float32
    assert out[f"{_NVFP4}.input_scale"].dtype == torch.float32
    assert out[f"{_NVFP4}.weight_scale_2"].item() == pytest.approx(0.25)
    assert out[f"{_NVFP4}.input_scale"].item() == pytest.approx(0.125)


def test_modelopt_checkpoint_passes_through_unchanged() -> None:
    weights = {
        f"{_NVFP4}.weight": torch.zeros(8, 4, dtype=torch.uint8),
        f"{_NVFP4}.weight_scale": torch.zeros(8, 2, dtype=torch.float8_e4m3fn),
        f"{_NVFP4}.weight_scale_2": torch.tensor([0.25]),
        f"{_NVFP4}.input_scale": torch.tensor([0.125]),
    }

    assert normalize_compressed_tensors_nvfp4_names(weights) is weights


def test_rename_collision_raises() -> None:
    weights = {
        f"{_NVFP4}.weight_packed": torch.zeros(8, 4, dtype=torch.uint8),
        f"{_NVFP4}.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
    }

    with pytest.raises(ValueError, match="collides"):
        normalize_compressed_tensors_nvfp4_names(weights)


def test_lazy_slices_are_materialized_for_global_scales_only() -> None:
    class _Slice:
        """Stand-in for a safetensors PySafeSlice (no torch ops, only ``[...]``)."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor

        def __getitem__(self, item):
            assert item is Ellipsis
            return self._tensor

    packed = _Slice(torch.zeros(8, 4, dtype=torch.uint8))
    weights = {
        f"{_NVFP4}.weight_packed": packed,
        f"{_NVFP4}.weight_global_scale": _Slice(torch.tensor([4.0])),
    }
    out = normalize_compressed_tensors_nvfp4_names(weights)

    assert out[f"{_NVFP4}.weight"] is packed
    assert out[f"{_NVFP4}.weight_scale_2"].item() == pytest.approx(0.25)


@pytest.mark.parametrize(
    "value, expected",
    [
        (4.0, 0.25),
        # Uncalibrated experts are stored as 0; a naive 1/x would emit +Inf,
        # which the fused-MoE max-reduction then spreads over every expert.
        (0.0, 0.0),
        (-1.0, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
    ],
)
def test_invert_global_scale_is_zero_safe(value: float, expected: float) -> None:
    out = invert_global_scale(torch.tensor([value]))

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert out.item() == pytest.approx(expected)


def test_invert_global_scale_mixed_vector() -> None:
    out = invert_global_scale(torch.tensor([4.0, 0.0, 8.0], dtype=torch.float64))

    assert torch.isfinite(out).all()
    assert out.max().item() == pytest.approx(0.25)
    torch.testing.assert_close(out, torch.tensor([0.25, 0.0, 0.125]))


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_dense_global_scale_must_be_positive_and_finite(value: float) -> None:
    weights = {f"{_NVFP4}.weight_global_scale": torch.tensor([value])}

    with pytest.raises(ValueError, match="dense NVFP4 layers require"):
        normalize_compressed_tensors_nvfp4_names(weights)


def test_moe_global_scale_sentinel_can_remain_zero() -> None:
    weights = {f"{_NVFP4}.weight_global_scale": torch.tensor([0.0])}

    out = normalize_compressed_tensors_nvfp4_names(weights, allow_zero_global_scales=True)

    assert out[f"{_NVFP4}.weight_scale_2"].item() == 0.0
