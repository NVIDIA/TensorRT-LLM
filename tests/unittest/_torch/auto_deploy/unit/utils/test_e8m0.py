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

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.utils import e8m0
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import (
    e8m0_to_fp32,
    e8m0_to_uint8,
    maybe_e8m0_to_fp32,
)

_HAS_E8M0 = hasattr(torch, "float8_e8m0fnu")
_requires_e8m0 = pytest.mark.skipif(
    not _HAS_E8M0,
    reason="torch.float8_e8m0fnu is not available in this PyTorch build",
)


def _e8m0_from_bytes(raw_bytes: list[int]) -> torch.Tensor:
    return torch.tensor(raw_bytes, dtype=torch.uint8, device="cpu").view(torch.float8_e8m0fnu)


@_requires_e8m0
def test_e8m0_to_uint8_preserves_raw_bytes() -> None:
    raw_bytes = torch.tensor([0, 1, 126, 127, 128, 254], dtype=torch.uint8, device="cpu")
    scale = raw_bytes.view(torch.float8_e8m0fnu)

    actual = e8m0_to_uint8(scale)

    assert actual.dtype == torch.uint8
    assert actual.device.type == "cpu"
    assert actual.data_ptr() == scale.data_ptr()
    torch.testing.assert_close(actual, raw_bytes, rtol=0, atol=0)


@_requires_e8m0
def test_view_uint8_differs_from_numeric_uint8_conversion_for_small_scales() -> None:
    scale = _e8m0_from_bytes([126, 127, 128])

    raw_bytes = e8m0_to_uint8(scale)
    numeric_uint8 = scale.to(torch.uint8)

    torch.testing.assert_close(
        raw_bytes,
        torch.tensor([126, 127, 128], dtype=torch.uint8, device="cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        numeric_uint8,
        torch.tensor([0, 1, 2], dtype=torch.uint8, device="cpu"),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(raw_bytes, numeric_uint8)


@_requires_e8m0
def test_e8m0_to_fp32_decodes_exponent_bytes() -> None:
    scale = _e8m0_from_bytes([126, 127, 128])

    actual = e8m0_to_fp32(scale)

    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    torch.testing.assert_close(
        actual,
        torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )


@_requires_e8m0
def test_maybe_e8m0_to_fp32_decodes_e8m0() -> None:
    scale = _e8m0_from_bytes([126, 127, 128])

    actual = maybe_e8m0_to_fp32(scale)

    torch.testing.assert_close(
        actual,
        torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_maybe_e8m0_to_fp32_returns_non_e8m0_scales_as_fp32(dtype: torch.dtype) -> None:
    scale = torch.tensor([0.25, 1.5, 8.0], dtype=dtype, device="cpu")

    actual = maybe_e8m0_to_fp32(scale)

    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    torch.testing.assert_close(
        actual,
        torch.tensor([0.25, 1.5, 8.0], dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )


@_requires_e8m0
def test_e8m0_helpers_reject_non_e8m0_inputs() -> None:
    scale = torch.ones(2, dtype=torch.float32, device="cpu")

    with pytest.raises(TypeError, match="expected an torch.float8_e8m0fnu tensor"):
        e8m0_to_uint8(scale)

    with pytest.raises(TypeError, match="expected an torch.float8_e8m0fnu tensor"):
        e8m0_to_fp32(scale)


def test_required_e8m0_helpers_raise_clear_error_when_dtype_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e8m0, "_get_e8m0_dtype", lambda: None)

    with pytest.raises(RuntimeError, match="torch.float8_e8m0fnu is not available"):
        e8m0_to_uint8(torch.ones(1, dtype=torch.float32, device="cpu"))

    with pytest.raises(RuntimeError, match="torch.float8_e8m0fnu is not available"):
        e8m0_to_fp32(torch.ones(1, dtype=torch.float32, device="cpu"))


def test_maybe_e8m0_to_fp32_handles_non_e8m0_when_dtype_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e8m0, "_get_e8m0_dtype", lambda: None)
    scale = torch.tensor([1.25, 2.5], dtype=torch.bfloat16, device="cpu")

    actual = maybe_e8m0_to_fp32(scale)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.tensor([1.25, 2.5], dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )
