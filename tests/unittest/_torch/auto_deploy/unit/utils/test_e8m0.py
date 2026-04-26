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

import inspect

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.utils import e8m0
from tensorrt_llm._torch.auto_deploy.utils.e8m0 import (
    e8m0_to_fp32,
    e8m0_to_uint8,
    e8m0_uint8_to_fp32,
    fp8_block_dequant_ref,
    fp8_block_quant_ref,
    fp8_block_scale_shape,
    fp32_to_e8m0,
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


def test_e8m0_to_uint8_helper_does_not_use_numeric_uint8_conversion() -> None:
    source = inspect.getsource(e8m0_to_uint8)

    assert ".to(torch.uint8)" not in source
    assert ".view(torch.uint8)" in source


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
def test_fp32_to_e8m0_emits_raw_exponent_bytes_not_numeric_values() -> None:
    scale = torch.tensor([0.25, 1.25, 2.0], dtype=torch.float32, device="cpu")

    actual = fp32_to_e8m0(scale)

    assert actual.dtype == torch.float8_e8m0fnu
    torch.testing.assert_close(
        e8m0_to_uint8(actual),
        torch.tensor([125, 128, 128], dtype=torch.uint8, device="cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        e8m0_to_fp32(actual),
        torch.tensor([0.25, 2.0, 2.0], dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )


def test_e8m0_uint8_to_fp32_decodes_raw_exponent_bytes() -> None:
    raw_bytes = torch.tensor([0, 1, 126, 127, 128, 254, 255], dtype=torch.uint8, device="cpu")

    actual = e8m0_uint8_to_fp32(raw_bytes)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual[:-1],
        torch.tensor(
            [
                2.0**-127,
                2.0**-126,
                0.5,
                1.0,
                2.0,
                2.0**127,
            ],
            dtype=torch.float32,
            device="cpu",
        ),
        rtol=0,
        atol=0,
    )
    assert torch.isnan(actual[-1])


@_requires_e8m0
def test_e8m0_to_fp32_matches_torch_decode_for_special_bytes() -> None:
    scale = _e8m0_from_bytes([0, 1, 126, 127, 128, 254, 255])

    actual = e8m0_to_fp32(scale)

    torch.testing.assert_close(actual, scale.to(torch.float32), rtol=0, atol=0, equal_nan=True)


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


def test_fp32_to_e8m0_fallback_returns_decoded_powers_when_dtype_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e8m0, "_get_e8m0_dtype", lambda: None)
    scale = torch.tensor([0.25, 1.25, 2.0], dtype=torch.float32, device="cpu")

    actual = fp32_to_e8m0(scale)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual,
        torch.tensor([0.25, 2.0, 2.0], dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("input_shape", "scale_shape"),
    [
        ((2, 448), (2, 4)),
        ((1024, 4096), (1024, 32)),
        ((256, 4096), (256, 32)),
        ((4096, 256), (4096, 2)),
    ],
)
def test_fp8_block_scale_shape_uses_exact_tail_block_count(
    input_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
) -> None:
    assert fp8_block_scale_shape(input_shape, block_size=128) == scale_shape


def test_fp8_block_quant_dequant_handles_nope_dim_448_tail_block() -> None:
    x = torch.zeros((2, 448), dtype=torch.float32, device="cpu")
    x[:, 384:] = torch.linspace(-224.0, 224.0, 64, dtype=torch.float32, device="cpu")

    x_fp8, scale = fp8_block_quant_ref(x, block_size=128)
    actual = fp8_block_dequant_ref(x_fp8, scale, block_size=128, dtype=torch.float32)

    assert x_fp8.shape == x.shape
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 4)
    scale_fp32 = maybe_e8m0_to_fp32(scale)
    torch.testing.assert_close(
        scale_fp32[:, -1],
        torch.full((2,), 0.5, dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(actual[:, :384], x[:, :384], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, -1], x[:, -1], rtol=0, atol=0)


@_requires_e8m0
def test_fp8_block_quant_scales_store_raw_e8m0_exponent_bytes() -> None:
    x = torch.tensor([[112.0, 224.0, 448.0, 0.0]], dtype=torch.float32, device="cpu")

    _, scale = fp8_block_quant_ref(x, block_size=2)

    assert scale.dtype == torch.float8_e8m0fnu
    torch.testing.assert_close(
        e8m0_to_uint8(scale),
        torch.tensor([[126, 127]], dtype=torch.uint8, device="cpu"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        e8m0_to_fp32(scale),
        torch.tensor([[0.5, 1.0]], dtype=torch.float32, device="cpu"),
        rtol=0,
        atol=0,
    )


def test_fp8_block_dequant_decodes_raw_uint8_e8m0_scale_bytes() -> None:
    x_fp8 = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]],
        dtype=torch.float32,
        device="cpu",
    ).to(torch.float8_e4m3fn)
    raw_scale = torch.tensor([[126, 127, 128]], dtype=torch.uint8, device="cpu")

    actual = fp8_block_dequant_ref(x_fp8, raw_scale, block_size=2, dtype=torch.float32)

    decoded_scale = e8m0_uint8_to_fp32(raw_scale).repeat_interleave(2, dim=-1)
    expected = x_fp8.to(torch.float32) * decoded_scale
    numeric_uint8_scale = raw_scale.to(torch.float32).repeat_interleave(2, dim=-1)
    numeric_uint8_expected = x_fp8.to(torch.float32) * numeric_uint8_scale
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, numeric_uint8_expected)


@pytest.mark.parametrize("shape", [(1024, 4096), (256, 4096), (4096, 256)])
def test_fp8_block_quant_dequant_shapes_for_observed_local_weight_shapes(
    shape: tuple[int, int],
) -> None:
    x = torch.zeros(shape, dtype=torch.bfloat16, device="cpu")

    x_fp8, scale = fp8_block_quant_ref(x, block_size=128)
    actual = fp8_block_dequant_ref(x_fp8, scale, block_size=128, dtype=torch.bfloat16)

    assert x_fp8.shape == shape
    assert scale.shape == fp8_block_scale_shape(shape, block_size=128)
    assert actual.shape == shape
    assert actual.dtype == torch.bfloat16


@_requires_e8m0
@pytest.mark.parametrize(
    "scale_shape",
    [
        (8, 32),
        (2, 32),
        (32, 2),
    ],
)
def test_e8m0_helpers_cover_observed_finegrained_fp8_scale_shapes(
    scale_shape: tuple[int, int],
) -> None:
    raw_bytes = torch.full(scale_shape, 127, dtype=torch.uint8, device="cpu")
    scale = raw_bytes.view(torch.float8_e8m0fnu)

    raw_actual = e8m0_to_uint8(scale)
    fp32_actual = e8m0_to_fp32(scale)

    assert raw_actual.shape == scale_shape
    assert fp32_actual.shape == scale_shape
    torch.testing.assert_close(raw_actual, raw_bytes, rtol=0, atol=0)
    torch.testing.assert_close(
        fp32_actual,
        torch.ones(scale_shape, dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_fp8_block_quant_all_zero_blocks_are_finite() -> None:
    x = torch.zeros((3, 448), dtype=torch.float32, device="cpu")

    x_fp8, scale = fp8_block_quant_ref(x, block_size=128)
    actual = fp8_block_dequant_ref(x_fp8, scale, block_size=128, dtype=torch.float32)
    scale_fp32 = maybe_e8m0_to_fp32(scale)

    assert torch.isfinite(x_fp8.to(torch.float32)).all()
    assert torch.isfinite(scale_fp32).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(scale_fp32, torch.ones((3, 4), dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_fp8_block_quant_fallback_uses_fp32_scales_when_e8m0_dtype_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e8m0, "_get_e8m0_dtype", lambda: None)
    x = torch.zeros((1, 448), dtype=torch.float32, device="cpu")
    x[:, -1] = 448.0

    x_fp8, scale = fp8_block_quant_ref(x, block_size=128)
    actual = fp8_block_dequant_ref(x_fp8, scale, block_size=128, dtype=torch.float32)

    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (1, 4)
    assert torch.isfinite(scale).all()
    torch.testing.assert_close(actual[:, -1], x[:, -1], rtol=0, atol=0)
