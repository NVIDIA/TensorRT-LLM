# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fused Wan DupUp3D output mapping."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan import dup_up3d as dup_up3d_module
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import DupUp3D


@pytest.mark.parametrize("first_chunk", [False, True])
@pytest.mark.parametrize("input_layout", ["contiguous", "channels_last_3d", "strided"])
@pytest.mark.parametrize(
    ("in_channels", "out_channels", "factor_t"),
    [
        pytest.param(4, 4, 2, id="temporal-repeat-8"),
        pytest.param(8, 4, 2, id="temporal-repeat-4"),
        pytest.param(8, 2, 2, id="temporal-repeat-2"),
        pytest.param(4, 4, 1, id="spatial-repeat-4"),
        pytest.param(8, 4, 1, id="spatial-repeat-2"),
    ],
)
def test_fused_dup_up3d_matches_eager(
    first_chunk: bool,
    input_layout: str,
    in_channels: int,
    out_channels: int,
    factor_t: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused DupUp3D kernel")

    input_width = 10 if input_layout == "strided" else 5
    x = torch.randn(
        2,
        in_channels,
        3,
        4,
        input_width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if input_layout == "channels_last_3d":
        x = x.contiguous(memory_format=torch.channels_last_3d)
    elif input_layout == "strided":
        x = x[..., ::2]
        assert not x.is_contiguous()
    module = DupUp3D(in_channels, out_channels, factor_t=factor_t, factor_s=2)

    expected = module(x.cpu(), first_chunk=first_chunk).cuda()
    actual = module(x, first_chunk=first_chunk)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.parametrize(
    ("shape", "first_chunk"),
    [
        ((0, 4, 2, 3, 4), False),
        ((1, 4, 0, 3, 4), True),
        ((1, 4, 2, 0, 4), False),
        ((1, 4, 2, 3, 0), False),
    ],
)
def test_fused_dup_up3d_empty_input_uses_eager(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    first_chunk: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the fused DupUp3D dispatch")

    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    module = DupUp3D(4, 4, factor_t=2, factor_s=2)
    expected = module(x.cpu(), first_chunk=first_chunk).cuda()

    def fail_if_launched(*args, **kwargs):
        pytest.fail("Empty inputs must bypass the fused DupUp3D kernel")

    monkeypatch.setattr(dup_up3d_module, "dup_up3d", fail_if_launched)
    actual = module(x, first_chunk=first_chunk)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_dup_up3d_int32_index_limit() -> None:
    assert dup_up3d_module._supports_triton_indexing(1 << 31)
    assert not dup_up3d_module._supports_triton_indexing((1 << 31) + 1)


def test_fused_dup_up3d_falls_back_above_index_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the fused DupUp3D dispatch")

    x = torch.randn(
        1,
        4,
        2,
        3,
        4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    module = DupUp3D(4, 4, factor_t=2, factor_s=2)
    expected = module(x.cpu(), first_chunk=True).cuda()

    monkeypatch.setattr(dup_up3d_module, "_MAX_TRITON_INDEXED_ELEMENTS", 0)
    actual = module(x, first_chunk=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
