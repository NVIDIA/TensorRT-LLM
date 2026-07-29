# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for native Wan VAE halo convolution geometry."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import WanCausalConvHalo
from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanCausalConv3d


@pytest.mark.parametrize(
    ("chunk_dim", "spatial_padding"),
    [
        (3, (0, 1)),
        (4, (1, 0)),
    ],
)
def test_native_halo_conv_emits_local_output_without_strip(
    monkeypatch: pytest.MonkeyPatch,
    chunk_dim: int,
    spatial_padding: tuple[int, int],
) -> None:
    conv = WanCausalConv3d(4, 4, 3, padding=1).float()
    halo = WanCausalConvHalo(conv, chunk_dim, [None], rank=0, world_size=2)
    x = torch.randn(1, 4, 3, 8, 8)
    reference = conv(x)

    def exchange_with_zero_boundaries(tensor: torch.Tensor) -> torch.Tensor:
        padding = [0, 0, 0, 0, 0, 0]
        padding_index = 2 * (4 - chunk_dim)
        padding[padding_index : padding_index + 2] = [1, 1]
        return torch.nn.functional.pad(tensor, padding)

    monkeypatch.setattr(halo, "_exchange_halos", exchange_with_zero_boundaries)
    monkeypatch.setattr(
        halo,
        "_strip_halo",
        lambda _: pytest.fail("native Wan halo Conv should emit local-width output directly"),
    )

    output = halo(x)

    assert halo._local_output_spatial_padding == spatial_padding
    torch.testing.assert_close(output, reference)


@pytest.mark.parametrize(
    ("stride", "dilation", "kernel_size"),
    [
        ((1, 1, 2), 1, 3),
        (1, (1, 1, 2), 3),
        (1, 1, (3, 3, 2)),
    ],
)
def test_native_halo_conv_falls_back_for_unsupported_geometry(
    monkeypatch: pytest.MonkeyPatch,
    stride: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int],
    kernel_size: int | tuple[int, int, int],
) -> None:
    conv = WanCausalConv3d(4, 4, kernel_size, stride=stride, padding=1).float()
    conv.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3

    halo = WanCausalConvHalo(conv, chunk_dim=4, adj_groups=[None], rank=0, world_size=2)

    assert halo._local_output_spatial_padding is None

    strip_called = False

    def strip_output(output: torch.Tensor) -> torch.Tensor:
        nonlocal strip_called
        strip_called = True
        return output

    monkeypatch.setattr(halo, "_exchange_halos", lambda tensor: tensor)
    monkeypatch.setattr(halo, "_strip_halo", strip_output)

    halo(torch.randn(1, 4, 3, 8, 8))

    assert strip_called
