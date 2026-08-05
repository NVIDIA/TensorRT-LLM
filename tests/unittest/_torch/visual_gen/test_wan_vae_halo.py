# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for native Wan VAE halo convolution geometry."""

from unittest import mock

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
    halo = WanCausalConvHalo(
        conv,
        chunk_dim,
        [mock.Mock(spec=torch.distributed.ProcessGroup)],
        rank=0,
        world_size=2,
    )
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


def test_unsafe_geometry_disables_local_output() -> None:
    """Assert fast-path selection only, not support for this stride-2 geometry."""
    conv = WanCausalConv3d(4, 4, 3, stride=(1, 1, 2), padding=1).float()
    halo = WanCausalConvHalo(
        conv,
        chunk_dim=4,
        adj_groups=[mock.Mock(spec=torch.distributed.ProcessGroup)],
        rank=0,
        world_size=2,
    )

    assert halo._local_output_spatial_padding is None
