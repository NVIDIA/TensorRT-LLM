# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helix padded-head exchange against a dense rank-ordered softmax reference."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.helix import (
    helix_target_combine,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping


@torch.inference_mode()
def test_padded_target_heads_and_empty_shards():
    world, rank = mpi_world_size(), mpi_rank()
    if world != 4:
        pytest.skip("Requires four MPI GPU ranks")
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=world,
        rank=rank,
        tp_size=1,
        cp_size=world,
        cp_config={"cp_type": "HELIX", "use_nccl_for_alltoall": True},
        gpus_per_node=4,
    )
    generator = torch.Generator(device="cuda").manual_seed(42)
    rows, heads, padded, latent = 5, 96, 128, 512
    partials = torch.randn(
        world, rows, padded, latent, generator=generator, device="cuda"
    ).bfloat16()
    stats = torch.randn(world, rows, padded, 2, generator=generator, device="cuda")
    stats[..., 1] = stats[..., 1].abs() + 1
    # Some queries see no keys on a rank; discarded heads also carry poison.
    empty = torch.zeros(world, rows, device="cuda", dtype=torch.bool)
    empty[1:, 0] = True
    empty[2:, 1] = True
    partials[empty] = float("nan")
    partials[:, :, heads:] = float("nan")
    metadata = SimpleNamespace(helix_kv_bounds=(~empty[rank]).int(), _helix_spec_tokens_valid=True)
    reference_stats = stats[:, :, :heads].clone()
    reference_stats[..., 0].masked_fill_(empty[:, :, None], -torch.inf)
    reference_stats[..., 1].masked_fill_(empty[:, :, None], 0)
    expected = helix_target_combine.reference(partials[:, :, :heads], reference_stats, rank)
    with nccl_window_tensor_scope(partials):
        actual = helix_target_combine(
            partials[rank].flatten(1), stats[rank], mapping, heads, latent, metadata, None, None
        )
        torch.testing.assert_close(actual.view_as(expected), expected, atol=0.02, rtol=0.02)
