# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Wan timestep modulation routing."""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

pytestmark = pytest.mark.cpu_only


def _routing_model(expand_timesteps: bool, token_sharded_tp: bool = False):
    """A Wan model whose forward runs on mocks (sharder, embedders, blocks), B=2, S=3, D=8."""
    hidden_size = 8

    model = WanTransformer3DModel.__new__(WanTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        patch_size=(1, 1, 1),
        hidden_size=hidden_size,
        expand_timesteps=expand_timesteps,
    )
    model.rope = mock.Mock(return_value=(torch.empty(0), torch.empty(0)))
    model.patch_embedding = torch.nn.Identity()
    model.sharder = mock.Mock()
    model.sharder.shard.side_effect = lambda tensor, **_kwargs: tensor
    model.sharder.shard_rope.return_value = None
    model.sharder.gather.side_effect = lambda tensor, **_kwargs: tensor
    model.sharder.shard_per_sample.side_effect = lambda tensor: tensor
    model.sharder.token_sharded_tp = token_sharded_tp

    def condition_embedder(
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
        timestep_seq_len: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if timestep_seq_len is None:
            temb_shape = (timestep.shape[0], hidden_size)
        else:
            temb_shape = (
                timestep.shape[0] // timestep_seq_len,
                timestep_seq_len,
                hidden_size,
            )
        temb = torch.zeros(temb_shape)
        temb_proj = torch.zeros((*temb_shape[:-1], 6 * hidden_size))
        return temb, temb_proj, encoder_hidden_states, encoder_hidden_states_image

    model.condition_embedder = mock.Mock(side_effect=condition_embedder)
    model._pertoken_adaln_runtime = mock.Mock()
    model._pertoken_adaln_runtime.prepare.side_effect = lambda x, _temb: x
    model.blocks = torch.nn.ModuleList()
    model.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 2, hidden_size))
    model.norm_out = torch.nn.Identity()
    model.proj_out = torch.nn.Identity()
    model.unpatchify = mock.Mock(side_effect=lambda tensor, _shape: tensor)
    return model


def _run(model) -> torch.Tensor:
    timestep = torch.tensor([0.25, 0.75])
    model(
        hidden_states=torch.zeros(2, 8, 1, 1, 3),
        timestep=timestep,
        encoder_hidden_states=torch.zeros(2, 1, 8),
    )
    return timestep


@pytest.mark.parametrize(
    ("expand_timesteps", "expected_temb_shape"),
    [(False, (2, 6, 8)), (True, (2, 3, 6, 8))],
)
def test_uniform_timestep_modulation_routing(
    expand_timesteps: bool, expected_temb_shape: tuple[int, ...]
) -> None:
    """Wan 5B expands T2V modulation only after the condition embedder."""
    model = _routing_model(expand_timesteps)
    timestep = _run(model)

    timestep_call = model.condition_embedder.call_args
    assert timestep_call.kwargs["timestep_seq_len"] is None
    torch.testing.assert_close(timestep_call.args[0], timestep * 1000)

    runtime_call = model._pertoken_adaln_runtime.prepare.call_args
    assert runtime_call.args[1].shape == expected_temb_shape
    if expand_timesteps:
        assert runtime_call.args[1].stride(1) == 0


@pytest.mark.parametrize("expand_timesteps", [False, True])
def test_token_sharded_tp_keeps_the_per_sample_table(expand_timesteps: bool) -> None:
    """Under token-sharded TP a uniform timestep keeps its per-sample table (the sharder
    gives each shard its samples' entries; no [B, S] broadcast) and the per-token AdaLN
    kernels stay off."""
    model = _routing_model(expand_timesteps, token_sharded_tp=True)
    _run(model)

    model._pertoken_adaln_runtime.prepare.assert_not_called()
    (table,), _ = model.sharder.shard_per_sample.call_args
    assert table.shape == (2, 6, 8)
