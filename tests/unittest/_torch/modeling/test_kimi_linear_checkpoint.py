# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-name and weight-shard tests for the Kimi Linear model."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("fla")

from tensorrt_llm._torch.models.modeling_kimi_linear import (  # noqa: E402
    KimiLinearForCausalLM,
    _helix_cp_v_b_shard,
    _shard_head_major_param,
)


def test_checkpoint_plan_preserves_external_attention_names():
    class _PlanHarness:
        checkpoint_name_plan = KimiLinearForCausalLM.checkpoint_name_plan
        model = SimpleNamespace(layers=[])

        def _trunk_parameters(self):
            return {
                "model.layers.0.linear_attn.q_proj.weight": torch.empty(0),
                "model.layers.1.self_attn.mixer.q_a_proj.weight": torch.empty(0),
                "lm_head.weight": torch.empty(0),
            }

    name_map, expected_keys, expert_jobs = _PlanHarness().checkpoint_name_plan("language_model.")

    assert name_map == {
        "model.layers.0.linear_attn.q_proj.weight": (
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ),
        "model.layers.1.self_attn.mixer.q_a_proj.weight": (
            "language_model.model.layers.1.self_attn.q_a_proj.weight"
        ),
        "lm_head.weight": "language_model.lm_head.weight",
    }
    assert expected_keys == set(name_map.values())
    assert expert_jobs == []


def _distinct(*shape: int) -> torch.Tensor:
    """A contiguous tensor with a distinct non-zero value at every position,
    so a wrong-rank slice is never equal to the right one (a no-op shard is
    also distinguishable from a correct slice)."""
    n = 1
    for d in shape:
        n *= d
    return torch.arange(1, n + 1, dtype=torch.float32).reshape(shape)


@pytest.mark.parametrize("kda_tp_size,kda_tp_rank", [(2, 0), (2, 1), (4, 3)])
def test_shard_kda_column_projection(kda_tp_size, kda_tp_rank):
    # Every head-major KDA projection except o_proj (q/k/v/g/f_b/b, conv,
    # dt_bias) is COLUMN-sharded on its output rows (dim 0) by kda_tp_size.
    local = 4
    src = _distinct(local * kda_tp_size, 8)
    param = torch.nn.Parameter(torch.empty(local, 8))
    out = _shard_head_major_param(
        "model.layers.0.linear_attn.q_proj.weight",
        src,
        param,
        kda_tp_size=kda_tp_size,
        kda_tp_rank=kda_tp_rank,
        model_tp_rank=0,
    )
    expected = src[kda_tp_rank * local : (kda_tp_rank + 1) * local]
    assert out.shape == param.shape
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("kda_tp_size,kda_tp_rank", [(2, 0), (2, 1), (4, 2)])
def test_shard_kda_o_proj_row(kda_tp_size, kda_tp_rank):
    # o_proj alone is ROW-sharded on its input columns (dim 1) by kda_tp_size.
    local = 4
    src = _distinct(6, local * kda_tp_size)
    param = torch.nn.Parameter(torch.empty(6, local))
    out = _shard_head_major_param(
        "model.layers.0.linear_attn.o_proj.weight",
        src,
        param,
        kda_tp_size=kda_tp_size,
        kda_tp_rank=kda_tp_rank,
        model_tp_rank=0,
    )
    expected = src[:, kda_tp_rank * local : (kda_tp_rank + 1) * local]
    assert out.shape == param.shape
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("scope", [".shared_experts.", ".mlp."])
@pytest.mark.parametrize("model_tp_rank", [0, 1, 3])
def test_shard_mlp_down_proj_row(scope, model_tp_rank):
    # down_proj is ROW-sharded on its input columns; tp comes from the shapes
    # and the shard index repeats modulo the parameter's shard count.
    local, tp = 4, 2
    src = _distinct(6, local * tp)
    param = torch.nn.Parameter(torch.empty(6, local))
    out = _shard_head_major_param(
        f"model.layers.3{scope}down_proj.weight",
        src,
        param,
        kda_tp_size=1,
        kda_tp_rank=0,
        model_tp_rank=model_tp_rank,
    )
    rank = model_tp_rank % tp
    expected = src[:, rank * local : (rank + 1) * local]
    assert out.shape == param.shape
    torch.testing.assert_close(out, expected)


def test_shard_passthrough_when_shape_matches():
    # Replicated KDA projections (f_a/g_a, o_norm) already match the param and
    # must be returned untouched.
    src = _distinct(4, 8)
    param = torch.nn.Parameter(torch.empty(4, 8))
    out = _shard_head_major_param(
        "model.layers.0.linear_attn.f_a_proj.weight",
        src,
        param,
        kda_tp_size=2,
        kda_tp_rank=1,
        model_tp_rank=0,
    )
    assert out is src


def test_shard_passthrough_for_mla_names():
    # MLA (.self_attn.) tensors are head-sharded by their own Linear modules, so
    # a shape mismatch here must pass through (not be treated as KDA/down_proj).
    src = _distinct(8, 4)
    param = torch.nn.Parameter(torch.empty(4, 4))
    out = _shard_head_major_param(
        "model.layers.1.self_attn.q_b_proj.weight",
        src,
        param,
        kda_tp_size=1,
        kda_tp_rank=0,
        model_tp_rank=0,
    )
    assert out is src


def test_shard_down_proj_non_divisible_raises():
    # The divisibility guard turns a misaligned checkpoint into a clear error
    # instead of a floor-tp silent misshard.
    src = _distinct(6, 6)
    param = torch.nn.Parameter(torch.empty(6, 4))  # 6 % 4 != 0
    with pytest.raises(AssertionError):
        _shard_head_major_param(
            "model.layers.3.mlp.down_proj.weight",
            src,
            param,
            kda_tp_size=1,
            kda_tp_rank=0,
            model_tp_rank=0,
        )


@pytest.mark.parametrize("cp_rank", [0, 1, 3])
def test_helix_cp_v_b_shard_slices_this_rank(cp_rank):
    # Helix cp_size=4: v_b_proj keeps only this rank's 1/cp head chunk, while
    # kv_b_proj / k_b_proj_trans (built from the un-sliced v_weight) keep every
    # tp-local head. This slice only fires when cp_size > 1, so no cp_size=1
    # smoke test exercises it.
    num_heads_tp, num_heads_tp_cp = 8, 2  # cp_size == 8 / 2 == 4
    v_weight = _distinct(num_heads_tp, 3, 5)  # [heads, v_head_dim, kv_lora_rank]
    out = _helix_cp_v_b_shard(v_weight, num_heads_tp_cp=num_heads_tp_cp, cp_rank=cp_rank)
    expected = v_weight[cp_rank * num_heads_tp_cp : (cp_rank + 1) * num_heads_tp_cp]
    assert out.shape[0] == num_heads_tp_cp
    torch.testing.assert_close(out, expected)


def test_helix_cp_v_b_shard_noop_without_cp():
    # cp_size == 1: num_heads_tp_cp equals the full tp-local head count, so the
    # tensor is returned untouched (no accidental slicing).
    v_weight = _distinct(8, 3, 5)
    out = _helix_cp_v_b_shard(v_weight, num_heads_tp_cp=8, cp_rank=0)
    assert out is v_weight
