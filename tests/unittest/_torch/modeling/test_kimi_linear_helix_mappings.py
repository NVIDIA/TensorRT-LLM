# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for KimiLinearForCausalLM._setup_helix_mappings precondition
validation and dual-mapping swap/restore logic."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

pytestmark = pytest.mark.cpu_only


def _make_model_config(mapping, **kwargs):
    return ModelConfig(mapping=mapping, quant_config=QuantConfig(), **kwargs)


def _make_cfg(num_attention_heads=96, kda_num_heads=96, num_experts=896):
    return SimpleNamespace(
        num_attention_heads=num_attention_heads,
        linear_attn_config={"num_heads": kda_num_heads},
        num_experts=num_experts,
    )


def _make_helix_mapping(tp_size=4, cp_size=2, rank=0, enable_attention_dp=False):
    mapping = Mapping(
        world_size=tp_size * cp_size,
        rank=rank,
        tp_size=tp_size,
        cp_size=cp_size,
        enable_attention_dp=enable_attention_dp,
    )
    return mapping


def _set_moe_attrs(mapping, moe_tp_size, moe_ep_size):
    mapping.moe_tp_size = moe_tp_size
    mapping.moe_ep_size = moe_ep_size
    mapping.moe_tp_ep_user_specified = True


def _make_non_helix_mapping():
    return Mapping(world_size=1, rank=0, tp_size=1)


def _get_setup_method():
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM

    return KimiLinearForCausalLM._setup_helix_mappings


class _FakeSelf:
    mapping_with_cp = None
    _repurposed_tp_mapping = None


def test_setup_helix_mappings_precondition_validation():
    setup = _get_setup_method()

    # enable_attention_dp raises
    mapping = _make_helix_mapping(tp_size=4, cp_size=2, enable_attention_dp=True)
    model_config = _make_model_config(mapping)
    cfg = _make_cfg(num_attention_heads=96, kda_num_heads=96)
    obj = _FakeSelf()
    with patch.object(type(mapping), "has_cp_helix", return_value=True):
        with pytest.raises(ValueError, match="enable_attention_dp"):
            setup(obj, model_config, cfg, None)

    # spec_config not None raises
    mapping = _make_helix_mapping(tp_size=4, cp_size=2, enable_attention_dp=False)
    model_config = _make_model_config(mapping)
    cfg = _make_cfg(num_attention_heads=96, kda_num_heads=96)
    obj = _FakeSelf()
    spec_config = SimpleNamespace()
    with patch.object(type(mapping), "has_cp_helix", return_value=True):
        with pytest.raises(ValueError, match="speculative"):
            setup(obj, model_config, cfg, spec_config)

    # num_attention_heads not divisible by tp*cp raises
    mapping = _make_helix_mapping(tp_size=4, cp_size=2)
    model_config = _make_model_config(mapping)
    cfg = _make_cfg(num_attention_heads=7, kda_num_heads=96)
    obj = _FakeSelf()
    with patch.object(type(mapping), "has_cp_helix", return_value=True):
        with pytest.raises(ValueError, match="MLA head count"):
            setup(obj, model_config, cfg, None)

    # KDA num_heads not divisible by tp*cp raises
    mapping = _make_helix_mapping(tp_size=4, cp_size=2)
    model_config = _make_model_config(mapping)
    cfg = _make_cfg(num_attention_heads=96, kda_num_heads=7)
    obj = _FakeSelf()
    with patch.object(type(mapping), "has_cp_helix", return_value=True):
        with pytest.raises(ValueError, match="KDA head count"):
            setup(obj, model_config, cfg, None)

    # moe_tp*moe_ep != tp*cp raises
    mapping = _make_helix_mapping(tp_size=4, cp_size=2)
    _set_moe_attrs(mapping, moe_tp_size=2, moe_ep_size=2)
    model_config = _make_model_config(mapping)
    cfg = _make_cfg(num_attention_heads=96, kda_num_heads=96, num_experts=896)
    obj = _FakeSelf()
    with patch.object(type(mapping), "has_cp_helix", return_value=True):
        with pytest.raises(ValueError, match="moe_tensor_parallel_size"):
            setup(obj, model_config, cfg, None)

    # num_experts not divisible by moe_ep raises (default EP = tp*cp = 8)
    mapping = _make_helix_mapping(tp_size=4, cp_size=2)
    model_config = _make_model_config(mapping)
    cfg = _make_cfg(num_attention_heads=96, kda_num_heads=96, num_experts=7)
    obj = _FakeSelf()
    with patch.object(type(mapping), "has_cp_helix", return_value=True):
        with pytest.raises(ValueError, match="MoE EP size"):
            setup(obj, model_config, cfg, None)

    # non-helix returns without raising, leaves None
    mapping = _make_non_helix_mapping()
    model_config = _make_model_config(mapping)
    cfg = _make_cfg()
    obj = _FakeSelf()
    setup(obj, model_config, cfg, None)
    assert obj.mapping_with_cp is None
    assert obj._repurposed_tp_mapping is None


def test_setup_helix_mappings_dual_mapping_swap_restore():
    setup = _get_setup_method()

    tp_size = 4
    cp_size = 2
    repurposed_tp = tp_size * cp_size  # 8
    rank = 3

    mapping = _make_helix_mapping(tp_size=tp_size, cp_size=cp_size, rank=rank)
    model_config = _make_model_config(mapping)
    cfg = _make_cfg(
        num_attention_heads=96,
        kda_num_heads=96,
        num_experts=896,
    )

    original_tp_size = model_config.mapping.tp_size
    original_cp_size = model_config.mapping.cp_size
    original_moe_tp_ep_user_specified = model_config.mapping.moe_tp_ep_user_specified

    obj = _FakeSelf()

    def fake_repurpose(self_mapping):
        repurposed = Mapping(
            world_size=repurposed_tp,
            rank=rank,
            tp_size=repurposed_tp,
            cp_size=1,
        )
        repurposed.moe_tp_ep_user_specified = True
        return repurposed

    with patch.object(type(mapping), "has_cp_helix", return_value=True):
        with patch.object(type(mapping), "repurpose_helix_cp_to_tp", fake_repurpose):
            setup(obj, model_config, cfg, None)

    # After _setup_helix_mappings, model_config.mapping is the repurposed mapping
    assert model_config.mapping.tp_size == repurposed_tp
    assert model_config.mapping.cp_size == 1

    # _helix_mapping_with_cp is the deep-copied CP original
    assert hasattr(model_config, "_helix_mapping_with_cp")
    assert model_config._helix_mapping_with_cp.tp_size == original_tp_size
    assert model_config._helix_mapping_with_cp.cp_size == original_cp_size

    # self.mapping_with_cp equals the CP original
    assert obj.mapping_with_cp is not None
    assert obj.mapping_with_cp.tp_size == original_tp_size
    assert obj.mapping_with_cp.cp_size == original_cp_size

    # Simulate the restore that __init__ performs after super().__init__
    model_config._frozen = False
    model_config.mapping = obj.mapping_with_cp
    model_config._frozen = True
    assert model_config.mapping.tp_size == original_tp_size
    assert model_config.mapping.cp_size == original_cp_size

    # _repurposed_tp_mapping.moe_tp_ep_user_specified equals the CP original's flag
    assert obj._repurposed_tp_mapping is not None
    assert obj._repurposed_tp_mapping.moe_tp_ep_user_specified == original_moe_tp_ep_user_specified

    # --- _load_trunk_params tp_rank selection via production code ---
    import tensorrt_llm._torch.models.modeling_kimi_linear as prod_mod

    def _capture_run_concurrently(fn, jobs, *args, **kwargs):
        for job in jobs:
            fn(*job)

    shard_size = 4
    full_size = shard_size * repurposed_tp  # 32
    fake_weights = {
        "model.layers.0.mlp.down_proj.weight": torch.arange(
            4 * full_size, dtype=torch.bfloat16
        ).reshape(4, full_size)
    }
    sharded_param_helix = torch.nn.Parameter(torch.zeros(4, shard_size, dtype=torch.bfloat16))
    sharded_name = "model.layers.0.mlp.down_proj.weight"
    sharded_ckpt_key = "model.layers.0.mlp.down_proj.weight"

    class _MinimalInstanceHelix:
        _repurposed_tp_mapping = obj._repurposed_tp_mapping
        model_config = SimpleNamespace(mapping=SimpleNamespace(tp_rank=0))
        model = SimpleNamespace(layers=[])

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    instance_helix = _MinimalInstanceHelix()

    with patch.object(prod_mod, "run_concurrently", _capture_run_concurrently):
        with patch.object(
            prod_mod, "_resolve_fp8_weight_read_gates", return_value=(False, False, False)
        ):
            prod_mod.KimiLinearForCausalLM._load_trunk_params(
                instance_helix,
                fake_weights,
                {sharded_name: sharded_param_helix},
                {sharded_name: sharded_ckpt_key},
            )

    expected_helix_tp_rank = obj._repurposed_tp_mapping.tp_rank
    expected_lo = (expected_helix_tp_rank % repurposed_tp) * shard_size
    expected_slice = fake_weights[sharded_ckpt_key][:, expected_lo : expected_lo + shard_size]
    assert torch.equal(sharded_param_helix.data, expected_slice.to(sharded_param_helix.dtype))

    # Non-helix branch: _repurposed_tp_mapping is None -> uses model_config.mapping.tp_rank
    sharded_param_no_helix = torch.nn.Parameter(torch.zeros(4, shard_size, dtype=torch.bfloat16))

    class _MinimalInstanceNoHelix:
        _repurposed_tp_mapping = None
        model_config = SimpleNamespace(mapping=SimpleNamespace(tp_rank=0))
        model = SimpleNamespace(layers=[])

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    instance_no_helix = _MinimalInstanceNoHelix()

    with patch.object(prod_mod, "run_concurrently", _capture_run_concurrently):
        with patch.object(
            prod_mod, "_resolve_fp8_weight_read_gates", return_value=(False, False, False)
        ):
            prod_mod.KimiLinearForCausalLM._load_trunk_params(
                instance_no_helix,
                fake_weights,
                {sharded_name: sharded_param_no_helix},
                {sharded_name: sharded_ckpt_key},
            )

    expected_slice_nh = fake_weights[sharded_ckpt_key][:, :shard_size]
    assert torch.equal(
        sharded_param_no_helix.data, expected_slice_nh.to(sharded_param_no_helix.dtype)
    )
