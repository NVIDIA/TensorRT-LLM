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
"""A planner says which deployments it supports; binding asks every planner and
installs the one that says yes. The architecture and the parallel topology are
the only inputs, so a built-in model and a modeling_v2 target bind alike."""

from pathlib import Path

import pytest
from transformers import PretrainedConfig

import tensorrt_llm._torch._experimental.modeling_v2 as modeling_v2
from tensorrt_llm._torch.disaggregation import planners
from tensorrt_llm._torch.disaggregation.planners.deepseek_mla_latent import (
    DeepseekMlaLatentTransferPlanner,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

DEEPSEEK = "DeepseekV3ForCausalLM"
DEP4 = Mapping(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True)
DEP8 = Mapping(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1, enable_attention_dp=True)
TP4 = Mapping(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=False)
DEP4_PP2 = Mapping(
    world_size=8, tp_size=4, pp_size=2, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True
)


def _model_config(architecture: str, mapping: Mapping) -> ModelConfig:
    return ModelConfig(
        pretrained_config=PretrainedConfig(architectures=[architecture]), mapping=mapping
    )


def test_mla_latent_planner_supports_attention_dp_deepseek_of_any_width():
    supports = DeepseekMlaLatentTransferPlanner.supports_deployment
    assert supports(DEEPSEEK, DEP4)
    assert supports(DEEPSEEK, DEP8)
    assert not supports("GptOssForCausalLM", DEP4)
    assert not supports(DEEPSEEK, TP4)
    assert not supports(DEEPSEEK, DEP4_PP2)


def test_planner_is_found_by_the_deployment_it_supports_and_nothing_else():
    assert planners.planner_for(DEEPSEEK, DEP4) is DeepseekMlaLatentTransferPlanner
    assert planners.planner_for("GptOssForCausalLM", Mapping()) is None


def test_no_target_source_names_a_planner():
    """The binding direction: planners judge deployments, never the reverse."""
    root = Path(modeling_v2.__file__).resolve().parent / "models"
    hits = [
        p.relative_to(root)
        for p in root.rglob("*.py")
        if "transfer_planner" in p.read_text() or "TransferPlanner" in p.read_text()
    ]
    assert hits == [], hits


def test_two_planners_supporting_one_deployment_is_an_error(monkeypatch):
    class Twin(DeepseekMlaLatentTransferPlanner):
        pass

    monkeypatch.setattr(planners, "PLANNERS", (DeepseekMlaLatentTransferPlanner, Twin))
    with pytest.raises(ValueError, match=DEEPSEEK):
        planners.planner_for(DEEPSEEK, DEP4)


class _StubModel:
    def build_transfer_planner(self, kv_cache_manager, self_rank_info, self_extractor):
        return "default"


def test_bind_leaves_an_unsupported_model_on_the_default_hook():
    stub = _StubModel()
    assert (
        planners.bind_transfer_planner(stub, _model_config("GptOssForCausalLM", Mapping())) is None
    )
    assert stub.build_transfer_planner(None, None, None) == "default"
    no_arch = ModelConfig(pretrained_config=PretrainedConfig())
    assert planners.bind_transfer_planner(stub, no_arch) is None


def test_bind_installs_the_supporting_planner_as_the_factory(monkeypatch):
    seen = {}

    def fake_init(self, self_rank_info, self_extractor):
        seen["args"] = (self_rank_info, self_extractor)

    monkeypatch.setattr(DeepseekMlaLatentTransferPlanner, "__init__", fake_init)
    stub = _StubModel()
    config = _model_config(DEEPSEEK, DEP4)
    assert planners.bind_transfer_planner(stub, config) is DeepseekMlaLatentTransferPlanner
    planner = stub.build_transfer_planner("kvm", "ri", "ext")
    assert isinstance(planner, DeepseekMlaLatentTransferPlanner)
    assert seen["args"] == ("ri", "ext")
