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
"""A planner covers deployments; a deployment is the architecture plus its parallel
topology, which a built-in model and a modeling_v2 target describe identically."""

from dataclasses import replace
from pathlib import Path

import pytest
from transformers import PretrainedConfig

import tensorrt_llm._torch._experimental.modeling_v2 as modeling_v2
from tensorrt_llm._torch.disaggregation import planners
from tensorrt_llm._torch.disaggregation.planners.deepseek_mla_latent import (
    DeepseekMlaLatentTransferPlanner,
)
from tensorrt_llm._torch.disaggregation.planners.deployment import Deployment
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

DEEPSEEK_DEP4 = Deployment(
    architecture="DeepseekV3ForCausalLM", tp=4, pp=1, cp=1, moe_ep=4, moe_tp=1, attention_dp=True
)
GPT_OSS_TP1 = Deployment(
    architecture="GptOssForCausalLM", tp=1, pp=1, cp=1, moe_ep=1, moe_tp=1, attention_dp=False
)
_DEP4_MAPPING = dict(
    world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True
)


def _model_config(architecture: str, **mapping_kwargs) -> ModelConfig:
    return ModelConfig(
        pretrained_config=PretrainedConfig(architectures=[architecture]),
        mapping=Mapping(**mapping_kwargs),
    )


def test_deployment_reads_the_architecture_and_the_parallel_topology():
    assert Deployment.of(_model_config("DeepseekV3ForCausalLM", **_DEP4_MAPPING)) == DEEPSEEK_DEP4
    assert Deployment.of(_model_config("GptOssForCausalLM")) == GPT_OSS_TP1
    assert Deployment.of(ModelConfig(pretrained_config=PretrainedConfig())) is None


def test_deployment_survives_the_wire():
    assert Deployment.from_bytes(DEEPSEEK_DEP4.to_bytes()) == DEEPSEEK_DEP4
    assert Deployment.from_bytes(b"") is None
    assert Deployment.from_bytes(b"not json") is None


def test_planner_is_found_by_the_deployment_it_covers_and_nothing_else():
    assert planners.planner_for(DEEPSEEK_DEP4) is DeepseekMlaLatentTransferPlanner
    assert planners.planner_for(GPT_OSS_TP1) is None
    # The same checkpoint on another topology is another deployment.
    assert planners.planner_for(replace(DEEPSEEK_DEP4, tp=8, moe_ep=8)) is None


def test_no_target_source_names_a_planner():
    """The binding direction: planners list deployments, never the reverse."""
    root = Path(modeling_v2.__file__).resolve().parent / "models"
    hits = [
        p.relative_to(root)
        for p in root.rglob("*.py")
        if "transfer_planner" in p.read_text() or "TransferPlanner" in p.read_text()
    ]
    assert hits == [], hits


def test_two_planners_covering_one_deployment_is_an_error(monkeypatch):
    class Twin(DeepseekMlaLatentTransferPlanner):
        pass

    monkeypatch.setattr(planners, "PLANNERS", (DeepseekMlaLatentTransferPlanner, Twin))
    with pytest.raises(ValueError, match="DeepseekV3ForCausalLM"):
        planners.planner_for(DEEPSEEK_DEP4)


class _StubModel:
    def build_transfer_planner(self, kv_cache_manager, self_rank_info, self_extractor):
        return "default"


def test_bind_leaves_an_uncovered_model_on_the_default_hook():
    stub = _StubModel()
    assert planners.bind_transfer_planner(stub, _model_config("GptOssForCausalLM")) is None
    assert stub.build_transfer_planner(None, None, None) == "default"
    no_arch = ModelConfig(pretrained_config=PretrainedConfig())
    assert planners.bind_transfer_planner(stub, no_arch) is None


def test_bind_installs_a_factory_that_knows_its_own_deployment(monkeypatch):
    """The class the model was built from is not an input: a built-in
    DeepseekV3ForCausalLM and a modeling_v2 target on the same topology bind
    to the same planner and publish the same description, which is what lets
    a context on one implementation feed a generation on the other."""
    seen = {}

    def fake_init(self, self_rank_info, self_extractor, *, self_deployment):
        seen["args"] = (self_rank_info, self_extractor, self_deployment)

    monkeypatch.setattr(DeepseekMlaLatentTransferPlanner, "__init__", fake_init)
    stub = _StubModel()
    config = _model_config("DeepseekV3ForCausalLM", **_DEP4_MAPPING)
    assert planners.bind_transfer_planner(stub, config) is DeepseekMlaLatentTransferPlanner
    planner = stub.build_transfer_planner("kvm", "ri", "ext")
    assert isinstance(planner, DeepseekMlaLatentTransferPlanner)
    assert seen["args"] == ("ri", "ext", DEEPSEEK_DEP4)
