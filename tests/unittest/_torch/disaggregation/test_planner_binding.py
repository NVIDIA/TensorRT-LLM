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
"""The user names a KV transfer planner through ``TRTLLM_KV_TRANSFER_PLANNER``; the
named planner checks that it supports the deployment and is bound to the model."""

from pathlib import Path

import pytest
from transformers import PretrainedConfig

import tensorrt_llm._torch._experimental.modeling_v2 as modeling_v2
from tensorrt_llm._torch.disaggregation import planners
from tensorrt_llm._torch.disaggregation.planners import PLANNER_ENV_VAR
from tensorrt_llm._torch.disaggregation.planners.deepseek_mla_latent import (
    DeepseekMlaLatentTransferPlanner,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

DEEPSEEK = "DeepseekV3ForCausalLM"
MLA_LATENT = DeepseekMlaLatentTransferPlanner.planner_id
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


class _StubModel:
    def build_transfer_planner(self, kv_cache_manager, self_rank_info, self_extractor):
        return "default"


def test_mla_latent_planner_supports_attention_dp_deepseek_of_any_width():
    supports = DeepseekMlaLatentTransferPlanner.supports_deployment
    assert supports(DEEPSEEK, DEP4)
    assert supports(DEEPSEEK, DEP8)
    assert not supports("GptOssForCausalLM", DEP4)
    assert not supports(DEEPSEEK, TP4)
    assert not supports(DEEPSEEK, DEP4_PP2)


def test_every_planner_is_keyed_by_its_planner_id():
    assert planners.PLANNERS == {MLA_LATENT: DeepseekMlaLatentTransferPlanner}
    assert PLANNER_ENV_VAR == "TRTLLM_KV_TRANSFER_PLANNER"


def test_no_target_source_names_a_planner():
    """Planners are chosen by the user, never by a model's source."""
    root = Path(modeling_v2.__file__).resolve().parent / "models"
    hits = [
        p.relative_to(root)
        for p in root.rglob("*.py")
        if "transfer_planner" in p.read_text() or "TransferPlanner" in p.read_text()
    ]
    assert hits == [], hits


def test_without_the_variable_every_model_keeps_the_default_hook(monkeypatch):
    monkeypatch.delenv(PLANNER_ENV_VAR, raising=False)
    stub = _StubModel()
    assert planners.bind_transfer_planner(stub, _model_config(DEEPSEEK, DEP4)) is None
    assert stub.build_transfer_planner(None, None, None) == "default"
    monkeypatch.setenv(PLANNER_ENV_VAR, "")
    assert planners.bind_transfer_planner(stub, _model_config(DEEPSEEK, DEP4)) is None


def test_the_named_planner_is_installed_as_the_factory(monkeypatch):
    seen = {}

    def fake_init(self, self_rank_info, self_extractor):
        seen["args"] = (self_rank_info, self_extractor)

    monkeypatch.setattr(DeepseekMlaLatentTransferPlanner, "__init__", fake_init)
    monkeypatch.setenv(PLANNER_ENV_VAR, MLA_LATENT)
    stub = _StubModel()
    assert (
        planners.bind_transfer_planner(stub, _model_config(DEEPSEEK, DEP4))
        is DeepseekMlaLatentTransferPlanner
    )
    planner = stub.build_transfer_planner("kvm", "ri", "ext")
    assert isinstance(planner, DeepseekMlaLatentTransferPlanner)
    assert seen["args"] == ("ri", "ext")


def test_an_unknown_key_is_refused_with_the_known_keys(monkeypatch):
    monkeypatch.setenv(PLANNER_ENV_VAR, "trtllm.no_such_planner")
    with pytest.raises(ValueError, match=MLA_LATENT):
        planners.bind_transfer_planner(_StubModel(), _model_config(DEEPSEEK, DEP4))


def test_a_planner_that_does_not_support_the_deployment_is_refused(monkeypatch):
    monkeypatch.setenv(PLANNER_ENV_VAR, MLA_LATENT)
    with pytest.raises(ValueError, match="GptOssForCausalLM"):
        planners.bind_transfer_planner(_StubModel(), _model_config("GptOssForCausalLM", DEP4))
    with pytest.raises(ValueError, match=DEEPSEEK):
        planners.bind_transfer_planner(_StubModel(), _model_config(DEEPSEEK, TP4))
    no_arch = ModelConfig(pretrained_config=PretrainedConfig())
    with pytest.raises(ValueError, match="architecture"):
        planners.bind_transfer_planner(_StubModel(), no_arch)
