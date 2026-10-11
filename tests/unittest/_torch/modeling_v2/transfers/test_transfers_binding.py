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
"""A planner declares the targets it covers; a target knows nothing about planners."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import transfers
from tensorrt_llm._torch._experimental.modeling_v2.transfers.deepseek_mla_latent import (
    DeepseekMlaLatentTransferPlanner,
)

pytestmark = pytest.mark.cpu_only

DEEPSEEK_DEP4 = "deepseek_v3/r1_0528_nvfp4__sm_103__dep4"
GPT_OSS_TP1 = "gpt_oss/gpt_oss_120b__sm_103__tp1"


def _target_cls(dotted_dir: str):
    family, name = dotted_dir.split("/")
    module = __import__(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.{family}.{name}.modeling",
        fromlist=["_"],
    )
    return next(
        v
        for k, v in vars(module).items()
        if k.startswith("ModelingV2")
        and isinstance(v, type)
        and v.__module__ == module.__name__
        and "Target" not in k
    )


def test_target_id_is_family_slash_directory():
    assert transfers.target_id_of(_target_cls(DEEPSEEK_DEP4)) == DEEPSEEK_DEP4
    assert transfers.target_id_of(_target_cls(GPT_OSS_TP1)) == GPT_OSS_TP1
    assert transfers.target_id_of(dict) is None  # not a modeling_v2 target


def test_planner_is_found_by_the_target_it_covers_and_nothing_else():
    assert transfers.planner_for_target(DEEPSEEK_DEP4) is DeepseekMlaLatentTransferPlanner
    assert transfers.planner_for_target(GPT_OSS_TP1) is None
    assert transfers.planner_for_target("nobody/such__sm_0__tp1") is None


def test_no_target_source_names_a_planner():
    """The binding direction: planners list targets, never the reverse."""
    from pathlib import Path

    root = Path(transfers.__file__).resolve().parents[1] / "models"
    hits = [
        p.relative_to(root)
        for p in root.rglob("*.py")
        if "transfer_planner" in p.read_text() or "TransferPlanner" in p.read_text()
    ]
    assert hits == [], hits


def test_two_planners_covering_one_target_is_an_error(monkeypatch):
    class Twin(DeepseekMlaLatentTransferPlanner):
        pass

    monkeypatch.setattr(transfers, "PLANNERS", (DeepseekMlaLatentTransferPlanner, Twin))
    with pytest.raises(ValueError, match=DEEPSEEK_DEP4):
        transfers.planner_for_target(DEEPSEEK_DEP4)


class _StubModel:
    def build_transfer_planner(self, kv_cache_manager, self_rank_info, self_extractor):
        return "default"


def test_bind_leaves_an_uncovered_model_on_the_default_hook():
    stub = _StubModel()
    assert transfers.bind_transfer_planner(stub, _target_cls(GPT_OSS_TP1)) is None
    assert stub.build_transfer_planner(None, None, None) == "default"
    assert transfers.bind_transfer_planner(stub, dict) is None


def test_bind_installs_a_factory_that_knows_its_own_target(monkeypatch):
    seen = {}

    def fake_init(self, self_rank_info, self_extractor, *, self_target):
        seen["args"] = (self_rank_info, self_extractor, self_target)

    monkeypatch.setattr(DeepseekMlaLatentTransferPlanner, "__init__", fake_init)
    stub = _StubModel()
    assert transfers.bind_transfer_planner(stub, _target_cls(DEEPSEEK_DEP4)) is (
        DeepseekMlaLatentTransferPlanner
    )
    planner = stub.build_transfer_planner("kvm", "ri", "ext")
    assert isinstance(planner, DeepseekMlaLatentTransferPlanner)
    assert seen["args"] == ("ri", "ext", DEEPSEEK_DEP4)
