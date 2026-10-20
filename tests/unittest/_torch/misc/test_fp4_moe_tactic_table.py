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
"""Opt-in FP4 MoE tactic table (TLLM_FP4_MOE_TACTIC_TABLE)."""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops

pytestmark = pytest.mark.cpu_only


def _runner(unique_id=(4, 3072, 64, 0), ep_size=2):
    specs = ops.FP4BlockScaleMoERunner.get_dynamic_tensor_specs(ep_size)
    return SimpleNamespace(
        unique_id=lambda: unique_id, tuning_config=SimpleNamespace(dynamic_tensor_specs=specs)
    )


@pytest.fixture
def table(tmp_path, monkeypatch):
    path = tmp_path / "fp4_moe_tactics.json"
    path.write_text(json.dumps({"4,3072,64,0": {"1024": [128, 23], "8192": [256, 7]}}))
    ops._load_fp4_moe_tactic_table.cache_clear()
    monkeypatch.setattr(ops, "_FP4_MOE_TACTIC_TABLE_PATH", str(path))
    yield path
    ops._load_fp4_moe_tactic_table.cache_clear()


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(ops, "_FP4_MOE_TACTIC_TABLE_PATH", None)
    assert ops._pinned_fp4_moe_tactic(_runner(), 1761) is None


def test_pinned_tactic_uses_the_tuning_bucket(table):
    # 1761 and 1160 rows both map to the 1024 tuning bucket.
    assert ops._pinned_fp4_moe_tactic(_runner(), 1761) == [128, 23]
    assert ops._pinned_fp4_moe_tactic(_runner(), 1160) == [128, 23]
    assert ops._pinned_fp4_moe_tactic(_runner(), 8192) == [256, 7]


def test_unlisted_bucket_or_shape_keeps_the_tuner_choice(table):
    assert ops._pinned_fp4_moe_tactic(_runner(), 300) is None
    assert ops._pinned_fp4_moe_tactic(_runner(unique_id=(8, 3072, 64, 0)), 1761) is None


def test_table_path_is_resolved_at_import(table, monkeypatch):
    # The environment is not re-read per call (CUDA-graph capture must see the same tactic as eager steps).
    monkeypatch.delenv(ops.FP4_MOE_TACTIC_TABLE_ENV, raising=False)
    assert ops._pinned_fp4_moe_tactic(_runner(), 1761) == [128, 23]
