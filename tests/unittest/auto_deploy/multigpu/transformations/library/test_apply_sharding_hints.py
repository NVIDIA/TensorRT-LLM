# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for ``apply_sharding_hints`` (hint-driven TP sharding).

``test_sharding`` — multi-GPU end-to-end: exports, transforms, and validates
    output correctness on real GPUs via ``run_test_transformed_gm``.
``test_apply_hints`` — single-process transform check: verifies graph
    rewriting (weight shapes, all_reduce replacement, skip conditions)
    without distributed execution.
"""

import pytest
import torch
import torch.nn as nn
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_test_transformed_gm

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

pytestmark = pytest.mark.threadleak(enabled=False)

FEATURES, HIDDEN = 32, 64


class HintedMLP(nn.Module):
    def __init__(self, features=FEATURES, hidden=HIDDEN):
        super().__init__()
        self.up = nn.Linear(features, hidden, bias=False)
        self.down = nn.Linear(hidden, features, bias=False)

    def forward(self, x):
        h = torch.ops.auto_deploy.torch_linear_simple(x, self.up.weight, None, tp_mode="colwise")
        h = torch.relu(h)
        h = torch.ops.auto_deploy.torch_linear_simple(h, self.down.weight, None, tp_mode="rowwise")
        h = torch.ops.auto_deploy.all_reduce(h)
        return h


# ---------------------------------------------------------------------------
# test_sharding — multi-GPU end-to-end (follows test_tp_sharding.py pattern)
# ---------------------------------------------------------------------------


def _run_sharding_job(rank: int, world_size: int) -> None:
    model = HintedMLP().to(device="cuda", dtype=torch.float16)
    x = torch.randn(4, 8, FEATURES, device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(None, {"apply_sharding_hints": {"stage": "sharding"}})(
        None, gm
    )

    op_ar = torch.ops.auto_deploy.torch_dist_all_reduce

    def check_transformed_graph(gm_mod) -> bool:
        has_dist = any(is_op(n, op_ar) for n in gm_mod.graph.nodes)
        return has_dist == (world_size > 1)

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=check_transformed_graph,
        _get_expected_num_params=lambda n: n // world_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("world_size", get_device_counts([2]))
def test_sharding(world_size: int):
    """Hint-based colwise/rowwise + all_reduce: end-to-end on real GPUs."""
    dist_common.spawn_multiprocess_job(job=_run_sharding_job, size=world_size)


# ---------------------------------------------------------------------------
# test_apply_hints — single-process transform checks (no distributed exec)
# ---------------------------------------------------------------------------


def _make_optimizer(world_size: int, rank: int = 0):
    opt = InferenceOptimizer(
        factory=None,
        config={"apply_sharding_hints": {"stage": "sharding"}},
    )
    opt.shared_config = SharedConfig(
        local_rank=rank,
        world_size=world_size,
        dist_config=DistConfig(world_size=world_size, rank=rank, tp_size=world_size),
    )
    return opt


def _export_hinted_mlp():
    model = HintedMLP().cuda()
    x = torch.randn(2, FEATURES, device="cuda")
    return torch_export_to_gm(model, args=(x,), clone=True), model, x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "world_size, expect_skipped, expect_up_shape, expect_down_shape",
    [
        (1, True, (HIDDEN, FEATURES), (FEATURES, HIDDEN)),
        (2, False, (HIDDEN // 2, FEATURES), (FEATURES, HIDDEN // 2)),
    ],
)
def test_apply_hints(world_size, expect_skipped, expect_up_shape, expect_down_shape):
    """Verify graph rewriting without distributed execution."""
    gm, _, _ = _export_hinted_mlp()
    gm_out = _make_optimizer(world_size)(None, gm)

    info = gm_out.meta["_autodeploy"]["transform_history"]["apply_sharding_hints"]
    assert info.skipped is expect_skipped

    assert gm_out.up.weight.shape == expect_up_shape
    assert gm_out.down.weight.shape == expect_down_shape

    has_dist_ar = any(
        is_op(n, torch.ops.auto_deploy.torch_dist_all_reduce.default) for n in gm_out.graph.nodes
    )
    assert has_dist_ar == (not expect_skipped)
