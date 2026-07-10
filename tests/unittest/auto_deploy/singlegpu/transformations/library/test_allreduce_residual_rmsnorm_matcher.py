# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Graph-level (non-distributed) match-count tests for the AR + residual + RMSNorm fusion.

These tests do NOT execute the fused graph (which would require a process group);
they only verify that the pattern matcher rewrites the expected subgraphs into a
single ``trtllm_fused_allreduce_residual_rmsnorm`` op. They specifically cover the
``reshape`` between the all-reduce and the residual add (e.g. a TP MLP/MoE output
all-reduced as a flattened ``[B*S, H]`` tensor and reshaped back to the 3D residual
``[B, S, H]``), which previously broke the match and left the op unfused.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy._compat import AllReduceStrategy
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device="cuda", dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ARResidualNorm(torch.nn.Module):
    """all_reduce -> [reshape] -> add(residual) -> rmsnorm.

    ``x`` is provided 2D when ``with_reshape`` so that the all-reduce output is
    reshaped (expanded) to the 3D residual shape before the add.
    """

    def __init__(self, hidden_size, strategy, add_order, with_reshape):
        super().__init__()
        self.norm = RMSNorm(hidden_size, dtype=torch.bfloat16)
        self.strategy = strategy
        self.add_order = add_order
        self.with_reshape = with_reshape

    def forward(self, x, residual):
        x = torch.ops.auto_deploy.trtllm_dist_all_reduce.default(x, self.strategy)
        if self.with_reshape:
            x = torch.reshape(x, residual.shape)
        y = residual + x if self.add_order == "residual_first" else x + residual
        return self.norm(y), y


def _count(gm, op):
    return sum(is_op(n, op) for n in gm.graph.nodes)


@pytest.mark.parametrize("add_order", ["residual_first", "x_first"])
@pytest.mark.parametrize("with_reshape", [False, True], ids=["base", "reshape"])
@pytest.mark.parametrize("strategy", ["AUTO", "ONESHOT"])
def test_allreduce_residual_rmsnorm_match(add_order, with_reshape, strategy):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device to build the test module.")

    hidden = 512
    bsz, seq = 2, 4
    if with_reshape:
        x = torch.randn(bsz * seq, hidden, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(bsz, seq, hidden, device="cuda", dtype=torch.bfloat16)
    else:
        x = torch.randn(bsz * seq, hidden, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(bsz * seq, hidden, device="cuda", dtype=torch.bfloat16)

    model = ARResidualNorm(hidden, strategy, add_order, with_reshape)

    # Dynamic leading dim to exercise the symbolic re-trace path of the matcher.
    # For the reshape variant x is the 2D [bs*seq, h] all-reduce output reshaped
    # (expanded) to the 3D residual [bs, seq, h], so the 2D token dim is the derived
    # dim seq * bs and the residual batch dim is bs.
    if with_reshape:
        bs = Dim("bs")
        dynamic_shapes = ({0: seq * bs}, {0: bs})
    else:
        tokens = Dim("tokens")
        dynamic_shapes = ({0: tokens}, {0: tokens})

    gm = torch_export_to_gm(model, args=(x, residual), dynamic_shapes=dynamic_shapes, clone=True)

    # Collapse the decomposed RMSNorm into a single torch_rmsnorm op (prerequisite for
    # the allreduce-residual-rmsnorm fusion).
    gm = InferenceOptimizer(None, {"match_rmsnorm_pattern": {"stage": "pattern_matcher"}})(None, gm)
    assert _count(gm, torch.ops.auto_deploy.torch_rmsnorm) == 1

    # Provide the allreduce strategy via the legacy container path (no distributed
    # process group / detect_sharding needed for a graph-only match test).
    gm._sharding_transform_container = SimpleNamespace(
        config=SimpleNamespace(allreduce_strategy=AllReduceStrategy[strategy])
    )
    gm = InferenceOptimizer(
        None, {"fuse_allreduce_residual_rmsnorm": {"stage": "post_load_fusion"}}
    )(None, gm)

    # The whole subgraph must collapse into exactly one fused op, with the standalone
    # all-reduce, residual add, and RMSNorm all removed.
    fused = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm)
    ]
    assert len(fused) == 1, f"expected 1 fused op, got {len(fused)}"
    assert fused[0].args[4] == strategy
    assert _count(gm, torch.ops.auto_deploy.trtllm_dist_all_reduce) == 0
    assert _count(gm, torch.ops.auto_deploy.torch_rmsnorm) == 0
