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

"""Drive _try_fuse over a hand-built six-node FP8 subgraph.

symbolic_trace cannot produce this: the quantized linear, the norm and the rotation are
custom ops, so a traced nn.Module gets rejected long before the guards under test. The graph
is built node by node against the real op schemas instead.
"""

import pytest
import torch

pytest.importorskip("nanojet_kernels")

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401,E402
from tensorrt_llm._torch.auto_deploy.custom_ops.linear.nanojet_fused_qkv_gemm_norm_rope import (  # noqa: E402
    register,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_nanojet_fused_qkv_gemm_norm_rope import (  # noqa: E402
    _HEAD_MAJOR_UNSQUEEZE_DIM,
    FuseNanojetFusedQKVGemmNormRope,
)

HIDDEN, HEAD_DIM, NUM_HEADS, NUM_KV_HEADS, EPS = 256, 128, 8, 4, 1e-6


def _build(
    kv_rows_match: bool = True,
    head_dim: int = HEAD_DIM,
    table_width: int | None = None,
    key_uses_different_activation: bool = False,
):
    """A q/k/v + norm + rope subgraph, returning (GraphModule, rope node).

    ``kv_rows_match=False`` gives V a different head count from K, which is what the stacked
    ``[q + 2kv, hidden]`` weight cannot represent. ``table_width`` controls the rope table.
    """
    module = torch.nn.Module()
    graph = torch.fx.Graph()

    if table_width is None:
        table_width = head_dim
    value_heads = NUM_KV_HEADS if kv_rows_match else NUM_KV_HEADS + 1
    tensors = {
        "query_weight": torch.zeros(NUM_HEADS * head_dim, HIDDEN, dtype=torch.float8_e4m3fn),
        "key_weight": torch.zeros(NUM_KV_HEADS * head_dim, HIDDEN, dtype=torch.float8_e4m3fn),
        "value_weight": torch.zeros(value_heads * head_dim, HIDDEN, dtype=torch.float8_e4m3fn),
        "input_scale": torch.tensor([0.5]),
        "weight_scale": torch.tensor([0.25]),
        "query_norm": torch.ones(head_dim),
        "key_norm": torch.ones(head_dim),
        "cos_table": torch.zeros(128, table_width),
        "sin_table": torch.zeros(128, table_width),
    }
    for name, tensor in tensors.items():
        module.register_buffer(name, tensor)

    attrs = {name: graph.get_attr(name) for name in tensors}
    hidden = graph.placeholder("hidden")
    key_hidden = graph.placeholder("key_hidden") if key_uses_different_activation else hidden
    positions = graph.placeholder("positions")

    def projection(weight_attr, activation=hidden):
        return graph.call_function(
            torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
            args=(activation, weight_attr, None, attrs["input_scale"], attrs["weight_scale"]),
        )

    def normed(projection_node, norm_attr):
        return graph.call_function(
            torch.ops.auto_deploy.torch_rmsnorm.default,
            args=(projection_node, norm_attr, EPS),
        )

    query = normed(projection(attrs["query_weight"]), attrs["query_norm"])
    key = normed(projection(attrs["key_weight"], key_hidden), attrs["key_norm"])
    projection(attrs["value_weight"])  # V has no norm; found by sharing the activation

    def gather(table):
        return graph.call_function(torch.ops.aten.index.Tensor, args=(table, [positions]))

    rope = graph.call_function(
        torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin.default,
        args=(
            query,
            key,
            gather(attrs["cos_table"]),
            gather(attrs["sin_table"]),
            _HEAD_MAJOR_UNSQUEEZE_DIM,
        ),
    )
    graph.output(rope)
    return torch.fx.GraphModule(module, graph), rope


def _reason(**kwargs):
    graph_module, rope = _build(**kwargs)
    return FuseNanojetFusedQKVGemmNormRope._try_fuse(graph_module, rope)


def test_harness_reaches_the_shape_guards():
    """The control: a well-formed subgraph must get past every earlier check.

    Without this, a rejection below could come from the harness rather than the guard, which
    is how a test like this quietly stops testing anything.
    """
    assert register(), "nanojet must be installed"
    result = _reason()
    assert not isinstance(result, str), result


def test_rejects_head_dim_not_supported_by_kernel():
    """The kernel's normalization and RoPE math is specialized for head_dim 128."""
    assert register(), "nanojet must be installed"
    assert _reason(head_dim=64) == "nanojet-declined"


def test_rejects_qkv_projections_with_different_activations():
    """One fused GEMM cannot reproduce projections that read different graph values."""
    assert register(), "nanojet must be installed"
    assert _reason(key_uses_different_activation=True) == "activation-mismatch"


def test_rejects_value_head_count_differing_from_key():
    """The stacked weight is indexed by n_q/n_kv, which only describes it if K and V match.

    A mismatch produces a correctly-shaped buffer the kernel then reads at wrong offsets —
    no error, just wrong numbers. Unreachable with Qwen3, whose K and V are both 8 heads.
    """
    assert register(), "nanojet must be installed"
    assert _reason(kv_rows_match=False) == "kv-row-mismatch"


def test_rejects_rope_table_narrower_than_head_dim():
    """The rewrite keeps each table's first half, which assumes HF duplicated both halves.

    A table already stored at half width would be silently truncated to a quarter.
    """
    assert register(), "nanojet must be installed"
    assert _reason(table_width=HEAD_DIM // 2) == "rope-table-width"
