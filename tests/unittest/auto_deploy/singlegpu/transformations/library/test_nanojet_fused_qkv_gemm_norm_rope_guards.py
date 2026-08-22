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

"""Regression tests for four silent-miscompute paths in fuse_nanojet_fused_qkv_gemm_norm_rope.

Each of these produced a plausible, wrong result with nothing to indicate it, and none is
reachable with a GQA checkpoint like Qwen3 (16/8 heads) — which is exactly why the end-to-end
cosine similarity stayed identical while they were broken. They are driven directly instead.

Covered here: the Q/K ordering contract and the buffer-naming rule. NOT covered: the
kv-row-mismatch and rope-table-width guards, which sit inside _try_fuse and need a full
six-node FP8 subgraph to reach — a traced nn.Module does not produce the quantized ops.
"""

import pytest

pytest.importorskip("nanojet_kernels")

from tensorrt_llm._torch.auto_deploy.transform.library.fuse_nanojet_fused_qkv_gemm_norm_rope import (  # noqa: E402
    _HEAD_MAJOR_UNSQUEEZE_DIM,
)


def test_unsqueeze_dim_constant_is_head_major():
    """The fused kernel is written against [batch, seq, heads, head_dim]."""
    assert _HEAD_MAJOR_UNSQUEEZE_DIM == 2


def test_query_and_key_come_from_the_op_schema_not_head_counts():
    """Q is args[0] by the rotation's own signature.

    The previous code inferred it from head counts, which is ambiguous the moment
    num_kv_heads == num_heads: `first_heads >= second_heads` is then always true, so a graph
    handing the rotation K first had Q and K silently swapped.
    """
    import torch as _torch

    schema = _torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin.default._schema
    names = [argument.name for argument in schema.arguments]
    assert names[:2] == ["q", "k"], f"rotation no longer names its inputs q,k: {names}"
    assert names[4] == "unsqueeze_dim"


def test_buffer_names_are_graph_unique_not_addresses():
    """``id()`` is neither stable across runs nor unique over time.

    A collected node's address can be handed to a later one, which would make two layers
    register the same buffer name and silently share one stacked weight.
    """
    import inspect

    from tensorrt_llm._torch.auto_deploy.transform.library import (
        fuse_nanojet_fused_qkv_gemm_norm_rope,
        fuse_nanojet_swiglu_gemm_fp8,
    )

    for module in (fuse_nanojet_fused_qkv_gemm_norm_rope, fuse_nanojet_swiglu_gemm_fp8):
        source = inspect.getsource(module)
        assert "id(rope_node)" not in source and "id(node)" not in source, (
            f"{module.__name__} derives a buffer name from an address"
        )
        assert ".name}" in source, f"{module.__name__} should key buffers on node.name"
