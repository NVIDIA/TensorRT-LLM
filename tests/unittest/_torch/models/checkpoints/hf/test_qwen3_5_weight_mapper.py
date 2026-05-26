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

import math

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import (
    Qwen3_5MoeHfWeightMapper,
)


def _make_mapper():
    # _split_qkv_scale_tensor never touches self, so we bypass __init__ to
    # avoid pulling in a full config + checkpoint loader for the unit test.
    return Qwen3_5MoeHfWeightMapper.__new__(Qwen3_5MoeHfWeightMapper)


def test_split_qkv_scale_tensor_fp8_block_layout():
    """FP8 block-wise (fp8_pb_wo) scales: leading dim is ceil(out_dim/128)."""
    expected_q = 4096
    expected_v = 4096
    q_blocks = math.ceil(expected_q / 128)  # 32
    v_blocks = math.ceil(expected_v / 128)  # 32
    trailing = 7  # arbitrary inner-block dim

    tensor = torch.arange(
        (q_blocks * 2 + v_blocks) * trailing, dtype=torch.float32
    ).reshape(q_blocks * 2 + v_blocks, trailing)

    mapper = _make_mapper()
    q, k, v = mapper._split_qkv_scale_tensor(tensor, expected_q, expected_v)

    assert q.shape == (q_blocks, trailing)
    assert k.shape == (q_blocks, trailing)
    assert v.shape == (v_blocks, trailing)
    assert torch.equal(q, tensor[:q_blocks])
    assert torch.equal(k, tensor[q_blocks : q_blocks * 2])
    assert torch.equal(v, tensor[q_blocks * 2 :])


def test_split_qkv_scale_tensor_awq_layout():
    """W4A16_AWQ scales: leading dim is the full output channel count.

    Regression for #14561: ModelOpt exports AWQ scales with shape
    (out_dim, in_dim / group_size). The preprocessor renames them to
    .weight_scale_inv so they reach the same split site as FP8 block scales,
    and this code path must accept either layout.
    """
    expected_q = 4096
    expected_v = 4096
    in_dim_over_group = 20  # matches reporter's (8192, 20) trailing dim

    tensor = torch.arange(
        (expected_q * 2 + expected_v) * in_dim_over_group, dtype=torch.float32
    ).reshape(expected_q * 2 + expected_v, in_dim_over_group)

    mapper = _make_mapper()
    q, k, v = mapper._split_qkv_scale_tensor(tensor, expected_q, expected_v)

    assert q.shape == (expected_q, in_dim_over_group)
    assert k.shape == (expected_q, in_dim_over_group)
    assert v.shape == (expected_v, in_dim_over_group)
    assert torch.equal(q, tensor[:expected_q])
    assert torch.equal(k, tensor[expected_q : expected_q * 2])
    assert torch.equal(v, tensor[expected_q * 2 :])


def test_split_qkv_scale_tensor_repro_14561_shape():
    """Exact reporter shape from #14561: (8192, 20) AWQ scales for q == v."""
    expected_q = expected_v = 4096
    tensor = torch.zeros(8192, 20)

    mapper = _make_mapper()
    q, k, v = mapper._split_qkv_scale_tensor(tensor, expected_q, expected_v)

    assert q.shape == (4096, 20)
    assert k.shape == (4096, 20)
    assert v.shape == (4096, 20)


def test_split_qkv_scale_tensor_unknown_layout_raises():
    """Tensors that match neither layout must raise with a message naming both
    expected leading dims, so a future quantization format that lands here is
    diagnosable from the error alone."""
    expected_q = 4096
    expected_v = 4096
    # Leading dim that matches neither per-channel (12288) nor per-block (96).
    tensor = torch.zeros(123, 7)

    mapper = _make_mapper()
    with pytest.raises(AssertionError, match="W4A16_AWQ.*FP8 block"):
        mapper._split_qkv_scale_tensor(tensor, expected_q, expected_v)


def test_split_qkv_scale_tensor_asymmetric_qv_block_layout():
    """FP8 block layout with expected_q != expected_v, exercising the
    ceil-division per side."""
    expected_q = 5000  # ceil(5000/128) = 40
    expected_v = 3000  # ceil(3000/128) = 24
    q_blocks = math.ceil(expected_q / 128)
    v_blocks = math.ceil(expected_v / 128)
    trailing = 4

    tensor = torch.arange(
        (q_blocks * 2 + v_blocks) * trailing, dtype=torch.float32
    ).reshape(q_blocks * 2 + v_blocks, trailing)

    mapper = _make_mapper()
    q, k, v = mapper._split_qkv_scale_tensor(tensor, expected_q, expected_v)

    assert q.shape == (q_blocks, trailing)
    assert k.shape == (q_blocks, trailing)
    assert v.shape == (v_blocks, trailing)
