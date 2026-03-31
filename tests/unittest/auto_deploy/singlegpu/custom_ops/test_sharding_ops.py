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

"""Unit tests for sharding hint custom ops in ``sharding_ops.py``."""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for auto_deploy sharding ops"
)


def test_view_matches_reshape():
    x = torch.randn(4, 8, device="cuda")
    shape = [32]
    out = torch.ops.auto_deploy.view(x, shape)
    ref = x.reshape(shape).clone()
    torch.testing.assert_close(out, ref)


def test_view_tp_scaled_dim_passthrough():
    x = torch.randn(4, 8, device="cuda")
    shape = [2, 16]
    default = torch.ops.auto_deploy.view(x, shape)
    explicit_neg1 = torch.ops.auto_deploy.view(x, shape, tp_scaled_dim=-1)
    nonzero_dim = torch.ops.auto_deploy.view(x, shape, tp_scaled_dim=0)
    torch.testing.assert_close(default, explicit_neg1)
    torch.testing.assert_close(default, nonzero_dim)


def test_view_accepts_layer_type():
    x = torch.randn(4, 8, device="cuda")
    shape = [32]
    out = torch.ops.auto_deploy.view(x, shape, layer_type="mha")
    ref = x.reshape(shape).clone()
    torch.testing.assert_close(out, ref)


def test_split_matches_torch_split():
    x = torch.randn(4, 8, device="cuda")
    split_sizes = [3, 5]
    dim = -1
    out = torch.ops.auto_deploy.split_with_sizes(x, split_sizes, dim)
    ref = list(torch.split(x, split_sizes, dim=dim))
    assert len(out) == len(ref)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b)


def test_split_shardable_flag():
    x = torch.randn(4, 8, device="cuda")
    split_sizes = [2, 2, 4]
    dim = -1
    a = torch.ops.auto_deploy.split_with_sizes(x, split_sizes, dim, shardable=False)
    b = torch.ops.auto_deploy.split_with_sizes(x, split_sizes, dim, shardable=True)
    assert len(a) == len(b)
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v)


def test_split_accepts_layer_type():
    x = torch.randn(4, 8, device="cuda")
    split_sizes = [4, 4]
    out = torch.ops.auto_deploy.split_with_sizes(x, split_sizes, dim=-1, layer_type="mlp")
    ref = list(torch.split(x, split_sizes, dim=-1))
    assert len(out) == len(ref)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b)


def test_all_reduce_is_identity():
    x = torch.randn(4, 8, device="cuda")
    out = torch.ops.auto_deploy.all_reduce(x)
    torch.testing.assert_close(out, x.clone())


def test_all_reduce_accepts_layer_type():
    x = torch.randn(4, 8, device="cuda")
    out = torch.ops.auto_deploy.all_reduce(x, layer_type="mha")
    torch.testing.assert_close(out, x.clone())
