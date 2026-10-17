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
"""Capability logging, error behaviour and engagement accounting of sparse_mla_decode.

* The static decision is logged once per layer, not once per process.
* A layer inside the sparse_mla_decode domain whose kernel is missing from the build or whose
  device cannot launch it raises instead of falling back to native on one rank;
  a layer outside the domain still takes the native library.
* ``engagement_snapshot`` counts the sparse_mla_decode calls per layer and the declined
  selections per reason.
"""

import pytest
import torch
from sparse_mla_harness import DsaGenHarness

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode import (  # noqa: E501
    sparse_mla,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.sparse_mla_decode.sparse_mla import (  # noqa: E501
    SparseMlaDecodeFmha,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not sparse_mla.op_available(),
    reason="needs an SM10x GPU and the sparse_mla_decode kernel",
)


def test_capability_logged_per_layer():
    harness = DsaGenHarness([256], seq_len_q=1, num_layers=3, seed=141)
    try:
        for layer in harness.layers:
            assert SparseMlaDecodeFmha.is_available(layer)
        snapshot = sparse_mla.engagement_snapshot()
        for layer in harness.layers:
            assert snapshot["static"][layer.layer_idx] == ""
            assert (
                dispatch._CAPABILITY_LOGGED[("sparse_mla_decode", f"layer{layer.layer_idx}")]
                is True
            )
    finally:
        harness.shutdown()


def test_selected_sparse_mla_raises_instead_of_falling_back(monkeypatch):
    harness = DsaGenHarness([256], seq_len_q=1, seed=142)
    try:
        layer = harness.layers[0]
        monkeypatch.setattr(sparse_mla, "device_supported", lambda device: False)
        with pytest.raises(RuntimeError, match="cannot launch its 16-CTA cluster kernel"):
            SparseMlaDecodeFmha.is_available(layer)
        monkeypatch.setattr(sparse_mla, "op_available", lambda: False)
        with pytest.raises(RuntimeError, match="does not register"):
            SparseMlaDecodeFmha.is_available(layer)
    finally:
        harness.shutdown()


def test_layer_outside_domain_takes_native():
    harness = DsaGenHarness([256], seq_len_q=1, kv_dtype=torch.bfloat16, seed=143)
    try:
        layer = harness.layers[0]
        assert not SparseMlaDecodeFmha.is_available(layer)
        assert sparse_mla.engagement_snapshot()["static"][layer.layer_idx] == "KV cache is not FP8"
    finally:
        harness.shutdown()


def test_engagement_counts_per_layer_and_declines():
    harness = DsaGenHarness([20000], seq_len_q=6, num_layers=2, seed=144)
    try:
        before = sparse_mla.engagement_snapshot()["engaged_calls"]
        topk = harness.random_topk(seed=146)
        for layer_idx in (0, 1):
            harness.select_library(layer_idx, use_sparse_mla=True)
            for _ in range(2):
                harness.forward(layer_idx, harness.make_step(layer_idx, seed=147), topk)
            assert harness.selected_library(layer_idx) == ["SparseMlaDecodeFmha"]
        torch.cuda.synchronize()
        after = sparse_mla.engagement_snapshot()["engaged_calls"]
        for layer in harness.layers:
            idx = layer.layer_idx
            assert after.get(idx, 0) - before.get(idx, 0) == 2, (before, after)
            assert dispatch._CAPABILITY_LOGGED[("sparse_mla_decode", f"call:layer{idx}")] is True
    finally:
        harness.shutdown()
    other = DsaGenHarness([16384], seq_len_q=6, index_topk=1024, seed=148)
    try:
        declined_before = sparse_mla.engagement_snapshot()["declined_selections"]
        gen = torch.Generator(device="cpu").manual_seed(149)
        topk = (
            torch.stack(
                [
                    torch.randperm(other.causal_len(r), generator=gen)[:1024]
                    for r in range(other.num_rows)
                ]
            )
            .to(torch.int32)
            .cuda()
        )
        other.select_library(0, use_sparse_mla=True)
        other.forward(0, other.make_step(0, seed=150), topk)
        assert "SparseMlaDecodeFmha" not in other.selected_library(0)
        declined_after = sparse_mla.engagement_snapshot()["declined_selections"]
        assert declined_after.get("top-k 1024", 0) == declined_before.get("top-k 1024", 0) + 1
    finally:
        other.shutdown()
