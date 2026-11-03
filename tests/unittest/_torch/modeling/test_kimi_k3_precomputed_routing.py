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
"""Kimi K3 routing method that hands out top-k results computed by the fused router."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_kimi_linear import _KimiK3RoutingMethod


def _routing(bias: torch.Tensor) -> _KimiK3RoutingMethod:
    return _KimiK3RoutingMethod(
        top_k=16,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )


def test_precomputed_result_is_returned_once():
    routing = _routing(torch.zeros(896))
    ids = torch.arange(16, dtype=torch.int32).view(1, 16)
    weights = torch.full((1, 16), 1.0 / 16)
    routing.precomputed = (ids, weights)
    out_ids, out_weights = routing.apply(torch.empty(1, 896))
    assert out_ids is ids and out_weights is weights
    assert routing.precomputed is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_without_precomputed_result_routes_the_logits():
    torch.manual_seed(0)
    bias = torch.randn(896, device="cuda") * 0.01
    routing = _routing(bias)
    logits = torch.randn(7, 896, device="cuda")
    ids, weights = routing.apply(logits)
    ref_ids, ref_weights = super(_KimiK3RoutingMethod, routing).apply(logits)
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, ref_weights, rtol=0, atol=0)
