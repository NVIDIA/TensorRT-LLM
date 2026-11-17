# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp4ep4 target's routed-expert routing on the generic path: its gate's ``KimiK3MoeRoutingMethod``
computes the top-16 outside the TRTLLM-Gen kernel. The checks are ``_kimi_k3_moe_routing``: the method host-side, and
on SM 10.0 one generic MoE call routed outside the kernel against the same call routed inside it."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    modeling as target,
)

__extra_import_path__ = [".."]
from _kimi_k3_moe_routing import (  # noqa: E402
    check_gate_routes_outside_the_kernel,
    check_routing_outside_the_kernel_matches_inside,
    sm_100,
)


def test_gate_routes_outside_the_kernel():
    check_gate_routes_outside_the_kernel(target)


@pytest.mark.skipif(
    not sm_100(), reason="the TRTLLM-Gen MXFP4 cubins and fused route + quant need SM100 or SM103"
)
def test_routing_outside_the_kernel_matches_inside(monkeypatch):
    check_routing_outside_the_kernel_matches_inside(target, monkeypatch)
