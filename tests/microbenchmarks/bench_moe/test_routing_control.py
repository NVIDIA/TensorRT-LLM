# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bench_moe's forced routing on a MoE module. Pure CPU."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from .routing.native_logits import _maybe_install_routing_control_patch


def test_supplied_routes_answer_the_moe_schedulers_routing_call():
    """The supplied routes answer the MoE scheduler's routing call.

    The scheduler calls ``moe.routing_method.apply(router_logits, input_ids)``; the patch returns the
    supplied routes and puts the routing method back afterwards.
    """
    ids = torch.tensor([[1, 2], [3, 0]])
    scales = torch.full((2, 2), 0.5)
    routing = SimpleNamespace(apply=lambda router_logits, input_ids=None: None)
    original = routing.apply
    moe = SimpleNamespace(routing_method=routing, forward_impl=None)
    with _maybe_install_routing_control_patch(moe, ids, scales, active=True):
        got_ids, got_scales = moe.routing_method.apply(torch.zeros(2, 4), torch.arange(2))
    assert torch.equal(got_ids, ids.int()) and torch.equal(got_scales, scales)
    assert moe.routing_method.apply is original
