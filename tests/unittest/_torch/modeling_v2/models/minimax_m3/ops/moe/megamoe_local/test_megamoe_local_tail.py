# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MegaMoE local backend's TAIL requests: every launch returns the reduced rows on this runtime."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_impl,
    local_plan,
)

pytestmark = pytest.mark.cpu_only


def _leaf(tail_enabled: bool) -> local_impl.TrtllmCutedslMegaMoeLocalNvfp4Impl:
    leaf = local_impl.TrtllmCutedslMegaMoeLocalNvfp4Impl.__new__(
        local_impl.TrtllmCutedslMegaMoeLocalNvfp4Impl
    )
    leaf._tail_enabled = tail_enabled
    return leaf


def test_a_request_declines_with_the_runtime_reason():
    leaf = _leaf(True)
    assert leaf.tail_static_decline_reason() == local_impl.TAIL_RUNTIME_DECLINE
    assert leaf.request_deferred_topk_reduce([8, 8], 8) == local_impl.TAIL_RUNTIME_DECLINE
    leaf.cancel_deferred_topk_reduce()


def test_the_lever_switch_names_itself():
    assert _leaf(False).request_deferred_topk_reduce(None, 0) == "the TAIL lever is off"
    assert local_plan.lever_enabled("TAIL", {})
    assert not local_plan.lever_enabled("TAIL", {f"{local_plan.DISABLE_ENV}_TAIL": "1"})
