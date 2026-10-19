# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The expert kernel's early start (``early_expert_ids``): its plan, constants and host rules.

Host-only. The early start of the persistent expert kernel from the ids the routed
pool kernel publishes builds on the persistent experts (and so on the replicated
experts) and on the one-chain prologue, and is installed after
``ring_buddy.install``. The early-ids module's host rules are pure Python and
checked here too: the trigger point and timing constants, the fallback count
readable without a device sync, and the placements the routed kernel publishes
(``pool_gather_route.EARLY_PLACES``; any other is refused at install).
"""

from __future__ import annotations

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
)

pytestmark = pytest.mark.cpu_only


def test_the_early_start_builds_on_the_persistent_experts_and_the_prologue():
    assert dispatch.closure({"early_expert_ids"}) >= {
        "persistent_experts",
        "moe_prologue",
    }
    with pytest.raises(ValueError, match="builds on"):
        dispatch.Plan(frozenset({"early_expert_ids", "replicated_experts", "persistent_experts"}))


def test_the_trigger_names():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
        early_expert_ids as ei,
    )

    assert ei.TRIGGERS == {"exit": 0, "exchange": 1, "entry": 2}
    assert ei.WORDS == 64
    # The routed kernel triggers after its logit exchange; the poll's deadline and back-off.
    assert (ei.TRIGGER, ei.DEADLINE_NS, ei.POLL_NS) == (ei.TRIGGER_EXCHANGE, 100_000, 32)


def _early_ids_module():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
        early_expert_ids as ei,
    )

    return ei


def test_a_buffer_takes_the_constants_and_counts_fallbacks(monkeypatch):
    import torch

    ei = _early_ids_module()
    monkeypatch.setattr(ei, "_BUFFERS", [])
    epoch = torch.zeros(4, dtype=torch.int32)
    assert ei.fallbacks_seen() is None
    early = ei.EarlyExpertIds(torch.device("cpu"), epoch, ("cpu", 4, 1))
    assert (early.trigger, early.deadline_ns, early.poll_ns) == (ei.TRIGGER, 100_000, 32)
    assert early.stamps is None
    assert ei.fallbacks_seen() == 0
    early.fallbacks[0] = 7
    assert ei.fallbacks_seen() == 7 and early.fallback_count() == 7


def test_only_the_placements_the_helper_computes_are_published():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.early_expert_ids import (  # noqa: E501
        early_ids as early_experts,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm import (  # noqa: E501
        pool_gather_route as route,
    )

    assert route.EARLY_PLACES == (
        route.PLACE_OWNER,
        route.PLACE_BALANCE,
        route.PLACE_STATIC,
        route.PLACE_BALANCE3,
    )
    # The three-window placement goes through the same helper and is published.
    places = {
        1: route.PLACE_BALANCE,
        2: route.PLACE_OWNER,
        3: route.PLACE_STATIC,
        5: route.PLACE_BALANCE3,
    }
    assert early_experts.unpublishable({**places, 4: route.PLACE_NONE}) == {}
    # A placement kind the helper does not compute is refused.
    assert early_experts.unpublishable({**places, 59: 7, 60: 7}) == {59: 7, 60: 7}
