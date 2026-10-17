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
"""Single-process checks of post_attention_hop (no collective, no GPU kernel).

The eight-rank correctness checks need two 4-GPU nodes of one NVLink domain
and run as collective tests.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_attention_hop import (  # noqa: E501
    hop,
)
from tensorrt_llm.mapping import Mapping

_HEADER = (
    Path(__file__).resolve().parents[9] / "cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHop.h"
)


def _header_constant(name: str) -> int:
    match = re.search(rf"constexpr int {name} = (\d+);", _HEADER.read_text())
    assert match, name
    return int(match.group(1))


def test_constants_match_kernel_header():
    assert hop.HIDDEN_SIZE == _header_constant("kHiddenDim")
    assert hop.NUM_RANKS == _header_constant("kNumRanks")
    assert hop.NUM_EXPERTS == _header_constant("kNumExperts")
    assert hop.MAX_ROWS == _header_constant("kMaxRows")
    assert hop.NUM_LAMPORT_BUFFERS == _header_constant("kNumLamportBuffers")


def test_entry_point_registered():
    assert (
        dispatch.get_impl("post_attention_hop", "make_post_attn_hop")
        == hop.PostAttnHop.get_or_create
    )
    assert dispatch.get_impl("post_attention_hop", "engagement_counts") == hop.engagement_counts


def test_router_layout_constants_match_kernel():
    """required_sms mirrors the kernel's launch configuration."""
    source = _HEADER.with_suffix(".cu").read_text()
    assert f"constexpr int kSlices = {hop.CLUSTER_CTAS};" in source
    for rows in range(1, hop.MAX_ROWS + 1):
        groups = min(rows, hop.ROUTER_MAX_ROW_GROUPS)
        assert (
            f"case {rows}: return launchTyped<{rows}, "
            f"{hop.ROUTER_EXPERT_GROUPS}, {groups}>(params);"
        ) in source


@pytest.mark.parametrize(
    "rows, router, expected",
    [
        (1, False, 8),
        (6, False, 48),
        (8, False, 64),
        (1, True, 32),
        (2, True, 64),
        (6, True, 96),
        (8, True, 96),
    ],
)
def test_required_sms(rows, router, expected):
    assert hop.required_sms(rows, router) == expected


def _mapping():
    return Mapping(world_size=8, tp_size=8, gpus_per_node=4, rank=0)


def test_native_allreduce_must_use_mnnvl():
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
    from tensorrt_llm.functional import AllReduceStrategy

    mapping = _mapping()
    nccl = SimpleNamespace(strategy=AllReduceStrategy.NCCL, mnnvl_allreduce=None)
    reason = hop.native_allreduce_ineligibility(nccl, mapping)
    assert reason is not None and "NCCL" in reason
    auto_single_node = SimpleNamespace(strategy=AllReduceStrategy.AUTO, mnnvl_allreduce=None)
    assert hop.native_allreduce_ineligibility(auto_single_node, mapping) is not None
    mnnvl = SimpleNamespace(strategy=AllReduceStrategy.AUTO, mnnvl_allreduce=object())
    assert hop.native_allreduce_ineligibility(mnnvl, mapping) is None
    # Call sites that do not pass their all-reduce: the TP group must have a
    # native MNNVL workspace.
    workspaces = MNNVLAllReduce.allreduce_mnnvl_workspaces
    assert mapping not in workspaces
    assert hop.native_allreduce_ineligibility(None, mapping) is not None
    workspaces[mapping] = {}
    try:
        assert hop.native_allreduce_ineligibility(None, mapping) is None
    finally:
        del workspaces[mapping]


def test_factory_declines_non_mnnvl_call_site_before_any_collective():
    """A non-MNNVL call site returns None without creating the workspace."""
    from tensorrt_llm.functional import AllReduceStrategy

    mapping = _mapping()
    nccl = SimpleNamespace(strategy=AllReduceStrategy.NCCL, mnnvl_allreduce=None)
    assert (
        hop.PostAttnHop.get_or_create(
            mapping, hop.HIDDEN_SIZE, torch.bfloat16, native_allreduce=nccl
        )
        is None
    )
    assert mapping not in hop.PostAttnHop._instances


def _enabled_hop(min_sm_count):
    h = hop.PostAttnHop.__new__(hop.PostAttnHop)
    h.enabled = True
    h.min_sm_count = min_sm_count
    return h


def test_call_eligibility_and_fallback_counters():
    h = _enabled_hop(152)
    x6 = torch.zeros(6, hop.HIDDEN_SIZE, dtype=torch.bfloat16)
    x9 = torch.zeros(9, hop.HIDDEN_SIZE, dtype=torch.bfloat16)
    gamma = torch.zeros(hop.HIDDEN_SIZE, dtype=torch.bfloat16)
    router = torch.zeros(hop.NUM_EXPERTS, hop.HIDDEN_SIZE, dtype=torch.bfloat16)
    hop.reset_engagement_counts()
    assert h.can_run(x6, x6, gamma, router)
    assert h.can_run(x6[:1], x6[:1], gamma)
    assert not h.can_run(x9, x9, gamma, router)
    assert not h.can_run(x6.half(), x6.half(), gamma)
    assert not h.can_run(x6, None, gamma)
    assert not h.can_run(x6, x6, gamma, router.float())
    counts = hop.engagement_counts()
    assert counts["native_fallback"] == 4
    assert counts[f"native_fallback:rows outside [1, {hop.MAX_ROWS}]"] == 1
    assert counts["native_fallback:dtype"] == 1
    assert counts["native_fallback:no residual"] == 1
    assert counts["native_fallback:router weight"] == 1
    assert "engaged" not in counts
    # A group whose smallest GPU has 64 SMs: the router layout needs 96 SMs
    # from 3 rows on; the router-less layout fits up to 8 rows.
    small = _enabled_hop(64)
    assert small.call_ineligibility(x6[:2], x6[:2], gamma, router) is None
    assert small.call_ineligibility(x6[:3], x6[:3], gamma, router) == "SM count"
    x8 = torch.zeros(8, hop.HIDDEN_SIZE, dtype=torch.bfloat16)
    assert small.call_ineligibility(x8, x8, gamma) is None
    hop.reset_engagement_counts()
    assert hop.engagement_counts() == {}


@pytest.mark.parametrize(
    "kwargs, hidden, dtype, expected",
    [
        (dict(world_size=8, tp_size=8, gpus_per_node=4), 6144, torch.bfloat16, None),
        (dict(world_size=4, tp_size=4, gpus_per_node=4), 6144, torch.bfloat16, "tp_size"),
        (
            dict(world_size=8, tp_size=8, gpus_per_node=4, enable_attention_dp=True),
            6144,
            torch.bfloat16,
            "attention DP",
        ),
        (dict(world_size=8, tp_size=8, gpus_per_node=4), 7168, torch.bfloat16, "hidden size"),
        (dict(world_size=8, tp_size=8, gpus_per_node=4), 6144, torch.float16, "dtype"),
    ],
)
def test_static_ineligibility(kwargs, hidden, dtype, expected):
    reason = hop.static_ineligibility(Mapping(rank=0, **kwargs), hidden, dtype)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


def test_disabled_hop_never_runs():
    disabled = hop.PostAttnHop.__new__(hop.PostAttnHop)
    disabled.enabled = False
    x = torch.zeros(6, hop.HIDDEN_SIZE, dtype=torch.bfloat16)
    assert not disabled.can_run(x, x, torch.zeros(hop.HIDDEN_SIZE, dtype=torch.bfloat16))


def test_hop_call_sites_keep_dense_layers_native():
    """Only PRE_MOE_FUSION sites (MoE and MTP layers) call the hop.

    The dense layers' native chain (plain all-reduce in o_proj, then a separate
    add+RMSNorm) normalizes the FP32 sum, which the hop does not reproduce, so
    the target must not route ``forward_mlp`` through the hop, and the hop's adoption
    must not touch o_proj's all-reduce.
    """
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import layers

    source = Path(layers.__file__).read_text()
    assert source.count("self._post_attn_hop(\n") == 2
    for method in (layers.GlmDecoderLayer.forward_MoE, layers.GlmMtpLayer.forward):
        assert "self._post_attn_hop(\n" in inspect.getsource(method)
    assert "post_attn_hop" not in inspect.getsource(layers.GlmDecoderLayer.forward_mlp)
    adoption = inspect.getsource(layers.GlmDecoderLayer._adopt_pre_moe)
    assert "enable_allreduce" not in adoption
    assert "o_proj" not in adoption


def test_adoption_passes_the_native_allreduce():
    """The hop serves a layer only if that layer's native AllReduce is MNNVL."""
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import layers

    adoption = inspect.getsource(layers.GlmDecoderLayer._adopt_pre_moe)
    assert '"make_post_attn_hop")' in adoption
    assert "native_allreduce=self.allreduce" in adoption
