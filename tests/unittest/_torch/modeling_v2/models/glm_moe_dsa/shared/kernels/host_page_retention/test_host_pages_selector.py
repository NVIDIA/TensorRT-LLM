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
"""host_page_retention selection and call site.

A plan runs ``host_page_retention`` with its closure, ``post_attention_hop``, and turning that
off turns it off; the target's executor events install the host policy once after warmup,
after the post-MoE hop health check.
"""

import os
import sys
import types

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (
    dispatch,
    runtime,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plan_testlib  # noqa: E402

B1 = (
    "absorbed_query,indexer_topk,attention_output_tail,post_moe_hop,draft_front,router_gemm,"
    "post_attention_hop,draft_tail,sparse_mla_decode,draft_absorbed_query,rope_kv_append,"
    "draft_loop,shared_indexer_absorbed_query,routed_moe,cross_rank_topk,"
    "attention_output_tail_prefetch,attention_output_into_hop,routed_moe_into_hop,deferred_norm,"
    "dense_mlp,deferred_input_prep"
)


def test_closure_is_attn_hop():
    assert dispatch.REQUIRED_FEATURES["host_page_retention"] == frozenset({"post_attention_hop"})
    assert "host_page_retention" not in dispatch.NUMERICAL_VARIANTS  # allocator policy only


def test_selected_with_adapter(monkeypatch):
    paths = frozenset(B1.split(",")) | {"host_page_retention"}
    monkeypatch.setattr(dispatch, "_log_info", lambda message: None)
    assert plan_testlib.select(dispatch, paths, monkeypatch) == paths
    assert dispatch.is_enabled("host_page_retention")
    plan = dispatch.resolve(
        dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: "post_attention_hop"}
    )
    assert not plan.enabled("host_page_retention")


def test_the_host_policy_is_installed_after_warmup_after_the_hop_health_check(monkeypatch):
    calls = []
    kernels = runtime.__name__.rsplit(".", 1)[0] + ".kernels"
    monkeypatch.setitem(
        sys.modules,
        kernels + ".post_moe_hop",
        types.SimpleNamespace(check_health=lambda *args, **kwargs: calls.append(("health", args))),
    )
    monkeypatch.setitem(
        sys.modules,
        kernels + ".host_page_retention",
        types.SimpleNamespace(install_host_policy=lambda where: calls.append(("policy", where))),
    )
    paths = {"post_moe_hop", "host_page_retention"}
    plan = types.SimpleNamespace(enabled=lambda name: name in paths)
    events = runtime.GlmExecutorEvents(plan, types.SimpleNamespace(enable_attention_dp=False))

    events.after_warmup()
    assert calls == [("health", ("after_warmup",)), ("policy", "after_warmup")]
    paths.discard("post_moe_hop")
    calls.clear()
    events.after_warmup()
    assert calls == [("policy", "after_warmup")]
