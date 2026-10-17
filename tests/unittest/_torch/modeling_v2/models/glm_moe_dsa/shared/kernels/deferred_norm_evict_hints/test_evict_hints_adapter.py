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
"""Control flow of the deferred_norm_evict_hints call sites in the target.

Without device work, checks that

* the plan runs deferred_norm_evict_hints with its closure ``(absorbed_query, post_moe_hop,
  deferred_norm)`` and turns it off with any of them;
* the DSA projection entry (``attention_forward._absorbed_query_packet``) hands a deferred_norm
  packet to deferred_norm_evict_hints's ``dsa_proj``
  exactly when deferred_norm_evict_hints is selected (deferred_norm's otherwise), and leaves other
  hidden states to the next entry;
* the layer adoption (``_adopt_post_moe``) asks deferred_norm_evict_hints's ``deferred_hop`` factory when
  deferred_norm_evict_hints is selected (deferred_norm's otherwise), with the decision taken on the
  host before the factory runs;
* deferred_input_prep knows deferred_norm_evict_hints's package directories and side selectors.
"""

import inspect
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plan_testlib  # noqa: E402

BASE = "absorbed_query,post_moe_hop,routed_moe_into_hop,deferred_norm,routed_moe"


@pytest.fixture
def select(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    monkeypatch.setattr(dispatch, "_log_info", lambda message: None)

    def _select(features):
        plan_testlib.select(dispatch, features, monkeypatch)
        return dispatch

    return _select


def test_selector_closure(select):
    dispatch = select(BASE + ",deferred_norm_evict_hints")
    assert dispatch.is_enabled("deferred_norm_evict_hints") and dispatch.is_enabled("deferred_norm")
    assert dispatch.REQUIRED_FEATURES["deferred_norm_evict_hints"] == frozenset(
        {"absorbed_query", "post_moe_hop", "deferred_norm"}
    )
    for base in ("absorbed_query", "post_moe_hop", "deferred_norm"):
        plan = dispatch.resolve(dispatch.ModelingV2Mode.AUTO, {dispatch.DISABLE_ENV: base})
        assert not plan.enabled("deferred_norm_evict_hints"), base


@pytest.mark.parametrize("with_evict_hints", [True, False])
def test_dsa_entry_routes_deferred_norm_packets(select, monkeypatch, with_evict_hints):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
        attention_forward,
    )

    dispatch = select(BASE + (",deferred_norm_evict_hints" if with_evict_hints else ""))
    calls = []
    packet = object()
    real_get_impl = dispatch.get_impl

    def fake_get_impl(feature, name):
        if name == "is_deferred":
            return lambda h: h is packet
        if name == "dsa_proj" and feature in ("deferred_norm", "deferred_norm_evict_hints"):
            return lambda *args: calls.append(feature) or f"{feature}-packet"
        return real_get_impl(feature, name)

    monkeypatch.setattr(dispatch, "get_impl", fake_get_impl)
    out = attention_forward._absorbed_query_packet(object(), None, packet, None)
    expect = "deferred_norm_evict_hints" if with_evict_hints else "deferred_norm"
    assert out == f"{expect}-packet" and calls == [expect]


def test_decoder_layer_selects_the_hop_factory():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmDecoderLayer,
    )

    src = inspect.getsource(GlmDecoderLayer._adopt_post_moe)
    # The formatter wraps long statements: compare with the whitespace collapsed.
    src = re.sub(r"(?<=[(\[])\s+|\s+(?=[)\]])", "", " ".join(src.split()))
    assert (
        '"deferred_norm_evict_hints" if dispatch.is_enabled("deferred_norm_evict_hints") else "deferred_norm"'
        in src
    )
    # The choice is made before the factory is called (host-side selector).
    head, tail = src.split(
        '"deferred_norm_evict_hints" if dispatch.is_enabled("deferred_norm_evict_hints") else "deferred_norm"',
        1,
    )
    assert head.rstrip().endswith("get_impl(")
    assert re.match(r', "deferred_hop",?\)\(self\.post_moe_hop\)', tail), tail[:60]


def test_evict_hints_registers_its_entry_points(select):
    dispatch = select(BASE + ",deferred_norm_evict_hints")
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        is_deferred,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints import (  # noqa: E501
        EvictHintsDeferredHop,
        evict_hints_dsa_proj,
    )

    assert dispatch.get_impl("deferred_norm_evict_hints", "deferred_hop") is EvictHintsDeferredHop
    assert dispatch.get_impl("deferred_norm_evict_hints", "dsa_proj") is evict_hints_dsa_proj
    assert dispatch.get_impl("deferred_norm_evict_hints", "is_deferred") is is_deferred
