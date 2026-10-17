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
"""The draft_routed_moe call site in the target's MoE.

* a plan runs draft_routed_moe alone and next to F's GEN set, draft_routed_moe extends no
  package and has a single numerical form; deferred_input_prep's gate
  classifies draft_routed_moe's launch-configuration selectors as draft_routed_moe's;
* ``GlmMoE.compute_routed_output`` dispatches to draft_routed_moe only when it is selected, before
  routed_moe, and keeps the native path when draft_routed_moe declines;
* an NVFP4 MoE layer (every target layer) declines draft_routed_moe before any device work,
  so routed_moe / hop_routing see exactly the calls they see in F;
* through ``GlmMoE.compute_routed_output`` on a decode step the MTP rank
  shard returns draft_routed_moe's output, bit-identical to the native leaf.
"""

import inspect
import os
import re
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "routed_moe"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "..", "..", "shared", "kernels"))
import plan_testlib  # noqa: E402

F_GEN = (
    "absorbed_query,indexer_topk,attention_output_tail,post_moe_hop,draft_front,router_gemm,"
    "post_attention_hop,draft_tail,sparse_mla_decode,draft_absorbed_query,rope_kv_append,"
    "draft_loop,shared_indexer_absorbed_query,routed_moe,cross_rank_topk,"
    "attention_output_tail_prefetch,attention_output_into_hop,routed_moe_into_hop,"
    "deferred_norm,dense_mlp,deferred_input_prep,hop_routing,oproj_weight_staging,"
    "fused_kv_indexer_projection,host_page_retention"
)


def _dispatch():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    return dispatch


def test_selector_facts():
    d = _dispatch()
    assert "draft_routed_moe" in d.SELECTOR_PATHS
    assert "draft_routed_moe" not in d.REQUIRED_FEATURES
    assert "draft_routed_moe" not in d.NUMERICAL_VARIANTS
    assert plan_testlib.plan_running(d, "draft_routed_moe").enabled("draft_routed_moe")
    no_input_prep = frozenset(F_GEN.split(",")) - {"deferred_input_prep"}
    assert plan_testlib.plan_running(d, no_input_prep | {"draft_routed_moe"}).enabled(
        "draft_routed_moe"
    )


def test_call_site_order():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmMoE,
    )

    src = inspect.getsource(GlmMoE.compute_routed_output)
    # The formatter wraps long statements: compare with the whitespace collapsed.
    src = re.sub(r"(?<=[(\[])\s+|\s+(?=[)\]])", "", " ".join(src.split()))
    draft_routed_moe = src.index('is_enabled("draft_routed_moe")')
    assert draft_routed_moe < src.index('is_enabled("routed_moe")')
    assert src.index('get_impl("draft_routed_moe", "routed_output")') > draft_routed_moe


def test_nvfp4_layer_declines():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    import draft_moe_jit
    import routed_moe_fixtures as wfx

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.draft_routed_moe import (  # noqa: E501
        routed_moe as rm,
    )

    # With the op loaded, the decline names the leaf type, not the missing op.
    draft_moe_jit.ensure_op(
        os.environ.get("GLM_MOE_DSA_DRAFT_ROUTED_MOE_JIT_ROOT", "/tmp/draft_routed_moe_jit")
    )
    layer = wfx.build_native_layer(10, 0)
    x = torch.zeros(6, 6144, dtype=torch.bfloat16, device="cuda")
    lg = torch.zeros(6, 256, dtype=torch.float32, device="cuda")
    ok, reason = rm.eligibility(layer.module, layer.bias, x, lg, True)
    print(f"GLM_MOE_DSA_DRAFT_ROUTED_MOE_NVFP4_DECLINE {reason}", flush=True)
    assert not ok and "BF16 leaf" in reason


def test_patched_call_site_bit_exact(monkeypatch):
    """``GlmMoE.compute_routed_output`` on a decode step, on a stand-in module that holds the MTP leaf."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    import draft_moe_fixtures as fx
    import draft_moe_jit

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.layers import (  # noqa: E501
        GlmMoE,
    )

    draft_moe_jit.ensure_op(
        os.environ.get("GLM_MOE_DSA_DRAFT_ROUTED_MOE_JIT_ROOT", "/tmp/draft_routed_moe_jit")
    )
    layer = fx.build_native(0)
    plan_testlib.select(_dispatch(), "draft_routed_moe", monkeypatch)

    class Gate:
        e_score_correction_bias = layer.bias

    class Stub:
        use_dp = False
        mapping = None
        router_gemm = None
        gate = Gate()
        experts = layer.module

    calls = {"n": 0}
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    inner = dispatch.get_impl("draft_routed_moe", "routed_output")

    def counted(*a, **k):
        out = inner(*a, **k)
        calls["n"] += out is not None
        return out

    dispatch.register("draft_routed_moe", "routed_output", counted)
    try:
        for rows in (1, 6):
            x = fx.hidden_rows(layer, rows, seed=40 + rows)
            lg = layer.logits(x)
            with dispatch.step(dispatch.Phase.DECODE):
                out = GlmMoE.compute_routed_output(Stub(), x, None, None, True, lg)
            assert torch.equal(out.view(torch.int16), layer.native(x, lg).view(torch.int16))
    finally:
        dispatch.register("draft_routed_moe", "routed_output", inner)
    assert calls["n"] == 2
