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
"""The query_front packet through the fused_kv_indexer_projection entry (``fused_kv_dsa_proj``)
against the host package's own path on the same hidden state (real weights, single GPU).

* full-indexer target layer 0 and the MTP layer 78 (draft_absorbed_query): absorbed_query's
  ``project`` with its native bridges (``fused_a`` + q_a norm next to the
  ``wk | weights_proj`` GEMM + k_norm on the aux stream) vs fused_kv_indexer_projection +
  the same norms + the same absorbed_query kernel. Every packet tensor that depends only
  on ``kva`` must be bit-identical; the tensors derived from ``kw`` (indexer
  key FP8 + scale, indexer weights) are bit-identical when ``kw`` is, and
  otherwise held to the frozen budget.
* shared-indexer target layer 3: shared_indexer_absorbed_query's ``project`` (native ``fused_a`` +
  shared_indexer_absorbed_query kernel) and absorbed_query's shared-layer path vs
  fused_kv_indexer_projection + the same kernels: bit-identical.
* static declines (layer kinds not selected, ineligible module) return
  ``None`` before any GPU work; a compile error propagates and is not
  cached.

The feature selection is set by replacing ``dispatch.enabled_features``
(test-only), so these tests do not need the target's model.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "absorbed_query"))

import absorbed_query_fixture  # noqa: E402
import fused_kv_utils as U  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not absorbed_query_fixture.checkpoint_available(),
    reason="needs a GPU and the GLM-5.2 checkpoint",
)

KW_EXACT = os.environ.get("GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_KW_EXACT", "0") == "1"
CTX = 491061
GEN_META = SimpleNamespace(num_contexts=0, num_generations=1)
# Packet slots (AbsorbedQueryPacket.proj_outputs) of a full-indexer layer.
KVA_SLOTS = (0, 1, 2, 3, 4, 8)  # q, compressed_kv, k_pe, latent, indexer q_fp8, q_scale
KW_SLOTS = (5, 6, 7)  # indexer k_fp8, k_scale, weights


@pytest.fixture
def features(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        entry,
    )

    def select(ids):
        monkeypatch.setattr(dispatch, "enabled_features", lambda: frozenset(ids))

    select(
        {
            "absorbed_query",
            "draft_absorbed_query",
            "shared_indexer_absorbed_query",
            "fused_kv_indexer_projection",
        }
    )
    monkeypatch.setattr(entry, "DEFAULT_LAYERS", ("full", "shared", "draft"))
    return select


def _reset(attn):
    for name in (
        "_absorbed_query",
        "_draft_absorbed_query",
        "_shared_indexer_query",
        "_fused_kv_projection",
    ):
        setattr(attn, name, None)


def _pos(rows):
    return (CTX + torch.arange(rows, dtype=torch.int32, device="cuda")).unsqueeze(0)


def _assert_same(a, b, tag):
    assert a.shape == b.shape and a.dtype == b.dtype, tag
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), tag


@pytest.mark.parametrize("rows", [1, 6, 8])
@pytest.mark.parametrize("layer", [0, U.MTP_LAYER])
@torch.inference_mode()
def test_full_layer_packet(features, layer, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        absorb_for,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.entry import (  # noqa: E501
        fused_kv_dsa_proj,
    )

    attn, gamma = U.module(layer)
    _reset(attn)
    h = U.native_hidden(rows, seed=31 + rows, gamma=gamma)
    pos = _pos(rows)
    ref = AbsorbedQueryProjection(attn).project(pos, h, absorb=absorb_for(attn))
    got = fused_kv_dsa_proj(attn, pos, h, GEN_META)
    torch.cuda.synchronize()
    assert got is not None and attn._fused_kv_projection not in (None, False)
    assert len(got.proj_outputs) == len(ref.proj_outputs) == 9
    for i in KVA_SLOTS:
        _assert_same(got.proj_outputs[i], ref.proj_outputs[i], f"slot {i}")
    assert (got.fused_q is None) == (ref.fused_q is None)
    if ref.fused_q is not None:
        _assert_same(got.fused_q[..., :512], ref.fused_q[..., :512], "fused_q")
    k_fp8, k_scale, weights = (got.proj_outputs[i] for i in KW_SLOTS)
    nk_fp8, nk_scale, nweights = (ref.proj_outputs[i] for i in KW_SLOTS)
    if KW_EXACT:
        for i in KW_SLOTS:
            _assert_same(got.proj_outputs[i], ref.proj_outputs[i], f"slot {i}")
    else:
        torch.testing.assert_close(
            k_fp8.float() * k_scale, nk_fp8.float() * nk_scale, rtol=0.1, atol=1e-2
        )
        torch.testing.assert_close(weights, nweights, rtol=0.1, atol=1e-2)
    diff = {
        f"slot{i}": float(
            (
                got.proj_outputs[i].contiguous().view(torch.uint8)
                != ref.proj_outputs[i].contiguous().view(torch.uint8)
            )
            .float()
            .mean()
        )
        for i in KW_SLOTS
    }
    print(
        f"GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_ENTRY layer={layer} rows={rows} kw_derived_byte_diff={diff}"
    )


@pytest.mark.parametrize("host", ["shared_indexer_absorbed_query", "absorbed_query"])
@pytest.mark.parametrize("rows", [1, 6, 8])
@torch.inference_mode()
def test_shared_layer_packet(features, host, rows):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.query_proj import (  # noqa: E501
        AbsorbedQueryProjection,
        absorb_for,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query.shared_indexer import (  # noqa: E501
        SharedIndexerProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.entry import (  # noqa: E501
        fused_kv_dsa_proj,
    )

    if host == "absorbed_query":
        features({"absorbed_query", "fused_kv_indexer_projection"})
    attn, gamma = U.module(3)
    _reset(attn)
    h = U.native_hidden(rows, seed=71 + rows, gamma=gamma)
    pos = _pos(rows)
    if host == "shared_indexer_absorbed_query":
        ref = SharedIndexerProjection(attn).project(h)
    else:
        ref = AbsorbedQueryProjection(attn).project(pos, h, absorb=absorb_for(attn))
    got = fused_kv_dsa_proj(attn, pos, h, GEN_META)
    torch.cuda.synchronize()
    assert got is not None
    assert len(got.proj_outputs) == len(ref.proj_outputs) == 4
    for i in range(4):
        _assert_same(got.proj_outputs[i], ref.proj_outputs[i], f"slot {i}")
    assert (got.fused_q is None) == (ref.fused_q is None)
    if ref.fused_q is not None:
        _assert_same(got.fused_q[..., :512], ref.fused_q[..., :512], "fused_q")


@torch.inference_mode()
def test_static_declines_before_gpu_work(features, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        entry,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.entry import (  # noqa: E501
        fused_kv_dsa_proj,
    )

    attn, gamma = U.module(0)
    h = U.native_hidden(6, seed=3, gamma=gamma)
    # Layer kind not selected.
    _reset(attn)
    monkeypatch.setattr(entry, "DEFAULT_LAYERS", ("shared",))
    assert fused_kv_dsa_proj(attn, _pos(6), h, GEN_META) is None
    assert attn._fused_kv_projection is False
    # fused_kv_indexer_projection not selected.
    _reset(attn)
    monkeypatch.setattr(entry, "DEFAULT_LAYERS", ("full",))
    features({"absorbed_query"})
    assert fused_kv_dsa_proj(attn, _pos(6), h, GEN_META) is None
    # Per call: context batch, too many rows, non-contiguous input.
    features({"absorbed_query", "fused_kv_indexer_projection"})
    _reset(attn)
    assert (
        fused_kv_dsa_proj(attn, _pos(6), h, SimpleNamespace(num_contexts=1, num_generations=0))
        is None
    )
    h9 = U.native_hidden(9, seed=4, gamma=gamma)
    assert fused_kv_dsa_proj(attn, _pos(9), h9, GEN_META) is None
    assert fused_kv_dsa_proj(attn, _pos(3), h9[::3], GEN_META) is None
    _reset(attn)


def test_layer_kinds_and_default_selection(monkeypatch):
    """Layer 3 is the dense-fed shared layer; the default serves only it."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        entry,
    )

    def mla(idx, full):
        return SimpleNamespace(
            layer_idx=idx,
            num_hidden_layers=78,
            mqa=SimpleNamespace(indexer=object() if full else None),
        )

    assert entry.layer_kind(mla(0, True)) == "full"
    assert entry.layer_kind(mla(3, False)) == "dense_shared"
    assert entry.layer_kind(mla(7, False)) == "shared"
    assert entry.layer_kind(mla(78, True)) == "draft"
    served = entry.served_layer_kinds()
    assert served == ("dense_shared",)
    assert entry.kind_selected("dense_shared", served)
    for kind in ("full", "shared", "draft"):
        assert not entry.kind_selected(kind, served)
    assert entry.kind_selected("dense_shared", ("shared",))
    assert not entry.kind_selected("shared", ("dense_shared",))


@torch.inference_mode()
def test_default_selection_serves_layer3_only(features, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        entry,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.entry import (  # noqa: E501
        fused_kv_dsa_proj,
    )

    monkeypatch.setattr(entry, "DEFAULT_LAYERS", ("dense_shared",))
    full, gamma0 = U.module(0)
    _reset(full)
    assert (
        fused_kv_dsa_proj(full, _pos(6), U.native_hidden(6, seed=8, gamma=gamma0), GEN_META) is None
    )
    assert full._fused_kv_projection is False
    shared, gamma3 = U.module(3)
    _reset(shared)
    got = fused_kv_dsa_proj(shared, _pos(6), U.native_hidden(6, seed=9, gamma=gamma3), GEN_META)
    assert got is not None
    _reset(full)
    _reset(shared)


@torch.inference_mode()
def test_compile_error_is_not_masked(features, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        projection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.entry import (  # noqa: E501
        fused_kv_dsa_proj,
    )

    def _fail(self):
        raise RuntimeError("injected projection compile failure")

    attn, gamma = U.module(0)
    _reset(attn)
    monkeypatch.setattr(projection.FusedKvProjection, "precompile", _fail)
    with pytest.raises(RuntimeError, match="injected projection compile failure"):
        fused_kv_dsa_proj(attn, _pos(6), U.native_hidden(6, seed=5, gamma=gamma), GEN_META)
    assert attn._fused_kv_projection is None, "a failed runner must not be cached"
    _reset(attn)
