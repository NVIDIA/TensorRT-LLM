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
"""Collective tests of the deferred_norm scope (8 ranks over 2 hosts, TP8).

The ``shared`` scope, this target's, keeps the deferred-norm hop only on boundaries whose next
layer is a shared-indexer layer. The tests that compare against the hop's unrestricted scope
``all`` build the hop with that scope directly. Checks:

* unrestricted scope: the hop is the default deferred_norm hop (engaged, no next-layer
  lookup, no per-boundary log key);
* ``shared``, next layer full: the hop is disengaged on every rank and the
  layer's choice (``can_run`` / ``can_run_routed`` false; the decoder layer
  keeps ``None``) is post_moe_hop's hop, whose output is post_moe_hop's by construction; the
  routed input is never enabled on such a hop;
* ``shared``, next layer shared: the hop is engaged and its packets (raw
  tuple and routed_moe_into_hop routed input in ``fp32`` and ``identity``) are bit-identical
  to the unrestricted hop's packets on the same inputs and workspace, and
  their materialization equals post_moe_hop's normalized output (``a_frag``, ``sumsq``,
  residual, ``materialize``);
* rank-differing scopes: one rank with another scope or an unknown
  next-layer type can only disengage a boundary on every rank (one agreement,
  the same number of collectives in every scope);
* the decoder-layer path: a stand-in with the target decoder layer's adoption order
  (``GlmDecoderLayer._adopt_post_moe``: post_moe_hop assigned, then the deferred_norm factory
  called with only the post_moe_hop) and the real GLM-5.2 model config resolves layers 5, 9, 73 ->
  full (post_moe_hop) and 3, 6, 76 -> shared (deferred_norm hop) on every rank.
"""

import importlib
import os
import sys
import types

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "post_moe_hop"))
sys.path.insert(0, os.path.join(_HERE, "..", "absorbed_query"))

import post_moe_hop_mpi_utils as M  # noqa: E402

FULL_NEXT = set(range(6, 78, 4))


def _env():
    env = M.mpi_env()
    if env.world != 8:
        pytest.skip("the deferred_norm hop is specialized to TP8")
    return env


_POST_MOE_HOP = {}


def _post_moe_hop(mode=None):
    """One post_moe_hop per routed-input mode (``None``: raw tuple only), collective."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.post_moe_hop import (  # noqa: E501
        PostMoeHop,
    )

    env = _env()
    if mode not in _POST_MOE_HOP:
        hop_stub = PostMoeHop(env.mapping, M.GLM_HIDDEN, M.GLM_TOP_K, require_multi_node=False)
        ok = hop_stub.engaged
        if mode is not None:
            if not hasattr(hop_stub, "enable_routed_input"):
                pytest.skip(
                    "the installed post_moe_hop hop has no routed_moe_into_hop routed input"
                )
            ok = ok and hop_stub.enable_routed_input(mode)
        assert M.all_ranks(ok), hop_stub.reason
        _POST_MOE_HOP[mode] = hop_stub
    return _POST_MOE_HOP[mode]


UNRESTRICTED = "all"
_HOP_MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa."
    "shared.kernels.deferred_norm.hop"
)


def _hop(monkeypatch, hop_stub, scope_value, **kwargs):
    """A deferred_norm hop built under this target's scope (``scope_value`` None or ``shared``).

    ``UNRESTRICTED``: the hop's unrestricted scope, which this target does not run; the hop
    module's scope reader returns it for this construction.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHop,
    )

    with monkeypatch.context() as patch:
        if scope_value == UNRESTRICTED:
            hop_module = importlib.import_module(_HOP_MODULE)
            patch.setattr(hop_module, "selected_scope", lambda: UNRESTRICTED)
        return DeferredNormHop(hop_stub, **kwargs)


def _metadata(rows):
    return types.SimpleNamespace(all_rank_num_tokens=None, num_contexts=0, num_generations=rows)


def _same_packet(pa, ra, pb, rb):
    """Same packet content (the fragment buffer holds 8 rows; rows >= R are not written)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        rows_from_frag,
    )

    torch.cuda.synchronize()
    rows, k = ra.shape
    return (
        torch.equal(rows_from_frag(pa.a_frag, rows, k), rows_from_frag(pb.a_frag, rows, k))
        and torch.equal(pa.sumsq, pb.sumsq)
        and torch.equal(ra, rb)
        and pa.eps == pb.eps
    )


def _materialized_is_post_moe_hop(packet, h_post_moe_hop):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        materialize,
    )

    h = materialize(packet)
    torch.cuda.synchronize()
    return torch.equal(h, h_post_moe_hop)


def test_default_scope_is_the_b11_hop(monkeypatch):
    hop_stub = _post_moe_hop()
    # The next-layer argument is ignored by the unrestricted scope.
    hop = _hop(monkeypatch, hop_stub, UNRESTRICTED, next_layer_full_indexer=True)
    ok = (
        hop.engaged
        and hop.scope == "all"
        and hop.next_layer_full is None
        and hop.next_layer_idx is None
        and hop.reason == ""
    )
    assert M.all_ranks(ok), (hop.scope, hop.reason)


def test_shared_scope_full_next_layer_keeps_post_moe_hop(monkeypatch):
    hop_stub = _post_moe_hop()
    hop = _hop(monkeypatch, hop_stub, None, next_layer_full_indexer=True)
    ok = (
        not hop.engaged
        and hop.scope == "shared"
        and hop.next_layer_full is True
        and "full-indexer layer" in hop.reason
        and not hop.can_run(6, _metadata(6))
        and not hop.can_run_routed(6, _metadata(6))
    )
    assert M.all_ranks(ok), hop.reason
    # The decoder layer enables the routed input only on a kept (engaged) hop;
    # called anyway, the enable declines on every rank without a collective.
    assert M.all_ranks(not hop.enable_routed_input() and not hop.routed_engaged)


@pytest.mark.parametrize("with_shared", [True, False])
@pytest.mark.parametrize("rows", [1, 2, 6, 8])
def test_shared_scope_shared_next_layer_equals_default(monkeypatch, rows, with_shared):
    hop_stub = _post_moe_hop()
    default = _hop(monkeypatch, hop_stub, UNRESTRICTED)
    shared = _hop(monkeypatch, hop_stub, None, next_layer_full_indexer=False)
    assert M.all_ranks(
        default.engaged
        and shared.engaged
        and shared.next_layer_full is False
        and shared.can_run(rows, _metadata(rows))
    ), shared.reason
    with torch.inference_mode():
        fc2, scales, perm, sh, residual, gamma = M.make_raw_tuple(
            rows, seed=900 + rows, with_shared=with_shared
        )
        h_post_moe_hop, res_post_moe_hop = hop_stub(
            fc2, scales, perm, sh, residual, gamma, M.GLM_EPS
        )
        pa, ra = default(fc2, scales, perm, sh, residual, gamma, M.GLM_EPS)
        pb, rb = shared(fc2, scales, perm, sh, residual, gamma, M.GLM_EPS)
        ok = (
            _same_packet(pa, ra, pb, rb)
            and torch.equal(rb, res_post_moe_hop)
            and _materialized_is_post_moe_hop(pb, h_post_moe_hop)
        )
    assert M.all_ranks(ok)


def _routed_source(mode, rows, seed):
    fc2, scales, perm, shared, residual, gamma = M.make_raw_tuple(rows, seed=seed)
    if mode == "identity":
        valid = perm >= 0
        safe = torch.where(valid, perm, torch.zeros_like(perm)).long()
        src = (
            fc2[safe.view(-1)].contiguous(),
            torch.where(valid, scales, torch.zeros_like(scales)).bfloat16().contiguous(),
        )
    else:
        valid = (perm >= 0).unsqueeze(-1)
        safe = torch.where(perm >= 0, perm, torch.zeros_like(perm)).long()
        prod = fc2[safe.view(-1)].view(rows, M.GLM_TOP_K, -1).float() * scales.float().unsqueeze(-1)
        src = torch.where(valid, prod, torch.zeros_like(prod)).sum(1).contiguous()
    return src, shared, residual, gamma


def _fill(hop_stub, mode, src, rows):
    if mode == "identity":
        buf, scales = hop_stub.routed_raw_buffers(rows)
        buf.copy_(src[0])
        scales.copy_(src[1])
        return buf
    buf = hop_stub.routed_acc_buffer(rows)
    buf.copy_(src)
    return buf


@pytest.mark.parametrize("rows", [1, 6, 8])
@pytest.mark.parametrize("mode", ["fp32", "identity"])
def test_shared_scope_routed_equals_default(monkeypatch, mode, rows):
    hop_stub = _post_moe_hop(mode)
    default = _hop(monkeypatch, hop_stub, UNRESTRICTED)
    shared = _hop(monkeypatch, hop_stub, None, next_layer_full_indexer=False)
    ok = default.enable_routed_input() and shared.enable_routed_input()
    assert M.all_ranks(ok and shared.can_run_routed(rows, _metadata(rows))), shared.routed_reason
    with torch.inference_mode():
        src, sh, residual, gamma = _routed_source(mode, rows, seed=950 + rows)
        h_post_moe_hop, res_post_moe_hop = hop_stub.hop_routed(
            _fill(hop_stub, mode, src, rows), sh, residual, gamma, M.GLM_EPS
        )
        pa, ra = default.hop_routed(
            _fill(hop_stub, mode, src, rows), sh, residual, gamma, M.GLM_EPS
        )
        pb, rb = shared.hop_routed(_fill(hop_stub, mode, src, rows), sh, residual, gamma, M.GLM_EPS)
        ok = (
            _same_packet(pa, ra, pb, rb)
            and torch.equal(rb, res_post_moe_hop)
            and _materialized_is_post_moe_hop(pb, h_post_moe_hop)
        )
        if mode == "fp32":
            torch.cuda.synchronize()
            ok = ok and bool((hop_stub.routed_acc_buffer(hop_stub.max_rows) == 0).all())
    assert M.all_ranks(ok)


@pytest.mark.parametrize("next_full", [True, False])
@pytest.mark.parametrize("odd_rank", [0, 3, 7])
def test_rank_differing_scope_agrees(monkeypatch, odd_rank, next_full):
    """One rank with the unrestricted scope: a full boundary is post_moe_hop everywhere, a
    shared boundary is deferred_norm everywhere."""
    env = _env()
    hop_stub = _post_moe_hop()
    value = UNRESTRICTED if env.rank == odd_rank else "shared"
    hop = _hop(monkeypatch, hop_stub, value, next_layer_full_indexer=next_full)
    if next_full:
        local = (
            "another rank is not capable" == hop.reason
            if env.rank == odd_rank
            else "full-indexer layer" in hop.reason
        )
        assert M.all_ranks(not hop.engaged and local), hop.reason
    else:
        assert M.all_ranks(hop.engaged), hop.reason


@pytest.mark.parametrize("odd_rank", [2, 5])
def test_unresolved_next_layer_on_one_rank(monkeypatch, odd_rank):
    """A rank that cannot resolve the next layer keeps post_moe_hop on every rank."""
    env = _env()
    hop_stub = _post_moe_hop()
    arg = None if env.rank == odd_rank else False
    hop = _hop(monkeypatch, hop_stub, "shared", next_layer_full_indexer=arg)
    if env.rank == odd_rank:
        local = "unknown" in hop.reason and "no constructing decoder layer found" in hop.reason
    else:
        local = hop.reason == "another rank is not capable"
    assert M.all_ranks(not hop.engaged and local), hop.reason


def _glm52_model_config():
    import absorbed_query_fixture

    if not absorbed_query_fixture.checkpoint_available():
        pytest.skip("needs the GLM-5.2 checkpoint config")
    return absorbed_query_fixture.build_model_config(M.mpi_env().rank)


class _DecoderLayerStandIn:
    """``GlmDecoderLayer._adopt_post_moe``'s deferred_norm wiring, in its order."""

    def __init__(self, model_config, layer_idx, post_moe_hop):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

        self.model_config = model_config
        self.layer_idx = layer_idx
        self.post_moe_hop = post_moe_hop
        hop = dispatch.get_impl("deferred_norm", "deferred_hop")(self.post_moe_hop)
        self.deferred_norm_hop = hop if hop.engaged else None
        self.deferred_norm_reason = hop.reason


def test_decoder_layer_resolves_the_next_layer():
    """Under ``shared``, this target's scope, full next layers keep the post-MoE hop."""
    hop_stub = _post_moe_hop()
    cfg = _glm52_model_config()
    ok = True
    for layer_idx in (3, 5, 6, 9, 73, 76):
        layer = _DecoderLayerStandIn(cfg, layer_idx, hop_stub)
        hop = layer.deferred_norm_hop
        if (layer_idx + 1) not in FULL_NEXT:
            good = (
                hop is not None
                and hop.next_layer_idx == layer_idx + 1
                and hop.next_layer_full is False
            )
        else:
            good = (
                hop is None
                and f"layer {layer_idx + 1} is a full-indexer layer" in layer.deferred_norm_reason
            )
        if not good:
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_SCOPE_LAYER_FAIL rank={M.mpi_env().rank} layer={layer_idx} "
                f"reason={layer.deferred_norm_reason}",
                flush=True,
            )
        ok = ok and good
    assert M.all_ranks(ok)
