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
"""fused_kv_indexer_projection entry into the absorbed_query / draft_absorbed_query /
shared_indexer_absorbed_query path (reachable only with ``fused_kv_indexer_projection``).

On a call whose hidden state is the native normalized tensor (not a deferred_norm
packet), fused_kv_indexer_projection replaces the two native GEMM bridges that consume it -- the
fused-A ``kv_a_proj_with_mqa`` GEMM and, on full-indexer layers, the indexer
``wk | weights_proj`` GEMM -- by one ``FusedKvProjection`` launch, and then runs
the unchanged consumer of the selected host package on its outputs:

* full-indexer target layer (absorbed_query) and the MTP draft layer (draft_absorbed_query): the
  native ``q_a_layernorm`` and indexer ``k_norm`` bridges on the projected outputs
  (``k_norm`` on the indexer's aux stream next to the q_a norm, joined
  before the kernel), then absorbed_query's fused indexer kernel;
* shared-indexer target layer: shared_indexer_absorbed_query's kernel when
  ``shared_indexer_absorbed_query`` is selected and the layer qualifies, else absorbed_query's query
  kernel.

The layer kinds are ``full``, ``shared``, ``dense_shared`` and ``draft``; the
served ones are ``DEFAULT_LAYERS`` (``served_layer_kinds``).
``dense_shared`` is the shared-indexer target layer whose input comes from the
post-MLP AllReduce of a dense layer (GLM-5.2: layer 3, after the dense layers
0-2); ``shared`` selects every shared-indexer layer. Every decline happens
before any GPU work and returns ``None``; the caller then runs the next path of
the adapter (absorbed_query / shared_indexer_absorbed_query / draft_absorbed_query, or native).
"""

from typing import Optional

from ...dispatch import is_enabled, log_capability
from ..absorbed_query.query_proj import (
    _KV_LORA,
    AbsorbedQueryPacket,
    AbsorbedQueryProjection,
    absorb_for,
    call_eligibility,
    check_package_selectors,
    module_eligibility,
)

#: How the decline reasons name the layer selection (``<kind> layers not selected (<label>)``).
LAYERS_LABEL = "TRTLLM_STAIRCASE_GLM_LL_FUSED_KV_INDEXER_PROJECTION_LAYERS"
LAYER_KINDS = ("full", "shared", "dense_shared", "draft")
#: The layer kinds this target serves.
DEFAULT_LAYERS = ("dense_shared",)
# GLM-5.2 runs dense MLPs in layers 0-2 (first_k_dense_replace = 3), so the
# shared-indexer layers up to layer 3 take their input from a dense layer's
# post-MLP AllReduce.
DENSE_FED_LAST_LAYER = 3


def served_layer_kinds() -> tuple:
    """The served layer kinds, ``DEFAULT_LAYERS``."""
    return DEFAULT_LAYERS


def layer_kind(mla) -> str:
    num_layers = getattr(mla, "num_hidden_layers", None)
    if num_layers is not None and mla.layer_idx is not None and mla.layer_idx >= num_layers:
        return "draft"
    if mla.mqa.indexer is not None:
        return "full"
    if mla.layer_idx is not None and mla.layer_idx <= DENSE_FED_LAST_LAYER:
        return "dense_shared"
    return "shared"


def kind_selected(kind: str, served: tuple) -> bool:
    """``shared`` also selects the ``dense_shared`` layer."""
    return kind in served or (kind == "dense_shared" and "shared" in served)


def _host_runner(mla, kind: str):
    """The absorbed_query / draft_absorbed_query / shared_indexer_absorbed_query runner that
    consumes the projection on this module (the same cached object the package's own entry point
    would create)."""
    if kind == "draft":
        if not is_enabled("draft_absorbed_query"):
            return False
        runner = getattr(mla, "_draft_absorbed_query", None)
        if runner is None:
            ok, reason = module_eligibility(mla, draft=True)
            log_capability("draft_absorbed_query", f"layer{mla.layer_idx}", ok, reason)
            runner = AbsorbedQueryProjection(mla) if ok else False
            mla._draft_absorbed_query = runner
        return runner
    if kind in ("shared", "dense_shared") and is_enabled("shared_indexer_absorbed_query"):
        runner = getattr(mla, "_shared_indexer_query", None)
        if runner is None:
            from ..absorbed_query.shared_indexer import (
                SharedIndexerProjection,
                shared_module_eligibility,
            )

            ok, reason = shared_module_eligibility(mla)
            log_capability("shared_indexer_absorbed_query", f"layer{mla.layer_idx}", ok, reason)
            runner = SharedIndexerProjection(mla) if ok else False
            mla._shared_indexer_query = runner
        if runner is not False:
            return runner
    runner = getattr(mla, "_absorbed_query", None)
    if runner is None:
        ok, reason = module_eligibility(mla)
        log_capability("absorbed_query", f"layer{mla.layer_idx}", ok, reason)
        runner = AbsorbedQueryProjection(mla) if ok else False
        mla._absorbed_query = runner
    return runner


def _fused_kv_runner(mla):
    """Per-module ``(projection, host runner, kind)`` (cached in
    ``mla._fused_kv_projection``) or False.

    Every decline is a static check of replicated module state and the
    environment, identical on all TP ranks. The kernel is compiled here (no
    GPU work); a compile error is not masked by a rank-local fallback and
    propagates.
    """
    cached = getattr(mla, "_fused_kv_projection", None)
    if cached is not None:
        return cached
    from .projection import FusedKvProjection, projection_eligibility

    check_package_selectors()
    kind = layer_kind(mla)
    ok, reason = True, ""
    if not kind_selected(kind, served_layer_kinds()):
        ok, reason = False, f"{kind} layers not selected ({LAYERS_LABEL})"
    kw = None
    if ok:
        ok, reason = module_eligibility(mla, draft=kind == "draft")
    if ok:
        ok, reason, kw = projection_eligibility(mla)
    host = False
    if ok:
        host = _host_runner(mla, kind)
        if host is False:
            ok, reason = (
                False,
                "no eligible absorbed_query / draft_absorbed_query / shared_indexer_absorbed_query consumer",
            )
    shared = kind in ("shared", "dense_shared")
    if ok and not shared and host.config.prefetch:
        ok, reason = False, "absorbed_query prefetch kernel configured"
    if ok and not shared and not host.indexer_fusable():
        ok, reason = False, "absorbed_query fused indexer kernel not eligible"
    runner = False
    if ok:
        proj = FusedKvProjection(mla, kw_weight=kw)
        why = proj.supported()
        if why is not None:
            ok, reason = False, why
        else:
            proj.precompile()
            runner = (proj, host, kind)
    log_capability("fused_kv_indexer_projection", f"layer{mla.layer_idx}", ok, reason or kind)
    mla._fused_kv_projection = runner
    return runner


def fused_kv_dsa_proj(
    mla, position_ids, hidden_states, attn_metadata
) -> Optional[AbsorbedQueryPacket]:
    """The query_front packet of a call whose hidden state is the native
    normalized tensor, with ``kva`` / ``kw`` from the fused_kv_indexer_projection.

    Same outputs and side effects as absorbed_query's ``dsa_proj``, draft_absorbed_query's
    ``mtp_dsa_proj`` or shared_indexer_absorbed_query's ``shared_dsa_proj`` on the same call;
    ``None`` (before any GPU work) outside the support domain.
    """
    if not (is_enabled("fused_kv_indexer_projection") and is_enabled("absorbed_query")):
        return None
    runner = _fused_kv_runner(mla)
    if runner is False:
        return None
    proj, host, kind = runner
    ok, _ = call_eligibility(mla, hidden_states, attn_metadata)
    if not ok or hidden_states.shape[0] > _max_rows() or not hidden_states.is_contiguous():
        return None
    kva, kw = proj.run(hidden_states)
    if kind in ("shared", "dense_shared"):
        if isinstance(host, AbsorbedQueryProjection):
            q, _, latent, fused_q = host.launch(kva, need_qr=False, absorb=absorb_for(mla))
        else:
            q, latent, fused_q = host.launch(kva)
        return AbsorbedQueryPacket([q, latent[:, :_KV_LORA], latent[:, _KV_LORA:], latent], fused_q)
    from ..deferred_norm.query_entry import indexer_query_from_projected

    return indexer_query_from_projected(host, kva, kw, position_ids, absorb_for(mla))


def _max_rows() -> int:
    from .projection import MAX_ROWS

    return MAX_ROWS
