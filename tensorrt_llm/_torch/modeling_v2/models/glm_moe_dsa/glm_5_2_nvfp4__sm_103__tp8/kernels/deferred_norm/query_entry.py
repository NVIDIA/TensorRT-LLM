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
"""deferred_norm entry into the absorbed_query / shared_indexer_absorbed_query path (reachable only
with ``deferred_norm``).

When the previous layer's hop deferred its RMSNorm (``deferred_norm``), the
hidden state of a target layer is a ``DeferredNormHidden`` packet. This entry
replaces exactly the two native GEMM bridges of absorbed_query / shared_indexer_absorbed_query that
consume the normalized hidden state -- the fused-A ``kv_a_proj_with_mqa`` GEMM and, on
full-indexer layers, the indexer ``wk | weights_proj`` GEMM -- by the deferred_norm
staged projection, and then runs the unchanged absorbed_query / shared_indexer_absorbed_query
kernels on their outputs, in the order the adapter uses: shared_indexer_absorbed_query's
shared-layer kernel when ``shared_indexer_absorbed_query`` is selected and the layer qualifies, else
absorbed_query's. On full-indexer layers absorbed_query's fused kernel takes the native
``q_a_layernorm`` and indexer ``k_norm`` outputs (absorbed_query's bridges); they run here on the
projected outputs. Like every absorbed_query-package entry point it validates the package selectors
(``check_package_selectors``) before its first per-module decision.

It returns ``None`` before any GPU work outside the support domain; the
caller then materializes the packet (post_moe_hop's normalized output) and runs the
next path. absorbed_query, draft_absorbed_query and shared_indexer_absorbed_query entry points never
reach this module.
"""

from typing import Optional

import torch

from ...dispatch import is_enabled, log_capability
from ..absorbed_query.query_proj import (
    _KV_LORA,
    AbsorbedQueryPacket,
    AbsorbedQueryProjection,
    absorb_for,
    call_eligibility,
    check_package_selectors,
    indexer_tail_from_projected,
    module_eligibility,
)

_SHARED_QUERY_MAX_ROWS = 8


def _deferred_norm_runner(mla):
    """Per-module deferred_norm projection (cached in ``mla._deferred_norm_projection``) or False.

    Every decline here is a static check of replicated module state and the
    environment, identical on all TP ranks. The kernel itself was compiled
    by the producing ``DeferredNormHop`` before its cross-rank agreement, so
    ``precompile`` is a cache hit; a compile error is not masked by a
    rank-local fallback (one rank rounding differently from the others would
    diverge the replicated caches and top-k sets) and propagates.
    """
    runner = getattr(mla, "_deferred_norm_projection", None)
    if runner is not None:
        return runner
    from .projection import DeferredNormProjection, projection_eligibility

    check_package_selectors()
    ok, reason = module_eligibility(mla)
    kw = None
    if ok:
        ok, reason, kw = projection_eligibility(mla)
    runner = False
    if ok:
        proj = DeferredNormProjection(mla, kw_weight=kw)
        why = proj.supported()
        if why is not None:
            ok, reason = False, why
        else:
            proj.precompile()
            runner = proj
    log_capability("deferred_norm", f"projection_layer{mla.layer_idx}", ok, reason)
    mla._deferred_norm_projection = runner
    return runner


def _shared_query_runner(mla):
    """shared_indexer_absorbed_query's cached shared-layer runner when
    ``shared_indexer_absorbed_query`` would serve this module."""
    if not is_enabled("shared_indexer_absorbed_query"):
        return False
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
    return runner


def _absorbed_query_runner(mla):
    runner = getattr(mla, "_absorbed_query", None)
    if runner is None:
        ok, reason = module_eligibility(mla)
        log_capability("absorbed_query", f"layer{mla.layer_idx}", ok, reason)
        runner = AbsorbedQueryProjection(mla) if ok else False
        mla._absorbed_query = runner
    return runner


# Placement of the native k_norm in the deferred_norm full-layer path: "aux" (on the
# indexer's aux stream next to the q_a norm when multi-stream is enabled and
# absorbed_query's indexer_k_on_aux is set, joined before absorbed_query's kernel), "k_first" or
# "q_first" (both on the current stream, in that order).
K_NORM_MODES = ("aux", "k_first", "q_first")
DEFAULT_K_NORM_MODE = "aux"


def indexer_query_from_projected(
    query_runner: AbsorbedQueryProjection,
    kva: torch.Tensor,
    kw: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    absorb: bool,
    k_norm_mode: Optional[str] = None,
) -> AbsorbedQueryPacket:
    """absorbed_query's fused full-layer kernel on projected ``kva`` and ``kw``.

    The kernel consumes the native ``q_a_layernorm`` of the q_a columns of
    ``kva`` and the native ``k_norm`` of the key columns of ``kw`` (absorbed_query's
    bridges). Both are computed here from the deferred_norm projection's outputs with
    the native calls absorbed_query's ``native_bridges`` makes; ``k_norm_mode`` (default
    ``DEFAULT_K_NORM_MODE``) only places them in the stream(s), the
    arithmetic is the same.
    """
    mla = query_runner.mla
    indexer = mla.mqa.indexer
    prenorm = query_runner.config.prenorm
    mode = DEFAULT_K_NORM_MODE if k_norm_mode is None else k_norm_mode
    assert mode in K_NORM_MODES, mode

    def main():
        return query_runner.native_q_norm(kva) if prenorm else None

    def aux():
        return indexer.k_norm(kw[:, : indexer.head_dim])

    if mode == "aux" and query_runner.config.indexer_k_on_aux:
        from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

        qn, k_ln = maybe_execute_in_parallel(
            main, aux, query_runner.events[0], query_runner.events[1], indexer.aux_stream
        )
    elif mode == "k_first":
        k_ln = aux()
        qn = main()
    else:
        qn = main()
        k_ln = aux()
    return query_runner.launch_indexer(kva, kw, k_ln, None, 0, position_ids, absorb, qn)


def deferred_norm_dsa_proj(
    mla, position_ids: Optional[torch.Tensor], packet, attn_metadata
) -> Optional[AbsorbedQueryPacket]:
    """The query_front packet of a target layer whose input is a deferred_norm packet.

    Same outputs and side effects as absorbed_query's ``dsa_proj`` (or
    shared_indexer_absorbed_query's ``shared_dsa_proj`` when ``shared_indexer_absorbed_query`` is
    selected), with the fused-A and indexer wk|weights_proj outputs taken from the deferred_norm
    projection.
    """
    if not (is_enabled("deferred_norm") and is_enabled("absorbed_query")):
        return None
    runner = _deferred_norm_runner(mla)
    if runner is False:
        return None
    ok, _ = call_eligibility(mla, packet.residual, attn_metadata)
    if not ok or packet.num_rows > runner_max_rows():
        return None
    indexer = mla.mqa.indexer
    shared_query = _shared_query_runner(mla) if indexer is None else False
    query_runner = _absorbed_query_runner(mla) if not shared_query else False
    if not shared_query and not query_runner:
        return None
    if query_runner and query_runner.config.prefetch:
        return None
    if query_runner and indexer is not None and not query_runner.indexer_fusable():
        fused_tail = False
    else:
        fused_tail = indexer is not None

    if runner.config.prefetch_next and not runner.prefetch:
        # Weights of the kernels that follow on this layer (L2 hint).
        nxt = [mla.q_b_proj.weight, mla.k_b_proj_trans]
        if indexer is not None:
            nxt.insert(1, indexer.wq_b.weight)
        runner.set_prefetch(nxt)
    kva, kw = runner.project(packet)
    if shared_query:
        q, latent, fused_q = shared_query.launch(kva)
        return AbsorbedQueryPacket([q, latent[:, :_KV_LORA], latent[:, _KV_LORA:], latent], fused_q)
    absorb = absorb_for(mla)
    if fused_tail:
        return indexer_query_from_projected(query_runner, kva, kw, position_ids, absorb)
    q, qr, latent, fused_q = query_runner.launch(kva, need_qr=indexer is not None, absorb=absorb)
    outs = [q, latent[:, :_KV_LORA], latent[:, _KV_LORA:], latent]
    if indexer is not None:
        outs.extend(indexer_tail_from_projected(indexer, qr, kw, position_ids))
    return AbsorbedQueryPacket(outs, fused_q)


def runner_max_rows() -> int:
    from .projection import MAX_ROWS

    return MAX_ROWS
