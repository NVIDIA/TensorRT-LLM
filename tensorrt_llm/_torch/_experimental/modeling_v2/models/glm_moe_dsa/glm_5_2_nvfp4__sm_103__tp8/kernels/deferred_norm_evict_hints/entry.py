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
"""deferred_norm_evict_hints entry into the absorbed_query / shared_indexer_absorbed_query path for
a deferred-norm packet.

Same decisions, outputs and side effects as deferred_norm's ``deferred_norm_dsa_proj``
(``deferred_norm/query_entry.py``), with the fused-A (and indexer) projection launched by
``EvictHintsProjection`` (bit-identical to ``DeferredNormProjection``). It returns ``None``
before any GPU work outside the support domain; the caller then
materializes the packet.
"""

from typing import Optional

import torch

from ...dispatch import is_enabled, log_capability
from ..absorbed_query.query_proj import (
    _KV_LORA,
    AbsorbedQueryPacket,
    absorb_for,
    call_eligibility,
    check_package_selectors,
    indexer_tail_from_projected,
    module_eligibility,
)
from ..deferred_norm.projection import MAX_ROWS, projection_eligibility
from ..deferred_norm.query_entry import (
    _absorbed_query_runner,
    _shared_query_runner,
    indexer_query_from_projected,
)


def _evict_hints_runner(mla):
    """Per-module deferred_norm_evict_hints projection (cached in ``mla._evict_hints_projection``) or False.

    Static, rank-identical checks only (as deferred_norm's runner); the kernel was
    compiled by the producing ``EvictHintsDeferredHop`` before its agreement.
    """
    runner = getattr(mla, "_evict_hints_projection", None)
    if runner is not None:
        return runner
    from .projection import EvictHintsProjection

    check_package_selectors()
    ok, reason = module_eligibility(mla)
    kw = None
    if ok:
        ok, reason, kw = projection_eligibility(mla)
    runner = False
    if ok:
        proj = EvictHintsProjection(mla, kw_weight=kw)
        why = proj.supported()
        if why is not None:
            ok, reason = False, why
        else:
            proj.precompile()
            runner = proj
    log_capability("deferred_norm_evict_hints", f"projection_layer{mla.layer_idx}", ok, reason)
    mla._evict_hints_projection = runner
    return runner


def evict_hints_dsa_proj(
    mla, position_ids: Optional[torch.Tensor], packet, attn_metadata
) -> Optional[AbsorbedQueryPacket]:
    """The query_front packet of a target layer whose input is a deferred_norm packet
    (deferred_norm_evict_hints projection)."""
    if not (
        is_enabled("deferred_norm_evict_hints")
        and is_enabled("deferred_norm")
        and is_enabled("absorbed_query")
    ):
        return None
    runner = _evict_hints_runner(mla)
    if runner is False:
        return None
    ok, _ = call_eligibility(mla, packet.residual, attn_metadata)
    if not ok or packet.num_rows > MAX_ROWS:
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
