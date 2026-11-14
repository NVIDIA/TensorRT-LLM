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
"""deferred_norm: deferred RMSNorm across the post-MoE hop and the next layer's
projection, with staged projection weights (the deferred-norm packet; a numerical variant).

Requires ``post_moe_hop`` (the hop) and ``absorbed_query`` (the next layer's query path). Registered
entry points (``dispatch.get_impl("deferred_norm", name)``):

* ``deferred_hop``: :class:`DeferredNormHop` factory, built at model
  construction from a layer's engaged ``PostMoeHop`` (collective); with
  ``routed_moe_into_hop`` also selected, its ``enable_routed_input()`` (collective) lets
  ``hop_routed`` serve the routed input routed_moe writes into the post_moe_hop;
* ``dsa_proj(mla, position_ids, packet, attn_metadata)``: the query_front
  packet of a target layer whose hidden state is a
  :class:`DeferredNormHidden` (absorbed_query / shared_indexer_absorbed_query kernels on the
  deferred_norm projection), or ``None`` to fall back;
* ``is_deferred(hidden_states)``: whether a hidden state is a deferred_norm packet.

Rounding changes versus native (see ``hop_kernel.py`` and
``proj_kernel.py``): the next-layer projection rounds ``bf16(x * gamma)``
instead of ``bf16((x * r) * gamma)`` and applies ``r`` after its fp32
accumulation (also for the indexer ``wk | weights_proj`` rows of
full-indexer layers), with its own K accumulation order. The residual
stream, ``r`` itself and the fallback materialization are bit-identical to
the post_moe_hop.

Scope (``scope.py``): the served scope is ``shared``, which serves only
boundaries into shared-indexer layers, so full-indexer layers keep the post-MoE
hop and their native or absorbed-query projection (no deferred-norm rounding
change on their indexer inputs); ``all`` would serve every boundary into a
target layer. The choice is made per layer at hop construction and agreed over
the TP group.

The hop runs on the post_moe_hop workspace with post_moe_hop's protocol, error handling
included: device errors land in the workspace's sticky error words, so the
post_moe_hop health check (``dispatch.get_impl("post_moe_hop", "health_check")``, called by the
post_moe_hop adapter after warmup and at executor cleanup) also covers deferred_norm hops, and
its call counter counts them. Every inline-PTX helper of the deferred_norm kernels is
deferred_norm's own (``ptx.py``).
"""

from ... import dispatch
from .hop import DeferredNormHop, materialize
from .packet import DeferredNormHidden, is_deferred
from .projection import DeferredNormProjConfig, DeferredNormProjection
from .query_entry import deferred_norm_dsa_proj

dispatch.register("deferred_norm", "deferred_hop", DeferredNormHop)
dispatch.register("deferred_norm", "dsa_proj", deferred_norm_dsa_proj)
dispatch.register("deferred_norm", "is_deferred", is_deferred)

__all__ = [
    "DeferredNormHidden",
    "DeferredNormHop",
    "DeferredNormProjConfig",
    "DeferredNormProjection",
    "is_deferred",
    "materialize",
    "deferred_norm_dsa_proj",
]
