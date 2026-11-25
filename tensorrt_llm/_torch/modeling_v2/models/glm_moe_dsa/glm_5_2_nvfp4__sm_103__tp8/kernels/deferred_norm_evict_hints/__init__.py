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
"""deferred_norm_evict_hints: the deferred_norm hop + next-layer projection chain without the HBM collision.

Requires ``absorbed_query``, ``post_moe_hop`` and ``deferred_norm``. deferred_norm_evict_hints keeps
deferred_norm's deferred-norm contract (same packet, residual, ``materialize()`` and projection
outputs, bit for bit) and changes only how the chain uses the memory system. By default:

* the hop (``EvictHintsHopKernel``) is deferred_norm's hop launch unchanged;
* the projection (``EvictHintsProjKernel``) stages its weights before its wait as
  deferred_norm's does, with an ``L2::evict_first`` policy on the weight copies: the
  next layer's 32 MB are read once per iteration, and the hint keeps them
  from evicting the lines of the hop that runs under that stream.

The other schedule knobs (``EvictHintsHopConfig``: L2 prefetch of the poll slice,
evict_last clears, the shared-memory carveout that lets every projection
CTA co-reside with the hop; ``EvictHintsProjConfig``: register caps, warp-parallel
or producer-warp copy issue, split / deferred staging) are experiment arms.

Registered entry points (``dispatch.get_impl("deferred_norm_evict_hints", name)``):

* ``deferred_hop``: :class:`EvictHintsDeferredHop` factory (same arguments and
  collective behaviour as deferred_norm's ``deferred_hop``);
* ``dsa_proj(mla, position_ids, packet, attn_metadata)``: the query_front
  packet of a target layer whose hidden state is a deferred_norm packet, through the
  deferred_norm_evict_hints projection, or ``None`` to fall back;
* ``is_deferred(hidden_states)``: deferred_norm's packet test.

The hop and the projection launch with ``EvictHintsHopConfig()`` and
``EvictHintsProjConfig()``.
"""

from ... import dispatch
from ..deferred_norm.packet import is_deferred
from .entry import evict_hints_dsa_proj
from .hop import EvictHintsDeferredHop, EvictHintsHopConfig
from .projection import EvictHintsProjConfig, EvictHintsProjection

dispatch.register("deferred_norm_evict_hints", "deferred_hop", EvictHintsDeferredHop)
dispatch.register("deferred_norm_evict_hints", "dsa_proj", evict_hints_dsa_proj)
dispatch.register("deferred_norm_evict_hints", "is_deferred", is_deferred)

__all__ = [
    "EvictHintsDeferredHop",
    "EvictHintsHopConfig",
    "EvictHintsProjConfig",
    "EvictHintsProjection",
    "is_deferred",
    "evict_hints_dsa_proj",
]
