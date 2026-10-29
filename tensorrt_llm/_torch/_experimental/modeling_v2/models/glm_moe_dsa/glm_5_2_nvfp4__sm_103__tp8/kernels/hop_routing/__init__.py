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
"""Package hop_routing: native top-8 routing in the pre-MoE into-hop entry.

Entry points registered with ``dispatch``:

* ``make_transport(into_hop)`` returns the :class:`~.hop.HopRoutingTransport` of an
  ``attention_output_into_hop`` transport (or None, the into-hop path). Call it at the same point
  of model construction on every rank.
* ``engagement_counts()`` returns the host-side routing-hop call counters.
* ``routed_output_into_hop(module, routing_bias, x, router_logits, routing,
  hop)`` and ``routed_output(module, routing_bias, x, router_logits, routing,
  do_finalize)``: routed_moe's entries of the same names with the hop's routing
  (``routing``, int32 ``[2, R, 8]``) in place of routed_moe's own; ``None`` means
  "run routed_moe on ``router_logits``" (decided before any device work).
"""

from ...dispatch import register
from .hop import HopRoutingTransport, engagement_counts, op_available, split_routing
from .routed_moe import routed_output, routed_output_into_hop

register("hop_routing", "make_transport", HopRoutingTransport.create)
register("hop_routing", "engagement_counts", engagement_counts)
register("hop_routing", "routed_output", routed_output)
register("hop_routing", "routed_output_into_hop", routed_output_into_hop)

__all__ = [
    "HopRoutingTransport",
    "engagement_counts",
    "op_available",
    "routed_output",
    "routed_output_into_hop",
    "split_routing",
]
