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
"""Package post_attention_hop: fused post-attention hop (cross-rank sum + residual + RMSNorm + router).

Entry points registered with ``dispatch``:

* ``make_post_attn_hop(mapping, hidden_size, dtype, native_allreduce=None)``
  returns the TP group's :class:`PostAttnHop` (or None for the native path)
  for a call site whose native ``AllReduce`` is ``native_allreduce``. Call it at
  the same point of model construction on every rank.
* ``engagement_counts()`` returns the host-side engaged/native-fallback call
  counters (eager calls and CUDA graph captures).
* ``make_into_hop_transport(hop)`` (``attention_output_into_hop`` only) returns the into-hop
  transport of a hop (see ``from_partials.py``), or None.
* ``into_hop_engagement_counts()`` (``attention_output_into_hop`` only) returns the
  host-side into-hop call counters; an into-hop call replaces a post_attention_hop call.
"""

from ...dispatch import register
from .hop import (
    PostAttnHop,
    engagement_counts,
    native_allreduce_ineligibility,
    required_sms,
    reset_engagement_counts,
    static_ineligibility,
)

register("post_attention_hop", "make_post_attn_hop", PostAttnHop.get_or_create)
register("post_attention_hop", "engagement_counts", engagement_counts)


def _make_into_hop_transport(hop):
    """``attention_output_into_hop`` only: the into-hop transport of ``hop`` (imports ``from_partials`` lazily, so
    the post_attention_hop itself is unaffected)."""
    from .from_partials import IntoHopTransport

    return IntoHopTransport.create(hop)


register("post_attention_hop", "make_into_hop_transport", _make_into_hop_transport)


def _into_hop_engagement_counts():
    """``attention_output_into_hop`` only: the into-hop call counters (imports ``from_partials`` lazily)."""
    from .from_partials import into_hop_engagement_counts

    return into_hop_engagement_counts()


register("post_attention_hop", "into_hop_engagement_counts", _into_hop_engagement_counts)

__all__ = [
    "PostAttnHop",
    "engagement_counts",
    "native_allreduce_ineligibility",
    "required_sms",
    "reset_engagement_counts",
    "static_ineligibility",
]
