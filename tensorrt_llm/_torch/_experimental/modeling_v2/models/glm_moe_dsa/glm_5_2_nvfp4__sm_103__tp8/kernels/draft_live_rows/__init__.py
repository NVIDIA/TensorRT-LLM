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
"""Package draft_live_rows: MTP draft step-0 live-row compaction at the post-attention hop.

Entry points registered with ``dispatch``:

* ``make_live_row_hop(post_attn_hop)`` returns the :class:`LiveRowHop` on the
  workspace of the MTP layer's post_attention_hop, or None (the layer keeps the post_attention_hop).
  Call it at the same point of model construction on every rank: the first
  call for a post_attention_hop agrees over its TP communicator whether every rank
  carries the operator, so the answer is the same on all ranks.
* ``select_step0(worker, draft_model, spec_metadata, draft_loop_ops)`` returns the
  draft step-0 arming (:class:`Step0LiveRows`) of one one-model MTP forward,
  or None to keep every row; decided on the host before any device work.
* ``engagement_counts()`` returns the host-side call counters (eager calls
  and CUDA graph captures).

The package imports no other package's code.
"""

from ...dispatch import register
from .hop import LiveRowHop, engagement_counts, op_available, required_sms, reset_engagement_counts
from .step0 import Step0LiveRows, select_step0, step0_ineligibility

register("draft_live_rows", "make_live_row_hop", LiveRowHop.for_post_attn_hop)
register("draft_live_rows", "select_step0", select_step0)
register("draft_live_rows", "engagement_counts", engagement_counts)

__all__ = [
    "LiveRowHop",
    "Step0LiveRows",
    "engagement_counts",
    "op_available",
    "required_sms",
    "reset_engagement_counts",
    "select_step0",
    "step0_ineligibility",
]
