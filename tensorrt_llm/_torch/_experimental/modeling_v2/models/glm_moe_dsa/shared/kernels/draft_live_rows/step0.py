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
"""Draft-loop side of package draft_live_rows: arm the MTP layer's live-row hop at step 0.

The target's one-model MTP draft loop (``GlmMtpWorker._forward_linear_draft_loop``)
asks :func:`select_step0` once per forward, before any device work, whether
draft step 0 may compact the MTP layer to its live rows. The answer is
rank-uniform: it depends on the selection, on draft_loop's draft-loop ops existing
for this forward (generation-only batch inside draft_loop's domain), on nothing
capturing the MTP layer's full output (``layers_to_capture`` empty) and on
the MTP layer owning a live-row hop. If it returns a :class:`Step0LiveRows`,
the loop runs step 0 inside :meth:`Step0LiveRows.armed`; when the MTP layer
took the live-row hop (``compacted``), the layer's output already holds one
row per request in ``gather_ids`` order, so draft_loop's step-0 row gather becomes the
identity.
"""

import contextlib
from typing import Any, Optional

import torch

from ...dispatch import log_capability
from .hop import _count

FEATURE = "draft_live_rows"


def step0_ineligibility(
    worker: Any, draft_model: Any, spec_metadata: Any, draft_loop_ops: Any
) -> Optional[str]:
    """Reason draft step 0 must keep every row after the MTP layer, or None (host only)."""
    if not getattr(worker, "is_mtp_eagle", False):
        return "not MTP Eagle one-model"
    if draft_loop_ops is None:
        return "no draft_loop draft-loop ops"
    if len(getattr(spec_metadata, "layers_to_capture", None) or ()) != 0:
        return "MTP layer output captured"
    layers = getattr(draft_model, "mtp_layers", None)
    if not layers:
        return "no MTP layer"
    if getattr(layers[0], "draft_live_row_hop", None) is None:
        return "MTP layer has no live-row hop"
    gather_ids = getattr(draft_loop_ops, "gather_ids", None)
    if (
        gather_ids is None
        or gather_ids.dtype != torch.int64
        or gather_ids.dim() != 1
        or gather_ids.shape[0] != draft_loop_ops.batch_size
    ):
        return "draft_loop gather ids"
    return None


class Step0LiveRows:
    """Arms the MTP layer with draft_loop's live-row ids for one draft step-0 call.

    The layer takes the ids through ``arm_live_rows`` and reports the outcome through
    ``disarm_live_rows``.
    """

    def __init__(self, mtp_layer: Any, gather_ids: torch.Tensor):
        self.mtp_layer = mtp_layer
        self.gather_ids = gather_ids
        self.compacted = False

    @contextlib.contextmanager
    def armed(self):
        """Run the step-0 MTP call inside this context; ``compacted`` tells the outcome."""
        layer = self.mtp_layer
        layer.arm_live_rows(self.gather_ids)
        self.compacted = False
        _count("step0_armed")
        try:
            yield self
        finally:
            compacted = layer.disarm_live_rows()
        self.compacted = compacted
        if compacted:
            _count("step0_compacted")


def select_step0(
    worker: Any, draft_model: Any, spec_metadata: Any, draft_loop_ops: Any
) -> Optional[Step0LiveRows]:
    """The step-0 arming of this forward, or None to keep every row (native draft_loop gather)."""
    reason = step0_ineligibility(worker, draft_model, spec_metadata, draft_loop_ops)
    log_capability(FEATURE, f"step0:{reason or 'engaged'}", reason is None, reason or "")
    if reason is not None:
        return None
    return Step0LiveRows(draft_model.mtp_layers[0], draft_loop_ops.gather_ids)
