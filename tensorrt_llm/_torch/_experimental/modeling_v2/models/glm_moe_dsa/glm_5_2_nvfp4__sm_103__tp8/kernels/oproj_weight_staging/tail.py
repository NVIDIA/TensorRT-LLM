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
"""oproj_weight_staging ``oproj_weight_staging`` adapter: the attention_output_tail attention tail
with staged o_proj rows.

``wrap_attn_tail(mla, tail)`` is called by the MLA adapter right after attention_output_tail's
``make_attn_tail`` when ``oproj_weight_staging`` is selected. It returns a ``StagedAttnTail``
that carries the same tail schedule (``TailConfig``) and takes every
per-call decision exactly as attention_output_tail's ``AttnTail`` (``arm``, ``defer_w_uv``, the
attention_output_into_hop request), so a declined call runs the native path with no
oproj_weight_staging work issued. Only the launch in ``project`` differs: ``kernels.stage_tail``
instead of ``attention_output_tail.kernels.attn_tail``, with the same arithmetic.

The decisions are static (logged once per rank and layer): at construction
oproj_weight_staging takes only one schedule, attention_output_tail's
``INTO_HOP_SPARSE_DECODE_SCHEDULE`` (128 CTAs, 32 early rows: the served tail behind
sparse_mla_decode with the attention_output_into_hop publish; otherwise attention_output_tail's
tail is returned unchanged); on the first ``arm``, together with attention_output_tail's own static
decision and before any device work, it checks the staging geometry on the layer's weights
(otherwise the layer runs attention_output_tail's ``project``).
"""

from typing import Optional

import torch

from ... import dispatch
from ..attention_output_tail.kernels import INTO_HOP_SPARSE_DECODE_SCHEDULE
from ..attention_output_tail.tail import MAX_INTO_HOP_ROWS, AttnTail
from .kernels import (
    DEFAULT_STAGE,
    StageConfig,
    check_stage_config,
    check_tail_config,
    stage_tail,
    wuv_stage_geometry,
)

FEATURE_ID = "oproj_weight_staging"


class StagedAttnTail(AttnTail):
    """attention_output_tail whose o_proj stages its after-wait ``w_o`` rows (oproj_weight_staging).

    Args:
        layer_key: name used in capability logs, e.g. ``"layer3"``.
        config: the attention_output_tail schedule (served: ``INTO_HOP_SPARSE_DECODE_SCHEDULE``).
        stage: the oproj_weight_staging configuration.
    """

    def __init__(self, layer_key: str, config, stage: StageConfig):
        super().__init__(layer_key, config)
        self.stage = stage
        self.staging_calls = 0
        self.staging_reason: Optional[str] = None
        """Why this layer's o_proj is not staged (``None`` if it is); set
        with attention_output_tail's static decision on the first ``arm``."""
        self._staging_ok: Optional[bool] = None
        self.trace: Optional[torch.Tensor] = None
        """Diagnostics only (benchmarks): int64 ``[grid * TRACE_POINTS]``
        buffer for the o_proj's ``%globaltimer`` stamps of every call."""

    def _check_static(self, mla) -> bool:
        ok = super()._check_static(mla)
        if self._staging_ok is None:
            reason = None if ok else "attention_output_tail tail not engaged on this layer"
            if reason is None:
                reason = geometry_reason(mla, self.config, self.stage)
            self.staging_reason = reason
            self._staging_ok = dispatch.log_capability(
                FEATURE_ID,
                self.layer_key,
                reason is None,
                reason
                or (
                    f"attention_output_tail tail o_proj (early_rows={self.config.early_rows}, "
                    f"grid={self.config.oproj_grid or 'SMs'}) with "
                    f"{self.stage}"
                ),
            )
        return ok

    def project(
        self, mla, attn_output: torch.Tensor, all_reduce_params=None
    ) -> Optional[torch.Tensor]:
        """As ``AttnTail.project`` for a ``cute_stream`` schedule without the
        attention_output_tail_prefetch, launching ``stage_tail``. Returns ``None`` (native
        projection) when ``defer_w_uv`` declined the call; runs attention_output_tail's own
        ``project`` when the staging geometry does not fit this layer."""
        if not self._staging_ok:
            return super().project(mla, attn_output, all_reduce_params)
        del all_reduce_params
        self._armed = False
        packet = self._pending
        if packet is None:
            self._join_prefetch()
            return None
        self._pending = None
        if attn_output.data_ptr() != packet.output_ptr or attn_output.shape[0] < packet.num_rows:
            self._join_prefetch()
            raise RuntimeError(
                f"[glm_moe_dsa] oproj_weight_staging/{self.layer_key}: tail packet does not match "
                "the output projection input"
            )
        hop_transport = None
        request, self._into_hop = self._into_hop, None
        if request is not None:
            transport, hop_rows = request
            # The hop polls hop_rows rows and later resets exactly those rows
            # of the buffer, so the producer must publish exactly them.
            if attn_output.shape[0] == hop_rows <= MAX_INTO_HOP_ROWS:
                hop_transport = transport
            else:
                self.into_hop_declined_calls += 1
        partial = stage_tail(
            packet.latent,
            mla.v_b_proj,
            mla.o_proj.weight,
            packet.num_rows,
            out_rows=attn_output.shape[0],
            config=self.config,
            stage=self.stage,
            hop_transport=hop_transport,
            trace=self.trace,
        )
        if hop_transport is not None:
            self._into_hop_published = True
            self.into_hop_published_calls += 1
        self.staging_calls += 1
        self._join_prefetch()
        return partial


def config_reason(config, stage: StageConfig) -> Optional[str]:
    """Static reason (tail schedule and staging configuration only) why oproj_weight_staging
    cannot wrap a tail with ``config``, or ``None``."""
    reason = check_stage_config(stage)
    if reason is not None:
        return reason
    if config.oproj_engine != "cute_stream":
        return f"o_proj engine {config.oproj_engine!r} is not cute_stream"
    if config.prefetch_l2 or config.successor_prefetch:
        return "attention_output_tail_prefetch off-path prefetch configs keep the attention_output_tail tail"
    if config.trigger_late:
        return "trigger_late is an experiment of the attention_output_tail tail"
    if config != INTO_HOP_SPARSE_DECODE_SCHEDULE:
        return (
            "oproj_weight_staging is measured only on the served attention_output_into_hop/sparse_mla_decode schedule "
            "(128 CTAs, 32 early rows)"
        )
    return None


def geometry_reason(mla, config, stage: StageConfig) -> Optional[str]:
    """Reason from the module's weights (first ``arm``) why the staging
    o_proj cannot run this layer, or ``None``."""
    w_o = mla.o_proj.weight
    n_dim, k_dim = w_o.shape
    sm_count = torch.cuda.get_device_properties(w_o.device).multi_processor_count
    reason = check_tail_config(config, n_dim, k_dim, sm_count)
    if reason is not None:
        return reason
    if stage.host == "wuv" and stage.stage == "l2":
        grid = config.oproj_grid or sm_count
        heads, v_dim, _ = mla.v_b_proj.shape
        if heads * (v_dim // config.wuv_block_v) < grid:
            return "W_UV grid smaller than the o_proj grid"
        early = config.early_rows if stage.early_rows is None else stage.early_rows
        if wuv_stage_geometry(n_dim, k_dim, grid, early, stage.rows) is None:
            return "W_UV host cannot cover the o_proj's after-wait rows"
    return None


def wrap_attn_tail(mla, tail: Optional[AttnTail], stage: Optional[StageConfig] = None):
    """Return a ``StagedAttnTail`` replacing attention_output_tail's ``tail``, or ``tail`` itself.

    Called once per MLA module at construction, after attention_output_tail's
    ``make_attn_tail``; ``None`` (no attention_output_tail on this layer) stays ``None``.
    ``stage`` defaults to ``DEFAULT_STAGE``. The schedule-level
    decision is made here (logged when it declines); the weight-geometry
    decision is made with attention_output_tail's static decision on the first ``arm``, before any
    device work.
    """
    del mla
    if tail is None:
        return None
    if stage is None:
        stage = DEFAULT_STAGE
    reason = config_reason(tail.config, stage)
    if reason is not None:
        dispatch.log_capability(FEATURE_ID, tail.layer_key, False, reason)
        return tail
    return StagedAttnTail(tail.layer_key, tail.config, stage)


__all__ = ["FEATURE_ID", "StagedAttnTail", "config_reason", "geometry_reason", "wrap_attn_tail"]
