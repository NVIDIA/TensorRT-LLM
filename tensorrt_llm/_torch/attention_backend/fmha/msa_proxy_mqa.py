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

"""Proxy MQA FMHA backed by MSA's ``fmha_sm100`` kernel.

This module provides :class:`MsaProxyMqaFmha`, an
:class:`IndexerProxyFmha` implementation that wraps MSA's
``fmha_sm100`` dense FMHA in ``output_maxscore`` mode for use by
sparse-attention indexers (currently the MiniMax-M3 indexer).

The kernel is SM100-only and the ``fmha_sm100`` Python package is an
optional external dependency (https://github.com/MiniMax-AI/MSA). On
hosts where either precondition is missing, :meth:`is_available`
returns ``False`` so the registry skips the class.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

import torch

from tensorrt_llm.logger import logger

from .indexer_proxy import IndexerProxyFmha


class MsaProxyMqaFmha(IndexerProxyFmha):
    """SM100 MQA proxy FMHA powered by MSA's ``fmha_sm100`` kernel.

    The MiniMax-M3 sparse-attention indexer (and, in the future,
    other sparse predictors that need to score every KV block against
    an MQA query) selects this backend when ``fmha_sm100`` is
    importable and the current device exposes compute capability
    10 (SM100 family).

    Hard requirements (checked at runtime in :meth:`forward_proxy`):
      * ``idx_q`` head dim is 128 -- the only ``fmha_sm100`` variant
        shipped today.
      * ``idx_k_paged`` is 4-D HND with ``num_kv_heads == 1`` (MQA).
    """

    HEAD_DIM = 128

    @classmethod
    def is_available(cls, attn=None) -> bool:
        # Probe with find_spec instead of importing. fmha_sm100's import
        # has module-level side effects: it imports tvm_ffi — the FFI
        # runtime flashinfer owns in this process — and registers global
        # functions into it. is_available() runs at attention-layer
        # construction for every layer, and pulling tvm_ffi up that
        # early (outside flashinfer's own initialization order)
        # intermittently corrupts the flashinfer dense-attention path
        # (~1/3 of processes emit garbage logits mid-decode). The real
        # import is deferred to first kernel use, by which point
        # flashinfer has initialized tvm_ffi itself.
        if importlib.util.find_spec("fmha_sm100") is None:
            logger.debug("MsaProxyMqaFmha is unavailable: fmha_sm100 package not installed.")
            return False
        if not torch.cuda.is_available():
            logger.debug("MsaProxyMqaFmha is unavailable: no CUDA device.")
            return False
        try:
            major, _ = torch.cuda.get_device_capability()
        except Exception:
            return False
        if major != 10:
            logger.debug(
                "MsaProxyMqaFmha is unavailable: requires SM100 (compute capability 10.x), "
                f"got compute capability {major}.x."
            )
            return False
        return True

    def forward_proxy(
        self,
        idx_q: torch.Tensor,
        idx_k_paged: torch.Tensor,
        *,
        qo_lens_cpu: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        qo_offset_cpu: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        """Run the MQA proxy FMHA and return ``[num_qo_heads, max_k_tiles, total_q]``.

        The implementation follows MSA's documented two-call pattern:
        ``fmha_sm100_plan`` builds the per-shape plan (CUDA-graph-stable
        worklists, KV-split workspaces) and ``fmha_sm100`` runs the
        kernel with ``output_o=False, output_maxscore=True`` so only
        the score tensor is materialized.

        This wrapper serves the eager prefill path only.  Decode runs
        through the in-tree graph-safe driver
        (``sparse.minimax_m3.decode_wrapper``) because
        ``fmha_sm100_plan`` is CUDA-graph-hostile (unpinned H2D
        staging, per-call device allocations, device-side cost sweep
        with ``.tolist()``).
        """
        # Imported here (not at module top) so the registry can still
        # advertise the class on hosts where fmha_sm100 is absent --
        # is_available() handles the off-host case.
        import fmha_sm100

        if idx_q.dim() != 3:
            raise ValueError(
                "MsaProxyMqaFmha expects idx_q with shape [total_q, num_qo_heads, head_dim]; "
                f"got {tuple(idx_q.shape)}."
            )
        if idx_q.shape[-1] != self.HEAD_DIM:
            raise NotImplementedError(
                f"MsaProxyMqaFmha currently supports head_dim={self.HEAD_DIM}; "
                f"got {idx_q.shape[-1]}."
            )
        if idx_k_paged.dim() != 4 or idx_k_paged.shape[1] != 1:
            raise ValueError(
                "MsaProxyMqaFmha expects MQA paged KV "
                "[num_pages, 1, page_size, head_dim]; "
                f"got {tuple(idx_k_paged.shape)}."
            )
        if idx_k_paged.shape[-1] != self.HEAD_DIM:
            raise NotImplementedError(
                f"MsaProxyMqaFmha currently supports head_dim={self.HEAD_DIM}; "
                f"got idx_k_paged head_dim={idx_k_paged.shape[-1]}."
            )

        page_size = int(idx_k_paged.shape[2])
        proxy_plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            idx_q.shape[1],  # num_qo_heads (= num_index_heads for M3)
            num_kv_heads=1,
            qo_offset=qo_offset_cpu,
            page_size=page_size,
            output_maxscore=True,
            causal=causal,
            num_kv_splits=1,
        )
        _, max_score = fmha_sm100.fmha_sm100(
            idx_q,
            idx_k_paged,
            idx_k_paged,  # v passthrough -- proxy ignores V via output_o=False
            proxy_plan,
            kv_indices=kv_indices,
            output_o=False,
            output_maxscore=True,
            sm_scale=sm_scale,
        )
        return max_score


__all__ = ["MsaProxyMqaFmha"]
