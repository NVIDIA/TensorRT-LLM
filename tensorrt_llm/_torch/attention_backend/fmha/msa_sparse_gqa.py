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

"""Block-sparse GQA FMHA backed by MSA's ``fmha_sm100`` kernel.

This module provides :class:`MsaSparseGqaFmha`, a
:class:`BlockSparseFmha` implementation that wraps MSA's
``fmha_sm100`` paged sparse GQA kernel. It consumes the
``kv_block_indexes`` produced by an upstream proxy + top-k pass (see
:class:`MsaProxyMqaFmha`) and runs the main attention on only the
selected KV blocks.

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

from .block_sparse import BlockSparseFmha


class MsaSparseGqaFmha(BlockSparseFmha):
    """SM100 block-sparse GQA FMHA powered by MSA's ``fmha_sm100`` kernel.

    Consumes ``kv_block_indexes`` (typically produced by
    :class:`MsaProxyMqaFmha` + ``sparse_topk_select`` upstream) and
    runs paged GQA attention over the selected blocks. Used by
    :mod:`tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend`
    for the MiniMax-M3 sparse layers' main attention pass.

    Hard requirements (checked at runtime in
    :meth:`forward_block_sparse`):
      * ``q``/``k``/``v`` head dim is 128 -- the only ``fmha_sm100``
        variant shipped today.
      * ``k_paged`` and ``v_paged`` are 4-D HND paged caches with
        matching ``num_kv_heads`` and ``page_size``.
    """

    HEAD_DIM = 128

    @classmethod
    def is_available(cls, attn=None) -> bool:
        # Probe with find_spec instead of importing — fmha_sm100's import
        # side effects (early tvm_ffi import + global-func registration)
        # intermittently corrupt the flashinfer dense-attention path.
        # See MsaProxyMqaFmha.is_available for the full story; the real
        # import happens at first kernel use in forward_block_sparse.
        if importlib.util.find_spec("fmha_sm100") is None:
            logger.debug("MsaSparseGqaFmha is unavailable: fmha_sm100 package not installed.")
            return False
        if not torch.cuda.is_available():
            logger.debug("MsaSparseGqaFmha is unavailable: no CUDA device.")
            return False
        try:
            major, _ = torch.cuda.get_device_capability()
        except Exception:
            return False
        if major != 10:
            logger.debug(
                "MsaSparseGqaFmha is unavailable: requires SM100 (compute capability 10.x), "
                f"got compute capability {major}.x."
            )
            return False
        return True

    def forward_block_sparse(
        self,
        q: torch.Tensor,
        k_paged: torch.Tensor,
        v_paged: torch.Tensor,
        kv_block_indexes: torch.Tensor,
        *,
        qo_lens_cpu: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        qo_offset_cpu: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        """Run block-sparse paged GQA and return ``[total_q, num_qo_heads, head_dim]``.

        Follows MSA's two-call pattern: ``fmha_sm100_plan`` builds the
        per-shape sparse plan (with ``kv_block_num`` derived from
        ``kv_block_indexes.shape[-1]``) and ``fmha_sm100`` runs the
        kernel with the block indices threaded through.
        """
        # Imported here (not at module top) so the registry can still
        # advertise the class on hosts where fmha_sm100 is absent --
        # is_available() handles the off-host case.
        import fmha_sm100

        if q.dim() != 3:
            raise ValueError(
                "MsaSparseGqaFmha expects q with shape [total_q, num_qo_heads, head_dim]; "
                f"got {tuple(q.shape)}."
            )
        if q.shape[-1] != self.HEAD_DIM:
            raise NotImplementedError(
                f"MsaSparseGqaFmha currently supports head_dim={self.HEAD_DIM}; got {q.shape[-1]}."
            )
        if k_paged.dim() != 4 or v_paged.dim() != 4:
            raise ValueError(
                "MsaSparseGqaFmha expects paged KV with shape "
                "[num_pages, num_kv_heads, page_size, head_dim]; "
                f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
            )
        if k_paged.shape != v_paged.shape:
            raise ValueError(
                f"MsaSparseGqaFmha requires k and v to share shape; "
                f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
            )
        if k_paged.shape[-1] != self.HEAD_DIM:
            raise NotImplementedError(
                f"MsaSparseGqaFmha currently supports head_dim={self.HEAD_DIM}; "
                f"got k_paged head_dim={k_paged.shape[-1]}."
            )

        num_qo_heads = int(q.shape[1])
        num_kv_heads = int(k_paged.shape[1])
        page_size = int(k_paged.shape[2])

        sparse_plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            num_qo_heads,
            num_kv_heads=num_kv_heads,
            qo_offset=qo_offset_cpu,
            page_size=page_size,
            kv_block_num=int(kv_block_indexes.shape[-1]),
            causal=causal,
            num_kv_splits=1,
        )
        out, _ = fmha_sm100.fmha_sm100(
            q,
            k_paged,
            v_paged,
            sparse_plan,
            kv_indices=kv_indices,
            kv_block_indexes=kv_block_indexes,
            sm_scale=sm_scale,
            output_maxscore=False,
        )
        return out


__all__ = ["MsaSparseGqaFmha"]
