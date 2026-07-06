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

"""Block-sparse GQA FMHA backed by MSA's `fmha_sm100` kernel.

`MsaSparseGqaFmha` wraps MSA's `fmha_sm100` paged sparse GQA kernel and
participates in the standard `TrtllmAttention.forward` dispatch loop. The
owning `MiniMaxM3MSATrtllmAttention` layer runs an `MsaIndexer` to select
the per-query KV blocks and publishes them on
`forward_args.sparse_prediction`; this class attends over them.

The kernel is SM100-only and `fmha_sm100` is an optional external
dependency (https://github.com/MiniMax-AI/MSA); `is_available` returns
False when it or an SM100 device is missing. `run_msa_sparse_gqa` is
importable for focused unit tests that drive the kernel directly.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm.logger import logger

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def _msa_metadata_cls() -> type:
    """The concrete metadata class that drives `MsaSparseGqaFmha`.

    Resolved lazily (the class is built inside a factory with a deferred
    `trtllm` import) so importing this module during attention-backend
    package init does not form an import cycle.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        get_minimax_m3_msa_attention_backend_cls,
    )

    return get_minimax_m3_msa_attention_backend_cls().Metadata


def run_msa_sparse_gqa(
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
    head_dim: int = 128,
) -> torch.Tensor:
    """Single kernel core: `fmha_sm100` block-sparse paged GQA.

    Follows MSA's two-call pattern: `fmha_sm100_plan` builds the per-shape
    sparse plan (with `kv_block_num` derived from
    `kv_block_indexes.shape[-1]`), then `fmha_sm100` runs the kernel with
    the block indices threaded through. Returns
    `[total_q, num_qo_heads, head_dim]` bfloat16.
    """
    # Imported here, not at module top, so the registry can still
    # advertise the class on hosts where fmha_sm100 is absent.
    # is_available() handles the off-host case.
    import fmha_sm100

    if q.dim() != 3:
        raise ValueError(
            "MsaSparseGqaFmha expects q with shape [total_q, num_qo_heads, head_dim]; "
            f"got {tuple(q.shape)}."
        )
    if q.shape[-1] != head_dim:
        raise NotImplementedError(
            f"MsaSparseGqaFmha currently supports head_dim={head_dim}; got {q.shape[-1]}."
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
    if k_paged.shape[-1] != head_dim:
        raise NotImplementedError(
            f"MsaSparseGqaFmha currently supports head_dim={head_dim}; "
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


def _sm100_fmha_available(class_name: str) -> bool:
    """Shared availability probe for the MSA `fmha_sm100` backend.

    Probes with `importlib.util.find_spec` instead of importing:
    `fmha_sm100`'s import side effects (an early `tvm_ffi` import plus
    global-func registration) can corrupt the flashinfer dense-attention
    path when pulled in at layer-construction time. The real import
    happens at first kernel use.
    """
    if importlib.util.find_spec("fmha_sm100") is None:
        logger.debug(f"{class_name} is unavailable: fmha_sm100 package not installed.")
        return False
    if not torch.cuda.is_available():
        logger.debug(f"{class_name} is unavailable: no CUDA device.")
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
    except RuntimeError:
        return False
    if major != 10:
        logger.debug(
            f"{class_name} is unavailable: requires SM100 (compute capability 10.x), "
            f"got compute capability {major}.x."
        )
        return False
    return True


class MsaSparseGqaFmha(Fmha):
    """SM100 block-sparse GQA FMHA powered by MSA's `fmha_sm100` kernel.

    Consumes the indexer's selected KV block indices on
    `forward_args.sparse_prediction.sparse_attn_indices` and runs paged GQA
    over them; `is_supported` claims only M3 MSA sparse requests. Inherits
    `Fmha` rather than `PhasedFmha` because the indexer, plan, and block
    indices span the whole batch and `fmha_sm100` handles mixed
    decode/prefill varlen batches in one call, so there is no
    context/generation split to reuse and `forward` dispatches the whole
    batch at once. Requires head_dim 128 and 4-D HND paged K/V.
    """

    HEAD_DIM = 128
    REQUIRES_PAGED_KV = True

    def __init__(self, attn: "TrtllmAttention"):
        # kv_factor and the out-head sizes mirror the other FMHA libs for
        # parity; the whole-batch path does not read them.
        super().__init__(attn)
        self.kv_factor = 2
        self.generation_out_head_size = self.HEAD_DIM
        self.context_out_head_size = self.HEAD_DIM

    @classmethod
    def is_available(cls, attn: Optional["TrtllmAttention"] = None) -> bool:
        return _sm100_fmha_available(cls.__name__)

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> bool:
        # Only claim MiniMax-M3 MSA sparse requests: the indexer must have
        # populated the per-query selected block indices via
        # sparse_attn_predict, and the metadata must be the M3 MSA
        # metadata that carries the pre-staged plans.
        sparse_prediction = getattr(forward_args, "sparse_prediction", None)
        if sparse_prediction is None:
            return False
        if getattr(sparse_prediction, "sparse_attn_indices", None) is None:
            return False
        return isinstance(metadata, _msa_metadata_cls())

    def _sm_scale(self) -> float:
        attn = self.attn
        q_scaling = getattr(attn, "q_scaling", 1.0) or 1.0
        return (self.HEAD_DIM**-0.5) / float(q_scaling)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> None:
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import (
            msa_paged_kv,
            write_msa_main_kv,
        )
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
            _whole_batch_lens,
            run_msa_sparse_decode,
        )

        attn = self.attn
        config = attn.m3_config
        layer_idx = attn.layer_idx
        kv_cache_manager = metadata.kv_cache_manager
        m3_meta = metadata.m3_meta
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")

        # fmha_sm100 reads the paged K/V cache directly, so (unlike the
        # standard C++ FMHA path) the new-token K/V must be written into
        # the cache here before the sparse GQA runs. The index-K write is
        # done by the indexer via sparse_attn_predict.
        if k is not None and v is not None:
            write_msa_main_kv(kv_cache_manager, layer_idx, metadata.m3_out_cache_loc, k, v)

        sparse_prediction = getattr(forward_args, "sparse_prediction", None)
        kv_block_indexes = (
            getattr(sparse_prediction, "sparse_attn_indices", None)
            if sparse_prediction is not None
            else None
        )
        if kv_block_indexes is None:
            raise RuntimeError(
                "MsaSparseGqaFmha invoked without sparse_attn_indices; "
                "MiniMaxM3MSATrtllmAttention.sparse_attn_predict must populate them."
            )

        num_tokens = int(q.shape[0])
        q3 = q.view(num_tokens, attn.num_heads, self.HEAD_DIM)
        out_view = output.view(num_tokens, attn.num_heads, self.HEAD_DIM)
        sm_scale = self._sm_scale()

        if m3_meta.is_prefill:
            # Context or mixed batch: eager fmha_sm100 (not graph captured).
            k_paged, v_paged = msa_paged_kv(kv_cache_manager, layer_idx)
            qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices = _whole_batch_lens(
                m3_meta, config.block_size
            )
            out = run_msa_sparse_gqa(
                q3,
                k_paged,
                v_paged,
                kv_block_indexes,
                qo_lens_cpu=qo_lens_cpu,
                kv_lens_cpu=kv_lens_cpu,
                qo_offset_cpu=qo_offset_cpu,
                kv_indices=kv_indices,
                sm_scale=sm_scale,
                causal=True,
                head_dim=self.HEAD_DIM,
            )
        else:
            # Pure decode: CUDA-graph captured, so route through the
            # graph-safe in-tree decode kernels rather than the
            # graph-hostile eager fmha_sm100_plan path.
            out = run_msa_sparse_decode(
                config,
                kv_cache_manager,
                layer_idx,
                m3_meta,
                q3,
                kv_block_indexes,
                sm_scale,
            )
        out_view.copy_(out.view_as(out_view))


__all__ = ["MsaSparseGqaFmha", "run_msa_sparse_gqa"]
