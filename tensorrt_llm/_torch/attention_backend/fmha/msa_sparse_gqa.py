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

`MsaSparseGqaFmha` is an `Fmha` that wraps MSA's `fmha_sm100` paged
sparse GQA kernel and participates in the standard
`TrtllmAttention.forward` dispatch loop. It mirrors `DSATrtllmAttention`:

  * The MiniMax-M3 MSA attention backend `MiniMaxM3MSATrtllmAttention`
    subclasses `TrtllmAttention` and owns this FMHA plus an `MsaIndexer`.
    The indexer runs the proxy MQA and top-k block selection, then
    publishes the per-query selected block indices onto
    `forward_args.sparse_prediction` via `sparse_attn_predict`.
  * This class inherits `Fmha` directly rather than `PhasedFmha`. The
    indexer, plan, and selected block indices are built over the whole
    batch, and `fmha_sm100` handles mixed decode/prefill varlen batches
    in one call, so there is no context/generation phase split to reuse.
    `forward` does its own whole-batch dispatch: eager `fmha_sm100` for
    prefill/mixed batches, the CUDA-graph-safe in-tree driver for pure
    decode. The paged K/V views, page table, and per-request lengths come
    from the owning metadata via `MsaSparseMetadataProtocol`.

The kernel is SM100-only and `fmha_sm100` is an optional external
dependency (https://github.com/MiniMax-AI/MSA). When either is missing,
`is_available` returns False so the registry skips the class.

The prefill path funnels through `run_msa_sparse_gqa`, which is also
importable for focused unit tests that drive the kernel directly.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Optional, Protocol, Tuple, runtime_checkable

import torch

from tensorrt_llm.logger import logger

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


@runtime_checkable
class MsaSparseMetadataProtocol(Protocol):
    """Fields a metadata must expose to drive `MsaSparseGqaFmha`.

    Implemented by `MiniMaxM3MSATrtllmAttentionMetadata`. The FMHA reads
    the whole-batch q and output straight from `Fmha.forward`'s
    arguments; this protocol supplies the extra MSA-specific staging
    (paged K/V views, whole-batch lengths, main-KV and index-K writes,
    graph-safe decode) that does not fit `AttentionForwardArgs`.
    """

    def msa_get_paged_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(k_paged, v_paged)` in HND layout.

        Each is `[num_pages, num_kv_heads, page_size, head_dim]`.
        """
        ...

    def msa_write_main_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write the new-token K/V into this layer's paged main cache.

        `k` and `v` are `[num_tokens, num_kv_heads * head_dim]` (or the
        equivalent per-head views). The write targets the per-new-token
        slots staged in the metadata's `out_cache_loc`.
        """
        ...

    def msa_is_prefill(self) -> bool:
        """True when the step routes through the extend/prefill kernel.

        Mixed context+decode batches are prefill (decode rows appear as
        1-token causal extends); pure-decode batches are not.
        """
        ...

    def msa_whole_batch_lens(
        self,
        *,
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Return whole-batch `(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices)`.

        Covers every request in the step (context and decode) so the
        kernel runs once over the whole batch, matching the whole-batch
        block indices the indexer produces. `qo_lens_cpu` and `kv_lens_cpu`
        are `[batch]` int32 on CPU; `qo_offset_cpu` is the per-request
        causal prefix offset on CPU when `causal=True`, else None;
        `kv_indices` is the flattened paged-KV page table for the whole
        batch, int32 on the cache device.
        """
        ...

    def msa_run_sparse_decode(
        self,
        *,
        layer_idx: int,
        q: torch.Tensor,
        kv_block_indexes: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run the decode sparse GQA via the CUDA-graph-safe in-tree driver.

        Decode is CUDA-graph captured, so it must not go through the eager
        `fmha_sm100_plan` host driver, which uses unpinned H2D staging,
        per-call device allocations, and a device-side cost sweep. Returns
        `[num_tokens, num_q_heads * head_dim]`.
        """
        ...


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

    Consumes the per-query selected KV block indices on
    `forward_args.sparse_prediction.sparse_attn_indices` (produced by the
    MiniMax-M3 MSA indexer) and runs paged GQA attention over the selected
    blocks. Participates in the standard `TrtllmAttention.forward`
    dispatch loop; `is_supported` returns True only for M3 MSA sparse
    requests.

    Inherits `Fmha` directly, not `PhasedFmha`: the indexer, plan, and
    selected block indices are built over the whole batch, and
    `fmha_sm100` handles mixed decode/prefill varlen batches in one call,
    so there is no context/generation split to reuse. `forward` does its
    own whole-batch dispatch.

    Hard requirements (checked at runtime):
      * q, k, v head dim is 128, the only supported `fmha_sm100` variant.
      * paged K/V are 4-D HND caches with matching `num_kv_heads` and
        `page_size` (supplied by `MsaSparseMetadataProtocol`).
    """

    HEAD_DIM = 128
    REQUIRES_PAGED_KV = True

    def __init__(self, attn: Optional["TrtllmAttention"] = None):
        # The registry constructs this with an owning TrtllmAttention (for
        # layer_idx, num_heads, scale). Owner-less construction is
        # tolerated for focused unit tests that only exercise
        # is_available() or run_msa_sparse_gqa().
        #
        # kv_factor and the out-head sizes are set for parity with the
        # other FMHA libs. The whole-batch path does not read them, but
        # keeping them avoids surprising a future caller.
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
        # sparse_attn_predict, and the metadata must expose the MSA
        # staging protocol.
        sparse_prediction = getattr(forward_args, "sparse_prediction", None)
        if sparse_prediction is None:
            return False
        if getattr(sparse_prediction, "sparse_attn_indices", None) is None:
            return False
        return isinstance(metadata, MsaSparseMetadataProtocol)

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
        # Whole-batch dispatch, which is why this class inherits Fmha
        # rather than PhasedFmha. The indexer's selected block indices
        # carry one row per query token across the entire batch, and the
        # plan lengths span the whole batch. Splitting q by phase would
        # mismatch the kernel's total_q against q.shape[0] (fmha_sm100's
        # sparse schedule split_counts validation). fmha_sm100 handles
        # mixed decode/prefill varlen batches (decode rows are 1-token
        # causal extends), so a single whole-batch call is correct.
        if not isinstance(metadata, MsaSparseMetadataProtocol):
            raise RuntimeError(
                "MsaSparseGqaFmha requires metadata implementing "
                "MsaSparseMetadataProtocol (MiniMaxM3MSATrtllmAttentionMetadata)."
            )
        attn = self.attn
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")

        # fmha_sm100 reads the paged K/V cache directly, so (unlike the
        # standard C++ FMHA path) the new-token K/V must be written into
        # the cache here before the sparse GQA runs. The index-K write is
        # done by the indexer via sparse_attn_predict.
        if k is not None and v is not None:
            metadata.msa_write_main_kv(attn.layer_idx, k, v)

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

        if metadata.msa_is_prefill():
            # Context or mixed batch: eager fmha_sm100 (not CUDA-graph
            # captured).
            k_paged, v_paged = metadata.msa_get_paged_kv(attn.layer_idx)
            qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices = metadata.msa_whole_batch_lens(
                causal=True
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
            # Pure decode: CUDA-graph captured. Route the sparse GQA through
            # the in-tree graph-safe driver (device-tensor launch args,
            # device-side top-k) instead of the graph-hostile eager
            # fmha_sm100_plan path.
            out = metadata.msa_run_sparse_decode(
                layer_idx=attn.layer_idx,
                q=q3,
                kv_block_indexes=kv_block_indexes,
                sm_scale=sm_scale,
            )
        out_view.copy_(out.view_as(out_view))


__all__ = ["MsaSparseGqaFmha", "MsaSparseMetadataProtocol", "run_msa_sparse_gqa"]
