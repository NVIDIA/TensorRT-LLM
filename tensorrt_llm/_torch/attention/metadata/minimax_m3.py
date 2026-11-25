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
"""MiniMax-M3 sparse attention: the state of its Triton and MSA implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Type

import torch

from ..backends.interface import AttentionMetadata
from ..backends.sparse.minimax_m3.common import MiniMaxM3SparseMetadataParams
from ..backends.sparse.minimax_m3.msa_backend import MiniMaxM3MsaSparseAttentionMetadata
from ..backends.sparse.minimax_m3.triton_metadata import (
    MiniMaxM3AttentionMetadata,
    build_runtime_metadata_from_kv_manager,
    derive_decode_cache_slots,
    derive_q_positions_and_cache_slots,
)
from .base import Metadata, _Registry
from .core import TrtllmCoreMetadata
from .sparse import SparseMetadata
from .spec_decoding import SpecDecodingMetadata

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs


class MiniMaxM3TritonMetadata(Metadata):
    """What the Triton implementation of MiniMax-M3 sparse attention reads.

    minimax_m3 holds the step's sparse metadata and the cache slots of its new
    tokens, built from the core's lengths on the device of the KV cache. Its
    logic is MiniMaxM3AttentionMetadata's, taken over method by method.
    """

    requires = (TrtllmCoreMetadata, SparseMetadata)

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        # {"metadata": ..., "out_cache_loc": ...} of the step, or None before the first one.
        self.minimax_m3 = None
        # Device buffers that keep the addresses a CUDA graph captured.
        self._m3_static_buffers = None

    def prepare(self, inputs: "BatchInputs") -> None:
        if inputs.stage_host:
            self._build_attachment()
        self._refresh_attachment()

    def _build_attachment(self) -> None:
        """Build the step's M3 metadata and cache slots from the staged lengths."""
        # Always rebuild the M3 metadata block on each prepare()
        # call so it reflects the current scheduler step's seq_lens
        # / request_ids / num_cached_tokens. Production
        # ``model_engine`` invokes ``prepare()`` outside any CUDA
        # graph capture window, so the (potentially expensive) build
        # is safe to perform here.
        #
        # When CUDA graph is enabled the inner ``build_runtime_metadata_from_kv_manager``
        # call writes into the persistent ``_m3_static_buffers`` so
        # the captured graph keeps reading from stable ``data_ptr()``s
        # across replays. Without this the captured ``index_select``
        # over ``req_to_token``/``slot_ids`` reads from freed warmup
        # memory and either produces wrong tokens or fires
        # ``Indexing.cu:1515`` ``srcIndex < srcSelectDimSize``.
        self.minimax_m3 = None

        # Production path: build the M3 metadata from the standard
        # AttentionMetadata fields. Requires kv_cache_manager + the
        # M3 sparse-cache contract.
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        if kv_cache_manager is None or not hasattr(kv_cache_manager, "get_index_k_buffer"):
            # Not an M3 KV cache manager: nothing to build. The
            # forward path will raise a clear error if the M3
            # backend ends up dispatched without the M3 cache.
            return
        request_ids = getattr(self, "request_ids", None)
        seq_lens = self.seq_lens
        if request_ids is None or seq_lens is None:
            return
        num_contexts = int(getattr(self, "num_contexts", 0) or 0)
        batch_size = int(seq_lens.shape[0])
        if batch_size == 0:
            return

        # The cache device hosts every paged buffer; this is the
        # device the forward path consumes.
        try:
            layer_buf = kv_cache_manager.get_buffers(0)
            cache_device = layer_buf.device
        except Exception:
            cache_device = torch.device(f"cuda:{torch.cuda.current_device()}")

        seq_lens_cpu = (
            getattr(self, "seq_lens_cpu", None)
            if hasattr(self, "seq_lens_cpu")
            else seq_lens.detach().to("cpu")
        )
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.detach().to("cpu")

        kv_cache_params = getattr(self, "kv_cache_params", None)
        num_cached_per_seq = (
            kv_cache_params.num_cached_tokens_per_seq
            if kv_cache_params is not None
            else [0] * batch_size
        )

        # ``attn_metadata.seq_lens`` from the PyExecutor is the
        # per-step new-token count. The M3 sparse-attention algorithm
        # consumes a *cumulative* kv length: ``minimax_m3_sparse_*``
        # masks reads against ``metadata.seq_lens`` as the per-request
        # K-side extent. Compute that cumulative kv length per request
        # and feed it into the algorithm metadata builder.
        kv_lens_cpu_list = [
            int(num_cached_per_seq[b]) + int(seq_lens_cpu[b].item()) for b in range(batch_size)
        ]
        kv_lens_cpu = torch.tensor(kv_lens_cpu_list, dtype=torch.int32)
        kv_lens_dev = kv_lens_cpu.to(device=cache_device, non_blocking=True)

        static_buffers = self._maybe_get_m3_static_buffers(cache_device, kv_cache_manager)

        # Multi-token generation rows (spec verify) also take the extend path;
        # plain decode stays one token per row.
        is_extend = num_contexts > 0 or int(seq_lens_cpu[:batch_size].max().item()) > 1
        if is_extend:
            prefix_lens_list = [int(num_cached_per_seq[b]) for b in range(batch_size)]
            extend_seq_lens_cpu = [
                kv_lens_cpu_list[b] - prefix_lens_list[b] for b in range(batch_size)
            ]
            prefix_lens = torch.tensor(
                prefix_lens_list,
                dtype=torch.int32,
                device=cache_device,
            )
            m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                seq_lens=kv_lens_dev,
                seq_lens_cpu=kv_lens_cpu,
                is_prefill=True,
                prefix_lens=prefix_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                device=cache_device,
                static_buffers=static_buffers,
            )
        else:
            m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                seq_lens=kv_lens_dev,
                seq_lens_cpu=kv_lens_cpu,
                is_prefill=False,
                device=cache_device,
                static_buffers=static_buffers,
            )

        self.minimax_m3 = {
            "metadata": m3_meta,
            "out_cache_loc": out_cache_loc,
        }

    def _refresh_attachment(self) -> None:
        """Bring the attachment in line with kv_lens_cuda, which a speculative step corrects on the device.

        Device-only and idempotent. seq_lens_cpu and max_seqlen_k keep the
        staged values; they only bound widths the kernels mask by seq_lens.
        """
        attachment = self.minimax_m3
        if not attachment:
            return
        meta = attachment["metadata"]
        out_cache_loc = attachment["out_cache_loc"]
        batch = int(meta.slot_ids.shape[0])
        kv_lens = self.kv_lens_cuda[:batch]
        meta.seq_lens[:batch].copy_(kv_lens)
        if meta.is_prefill:
            # Only the K-side prefix moves with rejections; the Q-side layout
            # (cu_seqlens_q, q_batch_row) is fixed per step.
            total_q = int(meta.q_positions.shape[0])
            cu = meta.cu_seqlens_q
            meta.prefix_lens[:batch].copy_(kv_lens - (cu[1 : batch + 1] - cu[:batch]))
            q_positions, cache_slots = derive_q_positions_and_cache_slots(
                meta.req_to_token,
                meta.prefix_lens[:batch],
                cu,
                meta.q_batch_row[:total_q],
            )
            meta.q_positions[:total_q].copy_(q_positions)
            out_cache_loc[:total_q].copy_(cache_slots)
        else:
            # No-op today (decode rows are not corrected); kept for symmetry.
            out_cache_loc[:batch].copy_(derive_decode_cache_slots(meta.req_to_token, kv_lens))


class MiniMaxM3MsaMetadata(Metadata):
    """What the MSA implementation of MiniMax-M3 sparse attention reads.

    The step's context rows run on fmha_sm100 and its generation rows on the
    dedicated decode kernels. This metadata stages the cache slots, page
    tables, lengths and plans both read, in buffers a CUDA graph can capture,
    and patches the lengths and what derives from them once the KV lengths of
    a speculative step are corrected. Its logic is
    MiniMaxM3MsaSparseAttentionMetadata's, taken over method by method.
    """

    requires = (TrtllmCoreMetadata, SparseMetadata, SpecDecodingMetadata)

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        params = self.sparse_metadata_params
        self._msa_params = params if isinstance(params, MiniMaxM3SparseMetadataParams) else None
        if config.is_cuda_graph:
            # The metadata of a CUDA graph batch size shares the buffers of the
            # one it is created from; see init_cuda_graph_metadata.
            return
        self._create_msa_buffers()
        self._validate_decode_kernel_support()

    def init_cuda_graph_metadata(self, graph_metadata: "MiniMaxM3MsaMetadata") -> None:
        """Start the MSA state of a CUDA graph batch size from this one.

        The buffers are sized for the worst-case geometry, so the graph
        metadata of every batch size share them instead of holding their own.
        """
        for name, value in vars(self).items():
            if name not in ("_registry", "config"):
                setattr(graph_metadata, name, value)

    def prepare(self, inputs: "BatchInputs") -> None:
        if inputs.stage_host:
            self._check_beam_width()
            # Everything below reads these.
            self._stage_host_lengths()
            # Set first: both _build_msa_fields and _build_step_plans skip the
            # fmha_sm100 preparation the decode kernels replace.
            self._set_decode_span()
            self._build_msa_fields()
            self._check_capture_is_pure_decode()
            self._build_step_plans()
        # A new forward: the first dense layer that reads the sub-page table expands it again (msa_subpage_rows).
        self._msa_subpage_expanded = False
        self._fill_tail_in_forward()
        self._update_kv_lens()

    def _generation_block_offsets_need_capacity(self) -> bool:
        """Whether generation block-offset rows need their configured capacity."""
        return (
            self.draft_kv_cache_manager is not None
            or self.is_spec_decoding_enabled
            or bool(self.kv_cache_params.num_extra_kv_tokens)
            or (
                self.runtime_features is not None
                and self.runtime_features.has_speculative_draft_tokens
            )
        )

    def _update_kv_lens(self) -> None:
        """Patch lengths, slots and valid-block counts to kv_lens_cuda.

        A speculative step under the overlap scheduler stages full-acceptance
        lengths and corrects kv_lens_cuda on the device. Shrinking keeps the
        staged page table valid, so only the per-row lengths and what derives
        from them are patched; the clamp to msa_kv_lens_staged enforces that.
        Device-only, capture-safe and idempotent; skipped without speculative
        decoding.
        """
        if not self._msa_fields_ready or not self._msa_kv_lens_dynamic:
            return
        batch = int(self.num_seqs)
        total_q = int(self.num_tokens)
        if batch <= 0 or total_q <= 0:
            return
        page = int(self.kv_cache_manager.tokens_per_block)
        n_valid_buf = (
            self._msa_prefill_n_valid_blocks
            if self._msa_prefill_n_valid_blocks is not None
            else self.msa_n_valid_blocks
        )
        kv_update = self.msa_policy.kv_update
        if kv_update is not None and kv_update(
            self.kv_lens_cuda,
            self.msa_kv_lens_staged,
            self.seq_lens_cuda,
            self.msa_q_batch_row,
            self.msa_q_intra,
            self.msa_block_table,
            page,
            batch,
            total_q,
            self.msa_seq_lens_cuda,
            None if self.msa_cu_kv_lens is None else self.msa_cu_kv_lens[: batch + 1],
            self.msa_out_cache_loc,
            None if n_valid_buf is None else n_valid_buf[:total_q],
        ):
            return
        # kv_lens_cuda is the attended length (no num_extra_kv_tokens), the same
        # domain as the staged bound and msa_seq_lens_cuda.
        kv_true = torch.minimum(self.kv_lens_cuda[:batch], self.msa_kv_lens_staged[:batch])
        self.msa_seq_lens_cuda[:batch].copy_(kv_true)
        if self.msa_cu_kv_lens is not None:
            self.msa_cu_kv_lens[0].zero_()
            torch.cumsum(kv_true, 0, out=self.msa_cu_kv_lens[1 : batch + 1])

        qbr = self.msa_q_batch_row[:total_q].to(torch.long)
        qo_dev = self.seq_lens_cuda[:batch]
        # Position each query token attends up to.
        pos = kv_true[qbr] - qo_dev[qbr] + self.msa_q_intra[:total_q]

        # KV/idx-K write slots, by the formula build_paged_kv_slot_mapping used
        # for the staged ones: block_table[request, pos // page] * page +
        # pos % page. Positions only move down, so the page is one prepare()
        # staged for that request; a padding row (pos -1) keeps its own first
        # slot, as on the host.
        page = int(self.kv_cache_manager.tokens_per_block)
        table = self.msa_block_table
        pos_slot = pos.clamp_min(0)
        block_col = torch.div(pos_slot, page, rounding_mode="floor").clamp(
            max=int(table.shape[1]) - 1
        )
        slots = table[qbr, block_col.to(torch.long)] * page + torch.remainder(pos_slot, page)
        self.msa_out_cache_loc[:total_q].copy_(slots)

        # Per-token valid-block counts for top-k, as per_token_valid_blocks
        # derives them on the host: ceil((pos + 1) / page), 0 for a row that
        # attends nothing (a CUDA-graph padding row), which the selector and
        # the decode kernel both accept.
        n_valid = torch.div((pos + 1).clamp_min(0) + (page - 1), page, rounding_mode="floor")
        n_valid_buf = (
            self._msa_prefill_n_valid_blocks
            if self._msa_prefill_n_valid_blocks is not None
            else self.msa_n_valid_blocks
        )
        if n_valid_buf is not None:
            n_valid_buf[:total_q].copy_(n_valid.to(torch.int32))

    def get_empty(
        self,
        buffers,
        tensor_shape: Sequence[int],
        dtype: torch.dtype,
        cache_name: str,
        capture_graph: bool = False,
    ) -> torch.Tensor:
        """A device buffer from the CUDA graph buffer pool, or a new one without a pool."""
        return AttentionMetadata.get_empty(
            buffers,
            list(tensor_shape),
            dtype=dtype,
            cache_name=self.config.buffer_name_prefix + cache_name,
            capture_graph=capture_graph,
        )


def _take_over(metadata_type: Type[Metadata], source: type, not_taken: frozenset) -> None:
    """Give metadata_type the methods, properties and class defaults of source."""
    # FIXME: move these functions from the old metadata classes into the classes above
    # once the feature-split metadata is the only path.
    for name, value in vars(source).items():
        if name.startswith("__") or name in not_taken or name in vars(metadata_type):
            continue
        setattr(metadata_type, name, value)


# Not taken: the hooks of the old metadata, whose work the classes above do in prepare.
_take_over(
    MiniMaxM3TritonMetadata,
    MiniMaxM3AttentionMetadata,
    frozenset({"prepare", "on_update_kv_lens"}),
)
_take_over(
    MiniMaxM3MsaMetadata,
    MiniMaxM3MsaSparseAttentionMetadata,
    frozenset({"prepare", "on_update_kv_lens"}),
)
