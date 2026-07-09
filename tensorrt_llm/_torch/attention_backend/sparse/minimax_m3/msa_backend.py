# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA-backed MiniMax-M3 sparse attention on the `TrtllmAttention` stack.

Mimics `DSATrtllmAttention`:

  * `MiniMaxM3MSATrtllmAttention` subclasses `TrtllmAttention` and reuses
    its inherited `forward` (the standard `fmha_libs` dispatch loop). It
    overrides only the sparse hooks (`sparse_attn_predict`,
    `sparse_kv_predict`) and owns an `MsaIndexer`. It does not inherit the
    Triton reference backend.
  * The main sparse GQA runs through the registered `MsaSparseGqaFmha`
    (an `Fmha` that does its own whole-batch dispatch) selected by the
    dispatch loop.
  * The indexer calls `fmha_sm100` directly (prefill) or the graph-safe
    decode driver (decode) to produce the per-query selected block
    indices, which the model layer threads through
    `forward_args.topk_indices`.
  * `MiniMaxM3MSATrtllmAttentionMetadata` subclasses
    `TrtllmAttentionMetadata`, so the inherited `forward` reads the
    standard metadata fields, while the MSA-specific paged views, page
    tables, and block writes are supplied through the
    `MsaSparseMetadataProtocol` methods added here (reusing the staging in
    `metadata`).

The backend and metadata classes are defined inside
`get_minimax_m3_msa_attention_backend_cls` (with a lazy `trtllm` import)
so that importing this module during attention-backend package init does
not pull in `trtllm` -> `fmha` -> `interface` and form an import cycle.
This mirrors the Triton reference backend's
`get_minimax_m3_attention_backend_cls` factory.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .common import (
    _MSA_REQUIRED_HEAD_DIM,
    _MSA_REQUIRED_TOPK,
    build_kv_indices_and_lens,
    cache_view_to_msa_paged,
    page_size_from_view,
    write_main_kv_slots,
)
from .indexer import MsaIndexer
from .metadata import M3StepState, MiniMaxM3SparseConfig, build_m3_sparse_metadata_and_plans

if TYPE_CHECKING:
    from .decode_wrapper.dispatch import M3DecodePlanner
    from .metadata import M3PlanTables, MiniMaxM3SparseAttentionMetadata


def _whole_batch_lens(
    m3_meta: "MiniMaxM3SparseAttentionMetadata",
    plans: Optional["M3PlanTables"],
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Whole-batch `(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices)`.

    Prefers the pre-staged plan tables (CUDA-graph-stable buffers) and
    falls back to an eager rebuild for focused tests and the first eager
    warmup pass.
    """
    if plans is not None:
        return (
            plans.qo_lens_cpu,
            plans.kv_lens_cpu,
            plans.qo_offset_cpu,
            plans.kv_indices,
        )
    seq_lens_cpu = m3_meta.seq_lens_cpu.to(torch.int32)
    if m3_meta.is_prefill:
        if m3_meta.extend_seq_lens_cpu is None or m3_meta.prefix_lens is None:
            raise RuntimeError("prefill metadata requires extend_seq_lens_cpu / prefix_lens")
        qo_lens_cpu = torch.tensor(m3_meta.extend_seq_lens_cpu, dtype=torch.int32)
        qo_offset_cpu = m3_meta.prefix_lens.detach().to(device="cpu", dtype=torch.int32)
    else:
        batch = int(seq_lens_cpu.shape[0])
        qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
        qo_offset_cpu = (seq_lens_cpu - 1).to(torch.int32)
    kv_indices, _ = build_kv_indices_and_lens(m3_meta, page_size)
    return qo_lens_cpu, seq_lens_cpu, qo_offset_cpu, kv_indices


@functools.lru_cache(maxsize=1)
def get_minimax_m3_msa_attention_backend_cls():
    """Return `MiniMaxM3MSATrtllmAttention` (selection entry point).

    Defined lazily (with a deferred `trtllm` import) so importing this
    module during attention-backend package init does not form the
    `trtllm` -> `fmha` -> `interface` import cycle.
    """
    from dataclasses import dataclass

    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

    @dataclass(init=False)
    class MiniMaxM3MSATrtllmAttentionMetadata(TrtllmAttentionMetadata):
        """`TrtllmAttentionMetadata` for MiniMax-M3 MSA sparse layers.

        Layers the MSA staging (paged HND views, page tables, per-request
        CPU lengths, main/index-K writes) on top of the standard metadata
        so `MsaSparseGqaFmha` can drive its whole-batch dispatch.
        Implements `MsaSparseMetadataProtocol`.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Per-rank sparse config, published once by the attention
            # layer's first (eager warmup) forward via ensure_m3_config().
            # CUDA-graph metadata instances receive it through the
            # shallow copy in create_cuda_graph_metadata(), which runs
            # after warmup and before capture.
            self._m3_config: Optional[MiniMaxM3SparseConfig] = None
            # Per-step state rebuilt by prepare().
            self.m3_step: Optional[M3StepState] = None
            # Per-instance CUDA-graph-stable staging: the paged-KV table
            # buffers and the decode planner. Each capture bucket has its
            # own metadata instance and therefore its own buffers,
            # mirroring the FlashInfer backend's per-instance wrappers.
            self._m3_static_buffers: Optional[dict] = None
            self._msa_plan_cache = None
            self._m3_decode_planner: Optional["M3DecodePlanner"] = None

        def ensure_m3_config(self, config: MiniMaxM3SparseConfig) -> None:
            """Publish the per-rank sparse config (first writer wins)."""
            if self._m3_config is None:
                self._m3_config = config

        def create_cuda_graph_metadata(self, *args, **kwargs):
            clone = super().create_cuda_graph_metadata(*args, **kwargs)
            if clone is self:
                return clone
            # The shallow copy shares this instance's mutable staging;
            # reset it so the graph bucket allocates and plans its own
            # buffers in its first prepare(). _m3_config is immutable and
            # intentionally carried over.
            clone.m3_step = None
            clone._m3_static_buffers = None
            clone._msa_plan_cache = None
            clone._m3_decode_planner = None
            return clone

        # lifecycle

        def _cache_device(self) -> torch.device:
            kv_cache_manager = getattr(self, "kv_cache_manager", None)
            if kv_cache_manager is not None:
                try:
                    return kv_cache_manager.get_buffers(0).device
                except Exception:
                    pass
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        def _maybe_static_buffers(self, cache_device: torch.device) -> Optional[dict]:
            """Return persistent M3 buffers when CUDA-graph stability is needed."""
            need_static = bool(getattr(self, "is_cuda_graph", False)) or (
                self._m3_static_buffers is not None
            )
            if not need_static:
                return None
            if (
                self._m3_static_buffers is not None
                and self._m3_static_buffers.get("device") == cache_device
            ):
                return self._m3_static_buffers
            placeholder: dict = {
                "device": cache_device,
                "max_num_sequences_hint": int(
                    getattr(self, "max_num_sequences", None) or self.max_num_requests
                ),
                "max_num_tokens_hint": int(
                    getattr(self, "max_num_tokens", None)
                    or (int(getattr(self, "max_num_sequences", None) or self.max_num_requests))
                ),
            }
            self._m3_static_buffers = placeholder
            return placeholder

        def prepare(self) -> None:
            super().prepare()
            self.m3_step = None
            cache_device = self._cache_device()
            static_buffers = self._maybe_static_buffers(cache_device)
            # The sparse KV block size equals the cache manager's
            # tokens_per_block (asserted by MiniMaxM3KVCacheManagerV2), so
            # the page-table staging needs no model geometry.
            kv_cache_manager = getattr(self, "kv_cache_manager", None)
            plan_block_size = (
                int(kv_cache_manager.tokens_per_block)
                if kv_cache_manager is not None
                and bool(getattr(kv_cache_manager, "use_msa", False))
                else None
            )
            self.m3_step, self._msa_plan_cache = build_m3_sparse_metadata_and_plans(
                self,
                static_buffers=static_buffers,
                plan_cache=self._msa_plan_cache,
                plan_block_size=plan_block_size,
            )
            # Decode: refresh the per-instance planner's CTA work split
            # for the current batch (host work, outside any capture).
            step = self.m3_step
            if (
                step is not None
                and step.plans is not None
                and not step.metadata.is_prefill
                and self._m3_config is not None
            ):
                batch = int(step.metadata.seq_lens.shape[0])
                self.m3_decode_planner(batch)

        # internal accessors

        def _require_step(self) -> M3StepState:
            if self.m3_step is None:
                raise RuntimeError(
                    "MiniMaxM3MSATrtllmAttentionMetadata.m3_step is not built; "
                    "prepare() must run before the sparse forward."
                )
            return self.m3_step

        @property
        def m3_meta(self) -> "MiniMaxM3SparseAttentionMetadata":
            return self._require_step().metadata

        @property
        def m3_out_cache_loc(self) -> torch.Tensor:
            return self._require_step().out_cache_loc

        @property
        def m3_plans(self) -> Optional["M3PlanTables"]:
            return self._require_step().plans

        def m3_decode_planner(self, batch: int) -> "M3DecodePlanner":
            """Per-instance decode planner, planned for `batch` requests.

            Created lazily on the first (eager, pre-capture) decode use;
            `prepare()` replans it every decode step, so calls during
            capture or replay hit the already-planned state and do no
            host work.
            """
            planner = self._m3_decode_planner
            if planner is None:
                from .decode_wrapper.dispatch import M3DecodeGeometry, M3DecodePlanner

                if self._m3_config is None:
                    raise RuntimeError(
                        "MiniMax-M3 MSA decode requires the sparse config; "
                        "the attention layer must call ensure_m3_config() "
                        "before the first decode step."
                    )
                config = self._m3_config
                m3_meta = self.m3_meta
                page_size = int(self._kv_views(0)[0].shape[1])
                max_batch = int(getattr(self, "max_num_sequences", None) or self.max_num_requests)
                geometry = M3DecodeGeometry(
                    num_q_heads=config.num_q_heads,
                    num_kv_heads=config.num_kv_heads,
                    num_index_heads=config.num_index_heads,
                    head_dim=config.head_dim,
                    page_size=page_size,
                    topk=config.topk,
                    init_blocks=config.init_blocks,
                    local_blocks=config.local_blocks,
                    max_batch=max(max_batch, batch),
                    max_kv_len=int(m3_meta.req_to_token.shape[1]),
                )
                planner = M3DecodePlanner(geometry, m3_meta.seq_lens.device)
                self._m3_decode_planner = planner
            planner.plan(batch)
            return planner

        def _kv_views(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """Per-layer NHD K/V slot views `[num_pages, page_size, num_kv_heads, head_dim]`."""
            buffers = self.kv_cache_manager.get_buffers(layer_idx)
            return buffers[:, 0], buffers[:, 1]

        # MsaSparseMetadataProtocol

        def msa_get_paged_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            k_view, v_view = self._kv_views(layer_idx)
            return cache_view_to_msa_paged(k_view), cache_view_to_msa_paged(v_view)

        def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
            """Raw paged index-K view for the indexer; HND conversion is done there."""
            return self.kv_cache_manager.get_index_k_buffer(layer_idx)

        def msa_write_main_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
            k_view, v_view = self._kv_views(layer_idx)
            num_kv_heads = int(k_view.shape[2])
            head_dim = int(k_view.shape[3])
            num_tokens = int(k.shape[0])
            out_cache_loc = self.m3_out_cache_loc
            write_main_kv_slots(
                k_view, out_cache_loc, k.reshape(num_tokens, num_kv_heads, head_dim)
            )
            write_main_kv_slots(
                v_view, out_cache_loc, v.reshape(num_tokens, num_kv_heads, head_dim)
            )

        def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
            idx_cache = self.msa_idx_k_cache(layer_idx)
            sparse_index_dim = int(idx_cache.shape[-1])
            num_tokens = int(idx_k.shape[0])
            write_main_kv_slots(
                idx_cache, self.m3_out_cache_loc, idx_k.reshape(num_tokens, 1, sparse_index_dim)
            )

        def msa_is_prefill(self) -> bool:
            return bool(self.m3_meta.is_prefill)

        def msa_whole_batch_lens(
            self,
            *,
            causal: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
            # The indexer and metadata are built over the whole batch, and
            # fmha_sm100 handles mixed decode/prefill varlen batches in one
            # call (decode rows are 1-token causal extends), so the FMHA
            # runs the whole batch at once rather than splitting into
            # per-request context/generation phases. Returning whole-batch
            # lengths keeps the plan's total_q consistent with the
            # whole-batch q the model passes and the whole-batch block
            # indices the indexer produced.
            page_size = int(self._kv_views(0)[0].shape[1])
            qo_all, kv_all, qo_off_all, kv_indices_all = _whole_batch_lens(
                self.m3_meta, self.m3_plans, page_size
            )
            return qo_all, kv_all, (qo_off_all if causal else None), kv_indices_all

        def msa_run_sparse_decode(
            self,
            *,
            layer_idx: int,
            q: torch.Tensor,
            kv_block_indexes: torch.Tensor,
            sm_scale: float,
        ) -> torch.Tensor:
            """CUDA-graph-safe decode sparse GQA via the in-tree planner.

            Uses this metadata instance's `M3DecodePlanner` (the same one
            the indexer used for proxy and block selection), planned by
            `prepare()`. Returns `[num_tokens, num_q_heads * head_dim]`.
            """
            plans = self.m3_plans
            if plans is None:
                raise RuntimeError(
                    "MiniMax-M3 MSA decode requires pre-staged plan tables; "
                    "prepare() did not build them (missing use_msa on the "
                    "KV cache manager)."
                )
            k_paged, v_paged = self.msa_get_paged_kv(layer_idx)
            seq_lens = self.m3_meta.seq_lens.to(torch.int32)
            batch = int(q.shape[0])
            planner = self.m3_decode_planner(batch)
            out = planner.sparse_attention(
                q,
                k_paged,
                v_paged,
                kv_block_indexes,
                seq_lens=seq_lens,
                kv_page_indptr=plans.kv_page_indptr,
                kv_indices=plans.kv_indices,
                sm_scale=sm_scale,
            )
            g = planner.geom
            return out.reshape(batch, g.num_q_heads * g.head_dim)

    class MiniMaxM3MSATrtllmAttention(TrtllmAttention):
        """MSA-backed MiniMax-M3 sparse attention (mimics `DSATrtllmAttention`)."""

        Metadata = MiniMaxM3MSATrtllmAttentionMetadata

        def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config=None,
            *,
            sparse_params,
            **kwargs,
        ):
            TrtllmAttention.__init__(
                self,
                layer_idx,
                num_heads,
                head_dim,
                num_kv_heads=num_kv_heads,
                quant_config=quant_config,
                sparse_params=sparse_params,
                **kwargs,
            )
            self.m3_config = MiniMaxM3SparseConfig.from_sparse_params(
                sparse_params,
                num_q_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
                head_dim=head_dim,
            )
            self.disable_index_value = bool(sparse_params.disable_index_value)
            self._validate_msa_preconditions()
            self.indexer = MsaIndexer(self.m3_config)

        def _validate_msa_preconditions(self) -> None:
            if not self.disable_index_value:
                raise NotImplementedError(
                    "MSA backend requires disable_index_value=True (MSA's proxy pass "
                    "only consumes max_score; an index-V path is not implemented)."
                )
            if self.m3_config.head_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires head_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.head_dim}."
                )
            if self.m3_config.sparse_index_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires sparse_index_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.sparse_index_dim}."
                )
            if self.m3_config.topk != _MSA_REQUIRED_TOPK:
                raise NotImplementedError(
                    f"MSA backend requires topk={_MSA_REQUIRED_TOPK}, got {self.m3_config.topk}."
                )

        @classmethod
        def support_fused_rope(cls) -> bool:
            # The MiniMax-M3 model layer applies partial RoPE to both the
            # main and index branches explicitly.
            return False

        # No create_fmha_libs override: MsaSparseGqaFmha is registered
        # first in FMHA_LIBS and its is_available() claims exactly the
        # MiniMaxM3MSATrtllmAttention layers, so the inherited dispatch
        # builds [MsaSparseGqaFmha, ...standard libs] here and the
        # standard libs stay reachable for dense (non-sparse) requests.

        # indexer entry point (called by the model layer, DSA-style)

        def run_indexer(
            self,
            idx_q: torch.Tensor,
            idx_k: torch.Tensor,
            metadata,
            *,
            idx_sm_scale: Optional[float] = None,
        ) -> torch.Tensor:
            """Write the index-K cache and return the selected block indices.

            Mirrors DSA's `indexer.sparse_attn_indexer`: the model layer
            runs this before `forward` and threads the result through
            `forward_args.topk_indices`. Returns `[total_q, num_kv_heads,
            topk]`.
            """
            config = self.m3_config
            metadata.ensure_m3_config(config)
            idx_sm_scale = (
                idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5
            )
            num_tokens = int(idx_q.shape[0])
            idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
            idx_k_view = idx_k.view(num_tokens, 1, config.sparse_index_dim)

            metadata.msa_write_idx_k(self.layer_idx, idx_k_view)
            idx_k_cache = metadata.msa_idx_k_cache(self.layer_idx)
            m3_meta = metadata.m3_meta
            page_size = page_size_from_view(metadata._kv_views(self.layer_idx)[0])

            if m3_meta.is_prefill:
                qo, kv, qo_off, kv_indices = _whole_batch_lens(
                    m3_meta, metadata.m3_plans, page_size
                )
                return self.indexer.select_blocks_prefill(
                    idx_q_view,
                    idx_k_cache,
                    idx_sm_scale=idx_sm_scale,
                    qo_lens_cpu=qo,
                    kv_lens_cpu=kv,
                    qo_offset_cpu=qo_off,
                    kv_indices=kv_indices,
                )
            return self.indexer.select_blocks_decode(
                idx_q_view,
                idx_k_cache,
                metadata,
                idx_sm_scale=idx_sm_scale,
                page_size=page_size,
            )

        # sparse hooks (consumed by inherited TrtllmAttention.forward)

        def sparse_attn_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            # The model layer runs run_indexer() and passes the selected
            # block indices through forward_args.topk_indices. Publish them
            # as the sparse attention indices the MsaSparseGqaFmha reads.
            return getattr(forward_args, "topk_indices", None), None

        def sparse_kv_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            return None, None

    return MiniMaxM3MSATrtllmAttention


__all__ = [
    "get_minimax_m3_msa_attention_backend_cls",
]
