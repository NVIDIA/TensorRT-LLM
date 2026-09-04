# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA-backed MiniMax-M3 sparse attention on the TrtllmAttention stack.

  * MiniMaxM3MsaSparseAttention subclasses TrtllmAttention and reuses its
    inherited forward, overriding only the sparse hooks and owning an
    MsaIndexer.
  * The main attention runs through the registered MsaPrefillFmha for context
    rows and MsaDecodeFmha for generation rows.
  * The indexer calls fmha_sm100 directly to produce the per-query selected
    block indices, which the model layer threads through
    forward_args.sparse_backend_args.
  * MiniMaxM3MsaSparseAttentionMetadata subclasses TrtllmAttentionMetadata and
    stores its per-forward MSA tensors in CUDA-graph-stable buffers.
    The buffers are allocated once in __post_init__ via
    get_empty(capture_graph=...), and prepare() copies the per-step values
    into them. The standard CUDAGraphRunner clones one metadata per graph
    batch size (create_cuda_graph_metadata), so no per-batch-size cache is
    needed here.

The classes subclass TrtllmAttention and TrtllmAttentionMetadata, imported at
module scope. That is cycle-free only because the dependency runs one way: the
two FMHA libraries reach the kernels through ...minimax_m3_kernels, never
through this package, so trtllm's import chain does not come back here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.bindings import DataType
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..minimax_m3_kernels.msa_utils import (
    MSA_REQUIRED_HEAD_DIM,
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    per_token_valid_blocks,
    require_msa_module,
)
from ..minimax_m3_kernels.trtllm_gen_dense_decode import (
    dense_decode_unsupported_reason,
    uniform_subpages_per_slot,
    write_subpage_block_table,
)
from .common import (
    MiniMaxM3SparseConfig,
    MiniMaxM3SparseMetadataParams,
    build_paged_kv_slot_mapping,
    write_kv_slots,
)
from .msa_indexer import MsaIndexer, cutedsl_score_runner


def _cache_device(meta) -> torch.device:
    """Device hosting the paged KV buffers, else the current CUDA device."""
    kv_cache_manager = meta.kv_cache_manager
    if kv_cache_manager is not None:
        try:
            return kv_cache_manager.get_buffers(0).device
        except (AttributeError, IndexError, KeyError):
            # A manager that exposes no layer-0 buffer, as in a focused test.
            pass
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def _worst_case_proxy_max_k_tiles(
    fmha_sm100,
    *,
    num_index_heads: int,
    kv_cache_manager,
    max_batch: int,
) -> int:
    """Return max_k_tiles for a proxy plan at the manager's max KV length."""
    page_size = int(kv_cache_manager.tokens_per_block)
    max_kv_len = int(kv_cache_manager.max_blocks_per_seq) * page_size
    qo_lens = torch.ones(max_batch, dtype=torch.int32)
    kv_lens = torch.full((max_batch,), max_kv_len, dtype=torch.int32)
    qo_offset = kv_lens - qo_lens
    proxy_plan = fmha_sm100.fmha_sm100_plan(
        qo_lens,
        kv_lens,
        num_index_heads,
        num_kv_heads=1,
        qo_offset=qo_offset,
        page_size=page_size,
        output_maxscore=True,
        num_kv_splits=1,
        causal=True,
    )
    return int(proxy_plan[3]["max_k_tiles"])


class MsaDecodeSpan(NamedTuple):
    """Generation rows served by MiniMax-M3's dedicated decode kernels.

    A named carrier for the pair each of them indexes by, in the style of
    PagedKvSlotMapping in common.py. See msa_decode_span.
    """

    # First generation row, which is also the step's context row count.
    row_first: int
    # Uniform query token count per generation request.
    query_len: int


@dataclass(init=False)
class MiniMaxM3MsaSparseAttentionMetadata(TrtllmAttentionMetadata):
    """TrtllmAttentionMetadata for MiniMax-M3 MSA sparse layers.

    A step is prepared for a fixed division of labour: its context rows run on
    fmha_sm100 through MsaPrefillFmha, and its generation rows on MiniMax-M3's
    dedicated decode kernels through MsaDecodeFmha. msa_decode_span says where
    the second range begins. Neither library chooses, so the staging below is
    unambiguous.

    Tensors read inside the captured forward are CUDA-graph-stable: the
    cache slots (msa_out_cache_loc), page tables (msa_kv_indices,
    msa_block_table), lengths (msa_seq_lens_cuda) and proxy scratch
    (msa_max_score, msa_n_valid_blocks) are allocated once from the manager's
    worst-case geometry. All of those except msa_max_score are refreshed via
    copy_; the fmha_sm100 proxy pass writes msa_max_score directly (see
    msa_proxy_max_score_view).

    Length inputs to fmha_sm100_plan (msa_qo_lens_cpu, msa_kv_lens_cpu,
    msa_qo_offset_cpu) are host properties of the base seq_lens/kv_lens, read
    only while building plans in prepare() (outside capture), so they need no
    graph-stable storage. The plans themselves (msa_prefill_*_plan) cover the
    context rows alone, and a step carrying those is never captured, so they
    need none either.
    """

    # Graph-stable buffers; consumers slice to the live count at the call
    # site. Filled once the current step's cache write is prepared.
    msa_out_cache_loc: Optional[torch.Tensor] = None
    msa_kv_indices: Optional[torch.Tensor] = None
    msa_max_score: Optional[torch.Tensor] = None
    msa_n_valid_blocks: Optional[torch.Tensor] = None
    # The same page table and lengths as msa_kv_indices / msa_kv_lens, in the
    # per-request 2-D form the decode kernels index directly
    # (block_table[request, block] and seq_lens[request]). fmha_sm100 instead
    # takes the flattened msa_kv_indices with the page count implied by its
    # plan, so both forms are staged rather than derived at the call site.
    msa_block_table: Optional[torch.Tensor] = None
    msa_seq_lens_cuda: Optional[torch.Tensor] = None
    # msa_block_table with each slot expanded into the K and V sub-pages the
    # trtllm-gen dense kernel indexes. _msa_subpages_per_slot is the expansion
    # factor, or 0 where the pool has no single one; see msa_subpage_rows.
    msa_subpage_block_table: Optional[torch.Tensor] = None
    _msa_subpages_per_slot: int = 0

    # _msa_buffers_ready gates the once-only device buffers;
    # _msa_fields_ready marks that the current step's buffers are populated.
    _msa_buffers_ready: bool = False
    _msa_fields_ready: bool = False
    # Sparse geometry the plans need.
    _msa_params: Optional[MiniMaxM3SparseMetadataParams] = None
    # This step's fmha_sm100 plans, plain tuples with no graph-stable buffers
    # because they cover the context rows alone and a step carrying those is
    # never captured. Built once per step in prepare() and reused by every
    # layer.
    _msa_prefill_proxy_plan: Optional[tuple] = None
    _msa_prefill_gqa_plan: Optional[tuple] = None
    _msa_prefill_dense_plan: Optional[tuple] = None
    # Per-token valid-block count for the prefill-side indexer proxy. It is
    # layer-invariant (a function of qo/kv lengths and page size), so it is
    # computed on the host and staged to the device once per step via a
    # non-blocking copy_, then reused by every sparse layer's indexer.
    # _msa_prefill_n_valid_buf is the persistent backing store for the view.
    _msa_prefill_n_valid_buf: Optional[torch.Tensor] = None
    _msa_prefill_n_valid_blocks: Optional[torch.Tensor] = None
    # This step's per-request host lengths, staged by _stage_host_lengths.
    _msa_qo_lens_cpu: Optional[torch.Tensor] = None
    _msa_kv_lens_cpu: Optional[torch.Tensor] = None
    _msa_qo_offset_cpu: Optional[torch.Tensor] = None
    # Set once per step by _set_decode_span(), ahead of every other preparation
    # step; see msa_decode_span.
    _msa_decode_span: Optional[MsaDecodeSpan] = None
    # See msa_max_kv_len.
    _msa_max_kv_len: int = 0
    # See msa_worst_case_max_k_tiles.
    _msa_worst_case_max_k_tiles: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        # A plain instance attribute rather than a dataclass field: this class
        # is init=False, so a field's default_factory would never run and the
        # attribute would not exist. See msa_mark_kv_written for what it holds.
        self._msa_kv_written_layers: set[int] = set()
        params = self.sparse_metadata_params
        self._msa_params = params if isinstance(params, MiniMaxM3SparseMetadataParams) else None
        self._create_msa_buffers()
        self._validate_decode_kernel_support()

    @property
    def msa_qo_lens_cpu(self) -> Optional[torch.Tensor]:
        """Per-request query length (host int32), from the base seq_lens."""
        return self._msa_qo_lens_cpu

    @property
    def msa_kv_lens_cpu(self) -> Optional[torch.Tensor]:
        """Per-request KV length, cached plus new tokens (host int32)."""
        return self._msa_kv_lens_cpu

    @property
    def msa_qo_offset_cpu(self) -> Optional[torch.Tensor]:
        """Per-request causal offset (kv_len - qo_len), the cached prefix length."""
        return self._msa_qo_offset_cpu

    def _stage_host_lengths(self) -> None:
        """Build this step's per-request host length tensors, once.

        Every planner and both FMHA libraries read these, several of them once
        per layer, and each build slices, casts to int32, and pins. Pinning is
        what lets the planners stage them with non-blocking copies instead of
        copying out of pageable memory, so it is worth paying for once a step.
        """

        def as_pinned_int32(lens: torch.Tensor) -> torch.Tensor:
            rows = lens[: self.num_seqs]
            return maybe_pin_memory(rows.to(torch.int32) if rows.dtype != torch.int32 else rows)

        seq_lens = self.seq_lens
        kv_lens = getattr(self, "kv_lens", None)
        self._msa_qo_lens_cpu = None if seq_lens is None else as_pinned_int32(seq_lens)
        self._msa_kv_lens_cpu = (
            None if seq_lens is None or kv_lens is None else as_pinned_int32(kv_lens)
        )
        self._msa_qo_offset_cpu = (
            None
            if self._msa_kv_lens_cpu is None
            else maybe_pin_memory(self._msa_kv_lens_cpu - self._msa_qo_lens_cpu)
        )

    # The attention plans cover the context rows; the proxy plan covers
    # whatever rows the CuTe DSL scorer did not take. See _msa_attn_plan_rows
    # and _msa_proxy_plan_rows.
    @property
    def msa_prefill_proxy_plan(self) -> Optional[tuple]:
        """Prebuilt indexer proxy plan for this step's fmha_sm100 rows."""
        return self._msa_prefill_proxy_plan

    @property
    def msa_prefill_gqa_plan(self) -> Optional[tuple]:
        """Prebuilt context-phase sparse GQA plan."""
        return self._msa_prefill_gqa_plan

    @property
    def msa_prefill_dense_plan(self) -> Optional[tuple]:
        """Prebuilt context-phase dense GQA plan."""
        return self._msa_prefill_dense_plan

    @property
    def msa_decode_span(self) -> Optional[MsaDecodeSpan]:
        """This step's generation rows, or None where it has none.

        A batch is ordered context-first, so the generation requests are the
        row suffix [row_first, num_seqs) and their query tokens the matching
        token suffix. Those rows run on MiniMax-M3's dedicated decode kernels,
        which address query tokens by the uniform per-request query_len; see
        _set_decode_span.
        """
        return self._msa_decode_span

    @property
    def msa_decode_query_len(self) -> Optional[int]:
        """Uniform per-request query length over this step's generation rows."""
        span = self._msa_decode_span
        return span.query_len if span is not None else None

    @property
    def msa_max_kv_len(self) -> int:
        """Staged max KV length over this step's generation rows.

        A scheduling upper bound for the decode kernels, taken over the
        generation rows alone so a long context request cannot inflate it.
        """
        return self._msa_max_kv_len

    @property
    def msa_worst_case_max_k_tiles(self) -> int:
        """max_k_tiles of a proxy plan at the manager's worst-case KV length.

        The bound the proxy scratch was allocated against, so it is valid for
        any step and lets a step that skipped the proxy plan still shape its
        max_score view.
        """
        return self._msa_worst_case_max_k_tiles

    @property
    def msa_prefill_n_valid_blocks(self) -> Optional[torch.Tensor]:
        """Device int32 valid-block count for the fmha_sm100 proxy rows, or None
        where the step has none (a pure-decode step or a structural test)."""
        return self._msa_prefill_n_valid_blocks

    def msa_subpage_rows(self, row_first: int, row_last: int) -> Tuple[Optional[torch.Tensor], int]:
        """Staged sub-page block table for the given rows, with its factor.

        (None, 0) when the pool has no single sub-pages-per-slot factor, which
        leaves the caller to expand its own layer's table.
        """
        table = self.msa_subpage_block_table
        if table is None:
            return None, 0
        return table[row_first:row_last], self._msa_subpages_per_slot

    def msa_mark_kv_written(self, layer_idx: int) -> bool:
        """Claim this step's main K/V cache write for one layer, once.

        A mixed batch runs the layer's context and generation phases as two
        FMHA calls, each of which reaches prepare_workspace, so the first
        caller claims the write and the second gets False. The claims are
        cleared in prepare(), so a captured graph carries exactly the one write
        the capture admitted.
        """
        if layer_idx in self._msa_kv_written_layers:
            return False
        self._msa_kv_written_layers.add(layer_idx)
        return True

    def _msa_main_kv_is_fp8(self) -> bool:
        """Whether the main paged K/V cache is stored as FP8 E4M3.

        The GQA and dense plans must pass use_fp8_kvcache so the inline
        sparse-prefill path selects the FP8 AOT kernels. Mirrors the k_paged
        dtype check in run_msa_prefill_gqa.
        """
        kv_cache_manager = self.kv_cache_manager
        return kv_cache_manager is not None and kv_cache_manager.dtype == DataType.FP8

    def _validate_decode_kernel_support(self) -> None:
        """Require the decode kernels to accept this run's cache geometry.

        The generation phase runs on them alone, so the geometry is settled
        once against the manager this metadata was built for. Doing it per step
        would only offer the choice of running the wrong kernel. It belongs
        here rather than in the attention's own validation because the manager,
        not the layer, fixes the page size, the index dtype and the sub-page
        pool.
        """
        params = self._msa_params
        kv_cache_manager = self.kv_cache_manager
        if params is None or kv_cache_manager is None:
            # No MSA geometry to check, as for a structural test's metadata.
            return
        page_size = int(kv_cache_manager.tokens_per_block)
        if not self._cutedsl_indexer_supported(
            num_index_heads=params.num_index_heads,
            page_size=page_size,
            # One query token per generation request; see msa_decode_span.
            decode_query_len=1,
        ):
            raise RuntimeError(
                "The MiniMax-M3 CuTe DSL indexer scorer does not support this "
                f"configuration: {params.num_index_heads} index heads, page size "
                f"{page_size}, index dtype {self._msa_index_kv_dtype()}."
            )
        dense_unsupported = dense_decode_unsupported_reason(kv_cache_manager, MSA_REQUIRED_HEAD_DIM)
        if dense_unsupported is not None:
            raise RuntimeError(
                "The MiniMax-M3 dense layers run on the trtllm-gen decode kernel, "
                f"but {dense_unsupported}"
            )

    def _create_msa_buffers(self) -> None:
        """Allocate the CUDA-graph-stable MSA device buffers.

        Buffers come from the shared graph buffer pool so they are reserved
        under capture. Sizing follows the worst-case graph geometry:
        max_num_tokens for cache slots, max_num_sequences * max_blocks_per_seq
        for the page table, and worst-case max_k_tiles for proxy scratch.
        """
        kv_cache_manager = self.kv_cache_manager
        self._msa_buffers_ready = False
        if kv_cache_manager is None or not hasattr(kv_cache_manager, "get_index_k_buffer"):
            return
        capture_graph = self.is_cuda_graph
        buffers = self.cuda_graph_buffers
        max_num_sequences = int(self.max_num_sequences)
        max_blocks_per_seq = int(kv_cache_manager.max_blocks_per_seq)
        max_total_pages = max_num_sequences * max_blocks_per_seq
        max_num_tokens = int(self.max_num_tokens)

        self.msa_out_cache_loc = self.get_empty(
            buffers,
            (max_num_tokens,),
            cache_name="msa_out_cache_loc",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.msa_kv_indices = self.get_empty(
            buffers,
            (max_total_pages,),
            cache_name="msa_kv_indices",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.msa_block_table = self.get_empty(
            buffers,
            (max_num_sequences, max_blocks_per_seq),
            cache_name="msa_block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.msa_seq_lens_cuda = self.get_empty(
            buffers,
            (max_num_sequences,),
            cache_name="msa_seq_lens_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Resolved once here rather than per step: the factor is fixed by the
        # pool's layout for the life of the manager.
        self._msa_subpages_per_slot = uniform_subpages_per_slot(kv_cache_manager)
        if self._msa_subpages_per_slot > 0:
            self.msa_subpage_block_table = self.get_empty(
                buffers,
                (max_num_sequences, 2, max_blocks_per_seq),
                cache_name="msa_subpage_block_table",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        # The proxy scratch needs the fmha_sm100 plan geometry. This metadata
        # exists only for the MSA backend, whose selection already required the
        # kernels, so a failed import here is a hard error rather than a reason
        # to skip allocation.
        params = self._msa_params
        if params is not None:
            fmha_sm100 = require_msa_module()
            max_k_tiles = _worst_case_proxy_max_k_tiles(
                fmha_sm100,
                num_index_heads=params.num_index_heads,
                kv_cache_manager=kv_cache_manager,
                max_batch=max_num_sequences,
            )
            self._msa_worst_case_max_k_tiles = int(max_k_tiles)
            self._alloc_msa_proxy_scratch(
                num_index_heads=params.num_index_heads,
                max_batch=max_num_sequences,
                max_k_tiles=max_k_tiles,
                capture_graph=capture_graph,
            )
        self._msa_buffers_ready = True

    def _alloc_msa_proxy_scratch(
        self,
        *,
        num_index_heads: int,
        max_batch: int,
        max_k_tiles: int,
        capture_graph: bool,
    ) -> None:
        """Allocate the flat proxy max-score store and the valid-block scratch.

        The store is sized for the worst-case max_k_tiles so one allocation
        serves every decode step. msa_proxy_max_score_view slices the per-step
        shape out of it.
        """
        buffers = self.cuda_graph_buffers
        self.msa_max_score = self.get_empty(
            buffers,
            (num_index_heads * max_k_tiles * max_batch,),
            cache_name="msa_max_score",
            dtype=torch.float32,
            capture_graph=capture_graph,
        )
        self.msa_n_valid_blocks = self.get_empty(
            buffers,
            (max_batch,),
            cache_name="msa_n_valid_blocks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    def _ensure_msa_decode_scratch_buffers(
        self,
        *,
        num_index_heads: int,
        max_batch: int,
        capture_graph: bool,
        required_max_k_tiles: int,
    ) -> None:
        """Ensure proxy scratch buffers exist and cover the current plan."""
        required_numel = num_index_heads * required_max_k_tiles * max_batch
        if self.msa_max_score is not None:
            if self.msa_max_score.numel() < required_numel:
                raise ValueError(
                    f"msa_max_score backing store ({self.msa_max_score.numel()} "
                    f"elements) is smaller than the decode plan needs "
                    f"({required_numel} = {num_index_heads} heads * "
                    f"{required_max_k_tiles} k-tiles * {max_batch} batch)."
                )
            return

        kv_cache_manager = self.kv_cache_manager
        if kv_cache_manager is None:
            return

        fmha_sm100 = require_msa_module()
        max_k_tiles = _worst_case_proxy_max_k_tiles(
            fmha_sm100,
            num_index_heads=num_index_heads,
            kv_cache_manager=kv_cache_manager,
            max_batch=max_batch,
        )
        if max_k_tiles < required_max_k_tiles:
            raise ValueError(
                f"Worst-case max_k_tiles ({max_k_tiles}) is less than the "
                f"decode plan ({required_max_k_tiles})."
            )
        self._msa_worst_case_max_k_tiles = int(max_k_tiles)
        self._alloc_msa_proxy_scratch(
            num_index_heads=num_index_heads,
            max_batch=max_batch,
            max_k_tiles=max_k_tiles,
            capture_graph=capture_graph,
        )

    def _ensure_prefill_n_valid_buffer(self, total_q: int, device: torch.device) -> torch.Tensor:
        """Return a persistent device int32 buffer for the valid-block count.

        A step carrying context rows is never CUDA-graph captured, so a plain
        device tensor, grown on demand and reused across steps, is sufficient.
        It is sized to the worst-case per-step query-token count.
        """
        buf = self._msa_prefill_n_valid_buf
        if buf is None or buf.numel() < total_q or buf.device != device:
            cap = max(int(total_q), int(getattr(self, "max_num_tokens", 0) or 0), 1)
            buf = torch.empty(cap, dtype=torch.int32, device=device)
            self._msa_prefill_n_valid_buf = buf
        return buf

    def prepare(self) -> None:
        super().prepare()
        self._msa_kv_written_layers.clear()
        self._check_beam_width()
        # Everything below reads these.
        self._stage_host_lengths()
        # Set first: both _build_msa_fields and _build_step_plans skip the
        # fmha_sm100 preparation the decode kernels replace.
        self._set_decode_span()
        self._build_msa_fields()
        self._check_capture_is_pure_decode()
        self._build_step_plans()

    def _check_beam_width(self) -> None:
        """Fail on beam search, which every MSA site assumes away.

        The decode kernels take one row per request, while a beam batch holds
        beam_width rows, so the block table and lengths handed to them would
        cover only the first 1 / beam_width of the batch.
        """
        if self.beam_width != 1:
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support beam search, but this "
                f"step has beam_width={self.beam_width}. Use beam_width=1 or the "
                "non-MSA MiniMax-M3 backend."
            )

    def _set_decode_span(self) -> None:
        """Describe this step's generation rows, ahead of any preparation work.

        The span is a description of the batch, not a choice between kernels:
        the generation rows always run on MiniMax-M3's dedicated decode kernels
        and the context rows always run on fmha_sm100. Whether those kernels
        can serve the run at all is settled once, up front, by
        ensure_msa_available and _validate_decode_kernel_support, so there is
        no per-step verdict here for the FMHA libraries to disagree about.

        The one property of the rows themselves that has to hold is a single
        positive query length across them, which the kernels derive the request
        id from, so a batch without it is rejected rather than served.
        """
        self._msa_decode_span = None
        self._msa_max_kv_len = 0
        qo_lens_cpu = self.msa_qo_lens_cpu
        kv_lens_cpu = self.msa_kv_lens_cpu
        if qo_lens_cpu is None or kv_lens_cpu is None:
            return
        row_first = int(self.num_contexts or 0)
        row_last = int(qo_lens_cpu.shape[0])
        if row_first >= row_last:
            # Pure prefill: no generation row to describe.
            return
        # Host-side tensors, so these reads do not sync the device.
        gen_qo_lens = qo_lens_cpu[row_first:]
        qo_min, qo_max = int(gen_qo_lens.min()), int(gen_qo_lens.max())
        if qo_max > 1:
            raise NotImplementedError(
                "MiniMax-M3 MSA attention does not support speculative decoding "
                "(multiple query tokens per decode step): generation rows "
                f"[{row_first}, {row_last}) carry up to {qo_max} query tokens. "
                "Disable speculative decoding or use the non-MSA MiniMax-M3 backend."
            )
        if qo_min != qo_max or qo_max <= 0:
            raise RuntimeError(
                "MiniMax-M3 MSA attention needs one positive query length across a "
                f"step's generation rows, which the decode kernels derive the "
                f"request id from, but rows [{row_first}, {row_last}) carry "
                f"{gen_qo_lens.tolist()}."
            )
        # Staged, i.e. before the overlap scheduler's correction, which only
        # shrinks lengths. That keeps it a valid upper bound even when it is
        # baked into a CUDA graph.
        self._msa_max_kv_len = int(kv_lens_cpu[row_first:].max())
        self._msa_decode_span = MsaDecodeSpan(row_first=row_first, query_len=qo_max)

    def _check_capture_is_pure_decode(self) -> None:
        """Fail if a CUDA graph step carries context rows.

        A context row is planned eagerly, as a plain tuple of per-step tensors
        (see _build_step_plans), so a graph that captured one would replay
        fmha_sm100 against the addresses of a step that has passed. Decode
        needs no such plan, which is why capture is confined to it.
        """
        if self.is_cuda_graph and int(self.num_contexts or 0) > 0:
            raise RuntimeError(
                "MiniMax-M3 MSA attention captured a CUDA graph for a step with "
                f"{int(self.num_contexts)} context requests. Only pure-decode steps "
                "are graph-safe here; see _build_step_plans."
            )

    def _msa_runs_no_fmha(self) -> bool:
        """Whether nothing this step reaches fmha_sm100.

        When True its whole per-step preparation is dead: the plans and the
        flattened msa_kv_indices page table. That is a pure-decode step, since
        the decode kernels then own every row. A mixed step never qualifies, as
        fmha_sm100 still runs the context prefix.
        """
        span = self._msa_decode_span
        return span is not None and span.row_first == 0

    def _msa_proxy_plan_rows(self) -> Optional[Tuple[int, int]]:
        """Batch rows this step's fmha_sm100 indexer proxy plan must cover.

        The indexer is not split by phase: it runs once per sparse layer over
        the whole batch, so its plan covers whatever the CuTe DSL scorer did
        not take.

        * pure prefill, so no span: the whole batch, the proxy scoring it all;
        * mixed: the context prefix only;
        * pure decode: None, no rows left for the proxy.
        """
        span = self._msa_decode_span
        if span is None:
            return (0, int(self.num_seqs))
        return (0, span.row_first) if span.row_first > 0 else None

    def _msa_attn_plan_rows(self) -> Optional[Tuple[int, int]]:
        """Batch rows the fmha_sm100 attention plans must cover.

        The context prefix, which is the whole of what fmha_sm100 attends: the
        generation rows are the decode kernels' and are never planned for it.
        """
        num_contexts = int(self.num_contexts or 0)
        return (0, num_contexts) if num_contexts > 0 else None

    def _msa_index_kv_dtype(self) -> torch.dtype:
        """dtype of the paged index-K cache, which index Q is cast to.

        The CuTe DSL scorer requires index Q and K to match, and run_indexer
        casts Q to the cache dtype, so the cache decides what the scorer sees.
        """
        indexer_kv_dtype = str(getattr(self.kv_cache_manager, "indexer_kv_dtype", "bf16"))
        return torch.float8_e4m3fn if indexer_kv_dtype == "fp8" else torch.bfloat16

    def _cutedsl_indexer_supported(
        self, *, num_index_heads: int, page_size: int, decode_query_len: int
    ) -> bool:
        """Whether the CuTe DSL scorer accepts this step's geometry."""
        runner = cutedsl_score_runner()
        if runner is None:
            return False
        return bool(
            runner.is_supported(
                q_dtype=self._msa_index_kv_dtype(),
                num_heads=int(num_index_heads),
                # Pinned to MSA_REQUIRED_HEAD_DIM by the backend's constructor.
                head_dim=MSA_REQUIRED_HEAD_DIM,
                page_size=int(page_size),
                max_decode_query_len=int(decode_query_len),
            )
        )

    def _build_step_plans(self) -> None:
        """Build the layer-invariant fmha_sm100 plans this step still needs.

        Runs in prepare(), outside CUDA graph capture. The proxy, GQA, and
        dense plans depend only on the per-step sparse geometry (qo/kv lengths,
        head counts, topk, page size), never on the layer, so they are built
        once here and reused by every layer.

        Each plan covers only the rows fmha_sm100 still runs, per
        _msa_proxy_plan_rows and _msa_attn_plan_rows, which leaves a pure
        decode step nothing to plan: its attention is the decode kernels' and
        its block selection the CuTe DSL scorer's. That is what keeps the plans
        off the CUDA-graph path, since a captured step is a pure-decode one
        (_check_capture_is_pure_decode), and so lets them be plain tuples of
        per-step tensors.
        """
        self._msa_prefill_proxy_plan = None
        self._msa_prefill_gqa_plan = None
        self._msa_prefill_dense_plan = None
        self._msa_prefill_n_valid_blocks = None
        if not self._msa_fields_ready:
            return
        # Geometry is captured in __post_init__; skip when it is unavailable.
        params = self._msa_params
        if params is None:
            return
        num_index_heads = params.num_index_heads
        qo_lens_cpu = self.msa_qo_lens_cpu
        kv_lens_cpu = self.msa_kv_lens_cpu
        qo_offset_cpu = self.msa_qo_offset_cpu
        if qo_lens_cpu is None or kv_lens_cpu is None or qo_offset_cpu is None:
            return
        batch = int(qo_lens_cpu.shape[0])
        page_size = int(self.kv_cache_manager.tokens_per_block)
        if self._msa_runs_no_fmha():
            # Pure decode: no plan to build, but the scorer still writes its
            # scores into the proxy scratch and reads the valid-block count.
            self._ensure_msa_decode_scratch_buffers(
                num_index_heads=num_index_heads,
                max_batch=int(self.max_num_sequences),
                capture_graph=self.is_cuda_graph,
                # No proxy plan, so the worst case is the only bound available.
                required_max_k_tiles=self._msa_worst_case_max_k_tiles,
            )
            n_valid = per_token_valid_blocks(
                qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, causal=True, block_size=page_size
            )
            self.msa_n_valid_blocks[:batch].copy_(n_valid.to(torch.int32), non_blocking=True)
            return

        fmha_sm100 = require_msa_module()
        num_q_heads, num_kv_heads = params.sharded_head_counts(self.mapping)
        # The main-attention GQA and dense plans need use_fp8_kvcache so the
        # inline sparse-prefill kernel selection matches an FP8 paged cache.
        # The proxy runs over the bf16 index-K cache, so it never needs the
        # flag.
        use_fp8 = self._msa_main_kv_is_fp8()

        def plan_for(rows: Optional[Tuple[int, int]], **plan_kwargs) -> Optional[tuple]:
            """Plan one site over the given rows, or None where it has none.

            Slicing keeps the length tensors' pinned backing, so a single-phase
            plan stages as cheaply as a whole-batch one.
            """
            if rows is None:
                return None
            first, last = rows
            whole = (first, last) == (0, batch)
            return fmha_sm100.fmha_sm100_plan(
                qo_lens_cpu if whole else qo_lens_cpu[first:last],
                kv_lens_cpu if whole else kv_lens_cpu[first:last],
                qo_offset=qo_offset_cpu if whole else qo_offset_cpu[first:last],
                page_size=page_size,
                num_kv_splits=1,
                causal=True,
                **plan_kwargs,
            )

        # Proxy plan: MQA (num_kv_heads=1) max-score pass over the index
        # branch; output_maxscore feeds the indexer's top-k block selection.
        self._msa_prefill_proxy_plan = plan_for(
            self._msa_proxy_plan_rows(),
            num_qo_heads=num_index_heads,
            num_kv_heads=1,
            output_maxscore=True,
        )
        attn_rows = self._msa_attn_plan_rows()
        # Sparse layers: kv_block_num=topk limits attention to top-k blocks.
        self._msa_prefill_gqa_plan = plan_for(
            attn_rows,
            num_qo_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            kv_block_num=params.topk,
            use_fp8_kvcache=use_fp8,
        )
        # Dense layers: no kv_block_num, so the full page table is attended.
        self._msa_prefill_dense_plan = plan_for(
            attn_rows,
            num_qo_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            use_fp8_kvcache=use_fp8,
        )
        # Stage the valid-block count to the device once for the whole step
        # (see msa_prefill_n_valid_blocks).
        n_valid_host = per_token_valid_blocks(
            qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, causal=True, block_size=page_size
        )
        total_q = int(n_valid_host.shape[0])
        if total_q > 0:
            dev_buf = self._ensure_prefill_n_valid_buffer(total_q, _cache_device(self))
            dev_buf[:total_q].copy_(n_valid_host.to(torch.int32), non_blocking=True)
            self._msa_prefill_n_valid_blocks = dev_buf[:total_q]

    def _build_msa_fields(self) -> None:
        """Populate the MSA cache-write buffers for this step.

        The page table and per-new-token cache slots are derived via the
        build_paged_kv_slot_mapping helper, then copied into the persistent
        buffers. The transient builder tensors are discarded.
        """
        self._msa_fields_ready = False
        if not self._msa_buffers_ready:
            return
        request_ids = self.request_ids
        qo_lens_cpu = self.msa_qo_lens_cpu
        kv_lens_cpu = self.msa_kv_lens_cpu
        qo_offset_cpu = self.msa_qo_offset_cpu
        if request_ids is None or qo_lens_cpu is None:
            return
        batch_size = int(qo_lens_cpu.shape[0])
        if batch_size == 0:
            return

        kv_cache_manager = self.kv_cache_manager
        cache_device = _cache_device(self)
        page_size = int(kv_cache_manager.tokens_per_block)

        # Built in prepare() (outside capture), so these transients are
        # fine: forwards read only the persistent buffers filled below.
        # qo_offset is the prefix length, so one build covers prefill
        # (num_cached) and decode (kv_len - 1 with qo_len 1).
        mapping = build_paged_kv_slot_mapping(
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            qo_lens_cpu=qo_lens_cpu,
            qo_offset_cpu=qo_offset_cpu,
            device=cache_device,
        )
        out_cache_loc = mapping.out_cache_loc
        # Only fmha_sm100 reads the flattened page table (the decode kernels
        # index msa_block_table directly), so a step with no fmha_sm100 work
        # left skips building and staging it.
        needs_flat_page_table = not self._msa_runs_no_fmha()
        kv_indices = (
            # Comes from the same host block ids the mapping was built from,
            # so it costs no device work.
            build_kv_page_indices(mapping.block_ids_cpu, kv_lens_cpu, page_size)
            if needs_flat_page_table
            else None
        )
        total_new_tokens = int(out_cache_loc.shape[0])
        if total_new_tokens > self.msa_out_cache_loc.shape[0]:
            raise ValueError(
                f"MSA out_cache_loc buffer ({self.msa_out_cache_loc.shape[0]}) is "
                f"smaller than the step's new-token count ({total_new_tokens})."
            )
        if kv_indices is not None and int(kv_indices.shape[0]) > self.msa_kv_indices.shape[0]:
            raise ValueError(
                f"MSA kv_indices buffer ({self.msa_kv_indices.shape[0]}) is "
                f"smaller than the step's page count ({int(kv_indices.shape[0])})."
            )
        block_ids_cpu = mapping.block_ids_cpu
        block_table_cols = int(block_ids_cpu.shape[1])
        if block_table_cols > self.msa_block_table.shape[1]:
            raise ValueError(
                f"MSA block_table buffer ({self.msa_block_table.shape[1]} columns) is "
                f"smaller than the step's per-request page count ({block_table_cols})."
            )

        self.msa_out_cache_loc[:total_new_tokens].copy_(out_cache_loc, non_blocking=True)
        if kv_indices is not None:
            self.msa_kv_indices[: int(kv_indices.shape[0])].copy_(kv_indices, non_blocking=True)

        # 2-D page table and per-request length for the decode kernels,
        # from the same host block ids the flat page table was built from.
        # Columns past a request's page count are left stale rather than
        # cleared: every consumer bounds its walk by seq_lens.
        self.msa_block_table[:batch_size, :block_table_cols].copy_(
            maybe_pin_memory(block_ids_cpu.to(torch.int32)), non_blocking=True
        )
        self.msa_seq_lens_cuda[:batch_size].copy_(kv_lens_cpu, non_blocking=True)
        # Sub-page expansion for the trtllm-gen dense layers, staged once here
        # instead of once per layer, outside capture into a graph-stable
        # buffer as with the slot table above.
        if self.msa_subpage_block_table is not None:
            write_subpage_block_table(
                self.msa_block_table[:batch_size],
                self._msa_subpages_per_slot,
                self.msa_subpage_block_table[:batch_size],
            )
        self._msa_fields_ready = True

    def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
        """Return the paged index-K cache in the HND layout MSA consumes."""
        return self.kv_cache_manager.get_index_k_buffer(layer_idx, kv_layout="HND")

    def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
        """Write the new-token index-K into the side cache at out_cache_loc."""
        cache = self.msa_idx_k_cache(layer_idx)
        sparse_index_dim = int(cache.shape[-1])
        num_tokens = int(idx_k.shape[0])
        write_kv_slots(
            cache,
            self.msa_out_cache_loc[:num_tokens],
            idx_k.reshape(num_tokens, 1, sparse_index_dim),
            layout="HND",
        )

    def msa_proxy_max_score_view(
        self, num_index_heads: int, plan_max_k_tiles: int, num_tokens: int
    ) -> torch.Tensor:
        """Return a contiguous [num_index_heads, plan_max_k_tiles, num_tokens] view.

        fmha_sm100 ignores the passed tensor's strides and writes a contiguous
        [num_index_heads, plan_max_k_tiles, total_q] block sized by the current
        decode plan, so it must receive a tensor contiguous in exactly that
        shape. The view is taken from the flat store's prefix starting at offset
        0, so its data_ptr is stable for CUDA graph replay. Capture builds the
        decode plan at the worst-case max_k_tiles, so replays only shrink it.
        """
        store = self.msa_max_score
        if plan_max_k_tiles <= 0:
            raise ValueError(
                "The proxy max-score view has no block extent (max_k_tiles="
                f"{plan_max_k_tiles}). Both the fmha_sm100 proxy and the CuTe "
                "DSL scorer address it by block id, so a zero extent would put "
                "their writes past the end of the view."
            )
        numel = num_index_heads * plan_max_k_tiles * num_tokens
        if numel > store.numel():
            raise ValueError(
                f"msa_max_score backing store ({store.numel()} elements) is "
                f"smaller than the proxy view needs ({numel} = {num_index_heads} "
                f"heads * {plan_max_k_tiles} k-tiles * {num_tokens} tokens)."
            )
        return store[:numel].view(num_index_heads, plan_max_k_tiles, num_tokens)


class MiniMaxM3MsaSparseAttention(TrtllmAttention):
    """MSA-backed MiniMax-M3 sparse attention."""

    Metadata = MiniMaxM3MsaSparseAttentionMetadata

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
        self.indexer_kv_dtype = str(sparse_params.indexer_kv_dtype)
        self._validate_msa_preconditions()
        self.indexer = MsaIndexer(self.m3_config)

    def _validate_msa_preconditions(self) -> None:
        config = self.m3_config
        if not self.disable_index_value:
            raise NotImplementedError(
                "MSA backend requires disable_index_value=True; the proxy pass "
                "consumes only the max score and has no index-V path."
            )
        if config.head_dim != MSA_REQUIRED_HEAD_DIM:
            raise NotImplementedError(
                f"MSA backend requires head_dim={MSA_REQUIRED_HEAD_DIM}, got {config.head_dim}."
            )
        if config.sparse_index_dim != MSA_REQUIRED_HEAD_DIM:
            raise NotImplementedError(
                f"MSA backend requires sparse_index_dim={MSA_REQUIRED_HEAD_DIM}, "
                f"got {config.sparse_index_dim}."
            )
        if config.topk != MSA_REQUIRED_TOPK:
            raise NotImplementedError(
                f"MSA backend requires topk={MSA_REQUIRED_TOPK}, got {config.topk}."
            )

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]) -> None:
        """Build the FMHA manager, then require the MSA pair among its libraries.

        The base class defers this past __init__ when weight creation is
        deferred, and reruns it whenever the quant config lands, so the pair
        check belongs here rather than with the other preconditions.
        """
        super().update_quant_config(new_quant_config)
        self._validate_fmha_pair()

    def _validate_fmha_pair(self) -> None:
        """Require both halves of the MSA pair on this layer.

        MsaPrefillFmha serves the context phase and MsaDecodeFmha the
        generation phase, and neither will take the other's. Losing one, as a
        TLLM_FMHA_LIBS subset would, leaves that phase to a library that
        refuses it. Checking as they are built reports it once per layer rather
        than on the first step that happens to carry those rows.
        """
        present = {type(fmha).__name__ for fmha in self._fmha_manager.fmha_libs}
        missing = sorted({"MsaPrefillFmha", "MsaDecodeFmha"} - present)
        if missing:
            raise RuntimeError(
                f"MiniMax-M3 MSA attention layer {self.layer_idx} is missing the FMHA "
                f"{'library' if len(missing) == 1 else 'libraries'} {', '.join(missing)}. "
                "The two are a pair covering one phase each; enable both (msa_prefill "
                "and msa_decode) or none."
            )

    @classmethod
    def support_fused_rope(cls) -> bool:
        # The MiniMax-M3 model layer applies partial RoPE to the main and
        # index branches explicitly.
        return False

    def run_indexer(
        self,
        idx_q: torch.Tensor,
        idx_k: Optional[torch.Tensor],
        metadata,
        *,
        idx_sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Write the index-K cache and return the selected block indices.

        The model layer runs this before forward and threads the result through
        forward_args.sparse_backend_args. Returns [total_q, num_kv_heads, topk].
        The generation rows are scored by the CuTe DSL kernel and any context
        rows by the fmha_sm100 proxy pass, over the plan prepare() built.
        """
        config = self.m3_config
        idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5
        num_tokens = int(idx_q.shape[0])
        # The span says how the scoring is split between the CuTe DSL kernel
        # and the fmha_sm100 proxy pass.
        span = metadata.msa_decode_span
        # Preserve split column views without allowing an implicit copy. The
        # scorer and cache writer below both honor their source strides.
        idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
        idx_k_cache = metadata.msa_idx_k_cache(self.layer_idx)
        configured_for_fp8 = self.indexer_kv_dtype == "fp8"
        expected_cache_dtype = torch.float8_e4m3fn if configured_for_fp8 else torch.bfloat16
        if idx_k_cache.dtype != expected_cache_dtype:
            raise ValueError(
                "MiniMax-M3 index-K cache dtype does not match indexer_kv_dtype="
                f"{self.indexer_kv_dtype!r}: expected {expected_cache_dtype}, "
                f"got {idx_k_cache.dtype}."
            )
        if configured_for_fp8:
            if idx_q_view.dtype != torch.float8_e4m3fn or idx_k is not None:
                raise ValueError(
                    "The MiniMax-M3 FP8 indexer requires fused FP8 index-Q and "
                    "an already-populated index-K cache (live index-K must be None)."
                )
        else:
            if idx_q_view.dtype != torch.bfloat16 or idx_k is None or idx_k.dtype != torch.bfloat16:
                live_k_dtype = None if idx_k is None else idx_k.dtype
                raise ValueError(
                    "The MiniMax-M3 BF16 indexer requires BF16 index-Q and a live "
                    f"BF16 index-K tensor; got Q={idx_q_view.dtype}, K={live_k_dtype}."
                )
            idx_k_view = idx_k.view(num_tokens, 1, config.sparse_index_dim)
            metadata.msa_write_idx_k(self.layer_idx, idx_k_view)
        # The FP8 indexer mirrors vLLM's unscaled E4M3 contract: normalized
        # index Q/K are cast directly and the proxy accumulates their QK scores
        # in FP32. Block ordering is invariant to the omitted positive scale.
        # The fused production path arrives here with E4M3 Q and an already
        # populated cache; the BF16 path writes its live K above.

        # Inputs for the CuTe DSL scorer, which takes this step's generation
        # span. Left None on a pure-prefill step, which has no span, so the
        # proxy plan scores the whole batch instead. gen_first is the span's
        # first query token: the scorer takes [gen_first, num_tokens) over rows
        # [ctx_rows, row_last), the proxy the context prefix ahead of both.
        block_table = None
        seq_lens_cuda = None
        decode_query_len = None
        gen_first = 0
        ctx_rows = 0
        if span is not None:
            ctx_rows = span.row_first
            decode_query_len = span.query_len
            row_last = int(metadata.num_seqs)
            # Derived from the row count and the uniform query length, as
            # PhasedFmha derives the attention phase's token offset, so the
            # scorer and the decode kernels agree on the boundary.
            gen_first = num_tokens - decode_query_len * (row_last - ctx_rows)
            block_table = metadata.msa_block_table[ctx_rows:row_last]
            seq_lens_cuda = metadata.msa_seq_lens_cuda[ctx_rows:row_last]
        # One selection path, over the scratch prepare() staged for whichever
        # scorer owns the rows: a span starting at row 0 leaves the CuTe DSL
        # kernel every row and the graph-stable valid-block buffer, anything
        # else keeps a proxy plan and the per-step count. When neither is
        # present (a standalone test that skips prepare) select_blocks plans
        # inline and computes the valid-block count itself.
        proxy_plan = metadata.msa_prefill_proxy_plan
        if proxy_plan is None and span is not None and span.row_first == 0:
            # No proxy plan to read max_k_tiles from for the contiguous score
            # view, so it is shaped to the worst case, which the scorer
            # accepts: it takes every score stride at runtime.
            max_score = metadata.msa_proxy_max_score_view(
                config.num_index_heads, metadata.msa_worst_case_max_k_tiles, num_tokens
            )
            n_valid_blocks = metadata.msa_n_valid_blocks[:num_tokens]
        else:
            n_valid_blocks = metadata.msa_prefill_n_valid_blocks
            if n_valid_blocks is not None:
                n_valid_blocks = n_valid_blocks[:num_tokens]
            # The scorer fills the buffer it is handed, shaped to the span's
            # tokens alone: the proxy writes its own half as a contiguous
            # [heads, k_tiles, tokens] block (see msa_proxy_max_score_view) and
            # so cannot take a slice of this one. The span's tokens are at most
            # a decode step's worth, which is what the store was sized for.
            max_score = (
                metadata.msa_proxy_max_score_view(
                    config.num_index_heads,
                    metadata.msa_worst_case_max_k_tiles,
                    num_tokens - gen_first,
                )
                if span is not None
                else None
            )
        return self.indexer.select_blocks(
            idx_q_view,
            idx_k_cache,
            idx_sm_scale=idx_sm_scale,
            kv_indices=metadata.msa_kv_indices,
            qo_lens_cpu=metadata.msa_qo_lens_cpu,
            kv_lens_cpu=metadata.msa_kv_lens_cpu,
            qo_offset_cpu=metadata.msa_qo_offset_cpu,
            proxy_plan=proxy_plan,
            max_score=max_score,
            n_valid_blocks=n_valid_blocks,
            require_cutedsl=span is not None,
            block_table=block_table,
            seq_lens_cuda=seq_lens_cuda,
            decode_query_len=decode_query_len,
            gen_token_first=gen_first,
            ctx_rows=ctx_rows,
        )

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata,
        forward_args: "AttentionForwardArgs",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # The model layer runs run_indexer and passes the selected blocks
        # through the sparse backend payload.
        sparse_backend_args = forward_args.sparse_backend_args
        topk_indices = sparse_backend_args.topk_indices if sparse_backend_args is not None else None
        return topk_indices, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata,
        forward_args: "AttentionForwardArgs",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None


__all__ = [
    "MiniMaxM3MsaSparseAttention",
    "MiniMaxM3MsaSparseAttentionMetadata",
]
