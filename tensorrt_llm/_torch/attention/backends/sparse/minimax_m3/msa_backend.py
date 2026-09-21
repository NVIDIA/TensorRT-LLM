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
  * With Eagle3 a decode row has 1 + draft_len query tokens. Slots and
    valid-block counts are per token, and the decode kernels take that uniform
    query length from msa_decode_span. on_update_kv_lens re-derives the
    per-request lengths, the slots and the counts after the overlap scheduler
    corrects kv_lens on device.

The classes subclass TrtllmAttention and TrtllmAttentionMetadata, imported at
module scope. That is cycle-free only because the dependency runs one way: the
two FMHA libraries reach the kernels through ...minimax_m3.kernels and never
load this module, so trtllm's import chain does not come back here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.bindings import DataType
from tensorrt_llm.models.modeling_utils import QuantConfig

from .common import (
    MiniMaxM3SparseConfig,
    MiniMaxM3SparseMetadataParams,
    build_paged_kv_slot_mapping,
    write_kv_slots,
)
from .kernels.msa_utils import (
    MSA_REQUIRED_HEAD_DIM,
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    per_token_valid_blocks,
    require_msa_module,
)
from .kernels.trtllm_gen_dense_decode import (
    dense_decode_unsupported_reason,
    uniform_subpages_per_slot,
    write_subpage_block_table,
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

    With Eagle3 a decode row has 1 + draft_len query tokens: the span's
    query_len is that count, slots and valid-block counts are per token, and
    the overlap scheduler corrects kv_lens on device after prepare(). The
    decode kernels read their lengths from msa_seq_lens_cuda, so
    on_update_kv_lens patches that buffer, the slots and the counts. The
    fmha_sm100 plans cover context rows only, whose lengths the correction
    never touches, so they need no patch.
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
    msa_cu_q_lens: Optional[torch.Tensor] = None
    msa_cu_kv_lens: Optional[torch.Tensor] = None
    # msa_block_table with each slot expanded into the K and V sub-pages the
    # trtllm-gen dense kernel indexes. _msa_subpages_per_slot is the expansion
    # factor, or 0 where the pool has no single one; see msa_subpage_rows.
    msa_subpage_block_table: Optional[torch.Tensor] = None
    _msa_subpages_per_slot: int = 0
    # Inputs for on_update_kv_lens: each query token's request row and offset
    # within the request, and the kv_lens prepare() staged (the upper bound the
    # correction is clamped to). The write slots are re-derived from
    # msa_block_table, so no per-token slot table is kept.
    msa_q_batch_row: Optional[torch.Tensor] = None
    msa_q_intra: Optional[torch.Tensor] = None
    msa_kv_lens_staged: Optional[torch.Tensor] = None

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
    # True only with speculative decoding; otherwise on_update_kv_lens and its
    # staging are skipped and non-speculative steps run as before.
    _msa_kv_lens_dynamic: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
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
        # The host kv_lens carries num_extra_kv_tokens (speculative draft
        # slots); MSA needs the attended length, which is what kv_lens_cuda
        # holds. Every consumer below, msa_seq_lens_cuda included, inherits
        # the correction from here.
        params = self.kv_cache_params
        extra = int(params.num_extra_kv_tokens) if params is not None else 0
        if kv_lens is not None and extra:
            kv_lens = kv_lens - extra
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

    def _msa_main_kv_is_fp8(self) -> bool:
        """Whether the main paged K/V cache is stored as FP8 E4M3.

        The GQA and dense plans must pass use_fp8_kvcache so the inline
        sparse-prefill path selects the FP8 AOT kernels. Mirrors the k_paged
        dtype check in run_msa_prefill_gqa.
        """
        kv_cache_manager = self.kv_cache_manager
        return kv_cache_manager is not None and kv_cache_manager.dtype in (
            DataType.FP8,
            DataType.NVFP4,
        )

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
        # 1 + draft_len query tokens per generation request under speculative
        # decoding, one otherwise; see msa_decode_span. The scorer tiles
        # num_index_heads * query tokens into one Q block, which bounds the
        # draft length it can verify.
        decode_query_len = self._msa_max_decode_query_len()
        num_index_heads = params.sharded_index_head_count(self.mapping)
        if not self._cutedsl_indexer_supported(
            num_index_heads=num_index_heads,
            page_size=page_size,
            decode_query_len=decode_query_len,
        ):
            raise RuntimeError(
                "The MiniMax-M3 CuTe DSL indexer scorer does not support this "
                f"configuration: {num_index_heads} index heads, page size "
                f"{page_size}, index dtype {self._msa_index_kv_dtype()}, up to "
                f"{decode_query_len} query tokens per generation request."
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
        if getattr(kv_cache_manager, "dtype", None) == DataType.NVFP4:
            self.msa_cu_q_lens = self.get_empty(
                buffers,
                (max_num_sequences + 1,),
                cache_name="msa_cu_q_lens",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.msa_cu_kv_lens = self.get_empty(
                buffers,
                (max_num_sequences + 1,),
                cache_name="msa_cu_kv_lens",
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
        # Inputs for on_update_kv_lens.
        self.msa_q_batch_row = self.get_empty(
            buffers,
            (max_num_tokens,),
            cache_name="msa_q_batch_row",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.msa_q_intra = self.get_empty(
            buffers,
            (max_num_tokens,),
            cache_name="msa_q_intra",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.msa_kv_lens_staged = self.get_empty(
            buffers,
            (max_num_sequences,),
            cache_name="msa_kv_lens_staged",
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
                num_index_heads=params.sharded_index_head_count(self.mapping),
                kv_cache_manager=kv_cache_manager,
                max_batch=max_num_sequences,
            )
            self._msa_worst_case_max_k_tiles = int(max_k_tiles)
            self._alloc_msa_proxy_scratch(
                num_index_heads=params.sharded_index_head_count(self.mapping),
                max_tokens=self._msa_max_decode_tokens(),
                max_k_tiles=max_k_tiles,
                capture_graph=capture_graph,
            )
        self._msa_buffers_ready = True

    def _msa_max_decode_query_len(self) -> int:
        """Worst-case query tokens per generation request: 1 + draft_len.

        The KV cache manager knows the speculative config when this metadata is
        built (__post_init__ runs before update_spec_dec_param), so it is the
        one source for both the up-front kernel validation and the scratch
        sizing. A run without speculative decoding reports 1.
        """
        draft_len = int(getattr(self.kv_cache_manager, "max_total_draft_tokens", 0) or 0)
        return 1 + max(0, draft_len)

    def _msa_max_decode_tokens(self) -> int:
        """Worst-case query tokens in one decode step: 1 + draft_len per row.

        Sizes the proxy scratch the CuTe DSL scorer writes one column per query
        token into, and the per-token valid-block buffer beside it. Capped at
        16384, which also keeps a whole-batch fmha_sm100 proxy plan under its
        65536 total_q * num_qo_heads limit with 4 index heads.
        """
        max_seqs = int(self.max_num_sequences)
        tokens = max_seqs * self._msa_max_decode_query_len()
        max_toks = int(self.max_num_tokens or 0)
        if max_toks > 0:
            tokens = min(tokens, max_toks)
        return max(max_seqs, min(tokens, 16384))

    def _alloc_msa_proxy_scratch(
        self,
        *,
        num_index_heads: int,
        max_tokens: int,
        max_k_tiles: int,
        capture_graph: bool,
    ) -> None:
        """Allocate the flat proxy max-score store and the valid-block scratch.

        Sized for the worst-case max_k_tiles and query-token count (more than
        the batch size under speculative verify), so one allocation serves every
        decode step. msa_proxy_max_score_view slices the per-step shape.
        """
        buffers = self.cuda_graph_buffers
        self.msa_max_score = self.get_empty(
            buffers,
            (num_index_heads * max_k_tiles * max_tokens,),
            cache_name="msa_max_score",
            dtype=torch.float32,
            capture_graph=capture_graph,
        )
        self.msa_n_valid_blocks = self.get_empty(
            buffers,
            (max_tokens,),
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
        max_tokens = max(int(max_batch), self._msa_max_decode_tokens())
        required_numel = num_index_heads * required_max_k_tiles * max_tokens
        if self.msa_max_score is not None:
            if self.msa_max_score.numel() < required_numel:
                raise ValueError(
                    f"msa_max_score backing store ({self.msa_max_score.numel()} "
                    f"elements) is smaller than the decode plan needs "
                    f"({required_numel} = {num_index_heads} heads * "
                    f"{required_max_k_tiles} k-tiles * {max_tokens} tokens)."
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
            max_tokens=max_tokens,
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
        id from, so a batch without it is rejected rather than served. Under
        speculative decoding that length is the verify window, 1 + draft_len,
        and may not exceed what _validate_decode_kernel_support settled the
        scorer for.
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
        max_query_len = self._msa_max_decode_query_len()
        if qo_max > max_query_len:
            raise RuntimeError(
                "MiniMax-M3 MSA attention validated its decode kernels for at most "
                f"{max_query_len} query tokens per generation request, but rows "
                f"[{row_first}, {row_last}) carry up to {qo_max}."
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

    def _msa_kv_lens_may_change(self) -> bool:
        """Whether kv_lens can change after prepare(): only with speculative decoding.

        max_total_draft_tokens is set by the engine's update_spec_dec_param; the
        other checks mirror TrtllmAttentionMetadata's spec_active.
        """
        params = self.kv_cache_params
        runtime_features = self.runtime_features
        return bool(
            self.max_total_draft_tokens
            or self.draft_kv_cache_manager is not None
            or self.is_spec_decoding_enabled
            or (params is not None and params.num_extra_kv_tokens)
            or (runtime_features is not None and runtime_features.has_speculative_draft_tokens)
        )

    def on_update_kv_lens(self) -> None:
        """Re-derive lengths, slots and valid-block counts from the corrected kv_lens_cuda.

        The overlap scheduler shortens kv_lens on device after prepare() staged
        full-acceptance values (CUDA-graph warmup restores them the same way
        between forwards). Shrinking keeps the staged page table valid, so only
        the per-row lengths and what derives from them are patched; the clamp
        to msa_kv_lens_staged enforces that. Device-only, capture-safe and
        idempotent; skipped without speculative decoding.

        The correction updates msa_seq_lens_cuda,
        which the CuTe DSL scorer, the Triton sparse decode and the trtllm-gen
        dense decode all read their lengths from; msa_out_cache_loc, the K/V
        and index-K write slots; and the per-token valid-block count the top-k
        selection is bounded by, on whichever buffer this step staged it.
        NVFP4 CSR attention also needs corrected cumulative KV lengths when
        extend_ctx promotes speculative generation rows into the context prefix.
        Staged host bounds remain valid upper bounds as lengths shrink.
        """
        super().on_update_kv_lens()
        if not self._msa_fields_ready or not self._msa_kv_lens_dynamic:
            return
        batch = int(self.num_seqs)
        total_q = int(self.num_tokens)
        if batch <= 0 or total_q <= 0:
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
        num_index_heads = params.sharded_index_head_count(self.mapping)
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
            # One entry per query token: batch * (1 + draft_len) under
            # speculative decoding, batch otherwise.
            n_valid = per_token_valid_blocks(
                qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, causal=True, block_size=page_size
            )
            total_q = int(n_valid.shape[0])
            self.msa_n_valid_blocks[:total_q].copy_(n_valid.to(torch.int32), non_blocking=True)
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
        buffers. The transient builder tensors are discarded. With speculative
        decoding the inputs for on_update_kv_lens are staged as well.
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
        # (num_cached) and decode (kv_len - qo_len).
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
        if self.msa_cu_q_lens is not None:
            self._msa_live_batch = batch_size
            self.msa_cu_q_lens[0].zero_()
            self.msa_cu_kv_lens[0].zero_()
            torch.cumsum(
                maybe_pin_memory(qo_lens_cpu).to(cache_device, non_blocking=True),
                0,
                out=self.msa_cu_q_lens[1 : batch_size + 1],
            )
            torch.cumsum(
                self.msa_seq_lens_cuda[:batch_size],
                0,
                out=self.msa_cu_kv_lens[1 : batch_size + 1],
            )
            self._msa_max_q_len = int(qo_lens_cpu.max().item())
            self._msa_max_kv_len_all = int(kv_lens_cpu.max().item())
            self._msa_total_k = int(kv_lens_cpu.to(torch.int64).sum().item())
            self._msa_total_k_rows = int(
                torch.div(
                    kv_lens_cpu.to(torch.int64) + page_size - 1,
                    page_size,
                    rounding_mode="floor",
                )
                .sum()
                .item()
            )
            # The same four bounds over the context prefix alone, which is what an
            # NVFP4 sparse layer's CSR kernel covers once the ported decode kernels
            # take the generation suffix. Sizing that call from the whole batch
            # would let a long generation row inflate its worklist.
            context_rows = min(int(self.num_contexts or 0), batch_size)
            if context_rows > 0:
                prefix_kv = kv_lens_cpu[:context_rows].to(torch.int64)
                self._msa_context_prefix_bounds = (
                    int(qo_lens_cpu[:context_rows].max().item()),
                    int(prefix_kv.max().item()),
                    int(prefix_kv.sum().item()),
                    int(
                        torch.div(prefix_kv + page_size - 1, page_size, rounding_mode="floor")
                        .sum()
                        .item()
                    ),
                )
            else:
                self._msa_context_prefix_bounds = (0, 0, 0, 0)
        # Sub-page expansion for the trtllm-gen dense layers, staged once here
        # instead of once per layer, outside capture into a graph-stable
        # buffer as with the slot table above.
        if self.msa_subpage_block_table is not None:
            write_subpage_block_table(
                self.msa_block_table[:batch_size],
                self._msa_subpages_per_slot,
                self.msa_subpage_block_table[:batch_size],
            )

        self._msa_kv_lens_dynamic = self._msa_kv_lens_may_change()
        if not self._msa_kv_lens_dynamic:
            self._msa_fields_ready = True
            return

        # Inputs for on_update_kv_lens: each query token's (request row, offset
        # in request). Pinned and non-blocking so the copies do not synchronize
        # the stream. The slots themselves come from msa_block_table above.
        qo_long = qo_lens_cpu.to(torch.long)
        batch_row_cpu = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32), qo_long
        )
        starts = torch.cumsum(qo_long, 0) - qo_long
        intra_cpu = (
            torch.arange(total_new_tokens, dtype=torch.int64)
            - torch.repeat_interleave(starts, qo_long)
        ).to(torch.int32)
        self.msa_q_batch_row[:total_new_tokens].copy_(
            maybe_pin_memory(batch_row_cpu), non_blocking=True
        )
        self.msa_q_intra[:total_new_tokens].copy_(maybe_pin_memory(intra_cpu), non_blocking=True)
        # The staged lens are the upper bound on_update_kv_lens clamps to: the
        # same attended lengths msa_seq_lens_cuda was just filled from, kept
        # apart so the clamp stays fixed however often the hook runs.
        self.msa_kv_lens_staged[:batch_size].copy_(kv_lens_cpu, non_blocking=True)
        self._msa_fields_ready = True

    def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
        """Return the paged index-K cache in the HND layout MSA consumes."""
        return self.kv_cache_manager.get_index_k_buffer(layer_idx)

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

    def write_layer_caches(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_k: Optional[torch.Tensor],
        metadata,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
    ) -> None:
        """Write this layer's new-token K, V and (bf16 indexer) index-K.

        One fused kernel launch when the source/cache layouts allow it, else
        the legacy per-cache writes. The model layer calls this first, so the
        index-K cache is populated before run_indexer's proxy pass reads it,
        and then hands forward() k=v=None: write_msa_phase_kv writes nothing
        for a phase without live K/V, so neither FMHA library repeats the
        write. `idx_k` is None on the FP8 indexer path, where the fused
        producer has already inserted E4M3 index-K into the side cache.
        `metadata` only supplies the step's write slots (msa_out_cache_loc,
        filled by prepare()) and the cache manager.
        """
        from .kernels.msa_scatter import fused_write_layer_caches, fused_write_layer_caches_nvfp4

        layer_idx = self.layer_idx
        manager = metadata.kv_cache_manager
        if getattr(manager, "is_nvfp4_layer", lambda _: False)(layer_idx):
            if kv_scale_orig_quant is None:
                raise RuntimeError("NVFP4 cache writes require quantization scales")
            buffers = manager.get_buffers(layer_idx, "HND")
            scales = manager.get_block_scale_buffers(layer_idx, "HND")
            idx_cache = metadata.msa_idx_k_cache(layer_idx) if idx_k is not None else None
            if not fused_write_layer_caches_nvfp4(
                buffers[:, 0],
                buffers[:, 1],
                scales[:, 0],
                scales[:, 1],
                idx_cache,
                metadata.msa_out_cache_loc[: k.shape[0]],
                k,
                v,
                idx_k,
                kv_scale_orig_quant,
            ):
                raise RuntimeError(
                    "NVFP4 cache writes require CUDA HND cache views and FP32 scales"
                )
            return
        buffers = metadata.kv_cache_manager.get_buffers(layer_idx, kv_layout="HND")
        k_view, v_view = buffers[:, 0], buffers[:, 1]
        idx_cache = metadata.msa_idx_k_cache(layer_idx) if idx_k is not None else None
        num_tokens = int(k.shape[0])
        out_cache_loc = metadata.msa_out_cache_loc[:num_tokens]
        if fused_write_layer_caches(k_view, v_view, idx_cache, out_cache_loc, k, v, idx_k):
            return
        num_kv_heads = int(k_view.shape[1])
        head_dim = int(k_view.shape[3])
        write_kv_slots(
            k_view,
            out_cache_loc,
            k.reshape(num_tokens, num_kv_heads, head_dim),
            layout="HND",
        )
        write_kv_slots(
            v_view,
            out_cache_loc,
            v.reshape(num_tokens, num_kv_heads, head_dim),
            layout="HND",
        )
        if idx_k is not None:
            write_kv_slots(
                idx_cache,
                out_cache_loc,
                idx_k.reshape(num_tokens, 1, int(idx_cache.shape[-1])),
                layout="HND",
            )

    def run_indexer(
        self,
        idx_q: torch.Tensor,
        idx_k: Optional[torch.Tensor],
        metadata,
        *,
        idx_sm_scale: Optional[float] = None,
        idx_k_prewritten: bool = False,
    ) -> torch.Tensor:
        """Write the index-K cache and return the selected block indices.

        The model layer runs this before forward and threads the result through
        forward_args.sparse_backend_args. Returns [total_q, num_kv_heads, topk].
        The generation rows are scored by the CuTe DSL kernel and any context
        rows by the fmha_sm100 proxy pass, over the plan prepare() built.
        `idx_k_prewritten` marks that the fused per-layer cache write
        (write_layer_caches) already stored this layer's index-K.
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
            # The fused per-layer write (write_layer_caches, signalled by
            # idx_k_prewritten) may already have stored this live bf16 index-K
            # ahead of the proxy pass; write it here only when it did not.
            if not idx_k_prewritten:
                idx_k_view = idx_k.view(num_tokens, 1, config.sparse_index_dim)
                metadata.msa_write_idx_k(self.layer_idx, idx_k_view)
        # The FP8 indexer mirrors vLLM's unscaled E4M3 contract: normalized
        # index Q/K are cast directly and the proxy accumulates their QK scores
        # in FP32. Block ordering is invariant to the omitted positive scale.
        # The fused production path arrives here with E4M3 Q and an already
        # populated cache; the BF16 path writes its live K above unless the
        # fused per-layer write already did.

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
