# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, FrozenSet, List, Optional, Tuple

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch import decode_switches
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.pyexecutor.input_commit import (
    InputCommit,
    active_input_commit,
    flush_active_input_commit,
)
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
from tensorrt_llm.logger import logger

from ..params import use_self_sampling_gvr
from .cache_manager import is_dsa_cache_manager
from .indexer import (
    _DG_SCHEDULE_BLOCK_KV,
    Indexer,
    IndexerPrefillChunkMetadata,
    _atom_ctx_offsets,
    _compute_slot_mappings,
    _effective_compress_ratio_divisor,
    _pick_dsl_split,
    _select_indexer_compress_ratio,
    use_cute_dsl_paged_mqa_logits,
)
from .kernels import (
    fused_dsa_decode_metadata,
    fused_dsa_helix_decode_metadata,
    helix_zero_or_empty_rows,
)
from .params import DSAMetadataParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

# Indexer MQA-logits are currently always fp32. The dtype is part of the
# CuTe DSL Top-K compile key, so warmup must use the runtime dtype.
_INDEXER_LOGITS_DTYPE = torch.float32
_MAX_RADIX_BLOCKS_PER_ROW = 10

if TYPE_CHECKING:
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager


@functools.lru_cache(maxsize=1)
def _fused_dsa_meta_enabled() -> bool:
    """Read the fused-DSA-metadata env gate (once, process-constant).

    Enabled by default. Set ``TRTLLM_DISABLE_FUSED_DSA_METADATA`` to any value
    other than "0" to keep the eager slot-mapping + gen-indptr chain instead of
    the fused Triton launch in kernels.py.

    Cached so the gate is fixed for the whole process: a mid-run flip would
    otherwise let the first fused launch happen inside CUDA-graph capture,
    voiding the "pre-compiled at warmup" capture-safety guarantee.
    """
    return os.environ.get("TRTLLM_DISABLE_FUSED_DSA_METADATA", "0") == "0"


def _fused_dsa_helix_meta_enabled() -> bool:
    """Build a Helix CP generation step's DSA decode metadata with one Triton launch
    (``fused_dsa_helix_decode_metadata``) instead of the eager chain and the Helix ownership
    rewrite: switchboard field ``dsa_helix_fused_meta`` (set by a target's generation profile). Set before
    the model is built, so the first launch happens at warm-up, before graph capture."""
    return decode_switches.active().dsa_helix_fused_meta


@functools.lru_cache(maxsize=1)
def _dg_sm100_schedule_in_kernel() -> bool:
    """Whether the fused Helix decode metadata also writes the DeepGEMM paged-MQA-logits schedules
    (``kernels._dg_sm100_schedule``): SM 10.x, where ``get_paged_mqa_logits_metadata`` runs
    DeepGEMM's SM100 schedule."""
    return get_sm_version() // 10 == 10


def _indexer_k_dtype(kv_cache_manager) -> str:
    """The indexer K dtype of a DSA-compatible cache manager: its ``indexer_k_dtype`` when it has
    one, else ``"fp4"`` or ``"fp8"`` from its ``use_fp4`` flag."""
    indexer_k_dtype = getattr(kv_cache_manager, "indexer_k_dtype", None)
    if indexer_k_dtype is not None:
        return indexer_k_dtype
    return "fp4" if getattr(kv_cache_manager, "use_fp4", False) else "fp8"


def build_req_idx_per_token(seq_lens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Map each flattened-batch token to its request index."""
    cu_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    token_idx = torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int32)
    return torch.searchsorted(cu_seq_lens, token_idx, right=True)


@dataclass(frozen=True)
class TokenMajorGenView:
    """One attention row per generation query token.

    Only the MLA RoPE and sparse-MLA generation ops consume this view. The
    request-major metadata remains unchanged for every other consumer.
    """

    num_rows: int
    sequence_length: torch.Tensor
    host_past_key_value_lengths: torch.Tensor
    host_context_lengths: torch.Tensor
    prompt_lens_cuda: torch.Tensor
    host_request_types: torch.Tensor
    kv_cache_block_offsets: torch.Tensor
    # Static row ceiling, not this step's row count. The attention op caches
    # workspace by max_num_requests, so a per-step value would thrash it.
    max_num_rows: int


@dataclass(init=False)
class DSAtrtllmAttentionMetadata(TrtllmAttentionMetadata):
    """Attention metadata for DSA (Dense Sparse Attention) with indexer state."""

    sparse_metadata_params: Optional[DSAMetadataParams] = None
    use_fp8_ds_mla: bool = field(default=False, init=False)
    # Store reference to indexer for preparation stage
    indexer: Optional["Indexer"] = None
    # Chunked prefill metadata for indexer (prefill-only, no CUDA graph needed)
    indexer_prefill_chunks: Optional[List[IndexerPrefillChunkMetadata]] = None
    # (local q_begin, local q_end, per-rank sizes) when indexer_prefill_chunks
    # cover only this TP rank's share of the context query tokens; the top-k
    # results are then all-gathered once after the chunks.
    indexer_prefill_q_split: Optional[Tuple[int, int, List[int]]] = None
    # Max chunk size for two-level chunking:
    # 1. Request-level: Pack multiple small requests into one chunk (up to indexer_max_chunk_size)
    # 2. Intra-request: Split large requests into Q-blocks when seq_len > max_chunk_size
    indexer_max_chunk_size: int
    # TopK for static token sparse attention
    num_sparse_topk: int
    # TopK for dynamic sparse MLA
    sparse_mla_topk: int
    radix_aux_indices: Optional[torch.Tensor] = field(default=None, init=False)
    radix_aux_logits: Optional[torch.Tensor] = field(default=None, init=False)
    # max number of draft tokens
    max_draft_tokens: int = 0
    # Indexer head dimension
    indexer_head_dim: int = 128
    # Elements per four-byte scale word, matching Indexer.prepare().
    indexer_quant_block_size: int = 128
    # Enable indexer skip for short sequences
    enable_indexer_skip: bool = False
    shared_topk_indices: Optional[torch.Tensor] = None
    indexer_skip_topk: bool = False
    in_mtp_draft_loop: bool = False
    mtp_num_accepted: Optional[torch.Tensor] = None
    # Whether skip the indexer for context requests
    skip_indexer_for_ctx_reqs: bool = False
    # Whether skip the indexer for generation requests
    skip_indexer_for_gen_reqs: bool = False
    # Whether to use the expanded buffers for MTP support
    use_expanded_buffers_for_mtp: bool = False
    # Whether to reshape the DSL paged MQA logits Q tensor into a kernel-
    # supported `effective_next_n` via caller-side atom-split (FP4: {1,2,3};
    # FP8: {1,2,3,4}; see `_pick_dsl_expand`). Reuses
    # `kv_lens_expanded_cuda` / `block_table_expanded` /
    # `scheduler_metadata_buffer_expanded`; runtime mutually exclusive with
    # `use_expanded_buffers_for_mtp` (the latter requires `not _use_dsl`).
    expand_for_dsl: bool = False
    # Cached (expand_factor, atom) decision from the wave-aware picker. Set at
    # `prepare()` time and read by forward call sites — avoids re-running the
    # picker per call and guarantees prepare/forward use the SAME decision
    # (otherwise the populated buffers would mismatch the kernel reshape).
    dsl_expand_factor: int = 1
    dsl_atom: int = 1
    # In-kernel uneven split (``_pick_dsl_split``): atom count, the window it
    # was picked for, and the per-atom causal stagger on device. Its schedule
    # reuses ``kv_lens_expanded_cuda`` / ``scheduler_metadata_buffer_expanded``.
    dsl_kernel_atoms: int = 1
    dsl_kernel_next_n: int = 0
    dsl_atom_ctx_offsets: Optional[torch.Tensor] = None
    dsl_atom_ctx_offsets_key: Tuple[int, int] = (0, 0)
    # Compression ratio for KV tokens
    compress_ratios: List[int] = field(default_factory=lambda: [1])
    # Number of compressed KV tokens for context requests
    num_ctx_kv_tokens: int = 0
    gen_indexer_kv_lens_cuda_runtime: Optional[torch.Tensor] = None
    # This layer's Helix combine mask when the key-sharded merge wrote it with the kept counts, else None.
    _helix_combine_mask: Optional[torch.Tensor] = field(default=None, init=False)
    # Rebound by on_update_kv_lens(); the decode graph recomputes it.
    graph_temporary_attrs: ClassVar[FrozenSet[str]] = frozenset(
        {"gen_indexer_kv_lens_cuda_runtime"}
    )
    # Temporal-GVR prior state: allocated only when the two-level dispatch
    # selects the temporal engine (the self-sampling engine keeps no
    # cross-step state).
    needs_gvr_prior: bool = field(default=False, init=False)
    gvr_prior_indices: Optional[torch.Tensor] = field(default=None, init=False)
    # Query tokens per generation request this step, in batch order. None
    # denotes the uniform runtime stride.
    ragged_verify_lens: Optional[List[int]] = None
    # Allocate the row-major and token-major buffers used only by ragged
    # verification. This is fixed before CUDA-graph metadata is created.
    enable_ragged_verification: bool = False
    # In device-window mode the host split determines only the captured shape;
    # the true windows are installed before replay.
    device_windows_mode: bool = False
    # Query tokens per generation request for a uniform runtime tier. Zero
    # falls back to the static draft-token ceiling.
    runtime_tokens_per_gen_step: int = 0

    @property
    def is_ragged_verify(self) -> bool:
        return self.ragged_verify_lens is not None

    def _validate_ragged_verification(self) -> None:
        """Reject layouts whose stable buffers were not enabled at allocation."""
        if self.ragged_verify_lens is not None and not self.enable_ragged_verification:
            raise RuntimeError(
                "ragged verification metadata was installed without enabling "
                "ragged-verification buffers"
            )

    @property
    def gen_token_stride(self) -> int:
        stride = int(self.runtime_tokens_per_gen_step)
        return stride if stride > 0 else 1 + self.max_draft_tokens

    def update_indexer_kv_lens_2d(self, gen_kv_lens: torch.Tensor) -> torch.Tensor:
        """Fill and return the active query-width view of persistent storage."""
        width = min(self.gen_token_stride, self.kv_lens_cuda_2d.shape[1])
        context_lens = self.kv_lens_cuda_2d[: self.num_generations, :width]
        context_lens.copy_(gen_kv_lens.unsqueeze(-1).expand(-1, width))
        return context_lens

    def mla_prepare_scheduler_buffers(self, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute prefixes for the same rows presented to MLA generation."""
        view = self.token_major_gen_view()
        if view is None:
            return super().mla_prepare_scheduler_buffers(num_heads)

        num_ctx = self.num_contexts
        n_gen = view.num_rows - num_ctx
        if not self._mla_scheduler_buffers_valid:
            # RoPE presents one query token per generation row. The device
            # extents are causal, pre-extra-KV lengths, just like the
            # cache_seq_lens scanned by the non-precomputed kernel path.
            torch.cumsum(
                view.prompt_lens_cuda[num_ctx:],
                0,
                dtype=torch.int32,
                out=self.ragged_mla_cu_q_rows[1 : n_gen + 1],
            )
            self.ragged_mla_cu_q_rows[1 : n_gen + 1] *= num_heads
            torch.cumsum(
                view.sequence_length[num_ctx:],
                0,
                dtype=torch.int32,
                out=self.ragged_mla_cu_kv_seqlens[1 : n_gen + 1],
            )
            self._mla_scheduler_buffers_valid = True
        return (
            self.ragged_mla_cu_q_rows[: n_gen + 1],
            self.ragged_mla_cu_kv_seqlens[: n_gen + 1],
        )

    def gen_token_repeat_list(self) -> List[int]:
        if not self.is_ragged_verify:
            return [self.gen_token_stride] * self.num_generations
        return list(self.ragged_verify_lens)

    def gen_token_repeats(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.tensor(self.gen_token_repeat_list(), dtype=torch.long, device=device)

    def expand_per_gen_token(self, values: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, int]:
        """Repeat request values once per generation query token."""
        repeats = self.gen_token_repeat_list()
        num_tokens = sum(repeats)
        if not self.is_ragged_verify:
            return values.repeat_interleave(self.gen_token_stride, dim=dim), num_tokens
        if values.device == self.gen_token_repeats_cuda.device:
            repeats_dev = self.gen_token_repeats_cuda[: self.num_generations]
        else:
            repeats_dev = torch.tensor(repeats, dtype=torch.long, device=values.device)
        return (
            values.repeat_interleave(repeats_dev, dim=dim, output_size=num_tokens),
            num_tokens,
        )

    def __init__(self, *args, **kwargs):
        """Initialize DSA metadata with SM count and indexer chunk size."""
        self.enable_ragged_verification = bool(kwargs.pop("enable_ragged_verification", False))
        sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        if (
            kwargs.get("sparse_metadata_params") is None
            and sparse_attention_config is not None
            and hasattr(sparse_attention_config, "to_sparse_metadata_params")
        ):
            kwargs["sparse_metadata_params"] = sparse_attention_config.to_sparse_metadata_params()
        self.num_sms = tensorrt_llm.deep_gemm.get_num_sms()
        # Cached step-invariant values for transform_local_topk_and_prepare_pool_view.
        # These are recomputed once per step in _ensure_pool_view_cached() and
        # reused across all layers to avoid redundant Python/CUDA overhead.
        # Initialized here as plain instance attributes (not class-level
        # annotations) to stay invisible to dataclass/torch.compile introspection.
        self._pool_cache_valid = False
        self._cached_kv_mgr_id = 0
        self._cached_pool_key = None
        self._cached_pool_view = None
        self._cached_num_pool_tokens = 0
        self._cached_tokens_per_block = 0
        self._cached_block_table_ctx = None
        self._cached_block_table_gen = None
        self._cached_req_idx_ctx = None
        self._cached_req_idx_gen = None
        # Cross-layer fan-out of the DSA index remap (TRTLLM_DISABLE_DSA_GROUP_REMAP).
        # `_group_remap_struct` is the (static) full+shared indexer
        # group layout; `_group_remap_batched` holds the current forward's
        # per-group batched remap output (leader writes, shared members read).
        self._group_remap_struct = None
        self._group_remap_struct_kv_id = 0
        self._group_remap_batched = {}
        self.num_ctx_mla_kv_tokens = 0
        self.nvfp4_mla_context_fp8_scratch = None
        self._radix_rows_per_sequence = 1
        super().__init__(*args, **kwargs)
        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.indexer_max_chunk_size = sparse_metadata_params.indexer_max_chunk_size

    @property
    def use_cute_dsl_paged_mqa_logits(self) -> bool:
        """The Indexer's decision: the config flag gated on SM100 and cutlass-dsl."""
        return use_cute_dsl_paged_mqa_logits(
            self.sparse_metadata_params.use_cute_dsl_paged_mqa_logits
        )

    def __post_init__(self):
        """Allocate indexer K-cache buffers and heuristic TopK metadata."""
        super().__post_init__()
        if self.is_cross:
            # Ragged verification is a target self-attention capability. A
            # cross-attention metadata object may be a shallow copy of one, but
            # must not inherit its verifier-only buffers or per-step layout.
            self.enable_ragged_verification = False
            self.ragged_verify_lens = None
        self._validate_ragged_verification()
        if not is_dsa_cache_manager(self.kv_cache_manager):
            has_deepseek_v4_cache_interface = all(
                hasattr(self.kv_cache_manager, attr)
                for attr in ("compressed_block_sizes", "get_cache_indices")
            )
            assert has_deepseek_v4_cache_interface, (
                "DSAtrtllmAttentionMetadata requires DSACacheManager-compatible "
                f"cache manager, got {type(self.kv_cache_manager)}"
            )
        self.use_fp8_ds_mla = getattr(self.kv_cache_manager, "use_fp8_ds_mla", False)

        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        self.num_sparse_topk = sparse_metadata_params.max_sparse_topk
        self.sparse_mla_topk = self.num_sparse_topk
        self.indexer_head_dim = sparse_metadata_params.index_head_dim
        self.indexer_quant_block_size = 128
        self.enable_indexer_skip = sparse_metadata_params.enable_indexer_skip
        self.use_cute_dsl_topk = (
            sparse_metadata_params.use_cute_dsl_topk and IS_CUTLASS_DSL_AVAILABLE
        )
        self.enable_gvr_topk = (
            sparse_metadata_params.enable_heuristic_topk and get_sm_version() >= 100
        )
        self.kv_lens_row_reorder = None
        capture_graph = self.is_cuda_graph
        # Plain DSA has no compression and uses the default [1]. DeepSeek-V4's
        # metadata params carry the model-specific compression ratios.
        self.compress_ratios = getattr(sparse_metadata_params, "compress_ratios", [1])

        # Effective tokens-per-block for the indexer k-cache slot mapping.
        # DeepSeek-V4's indexer cache uses layer-dependent compressed block sizes
        # (tokens_per_block // compress_ratio), so slot mappings must be built
        # against that stride — not kv_cache_manager.tokens_per_block directly.
        tpb = self.kv_cache_manager.tokens_per_block
        self._indexer_compress_ratio = _select_indexer_compress_ratio(self.compress_ratios)
        if hasattr(self.kv_cache_manager, "compressed_block_sizes"):
            tpb = tpb // _effective_compress_ratio_divisor(self._indexer_compress_ratio)
        self._tokens_per_block = tpb
        # Mirror the indexer's two-level GVR decision. The representative
        # compress ratio matches every live indexer: the DeepSeek-V4 backend
        # only builds indexers on cr=4 layers and plain DSA uses cr=1.
        self.use_self_sampling_topk = use_self_sampling_gvr(
            enable_heuristic_topk=self.enable_gvr_topk,
            use_self_sampling_topk=sparse_metadata_params.use_self_sampling_topk,
            index_topk=self.num_sparse_topk,
            compress_ratio=self._indexer_compress_ratio,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=get_sm_version(),
        )
        self.needs_gvr_prior = (
            self.enable_gvr_topk
            and IS_CUTLASS_DSL_AVAILABLE
            and get_sm_version() in (100, 103)
            and not self.use_self_sampling_topk
        )

        self.create_buffers_for_mla_rope_append(capture_graph=capture_graph)
        self.create_buffers_for_indexer(capture_graph=capture_graph)
        self._create_nvfp4_mla_generation_buffers(capture_graph=capture_graph)
        self._create_helix_sparse_row_buffers(capture_graph=capture_graph)

    def _helix_local_kv_lens(self, cached_token_lens: torch.Tensor) -> torch.Tensor:
        """Rank-local KV lengths under Helix CP, as TrtllmAttentionMetadata computes them.

        Context rows append all their tokens. A generation row appends only the
        new tokens this rank owns: one token on the active rank without
        speculation, and a per-sequence count when a verify group can straddle a
        page boundary. The Helix per-sequence buffers are generation-relative.
        Under the overlap scheduler these host values are provisional;
        on_update_kv_lens() rederives the device state.
        """
        kv_lens = cached_token_lens + self.seq_lens_kv
        gen = slice(self.num_contexts, self.num_seqs)
        num_gen = self.num_generations
        if self._helix_spec_tokens_valid:
            kv_lens[gen] = cached_token_lens[gen] + self.helix_owned_new_tokens_cpu[:num_gen]
        else:
            inactive = self.helix_is_inactive_rank_cpu[:num_gen]
            kv_lens[gen] = torch.where(inactive, cached_token_lens[gen], kv_lens[gen])
        return kv_lens

    def _create_helix_sparse_row_buffers(self, capture_graph=False):
        """Allocate the token-major rows token-sparse MLA reads under Helix.

        After the Helix Top-K merge a rank keeps only its share of each row's
        global Top-K: the kept entries come first and the rest of the row is
        -1. The sparse-MLA kernel attends to the first min(top_k, kv_len)
        entries of a row and does not mask -1 inside that range, so every
        generation token is presented as its own row whose length is its kept
        count (``set_helix_sparse_row_lens``).
        """
        names = (
            "helix_row_kv_lens_cuda",
            "helix_row_kv_lens_host",
            "helix_row_req_idx_cuda",
            "helix_row_req_idx_host",
            "helix_row_prompt_lens_cuda",
            "helix_row_prompt_lens_cpu",
            "helix_row_request_types_host",
            "helix_row_block_offsets",
        )
        for name in names:
            setattr(self, name, None)
        self._helix_num_rows = 0
        self._helix_rows_set = False
        if not self.enable_helix or self.kv_cache_manager is None:
            return
        rows_cap = self.max_num_sequences * (2 + self._draft_sizing_cap)
        for name, dtype in (
            ("helix_row_kv_lens_cuda", torch.int32),
            ("helix_row_req_idx_cuda", torch.long),
            ("helix_row_prompt_lens_cuda", torch.int32),
        ):
            setattr(
                self,
                name,
                self.get_empty(
                    self.cuda_graph_buffers,
                    (rows_cap,),
                    cache_name=name,
                    dtype=dtype,
                    capture_graph=capture_graph,
                ),
            )
        pinned = prefer_pinned()
        self.helix_row_kv_lens_host = torch.zeros((rows_cap,), dtype=torch.int32, pin_memory=pinned)
        self.helix_row_req_idx_host = torch.zeros((rows_cap,), dtype=torch.long, pin_memory=pinned)
        self.helix_row_prompt_lens_cpu = torch.zeros(
            (rows_cap,), dtype=torch.int32, pin_memory=pinned
        )
        self.helix_row_request_types_host = torch.ones(
            (rows_cap,), dtype=torch.int32, pin_memory=pinned
        )
        num_attention_op_pools = getattr(
            self.kv_cache_manager, "num_attention_op_pools", self.kv_cache_manager.num_pools
        )
        self.helix_row_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            (num_attention_op_pools, rows_cap, 2, self.kv_cache_manager.max_blocks_per_seq),
            cache_name="helix_row_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    def _prepare_helix_sparse_rows(self, kv_lens: torch.Tensor) -> None:
        """Lay out one attention row per context request and generation token.

        Host lengths are upper bounds for the op's host-side planning; the
        device lengths start there and the indexer narrows each generation
        row to its kept count layer by layer.
        """
        self._helix_num_rows = 0
        if self.helix_row_kv_lens_cuda is None or self.num_generations <= 0:
            return
        num_contexts = self.num_contexts
        gen_seq_lens = self.seq_lens[num_contexts : self.num_seqs].to(torch.long)
        num_rows = num_contexts + int(gen_seq_lens.sum())
        assert num_rows <= self.helix_row_kv_lens_cuda.shape[0], (
            f"Helix sparse rows need {num_rows} rows, capacity "
            f"{self.helix_row_kv_lens_cuda.shape[0]}"
        )
        req_idx = torch.cat(
            [
                torch.arange(num_contexts, dtype=torch.long),
                torch.repeat_interleave(
                    torch.arange(num_contexts, self.num_seqs, dtype=torch.long), gen_seq_lens
                ),
            ]
        )
        self.helix_row_req_idx_host[:num_rows].copy_(req_idx)
        # A generation row reads at most top_k entries and never more than its
        # sequence holds locally after this step's appends (an upper bound
        # under the overlap scheduler's provisional lengths).
        upper = (kv_lens[: self.num_seqs].to(torch.long) + self.seq_lens_kv.to(torch.long)).clamp(
            max=self.num_sparse_topk
        )
        upper[:num_contexts] = kv_lens[:num_contexts]
        self.helix_row_kv_lens_host[:num_rows].copy_(upper[req_idx].to(torch.int32))
        self.helix_row_prompt_lens_cpu[:num_contexts].copy_(
            self.prompt_lens_cpu[:num_contexts].to(torch.int32)
        )
        self.helix_row_prompt_lens_cpu[num_contexts:num_rows].fill_(1)
        self.helix_row_request_types_host[:num_contexts].zero_()
        self.helix_row_request_types_host[num_contexts:num_rows].fill_(1)
        for dst, src in (
            (self.helix_row_kv_lens_cuda, self.helix_row_kv_lens_host),
            (self.helix_row_req_idx_cuda, self.helix_row_req_idx_host),
            (self.helix_row_prompt_lens_cuda, self.helix_row_prompt_lens_cpu),
        ):
            self._stage_or_copy(dst[:num_rows], src[:num_rows])
        source = self.kv_cache_block_offsets
        num_pools = min(source.shape[0], self.helix_row_block_offsets.shape[0])
        width = min(source.shape[-1], self.helix_row_block_offsets.shape[-1])

        def gather_rows():
            torch.index_select(
                source[:num_pools, :, :, :width],
                1,
                self.helix_row_req_idx_cuda[:num_rows],
                out=self.helix_row_block_offsets[:num_pools, :num_rows, :, :width],
            )

        if self._staging_commit is not None:
            # The gather reads the staged row index and block offsets: right after the commit.
            self._staging_commit.after_commit(gather_rows)
        else:
            gather_rows()
        self._helix_num_rows = num_rows
        self._helix_rows_set = False
        self._helix_combine_mask = None

    def set_helix_sparse_row_lens(self, gen_rows: torch.Tensor) -> None:
        """Present each generation row with exactly its kept Top-K entries.

        Called by every indexer layer after its Helix merge; a layer that
        shares the previous layer's Top-K reads the same lengths.
        """
        if self._helix_num_rows <= 0:
            return
        num_contexts = self.num_contexts
        torch.sum(
            gen_rows >= 0,
            dim=1,
            dtype=torch.int32,
            out=self.helix_row_kv_lens_cuda[num_contexts : num_contexts + gen_rows.shape[0]],
        )
        self._helix_rows_set = True
        self._helix_combine_mask = None

    def helix_sparse_row_lens_out(self, num_gen_rows: int) -> Optional[torch.Tensor]:
        """The generation rows' slice of the token-sparse MLA row lengths, for a producer that writes the kept
        counts itself (then ``mark_helix_sparse_row_lens_set``); None without Helix sparse rows."""
        if self._helix_num_rows <= 0:
            return None
        num_contexts = self.num_contexts
        return self.helix_row_kv_lens_cuda[num_contexts : num_contexts + num_gen_rows]

    def mark_helix_sparse_row_lens_set(self, combine_mask: Optional[torch.Tensor] = None) -> None:
        """The generation rows' lengths were written through ``helix_sparse_row_lens_out``; ``combine_mask``, if
        the producer wrote it too, is their ``helix_zero_or_empty_rows``."""
        if self._helix_num_rows > 0:
            self._helix_rows_set = True
            self._helix_combine_mask = combine_mask

    def helix_sparse_empty_rows(self, num_gen_tokens: int) -> Optional[torch.Tensor]:
        """Generation rows whose current Top-K kept nothing on this rank, or None.

        Such a row attends to no local KV, so its partial output must not
        enter the Helix combine.
        """
        if self._helix_num_rows <= 0 or not self._helix_rows_set:
            return None
        num_contexts = self.num_contexts
        return self.helix_row_kv_lens_cuda[num_contexts : num_contexts + num_gen_tokens] == 0

    def helix_zero_or_empty_rows(self, num_gen_tokens: int) -> Optional[torch.Tensor]:
        """The Helix combine mask of the generation rows in one launch (``mla.py``,
        ``helix_folded_zero_kv_mask``).

        ``(helix_kv_bounds == 0) | (kept count == 0)``: the rows this rank owns no KV for (their
        per-token bound, as ``helix_gen_zero_kv_mask`` takes it on a speculative step) or whose
        current Top-K kept nothing here (``helix_sparse_empty_rows``). None where either half is not
        the per-token form, so the caller keeps the generic path.
        """
        bounds = getattr(self, "helix_kv_bounds", None)
        if bounds is None or not getattr(self, "_helix_spec_tokens_valid", False):
            return None
        if self._helix_num_rows <= 0 or not self._helix_rows_set:
            return None
        mask = getattr(self, "_helix_combine_mask", None)
        if mask is not None and mask.shape[0] == num_gen_tokens:
            return mask
        num_contexts = self.num_contexts
        return helix_zero_or_empty_rows(
            bounds[:num_gen_tokens],
            self.helix_row_kv_lens_cuda[num_contexts : num_contexts + num_gen_tokens],
        )

    def helix_spec_row_kv_lens(self, num_gen_tokens: int) -> Optional[torch.Tensor]:
        """Per-token rank-local KV bounds of Helix verify groups, or None."""
        if not (self.enable_helix and self._helix_spec_tokens_valid):
            return None
        return self.helix_kv_bounds[:num_gen_tokens]

    # The input commit the running prepare() stages its uploads into, or None (_stage_or_copy).
    _staging_commit: Optional[InputCommit] = None

    def prepare(self):
        self._validate_ragged_verification()
        super().prepare()
        commit = active_input_commit()
        staging = commit is not None and commit.is_open
        flushed = staging and not self._staged_prepare_needs_no_flush()
        if flushed:
            # The DSA preparation below reads the KV lengths and offsets on the device.
            flush_active_input_commit()
        self._staging_commit = commit if staging else None
        self._num_staged_uploads = 0
        try:
            self._prepare_dsa()
        finally:
            self._staging_commit = None
        if staging and self._num_staged_uploads:
            logger.info_once(
                f"DSA prepare stages its uploads into the step's input commit "
                f"({self._num_staged_uploads} entries at num_seqs={self.num_seqs}, "
                f"num_tokens={self.num_tokens}; flushed first: {flushed})",
                key=f"dsa_prepare_staged_{flushed}",
            )

    def _staged_prepare_needs_no_flush(self) -> bool:
        """Whether :meth:`_prepare_dsa` of a staged step runs without the inputs staged so far
        written: a Helix speculative generation step, where every device read of a staged
        input is deferred past the commit (the sparse-row gather, ``after_commit``) and the
        DeepGEMM schedules are left to ``on_update_kv_lens``. Context rows, ragged
        verification and the CuTe DSL row expansion read the device lengths, so they flush."""
        return (
            self.num_contexts == 0
            and self.enable_helix
            and self._helix_spec_tokens_valid
            and not self.is_ragged_verify
        )

    _num_staged_uploads: int = 0

    def helix_host_gen_slot_rows(self, indexer_params) -> Optional[slice]:
        """Rows of the host slot mappings this metadata rewrote for Helix generation tokens (the
        indexer uploads them with its own), or None: the engine keeps the generic mapping."""
        return None

    def _stage_or_copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """``dst.copy_(src, non_blocking=True)`` of a host ``src``; staged into the step's input
        commit instead while :meth:`prepare` runs inside one (a CUDA-graph generation step of the
        decode-step shell, ``input_commit.py``). The commit kernel writes the same bytes before the
        replay, and nothing in :meth:`prepare` reads a staged destination on the device before
        it: the clamps are applied on the host, the sparse-row gather runs after the commit and
        the DeepGEMM schedules are left to the step's ``on_update_kv_lens`` (see the callers)."""
        commit = self._staging_commit
        if commit is None:
            dst.copy_(src, non_blocking=True)
        elif dst.is_contiguous():
            commit.stage(dst, src)
            self._num_staged_uploads += 1
        else:
            # Rows of a 2-D slice of a wider buffer: one contiguous entry per row.
            for row in range(dst.shape[0]):
                commit.stage(dst[row], src[row])
            self._num_staged_uploads += dst.shape[0]

    def _clamp_staged_buffer_once(self, buf: torch.Tensor) -> None:
        """Before the first staged upload of a block-offset buffer, apply the full-buffer
        ``clamp_(min=0)`` of the copy path once. Staged uploads are clamped on the host, so the
        rows a step leaves alone stay non-negative afterwards, as the copy path keeps them."""
        done = self.__dict__.setdefault("_staged_clamped_buffers", set())
        key = buf.data_ptr()
        if key not in done:
            buf.clamp_(min=0)
            done.add(key)

    def _prepare_dsa(self):
        self._invalidate_pool_view_cache()

        # Get kv lengths
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        if self.enable_helix:
            kv_lens = self._helix_local_kv_lens(cached_token_lens)
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv

        kv_lens = self._prepare_device_window_kv_lens(kv_lens)

        # For mla_rope_append_paged_kv_assign_q
        self.prepare_for_mla_rope_append(cached_token_lens, kv_lens)

        # Prepare to support skip indexer
        self.prepare_for_skip_indexer(kv_lens)

        # For indices conversion
        self.prepare_for_indices_conversion()

        # For indexer k cache
        self.prepare_for_indexer_k_cache()

        # For spec decode
        self.prepare_for_spec_decode(kv_lens)

        # Prepare metadata for indexer
        Indexer.prepare(metadata=self)

        if self.enable_helix:
            self._prepare_helix_sparse_rows(kv_lens)

    def _prepare_device_window_kv_lens(self, kv_lens: torch.Tensor) -> torch.Tensor:
        """Bound host-side KV extents until device-selected windows are installed."""
        if self.device_windows_mode and self.is_ragged_verify:
            nc, ns = self.num_contexts, self.num_seqs
            kv_lens = kv_lens.clone()
            kv_lens[nc:ns] += (1 + self.max_draft_tokens) - self.seq_lens_kv[nc:ns]
            self.kv_lens[nc:ns] = kv_lens[nc:ns]
        return kv_lens

    def prepare_for_draft_forward(self) -> dict | None:
        """Select native DSA indexer metadata for a draft forward."""
        # DeepSeek-V4 metadata inherits DSA metadata, but its cache manager uses a
        # different dual-pool layout. Only native DSA cache managers use the DSA
        # draft-replay buffers below.
        if not is_dsa_cache_manager(self.kv_cache_manager):
            return None

        saved_state = {
            "host_indexer_k_cache_block_offsets": self.host_indexer_k_cache_block_offsets,
            "indexer_k_cache_block_offsets": self.indexer_k_cache_block_offsets,
            "host_slot_mapping_fp8": self.host_slot_mapping_fp8,
            "host_slot_mapping_scale": self.host_slot_mapping_scale,
            "slot_mapping_fp8": self.slot_mapping_fp8,
            "slot_mapping_scale": self.slot_mapping_scale,
            "block_table": self.block_table,
            "block_table_expanded": self.block_table_expanded,
            "host_block_table_expanded": self.host_block_table_expanded,
        }
        # The cached-KV feature owns these references even when an optimized
        # path aliases them to slot_mapping_*. With the feature disabled, the
        # aliases are lazy and may not exist on the first generation replay.
        if self.enable_context_mla_with_cached_kv:
            saved_state.update(
                {
                    "slot_mapping_fp8_fullkv": self.slot_mapping_fp8_fullkv,
                    "slot_mapping_scale_fullkv": self.slot_mapping_scale_fullkv,
                }
            )

        # Rebind to the draft manager's dedicated buffers instead of
        # overwriting the target tensors in place. Rebinding is invisible to
        # CUDA graph capture, so the target and draft segments of the graph
        # bake distinct addresses (like draft_kv_cache_block_offsets) and no
        # graph-recorded copy from a transient host buffer is needed.
        self.swap_for_draft(
            "host_indexer_k_cache_block_offsets", self.host_draft_indexer_k_cache_block_offsets
        )
        self.swap_for_draft(
            "indexer_k_cache_block_offsets", self.draft_indexer_k_cache_block_offsets
        )
        self.swap_for_draft("host_slot_mapping_fp8", self.host_draft_slot_mapping_fp8)
        self.swap_for_draft("slot_mapping_fp8", self.draft_slot_mapping_fp8)
        self.swap_for_draft("host_slot_mapping_scale", self.host_draft_slot_mapping_scale)
        self.swap_for_draft("slot_mapping_scale", self.draft_slot_mapping_scale)
        self.swap_for_draft("block_table", self.draft_block_table)
        self.swap_for_draft("block_table_expanded", self.draft_block_table_expanded)
        self.swap_for_draft("host_block_table_expanded", self.host_draft_block_table_expanded)
        # recompute_context_kv_gather_mappings() rebinds the aliases below.
        self.record_draft_swap("slot_mapping_fp8_fullkv")
        self.record_draft_swap("slot_mapping_scale_fullkv")
        self._invalidate_pool_view_cache()

        # Recording a capture executes no kernels, so the draft mappings only
        # need refreshing when the transfers actually run: eager forwards
        # (warmup) and the pre-replay call from model_engine. The per-step
        # advance inside the captured graph re-derives slot mappings on
        # device from the rebound block-offset buffer.
        # kv_cache_manager was already swapped to the draft manager above.
        if not torch.cuda.is_current_stream_capturing():
            self.prepare_for_indexer_k_cache()
            self._refresh_expanded_block_table()
            Indexer.recompute_slot_mappings(self)
        Indexer.recompute_context_kv_gather_mappings(self)

        return saved_state

    def restore_after_draft_forward(self, saved_state: dict | None) -> None:
        """Restore native DSA indexer metadata after a draft forward."""
        if saved_state is None:
            return

        # restore_draft_swaps() has already rebound the swapped fields.
        self._invalidate_pool_view_cache()
        if "slot_mapping_fp8_fullkv" not in saved_state:
            # Without cached-KV context MLA the aliases track the target mappings.
            self.slot_mapping_fp8_fullkv = self.slot_mapping_fp8
            self.slot_mapping_scale_fullkv = self.slot_mapping_scale

    def get_indexer_kv_lens(self, kv_lens: torch.Tensor) -> torch.Tensor:
        if self._indexer_compress_ratio <= 1:
            return kv_lens
        return kv_lens // self._indexer_compress_ratio

    def get_indexer_max_seq_len(self) -> int:
        if self._indexer_compress_ratio <= 1:
            return self.kv_cache_manager.max_seq_len
        return max(1, self.kv_cache_manager.max_seq_len // self._indexer_compress_ratio)

    def warmup_cute_dsl_radix_topk(self, next_n: int) -> None:
        """Pre-compile CuTe DSL radix variants not covered by engine warmup."""
        sparse_params = self.sparse_metadata_params
        if not self.use_cute_dsl_topk or (
            sparse_params.enable_heuristic_topk and get_sm_version() >= 100
        ):
            return
        if self.kv_cache_manager is None:
            return
        top_k = self.sparse_mla_topk
        if not top_k:
            return
        if self._indexer_compress_ratio > 1 and next_n > 1:
            return
        try:
            from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
                warmup_cute_dsl_radix_topk_decode,
            )
        except ImportError:
            return

        warmup_cute_dsl_radix_topk_decode(
            top_k=int(top_k),
            num_cols=int(self.get_indexer_max_seq_len()),
            next_n=next_n,
            dtype=_INDEXER_LOGITS_DTYPE,
            num_sms=self.num_sms,
        )

    def warmup_selfsampling_topk(
        self, next_n: int, batch_sizes: Optional[List[int]] = None
    ) -> None:
        """Pre-compile the self-sampling GVR varlen engine during warmup.

        Mirrors ``warmup_cute_dsl_radix_topk``. The varlen launcher is keyed
        by the exact row count AND the logits row stride: rows cover the
        small eager batches plus every configured CUDA-graph batch size, and
        the stride mirrors what the active paged-MQA producer emits, so the
        warmed keys are the ones dispatch actually looks up. Batches outside
        this set still compile lazily on first touch. The helper enumerates
        one representative row per distinct engine compile key, so large
        batch lists warm in bounded time and memory. No-op unless the
        two-level dispatch (enable_heuristic_topk + use_self_sampling_topk)
        selects the self-sampling engine.
        """
        # same two-level dispatch and hardware gates as the indexer __init__:
        # never compile these kernels on unsupported stacks during warmup
        if not IS_CUTLASS_DSL_AVAILABLE or get_sm_version() not in (100, 103, 107):
            return
        if not self.enable_gvr_topk or self.kv_cache_manager is None:
            return
        if not self.use_self_sampling_topk:
            return
        top_k = self.sparse_mla_topk
        if not top_k or int(top_k) not in (512, 1024, 2048):
            return
        cr = int(self._indexer_compress_ratio) if self._indexer_compress_ratio else 1
        if cr not in (1, 4):
            return
        try:
            from tensorrt_llm._torch.kernels.blackwell.top_k import (
                gvr_topk_decode_self_sampling_host as _ss_host,
            )
        except ImportError:
            return
        nn = int(next_n)
        # warm the small row counts eager mixed batches commonly produce;
        # larger eager row counts lazy-JIT on first touch, and CUDA-graph
        # geometries are covered through ``batch_sizes`` below
        eager_warm_rows = 32
        rows = set(range(nn, eager_warm_rows + 1, nn)) or {nn}
        for bs in batch_sizes or ():
            rows.add(int(bs) * nn)
        msl_c = int(self.get_indexer_max_seq_len())
        # Prefill leg first: the decode-stride guard below may return early for
        # an odd msl_c, but the DeepGEMM prefill stride is always 256-aligned.
        # forwarded only when enabled (keeps the default warmup calls unchanged)
        det_kw = (
            {"deterministic": True}
            if getattr(self.sparse_metadata_params, "gvr_deterministic_topk", False)
            else {}
        )
        try:
            _ss_host.warmup_prefill(int(top_k), max(msl_c, 32768), **det_kw)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "self-sampling GVR prefill warmup ran out of memory; prefill "
                "engines will JIT-compile lazily on first touch instead."
            )
        if self.use_cute_dsl_paged_mqa_logits:
            # mirror the DSL paged-MQA arena stride (rows round up to 256
            # elements). A drift here only degrades warmup to unused keys —
            # dispatch still lazy-JITs the true key outside capture.
            row_stride = (msl_c + 255) // 256 * 256
        else:
            # DeepGEMM emits exact-width rows; a non-float4 width falls
            # through at the dispatch format gate, so there is nothing to warm
            row_stride = msl_c
            if row_stride % 4:
                return
        # The helper takes max_seq_len in KV-token space;
        # get_indexer_max_seq_len is compressed, so multiply it back as at
        # the dispatch seam.
        try:
            _ss_host.warmup_varlen(
                int(top_k),
                msl_c * cr,
                compress_ratio=cr,
                next_n=int(next_n),
                num_rows_list=tuple(sorted(rows)),
                row_stride=row_stride,
                **det_kw,
            )
        except torch.cuda.OutOfMemoryError:
            # warmup is best-effort: the dispatch works without it (engines
            # JIT lazily outside capture), so do not fail engine init
            logger.warning(
                "self-sampling GVR warmup ran out of memory; varlen engines "
                "will JIT-compile lazily on first touch instead."
            )

    def _apply_helix_gen_ownership(self) -> None:
        """Give generation rows this CP rank's indexer-K slot, or -1 (skipped) for tokens
        another rank owns; verify groups score one row per token against ``helix_kv_bounds``.
        Device-only with no host sync, so it can be captured."""
        num_ctx_tokens = self.num_ctx_tokens
        num_gen_tokens = self.num_tokens - num_ctx_tokens
        if num_gen_tokens <= 0:
            return
        if self._helix_spec_tokens_valid:
            local = self.helix_local_slots[:num_gen_tokens].to(torch.int64)
            owned = local >= 0
            req_indices = self.req_idx_per_token[num_ctx_tokens : self.num_tokens].to(torch.int64)
        else:
            # One token per generation row; the active rank appends it locally.
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs].to(torch.int64)
            local = gen_kv_lens - 1
            owned = ~self.helix_is_inactive_rank[: self.num_generations]
            req_indices = torch.arange(
                self.num_contexts, self.num_seqs, dtype=torch.int64, device=local.device
            )
        index_head_dim = self.kv_cache_manager.index_head_dim
        use_fp4 = getattr(self.kv_cache_manager, "use_fp4", False)
        data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
        fp8_indices, scale_indices = _compute_slot_mappings(
            local.clamp(min=0),
            self.indexer_k_cache_block_offsets,
            req_indices,
            index_head_dim,
            self._tokens_per_block,
            self.indexer_quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )
        gen_rows = slice(num_ctx_tokens, self.num_tokens)
        self.slot_mapping_fp8[gen_rows] = torch.where(owned, fp8_indices, -1)
        self.slot_mapping_scale[gen_rows] = torch.where(owned, scale_indices, -1)

        if self._helix_spec_tokens_valid and self.use_expanded_buffers_for_mtp:
            # A rank may see no KV for a token (bound 0); scoring keeps at least
            # one row slot so the schedule never sees an all-empty batch, and
            # the indexer drops those rows' candidates (helix_spec_row_kv_lens).
            bounds = self.helix_kv_bounds[:num_gen_tokens]
            self.kv_lens_expanded_cuda[:num_gen_tokens].copy_(bounds.clamp(min=1))
            scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                self.kv_lens_expanded_cuda[:num_gen_tokens].view(-1, 1),
                _DG_SCHEDULE_BLOCK_KV,
                self.num_sms,
            )
            self.scheduler_metadata_buffer_expanded.copy_(
                scheduler_metadata_buffer_expanded, non_blocking=True
            )

    def on_update_kv_lens(self) -> None:
        # After changing the kv_lens/kv_lens_cuda, we may need to update other metadatas.
        # Especially for the changes in DecoderInputPreparer.preprocess_inputs().
        #
        # NOTE:
        # In overlap scheduler + speculative decoding, kv_lens_cuda can be corrected at runtime
        # (inside preprocess_inputs) to account for variable accepted tokens. The indexer
        # slot_mapping_* buffers also depend on these effective cached lengths. If we do not
        # refresh slot mappings here, indexer K-cache updates can be written with stale offsets.

        super().on_update_kv_lens()

        # preprocess_inputs() also uses this as a general hook to "invalidate per-forward-pass
        # caches so they are recomputed (and captured) on every _forward_step". Invalidate the
        # pool_view cache here so it is recomputed on the next
        # transform_local_topk_and_prepare_pool_view() call.
        self._invalidate_pool_view_cache()

        if self._helix_fused_meta_applies():
            self._update_kv_lens_helix_fused()
            return

        # Default fused path: collapse the eager DSA decode-metadata chain
        # (req_idx_per_token + slot mappings + the two gen indptr cumsums) into
        # one fused Triton launch. Only for the pure-decode/generation step
        # (num_contexts == 0); see kernels.py and
        # _run_fused_dsa_decode_metadata().
        #
        # The fused kernel produces the five shared decode-metadata outputs but
        # NOT token_positions_cuda, which the eager slot block writes only when
        # use_fp8_ds_mla is set (consumed by the FlashInfer sparse-attention
        # paths for RoPE). Exclude that cache mode so the fused path never leaves
        # it stale; those configs keep the eager chain.
        fused_eligible = (
            _fused_dsa_meta_enabled()
            and self.kv_cache_manager is not None
            and self.num_tokens > 0
            and self.num_generations > 0
            and self.num_contexts == 0
            and not self.use_fp8_ds_mla
            # Helix rewrites the generation slots per rank below.
            and not self.enable_helix
        )

        # The draft loop can rewrite seq_lens after prepare() built this map.
        if self.num_tokens > 0 and not fused_eligible:
            self.req_idx_per_token[: self.num_tokens] = build_req_idx_per_token(
                self.seq_lens_cuda[: self.num_seqs], self.num_tokens
            ).to(self.req_idx_per_token.dtype)

        # The overlap correction may have changed request KV lengths after
        # prepare(); rebuild each ragged row's causal extent on device.
        if self.enable_ragged_verification:
            self.refresh_ragged_row_kv_lens()
            self.refresh_token_major_gen_rows()

        # Uncached DSA prefill aliases these scatter maps as gather maps.
        if self.kv_cache_manager is not None and self.num_tokens > 0 and not fused_eligible:
            seq_lens = self.seq_lens_cuda[: self.num_seqs]
            # Runtime cached lengths after overlap/spec-dec correction.
            start_positions = self.kv_lens_cuda[: self.num_seqs] - seq_lens

            req_indices = self.req_idx_per_token[: self.num_tokens].to(dtype=torch.int64)
            seq_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
            token_offsets = (
                torch.arange(self.num_tokens, device=seq_lens.device, dtype=torch.int64)
                - seq_starts[req_indices]
            )

            global_positions = start_positions[req_indices] + token_offsets
            if self.use_fp8_ds_mla:
                self.token_positions_cuda[: self.num_tokens] = global_positions.to(torch.int32)
            # Honor MXFP4 indexer K cache layout (½ byte per value vs FP8's
            # 1 byte) when the cache manager exposes a use_fp4 flag.
            index_head_dim = self.kv_cache_manager.index_head_dim
            use_fp4 = getattr(self.kv_cache_manager, "use_fp4", False)
            data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim
            fp8_indices, scale_indices = _compute_slot_mappings(
                global_positions,
                self.indexer_k_cache_block_offsets,
                req_indices,
                index_head_dim,
                self._tokens_per_block,
                self.indexer_quant_block_size,
                data_bytes_per_token=data_bytes_per_token,
            )
            self.slot_mapping_fp8[: self.num_tokens] = fp8_indices
            self.slot_mapping_scale[: self.num_tokens] = scale_indices

        if self.num_generations > 0:
            if not fused_eligible:
                torch.cumsum(
                    self.kv_lens_cuda[
                        self.num_contexts : self.num_seqs
                    ],  # num_contexts should be 0
                    dim=0,
                    dtype=torch.int64,
                    out=self.gen_kv_indptr[1 : self.num_generations + 1],
                )
                torch.cumsum(
                    (
                        self.kv_lens_cuda[self.num_contexts : self.num_seqs]
                        - self.seq_lens_cuda[self.num_contexts : self.num_seqs]
                    ),
                    dim=0,
                    dtype=torch.int64,
                    out=self.gen_cached_token_indptr[1 : self.num_generations + 1],
                )
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs]
            gen_indexer_kv_lens = self.get_indexer_kv_lens(gen_kv_lens)
            self.gen_indexer_kv_lens_cuda_runtime = gen_indexer_kv_lens
            scheduler_context_lens = self.update_indexer_kv_lens_2d(gen_indexer_kv_lens)
            scheduler_metadata_buffer = get_paged_mqa_logits_metadata(
                gen_indexer_kv_lens.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, self.num_sms
            )
            self.scheduler_metadata_buffer.copy_(scheduler_metadata_buffer, non_blocking=True)
            if self.max_draft_tokens > 0 and not self.use_expanded_buffers_for_mtp:
                scheduler_metadata_buffer_full_next_n = get_paged_mqa_logits_metadata(
                    scheduler_context_lens.contiguous(),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_full_next_n.copy_(
                    scheduler_metadata_buffer_full_next_n, non_blocking=True
                )
            if self.use_expanded_buffers_for_mtp:
                kv_lens_expanded, num_tokens = self.expand_per_gen_token(gen_indexer_kv_lens)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(kv_lens_expanded)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True
                )
            if self.expand_for_dsl and self.dsl_kernel_atoms > 1:
                num_rows = self.num_generations * self.dsl_kernel_atoms
                atom_lens = gen_indexer_kv_lens.unsqueeze(1) - self.dsl_atom_ctx_offsets
                self.kv_lens_expanded_cuda[:num_rows].copy_(atom_lens.reshape(-1))
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_rows].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True
                )
            if self.expand_for_dsl and self.dsl_expand_factor > 1:
                expand_factor = self.dsl_expand_factor
                num_tokens = self.num_generations * expand_factor
                gen_kv_lens_expanded = gen_indexer_kv_lens.repeat_interleave(expand_factor)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(gen_kv_lens_expanded)
                scheduler_metadata_buffer_expanded = get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                )
                self.scheduler_metadata_buffer_expanded.copy_(
                    scheduler_metadata_buffer_expanded, non_blocking=True
                )

        if fused_eligible:
            if not getattr(self, "_fused_dsa_meta_armed", False):
                logger.info(
                    "Fused DSA decode-metadata "
                    f"kernel armed (num_seqs={self.num_seqs}, "
                    f"num_tokens={self.num_tokens})."
                )
                self._fused_dsa_meta_armed = True
            self._run_fused_dsa_decode_metadata()

        if self.enable_helix and self.kv_cache_manager is not None and self.num_generations > 0:
            self._apply_helix_gen_ownership()

        self._compute_kv_lens_row_reorder()
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def _helix_fused_meta_applies(self) -> bool:
        """Whether ``on_update_kv_lens`` takes the fused Helix path
        (:func:`_fused_dsa_helix_meta_enabled`).

        The step must be a Helix CP generation step with uniform verify groups and the per-token
        Helix buffers armed: no context request, the expanded per-token schedule, no ragged
        verification, no DSL split rows, no fp8_ds_mla token positions and an uncompressed
        indexer. Any other step keeps the eager chain.
        """
        return (
            _fused_dsa_helix_meta_enabled()
            and self.enable_helix
            and self.kv_cache_manager is not None
            and getattr(self, "_helix_spec_tokens_valid", False)
            and self.num_contexts == 0
            and self.num_ctx_tokens == 0
            and self.num_generations > 0
            and self.num_tokens == self.num_generations * self.gen_token_stride
            and getattr(self, "use_expanded_buffers_for_mtp", False)
            and not getattr(self, "expand_for_dsl", True)
            and not self.enable_ragged_verification
            and not self.is_ragged_verify
            and not self.use_fp8_ds_mla
            and self._indexer_compress_ratio <= 1
        )

    def _update_kv_lens_helix_fused(self) -> None:
        """``on_update_kv_lens`` of a Helix CP generation step with the fused metadata kernel.

        Leaves the state of the eager chain (``req_idx_per_token``, the slot mappings after the
        Helix ownership rewrite, the generation indptrs, the 2D and per-token indexer lengths and
        both DeepGEMM schedules) with one Triton launch plus the two schedule kernels; the
        generic slots and the generic per-token schedule the eager chain builds and then
        overwrites are not computed.
        """
        num_seqs = self.num_seqs
        num_tokens = self.num_tokens
        index_head_dim = self.kv_cache_manager.index_head_dim
        # Indexer K payload bytes of one token, without its scale (FP4 packs two per byte).
        data_bytes_per_token = (
            index_head_dim // 2
            if _indexer_k_dtype(self.kv_cache_manager) == "fp4"
            else index_head_dim
        )
        gen_kv_lens = self.kv_lens_cuda[:num_seqs]
        # On SM 10.x the launch also writes both DeepGEMM schedules (DeepGEMM's SM100 metadata).
        in_kernel_schedule = _dg_sm100_schedule_in_kernel()
        self._launch_fused_dsa_helix_decode_metadata(
            self.seq_lens_cuda[:num_seqs],
            gen_kv_lens,
            self.indexer_k_cache_block_offsets,
            self.helix_local_slots,
            self.helix_kv_bounds,
            self.req_idx_per_token,
            self.slot_mapping_fp8,
            self.slot_mapping_scale,
            self.gen_kv_indptr,
            self.gen_cached_token_indptr,
            self.kv_lens_cuda_2d,
            self.kv_lens_expanded_cuda,
            num_tokens=num_tokens,
            width=min(self.gen_token_stride, self.kv_lens_cuda_2d.shape[1]),
            tokens_per_block=self._tokens_per_block,
            index_head_dim=index_head_dim,
            quant_block_size=self.indexer_quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
            schedule=self.scheduler_metadata_buffer if in_kernel_schedule else None,
            schedule_expanded=(
                self.scheduler_metadata_buffer_expanded if in_kernel_schedule else None
            ),
        )
        if not getattr(self, "_fused_dsa_helix_meta_armed", False):
            logger.info(
                "Fused Helix DSA decode-metadata kernel armed "
                f"(num_seqs={num_seqs}, num_tokens={num_tokens}, "
                f"DeepGEMM schedules in the kernel: {in_kernel_schedule})."
            )
            self._fused_dsa_helix_meta_armed = True
        # The indexer compress ratio is 1 on this path (_helix_fused_meta_applies).
        gen_indexer_kv_lens = self.get_indexer_kv_lens(gen_kv_lens)
        self.gen_indexer_kv_lens_cuda_runtime = gen_indexer_kv_lens
        if not in_kernel_schedule:
            self.scheduler_metadata_buffer.copy_(
                get_paged_mqa_logits_metadata(
                    gen_indexer_kv_lens.view(-1, 1), _DG_SCHEDULE_BLOCK_KV, self.num_sms
                ),
                non_blocking=True,
            )
            self.scheduler_metadata_buffer_expanded.copy_(
                get_paged_mqa_logits_metadata(
                    self.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
                    _DG_SCHEDULE_BLOCK_KV,
                    self.num_sms,
                ),
                non_blocking=True,
            )
        self._compute_kv_lens_row_reorder()
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def _launch_fused_dsa_helix_decode_metadata(self, *args, **kwargs) -> None:
        """The launch of ``_update_kv_lens_helix_fused``: ``fused_dsa_helix_decode_metadata(*args,
        **kwargs)``. A subclass may make it part of another launch that writes the same outputs
        from the same inputs."""
        fused_dsa_helix_decode_metadata(*args, **kwargs)

    def _run_fused_dsa_decode_metadata(self):
        """Fill req_idx_per_token + slot mappings + gen indptrs via one Triton
        launch (see kernels.py), replacing the eager chain for the
        pure-decode step. Capture-safe: the launch is pre-compiled at warmup and
        replays inside the decode CUDA graph."""
        num_tokens = self.num_tokens
        num_seqs = self.num_seqs
        # The fused header writes gen_*_indptr[1 : num_seqs + 1] while the eager
        # chain writes [1 : num_generations + 1]; they coincide only because the
        # eligibility gate requires num_contexts == 0 (=> num_seqs ==
        # num_generations). Pin that coupling down explicitly.
        assert num_seqs == self.num_generations, (
            f"fused DSA metadata expects num_seqs ({num_seqs}) == "
            f"num_generations ({self.num_generations})"
        )
        index_head_dim = self.kv_cache_manager.index_head_dim
        use_fp4 = getattr(self.kv_cache_manager, "use_fp4", False)
        data_bytes_per_token = index_head_dim // 2 if use_fp4 else index_head_dim

        fused_dsa_decode_metadata(
            self.seq_lens_cuda[:num_seqs],
            self.kv_lens_cuda[:num_seqs],
            self.indexer_k_cache_block_offsets[:num_seqs],
            self.req_idx_per_token[:num_tokens],
            self.slot_mapping_fp8[:num_tokens],
            self.slot_mapping_scale[:num_tokens],
            self.gen_kv_indptr[: num_seqs + 1],
            self.gen_cached_token_indptr[: num_seqs + 1],
            num_tokens=num_tokens,
            max_query_len=1 + self.max_draft_tokens,
            tokens_per_block=self._tokens_per_block,
            index_head_dim=index_head_dim,
            quant_block_size=self.indexer_quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

    def _compute_kv_lens_row_reorder(self) -> None:
        """Prepare the longest-job-first GVR row order once per forward step."""
        next_n = self.gen_token_stride
        if self.needs_gvr_prior and self.num_generations * next_n >= 2 * self.num_sms:
            gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs]
            order = torch.argsort(gen_kv_lens, descending=True).to(torch.int32)
            self.kv_lens_row_reorder_buffer[: self.num_generations].copy_(order)
            self.kv_lens_row_reorder = self.kv_lens_row_reorder_buffer[: self.num_generations]
        else:
            self.kv_lens_row_reorder = None

    def update_for_spec_dec(self, refresh_device_metadata: bool = True):
        super().update_for_spec_dec(refresh_device_metadata)
        # host
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1

        # device
        self.on_update_kv_lens()

    # Create buffers for mla_rope_append_paged_kv_assign_q
    def create_buffers_for_mla_rope_append(self, capture_graph=False):
        # New context buffers for dsa
        if not self.enable_context_mla_with_cached_kv:
            self.ctx_cached_token_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1,),
                cache_name="ctx_cached_token_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_cached_token_indptr = torch.zeros_like(
                self.ctx_cached_token_indptr,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.ctx_kv_indptr = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_requests + 1,),
                cache_name="ctx_kv_indptr",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_ctx_kv_indptr = torch.zeros_like(
                self.ctx_kv_indptr,
                device="cpu",
                pin_memory=prefer_pinned(),
            )

        # New generation buffers for dsa
        self.gen_cached_token_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1,),
            cache_name="gen_cached_token_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_cached_token_indptr = torch.zeros_like(
            self.gen_cached_token_indptr,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.gen_kv_indptr = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_requests + 1,),
            cache_name="gen_kv_indptr",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_kv_indptr = torch.zeros_like(
            self.gen_kv_indptr,
            device="cpu",
            pin_memory=prefer_pinned(),
        )

    def _create_nvfp4_mla_generation_buffers(self, capture_graph=False):
        """Allocate stable NVFP4 gather outputs for all MTP generation rows."""
        self.nvfp4_mla_fp8_scratch = None
        self.nvfp4_mla_compact_indices = None
        if getattr(self.kv_cache_manager, "dtype", None) != tensorrt_llm.bindings.DataType.NVFP4:
            return

        # MTP verifies the current token and max_draft_tokens draft tokens for
        # every generation request. Each query row has its own TopK selection,
        # so the dequantized FP8 rows and compact offsets must cover the full
        # verification window. The buffers have stable addresses for CUDA
        # graphs and are reused sequentially by every attention layer.
        max_gen_tokens = self.max_num_sequences * (1 + self.max_draft_tokens)
        self.nvfp4_mla_fp8_scratch = self.get_empty(
            self.cuda_graph_buffers,
            (
                max_gen_tokens,
                self.num_sparse_topk,
                self.kv_cache_manager.head_dim,
            ),
            cache_name="nvfp4_mla_fp8_scratch",
            dtype=torch.float8_e4m3fn,
            capture_graph=capture_graph,
        )
        self.nvfp4_mla_compact_indices = self.get_empty(
            self.cuda_graph_buffers,
            (max_gen_tokens, self.num_sparse_topk),
            cache_name="nvfp4_mla_compact_indices",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    def _create_radix_aux_buffers(self, capture_graph: bool) -> None:
        max_num_rows = self.max_num_sequences * self._radix_rows_per_sequence
        radix_shape = (max_num_rows, _MAX_RADIX_BLOCKS_PER_ROW, self.num_sparse_topk)
        self.radix_aux_indices = self.get_empty(
            self.cuda_graph_buffers,
            radix_shape,
            cache_name="radix_aux_indices",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.radix_aux_logits = self.get_empty(
            self.cuda_graph_buffers,
            radix_shape,
            cache_name="radix_aux_logits",
            dtype=torch.float32,
            capture_graph=capture_graph,
        )

    def create_buffers_for_indexer(self, capture_graph=False):
        sparse_metadata_params = self.sparse_metadata_params
        if not isinstance(sparse_metadata_params, DSAMetadataParams):
            raise ValueError("DSA sparse attention metadata params are not set")
        # A cache whose index-key pages outnumber its blocks (DKV's whole index keys)
        # states the table's width.
        indexer_columns = getattr(
            self.kv_cache_manager,
            "indexer_max_blocks_per_seq",
            self.kv_cache_manager.max_blocks_per_seq,
        )
        self.indexer_k_cache_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            [self.max_num_sequences, indexer_columns],
            cache_name="indexer_k_cache_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_indexer_k_cache_block_offsets = torch.zeros_like(
            self.indexer_k_cache_block_offsets,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # Local TopK reused across layers or MTP draft steps.
        if (
            sparse_metadata_params.has_shared_indexer_layers
            or sparse_metadata_params.mtp_index_share
        ):
            self.shared_topk_indices = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="shared_topk_indices",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        else:
            self.shared_topk_indices = None

        self.indexer_skip_topk = False
        self.in_mtp_draft_loop = False
        self.mtp_num_accepted = None

        # Indexer metadata
        # Separate slot mappings for non-interleaved layout (flat byte indices)
        self.slot_mapping_fp8 = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="slot_mapping_fp8",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_fp8 = torch.zeros_like(
            self.slot_mapping_fp8,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.slot_mapping_scale = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="slot_mapping_scale",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_slot_mapping_scale = torch.zeros_like(
            self.slot_mapping_scale,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.token_positions_cuda = None
        if self.use_fp8_ds_mla:
            self.token_positions_cuda = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name="token_positions_cuda",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        # Allocate separate indexer block-offset and slot-mapping buffers for
        # the draft KV cache manager, mirroring draft_kv_cache_block_offsets:
        # the draft-replay context swaps these in by rebinding, so CUDA graph
        # capture bakes distinct addresses for the target and draft segments
        # and both sides can be refreshed eagerly outside the graph.
        if is_dsa_cache_manager(self.draft_kv_cache_manager):
            self.draft_indexer_k_cache_block_offsets = self.get_empty(
                self.cuda_graph_buffers,
                [self.max_num_sequences, self.draft_kv_cache_manager.max_blocks_per_seq],
                cache_name="draft_indexer_k_cache_block_offsets",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_draft_indexer_k_cache_block_offsets = torch.zeros_like(
                self.draft_indexer_k_cache_block_offsets,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.draft_slot_mapping_fp8 = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name="draft_slot_mapping_fp8",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_draft_slot_mapping_fp8 = torch.zeros_like(
                self.draft_slot_mapping_fp8,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.draft_slot_mapping_scale = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens,),
                cache_name="draft_slot_mapping_scale",
                dtype=torch.int64,
                capture_graph=capture_graph,
            )
            self.host_draft_slot_mapping_scale = torch.zeros_like(
                self.draft_slot_mapping_scale,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        # Only when MLA chunked prefill is enabled, we need to gather the full KV for indexer's logit computation.
        # Allocate these buffers dynamically in Indexer.prepare()
        # based on the actual total_kv_len to save memory.
        if self.enable_context_mla_with_cached_kv:
            self.slot_mapping_fp8_fullkv = None
            self.slot_mapping_scale_fullkv = None
        # Per-token request index buffer for topk_indices conversion
        self.req_idx_per_token = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="req_idx_per_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_req_idx_per_token = torch.empty_like(
            self.req_idx_per_token, device="cpu", pin_memory=prefer_pinned()
        )
        # Block table for topk_indices conversion (shared for context and generation); it holds
        # the index-key pages, as wide as their table.
        self.block_table = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences,
                getattr(
                    self.kv_cache_manager,
                    "indexer_max_blocks_per_seq",
                    self.kv_cache_manager.max_blocks_per_seq,
                ),
            ],
            cache_name="block_table",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        if is_dsa_cache_manager(self.draft_kv_cache_manager):
            self.draft_block_table = self.get_empty(
                self.cuda_graph_buffers,
                [
                    self.max_num_sequences,
                    self.draft_kv_cache_manager.max_blocks_per_seq,
                ],
                cache_name="draft_block_table",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        self.scheduler_metadata_buffer = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # When MTP runs without the expanded-tokens path, the same forward step
        # alternates between full-window calls (next_n == 1 + max_draft_tokens)
        # and per-token draft calls (next_n == 1). The 2D DeepGEMM metadata
        # API encodes next_n into the schedule, so the precomputed schedule
        # for one shape cannot be reused for the other. Maintain a second
        # buffer holding the schedule for the full next_n window; the draft
        # path keeps using `scheduler_metadata_buffer`. Always allocate (a
        # few KB) so transitions in `update_spec_dec_param` don't have to
        # special-case its existence.
        self.scheduler_metadata_buffer_full_next_n = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_full_next_n",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Pre-allocated 2D kv_lens buffer for the new DeepGEMM 2D context_lens
        # API. Shape: (max_num_sequences, 1 + max_draft_tokens). Each row
        # broadcasts the same kv_len across next_n positions; kernel reads a
        # slice per forward. Avoids per-forward .expand().contiguous()
        # allocations that would break CUDA graphs.
        self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
        self.cu_seqlen_ks = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="cu_seqlen_ks",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.cu_seqlen_ke = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_tokens,),
            cache_name="cu_seqlen_ke",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        # Topk indices buffer to support skip indexer for requests with short sequence lengths
        if self.enable_indexer_skip:
            self.topk_indices_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_tokens, self.num_sparse_topk),
                cache_name="topk_indices_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_topk_indices_buffer = torch.zeros_like(
                self.topk_indices_buffer,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        if self.needs_gvr_prior:
            self.gvr_prior_indices = self.get_empty(
                self.cuda_graph_buffers,
                (
                    self.kv_cache_manager.num_local_layers,
                    self.max_num_sequences,
                    self.num_sparse_topk,
                ),
                cache_name="gvr_prior_indices",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.gvr_prior_indices.zero_()
            self.kv_lens_row_reorder_buffer = self.get_empty(
                self.cuda_graph_buffers,
                (self.max_num_sequences,),
                cache_name="kv_lens_row_reorder_buffer",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
        self._create_radix_aux_buffers(capture_graph)
        # Create expanded buffers for MTP support
        self.create_expanded_buffers(capture_graph=capture_graph)

    @property
    def _draft_sizing_cap(self) -> int:
        """Process-wide draft ceiling used only for persistent buffer sizing."""
        cap = getattr(self, "_draft_alloc_cap", None)
        if cap is None:
            return self.max_draft_tokens
        return max(int(cap), self.max_draft_tokens)

    def _create_kv_lens_2d_buffer(self, capture_graph=False):
        """Pre-allocated buffer for the DeepGEMM 2D context_lens API.

        Preparation and on_update_kv_lens fill the persistent backing storage
        in place. Scheduling receives a contiguous view at the active runtime
        query width, which may be smaller than the allocation ceiling.
        """
        self.kv_lens_cuda_2d = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences, 1 + self._draft_sizing_cap),
            cache_name="kv_lens_cuda_2d",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # TODO: remove these expanded buffers when fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def create_expanded_buffers(self, capture_graph=False):
        """Create expanded KV-length and block-table buffers for speculative decoding."""
        self.kv_lens_expanded_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences * (1 + self._draft_sizing_cap),),
            cache_name="kv_lens_expanded_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.kv_lens_expanded_host = torch.zeros_like(
            self.kv_lens_expanded_cuda,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.gen_token_repeats_cuda = None
        self.host_gen_token_repeats = None
        self._ragged_num_rows = 0
        self._attn_num_rows = 0
        self.row_kv_lens_cuda = None
        self.row_kv_lens_host = None
        self.row_kv_correction_cuda = None
        self.row_kv_correction_host = None
        self.row_req_idx_cuda = None
        self.row_req_idx_host = None
        self.attn_row_kv_lens_cuda = None
        self.attn_row_kv_correction_cuda = None
        self.attn_row_req_idx_cuda = None
        self.attn_row_kv_lens_host = None
        self.attn_row_kv_correction_host = None
        self.attn_row_req_idx_host = None
        self.attn_row_request_types_host = None
        self.attn_row_prompt_lens_cuda = None
        self.attn_row_prompt_lens_cpu = None
        self.attn_row_block_offsets = None
        self.ragged_mla_cu_q_rows = None
        self.ragged_mla_cu_kv_seqlens = None
        if self.enable_ragged_verification:
            self._create_ragged_verification_buffers(capture_graph)
        self.block_table_expanded = self.get_empty(
            self.cuda_graph_buffers,
            [
                self.max_num_sequences * (1 + self._draft_sizing_cap),
                self.kv_cache_manager.max_blocks_per_seq,
            ],
            cache_name="block_table_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.host_block_table_expanded = torch.zeros_like(
            self.block_table_expanded,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        if is_dsa_cache_manager(self.draft_kv_cache_manager):
            self.draft_block_table_expanded = self.get_empty(
                self.cuda_graph_buffers,
                [
                    self.max_num_sequences * (1 + self._draft_sizing_cap),
                    self.draft_kv_cache_manager.max_blocks_per_seq,
                ],
                cache_name="draft_block_table_expanded",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self.host_draft_block_table_expanded = torch.zeros_like(
                self.draft_block_table_expanded,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
        self.scheduler_metadata_buffer_expanded = self.get_empty(
            self.cuda_graph_buffers,
            (self.num_sms + 1, 2),
            cache_name="scheduler_metadata_buffer_expanded",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    def _create_ragged_verification_buffers(self, capture_graph: bool) -> None:
        """Allocate persistent row layouts required only by ragged verification."""
        # Stable-address repeat vector used by in-graph ragged expansions.
        self.gen_token_repeats_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (self.max_num_sequences,),
            cache_name="gen_token_repeats_cuda",
            dtype=torch.int64,
            capture_graph=capture_graph,
        )
        self.host_gen_token_repeats = torch.empty_like(
            self.gen_token_repeats_cuda,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        row_cap = self.max_num_sequences * (1 + self._draft_sizing_cap)
        for name in ("ragged_mla_cu_q_rows", "ragged_mla_cu_kv_seqlens"):
            prefix = self.get_empty(
                self.cuda_graph_buffers,
                (row_cap + 1,),
                cache_name=name,
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            prefix[:1].zero_()
            setattr(self, name, prefix)
        self.row_kv_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (row_cap,),
            cache_name="row_kv_lens_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.row_kv_lens_host = torch.zeros_like(
            self.row_kv_lens_cuda, device="cpu", pin_memory=prefer_pinned()
        )
        self.row_kv_correction_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (row_cap,),
            cache_name="row_kv_correction_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.row_kv_correction_host = torch.zeros_like(
            self.row_kv_correction_cuda, device="cpu", pin_memory=prefer_pinned()
        )
        self.row_req_idx_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (row_cap,),
            cache_name="row_req_idx_cuda",
            dtype=torch.long,
            capture_graph=capture_graph,
        )
        self.row_req_idx_host = torch.zeros_like(
            self.row_req_idx_cuda, device="cpu", pin_memory=prefer_pinned()
        )

        # MLA RoPE and sparse-MLA generation consume one row per query token;
        # every other consumer keeps using request-major metadata.
        attn_rows_cap = self.max_num_sequences * (2 + self._draft_sizing_cap)
        self.attn_row_kv_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (attn_rows_cap,),
            cache_name="attn_row_kv_lens_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.attn_row_kv_correction_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (attn_rows_cap,),
            cache_name="attn_row_kv_correction_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.attn_row_req_idx_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (attn_rows_cap,),
            cache_name="attn_row_req_idx_cuda",
            dtype=torch.long,
            capture_graph=capture_graph,
        )
        self.attn_row_kv_lens_host = torch.zeros(
            (attn_rows_cap,), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self.attn_row_kv_correction_host = torch.zeros(
            (attn_rows_cap,), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self.attn_row_req_idx_host = torch.zeros(
            (attn_rows_cap,), dtype=torch.long, pin_memory=prefer_pinned()
        )
        self.attn_row_request_types_host = torch.ones(
            (attn_rows_cap,), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self.attn_row_prompt_lens_cuda = self.get_empty(
            self.cuda_graph_buffers,
            (attn_rows_cap,),
            cache_name="attn_row_prompt_lens_cuda",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
        self.attn_row_prompt_lens_cpu = torch.zeros(
            (attn_rows_cap,), dtype=torch.int32, pin_memory=prefer_pinned()
        )
        num_attention_op_pools = getattr(
            self.kv_cache_manager,
            "num_attention_op_pools",
            self.kv_cache_manager.num_pools,
        )
        self.attn_row_block_offsets = self.get_empty(
            self.cuda_graph_buffers,
            (
                num_attention_op_pools,
                attn_rows_cap,
                2,
                self.kv_cache_manager.max_blocks_per_seq,
            ),
            cache_name="attn_row_block_offsets",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )

    # This function is only used to create the expanded buffers when the max_draft_tokens is changed.
    # TODO: remove this function once fp8_paged_mqa_logits supports an arbitrary number of MTP draft tokens.
    def update_spec_dec_param(
        self,
        batch_size,
        is_spec_decoding_enabled,
        is_spec_dec_tree,
        is_spec_dec_dynamic_tree,
        max_draft_len,
        max_total_draft_tokens,
        spec_metadata: Optional["SpecMetadata"] = None,
        spec_tree_manager: Optional["SpecTreeManager"] = None,
        num_contexts: int = 0,
    ):
        """Update speculative decoding parameters and create expanded buffers."""
        super().update_spec_dec_param(
            batch_size,
            is_spec_decoding_enabled,
            is_spec_dec_tree,
            is_spec_dec_dynamic_tree,
            max_draft_len,
            max_total_draft_tokens,
            spec_metadata,
            spec_tree_manager,
            num_contexts=num_contexts,
        )
        self.max_draft_tokens = max_draft_len
        rows_per_sequence = max(1 + max_draft_len, 1 + (max_total_draft_tokens or 0))
        if is_spec_dec_dynamic_tree and spec_tree_manager is not None:
            rows_per_sequence = max(rows_per_sequence, spec_tree_manager._internal_buf_dim)
        resize_radix_workspace = rows_per_sequence != self._radix_rows_per_sequence
        self._radix_rows_per_sequence = rows_per_sequence
        self._draft_alloc_cap = max(
            int(max_total_draft_tokens),
            int(getattr(self, "_draft_alloc_cap", 0) or 0),
        )
        capture_graph = self.is_cuda_graph
        max_gen_tokens = self.max_num_sequences * (1 + self.max_draft_tokens)
        if (
            self.nvfp4_mla_fp8_scratch is not None
            and self.nvfp4_mla_fp8_scratch.shape[0] != max_gen_tokens
        ):
            self._create_nvfp4_mla_generation_buffers(capture_graph=capture_graph)
        previous_cap = int(getattr(self, "_expanded_alloc_draft_cap", -1))
        allocation_cap = max(self._draft_sizing_cap, previous_cap)
        if allocation_cap > previous_cap:
            self._expanded_alloc_draft_cap = allocation_cap
            self._create_kv_lens_2d_buffer(capture_graph=capture_graph)
            self.create_expanded_buffers(capture_graph=capture_graph)
            # One Helix sparse row per verify token.
            self._create_helix_sparse_row_buffers(capture_graph=capture_graph)
        if resize_radix_workspace:
            self._create_radix_aux_buffers(capture_graph)

    def _update_indexer_k_cache_block_offsets(self) -> torch.Tensor:
        """Refresh INDEX_KEY offsets and return their physical pool slots."""
        cache_manager = self.kv_cache_manager
        # Raw block IDs can exceed the primary pool after host-cache offload;
        # the manager resolves their current physical slots.
        pool_indices = cache_manager.get_pool_block_indices(
            self.num_seqs,
            request_ids=self.request_ids,
            num_contexts=self.num_contexts,
            beam_width=self.beam_width,
        )
        page_indices = pool_indices * cache_manager.indexer_k_cache_page_scale
        num_blocks = page_indices.shape[1]
        self.host_indexer_k_cache_block_offsets[: self.num_seqs, :num_blocks].copy_(page_indices)
        if self._staging_commit is not None:
            self._clamp_staged_buffer_once(self.indexer_k_cache_block_offsets)
            self._stage_or_copy(
                self.indexer_k_cache_block_offsets[: self.num_seqs],
                self.host_indexer_k_cache_block_offsets[: self.num_seqs].clamp(min=0),
            )
            return pool_indices
        self.indexer_k_cache_block_offsets[: self.num_seqs].copy_(
            self.host_indexer_k_cache_block_offsets[: self.num_seqs], non_blocking=True
        )
        # Sanitize graph-padding entries that may be stale after cache
        # eviction or host-cache onboarding.
        self.indexer_k_cache_block_offsets.clamp_(min=0)
        return pool_indices

    def set_skip_topk(self, skip: bool) -> None:
        self.indexer_skip_topk = skip

    def set_in_mtp_draft_loop(self, active: bool) -> None:
        self.in_mtp_draft_loop = active

    def set_mtp_num_accepted(self, num_accepted: Optional[torch.Tensor]) -> None:
        self.mtp_num_accepted = num_accepted

    def _invalidate_pool_view_cache(self):
        """Invalidate the cached pool view and related step-invariant values.

        Must be called at the start of each forward step (in prepare()) so that
        _ensure_pool_view_cached() recomputes them for the new batch.
        """
        self._pool_cache_valid = False
        # The grouped remap batches (leader-written, follower-read) are only
        # valid within one forward. Drop them at the step boundary so a follower
        # can never read a previous step's batch, and so the tensors are not
        # retained across idle steps. Python-only -> safe under graph replay
        # (replay does not re-run this).
        self._group_remap_batched.clear()
        self.nvfp4_mla_context_fp8_scratch = None

    def _ensure_pool_view_cached(self, local_layer_idx: int | None = None):
        """Compute and cache values used by
        transform_local_topk_and_prepare_pool_view().

        These values (pool view, block table slices, and request index slices)
        are constant across all layers sharing the same KV pool and batch
        dimensions within a forward pass. Caching them avoids redundant
        Python/CUDA overhead per layer.

        Safety: _invalidate_pool_view_cache() is called unconditionally at the
        start of every step (prepare() and on_update_kv_lens()), so the boolean
        flag is always cleared before the first per-layer call within a step.
        """
        kv_cache_manager = self.kv_cache_manager
        if local_layer_idx is None:
            # The manager's one primary pool, of the manager's KV dtype.
            pool_key = (id(kv_cache_manager), None)
            if self._pool_cache_valid and self._cached_pool_key == pool_key:
                return
            pool = kv_cache_manager.get_unique_primary_pool()
            packed_fp4 = kv_cache_manager.dtype == tensorrt_llm.bindings.DataType.NVFP4
        else:
            # A manager with a pool per storage dtype: the layer's pool, which carries its own
            # dtype (FP4 is packed int8).
            pool = kv_cache_manager.get_primary_pool(local_layer_idx)
            pool_key = (id(kv_cache_manager), pool.data_ptr())
            if self._pool_cache_valid and self._cached_pool_key == pool_key:
                return
            packed_fp4 = pool.dtype == torch.int8

        num_blocks, num_layers = pool.shape[:2]
        self._cached_tokens_per_block = kv_cache_manager.tokens_per_block
        head_dim = kv_cache_manager.head_dim
        self._cached_num_pool_tokens = num_blocks * num_layers * self._cached_tokens_per_block
        if packed_fp4:
            # FP4 packs two values per byte and therefore cannot use the
            # ordinary [token, head_dim] view. Its gather op reads the raw
            # data/scale pools through their stable host pointers instead.
            self._cached_pool_view = None
        else:
            self._cached_pool_view = pool.squeeze(2).view(-1, 1, head_dim)
        self._cached_block_table_ctx = self.block_table[: self.num_contexts]
        self._cached_block_table_gen = self.block_table[self.num_contexts : self.num_seqs]
        self._cached_req_idx_ctx = self.req_idx_per_token[: self.num_ctx_tokens]
        self._cached_req_idx_gen = (
            self.req_idx_per_token[self.num_ctx_tokens : self.num_tokens] - self.num_contexts
        )
        self._cached_kv_mgr_id = id(kv_cache_manager)
        self._cached_pool_key = pool_key
        self._pool_cache_valid = True

    def _ensure_group_remap_struct(self):
        """Build (once) the static full+shared indexer group layout used by the
        cross-layer fan-out remap (``TRTLLM_DISABLE_DSA_GROUP_REMAP``).

        Groups are runs of consecutive local layers starting at a full-indexer
        layer (``indexer_k_cache_local_layer_mask[L] == True``, i.e.
        ``is_full_indexer_layer``); the shared layers that follow reuse the
        leader's top-k. Within a group, the remap output for each member differs
        from the leader's only by the additive ``layer_offset * tokens_per_block``
        term, so one launch (grid.z = group_size) covers the whole group.

        A group is marked active only when it has >= 2 members (a genuine
        fan-out) and a uniform primary-pool page scale (required for a single
        stride_factor). Singleton / non-uniform groups fall back to the
        per-layer single-op path (identical to flag-off behavior).
        """
        kvm = self.kv_cache_manager
        if self._group_remap_struct is not None and self._group_remap_struct_kv_id == id(kvm):
            return self._group_remap_struct

        # Without shared-indexer layers every group is a singleton. Skipping
        # the scan also keeps hybrid managers, whose local layers include
        # recurrent layers that own no indexer, from forming groups.
        sparse_metadata_params = getattr(self, "sparse_metadata_params", None)
        if sparse_metadata_params is not None and not getattr(
            sparse_metadata_params, "has_shared_indexer_layers", True
        ):
            self._group_remap_struct = {}
            self._group_remap_struct_kv_id = id(kvm)
            return self._group_remap_struct

        # Building the small per-group layer-offset device tensors must happen
        # eagerly (host->device copy), never inside a CUDA-graph capture. TRT-LLM
        # runs an eager warmup before capture, so return an uncached empty struct
        # if somehow reached first during capture; it is (re)built on the next
        # eager step and grouping simply falls back to per-layer until then.
        if torch.cuda.is_current_stream_capturing():
            return {}

        mask = getattr(kvm, "indexer_k_cache_local_layer_mask", None)
        if mask is None:
            self._group_remap_struct = {}
            self._group_remap_struct_kv_id = id(kvm)
            return self._group_remap_struct

        n = len(mask)
        leader_of = [-1] * n
        slot_of = [0] * n
        members = {}  # leader local idx -> [member local idx, ...]
        cur_leader = None
        for local_idx in range(n):
            if mask[local_idx]:
                cur_leader = local_idx
                members[cur_leader] = [local_idx]
            else:
                if cur_leader is None:
                    # Shouldn't happen (layer 0 forced full); be safe.
                    cur_leader = local_idx
                    members[cur_leader] = [local_idx]
                else:
                    members[cur_leader].append(local_idx)
            leader_of[local_idx] = cur_leader
            slot_of[local_idx] = len(members[cur_leader]) - 1

        # Device for the small per-group layer-offset tensors.
        device = None
        if self.shared_topk_indices is not None:
            device = self.shared_topk_indices.device
        elif self.block_table is not None:
            device = self.block_table.device

        group_active = {}
        group_scale = {}
        group_size = {}
        group_layer_ids = {}
        for leader, member_list in members.items():
            group_size[leader] = len(member_list)
            params = [kvm.get_primary_pool_page_index_params(m) for m in member_list]
            scale0 = params[0][0]
            uniform_scale = all(p[0] == scale0 for p in params)
            offsets = [int(p[1]) for p in params]
            active = len(member_list) >= 2 and uniform_scale and device is not None
            group_active[leader] = active
            group_scale[leader] = int(scale0)
            if active:
                group_layer_ids[leader] = torch.tensor(offsets, dtype=torch.int32, device=device)

        struct = {
            "leader_of": leader_of,
            "slot_of": slot_of,
            "group_active": group_active,
            "group_scale": group_scale,
            "group_size": group_size,
            "group_layer_ids": group_layer_ids,
        }
        self._group_remap_struct = struct
        self._group_remap_struct_kv_id = id(kvm)
        return struct

    @maybe_compile(dynamic=True)
    def _get_dense_topk_indices(self, seq_lens, kv_lens, num_tokens):
        device = kv_lens.device
        past_kv_lens = kv_lens - seq_lens
        # get position ids
        seq_ends = torch.cumsum(seq_lens, dim=0)
        seq_starts = seq_ends - seq_lens
        per_seq_offsets = past_kv_lens - seq_starts  # Shape: [batch_size]
        global_indices = torch.arange(num_tokens, device=device)
        batch_indices = torch.searchsorted(seq_ends, global_indices, side="right")
        repeated_offsets = per_seq_offsets[batch_indices]
        position_ids = global_indices + repeated_offsets
        # get the dense topk indices with causal mask
        range_row = torch.arange(self.num_sparse_topk, device=device)
        mask = range_row <= position_ids.unsqueeze(1)
        return torch.where(mask, range_row, -1)

    def prepare_dense_topk_indices(self, kv_lens, device=False):  # device=False means use CPU
        """Prepare dense TopK indices for short sequences that skip the indexer."""

        if self.num_contexts > 0 and self.skip_indexer_for_ctx_reqs:
            ctx_range = slice(self.num_ctx_tokens)
            if device:
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[: self.num_contexts],
                        kv_lens[: self.num_contexts],
                        self.num_ctx_tokens,
                    ),
                    non_blocking=True,
                )
            else:
                self.host_topk_indices_buffer[ctx_range, :] = self._get_dense_topk_indices(
                    self.seq_lens[: self.num_contexts],
                    kv_lens[: self.num_contexts],
                    self.num_ctx_tokens,
                )
                self.topk_indices_buffer[ctx_range, :].copy_(
                    self.host_topk_indices_buffer[ctx_range, :], non_blocking=True
                )

        if self.num_generations > 0 and self.skip_indexer_for_gen_reqs:
            gen_range = slice(self.num_ctx_tokens, self.num_tokens)
            if device:
                self.topk_indices_buffer[gen_range, :].copy_(
                    self._get_dense_topk_indices(
                        self.seq_lens_cuda[self.num_contexts : self.num_seqs],
                        kv_lens[self.num_contexts : self.num_seqs],
                        self.num_tokens - self.num_ctx_tokens,
                    ),
                    non_blocking=True,
                )
            else:
                self.host_topk_indices_buffer[gen_range, :] = self._get_dense_topk_indices(
                    self.seq_lens[self.num_contexts : self.num_seqs],
                    kv_lens[self.num_contexts : self.num_seqs],
                    self.num_tokens - self.num_ctx_tokens,
                )
                self.topk_indices_buffer[gen_range, :].copy_(
                    self.host_topk_indices_buffer[gen_range, :], non_blocking=True
                )

    def prepare_for_spec_decode(self, kv_lens: torch.Tensor):
        # The DeepGEMM paged-MQA kernel (fp8_paged_mqa_logits) runs a native
        # next_n >= 1 on sm100+: its scheduler tiles the query tokens into
        # BLOCK_Q-sized blocks (num_q_blocks = ceil_div(num_q_tokens, BLOCK_Q)),
        # so any MTP depth is handled without a per-draft-token Q flatten /
        # kv_lens / block_table expansion on Blackwell. sm90 still lacks native
        # MTP support (seq_len 1/2 only) and must expand for max_draft_tokens > 1.
        # TODO:
        # - Drop this sm90 branch (and the expanded buffers) once
        #   fp8_paged_mqa_logits supports an arbitrary next_n on sm90 too.
        use_dsl = self.use_cute_dsl_paged_mqa_logits
        if not self.is_ragged_verify:
            self._ragged_num_rows = 0
            self._attn_num_rows = 0
        else:
            # The DSL paged-MQA path reconstructs rows from one scalar window
            # and cannot express a per-request split.
            use_dsl = False
            assert not self.is_spec_decoding_enabled, (
                "ragged verification requires dense spec-decoding metadata to be disabled"
            )

        helix_spec = self.enable_helix and self._helix_spec_tokens_valid
        if helix_spec:
            # Each Helix verify token has its own rank-local KV bound
            # (helix_kv_bounds), which the in-kernel next_n stagger cannot
            # express, so score one row per token with explicit lengths.
            use_dsl = False
        self.use_expanded_buffers_for_mtp = not use_dsl and (
            helix_spec
            or self.is_ragged_verify
            or (self.max_draft_tokens > 1 and get_sm_version() == 90)
        )
        if self.use_expanded_buffers_for_mtp:
            if self.is_ragged_verify:
                n_gen = self.num_generations
                self.host_gen_token_repeats[:n_gen].copy_(
                    torch.tensor(self.gen_token_repeat_list(), dtype=torch.int64)
                )
                self.gen_token_repeats_cuda[:n_gen].copy_(
                    self.host_gen_token_repeats[:n_gen], non_blocking=True
                )

            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
            gen_kv_lens_expanded, num_tokens = self.expand_per_gen_token(gen_kv_lens)
            self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
            self._stage_or_copy(
                self.kv_lens_expanded_cuda[:num_tokens], self.kv_lens_expanded_host[:num_tokens]
            )

            if self.kv_cache_manager is not None and self.num_generations > 0:
                max_len = self.host_indexer_k_cache_block_offsets.shape[1]
                gen_blocks = self.host_indexer_k_cache_block_offsets[
                    self.num_contexts : self.num_seqs, :max_len
                ]
                expanded_blocks, _ = self.expand_per_gen_token(gen_blocks, dim=0)
                self.host_block_table_expanded[:num_tokens, :max_len].copy_(expanded_blocks)
                if self._staging_commit is not None:
                    self._clamp_staged_buffer_once(self.block_table_expanded)
                    self._stage_or_copy(
                        self.block_table_expanded[:num_tokens],
                        self.host_block_table_expanded[:num_tokens].clamp(min=0),
                    )
                else:
                    self.block_table_expanded[:num_tokens].copy_(
                        self.host_block_table_expanded[:num_tokens], non_blocking=True
                    )
                    self.block_table_expanded.clamp_(min=0)

            if self.is_ragged_verify:
                self._prepare_ragged_row_kv_lens(kv_lens)
                self._prepare_token_major_gen_rows(kv_lens)

        self.expand_for_dsl = (
            use_dsl and self.kv_cache_manager is not None and self.max_draft_tokens >= 1
        )
        if self.expand_for_dsl and self.num_generations > 0:
            next_n = self.gen_token_stride
            kernel_atoms = (1, 2, 3) if self.kv_cache_manager.use_fp4 else (1, 2, 3, 4)
            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[self.num_contexts : self.num_seqs])
            max_ctx = int(gen_kv_lens.max().item()) if gen_kv_lens.numel() else 0
            expand_factor, atom, kernel_split_atoms = _pick_dsl_split(
                next_n,
                batch_size=self.num_generations,
                max_ctx=max_ctx,
                num_sms=self.num_sms,
                kernel_atoms=kernel_atoms,
            )
            self.dsl_expand_factor = expand_factor
            self.dsl_atom = atom
            self.dsl_kernel_atoms = kernel_split_atoms
            self.dsl_kernel_next_n = next_n
            if kernel_split_atoms > 1:
                offsets = _atom_ctx_offsets(next_n, kernel_split_atoms)
                if self.dsl_atom_ctx_offsets_key != (next_n, kernel_split_atoms):
                    # Built outside graph capture; on_update_kv_lens only subtracts.
                    self.dsl_atom_ctx_offsets = torch.tensor(
                        offsets, dtype=torch.int32, device="cuda"
                    )
                    self.dsl_atom_ctx_offsets_key = (next_n, kernel_split_atoms)
                num_rows = self.num_generations * kernel_split_atoms
                atom_lens = gen_kv_lens.unsqueeze(1) - torch.tensor(
                    offsets, dtype=gen_kv_lens.dtype, device=gen_kv_lens.device
                )
                atom_lens = atom_lens.reshape(-1)
                self.kv_lens_expanded_host[:num_rows].copy_(atom_lens)
                self.kv_lens_expanded_cuda[:num_rows].copy_(
                    self.kv_lens_expanded_host[:num_rows], non_blocking=True
                )
            if expand_factor > 1:
                num_tokens = self.num_generations * expand_factor
                gen_kv_lens_expanded = gen_kv_lens.repeat_interleave(expand_factor)
                self.kv_lens_expanded_host[:num_tokens].copy_(gen_kv_lens_expanded)
                self.kv_lens_expanded_cuda[:num_tokens].copy_(
                    self.kv_lens_expanded_host[:num_tokens], non_blocking=True
                )
                self._refresh_expanded_block_table(expand_factor)
        else:
            self.dsl_expand_factor = 1
            self.dsl_atom = self.gen_token_stride
            self.dsl_kernel_atoms = 1

    def _prepare_ragged_row_kv_lens(self, kv_lens: torch.Tensor) -> None:
        """Populate the causal KV extent of each ragged generation row."""
        verify_lens = self.ragged_verify_lens
        if not verify_lens:
            return
        gen_kv_lens = kv_lens[self.num_contexts : self.num_seqs].tolist()
        assert len(gen_kv_lens) == len(verify_lens), (
            f"ragged verify lengths {len(verify_lens)} != generation requests {len(gen_kv_lens)}"
        )
        expected_gen_tokens = self.num_tokens - self.num_ctx_tokens
        assert sum(verify_lens) == expected_gen_tokens, (
            f"ragged verify windows sum to {sum(verify_lens)} but this step has "
            f"{expected_gen_tokens} generation tokens"
        )

        rows: List[int] = []
        corrections: List[int] = []
        request_indices: List[int] = []
        for request_idx, (kv_len, verify_len) in enumerate(zip(gen_kv_lens, verify_lens)):
            verify_len = int(verify_len)
            base = int(kv_len) - verify_len + 1
            rows.extend(range(base, base + verify_len))
            corrections.extend(range(1 - verify_len, 1))
            request_indices.extend([request_idx] * verify_len)

        num_rows = len(rows)
        assert num_rows <= self.row_kv_lens_cuda.shape[0]
        self._ragged_num_rows = num_rows
        self.row_kv_lens_host[:num_rows].copy_(torch.tensor(rows, dtype=torch.int32))
        self.row_kv_correction_host[:num_rows].copy_(torch.tensor(corrections, dtype=torch.int32))
        self.row_req_idx_host[:num_rows].copy_(torch.tensor(request_indices, dtype=torch.long))
        self.row_kv_lens_cuda[:num_rows].copy_(self.row_kv_lens_host[:num_rows], non_blocking=True)
        self.row_kv_correction_cuda[:num_rows].copy_(
            self.row_kv_correction_host[:num_rows], non_blocking=True
        )
        self.row_req_idx_cuda[:num_rows].copy_(self.row_req_idx_host[:num_rows], non_blocking=True)

    def refresh_ragged_row_kv_lens(self) -> None:
        """Refresh ragged extents after overlap changes ``kv_lens_cuda``."""
        num_rows = self._ragged_num_rows
        if not self.is_ragged_verify or num_rows <= 0:
            return
        gen_kv_lens = self.kv_lens_cuda[self.num_contexts : self.num_seqs]
        row_kv_lens = self.row_kv_lens_cuda[:num_rows]
        torch.index_select(
            gen_kv_lens,
            0,
            self.row_req_idx_cuda[:num_rows],
            out=row_kv_lens,
        )
        row_kv_lens.add_(self.row_kv_correction_cuda[:num_rows])

    def _prepare_token_major_gen_rows(self, kv_lens: torch.Tensor) -> None:
        """Build the temporary one-row-per-query-token attention view."""
        verify_lens = self.ragged_verify_lens
        num_gen_rows = self._ragged_num_rows
        if not verify_lens or num_gen_rows <= 0:
            self._attn_num_rows = 0
            return

        num_contexts = self.num_contexts
        num_rows = num_contexts + num_gen_rows
        capacity = self.attn_row_kv_lens_cuda.shape[0]
        assert num_rows <= capacity, (
            f"token-major generation needs {num_rows} rows but capacity is {capacity}"
        )

        if num_contexts:
            self.attn_row_kv_lens_host[:num_contexts].copy_(kv_lens[:num_contexts].to(torch.int32))
            self.attn_row_kv_correction_host[:num_contexts].zero_()
            self.attn_row_req_idx_host[:num_contexts].copy_(
                torch.arange(num_contexts, dtype=torch.long)
            )
            self.attn_row_prompt_lens_cpu[:num_contexts].copy_(
                self.prompt_lens_cpu[:num_contexts].to(torch.int32)
            )

        self.attn_row_kv_lens_host[num_contexts:num_rows].copy_(
            self.row_kv_lens_host[:num_gen_rows]
        )
        self.attn_row_kv_correction_host[num_contexts:num_rows].copy_(
            self.row_kv_correction_host[:num_gen_rows]
        )
        torch.add(
            self.row_req_idx_host[:num_gen_rows],
            num_contexts,
            out=self.attn_row_req_idx_host[num_contexts:num_rows],
        )
        self.attn_row_prompt_lens_cpu[num_contexts:num_rows].fill_(1)
        if num_contexts:
            self.attn_row_request_types_host[:num_contexts].zero_()
        self.attn_row_request_types_host[num_contexts:num_rows].fill_(1)

        self.attn_row_kv_correction_cuda[:num_rows].copy_(
            self.attn_row_kv_correction_host[:num_rows], non_blocking=True
        )
        self.attn_row_kv_lens_cuda[:num_rows].copy_(
            self.attn_row_kv_lens_host[:num_rows], non_blocking=True
        )
        self.attn_row_req_idx_cuda[:num_rows].copy_(
            self.attn_row_req_idx_host[:num_rows], non_blocking=True
        )
        self.attn_row_prompt_lens_cuda[:num_rows].copy_(
            self.attn_row_prompt_lens_cpu[:num_rows], non_blocking=True
        )
        self._attn_num_rows = num_rows
        self.refresh_token_major_block_table()

    def refresh_token_major_gen_rows(self) -> None:
        """Refresh token-major KV extents after overlap correction."""
        num_rows = self._attn_num_rows
        if not self.is_ragged_verify or num_rows <= 0:
            return
        torch.index_select(
            self.kv_lens_cuda[: self.num_seqs],
            0,
            self.attn_row_req_idx_cuda[:num_rows],
            out=self.attn_row_kv_lens_cuda[:num_rows],
        )
        self.attn_row_kv_lens_cuda[:num_rows].add_(self.attn_row_kv_correction_cuda[:num_rows])

    def refresh_token_major_block_table(self) -> None:
        """Expand the main attention block table along its sequence axis."""
        num_rows = self._attn_num_rows
        if num_rows <= 0 or self.kv_cache_block_offsets is None:
            return
        source = self.kv_cache_block_offsets
        num_pools = min(source.shape[0], self.attn_row_block_offsets.shape[0])
        width = min(source.shape[-1], self.attn_row_block_offsets.shape[-1])
        torch.index_select(
            source[:num_pools, :, :, :width],
            1,
            self.attn_row_req_idx_cuda[:num_rows],
            out=self.attn_row_block_offsets[:num_pools, :num_rows, :, :width],
        )

    def token_major_gen_view(self, token_sparse: bool = False) -> Optional[TokenMajorGenView]:
        """Return the ragged generation view expected by ``trtllm.py``.

        Under Helix, token-sparse MLA is presented one row per generation token
        instead, each row as long as its kept Top-K entries; see
        ``_create_helix_sparse_row_buffers``.
        """
        if token_sparse and not self.is_ragged_verify and self._helix_num_rows > 0:
            num_rows = self._helix_num_rows
            return TokenMajorGenView(
                num_rows=num_rows,
                sequence_length=self.helix_row_kv_lens_cuda[:num_rows],
                host_past_key_value_lengths=self.helix_row_kv_lens_host[:num_rows],
                host_context_lengths=self.helix_row_prompt_lens_cpu[:num_rows],
                prompt_lens_cuda=self.helix_row_prompt_lens_cuda[:num_rows],
                host_request_types=self.helix_row_request_types_host[:num_rows],
                kv_cache_block_offsets=self.helix_row_block_offsets[:, :num_rows],
                max_num_rows=self.helix_row_kv_lens_cuda.shape[0],
            )
        num_rows = self._attn_num_rows
        if not self.is_ragged_verify or num_rows <= 0:
            return None
        return TokenMajorGenView(
            num_rows=num_rows,
            sequence_length=self.attn_row_kv_lens_cuda[:num_rows],
            host_past_key_value_lengths=self.attn_row_kv_lens_host[:num_rows],
            host_context_lengths=self.attn_row_prompt_lens_cpu[:num_rows],
            prompt_lens_cuda=self.attn_row_prompt_lens_cuda[:num_rows],
            host_request_types=self.attn_row_request_types_host[:num_rows],
            kv_cache_block_offsets=self.attn_row_block_offsets[:, :num_rows],
            max_num_rows=self.attn_row_kv_lens_cuda.shape[0],
        )

    def ragged_row_kv_lens(self, num_tokens: int) -> Optional[torch.Tensor]:
        if not self.is_ragged_verify:
            return None
        assert num_tokens == self._ragged_num_rows, (
            f"ragged row count moved between prepare ({self._ragged_num_rows}) "
            f"and forward ({num_tokens})"
        )
        return self.row_kv_lens_cuda[:num_tokens]

    def apply_device_ragged_layout(
        self,
        verify_lens: torch.Tensor,
        req_idx: torch.Tensor,
        kv_correction: torch.Tensor,
    ) -> None:
        """Install device-selected windows into stable captured buffers."""
        self._validate_ragged_verification()
        if not self.is_ragged_verify:
            raise RuntimeError("device ragged layout requires enabled ragged verification")
        num_contexts = self.num_contexts
        num_rows = self._ragged_num_rows
        num_generations = self.num_generations
        self.seq_lens_cuda[num_contexts : self.num_seqs] = verify_lens.to(self.seq_lens_cuda.dtype)
        self.gen_token_repeats_cuda[:num_generations] = verify_lens.to(torch.int64)
        self.row_req_idx_cuda[:num_rows] = req_idx
        self.row_kv_correction_cuda[:num_rows] = kv_correction
        attn_row_start = num_contexts
        attn_row_end = attn_row_start + num_rows
        self.attn_row_req_idx_cuda[attn_row_start:attn_row_end] = req_idx + num_contexts
        self.attn_row_kv_correction_cuda[attn_row_start:attn_row_end] = kv_correction
        token_start = self.num_ctx_tokens
        token_end = token_start + num_rows
        self.req_idx_per_token[token_start:token_end] = (req_idx + num_contexts).to(
            self.req_idx_per_token.dtype
        )
        self.refresh_token_major_block_table()
        width = min(
            self.indexer_k_cache_block_offsets.shape[-1],
            self.block_table_expanded.shape[-1],
        )
        self.block_table_expanded[:num_rows, :width] = (
            self.indexer_k_cache_block_offsets[num_contexts : self.num_seqs, :width]
            .index_select(0, req_idx)
            .clamp_(min=0)
        )

    def _refresh_expanded_block_table(self, repeat_factor: Optional[int] = None):
        """Refresh the active cache's expanded INDEX_KEY page table."""
        if self.kv_cache_manager is None or self.num_generations == 0:
            return
        use_runtime_layout = False
        if repeat_factor is None:
            if self.use_expanded_buffers_for_mtp:
                use_runtime_layout = True
            elif self.expand_for_dsl and self.dsl_expand_factor > 1:
                repeat_factor = self.dsl_expand_factor
            else:
                return

        max_len = self.host_indexer_k_cache_block_offsets.shape[1]
        gen_block_tensor = self.host_indexer_k_cache_block_offsets[
            self.num_contexts : self.num_seqs, :max_len
        ]
        if use_runtime_layout:
            expanded_blocks, num_tokens = self.expand_per_gen_token(gen_block_tensor, dim=0)
        else:
            num_tokens = self.num_generations * repeat_factor
            expanded_blocks = gen_block_tensor.repeat_interleave(repeat_factor, dim=0)
        self.host_block_table_expanded[:num_tokens, :max_len].copy_(
            expanded_blocks, non_blocking=True
        )
        self.block_table_expanded[:num_tokens].copy_(
            self.host_block_table_expanded[:num_tokens], non_blocking=True
        )
        self.block_table_expanded.clamp_(min=0)

    def prepare_for_indexer_k_cache(self):
        if self.kv_cache_manager is None:
            return
        # Keep physical slots for the primary-pool TopK conversion below.
        pool_indices = self._update_indexer_k_cache_block_offsets()

        # Build block_table for topk_indices conversion (actual block allocation)
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq,
            dtype=torch.int,
            device="cpu",
        )
        if self.enable_helix:
            kv_lens = self._helix_local_kv_lens(cached_token_lens)
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv
        # A cache whose index-key table column spans several pages (DKV's sharded index keys:
        # one column per DKV block) states the tokens of a column.
        tokens_per_block = getattr(
            self.kv_cache_manager,
            "indexer_tokens_per_block",
            self.kv_cache_manager.tokens_per_block,
        )
        num_blocks_per_seq = (kv_lens[: self.num_seqs] + tokens_per_block - 1) // tokens_per_block
        max_blocks_used = num_blocks_per_seq.max().item() if self.num_seqs > 0 else 1
        # pool_indices already has correct values; set padding to -1.
        # Stage through a fresh pinned buffer: an async H2D from pageable
        # memory would block the host behind the busy execution stream.
        host_block_table = torch.empty(
            (pool_indices.shape[0], max_blocks_used),
            dtype=pool_indices.dtype,
            pin_memory=prefer_pinned(),
        )
        host_block_table.copy_(pool_indices[:, :max_blocks_used])
        pad_cols = torch.arange(max_blocks_used, dtype=num_blocks_per_seq.dtype)
        host_block_table.masked_fill_(
            pad_cols.unsqueeze(0) >= num_blocks_per_seq[: self.num_seqs].unsqueeze(1), -1
        )
        # Copy to GPU
        self.block_table[: self.num_seqs, :max_blocks_used].copy_(
            host_block_table, non_blocking=True
        )

    def prepare_for_skip_indexer(self, kv_lens: torch.Tensor):
        num_extra_kv_tokens = self.kv_cache_params.num_extra_kv_tokens
        if self.num_contexts > 0 and self.enable_indexer_skip:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_ctx_reqs = (
                kv_lens[: self.num_contexts].max().item()
                <= self.num_sparse_topk - num_extra_kv_tokens
            )
        else:
            self.skip_indexer_for_ctx_reqs = False

        # Under Helix kv_lens are this rank's share of each sequence, which does
        # not bound the global Top-K; the indexer runs and merge_helix_topk
        # reduces the per-rank selections instead.
        if self.num_generations > 0 and self.enable_indexer_skip and not self.enable_helix:
            # Minus the number of extra KV tokens because when using one-model MTP, the
            # draft layers needs more KV tokens for the next draft forwards.
            self.skip_indexer_for_gen_reqs = (
                kv_lens[self.num_contexts : self.num_seqs].max().item()
                <= self.num_sparse_topk - num_extra_kv_tokens
            )
        else:
            self.skip_indexer_for_gen_reqs = False
        self.prepare_dense_topk_indices(kv_lens)

    def prepare_for_mla_rope_append(self, cached_token_lens: torch.Tensor, kv_lens: torch.Tensor):
        if self.num_contexts > 0:
            self.num_ctx_cached_tokens = cached_token_lens[: self.num_contexts].sum().item()
            self.num_ctx_mla_kv_tokens = kv_lens[: self.num_contexts].sum().item()
            self.max_ctx_kv_len = kv_lens[: self.num_contexts].max().item()
            self.max_ctx_seq_len = self.seq_lens[: self.num_contexts].max().item()
            # context cached token indptr
            torch.cumsum(
                cached_token_lens[: self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_cached_token_indptr[1 : self.num_contexts + 1],
            )
            self.ctx_cached_token_indptr[: self.num_contexts + 1].copy_(
                self.host_ctx_cached_token_indptr[: self.num_contexts + 1], non_blocking=True
            )
            # context kv indptr
            torch.cumsum(
                kv_lens[: self.num_contexts],
                dim=0,
                dtype=torch.int64,
                out=self.host_ctx_kv_indptr[1 : self.num_contexts + 1],
            )
            self.ctx_kv_indptr[: self.num_contexts + 1].copy_(
                self.host_ctx_kv_indptr[: self.num_contexts + 1], non_blocking=True
            )
        else:
            self.num_ctx_cached_tokens = 0
            self.num_ctx_mla_kv_tokens = 0
            self.max_ctx_kv_len = 0
            self.max_ctx_seq_len = 0

        if self.num_generations > 0:
            self.max_gen_seq_len = self.seq_lens[self.num_contexts : self.num_seqs].max().item()
            # generation cached token indptr
            torch.cumsum(
                cached_token_lens[self.num_contexts : self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_cached_token_indptr[1 : self.num_generations + 1],
            )
            self._stage_or_copy(
                self.gen_cached_token_indptr[: self.num_generations + 1],
                self.host_gen_cached_token_indptr[: self.num_generations + 1],
            )
            # generation kv indptr
            torch.cumsum(
                kv_lens[self.num_contexts : self.num_seqs],
                dim=0,
                dtype=torch.int64,
                out=self.host_gen_kv_indptr[1 : self.num_generations + 1],
            )
            self._stage_or_copy(
                self.gen_kv_indptr[: self.num_generations + 1],
                self.host_gen_kv_indptr[: self.num_generations + 1],
            )
        else:
            self.max_gen_seq_len = 0

    def prepare_for_indices_conversion(self):
        # Build req_idx_per_token for topk_indices conversion
        # Use pinned staging buffer to avoid pageable H2D memcpy
        self.host_req_idx_per_token[: self.num_tokens] = torch.repeat_interleave(
            torch.arange(self.num_seqs, dtype=torch.int32),
            self.seq_lens,
            dim=0,
        )
        self._stage_or_copy(
            self.req_idx_per_token[: self.num_tokens],
            self.host_req_idx_per_token[: self.num_tokens],
        )
