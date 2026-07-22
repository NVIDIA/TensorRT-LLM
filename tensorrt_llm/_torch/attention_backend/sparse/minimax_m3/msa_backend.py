# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA-backed MiniMax-M3 sparse attention on the TrtllmAttention stack.

  * MiniMaxM3MsaSparseAttention subclasses TrtllmAttention and reuses its
    inherited forward, overriding only the sparse hooks and owning an
    MsaIndexer.
  * The main sparse GQA runs through the registered MsaSparseGqaFmha.
  * The indexer calls fmha_sm100 directly to produce the per-query selected
    block indices, which the model layer threads through
    forward_args.topk_indices.
  * MiniMaxM3MsaSparseAttentionMetadata subclasses TrtllmAttentionMetadata and
    stores its per-forward MSA tensors in CUDA-graph-stable buffers.
    The buffers are allocated once in __post_init__ via
    get_empty(capture_graph=...), and prepare() copies the per-step values
    into them. The standard CUDAGraphRunner clones one metadata per graph
    batch size (create_cuda_graph_metadata), so no per-batch-size cache is
    needed here.

The classes subclass TrtllmAttention and TrtllmAttentionMetadata, imported at
module scope. This is cycle-free because the fmha registry defers its
MsaSparseGqaFmha import (see fmha/registry.py), so trtllm's import chain does
not reach this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata

from .common import (
    MiniMaxM3SparseConfig,
    MiniMaxM3SparseMetadataParams,
    build_paged_kv_slot_mapping,
    write_kv_slots,
)
from .msa_indexer import MsaIndexer
from .msa_utils import (
    MSA_REQUIRED_HEAD_DIM,
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    per_token_valid_blocks,
    require_msa_module,
)


def _cache_device(meta) -> torch.device:
    """Device hosting the paged KV buffers, else the current CUDA device."""
    kv_cache_manager = meta.kv_cache_manager
    if kv_cache_manager is not None:
        try:
            return kv_cache_manager.get_buffers(0).device
        except Exception:
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


# Per-step fmha_sm100 plan tensors that must live in CUDA-graph-stable buffers.
# At num_kv_splits=1 the plan carries no split-KV workspaces, and
# cute_workspace_buffer is the vendor's cached scratch (kept by reference, not
# copied).
_MSA_PLAN_STABLE_KEYS = (
    "packed_work_range",
    "packed_work_info",
    "qo_segment_offsets",
    "kv_segment_offsets",
    "kv_page_indptr",
    "qo_segment_lens",
    "kv_segment_lens",
    "qo_offset",
)
_MSA_PLAN_INT64_KEYS = ("packed_work_range", "packed_work_info")
# fmha_sm100 sizes packed_work_info at 131072 * max(num_kv_splits, 1); forcing
# num_kv_splits=1 pins this worklist width.
_MSA_PACKED_WORK_INFO_LEN = 131072
_MSA_SPLIT_KV_KEYS = (
    "kv_tile_begin_indices",
    "kv_tile_end_indices",
    "kv_split_indices",
    "num_kv_splits_per_row",
    "workspace_o",
    "workspace_lse",
)


class _MsaGraphSafePlan:
    """CUDA-graph-stable mirror of one fmha_sm100 decode plan.

    Owns fixed device buffers for the per-step plan worklists. refresh() copies
    a freshly built plan into them and returns a plan tuple pointing at the
    stable buffers, so the captured fmha_sm100 run reads addresses that do not
    change across replays. Mirrors FlashInfer's fixed indptr/indices buffers.

    Only valid at num_kv_splits=1: the plan then has no split-KV workspaces
    (refresh() asserts this), and cute_workspace_buffer and the scalar fields
    pass through unchanged.
    """

    def __init__(self, metadata, name: str, *, max_batch: int, num_ctas: int, capture_graph: bool):
        buffers = metadata.cuda_graph_buffers
        self._buf = {}
        # Set by refresh(), read through the plan property.
        self._plan: Optional[tuple] = None
        # cute_workspace_buffer must keep a fixed address across steps for the
        # captured graph to replay correctly. Pin it on first use and fail if
        # it moves.
        self._ws_ptr: Optional[int] = None
        for key in _MSA_PLAN_STABLE_KEYS:
            if key == "packed_work_range":
                shape = (num_ctas,)
            elif key == "packed_work_info":
                shape = (_MSA_PACKED_WORK_INFO_LEN,)
            elif key in ("qo_segment_offsets", "kv_segment_offsets", "kv_page_indptr"):
                shape = (max_batch + 1,)
            else:
                shape = (max_batch,)
            dtype = torch.int64 if key in _MSA_PLAN_INT64_KEYS else torch.int32
            self._buf[key] = metadata.get_empty(
                buffers,
                shape,
                cache_name=f"{name}_{key}",
                dtype=dtype,
                capture_graph=capture_graph,
            )

    @property
    def plan(self) -> Optional[tuple]:
        """The current graph-safe plan tuple, or None if no decode plan is live."""
        return self._plan

    def reset(self) -> None:
        """Drop the live plan tuple (e.g. for a prefill/mixed or captured step)."""
        self._plan = None

    def refresh(self, plan_tuple) -> tuple:
        has_mixed, split, batch, decode, prefill = plan_tuple
        if has_mixed:
            raise RuntimeError(
                "MSA decode expects a single (non-mixed) fmha_sm100 plan; a decode "
                "batch must be pure decode."
            )
        for key in _MSA_SPLIT_KV_KEYS:
            if decode.get(key) is not None:
                raise RuntimeError(
                    f"MSA decode plan used split-KV workspace {key!r}; num_kv_splits=1 "
                    "is required for graph-safe decode."
                )
        ws = decode.get("cute_workspace_buffer")
        if ws is not None:
            if self._ws_ptr is None:
                self._ws_ptr = ws.data_ptr()
            elif ws.data_ptr() != self._ws_ptr:
                raise RuntimeError(
                    "cute_workspace_buffer moved across steps; the fmha_sm100 plan "
                    "is not CUDA-graph safe."
                )
        rebuilt = dict(decode)
        for key in _MSA_PLAN_STABLE_KEYS:
            src = decode.get(key)
            if src is None:
                continue
            n = int(src.shape[0])
            dst = self._buf[key]
            if n > dst.shape[0]:
                raise ValueError(
                    f"MSA plan buffer {key} ({dst.shape[0]}) is smaller than the plan tensor ({n})."
                )
            dst[:n].copy_(src, non_blocking=True)
            rebuilt[key] = dst[:n]
        self._plan = (has_mixed, split, batch, rebuilt, prefill)
        return self._plan


@dataclass(init=False)
class MiniMaxM3MsaSparseAttentionMetadata(TrtllmAttentionMetadata):
    """TrtllmAttentionMetadata for MiniMax-M3 MSA sparse layers.

    Tensors read inside the captured forward are CUDA-graph-stable: the
    cache slots (msa_out_cache_loc), page table (msa_kv_indices), and proxy
    scratch (msa_max_score, msa_n_valid_blocks) are allocated once from the
    manager's worst-case geometry. msa_out_cache_loc, msa_kv_indices, and
    msa_n_valid_blocks are refreshed via copy_, while the fmha_sm100 proxy pass
    writes msa_max_score directly (see msa_proxy_max_score_view). Decode-plan
    worklists live on _MsaGraphSafePlan owners, surfaced via msa_decode_*_plan.

    Length inputs to fmha_sm100_plan (msa_qo_lens_cpu, msa_kv_lens_cpu,
    msa_qo_offset_cpu) are host properties of the base seq_lens/kv_lens,
    read only while building plans in prepare() (outside capture), so they
    need no graph-stable storage. Plans are built in _build_step_plans:
    pure-decode batches use the graph-safe owners (msa_decode_*_plan) while
    prefill/mixed batches keep plain eager tuples (msa_eager_*_plan).
    """

    # Graph-stable buffers; consumers slice to the live count at the call
    # site. Filled once the current step's cache write is prepared.
    msa_out_cache_loc: Optional[torch.Tensor] = None
    msa_kv_indices: Optional[torch.Tensor] = None
    msa_max_score: Optional[torch.Tensor] = None
    msa_n_valid_blocks: Optional[torch.Tensor] = None

    # _msa_buffers_ready gates the once-only device buffers;
    # _msa_fields_ready marks that the current step's buffers are populated.
    _msa_buffers_ready: bool = False
    _msa_fields_ready: bool = False
    # Sparse geometry the decode plans need.
    _msa_params: Optional[MiniMaxM3SparseMetadataParams] = None
    # Plan owners, created lazily when the decode plans are first built and
    # reused across steps. Each owns its graph-safe plan buffers and the
    # current refreshed plan tuple.
    _msa_proxy_plan: Optional["_MsaGraphSafePlan"] = None
    _msa_gqa_plan: Optional["_MsaGraphSafePlan"] = None
    _msa_dense_plan: Optional["_MsaGraphSafePlan"] = None
    # Eager (prefill/mixed) plans, plain tuples with no graph-stable buffers
    # since prefill runs eagerly and is never CUDA-graph captured. Built once
    # per step in prepare() and reused by every layer.
    _msa_eager_proxy_plan: Optional[tuple] = None
    _msa_eager_gqa_plan: Optional[tuple] = None
    _msa_eager_dense_plan: Optional[tuple] = None
    # Eager (prefill/mixed) per-token valid-block count. It is layer-invariant
    # (a function of qo/kv lengths and page size), so it is computed on the host
    # and staged to the device once per step via a non-blocking copy_, then
    # reused by every sparse layer's indexer. _msa_eager_all_blocks_empty caches
    # the host-side empty-selection result so the indexer can short-circuit without
    # a device read. _msa_eager_n_valid_buf is the persistent backing store for the view.
    _msa_eager_n_valid_buf: Optional[torch.Tensor] = None
    _msa_eager_n_valid_blocks: Optional[torch.Tensor] = None
    _msa_eager_all_blocks_empty: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        params = self.sparse_metadata_params
        self._msa_params = params if isinstance(params, MiniMaxM3SparseMetadataParams) else None
        # See on_update_kv_lens.
        self._msa_live_batch = 0
        self._msa_live_total_q = 0
        self._msa_page_size = 0
        self._msa_corrected_kv_lens_cpu: Optional[torch.Tensor] = None
        self._create_msa_buffers()

    @property
    def msa_qo_lens_cpu(self) -> Optional[torch.Tensor]:
        """Per-request query length (host int32), from the base seq_lens."""
        seq_lens = self.seq_lens
        if seq_lens is None:
            return None
        out = seq_lens[: self.num_seqs]
        return out if out.dtype == torch.int32 else out.to(torch.int32)

    @property
    def msa_kv_lens_cpu(self) -> Optional[torch.Tensor]:
        """Per-request KV length, cached plus new tokens (host int32).

        The base ``kv_lens`` includes ``num_extra_kv_tokens`` (speculative
        draft-loop slots consumed by the C++ kernels); the MSA plans, ladder
        slots and page counts need the true attended length, so it is
        excluded here.
        """
        if self._msa_corrected_kv_lens_cpu is not None:
            return self._msa_corrected_kv_lens_cpu
        kv_lens = getattr(self, "kv_lens", None)
        if self.seq_lens is None or kv_lens is None:
            return None
        out = kv_lens[: self.num_seqs]
        params = self.kv_cache_params
        extra = params.num_extra_kv_tokens if params is not None else 0
        if extra:
            out = out - extra
        return out if out.dtype == torch.int32 else out.to(torch.int32)

    @property
    def msa_qo_offset_cpu(self) -> Optional[torch.Tensor]:
        """Per-request causal offset (kv_len - qo_len), the cached prefix length."""
        qo = self.msa_qo_lens_cpu
        kv = self.msa_kv_lens_cpu
        if qo is None or kv is None:
            return None
        return kv - qo

    @property
    def msa_decode_proxy_plan(self) -> Optional[tuple]:
        """Proxy (max-score) plan tuple, or None outside decode."""
        plan = self._msa_proxy_plan
        return plan.plan if plan is not None else None

    @property
    def msa_decode_gqa_plan(self) -> Optional[tuple]:
        """Sparse GQA plan tuple, or None outside decode."""
        plan = self._msa_gqa_plan
        return plan.plan if plan is not None else None

    @property
    def msa_decode_dense_plan(self) -> Optional[tuple]:
        """Dense GQA plan tuple, shared by dense layers 0 to 2."""
        plan = self._msa_dense_plan
        return plan.plan if plan is not None else None

    @property
    def msa_eager_proxy_plan(self) -> Optional[tuple]:
        """Prebuilt indexer proxy plan for the eager (prefill/mixed) path."""
        return self._msa_eager_proxy_plan

    @property
    def msa_eager_gqa_plan(self) -> Optional[tuple]:
        """Prebuilt sparse GQA plan for the eager (prefill/mixed) path."""
        return self._msa_eager_gqa_plan

    @property
    def msa_eager_dense_plan(self) -> Optional[tuple]:
        """Prebuilt dense GQA plan for the eager (prefill/mixed) path."""
        return self._msa_eager_dense_plan

    @property
    def msa_eager_n_valid_blocks(self) -> Optional[torch.Tensor]:
        """Device int32 valid-block count for the eager path, or None if no eager
        step was prepared (a decode step or a structural test)."""
        return self._msa_eager_n_valid_blocks

    @property
    def msa_eager_all_blocks_empty(self) -> bool:
        """Whether the eager step has no valid KV blocks for any query token."""
        return self._msa_eager_all_blocks_empty

    def _msa_main_kv_is_fp8(self) -> bool:
        """Whether the main paged K/V cache is stored as FP8 E4M3.

        The eager GQA and dense plans must pass use_fp8_kvcache so the inline
        sparse-prefill path selects the FP8 AOT kernels; it is a no-op for the
        decode planner. Mirrors the k_paged.dtype check in run_msa_paged_gqa.
        """
        kv_cache_manager = self.kv_cache_manager
        if kv_cache_manager is None:
            return False
        try:
            buffers = kv_cache_manager.get_buffers(0, kv_layout="HND")
        except TypeError:
            buffers = kv_cache_manager.get_buffers(0)
        except Exception:
            return False

        try:
            return buffers[:, 0].dtype == torch.float8_e4m3fn
        except Exception:
            return False

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
        # Staging for on_update_kv_lens: re-derives slots/bounds on device
        # from the corrected kv_lens_cuda, sync-free.
        tokens_per_block = int(kv_cache_manager.tokens_per_block)
        self.msa_req_to_token = self.get_empty(
            buffers,
            (max_num_sequences, max_blocks_per_seq * tokens_per_block),
            cache_name="msa_req_to_token",
            dtype=torch.int32,
            capture_graph=capture_graph,
        )
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
        self.msa_qo_lens_dev = self.get_empty(
            buffers,
            (max_num_sequences,),
            cache_name="msa_qo_lens_dev",
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
            self._alloc_msa_proxy_scratch(
                num_index_heads=params.num_index_heads,
                max_tokens=self._msa_max_decode_tokens(),
                max_k_tiles=max_k_tiles,
                capture_graph=capture_graph,
            )
        self._msa_buffers_ready = True

    def _msa_max_decode_tokens(self) -> int:
        """Worst-case decode-step query tokens (spec verify emits 1 + draft_len
        per row), bounded by max_num_tokens. getattr fallbacks cover metadata
        built via ``__new__`` in structural tests.
        """
        max_seqs = int(getattr(self, "max_num_sequences", 0) or 0)
        max_toks = int(getattr(self, "max_num_tokens", 0) or 0)
        if max_toks <= 0:
            return max_seqs
        # 16384 keeps total_q * num_qo_heads under the fmha_sm100 planner cap
        # (65536, fmha_sm100/api.py) at 4 sharded index heads.
        return max(max_seqs, min(max_toks, 16384))

    def _alloc_msa_proxy_scratch(
        self,
        *,
        num_index_heads: int,
        max_tokens: int,
        max_k_tiles: int,
        capture_graph: bool,
    ) -> None:
        """Allocate the flat proxy max-score store and the valid-block scratch.

        The store is sized for the worst-case max_k_tiles and the worst-case
        per-step query-token count (which exceeds the batch size under
        speculative multi-token verify), so one allocation serves every
        decode step. msa_proxy_max_score_view slices the per-step shape out
        of it.
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
        self._alloc_msa_proxy_scratch(
            num_index_heads=num_index_heads,
            max_tokens=max_tokens,
            max_k_tiles=max_k_tiles,
            capture_graph=capture_graph,
        )

    def _ensure_eager_n_valid_buffer(self, total_q: int, device: torch.device) -> torch.Tensor:
        """Return a persistent device int32 buffer for the eager valid-block count.

        The eager path is never CUDA-graph captured, so a plain device tensor,
        grown on demand and reused across steps, is sufficient. It is sized to
        the worst-case per-step query-token count.
        """
        buf = self._msa_eager_n_valid_buf
        if buf is None or buf.numel() < total_q or buf.device != device:
            cap = max(int(total_q), int(getattr(self, "max_num_tokens", 0) or 0), 1)
            buf = torch.empty(cap, dtype=torch.int32, device=device)
            self._msa_eager_n_valid_buf = buf
        return buf

    def prepare(self) -> None:
        super().prepare()
        self._msa_corrected_kv_lens_cpu = None
        self._build_msa_fields()
        self._build_step_plans()

    def on_update_kv_lens(self) -> None:
        """Re-derive length-dependent MSA state from the corrected kv_lens_cuda.

        The overlap scheduler corrects kv_lens_cuda on device after prepare()
        staged optimistic (full-acceptance) lens. Decode steps are patched
        with pure device ops (capture-safe); the correction only shrinks
        lengths, so the host-baked plan worklists and the page-table layout
        stay valid. Eager mixed batches install corrected host lens and
        rebuild (small D2H sync). Idempotent.
        """
        super().on_update_kv_lens()
        if not self._msa_fields_ready:
            return
        batch = self._msa_live_batch
        total_q = self._msa_live_total_q
        if batch <= 0 or total_q <= 0:
            return
        if self.msa_decode_proxy_plan is None:
            # Prefill/mixed step: the eager plans and the page-table layout
            # were built from the optimistic host mirrors, so install the
            # corrected lens and rebuild both. Mixed steps are never captured.
            if torch.cuda.is_current_stream_capturing():
                return
            self._msa_corrected_kv_lens_cpu = self.kv_lens_cuda[:batch].to("cpu", torch.int32)
            self._build_msa_fields()
            self._build_step_plans()
            return

        kv_true = self.kv_lens_cuda[:batch]
        qbr = self.msa_q_batch_row[:total_q].to(torch.long)
        qo_dev = self.msa_qo_lens_dev[:batch]
        kv_true_tok = kv_true[qbr]
        pos = kv_true_tok - qo_dev[qbr] + self.msa_q_intra[:total_q]

        # KV/idx-K write slots: slot[j] = req_to_token[request[j], pos[j]].
        width = int(self.msa_req_to_token.shape[1])
        idx = pos.to(torch.long).clamp(min=0, max=width - 1)
        slots = self.msa_req_to_token.reshape(-1).index_select(0, qbr * width + idx)
        self.msa_out_cache_loc[:total_q].copy_(slots)

        # Per-token valid-block counts for top-k selection. clamp_min(1)
        # keeps degenerate (graph-padding) rows from -inf-masking every
        # block, which would NaN the fully-masked GQA row.
        page = self._msa_page_size
        n_valid = torch.div((pos + 1).clamp_min(1) + (page - 1), page, rounding_mode="floor")
        self.msa_n_valid_blocks[:total_q].copy_(n_valid.to(torch.int32))

        # Plan mirrors: proxy and dense keep one row per request; the sparse
        # GQA plan is row-expanded per token with the ladder in the per-row
        # offset. qo_offset must stay non-negative: negative values hit the
        # kernel's packed-length sentinel fallback.
        off_req = (kv_true - qo_dev).clamp_min(0)
        for owner, expanded in (
            (self._msa_proxy_plan, False),
            (self._msa_gqa_plan, True),
            (self._msa_dense_plan, False),
        ):
            decode = owner.plan[3]
            seg_lens = decode.get("kv_segment_lens")
            qo_off = decode.get("qo_offset")
            if expanded:
                if seg_lens is not None:
                    seg_lens[:total_q].copy_(kv_true_tok.to(seg_lens.dtype))
                if qo_off is not None:
                    qo_off[:total_q].copy_(pos.clamp_min(0).to(qo_off.dtype))
            else:
                if seg_lens is not None:
                    seg_lens[:batch].copy_(kv_true.to(seg_lens.dtype))
                if qo_off is not None:
                    qo_off[:batch].copy_(off_req.to(qo_off.dtype))

    def _build_step_plans(self) -> None:
        """Build the three layer-invariant fmha_sm100 plans once per step.

        Runs in prepare(), outside CUDA graph capture. The proxy, GQA, and
        dense plans depend only on the per-step sparse geometry (qo/kv lengths,
        head counts, topk, page size), never on the layer, so they are built
        once here and reused by every layer:

        * Pure-decode batches mirror the plans into the CUDA-graph-stable
          _MsaGraphSafePlan buffers (surfaced by msa_decode_*_plan), because
          decode is captured and the plan worklists must keep a fixed address
          across replays.
        * Prefill, chunked-prefill, and mixed batches run eagerly (never
          captured), so the plans are stored as plain tuples (msa_eager_*_plan)
          that every sparse and dense layer reuses.
        """
        # Drop any plan tuples from the previous step; the msa_decode_*_plan and
        # msa_eager_*_plan properties then report None until rebuilt below.
        for plan in (self._msa_proxy_plan, self._msa_gqa_plan, self._msa_dense_plan):
            if plan is not None:
                plan.reset()
        self._msa_eager_proxy_plan = None
        self._msa_eager_gqa_plan = None
        self._msa_eager_dense_plan = None
        self._msa_eager_n_valid_blocks = None
        self._msa_eager_all_blocks_empty = False
        if not self._msa_fields_ready:
            return
        # Geometry is captured in __post_init__; skip when it is unavailable.
        params = self._msa_params
        if params is None:
            return
        num_index_heads = params.num_index_heads
        num_q_heads, num_kv_heads = params.sharded_head_counts(self.mapping)
        topk = params.topk

        fmha_sm100 = require_msa_module()
        qo_lens_cpu = self.msa_qo_lens_cpu
        kv_lens_cpu = self.msa_kv_lens_cpu
        qo_offset_cpu = self.msa_qo_offset_cpu
        if qo_lens_cpu is None or kv_lens_cpu is None or qo_offset_cpu is None:
            return
        device = _cache_device(self)
        page_size = int(self.kv_cache_manager.tokens_per_block)
        capture_graph = self.is_cuda_graph
        max_batch = int(self.max_num_sequences)
        # A decode batch is pure generation (no context requests). Only that
        # path is CUDA-graph captured and uses the graph-stable plan buffers.
        is_decode = int(self.num_contexts or 0) == 0
        # The main-attention GQA and dense plans need use_fp8_kvcache so the
        # eager (inline sparse-prefill) kernel selection matches an FP8 paged
        # cache; it is a no-op for the decode planner. The proxy runs over the
        # bf16 index-K cache, so it never needs the flag.
        use_fp8 = self._msa_main_kv_is_fp8()

        # Proxy plan: MQA (num_kv_heads=1) max-score pass over the index
        # branch; output_maxscore feeds the indexer's top-k block selection.
        proxy_plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            num_index_heads,
            num_kv_heads=1,
            qo_offset=qo_offset_cpu,
            page_size=page_size,
            output_maxscore=True,
            num_kv_splits=1,
            causal=True,
        )
        # Sparse-layer plan: kv_block_num=topk limits attention to top-k blocks.
        gqa_plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            num_q_heads,
            num_kv_heads=num_kv_heads,
            qo_offset=qo_offset_cpu,
            page_size=page_size,
            kv_block_num=topk,
            num_kv_splits=1,
            causal=True,
            use_fp8_kvcache=use_fp8,
        )
        # Dense-layer plan: no kv_block_num, so it attends the full page table.
        dense_plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            num_q_heads,
            num_kv_heads=num_kv_heads,
            qo_offset=qo_offset_cpu,
            page_size=page_size,
            num_kv_splits=1,
            causal=True,
            use_fp8_kvcache=use_fp8,
        )

        if not is_decode:
            # Prefill and mixed batches run eagerly, so keep the plain plan
            # tuples and leave the graph-safe owners reset.
            self._msa_eager_proxy_plan = proxy_plan
            self._msa_eager_gqa_plan = gqa_plan
            self._msa_eager_dense_plan = dense_plan
            # Stage the valid-block count to the device once for the whole step
            # (see _msa_eager_n_valid_blocks). The empty-selection check runs
            # here on the host, so run_indexer can short-circuit cheaply.
            n_valid_host = per_token_valid_blocks(
                qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, causal=True, block_size=page_size
            )
            total_q = int(n_valid_host.shape[0])
            self._msa_eager_all_blocks_empty = total_q == 0 or int(n_valid_host.max().item()) <= 0
            if total_q > 0:
                dev_buf = self._ensure_eager_n_valid_buffer(total_q, device)
                dev_buf[:total_q].copy_(n_valid_host.to(torch.int32), non_blocking=True)
                self._msa_eager_n_valid_blocks = dev_buf[:total_q]
            return

        required_max_k_tiles = int(proxy_plan[3]["max_k_tiles"])
        self._ensure_msa_decode_scratch_buffers(
            num_index_heads=num_index_heads,
            max_batch=max_batch,
            capture_graph=capture_graph,
            required_max_k_tiles=required_max_k_tiles,
        )

        # Allocate the graph-safe plan owners once per metadata; later steps
        # only refresh their contents below. The plan worklists are sized per
        # expanded row (the planner splits qo_len > 1 requests into per-token
        # rows under speculative multi-token verify), so use the worst-case
        # decode-step token count rather than the batch size.
        if self._msa_proxy_plan is None:
            max_plan_rows = max(max_batch, self._msa_max_decode_tokens())
            num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
            self._msa_proxy_plan = _MsaGraphSafePlan(
                self,
                "msa_proxy_plan",
                max_batch=max_plan_rows,
                num_ctas=num_ctas,
                capture_graph=capture_graph,
            )
            self._msa_gqa_plan = _MsaGraphSafePlan(
                self,
                "msa_gqa_plan",
                max_batch=max_plan_rows,
                num_ctas=num_ctas,
                capture_graph=capture_graph,
            )
            self._msa_dense_plan = _MsaGraphSafePlan(
                self,
                "msa_dense_plan",
                max_batch=max_plan_rows,
                num_ctas=num_ctas,
                capture_graph=capture_graph,
            )

        # refresh() stores each plan tuple on its owner, surfaced by the
        # msa_decode_*_plan properties.
        self._msa_proxy_plan.refresh(proxy_plan)
        self._msa_gqa_plan.refresh(gqa_plan)
        self._msa_dense_plan.refresh(dense_plan)

        n_valid = per_token_valid_blocks(
            qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, causal=True, block_size=page_size
        )
        # One entry per query token (qo_len > 1 under spec verify).
        total_q = int(n_valid.shape[0])
        self.msa_n_valid_blocks[:total_q].copy_(n_valid.to(torch.int32), non_blocking=True)

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
        req_to_token, slot_ids, out_cache_loc = build_paged_kv_slot_mapping(
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            qo_lens_cpu=qo_lens_cpu,
            qo_offset_cpu=qo_offset_cpu,
            device=cache_device,
        )
        kv_indices = build_kv_page_indices(req_to_token, slot_ids, kv_lens_cpu, page_size)

        total_new_tokens = int(out_cache_loc.shape[0])
        total_pages = int(kv_indices.shape[0])
        if total_new_tokens > self.msa_out_cache_loc.shape[0]:
            raise ValueError(
                f"MSA out_cache_loc buffer ({self.msa_out_cache_loc.shape[0]}) is "
                f"smaller than the step's new-token count ({total_new_tokens})."
            )
        if total_pages > self.msa_kv_indices.shape[0]:
            raise ValueError(
                f"MSA kv_indices buffer ({self.msa_kv_indices.shape[0]}) is "
                f"smaller than the step's page count ({total_pages})."
            )

        self.msa_out_cache_loc[:total_new_tokens].copy_(out_cache_loc, non_blocking=True)
        self.msa_kv_indices[:total_pages].copy_(kv_indices, non_blocking=True)

        # Staging for on_update_kv_lens.
        step_width = int(req_to_token.shape[1])
        self.msa_req_to_token[:batch_size, :step_width].copy_(req_to_token, non_blocking=True)
        qo_long = qo_lens_cpu.to(torch.long)
        batch_row_cpu = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32), qo_long
        )
        starts = torch.cumsum(qo_long, 0) - qo_long
        intra_cpu = (
            torch.arange(total_new_tokens, dtype=torch.int64)
            - torch.repeat_interleave(starts, qo_long)
        ).to(torch.int32)
        self.msa_q_batch_row[:total_new_tokens].copy_(batch_row_cpu)
        self.msa_q_intra[:total_new_tokens].copy_(intra_cpu)
        self.msa_qo_lens_dev[:batch_size].copy_(qo_lens_cpu)
        self._msa_live_batch = batch_size
        self._msa_live_total_q = total_new_tokens
        self._msa_page_size = page_size
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

    @classmethod
    def support_fused_rope(cls) -> bool:
        # The MiniMax-M3 model layer applies partial RoPE to the main and
        # index branches explicitly.
        return False

    def run_indexer(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        metadata,
        *,
        idx_sm_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Write the index-K cache and return the selected block indices.

        The model layer runs this before forward and threads the result through
        forward_args.topk_indices. Returns [total_q, num_kv_heads, topk].
        Decode uses the prebuilt graph-safe proxy plan; prefill and mixed
        batches use the prebuilt eager proxy plan.
        """
        config = self.m3_config
        idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5
        num_tokens = int(idx_q.shape[0])
        # idx_q and idx_k may be strided column-views of a fused buffer, so
        # reshape to keep them zero-copy. The proxy fmha_sm100 and the index-K
        # scatter below both honor the source strides.
        idx_q_view = idx_q.reshape(num_tokens, config.num_index_heads, config.sparse_index_dim)
        idx_k_view = idx_k.reshape(num_tokens, 1, config.sparse_index_dim)

        metadata.msa_write_idx_k(self.layer_idx, idx_k_view)
        idx_k_cache = metadata.msa_idx_k_cache(self.layer_idx)

        # One selection path. Decode passes the graph-safe proxy plan plus the
        # proxy scratch shaped to the live query count. Prefill and mixed batches
        # pass the eager proxy plan and the device-staged valid-block count. When
        # neither is present (a standalone test that skips prepare) select_blocks
        # plans inline and computes the valid-block count itself.
        proxy_plan = metadata.msa_decode_proxy_plan
        if proxy_plan is not None:
            # proxy_plan is (has_mixed, split, batch, decode_dict, prefill);
            # decode_dict carries max_k_tiles for the contiguous score view.
            plan_max_k_tiles = int(proxy_plan[3]["max_k_tiles"])
            max_score = metadata.msa_proxy_max_score_view(
                config.num_index_heads, plan_max_k_tiles, num_tokens
            )
            n_valid_blocks = metadata.msa_n_valid_blocks[:num_tokens]
        else:
            proxy_plan = metadata.msa_eager_proxy_plan
            max_score = None
            # The empty case was resolved on the host, so short-circuit here.
            if metadata.msa_eager_all_blocks_empty:
                return torch.full(
                    (num_tokens, config.num_kv_heads, MSA_REQUIRED_TOPK),
                    -1,
                    dtype=torch.int32,
                    device=idx_q.device,
                )
            n_valid_blocks = metadata.msa_eager_n_valid_blocks
            if n_valid_blocks is not None:
                n_valid_blocks = n_valid_blocks[:num_tokens]
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
        )

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata,
        forward_args: "AttentionForwardArgs",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # The model layer runs run_indexer and passes the selected block
        # indices through forward_args.topk_indices. Publish them as the
        # sparse attention indices MsaSparseGqaFmha reads.
        return forward_args.topk_indices, None

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
