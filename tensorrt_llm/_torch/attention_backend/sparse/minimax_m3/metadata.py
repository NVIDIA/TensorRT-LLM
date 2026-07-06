# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 sparse attention configuration + per-forward metadata.

Contains:

  * :class:`MiniMaxM3SparseConfig`            -- post-TP-shard kernel
                                                 parameter bundle.
  * :class:`MiniMaxM3SparseAttentionMetadata` -- per-forward metadata
                                                 dataclass with a
                                                 CUDA-graph-safe
                                                 :meth:`prepare`.
  * Helpers to migrate metadata across devices, build it from a real
    :class:`KVCacheManagerV2`, and pre-allocate CUDA-graph-stable
    buffers.
  * :func:`get_minimax_m3_attention_metadata_cls` -- lazy factory for
    the :class:`AttentionMetadata` subclass the pyexecutor wires into
    the M3 sparse layer's forward path.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..params import SparseParams

# ---------------------------------------------------------------------------
# MSA per-rank geometry and CUDA-graph-stable paged-KV table staging
# ---------------------------------------------------------------------------

# The per-rank sparse geometry needed to stage the MSA plans is the
# layer-invariant `MiniMaxM3SparseConfig`. There is no separate geometry
# struct: the config is the single source of truth, and the decode driver
# derives its own alloc-time key (`M3DecodeGeometry`) from it.

_GLOBAL_MSA_GEOMETRY: Optional["MiniMaxM3SparseConfig"] = None


def set_global_msa_geometry(geometry: "MiniMaxM3SparseConfig") -> None:
    """Register the per-rank M3 sparse config process-wide.

    Called from the attention layer's constructor, before any forward and
    so before any CUDA graph capture. Every metadata instance's prepare()
    reads this to pre-build the MSA plans; registering at construction
    ensures graph-capture metadata has a config and does not fall back to
    in-forward planning that would freeze host values into each replay.

    All sparse layers on a rank share one config; the first writer wins.
    """
    global _GLOBAL_MSA_GEOMETRY
    if _GLOBAL_MSA_GEOMETRY is None:
        _GLOBAL_MSA_GEOMETRY = geometry
        logger.info("MiniMax-M3: using MSA (fmha_sm100) sparse attention kernels.")


def get_global_msa_geometry() -> Optional["MiniMaxM3SparseConfig"]:
    return _GLOBAL_MSA_GEOMETRY


def build_stable_kv_indices(
    *,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    page_size: int,
    dst: torch.Tensor,
    kv_page_indptr_dst: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the flat paged-KV page table into `dst`.

    Computes the page table with vectorized ops into a preallocated
    buffer whose `data_ptr()` is stable under CUDA graph replay.

    Parameters
    ----------
    req_to_token : `[max_reqs, max_kv_len]` int32 on cache device.
    slot_ids : `[batch]` int32 on cache device; `req_to_token` row
        indices for the current batch.
    seq_lens : `[batch]` int32 on cache device; per-request effective KV
        length.
    seq_lens_cpu : `[batch]` int32 or int64 on CPU; same values, used for
        the CPU-side sizing math so we do not force a D2H sync.
    page_size : int, must equal the `req_to_token` block width (the M3 KV
        cache manager enforces this).
    dst : preallocated int32 buffer sized `>= max_batch * max_pages`. The
        output `kv_indices` view is `dst[:total_pages]`.
    kv_page_indptr_dst : preallocated int32 buffer sized `>= batch + 1`.
        The output `kv_page_indptr` view is `kv_page_indptr_dst[:batch+1]`.

    Returns
    -------
    (kv_indices, kv_page_indptr) as views into `dst` / `kv_page_indptr_dst`.
    Their `data_ptr()` is stable across calls because they alias the
    destination buffers.
    """
    device = req_to_token.device
    batch = int(seq_lens_cpu.shape[0])
    max_kv_len = int(req_to_token.shape[1])
    max_pages_per_seq = max_kv_len // page_size

    if batch == 0:
        return dst[:0], kv_page_indptr_dst[:1].zero_()

    # Number of pages per request: ceil(seq_len / page_size). Do it on
    # CPU so kv_page_indptr can be prepared as ints for the row/col
    # gather without triggering a D2H sync on seq_lens.
    seq_lens_cpu_long = seq_lens_cpu.to(torch.long).cpu()
    num_pages_cpu = (seq_lens_cpu_long + page_size - 1) // page_size
    num_pages_cpu = num_pages_cpu.clamp_min(0)
    total_pages = int(num_pages_cpu.sum().item())
    if total_pages > int(dst.shape[0]):
        raise RuntimeError(
            f"MSA kv_indices persistent buffer too small: capacity {int(dst.shape[0])} "
            f"< total pages {total_pages}. Increase max_kv_indices on allocate()."
        )

    # Build kv_page_indptr on CPU then copy the prefix into the
    # persistent buffer. Values are per-batch cumulative page counts,
    # starting at 0.
    kv_page_indptr_cpu = torch.empty(batch + 1, dtype=torch.int32)
    kv_page_indptr_cpu[0] = 0
    kv_page_indptr_cpu[1:].copy_(num_pages_cpu.to(torch.int32).cumsum(0))
    kv_page_indptr_dst[: batch + 1].copy_(
        kv_page_indptr_cpu.to(device=device, non_blocking=True), non_blocking=True
    )

    # Vectorized page-index gather:
    #   req_rows = req_to_token[slot_ids] gives [batch, max_kv_len] slot
    #   ids. For each request b, valid pages are indices 0..num_pages[b]-1;
    #   the p-th page's first-slot column is p * page_size, and its page
    #   id is req_rows[b, p*page_size] // page_size. We build a max-sized
    #   (batch, max_pages_per_seq) grid, gather with clamped column
    #   indices, then mask trailing invalid pages before packing into dst.
    slot_ids_long = slot_ids.to(torch.long)
    req_rows = req_to_token.index_select(0, slot_ids_long).to(torch.long)

    max_valid_pages = max(1, max_pages_per_seq)
    pages_grid = torch.arange(max_valid_pages, device=device, dtype=torch.long)
    # Column index of the first slot of page p: p * page_size, clamped to
    # max_kv_len - 1 for out-of-range pages so the gather does not fault.
    # Out-of-range page ids are trimmed by the batch mask below.
    col_idx = (pages_grid * page_size).clamp_max(max(0, max_kv_len - 1))
    # Broadcast to [batch, max_pages_per_seq].
    col_idx_b = col_idx.unsqueeze(0).expand(batch, -1)
    # The gathered values are global page ids into the paged cache.
    # req_rows holds valid slot ids by construction, so no value clamp is
    # applied: clamping to the per-request page count would collapse the
    # page table for any request whose global page ids exceed that count
    # and corrupt every request after the first.
    gathered = (req_rows.gather(1, col_idx_b) // page_size).to(torch.int32)

    # Build a valid-page mask per request:
    #   mask[b, p] = p < num_pages[b]
    num_pages_dev = num_pages_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    mask = pages_grid.unsqueeze(0) < num_pages_dev.unsqueeze(1)  # [batch, max_pages_per_seq]

    # Compact into the flat dst prefix using boolean indexing.
    # torch.masked_select preserves row-major (batch, page) order, which
    # matches the concat([pages_of(seq_0), pages_of(seq_1), ...]) layout
    # kv_page_indptr encodes.
    packed = torch.masked_select(gathered, mask)
    dst[:total_pages].copy_(packed, non_blocking=True)

    return dst[:total_pages], kv_page_indptr_dst[: batch + 1]


def whole_batch_qo_lens(
    m3_meta: "MiniMaxM3SparseAttentionMetadata",
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """Per-request `(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu)` for the whole batch.

    Single source of truth for the CPU length/offset derivation shared by
    the plan staging (`_build_msa_plans_for_metadata`) and the eager
    whole-batch fallback (`msa_backend._whole_batch_lens`): prefill reads
    the extend lengths and prefix offsets; decode is one query token per
    request at position `kv_len - 1`. `kv_lens_cpu` is the per-request
    effective KV length (int32, CPU).

    Returns ``None`` when prefill metadata is incomplete (missing
    `extend_seq_lens_cpu` / `prefix_lens`) so callers can choose to skip
    staging or raise.
    """
    seq_lens_cpu = m3_meta.seq_lens_cpu.to(torch.int32)
    if m3_meta.is_prefill:
        if m3_meta.extend_seq_lens_cpu is None or m3_meta.prefix_lens is None:
            return None
        qo_lens_cpu = torch.tensor(m3_meta.extend_seq_lens_cpu, dtype=torch.int32)
        qo_offset_cpu = m3_meta.prefix_lens.detach().to(device="cpu", dtype=torch.int32)
    else:
        batch = int(seq_lens_cpu.shape[0])
        qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
        qo_offset_cpu = (seq_lens_cpu - 1).to(torch.int32)
    return qo_lens_cpu, seq_lens_cpu, qo_offset_cpu


@dataclass(frozen=True)
class MiniMaxM3SparseParams(SparseParams):
    """Lowered runtime parameters for the MiniMax-M3 sparse backend."""

    algorithm: Literal["minimax_m3"] = field(init=False, default="minimax_m3")
    num_index_heads: int = 4
    # Sparse index heads paired with each KV head
    # (index_group = num_index_heads_global // num_kv_heads_global). Under
    # TP a rank holding KV heads `[s, e)` scores with index heads
    # `[s*index_group, e*index_group)`. `None` means "derive from the
    # per-rank KV head count" (single-GPU / tests, where per-rank equals
    # global).
    index_group: Optional[int] = None
    sparse_index_dim: int = 128
    block_size: int = 128
    topk: int = 16
    init_blocks: int = 0
    local_blocks: int = 1
    score_type: str = "max"
    disable_index_value: bool = True
    # When True, the layer dispatches the sparse forward through the
    # MSA-backed FMHA runtime (`fmha_sm100` + `sparse_topk_select`)
    # instead of the in-tree Triton + SDPA reference path. The MSA stack
    # is only available on SM100 and requires the external
    # `fmha_sm100` package; the layer raises a descriptive error if
    # it is requested without those preconditions met.
    use_msa: bool = False

    @property
    def indices_block_size(self) -> int:
        """Block granularity of the sparse attention indices.

        `TrtllmAttention.forward` reads this off the sparse params to
        stamp `forward_args.sparse_prediction.sparse_attn_indices_block_size`.
        For MiniMax-M3 the per-query selected indices are KV *block* indices,
        so the granularity is the sparse block size.
        """
        return self.block_size


@dataclass(frozen=True)
class MiniMaxM3SparseConfig:
    """Per-rank kernel parameter bundle for MiniMax-M3 sparse attention.

    This is **not** a user-facing config (use
    :class:`tensorrt_llm.llmapi.llm_args.MiniMaxM3SparseAttentionConfig`
    for that). It is the layer-invariant, post-TP-shard parameter bundle
    that backend kernels and reference helpers consume. The user knobs
    come from :class:`MiniMaxM3SparseParams`; ``num_q_heads`` /
    ``num_kv_heads`` / ``head_dim`` come from the per-rank model
    geometry and must be supplied by the caller (typically via
    :meth:`from_sparse_params`).
    """

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_index_heads: int
    sparse_index_dim: int
    block_size: int
    topk: int
    init_blocks: int = 0
    local_blocks: int = 1
    score_type: str = "max"

    def __post_init__(self) -> None:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.num_index_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_index_heads ({self.num_index_heads}) must be divisible "
                f"by num_kv_heads ({self.num_kv_heads})"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.topk <= 0:
            raise ValueError(f"topk must be > 0, got {self.topk}")
        if self.init_blocks < 0:
            raise ValueError(f"init_blocks must be >= 0, got {self.init_blocks}")
        if self.local_blocks < 0:
            raise ValueError(f"local_blocks must be >= 0, got {self.local_blocks}")
        if self.score_type != "max":
            # SGLang exposes only "max" today and that is what the MiniMax-M3
            # checkpoint config specifies. Reject anything else explicitly so
            # a config drift surfaces immediately.
            raise ValueError(
                f"score_type={self.score_type!r} is not supported "
                "(only 'max' matches the SGLang reference)"
            )

    @classmethod
    def from_sparse_params(
        cls,
        sparse_params: "MiniMaxM3SparseParams",
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> "MiniMaxM3SparseConfig":
        """Build a kernel param bundle from lowered ``MiniMaxM3SparseParams``
        and the per-rank model geometry.

        The per-rank index-head count is `index_group` heads per local KV
        head, so index head `i` stays paired with KV head `i` under TP
        (the model layer slices `idx_q` with the matching offsets; see
        `modeling_minimaxm3.py`). Selecting blocks from all index heads'
        scores would give every KV head the union/max over all index heads
        instead of its own head's top-k.
        """
        index_group = sparse_params.index_group
        if index_group is None:
            # Single-GPU / tests: per-rank KV heads equal the global count.
            if int(sparse_params.num_index_heads) % int(num_kv_heads) != 0:
                raise ValueError(
                    f"num_index_heads ({sparse_params.num_index_heads}) must be "
                    f"divisible by num_kv_heads ({num_kv_heads})"
                )
            index_group = int(sparse_params.num_index_heads) // int(num_kv_heads)
        index_group = int(index_group)
        # Per-rank KV heads never exceed the global count, so this is just
        # index_group heads per local KV head.
        num_index_heads_local = index_group * int(num_kv_heads)
        return cls(
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            num_index_heads=num_index_heads_local,
            sparse_index_dim=int(sparse_params.sparse_index_dim),
            block_size=int(sparse_params.block_size),
            topk=int(sparse_params.topk),
            init_blocks=int(sparse_params.init_blocks),
            local_blocks=int(sparse_params.local_blocks),
            score_type=str(sparse_params.score_type),
        )


@dataclass
class MiniMaxM3SparseAttentionMetadata:
    """Per-forward metadata for MiniMax-M3 sparse attention.

    Built from the paged-cache view `KVCacheManagerV2` exposes
    (`req_to_token[req_idx, pos] -> slot_id` plus `slot_ids[batch_idx]`).
    `prepare()` fills the CUDA-graph-safe scalars `max_seqlen_q` /
    `max_seqlen_k` and, for prefill, the `q_batch_row` / `q_positions`
    tensors.

    `is_prefill=True` routes through the extend kernel, which handles both
    pure prefill and mixed prefill+decode batches (decode rows appear as
    1-slot extends). `is_prefill=False` is a perf-only specialization for
    pure-decode batches, mathematically equivalent to the extend path with
    `extend_seq_len=[1]*batch`.
    """

    is_prefill: bool
    req_to_token: torch.Tensor
    slot_ids: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    prefix_lens: Optional[torch.Tensor] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    q_batch_row: Optional[torch.Tensor] = None
    q_positions: Optional[torch.Tensor] = None
    max_seqlen_q: int = field(default=1)
    max_seqlen_k: int = field(default=1)
    # MSA per-step plan values, written by `_build_msa_plans_for_metadata`
    # when the KV cache manager has `use_msa=True` (None on the Triton
    # path). The MSA decode kernels read them directly off this metadata.
    msa_kv_indices: Optional[torch.Tensor] = None
    msa_kv_page_indptr: Optional[torch.Tensor] = None
    msa_qo_lens_cpu: Optional[torch.Tensor] = None
    msa_kv_lens_cpu: Optional[torch.Tensor] = None
    msa_qo_offset_cpu: Optional[torch.Tensor] = None
    msa_max_batch: int = 0
    msa_max_kv_len: int = 0

    def prepare(self) -> None:
        """Compute CUDA-graph-safe scalar max lengths from CPU tensors.

        ``max_seqlen_q`` and ``max_seqlen_k`` are stored as plain Python
        ints so they can be used as static shapes / loop bounds inside
        forward kernels without triggering a GPU-CPU sync at graph
        capture or replay. For prefill metadata, this method also
        precomputes the ``q_batch_row`` and ``q_positions`` tensors so
        the forward path does not need to read CPU tensors during
        capture.

        When :attr:`q_batch_row` and :attr:`q_positions` are already
        populated (the caller has written into pre-allocated static
        buffers for CUDA-graph-stable addresses) the prefill branch
        leaves them alone; only the scalar ``max_seqlen_*`` derivation
        runs. Otherwise prepare() allocates fresh tensors as before.
        """
        batch_size = int(self.slot_ids.shape[0])
        if self.is_prefill:
            if self.extend_seq_lens_cpu is None:
                raise ValueError(
                    "prefill metadata requires extend_seq_lens_cpu for CUDA-graph-safe max_seqlen_q"
                )
            if self.cu_seqlens_q is None:
                raise ValueError("prefill metadata requires cu_seqlens_q")
            if self.prefix_lens is None:
                raise ValueError("prefill metadata requires prefix_lens")
            if not self.extend_seq_lens_cpu:
                self.max_seqlen_q = 1
            else:
                self.max_seqlen_q = int(max(self.extend_seq_lens_cpu))

            if self.q_batch_row is None or self.q_positions is None:
                # Precompute per-Q-token batch_row + K-side position once,
                # CPU-side, so the forward path is sync-free.
                total_q = int(self.cu_seqlens_q[-1].item())
                cu = self.cu_seqlens_q.to(torch.long).tolist()
                pref = self.prefix_lens.to(torch.long).tolist()
                device = self.slot_ids.device
                q_batch_row = torch.empty(total_q, dtype=torch.int32, device=device)
                q_positions = torch.empty(total_q, dtype=torch.int32, device=device)
                for b in range(batch_size):
                    start, end = cu[b], cu[b + 1]
                    q_batch_row[start:end] = b
                    offsets = (
                        torch.arange(start, end, device=device, dtype=torch.int32)
                        - start
                        + int(pref[b])
                    )
                    q_positions[start:end] = offsets
                self.q_batch_row = q_batch_row
                self.q_positions = q_positions
            # else: caller (build_runtime_metadata_from_kv_manager with
            # static_buffers) has already populated q_batch_row /
            # q_positions in-place into stable buffers. Do not reallocate.
        else:
            self.max_seqlen_q = 1
            # Decode keeps q_batch_row / q_positions as None unless the
            # caller has wired pre-allocated buffers (e.g. for CUDA graph
            # capture safety the decode path uses
            # ``torch.arange(batch_size)`` inline so no static buffer is
            # required). Preserve the existing post-prepare contract.
            if self.q_batch_row is None:
                self.q_positions = None
        if batch_size == 0:
            self.max_seqlen_k = 1
        else:
            self.max_seqlen_k = int(self.seq_lens_cpu[:batch_size].max().item())


def ensure_metadata_on_device(
    metadata: "MiniMaxM3SparseAttentionMetadata",
    device: torch.device,
) -> "MiniMaxM3SparseAttentionMetadata":
    """Return ``metadata`` with every GPU-consumed tensor on ``device``.

    Constructs a new :class:`MiniMaxM3SparseAttentionMetadata` whose
    tensor fields are migrated to ``device`` when they live elsewhere.
    The CPU-side mirror ``seq_lens_cpu`` is preserved because the
    algorithm reads it only when explicitly noted (e.g. for
    sync-free ``max_seqlen_*`` derivation on the host).

    This is the integration shim between the pyexecutor-driven
    host-side metadata (``seq_lens.device`` is typically CPU) and the
    M3 algorithm (which runs on the cache device). Calling it once at
    the top of ``forward_sparse`` avoids scattering ``.to(device)``
    calls through every algorithm helper.
    """

    def _move(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None or not isinstance(t, torch.Tensor):
            return t
        if t.device == device:
            return t
        return t.to(device)

    return dataclasses.replace(
        metadata,
        req_to_token=_move(metadata.req_to_token),
        slot_ids=_move(metadata.slot_ids),
        seq_lens=_move(metadata.seq_lens),
        prefix_lens=_move(metadata.prefix_lens),
        cu_seqlens_q=_move(metadata.cu_seqlens_q),
        q_batch_row=_move(metadata.q_batch_row),
        q_positions=_move(metadata.q_positions),
    )


def allocate_minimax_m3_static_buffers(
    *,
    max_num_sequences: int,
    max_num_tokens: int,
    max_kv_len: int,
    device: torch.device,
) -> dict:
    """Allocate persistent per-graph buffers for MiniMax-M3 metadata.

    Under CUDA-graph replay every tensor the captured kernels read must
    keep the same `data_ptr()`. Fresh per-call allocations break this: the
    captured kernel reads freed warmup memory, producing wrong tokens or an
    out-of-bounds `index_select` (`Indexing.cu:1515` `srcIndex <
    srcSelectDimSize`). `build_runtime_metadata_from_kv_manager` writes
    per-call values into these buffers in place via `.copy_()`.

    The buffers are sized for the largest batch the graph will see
    (`max_num_sequences` requests, `max_kv_len` cache slots, `max_num_tokens`
    Q tokens) and live on `device`. The returned dict also carries the
    allocation-time geometry so callers can detect a geometry change.
    """
    if max_num_sequences <= 0:
        raise ValueError("max_num_sequences must be positive")
    if max_num_tokens <= 0:
        raise ValueError("max_num_tokens must be positive")
    if max_kv_len <= 0:
        raise ValueError("max_kv_len must be positive")

    return {
        "max_num_sequences": int(max_num_sequences),
        "max_num_tokens": int(max_num_tokens),
        "max_kv_len": int(max_kv_len),
        "device": device,
        "req_to_token": torch.zeros(
            (max_num_sequences, max_kv_len),
            dtype=torch.int32,
            device=device,
        ),
        "slot_ids": torch.arange(
            max_num_sequences,
            dtype=torch.int32,
            device=device,
        ),
        "seq_lens_dev": torch.ones(
            (max_num_sequences,),
            dtype=torch.int32,
            device=device,
        ),
        "prefix_lens": torch.zeros(
            (max_num_sequences,),
            dtype=torch.int32,
            device=device,
        ),
        "cu_seqlens_q": torch.zeros(
            (max_num_sequences + 1,),
            dtype=torch.int32,
            device=device,
        ),
        "out_cache_loc": torch.zeros(
            (max_num_tokens,),
            dtype=torch.int32,
            device=device,
        ),
        "q_batch_row": torch.zeros(
            (max_num_tokens,),
            dtype=torch.int32,
            device=device,
        ),
        "q_positions": torch.zeros(
            (max_num_tokens,),
            dtype=torch.int32,
            device=device,
        ),
    }


def m3_cache_device(meta) -> torch.device:
    """Device hosting the paged KV buffers, else the current CUDA device.

    Shared by both M3 metadata classes and the plan builder so the
    cache-device probe lives in one place.
    """
    kv_cache_manager = meta.kv_cache_manager
    if kv_cache_manager is not None:
        try:
            return kv_cache_manager.get_buffers(0).device
        except Exception:
            pass
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def maybe_build_static_buffers_placeholder(
    meta,
    cache_device: torch.device,
) -> Optional[dict]:
    """Return the persistent M3 buffer dict when CUDA-graph stability is needed.

    Shared by both M3 attention-metadata classes (the Triton
    :class:`MiniMaxM3AttentionMetadata` and the MSA
    ``MiniMaxM3MSATrtllmAttentionMetadata``). Manages ``meta._m3_static_buffers``
    in place and returns:

      * ``None`` when static buffers are not needed (eager-only paths that
        rely on per-call allocations), OR
      * a placeholder dict (only capacity hints) on first use, which
        :func:`build_runtime_metadata_from_kv_manager` lazily fills with the
        real persistent tensors once the current step's geometry is known,
        OR
      * the already-allocated buffer dict on subsequent steps.

    Static buffers are used when either ``is_cuda_graph`` is set (captured
    graph needs stable ``data_ptr()`` across replays) or a previous
    prepare() already allocated them (so the algorithm keeps seeing the
    same addresses when the engine alternates eager warmup and graph
    replay).
    """
    existing = getattr(meta, "_m3_static_buffers", None)
    need_static = bool(meta.is_cuda_graph) or existing is not None
    if not need_static:
        return None
    if existing is not None and existing.get("device") == cache_device:
        return existing

    max_num_sequences_hint = int(meta.max_num_sequences or meta.max_num_requests)
    placeholder: dict = {
        "device": cache_device,
        "max_num_sequences_hint": max_num_sequences_hint,
        "max_num_tokens_hint": int(meta.max_num_tokens or max_num_sequences_hint),
    }
    meta._m3_static_buffers = placeholder
    return placeholder


def build_runtime_metadata_from_kv_manager(
    *,
    kv_cache_manager,
    request_ids,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    is_prefill: bool,
    prefix_lens: Optional[torch.Tensor] = None,
    extend_seq_lens_cpu: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    static_buffers: Optional[dict] = None,
) -> Tuple[MiniMaxM3SparseAttentionMetadata, torch.Tensor]:
    """Build a `MiniMaxM3SparseAttentionMetadata` from a real
    `MiniMaxM3KVCacheManagerV2`.

    Returns the metadata plus an `out_cache_loc` tensor `[num_new_tokens]`
    of the per-new-token slot ids the caller must write K/V/idx_K to. The
    `req_to_token[req_idx, pos] -> slot_id` view is built by expanding each
    of `get_block_ids_per_seq(request_ids)`'s block ids into a contiguous
    range of `tokens_per_block` slot ids.

    `device` places the returned tensors (default `seq_lens.device`). When
    `static_buffers` is provided, per-call values are written into those
    persistent buffers in place so `data_ptr()` stays constant across CUDA
    graph replays; otherwise fresh tensors are allocated per call.
    """
    tokens_per_block = int(kv_cache_manager.tokens_per_block)
    # block_ids_per_seq is a [batch_size, max_blocks_per_seq] tensor; row b
    # holds the block ids assigned to request_ids[b] in order.
    block_ids = kv_cache_manager.get_block_ids_per_seq(list(request_ids))
    batch = int(seq_lens.shape[0])
    max_blocks = int(block_ids.shape[1])
    max_kv_len = max_blocks * tokens_per_block
    if device is None:
        device = seq_lens.device

    if static_buffers is not None:
        if static_buffers.get("device") != device:
            raise ValueError(
                f"static_buffers device {static_buffers.get('device')} does not "
                f"match requested device {device}"
            )
        if "req_to_token" not in static_buffers:
            # Lazy first-call allocation: the caller passes a placeholder
            # dict (with only hints) and we allocate the persistent
            # tensors using the current scheduler step's actual geometry.
            # This handles the case where the manager's ``max_seq_len``
            # is smaller than what the CUDA-graph warmup actually
            # requests. The first allocation pins the buffer addresses
            # for the rest of this metadata instance's lifetime, so all
            # subsequent prepare() calls reuse the same ``data_ptr()``s
            # and CUDA graph capture/replay stays valid.
            max_num_sequences_hint = int(static_buffers.get("max_num_sequences_hint", batch))
            max_num_sequences = max(batch, max_num_sequences_hint)
            # For decode every request emits one Q token; for prefill the
            # builder will validate ``total_q`` against this size below.
            # The hint typically equals ``max_num_sequences`` for the
            # decode-only graph and ``max_seq_len`` for the prefill
            # graph; we use it as a floor and grow if the actual call
            # needs more.
            max_num_tokens_hint = int(static_buffers.get("max_num_tokens_hint", max_num_sequences))
            max_num_tokens = max(max_num_sequences, max_num_tokens_hint)
            # Grow ``max_kv_len`` to the current call's actual width plus
            # one extra block of headroom so any decode-step growth (one
            # newly-allocated block per generated token) does not force
            # a reallocation. The captured graph references the
            # ``data_ptr()`` we pick here, so we cannot grow this buffer
            # after capture.
            max_kv_len_alloc = max(
                max_kv_len + tokens_per_block, max_num_sequences_hint * tokens_per_block
            )

            allocated = allocate_minimax_m3_static_buffers(
                max_num_sequences=max_num_sequences,
                max_num_tokens=max_num_tokens,
                max_kv_len=max_kv_len_alloc,
                device=device,
            )
            # Mutate the caller-supplied dict in-place so the owner sees
            # the allocation on subsequent calls.
            static_buffers.update(allocated)
        if batch > static_buffers["max_num_sequences"]:
            raise ValueError(
                f"static_buffers max_num_sequences={static_buffers['max_num_sequences']} "
                f"is smaller than current batch={batch}; capture/replay would overflow"
            )
        if max_kv_len > static_buffers["max_kv_len"]:
            raise ValueError(
                f"static_buffers max_kv_len={static_buffers['max_kv_len']} "
                f"is smaller than current max_kv_len={max_kv_len}; "
                f"capture/replay would overflow"
            )

    # Ensure the per-sequence seq_lens lands on the target device; the
    # algorithm consumes it on the cache device and a stray CPU copy
    # would force ``.to(device)`` inside the forward (unsafe under
    # CUDA-graph capture). ``seq_lens_cpu`` is the CPU mirror used by
    # ``prepare()`` to compute scalar max lengths sync-free.
    if static_buffers is not None:
        # Write into the persistent seq_lens buffer in-place, then
        # expose a [:batch] slice. The slice shares ``data_ptr()`` with
        # the persistent buffer (slice starts at offset 0), so the
        # captured CUDA graph keeps reading from a stable address even
        # though the metadata's seq_lens field is a new view object each
        # call.
        seq_lens_buf = static_buffers["seq_lens_dev"]
        seq_lens_src = seq_lens.to(device=device, dtype=torch.int32, non_blocking=True)
        seq_lens_buf[:batch].copy_(seq_lens_src, non_blocking=True)
        seq_lens_dev = seq_lens_buf[:batch]
    else:
        seq_lens_dev = seq_lens.to(device) if seq_lens.device != device else seq_lens

    # Expand block ids -> per-token slot ids.
    # slot_id = block_id * tokens_per_block + offset_within_block
    block_ids_dev = block_ids.to(device).to(torch.int64)
    within_block = torch.arange(tokens_per_block, device=device, dtype=torch.int64)
    # Outer product per batch entry: [batch, max_blocks, tokens_per_block]
    slot_grid = block_ids_dev.unsqueeze(-1) * tokens_per_block + within_block
    req_to_token_fresh = slot_grid.reshape(batch, max_kv_len).to(torch.int32)

    if static_buffers is not None:
        req_to_token_buf = static_buffers["req_to_token"]
        # Write the batch's rows / columns into the persistent buffer.
        req_to_token_buf[:batch, :max_kv_len].copy_(req_to_token_fresh, non_blocking=True)
        # The algorithm reads ``req_to_token`` via
        # ``index_select(0, slot_ids)``; rows past ``batch`` are
        # never selected because ``slot_ids`` only enumerates
        # ``[0, batch)``. Exposing the full buffer keeps the captured
        # ``data_ptr()`` stable; the shape behaviour is identical to
        # capturing/replaying a graph with the buffer's max shape.
        req_to_token = req_to_token_buf
        slot_ids = static_buffers["slot_ids"][:batch]
    else:
        req_to_token = req_to_token_fresh
        slot_ids = torch.arange(batch, device=device, dtype=torch.int32)

    # Compute out_cache_loc: per-new-token slot ids, in flattened order
    # matching the q-token order the model layer projects. The Python
    # loops below run on CPU lists derived from the CPU-resident
    # ``seq_lens_cpu`` / ``prefix_lens`` / ``extend_seq_lens_cpu``, so
    # no GPU sync is needed at this point. The resulting
    # ``out_cache_loc`` tensor is constructed directly on ``device``.
    # The ``int(...item())`` reads against ``req_to_token`` are a CPU
    # sync but only ever run from ``prepare()`` (outside any CUDA-graph
    # capture window) — they are not in the forward path.
    if is_prefill:
        if extend_seq_lens_cpu is None:
            raise ValueError("prefill metadata requires extend_seq_lens_cpu")
        if prefix_lens is None:
            raise ValueError("prefill metadata requires prefix_lens")
        prefix_lens_cpu = prefix_lens.to("cpu").tolist()
        if static_buffers is not None:
            prefix_buf = static_buffers["prefix_lens"]
            prefix_src = prefix_lens.to(device=device, dtype=torch.int32, non_blocking=True)
            prefix_buf[:batch].copy_(prefix_src, non_blocking=True)
            prefix_lens_dev = prefix_buf[:batch]
        else:
            prefix_lens_dev = (
                prefix_lens.to(device) if prefix_lens.device != device else prefix_lens
            )
        out_cache_loc_list: List[int] = []
        cu_q: List[int] = [0]
        req_to_token_cpu = req_to_token_fresh.to("cpu")
        for b in range(batch):
            pref = int(prefix_lens_cpu[b])
            ext = int(extend_seq_lens_cpu[b])
            for offset in range(ext):
                slot = int(req_to_token_cpu[b, pref + offset].item())
                out_cache_loc_list.append(slot)
            cu_q.append(cu_q[-1] + ext)
        total_q = cu_q[-1]
        if static_buffers is not None:
            if total_q > static_buffers["max_num_tokens"]:
                raise ValueError(
                    f"static_buffers max_num_tokens={static_buffers['max_num_tokens']} "
                    f"is smaller than current total_q={total_q}"
                )
            out_cache_loc_buf = static_buffers["out_cache_loc"]
            out_cache_loc_src = torch.tensor(out_cache_loc_list, dtype=torch.int32, device=device)
            out_cache_loc_buf[:total_q].copy_(out_cache_loc_src, non_blocking=True)
            out_cache_loc = out_cache_loc_buf[:total_q]
            cu_seqlens_q_buf = static_buffers["cu_seqlens_q"]
            cu_seqlens_q_src = torch.tensor(cu_q, dtype=torch.int32, device=device)
            cu_seqlens_q_buf[: batch + 1].copy_(cu_seqlens_q_src, non_blocking=True)
            cu_seqlens_q = cu_seqlens_q_buf[: batch + 1]
            # Populate persistent q_batch_row / q_positions in-place so
            # the inner metadata's prepare() can leave them alone.
            q_batch_row_buf = static_buffers["q_batch_row"]
            q_positions_buf = static_buffers["q_positions"]
            for b in range(batch):
                start, end = cu_q[b], cu_q[b + 1]
                if end > start:
                    q_batch_row_buf[start:end] = b
                    pref = int(prefix_lens_cpu[b])
                    offsets = (
                        torch.arange(start, end, device=device, dtype=torch.int32) - start + pref
                    )
                    q_positions_buf[start:end].copy_(offsets, non_blocking=True)
            q_batch_row = q_batch_row_buf[:total_q]
            q_positions = q_positions_buf[:total_q]
        else:
            out_cache_loc = torch.tensor(out_cache_loc_list, dtype=torch.int32, device=device)
            cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
            q_batch_row = None
            q_positions = None
        meta = MiniMaxM3SparseAttentionMetadata(
            is_prefill=True,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_dev,
            seq_lens_cpu=seq_lens_cpu,
            prefix_lens=prefix_lens_dev,
            cu_seqlens_q=cu_seqlens_q,
            extend_seq_lens_cpu=list(extend_seq_lens_cpu),
            q_batch_row=q_batch_row,
            q_positions=q_positions,
        )
    else:
        # Decode: the new token sits at position seq_lens[b] - 1.
        seq_lens_cpu_list = seq_lens_cpu.to("cpu").tolist()
        out_cache_loc_list = []
        req_to_token_cpu = req_to_token_fresh.to("cpu")
        for b in range(batch):
            pos = int(seq_lens_cpu_list[b]) - 1
            out_cache_loc_list.append(int(req_to_token_cpu[b, pos].item()))
        if static_buffers is not None:
            if batch > static_buffers["max_num_tokens"]:
                raise ValueError(
                    f"static_buffers max_num_tokens={static_buffers['max_num_tokens']} "
                    f"is smaller than current batch={batch}"
                )
            out_cache_loc_buf = static_buffers["out_cache_loc"]
            out_cache_loc_src = torch.tensor(out_cache_loc_list, dtype=torch.int32, device=device)
            out_cache_loc_buf[:batch].copy_(out_cache_loc_src, non_blocking=True)
            out_cache_loc = out_cache_loc_buf[:batch]
        else:
            out_cache_loc = torch.tensor(out_cache_loc_list, dtype=torch.int32, device=device)
        meta = MiniMaxM3SparseAttentionMetadata(
            is_prefill=False,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_dev,
            seq_lens_cpu=seq_lens_cpu,
        )
    meta.prepare()
    return meta, out_cache_loc


def _build_msa_plans_for_metadata(
    *,
    m3_meta: "MiniMaxM3SparseAttentionMetadata",
    geometry: "MiniMaxM3SparseConfig",
    cache_device: torch.device,
    max_batch: int,
    kv_indices_buf: Optional[torch.Tensor],
    kv_page_indptr_buf: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Refresh the persistent paged-KV staging for one scheduler step.

    Returns the persistent `(kv_indices_buf, kv_page_indptr_buf)` staging
    buffers (stable `data_ptr()` across CUDA graph replays) so the caller
    can retain them for the next step. The per-step plan values (the staged
    page-table views plus the per-request CPU lens/offsets the forward path
    consumes) are written directly onto `m3_meta`'s `msa_*` attributes.
    """
    # 1) Derive per-request CPU tensors via the shared helper (same values
    #    the eager forward fallback in `msa_backend._whole_batch_lens`
    #    produces).
    lens = whole_batch_qo_lens(m3_meta)
    if lens is None:
        # Prefill metadata is incomplete; skip staging. The sparse
        # forward's eager fallback builds the page table in-forward
        # (safe when outside capture).
        return kv_indices_buf, kv_page_indptr_buf
    qo_lens_cpu, seq_lens_cpu, qo_offset_cpu = lens

    # 2) Allocate the staging buffers lazily on the first call. Sizes
    #    are picked so any padded batch up to `max_batch` (from
    #    `AttentionMetadata.max_num_sequences`) fits. The first allocation
    #    pins the buffer addresses for the rest of the metadata's
    #    lifetime, so CUDA graph capture/replay keeps reading stable
    #    `data_ptr()`s.
    if kv_indices_buf is None:
        # kv_indices is max_batch * max_pages_per_seq; page_size is the
        # sparse config's block_size (128 for M3) so max_pages_per_seq
        # comes from req_to_token's max_kv_len column dimension.  Use
        # the current metadata's `req_to_token` as the size witness.
        max_kv_len = int(m3_meta.req_to_token.shape[1])
        max_pages_per_seq = max(1, max_kv_len // int(geometry.block_size))
        max_kv_indices = max_batch * max_pages_per_seq
        kv_indices_buf = torch.zeros(max_kv_indices, dtype=torch.int32, device=cache_device)
        kv_page_indptr_buf = torch.zeros(max_batch + 1, dtype=torch.int32, device=cache_device)

    # 3) Refresh the page table in-place into the persistent buffers.
    kv_indices, kv_page_indptr = build_stable_kv_indices(
        req_to_token=m3_meta.req_to_token,
        slot_ids=m3_meta.slot_ids,
        seq_lens=m3_meta.seq_lens,
        seq_lens_cpu=m3_meta.seq_lens_cpu,
        page_size=int(geometry.block_size),
        dst=kv_indices_buf,
        kv_page_indptr_dst=kv_page_indptr_buf,
    )

    # Write the per-step plan values directly onto the sparse attention
    # metadata. The decode capacity constants stay stable across steps so
    # the decode state's geometry key is constant.
    m3_meta.msa_kv_indices = kv_indices
    m3_meta.msa_kv_page_indptr = kv_page_indptr
    m3_meta.msa_qo_lens_cpu = qo_lens_cpu
    m3_meta.msa_kv_lens_cpu = seq_lens_cpu
    m3_meta.msa_qo_offset_cpu = qo_offset_cpu
    m3_meta.msa_max_batch = int(max_batch)
    m3_meta.msa_max_kv_len = int(m3_meta.req_to_token.shape[1])
    return kv_indices_buf, kv_page_indptr_buf


def build_m3_sparse_metadata_and_plans(
    meta,
    *,
    geometry: Optional["MiniMaxM3SparseConfig"],
) -> Optional[dict]:
    """Build the per-step MiniMax-M3 sparse attachment and MSA plans.

    Backend-neutral so both the Triton and MSA metadata classes produce
    the identical attachment without duplicating the logic. `meta` must
    expose the standard attention-metadata attributes; `geometry` is the
    layer-invariant `MiniMaxM3SparseConfig`.

    All CUDA-graph-stable buffers are owned by `meta` (the static per-graph
    buffers and the MSA paged-KV staging buffers). This function reads them
    off `meta` and writes any newly allocated buffers back, so their
    `data_ptr()` stays constant across replays.

    Publishes the built per-forward sparse metadata as `meta.m3_sparse_metadata`
    and the per-new-token slot ids as `meta.m3_out_cache_loc`, and returns the
    sparse metadata (or None when the manager is not an M3 sparse cache or
    the batch is empty).
    """
    kv_cache_manager = meta.kv_cache_manager
    if kv_cache_manager is None or not hasattr(kv_cache_manager, "get_index_k_buffer"):
        return None
    request_ids = meta.request_ids
    seq_lens = meta.seq_lens
    if request_ids is None or seq_lens is None:
        return None
    num_contexts = int(meta.num_contexts or 0)
    batch_size = int(seq_lens.shape[0])
    if batch_size == 0:
        return None

    cache_device = m3_cache_device(meta)
    # All graph-stable buffers live on the metadata; pull the current
    # ones (allocated lazily on first use) so the builder can refresh
    # them in place.
    static_buffers = maybe_build_static_buffers_placeholder(meta, cache_device)
    kv_indices_buf = getattr(meta, "_msa_kv_indices_buf", None)
    kv_page_indptr_buf = getattr(meta, "_msa_kv_page_indptr_buf", None)

    # `seq_lens_cpu` is a `TrtllmAttentionMetadata` field (the MSA metadata
    # path) but not part of the base `AttentionMetadata` (the Triton path),
    # so probe for it and fall back to a D2H copy.
    seq_lens_cpu = getattr(meta, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = seq_lens.detach().to("cpu")

    kv_cache_params = meta.kv_cache_params
    num_cached_per_seq = (
        kv_cache_params.num_cached_tokens_per_seq
        if kv_cache_params is not None
        else [0] * batch_size
    )
    kv_lens_cpu_list = [
        int(num_cached_per_seq[b]) + int(seq_lens_cpu[b].item()) for b in range(batch_size)
    ]
    kv_lens_cpu = torch.tensor(kv_lens_cpu_list, dtype=torch.int32)
    kv_lens_dev = kv_lens_cpu.to(device=cache_device, non_blocking=True)

    use_msa = bool(getattr(kv_cache_manager, "use_msa", False))

    is_extend = num_contexts > 0
    if is_extend:
        prefix_lens_list = [int(num_cached_per_seq[b]) for b in range(batch_size)]
        extend_seq_lens_cpu = [kv_lens_cpu_list[b] - prefix_lens_list[b] for b in range(batch_size)]
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=cache_device)
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
        # The MSA decode path assumes a single query token per request
        # (decode_wrapper/dispatch.py packs qo_len=1). Speculative
        # decoding emits multiple draft query tokens per generation step,
        # which this path cannot represent, so reject it with a clear
        # error rather than silently mis-staging out_cache_loc. The Triton
        # reference path shares this builder but keeps its prior behavior,
        # so gate the rejection on the MSA backend only.
        if use_msa and batch_size > 0 and int(seq_lens_cpu[:batch_size].max().item()) > 1:
            raise NotImplementedError(
                "MiniMax-M3 MSA sparse attention does not support speculative "
                "decoding (multiple query tokens per decode step). Disable "
                "speculative decoding or use the non-MSA MiniMax-M3 backend."
            )
        m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            seq_lens=kv_lens_dev,
            seq_lens_cpu=kv_lens_cpu,
            is_prefill=False,
            device=cache_device,
            static_buffers=static_buffers,
        )

    # Publish the built sparse metadata and per-new-token slot ids as
    # direct attributes on the attention metadata.
    meta.m3_sparse_metadata = m3_meta
    meta.m3_out_cache_loc = out_cache_loc

    if use_msa and geometry is not None:
        max_batch = int(meta.max_num_sequences or meta.max_num_requests)
        # Persist the (possibly first-allocated) staging buffers back onto
        # the metadata so the next step reuses the same data_ptr(). The
        # per-step plan values are written onto `m3_meta` by the helper.
        meta._msa_kv_indices_buf, meta._msa_kv_page_indptr_buf = _build_msa_plans_for_metadata(
            m3_meta=m3_meta,
            geometry=geometry,
            cache_device=cache_device,
            max_batch=max_batch,
            kv_indices_buf=kv_indices_buf,
            kv_page_indptr_buf=kv_page_indptr_buf,
        )
    return m3_meta


@functools.lru_cache(maxsize=1)
def get_minimax_m3_attention_metadata_cls():
    """Return :class:`MiniMaxM3AttentionMetadata` (lazy import)."""
    from ...interface import AttentionMetadata

    class MiniMaxM3AttentionMetadata(AttentionMetadata):
        """:class:`AttentionMetadata` that pre-builds MiniMax-M3 metadata.

        `prepare()` builds the per-forward `MiniMaxM3SparseAttentionMetadata`
        and per-new-token `out_cache_loc` once per scheduler step, outside
        the CUDA-graph capture window, and publishes them as
        `self.m3_sparse_metadata` / `self.m3_out_cache_loc` so the forward
        reads them with no capture-time CPU->GPU copies. Test paths may set
        those attributes directly instead of going through `prepare()`.
        """

        m3_sparse_metadata: Optional["MiniMaxM3SparseAttentionMetadata"] = None
        m3_out_cache_loc: Optional[torch.Tensor] = None
        # Persistent buffers that keep the sparse-metadata tensor addresses
        # stable across CUDA-graph replays (see
        # allocate_minimax_m3_static_buffers); allocated lazily on first use.
        _m3_static_buffers: Optional[dict] = None
        # MSA paged-KV page-table staging buffers, allocated lazily when the
        # cache manager has use_msa=True.
        _msa_kv_indices_buf: Optional[torch.Tensor] = None
        _msa_kv_page_indptr_buf: Optional[torch.Tensor] = None

        def prepare(self) -> None:
            super().prepare()
            # Rebuild the M3 metadata each step to reflect the current
            # seq_lens / request_ids / num_cached_tokens. model_engine runs
            # prepare() outside capture; the builder writes into the
            # persistent static/staging buffers so the captured forward
            # reads stable data_ptr()s across replays.
            self.m3_sparse_metadata = None
            self.m3_out_cache_loc = None
            build_m3_sparse_metadata_and_plans(self, geometry=get_global_msa_geometry())

    return MiniMaxM3AttentionMetadata


__all__ = [
    "MiniMaxM3SparseConfig",
    "MiniMaxM3SparseAttentionMetadata",
    "allocate_minimax_m3_static_buffers",
    "build_m3_sparse_metadata_and_plans",
    "build_runtime_metadata_from_kv_manager",
    "build_stable_kv_indices",
    "ensure_metadata_on_device",
    "get_global_msa_geometry",
    "get_minimax_m3_attention_metadata_cls",
    "set_global_msa_geometry",
]
