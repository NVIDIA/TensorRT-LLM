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
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


@dataclass(frozen=True)
class MiniMaxM3SparseConfig:
    """Per-rank kernel parameter bundle for MiniMax-M3 sparse attention.

    This is **not** a user-facing config (use
    :class:`tensorrt_llm.llmapi.llm_args.MiniMaxM3SparseAttentionConfig`
    for that). It is the layer-invariant, post-TP-shard parameter bundle
    that backend kernels and reference helpers consume. The user knobs
    come from the Pydantic config; ``num_q_heads`` / ``num_kv_heads`` /
    ``head_dim`` come from the per-rank model geometry and must be
    supplied by the caller (typically via
    :meth:`from_sparse_attention_config`).
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
    def from_sparse_attention_config(
        cls,
        sparse_attention_config: "MiniMaxM3SparseAttentionConfig",
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> "MiniMaxM3SparseConfig":
        """Build a kernel param bundle from the user-facing Pydantic config.

        The Pydantic config supplies the algorithm knobs; the per-rank
        model geometry (``num_q_heads``, ``num_kv_heads``, ``head_dim``)
        is supplied separately by the caller because it is not part of
        the user-facing config — it is derived from the model and the TP
        mapping. This mirrors the DSA pattern where the
        :class:`Indexer` reads the Pydantic config plus its own
        constructor-provided geometry.
        """
        cfg = sparse_attention_config
        return cls(
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            num_index_heads=int(cfg.sparse_num_index_heads),
            sparse_index_dim=int(cfg.sparse_index_dim),
            block_size=int(cfg.sparse_block_size),
            topk=int(cfg.sparse_topk_blocks),
            init_blocks=int(cfg.sparse_init_blocks),
            local_blocks=int(cfg.sparse_local_blocks),
            score_type=str(cfg.sparse_score_type),
        )


@dataclass
class MiniMaxM3SparseAttentionMetadata:
    """Per-forward metadata for MiniMax-M3 sparse attention.

    Mirrors the shape of SGLang's
    :class:`MiniMaxSparseAttnBackend`-side metadata but is constructed
    from the paged-cache view that :class:`KVCacheManagerV2` exposes:

      * ``req_to_token[req_idx, pos] -> slot_id``
      * ``slot_ids[batch_idx]``  -- which ``req_to_token`` row the
        ``batch_idx``-th sequence in this forward corresponds to.

    Fields
    ------
    is_prefill : bool
        ``True`` if this metadata describes a prefill / extend forward
        (multiple Q tokens per sequence). ``False`` for decode (1 Q
        token per sequence).
    req_to_token : torch.Tensor
        Paged ``[max_reqs, max_kv_len]`` int32 mapping from
        ``(req_idx, pos)`` to slot index. Shared across layers.
    slot_ids : torch.Tensor
        ``[batch_size]`` int32 mapping from ``batch_idx`` to
        ``req_to_token`` row index.
    seq_lens : torch.Tensor
        ``[batch_size]`` int32 total K length per sequence (prefix +
        current chunk, or full context for decode).
    prefix_lens : torch.Tensor or None
        ``[batch_size]`` int32 prefix length per sequence; required for
        prefill, ignored for decode.
    cu_seqlens_q : torch.Tensor or None
        ``[batch_size + 1]`` int32 cumulative Q-length offsets; required
        for prefill, ignored for decode.
    seq_lens_cpu : torch.Tensor
        CPU mirror of ``seq_lens`` used by :meth:`prepare` to compute
        :attr:`max_seqlen_k` without a GPU sync.
    extend_seq_lens_cpu : list[int] or None
        Per-sequence Q-length list (CPU) used by :meth:`prepare` to
        compute :attr:`max_seqlen_q` for prefill without a GPU sync.
        Ignored for decode (decode always has 1 Q token / sequence).
    q_batch_row : torch.Tensor or None
        ``[total_q_tokens]`` int32, only meaningful for prefill: for
        each Q token, which batch row it belongs to. Built by
        :meth:`prepare` from ``cu_seqlens_q``.
    q_positions : torch.Tensor or None
        ``[total_q_tokens]`` int32, only meaningful for prefill: each
        Q token's K-side position (prefix_lens[b] + offset). Built by
        :meth:`prepare` from ``cu_seqlens_q`` and ``prefix_lens``.
    max_seqlen_q : int
        Populated by :meth:`prepare`. CUDA-graph safe scalar.
    max_seqlen_k : int
        Populated by :meth:`prepare`. CUDA-graph safe scalar.
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


def replace_metadata(
    metadata: MiniMaxM3SparseAttentionMetadata,
    **changes,
) -> MiniMaxM3SparseAttentionMetadata:
    """Helper around :func:`dataclasses.replace` for ``metadata``.

    Provided so callers can build a decode metadata from a prefill
    metadata without manually re-typing every field.
    """
    return dataclasses.replace(metadata, **changes)


def allocate_minimax_m3_static_buffers(
    *,
    max_num_sequences: int,
    max_num_tokens: int,
    max_kv_len: int,
    device: torch.device,
) -> dict:
    """Allocate persistent per-graph buffers for MiniMax-M3 metadata.

    Under CUDA-graph capture/replay every tensor the captured kernels
    read must keep the same ``data_ptr()`` across replays. The default
    :func:`build_runtime_metadata_from_kv_manager` path allocates fresh
    tensors per call, which silently breaks graph replay: the captured
    kernel keeps reading from the warmup tensor's freed memory, so the
    enabled-graph run either produces wrong tokens (numerical drift) or
    crashes inside ``index_select`` when the stale memory contains
    out-of-bounds indices (``Indexing.cu:1515`` ``srcIndex <
    srcSelectDimSize`` assert).

    The buffers returned here are sized for the largest batch the
    captured graph will ever see: ``max_num_sequences`` requests with
    up to ``max_kv_len`` cache slots, plus ``max_num_tokens`` total Q
    tokens. They live on ``device`` so the forward path consumes them
    without any device migration. ``build_runtime_metadata_from_kv_manager``
    writes per-call values into these buffers in-place via ``.copy_()``
    so the captured graph always sees the same addresses.

    Returns a dict carrying the persistent tensors plus the geometry
    parameters used at allocation time; callers compare those parameters
    on subsequent prepare() calls to detect a geometry change (which
    would be a workflow bug under fixed-batch CUDA graph).
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
    """Build a :class:`MiniMaxM3SparseAttentionMetadata` from a real
    :class:`MiniMaxM3KVCacheManagerV2`.

    Returns the populated metadata plus an ``out_cache_loc`` tensor
    ``[num_new_tokens]`` listing the per-new-token slot ids the caller
    must write the projected K/V/idx_K to before calling the algorithm.

    The slot view of the paged main K/V cache is built by combining
    ``kv_cache_manager.get_block_ids_per_seq(request_ids)`` (per-request
    block ids in order) with the configured ``tokens_per_block`` to
    expand each block id into a contiguous range of ``tokens_per_block``
    slot ids. The resulting ``req_to_token[req_idx, pos] -> slot_id``
    matches the main cache's per-token addressing exactly.

    ``device`` selects the placement of the returned tensors. Default
    is ``seq_lens.device`` so existing callers (focused tests) keep
    their behaviour; the production
    :class:`MiniMaxM3AttentionMetadata.prepare` path passes the cache
    device explicitly so the forward never needs a CPU->GPU copy.

    ``static_buffers`` is the CUDA-graph-stable buffer dict produced by
    :func:`allocate_minimax_m3_static_buffers`. When provided, this
    function writes every per-call tensor into the persistent buffers
    in-place (via ``.copy_()`` / slice assignment) and returns metadata
    pointing into those persistent buffers. That keeps ``data_ptr()``
    constant across replays, which is the contract the CUDA graph
    runner expects. When omitted, the function falls back to fresh
    per-call allocations (preserves existing focused-test behaviour).

    This helper is the integration glue between the
    pyexecutor-driven runtime metadata and the MiniMax-M3
    algorithm's metadata shape. Tests can call it directly to verify
    the end-to-end runtime path without going through the full LLM
    forward.
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


@functools.lru_cache(maxsize=1)
def get_minimax_m3_attention_metadata_cls():
    """Return :class:`MiniMaxM3AttentionMetadata` (lazy import).

    The class extends :class:`AttentionMetadata` so the pyexecutor's
    metadata-creation/prepare hooks (model_engine.py) drive M3 metadata
    construction outside the CUDA-graph capture window. Building the
    M3-sparse ``req_to_token`` / ``slot_ids`` / ``out_cache_loc``
    tensors during ``prepare()`` lands them on the GPU **before** the
    forward call; the forward path then reads from the pre-built
    attachment and performs no CPU->GPU copies, which is required for
    CUDA-graph capture safety (``cudaErrorStreamCaptureUnsupported``
    fires for CPU->GPU ``memcpyAsync`` calls inside a captured stream).
    """
    from ...interface import AttentionMetadata

    class MiniMaxM3AttentionMetadata(AttentionMetadata):
        """:class:`AttentionMetadata` that pre-builds MiniMax-M3 metadata.

        Overrides :meth:`prepare` so the M3-sparse
        :class:`MiniMaxM3SparseAttentionMetadata` and the per-new-token
        ``out_cache_loc`` are built **once per scheduler step**, on the
        cache device, before the model forward runs.  The result is
        stored as ``self.minimax_m3 = {"metadata": m3_meta,
        "out_cache_loc": out_cache_loc}`` so the model layer's
        ``_dense_forward`` and ``_sparse_forward`` can read it without
        any device migration.

        Test paths that build their own metadata can short-circuit by
        attaching ``attn_metadata.minimax_m3`` directly before calling
        the forward; those paths do not go through :meth:`prepare`.
        """

        minimax_m3: Optional[dict] = None
        # Lazily allocated dict of persistent device buffers used to keep
        # ``MiniMaxM3SparseAttentionMetadata`` tensor addresses stable
        # across CUDA-graph capture/replay. None until the first
        # ``prepare()`` call decides to use them (``is_cuda_graph`` /
        # graph-stable mode).
        _m3_static_buffers: Optional[dict] = None

        def _maybe_get_m3_static_buffers(
            self, cache_device: torch.device, kv_cache_manager
        ) -> Optional[dict]:
            """Return persistent M3 buffers when graph stability is
            required.

            Allocates the persistent buffer dict the first time it is
            needed and caches it on ``self._m3_static_buffers``. We
            allocate the buffers under two conditions:

              * ``self.is_cuda_graph`` is True -- the captured graph
                requires stable ``data_ptr()`` across replays; OR
              * the previous prepare() already allocated buffers --
                we keep using them so the algorithm sees the same
                addresses even between non-graph and graph-mode calls
                (which can happen when the model engine alternates
                between eager warmup and graph replay).

            Returns ``None`` when no static buffers should be used (e.g.
            eager-only test paths that rely on per-call allocations).
            """
            need_static = (
                bool(getattr(self, "is_cuda_graph", False)) or self._m3_static_buffers is not None
            )
            if not need_static:
                return None
            if self._m3_static_buffers is not None:
                bufs = self._m3_static_buffers
                if bufs.get("device") == cache_device:
                    return bufs

            # First-time use: return an empty placeholder dict.
            # ``build_runtime_metadata_from_kv_manager`` performs the
            # actual allocation lazily on the first call where the
            # current scheduler step's geometry (max_kv_len from the
            # manager's block-id table, total_q from extend_seq_lens,
            # actual batch size after CUDA-graph padding) is known.
            # That removes the need to predict the warmup geometry up
            # front. The first allocation pins the buffer addresses
            # for the rest of this metadata instance's lifetime, so all
            # subsequent prepare() calls reuse the same ``data_ptr()``s
            # and CUDA graph capture/replay stays valid.
            placeholder: dict = {
                "device": cache_device,
                # Caller-provided hints used by the lazy allocator below
                # when it sizes the persistent buffers on the first real
                # prepare() call.
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

            # Any batch containing a context (prefill or chunked extend)
            # request takes the extend path. For prefill rows
            # ``num_cached_per_seq`` is ``prefix_lens`` and the full new
            # chunk is ``extend_seq_len``; for decode rows
            # ``num_cached`` is ``kv_len - 1`` and ``extend_seq_len`` is
            # 1, so the same builder produces the correct one-slot
            # entry. Pure-decode batches (``num_contexts == 0``) still
            # take the decode optimization for CUDA-graph warmup
            # geometry.
            is_extend = num_contexts > 0
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

    return MiniMaxM3AttentionMetadata


__all__ = [
    "MiniMaxM3SparseConfig",
    "MiniMaxM3SparseAttentionMetadata",
    "allocate_minimax_m3_static_buffers",
    "build_runtime_metadata_from_kv_manager",
    "ensure_metadata_on_device",
    "get_minimax_m3_attention_metadata_cls",
    "replace_metadata",
]
