# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 Triton reference per-forward metadata.

Contains:

  * :class:`MiniMaxM3TritonSparseAttentionMetadata` -- per-forward metadata
    dataclass with a CUDA-graph-safe :meth:`prepare`.
  * Helpers to migrate metadata across devices, build it from a real
    :class:`KVCacheManagerV2`, and pre-allocate CUDA-graph-stable buffers.
  * :class:`MiniMaxM3AttentionMetadata` -- the :class:`AttentionMetadata`
    subclass the pyexecutor wires into the M3 sparse layer's forward path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from ...trtllm import TrtllmAttentionMetadata
from .common import build_paged_kv_slot_mapping


@dataclass
class MiniMaxM3TritonSparseAttentionMetadata:
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
        ``True`` routes through the *extend* kernel
        (``minimax_m3_sparse_prefill``), which handles both pure
        prefill AND mixed prefill+decode batches — decode rows in a
        mixed batch appear as 1-slot extends (``extend_seq_len=1``,
        ``prefix_len=num_cached``).
        ``False`` is a perf-only specialization for pure-decode
        batches (``num_contexts == 0``); it is mathematically
        equivalent to the ``True`` path with
        ``extend_seq_len=[1]*batch``. See
        ``test_iter131_metadata_prepare_mixed_batch_uses_extend_path``.
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
    # Query tokens per request on decode-shaped metadata: 1 normally,
    # 1 + draft_len under one-model Eagle3 verify. Consumed by the dense
    # SDPA decode ladder mask.
    decode_qo_len: int = 1
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
            max_k = int(self.seq_lens_cpu[:batch_size].max().item())
            # Optimistic overlap+spec lengths can overhang the page table at
            # a page boundary; SDPA consumes max_seqlen_k as the exact
            # mask/gather width, so clamp to the table.
            self.max_seqlen_k = min(max_k, int(self.req_to_token.shape[1]))


def ensure_metadata_on_device(
    metadata: "MiniMaxM3TritonSparseAttentionMetadata",
    device: torch.device,
) -> "MiniMaxM3TritonSparseAttentionMetadata":
    """Return ``metadata`` with every GPU-consumed tensor on ``device``.

    Constructs a new :class:`MiniMaxM3TritonSparseAttentionMetadata` whose
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


def _build_runtime_metadata_fresh(
    *,
    kv_cache_manager,
    request_ids,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    is_prefill: bool,
    prefix_lens: Optional[torch.Tensor],
    extend_seq_lens_cpu: Optional[List[int]],
    device: torch.device,
) -> Tuple[MiniMaxM3TritonSparseAttentionMetadata, torch.Tensor]:
    """Fresh-allocation build of the Triton reference metadata.

    Delegates the backend-neutral req_to_token, slot_ids and out_cache_loc
    derivation to common.build_paged_kv_slot_mapping and adds the Triton-only
    fields: cu_seqlens_q, prefix_lens on device, and the scalar max_seqlen
    values and per-query-token tensors that prepare() computes. Used when no
    graph-stable static_buffers are supplied; the static_buffers path in
    build_runtime_metadata_from_kv_manager keeps its own in-place buffer writes.
    """
    batch = int(seq_lens.shape[0])
    seq_lens_dev = seq_lens.to(device) if seq_lens.device != device else seq_lens

    if is_prefill:
        if extend_seq_lens_cpu is None:
            raise ValueError("prefill metadata requires extend_seq_lens_cpu")
        if prefix_lens is None:
            raise ValueError("prefill metadata requires prefix_lens")
        prefix_lens_dev = prefix_lens.to(device) if prefix_lens.device != device else prefix_lens
        qo_lens_cpu = torch.tensor([int(x) for x in extend_seq_lens_cpu], dtype=torch.int32)
        qo_offset_cpu = prefix_lens.detach().to(device="cpu", dtype=torch.int32)
        req_to_token, slot_ids, out_cache_loc = build_paged_kv_slot_mapping(
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            qo_lens_cpu=qo_lens_cpu,
            qo_offset_cpu=qo_offset_cpu,
            device=device,
        )
        cu_q: List[int] = [0]
        for ext in extend_seq_lens_cpu:
            cu_q.append(cu_q[-1] + int(ext))
        cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
        meta = MiniMaxM3TritonSparseAttentionMetadata(
            is_prefill=True,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_dev,
            seq_lens_cpu=seq_lens_cpu,
            prefix_lens=prefix_lens_dev,
            cu_seqlens_q=cu_seqlens_q,
            extend_seq_lens_cpu=list(extend_seq_lens_cpu),
            q_batch_row=None,
            q_positions=None,
        )
    else:
        # Decode: the new token sits at position seq_lens[b] - 1.
        qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
        qo_offset_cpu = seq_lens_cpu.detach().to(device="cpu", dtype=torch.int32) - 1
        req_to_token, slot_ids, out_cache_loc = build_paged_kv_slot_mapping(
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            qo_lens_cpu=qo_lens_cpu,
            qo_offset_cpu=qo_offset_cpu,
            device=device,
        )
        meta = MiniMaxM3TritonSparseAttentionMetadata(
            is_prefill=False,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_dev,
            seq_lens_cpu=seq_lens_cpu,
        )
    meta.prepare()
    return meta, out_cache_loc


def derive_q_positions_and_cache_slots(
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q_batch_row: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-Q-token K-side positions and KV slot ids, on device, sync-free.

    Shared by the metadata builder and the ``on_update_kv_lens``
    re-derivation so the two cannot drift.
    """
    total_q = int(q_batch_row.shape[0])
    qbr = q_batch_row.to(torch.long)
    tok = torch.arange(total_q, dtype=torch.int32, device=q_batch_row.device)
    q_positions = prefix_lens[qbr] + (tok - cu_seqlens_q[qbr])
    # Optimistic prefix_lens can overhang the last allocated page; the
    # overhanging slots are placeholders (on_update_kv_lens re-derives them
    # before any forward reads them) but the gather must stay in bounds.
    # Clamp only the table index — non-inplace, since ``.to`` aliases int64
    # inputs.
    idx = q_positions.to(torch.long).clamp(min=0, max=req_to_token.shape[1] - 1)
    flat = qbr * req_to_token.shape[1] + idx
    return q_positions, req_to_token.reshape(-1).index_select(0, flat)


def derive_decode_cache_slots(req_to_token: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Decode-row KV slot ids (new token at position ``seq_lens[b] - 1``).

    Same in-bounds-placeholder clamp contract as
    :func:`derive_q_positions_and_cache_slots`; the ``min=0`` floor
    additionally covers zero-length dummy rows indexing ``-1``.
    """
    rows = torch.arange(seq_lens.shape[0], device=seq_lens.device, dtype=torch.long)
    idx = (seq_lens.to(torch.long) - 1).clamp_(min=0, max=req_to_token.shape[1] - 1)
    flat = rows * req_to_token.shape[1] + idx
    return req_to_token.reshape(-1).index_select(0, flat)


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
) -> Tuple[MiniMaxM3TritonSparseAttentionMetadata, torch.Tensor]:
    """Build a :class:`MiniMaxM3TritonSparseAttentionMetadata` from a real
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
    if device is None:
        device = seq_lens.device

    if static_buffers is None:
        return _build_runtime_metadata_fresh(
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            is_prefill=is_prefill,
            prefix_lens=prefix_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            device=device,
        )

    tokens_per_block = int(kv_cache_manager.tokens_per_block)
    # block_ids_per_seq is a [batch_size, max_blocks_per_seq] tensor; row b
    # holds the block ids assigned to request_ids[b] in order.
    block_ids = kv_cache_manager.get_block_ids_per_seq(list(request_ids))
    batch = int(seq_lens.shape[0])
    max_blocks = int(block_ids.shape[1])
    max_kv_len = max_blocks * tokens_per_block

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

    # out_cache_loc must be flattened in the q-token order the model layer
    # projects, or K/V lands in the wrong requests' slots.
    if is_prefill:
        if extend_seq_lens_cpu is None:
            raise ValueError("prefill metadata requires extend_seq_lens_cpu")
        if prefix_lens is None:
            raise ValueError("prefill metadata requires prefix_lens")
        if static_buffers is not None:
            prefix_buf = static_buffers["prefix_lens"]
            prefix_src = prefix_lens.to(device=device, dtype=torch.int32, non_blocking=True)
            prefix_buf[:batch].copy_(prefix_src, non_blocking=True)
            prefix_lens_dev = prefix_buf[:batch]
        else:
            prefix_lens_dev = (
                prefix_lens.to(device) if prefix_lens.device != device else prefix_lens
            )
        cu_q: List[int] = [0]
        for ext in extend_seq_lens_cpu:
            cu_q.append(cu_q[-1] + int(ext))
        total_q = cu_q[-1]
        cu_seqlens_q_src = torch.tensor(cu_q, dtype=torch.int32, device=device)
        q_batch_row_src = torch.repeat_interleave(
            torch.arange(batch, device=device, dtype=torch.int32),
            torch.tensor(extend_seq_lens_cpu, dtype=torch.int64, device=device),
        )
        q_positions_src, out_cache_loc_src = derive_q_positions_and_cache_slots(
            req_to_token, prefix_lens_dev, cu_seqlens_q_src, q_batch_row_src
        )
        if static_buffers is not None:
            if total_q > static_buffers["max_num_tokens"]:
                raise ValueError(
                    f"static_buffers max_num_tokens={static_buffers['max_num_tokens']} "
                    f"is smaller than current total_q={total_q}"
                )
            out_cache_loc_buf = static_buffers["out_cache_loc"]
            out_cache_loc_buf[:total_q].copy_(out_cache_loc_src, non_blocking=True)
            out_cache_loc = out_cache_loc_buf[:total_q]
            cu_seqlens_q_buf = static_buffers["cu_seqlens_q"]
            cu_seqlens_q_buf[: batch + 1].copy_(cu_seqlens_q_src, non_blocking=True)
            cu_seqlens_q = cu_seqlens_q_buf[: batch + 1]
            q_batch_row_buf = static_buffers["q_batch_row"]
            q_positions_buf = static_buffers["q_positions"]
            q_batch_row_buf[:total_q].copy_(q_batch_row_src, non_blocking=True)
            q_positions_buf[:total_q].copy_(q_positions_src, non_blocking=True)
            q_batch_row = q_batch_row_buf[:total_q]
            q_positions = q_positions_buf[:total_q]
        else:
            out_cache_loc = out_cache_loc_src
            cu_seqlens_q = cu_seqlens_q_src
            q_batch_row = q_batch_row_src
            q_positions = q_positions_src
        meta = MiniMaxM3TritonSparseAttentionMetadata(
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
        out_cache_loc_src = derive_decode_cache_slots(req_to_token, seq_lens_dev)
        if static_buffers is not None:
            if batch > static_buffers["max_num_tokens"]:
                raise ValueError(
                    f"static_buffers max_num_tokens={static_buffers['max_num_tokens']} "
                    f"is smaller than current batch={batch}"
                )
            out_cache_loc_buf = static_buffers["out_cache_loc"]
            out_cache_loc_buf[:batch].copy_(out_cache_loc_src, non_blocking=True)
            out_cache_loc = out_cache_loc_buf[:batch]
        else:
            out_cache_loc = out_cache_loc_src
        meta = MiniMaxM3TritonSparseAttentionMetadata(
            is_prefill=False,
            req_to_token=req_to_token,
            slot_ids=slot_ids,
            seq_lens=seq_lens_dev,
            seq_lens_cpu=seq_lens_cpu,
        )
    meta.prepare()
    return meta, out_cache_loc


class MiniMaxM3AttentionMetadata(TrtllmAttentionMetadata):
    """:class:`TrtllmAttentionMetadata` that pre-builds MiniMax-M3 metadata.

    Subclasses :class:`TrtllmAttentionMetadata` (precedent:
    ``DSAtrtllmAttentionMetadata``): one-model Eagle3 draft layers run
    :class:`TrtllmAttention` against this shared per-step metadata, and the
    engine gates spec-dec plumbing on the TRTLLM ``isinstance``.

    Overrides :meth:`prepare` so the M3-sparse
    :class:`MiniMaxM3TritonSparseAttentionMetadata` and the per-new-token
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
    # ``MiniMaxM3TritonSparseAttentionMetadata`` tensor addresses stable
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
        #
        # Mixed prefill+decode batches always take the extend path:
        # the prefill kernel handles decode rows as 1-slot extends.
        # The decode branch below is a pure-decode-only perf
        # specialization. (iter-131 regression: previously a wrong
        # predicate routed mixed batches into the decode branch and
        # crashed in index_copy_.)
        # Multi-token gen rows (spec verify) also route through the extend
        # path as prefix+window extends; decode stays one-token-per-row.
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

    def on_update_kv_lens(self) -> None:
        """Re-derive the M3 attachment from the corrected ``kv_lens_cuda``.

        Under the overlap scheduler + speculative decoding, prepare()
        runs with optimistic cached counts (full draft acceptance) and
        the engine corrects ``kv_lens_cuda`` on device before invoking
        this hook (same pattern as ``DSAtrtllmAttentionMetadata``).
        On-device, sync-free, and idempotent; ``seq_lens_cpu`` /
        ``max_seqlen_k`` keep the optimistic values — they only bound
        arange widths that the kernels mask by ``seq_lens``.
        """
        super().on_update_kv_lens()
        attachment = self.minimax_m3
        if not attachment:
            return
        meta = attachment["metadata"]
        out_cache_loc = attachment["out_cache_loc"]
        batch = int(meta.slot_ids.shape[0])
        kv_lens = self.kv_lens_cuda[:batch]
        meta.seq_lens[:batch].copy_(kv_lens)
        if meta.is_prefill:
            # Only the K-side prefix moves with rejections; the Q-side
            # structure (cu_seqlens_q, q_batch_row) is fixed per step.
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
            # Identity today; keeps 0-draft corrections right if they
            # become reachable.
            out_cache_loc[:batch].copy_(derive_decode_cache_slots(meta.req_to_token, kv_lens))


__all__ = [
    "MiniMaxM3AttentionMetadata",
    "MiniMaxM3TritonSparseAttentionMetadata",
    "allocate_minimax_m3_static_buffers",
    "build_runtime_metadata_from_kv_manager",
    "derive_decode_cache_slots",
    "derive_q_positions_and_cache_slots",
    "ensure_metadata_on_device",
]
