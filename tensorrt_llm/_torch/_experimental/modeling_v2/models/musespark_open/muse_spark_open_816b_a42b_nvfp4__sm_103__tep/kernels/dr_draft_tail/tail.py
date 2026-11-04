# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash 2 draft head's tail after the vocab-sharded LM-head GEMV as three CUDA launches (``drafter_folds``).

Part of the ``drafter_folds`` fast path, on top of ``draft_head`` (``install``, from the shell's drafter install,
collective). ``VARIANT`` selects the kernels: variant 3, whose walk reduces with ``redux.sync``, loads its tail
operands before its wait and issues its edge loads together (variant 2 is the walk without the grouped edge
loads, variant 1 the walk without any of the three; neither is used); the same values in all. On a one-request
draft step of ``ROWS`` positions it replaces

    draft_shard_topk -> MNNVL FP32 all-reduce of the packed keys -> draft_topk_merge -> (einsum) -> draft_path_walk_tail

by

    select (shard keys, top 16 per chunk, one-shot multicast publish) -> merge (poll, top 16, torch's
    arrangement and bitonic network, codebook operands) -> (the same einsum) -> walk + tail

(``dr_draft_tail.cu``; its header has the transport's ownership and progress argument). Every value the replaced
kernels produce is reproduced exactly -- integer keys and selections, exact FP32 products of widened BF16 values,
the walk's one FP32 add and its comparisons -- so the draft tokens, the next input tokens and the KV-length rewind
are the built-in ones bit for bit. The selector's ``hidden_projection`` (and with ``selector_fork`` its stream join)
is issued before the select launch instead of after the key exchange; it reads only the block rows.

Whether a step takes this path depends only on rank-uniform facts (shapes, dtypes, the request count), so every
rank runs the same sequence of select / merge calls, which the epoch protocol requires.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch

from tensorrt_llm.logger import logger

#: The kernels' variant (``dr_draft_tail.cu``'s header).
VARIANT = 3
#: Draft positions of the step the tail takes: K = 7 drafted rows of one request (``drafter_pdl_chain.HEAD_ROWS``).
ROWS = 7
TOP_K = 16
CHUNK = 4096
RANK_DIM = 256
MAX_WORLD = 8
#: A poll waiting longer than this traps (a peer that never published).
POLL_TIMEOUT_S = 600
LIBRARY_NAME = "musespark_open_dr_tail"
SOURCE = Path(__file__).resolve().parent / "dr_draft_tail.cu"
NVCC_FLAGS = ("-O3", "-lineinfo")

_LOCK = threading.Lock()
_LOADED: Optional[str] = None


def arch_target() -> str:
    capability = torch.cuda.get_device_capability()
    if capability == (10, 3):
        return "103a"
    if capability == (10, 7):
        return "107a"
    raise RuntimeError(
        f"the dr draft tail is built for sm_103a / sm_107a, not sm_{capability[0]}{capability[1]}"
    )


def load_library() -> str:
    """Build (if needed) and load the library once per process; its path."""
    global _LOADED
    with _LOCK:
        if _LOADED is None:
            from torch.utils.cpp_extension import load

            target = arch_target()
            path = load(
                name=f"{LIBRARY_NAME}_sm{target}",
                sources=[str(SOURCE)],
                extra_cuda_cflags=list(NVCC_FLAGS)
                + [f"-gencode=arch=compute_{target},code=sm_{target}"],
                is_python_module=False,
                verbose=False,
            )
            _LOADED = str(path) if path else LIBRARY_NAME
        return _LOADED


def chunks_of(shard: int) -> int:
    return (shard + CHUNK - 1) // CHUNK


def area_bytes(rows: int, world: int, shard: int) -> int:
    """Bytes of one key area: one 64-bit LL element per key of every (row, rank)."""
    return rows * world * chunks_of(shard) * TOP_K * 8


def unsupported_reason(world: int, shard: int, selector) -> Optional[str]:
    """Why the tail cannot run at this geometry, or None. Rank-uniform facts only."""
    if not 1 <= world <= MAX_WORLD:
        return f"the tail serves up to {MAX_WORLD} ranks, not {world}"
    if not (TOP_K <= shard <= 65536 and shard % 8 == 0):
        return f"a shard of {shard} columns (16..65536, a multiple of 8)"
    if getattr(selector, "top_k", None) != TOP_K:
        return f"selector_top_k {getattr(selector, 'top_k', None)} is not {TOP_K}"
    for name in ("predecessor_codebook", "successor_codebook"):
        cb = getattr(selector, name, None)
        if (
            cb is None
            or cb.dtype != torch.bfloat16
            or cb.dim() != 2
            or cb.shape[1] != RANK_DIM
            or cb.stride(1) != 1
            or cb.shape[0] != world * shard
        ):
            return f"selector {name} is not a [{world * shard}, {RANK_DIM}] BF16 tensor"
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        return "the tail runs on SM103 or SM107 only"
    return None


class DrDraftTail:
    """This rank's draft-head tail: the key areas, the epoch counters, the three launches."""

    def __init__(
        self,
        mapping,
        shard: int,
        device: torch.device,
        rows: int = ROWS,
        *,
        emulated_world: Optional[int] = None,
    ) -> None:
        """Collective over the TP group of ``mapping``.

        ``emulated_world`` (tests only, no ``mapping``): one process stands in for that many ranks on a
        plain device buffer that is both the multicast and the receive area; ``select(..., rank=r)``
        publishes rank ``r``'s keys with that rank's own epoch counters.
        """
        self.shard = int(shard)
        self.rows = int(rows)
        self.chunks = chunks_of(self.shard)
        self.device = device
        self.pdl = True
        self.timeout_ns = int(POLL_TIMEOUT_S * 1e9)
        self.variant = VARIANT
        self.calls = 0
        if emulated_world is not None:
            self.world = int(emulated_world)
            self.rank = 0
            load_library()
            size = 2 * area_bytes(self.rows, self.world, self.shard)
            self._area = torch.zeros(size // 4, dtype=torch.int32, device=device)
            self._mc_ptr = self._uc_ptr = int(self._area.data_ptr())
            self.select_counters = [
                torch.zeros(self.rows * self.chunks, dtype=torch.int32, device=device)
                for _ in range(self.world)
            ]
            self.merge_counters = torch.zeros(self.rows, dtype=torch.int32, device=device)
            self._dry()
            torch.cuda.synchronize(device)
            return
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )

        self.world = int(mapping.tp_size)
        self.rank = int(mapping.tp_rank)
        self._comm = _get_mnnvl_workspace_comm(mapping)
        error: Optional[BaseException] = None
        try:
            load_library()
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] dr draft tail: the library failed to load on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error
        size = 2 * area_bytes(self.rows, self.world, self.shard)
        try:
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, size, mapping, mapping.is_multi_node()
            )
            self._uc = self._handle.get_uc_buffer(self.rank, (size // 4,), torch.int32, 0)
            self._mc = self._handle.get_mc_buffer((size // 4,), torch.int32, 0)
            self._mc_ptr = int(self._mc.data_ptr())
            self._uc_ptr = int(self._uc.data_ptr())
            with torch.cuda.device(device):
                self._uc.zero_()
                self.select_counters = [
                    torch.zeros(self.rows * self.chunks, dtype=torch.int32, device=device)
                ]
                self.merge_counters = torch.zeros(self.rows, dtype=torch.int32, device=device)
                self._dry()
                torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "[musespark_open] dr draft tail: the workspace failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def _dry(self) -> None:
        dummy = torch.empty(1, self.shard, dtype=torch.bfloat16, device=self.device)
        torch.ops.musespark_open_dr_tail.select(
            dummy, self.merge_counters, 0, 0, 0, 1, self.pdl, True
        )

    @property
    def area_bytes(self) -> int:
        return area_bytes(self.rows, self.world, self.shard)

    def takes(
        self, local_logits: torch.Tensor, block_hidden_states: torch.Tensor, selector
    ) -> bool:
        """Whether this draft step runs on the tail. Rank-uniform facts only."""
        return (
            local_logits.dim() == 3
            and tuple(local_logits.shape) == (1, self.rows, self.shard)
            and local_logits.dtype == torch.bfloat16
            and block_hidden_states.dim() == 3
            and block_hidden_states.shape[:2] == (1, self.rows)
            and getattr(selector, "top_k", None) == TOP_K
            and selector.predecessor_codebook.shape == (self.world * self.shard, RANK_DIM)
        )

    def select(self, local: torch.Tensor, rank: Optional[int] = None) -> None:
        """Publish this rank's top-16 keys of every chunk of ``local`` ``[rows, shard]``."""
        if local.stride(-1) != 1 or local.stride(0) % 8 or local.data_ptr() % 16:
            local = local.contiguous()
        r = self.rank if rank is None else int(rank)
        torch.ops.musespark_open_dr_tail.select(
            local,
            self.select_counters[r if len(self.select_counters) > 1 else 0],
            self._mc_ptr,
            self.area_bytes,
            r,
            self.world,
            self.pdl,
            False,
        )

    def merge(self, gate: torch.Tensor, anchor: torch.Tensor, selector):
        """The merge of every rank's keys: ``(ids, unary, predecessors, successors)`` as ``draft_topk_merge``.

        Launched right after this rank's ``select`` on the same stream. The kernel reads ``gate``, ``anchor``
        and the codebooks before its grid-dependency wait, so their producers must precede that ``select``
        launch on the stream (``draft_tokens`` issues the projection first); a producer launched between
        ``select`` and ``merge`` may still be writing when the merge reads.
        """
        rows = self.rows
        gate2 = gate.reshape(rows, gate.shape[-1])
        if gate2.stride(-1) != 1:
            gate2 = gate2.contiguous()
        device = gate.device
        ids = torch.empty((1, rows, TOP_K), dtype=torch.int64, device=device)
        unary = torch.empty((1, rows, TOP_K), dtype=torch.float32, device=device)
        preds = torch.empty((1, rows, TOP_K, RANK_DIM), dtype=torch.float32, device=device)
        succs = torch.empty_like(preds)
        torch.ops.musespark_open_dr_tail.merge(
            self.merge_counters,
            self._uc_ptr,
            self.area_bytes,
            self.world,
            self.shard,
            gate2,
            anchor.reshape(-1),
            selector.predecessor_codebook,
            selector.successor_codebook,
            ids,
            unary,
            preds,
            succs,
            self.pdl,
            self.timeout_ns,
            self.variant,
        )
        self.calls += 1
        return ids, unary, preds, succs

    def walk_tail(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        ids: torch.Tensor,
        accepted: torch.Tensor,
        num_accepted: torch.Tensor,
        batch_indices: torch.Tensor,
        kv_lens: Optional[torch.Tensor],
        rewind: Optional[torch.Tensor],
    ):
        """``draft_path_walk_tail``: ``(tokens [B, K] int32, next_tokens [B, K + 1] int32)``; rewinds in place.

        Variants 2 and 3 read ``accepted`` / ``num_accepted`` / ``batch_indices`` / ``kv_lens`` / ``rewind`` before its
        grid-dependency wait: their producers must precede the launch right before this one (in ``draft_tokens``
        they precede the select launch; the einsum is the walk's predecessor).
        """
        requests, positions, _ = unary.shape
        tokens = torch.empty((requests, positions), dtype=torch.int32, device=unary.device)
        next_tokens = torch.empty((requests, positions + 1), dtype=torch.int32, device=unary.device)
        rewinds = kv_lens is not None and rewind is not None
        if not edges.is_contiguous():
            edges = edges.contiguous()
        torch.ops.musespark_open_dr_tail.walk_tail(
            edges,
            unary,
            ids,
            tokens,
            next_tokens,
            accepted,
            num_accepted,
            batch_indices,
            kv_lens if rewinds else tokens,
            rewind if rewinds else tokens,
            rewinds,
            self.pdl,
            self.variant,
        )
        return tokens, next_tokens

    def draft_tokens(
        self,
        local_logits: torch.Tensor,
        block_hidden_states: torch.Tensor,
        anchor_tokens: torch.Tensor,
        selector,
        walk,
        tail_walk,
    ):
        """``draft_head.selector_draft_tokens`` on the tail (the caller checked ``takes``)."""
        # The built-in selector's projection call, operand for operand (its fork joins here).
        gate = selector.hidden_projection(
            block_hidden_states.to(selector.hidden_projection.weight.dtype)
        )
        self.select(local_logits.reshape(self.rows, self.shard))
        ids, unary, predecessors, successors = self.merge(gate, anchor_tokens, selector)
        # dflash2_score_edges' einsum on tensors of its shapes and strides: the same kernel.
        edges = torch.einsum("blpr,blcr->blpc", predecessors, successors)
        if tail_walk is not None:
            return self.walk_tail(edges, unary, ids, *tail_walk)
        return walk(edges, unary, ids)


def installed(llm) -> bool:
    worker = getattr(llm, "spec_worker", None)
    return isinstance(getattr(worker, "__dict__", {}).get("_dr_draft_tail"), DrDraftTail)


def install(llm, mapping) -> None:
    """Build the tail on ``llm``'s draft-head worker (``drafter_folds``). Collective.

    Needs the ``draft_head`` fast path (the worker's sharded selector) and a gathered column-parallel
    draft LM head; the refusal is agreed over the TP group before the collective construction.
    Idempotent.
    """
    if installed(llm):
        return
    from ... import dispatch as _tp_agreement
    from ..draft_head import head as _draft_head

    worker = getattr(llm, "spec_worker", None)
    draft_model = getattr(llm, "draft_model", None)
    reason = None
    if not isinstance(worker, _draft_head.ShardedDraftHeadDFlashWorker):
        reason = "the draft_head fast path is not installed"
    elif worker.__dict__.get("_draft_head_allreduce") is None:
        reason = "the draft head has no MNNVL key exchange"
    else:
        lm_head = getattr(draft_model, "lm_head", None)
        selector = getattr(draft_model, "candidate_selector", None)
        if not _draft_head.gathers_column_shards(lm_head):
            reason = "the draft LM head is not a gathered column-parallel head"
        else:
            reason = unsupported_reason(int(mapping.tp_size), int(lm_head.out_features), selector)
    reason = _tp_agreement.agreed_refusal(
        mapping, None if reason is None else f"[musespark_open] dr draft tail: {reason}"
    )
    if reason is not None:
        raise ValueError(reason)
    device = draft_model.candidate_selector.predecessor_codebook.device
    tail = DrDraftTail(mapping, int(draft_model.lm_head.out_features), device)
    worker.__dict__["_dr_draft_tail"] = tail
    logger.info(
        "[musespark_open] dr draft tail: select + one-shot key publish, merge, walk + tail as CUDA "
        f"launches, variant {tail.variant} (shard {tail.shard}, {tail.world} ranks, {tail.rows} positions)"
    )
