# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 MLA drafter's context in the KV cache manager's draft pages.

With a unified draft cache (KV cache manager V2 built with standalone draft layers; always under
disaggregated serving) the target manager holds, per request, the
drafter's context latents of its last sliding window: one ``kv_lora_rank + qk_rope_head_dim``
latent per position and drafter layer, at the absolute position, in pages of the target's
``tokens_per_block`` (``get_draft_buffers`` views ``[pages, 1, 1, tokens_per_block, latent]``).
Those pages are a carrier and a reuse store; the drafter itself decodes from its private ring
(:mod:`.dflash2_ring`):

* a context chunk ``[c0, c1)`` writes the latents of its last ``window`` positions, so the
  window ending at every chunk end is in the pages: the earlier positions of a short chunk's
  window were written by the chunk (or the reused prefix) before it. K3.1's recurrent-state
  snapshots split context chunks, and a recurrent reuse match ends at a snapshot, so a reused
  prefix always brings a full window; the manager's match requires those pages;
* the draft history ``(valid_length, position)`` is published at every chunk end, once the
  step's forward is done (``DraftHistoryUpdate``); a disaggregated generation server derives it
  from the prompt whose pages it receives;
* a request that arrives with a history (a generation request of a disaggregated server, a
  context request whose prefix was reused) has its ring seeded once from the pages, rows
  ``[position - valid_length, position)`` at ``p mod ring_rows``, and then decodes as one
  whose ring was filled by its own prefill.

Everything here is host bookkeeping or plain index copies (bitwise), run eagerly on the
execution stream outside any CUDA graph.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_prefill_share import (
    disagg_role,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.unified_draft_cache import (
    DraftHistoryUpdate,
    StandaloneDraftHistory,
)
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger


def dflash2_attention_mode(pretrained_config) -> Optional[str]:
    """``dflash_config.attention_mode`` of a drafter config (None when absent)."""
    dflash_config = getattr(pretrained_config, "dflash_config", None) or {}
    return dflash_config.get("attention_mode")


def dflash2_sliding_window(pretrained_config) -> Optional[int]:
    """The DFlash2 MLA drafter's attention window in positions (the query's own included).

    The one reading of the window: the drafter decodes with it and the KV cache manager keeps it
    of the drafter's context. ``pretrained_config`` is the drafter's parsed config (the
    ``pretrained_config`` of its ``ModelConfig``, :func:`dflash2_drafter_config`). Every layer
    must be a ``sliding_attention`` layer of one ``sliding_window``: the block decode builds one
    set of window index lists for all layers.

    Returns:
        The window, or None for a drafter whose ``dflash_config.attention_mode`` is not ``mla``.

    Raises:
        ValueError: for an MLA drafter without one sliding window over all its layers, or
            without causal block attention.
    """
    if dflash2_attention_mode(pretrained_config) != "mla":
        return None
    window = getattr(pretrained_config, "sliding_window", None)
    layer_types = getattr(pretrained_config, "layer_types", None) or []
    if not window or any(t != "sliding_attention" for t in layer_types):
        raise ValueError(
            "The DFlash2 MLA drafter needs every layer to be a sliding_attention layer of one "
            f"sliding_window; got sliding_window={window!r}, layer_types={layer_types!r}."
        )
    if not getattr(pretrained_config, "is_causal", True):
        raise ValueError("The DFlash2 MLA drafter's block attention is causal (is_causal: true).")
    return int(window)


@functools.lru_cache(maxsize=8)
def dflash2_drafter_config(checkpoint: str):
    """The drafter checkpoint's config, parsed as the drafter's ``ModelConfig`` parses it."""
    return ModelConfig.from_pretrained(checkpoint, trust_remote_code=True).pretrained_config


def dflash2_draft_window(spec_config) -> Optional[int]:
    """The sliding window of the DFlash2 MLA drafter ``spec_config`` names, or None.

    :func:`dflash2_sliding_window` of the drafter's config: the window the KV cache manager
    keeps of the drafter's context. None for any other speculative config.
    """
    if spec_config is None or not spec_config.spec_dec_mode.is_dflash():
        return None
    checkpoint = getattr(spec_config, "speculative_model", None)
    if checkpoint is None:
        return None
    return dflash2_sliding_window(dflash2_drafter_config(str(checkpoint)))


def dflash2_context_kv_only(spec_config) -> bool:
    """Whether this server builds a context-only DFlash drafter (``context_kv_only``).

    A disaggregated context server (:func:`disagg_role` ``context``) never drafts: its drafter
    only builds the prompt's context latents, which the draft pages carry to the generation server.
    """
    return (
        spec_config is not None
        and spec_config.spec_dec_mode.is_dflash()
        and disagg_role() == "context"
    )


def dflash2_draft_pages_manager(kv_cache_manager) -> Optional[object]:
    """``kv_cache_manager`` when it holds the drafter's context pages (standalone draft layers)."""
    if getattr(kv_cache_manager, "has_standalone_draft_cache", False) is True:
        return kv_cache_manager
    return None


def dflash2_draft_page_pools(manager, num_layers: int, latent_dim: int) -> list[torch.Tensor]:
    """Per-layer ``[pages, tokens_per_block, latent]`` views of the manager's draft pages."""
    pools = []
    for layer_idx in range(num_layers):
        view = manager.get_draft_buffers(layer_idx, kv_layout="HND")
        if view.dim() != 5 or tuple(view.shape[1:3]) != (1, 1) or view.shape[-1] != latent_dim:
            raise ValueError(
                "The DFlash2 drafter's draft pages must hold one latent of "
                f"{latent_dim} values per token and layer, got a {tuple(view.shape)} view."
            )
        pools.append(view[:, 0, 0])
    return pools


def dflash2_chunk_tail_start(chunk_start: int, chunk_len: int, window: int) -> int:
    """First position of the chunk ``[chunk_start, chunk_start + chunk_len)`` whose latent the
    draft pages keep: the chunk's last ``window`` positions."""
    return chunk_start + max(0, int(chunk_len) - int(window))


def dflash2_history_after_chunk(
    prior: Optional[StandaloneDraftHistory], chunk_start: int, chunk_len: int, window: int
) -> tuple[int, int]:
    """``(valid_length, position)`` of the draft history once a chunk's tail is stored.

    The window before the chunk is valid for ``prior.valid_length`` positions when ``prior``
    ends where the chunk starts, and for none otherwise.
    """
    before = prior.valid_length if prior is not None and prior.position == chunk_start else 0
    end = chunk_start + int(chunk_len)
    return min(int(window), before + int(chunk_len)), end


class DraftPageWrites(NamedTuple):
    """Where a step's drafter latents go in the draft pages, one entry per stored position."""

    #: ``[N]`` int64: the latent's row in the caller's per-step latent (or capture) rows.
    rows: torch.Tensor
    #: ``[N]`` int64: absolute positions (the RoPE positions of the latents).
    positions: torch.Tensor
    #: ``[N]`` int64: draft page of each position.
    pages: torch.Tensor
    #: ``[N]`` int64: token offset of each position in its page.
    offsets: torch.Tensor
    #: ``[N]`` int64 on the host: ``positions`` before the upload.
    host_positions: Optional[np.ndarray] = None


class DraftPagePlanner:
    """Collects the positions a step stores, request by request, then uploads them at once.

    ``known`` maps a request to the ``(valid_length, position)`` its previous chunk left, which the
    caller keeps across steps: the manager publishes a step's histories only after the step's
    sampling, which an overlapping schedule runs after the next step's forward.
    """

    def __init__(self, manager, window: int, known: Optional[dict] = None) -> None:
        self.manager = manager
        self.window = int(window)
        self.known = known if known is not None else {}
        self.page_size = int(manager.tokens_per_block)
        self._rows: list[np.ndarray] = []
        self._positions: list[np.ndarray] = []
        self._pages: list[np.ndarray] = []
        self.request_ids: list[int] = []
        self.histories: list[tuple[int, int]] = []

    def add_chunk(self, request_id: int, chunk_start: int, chunk_len: int, row_base: int) -> int:
        """Plan a context chunk's tail; returns the first position stored.

        Args:
            request_id: the request.
            chunk_start: absolute position of the chunk's first token.
            chunk_len: tokens in the chunk.
            row_base: the caller's row of the chunk's first token; the stored positions take
                rows ``row_base + (p - chunk_start)``.
        """
        prior = self._prior(request_id, chunk_start)
        tail = dflash2_chunk_tail_start(chunk_start, chunk_len, self.window)
        end = chunk_start + int(chunk_len)
        if tail < end:
            pages = self._draft_pages(request_id, tail)
            positions = np.arange(tail, end, dtype=np.int64)
            page_idx = positions // self.page_size
            if page_idx[-1] >= pages.size or np.any(pages[page_idx] < 0):
                raise ValueError(
                    f"Request {request_id}: the draft pages do not cover positions "
                    f"[{tail}, {end}) of its context chunk."
                )
            self._rows.append(
                np.arange(row_base + tail - chunk_start, row_base + end - chunk_start)
            )
            self._positions.append(positions)
            self._pages.append(pages[page_idx])
        history = dflash2_history_after_chunk(prior, chunk_start, chunk_len, self.window)
        self.known[int(request_id)] = history
        self.request_ids.append(int(request_id))
        self.histories.append(history)
        return tail

    def _prior(self, request_id: int, chunk_start: int) -> Optional[StandaloneDraftHistory]:
        """The history the chunk starts from: the caller's record of the request's previous chunk,
        else the manager's (a fresh request's, or the one a reused prefix set)."""
        known = self.known.get(int(request_id))
        if known is not None and known[1] == chunk_start:
            return StandaloneDraftHistory(*known)
        return self.manager.get_draft_history(request_id)

    def _draft_pages(self, request_id: int, first_stored: int) -> np.ndarray:
        """The request's draft page indices, validated from the first stored position on (pages
        before it may already have left the manager's window)."""
        history = StandaloneDraftHistory(0, int(first_stored))
        (indices,) = self.manager.get_draft_block_indices([request_id], [history])
        return np.asarray(indices, dtype=np.int64)

    def writes(self, device: torch.device) -> Optional[DraftPageWrites]:
        """The planned writes on ``device`` (one upload), or None when nothing is stored."""
        if not self._rows:
            return None
        positions = np.concatenate(self._positions)
        host = np.stack(
            (
                np.concatenate(self._rows),
                positions,
                np.concatenate(self._pages),
                positions % self.page_size,
            )
        )
        stacked = maybe_pin_memory(torch.from_numpy(host))
        rows, positions_t, pages, offsets = stacked.to(device, non_blocking=True).unbind(0)
        return DraftPageWrites(rows, positions_t, pages, offsets, positions)

    def history_update(self) -> Optional[DraftHistoryUpdate]:
        """The draft histories after this step's chunks, for their manager to publish."""
        if not self.request_ids:
            return None
        logger.info_once(
            f"DFlash2: context chunks store the drafter latents of their last {self.window} "
            "positions in the draft pages",
            key="k31_dflash2_draft_pages_store",
        )
        values = torch.tensor(self.histories, dtype=torch.long)
        return DraftHistoryUpdate.capture(self.manager, tuple(self.request_ids), values)


def dflash2_store_draft_pages(
    pools: Sequence[torch.Tensor], latents: torch.Tensor, pages: torch.Tensor, offsets: torch.Tensor
) -> None:
    """Write ``latents`` ``[N, L, 1, latent]`` (or ``[N, L, latent]``) into the draft pages, in the pool dtype.

    Raises:
        RuntimeError: when called during CUDA graph capture. The stores run in the speculative
            worker after the model body, which a breakable prefill CUDA graph captures alone.
    """
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DFlash2 draft-page stores run eagerly: they are host-planned per step and must stay "
            "outside every captured CUDA graph segment."
        )
    if latents.dim() == 4:
        latents = latents[:, :, 0]
    for layer_idx, pool in enumerate(pools):
        new = latents[:, layer_idx].to(pool.dtype)
        pool[pages, offsets] = new


def dflash2_rope_rows(
    draft_model, positions: np.ndarray, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The drafter's YaRN RoPE rows at ``positions``, bitwise its device table's, and their indices.

    ``build_mla_yarn_freqs_cis`` (``ops/spec/mla_norm_rope``) computes the table on the host and copies it to
    the device, so the rows gathered from the host table (built once per drafter,
    ``draft_model._mla_rope_kwargs()``) equal the device table's. A context-only drafter, which rotates only the
    positions it stores, then holds no device table (``max_positions`` x ``rope_dim / 2`` complex64, 256 MiB at
    1M positions).

    Returns:
        ``([N, rope_dim / 2]`` complex64 rows on ``device``, ``[N]`` int64 ``0 .. N-1`` on ``device``):
        the table and positions arguments of the drafter's norm / RoPE kernels.
    """
    from .mla_norm_rope import build_mla_yarn_freqs_cis

    kwargs = draft_model._mla_rope_kwargs()
    cached = draft_model.__dict__.get("_k31_host_mla_freqs")
    if cached is None or cached[0] != kwargs:
        cached = (kwargs, build_mla_yarn_freqs_cis(device="cpu", **kwargs))
        draft_model.__dict__["_k31_host_mla_freqs"] = cached
    rows = cached[1].index_select(
        0, torch.from_numpy(np.ascontiguousarray(positions, dtype=np.int64))
    )
    rows = maybe_pin_memory(rows).to(device, non_blocking=True)
    return rows, torch.arange(rows.shape[0], dtype=torch.int64, device=device)


class RingSeed(NamedTuple):
    """One request whose private ring is seeded from its draft pages."""

    request_id: int
    slot: int
    valid_length: int
    position: int


def dflash2_seed_ring_from_pages(
    manager,
    pools: Sequence[torch.Tensor],
    ring: torch.Tensor,
    ring_table: torch.Tensor,
    ring_rows: Optional[int],
    ring_page_size: int,
    seeds: Sequence[RingSeed],
) -> None:
    """Copy each seed's window ``[position - valid_length, position)`` from the draft pages into
    its slot of the private ring (row ``p mod ring_rows``; row ``p`` without a ring).

    Args:
        manager: the KV cache manager holding the draft pages.
        pools: per-layer ``[pages, tokens_per_block, latent]`` draft page views.
        ring: ``[L, ring_pages, 1, 1, ring_page_size, latent]`` private arena, the pools' dtype.
        ring_table: ``[slots, pages_per_slot]`` int32 page table of the arena.
        ring_rows: rows per slot of the ring, None for a dense arena.
        ring_page_size: tokens per arena page.
        seeds: the requests to seed.
    """
    if not seeds:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DFlash2 ring seeding runs eagerly, outside CUDA graph capture.")
    if ring.dtype != pools[0].dtype:
        raise ValueError(
            f"The drafter's ring ({ring.dtype}) and draft pages ({pools[0].dtype}) must share a "
            "dtype: both hold the drafter's context latents in the drafter's dtype."
        )
    page_size = int(manager.tokens_per_block)
    histories = [StandaloneDraftHistory(s.valid_length, s.position) for s in seeds]
    indices = manager.get_draft_block_indices([s.request_id for s in seeds], histories)
    src_pages, positions, slots = [], [], []
    for seed, row in zip(seeds, indices):
        p = np.arange(seed.position - seed.valid_length, seed.position, dtype=np.int64)
        src_pages.append(np.asarray(row, dtype=np.int64)[p // page_size])
        positions.append(p)
        slots.append(np.full(p.size, seed.slot, dtype=np.int64))
    positions_np = np.concatenate(positions)
    ring_pos = positions_np % ring_rows if ring_rows else positions_np
    host = np.stack(
        (
            np.concatenate(src_pages),
            positions_np % page_size,
            np.concatenate(slots),
            ring_pos // ring_page_size,
            ring_pos % ring_page_size,
        )
    )
    stacked = maybe_pin_memory(torch.from_numpy(host))
    src_page, src_off, slot, ring_col, ring_off = stacked.to(ring.device, non_blocking=True).unbind(
        0
    )
    ring_page = ring_table[slot, ring_col].long()
    for layer_idx, pool in enumerate(pools):
        dst = ring[layer_idx]
        dst[ring_page, 0, 0, ring_off] = pool[src_page, src_off]
    logger.debug(
        f"DFlash2: seeded {len(seeds)} drafter ring slot(s) from the draft pages "
        f"({positions_np.size} positions)"
    )


def _seed_count_logged(arrived: int) -> bool:
    """Whether the seeding counts are logged when this many requests arrived with drafter context:
    at the first, every 16th below 1024, then every 1000th."""
    return arrived == 1 or (arrived < 1024 and arrived % 16 == 0) or arrived % 1000 == 0


class DFlash2DraftPagesWorkerMixin:
    """The K3.1 DFlash worker's side of the draft pages, ahead of the engine ``DFlashWorker`` in the MRO.

    It implements the hooks a unified draft cache uses (``get_draft_kv_cache_manager``, and the
    runner's ``prepare_managed_draft_cache`` and ``snapshot_managed_draft_history``) with the
    drafter's context still in the worker's private ring: the pages are written per context chunk
    (each chunk's last window), their history published at chunk ends only, and a request's ring
    seeded from them once when it arrives with them. Decode steps do no draft-page work. Before the
    forward the hook also maps the step's requests to ring slots (``_map_step_slots``) on the step
    metadata the worker's ``_forward_impl`` marks (``_dflash2_worker``). The worker's prefill store
    calls :meth:`_plan_page_tail` / :meth:`_seed_ring_at` (or :meth:`_store_context_window` for a
    context-only drafter); a worker that defines its own ``_assign_slot`` calls
    :meth:`_queue_draft_page_seed` from it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Generation requests given a ring slot this step, seeded before the forward.
        self._draft_page_seeds: list[int] = []
        # The draft histories after this step's context chunks, for the executor to publish.
        self._draft_page_update: Optional[DraftHistoryUpdate] = None
        # (manager, per-layer views) of the draft pages, rebound when the manager is replaced.
        self._draft_page_views = None
        # Request -> (valid_length, position) after its last stored context chunk.
        self._draft_page_histories: dict[int, tuple[int, int]] = {}
        # New generation slots queued for seeding; requests among them that arrived with drafter
        # context in their draft pages; rings seeded from it (see _seed_count_logged); slots given
        # to requests that had already generated past their prompt end.
        self._draft_page_queued = 0
        self._draft_page_arrived = 0
        self._draft_page_seeded = 0
        self._draft_page_late = 0

    def _draft_page_planner(self, manager, window: int) -> DraftPagePlanner:
        """A planner for this step's context chunks, sharing the worker's chunk histories (pruned of
        requests the manager no longer holds)."""
        known = self._draft_page_histories
        if len(known) > 4 * max(1, len(manager.kv_cache_map)):
            for request_id in [r for r in known if r not in manager.kv_cache_map]:
                del known[request_id]
        return DraftPagePlanner(manager, window, known)

    def get_draft_kv_cache_manager(self, resource_manager):
        """No draft KV cache manager for the drafter's decode: its context is the private ring; the
        draft pages are only a carrier, which the hooks reach through the target manager."""
        from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
        from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

        manager = SpecWorkerBase.get_draft_kv_cache_manager(self, resource_manager)
        # A target manager that holds the draft layers names itself as the draft view.
        if resource_manager is not None and manager is resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        ):
            return None
        return manager

    @staticmethod
    def _context_only_server(draft_model, attn_metadata) -> bool:
        """Whether this server only runs the context phase of disaggregated requests and stores its
        drafter context in draft pages: a context-only drafter (:func:`dflash2_context_kv_only`),
        or a full drafter on a KV cache manager that serves the context phase only
        (``kda_context_only``) and holds draft pages. It never drafts, so a prefill stores only each
        chunk's last window, into the draft pages. Without draft pages a full drafter fills its own
        context, as on any other server."""
        if getattr(draft_model, "context_kv_only", False):
            return True
        manager = getattr(attn_metadata, "kv_cache_manager", None)
        return (
            getattr(manager, "kda_context_only", False) is True
            and dflash2_draft_pages_manager(manager) is not None
        )

    @staticmethod
    def _chunk_start(i, offset, slen, host_pos, cached_tokens, position_ids) -> int:
        """Absolute position of context request ``i``'s chunk (its first token)."""
        if slen == 0:
            return 0
        if host_pos is not None:
            return int(host_pos[offset])
        if cached_tokens is not None and i < len(cached_tokens):
            return int(cached_tokens[i])
        return int(position_ids[offset].item())

    def _draft_page_pools(self, draft_model, manager) -> list:
        """Per-layer views of ``manager``'s draft pages (rebound when the manager is replaced)."""
        if self._draft_page_views is None or self._draft_page_views[0] is not manager:
            pools = dflash2_draft_page_pools(
                manager, draft_model._num_attn_layers, draft_model._head_dim
            )
            self._draft_page_views = (manager, pools)
        return self._draft_page_views[1]

    def _plan_page_tail(
        self, planner, page_latents, draft_model, req_id, first_pos, chunk_proj, ring_k
    ) -> None:
        """Queue the latents of a context chunk's last window positions for the draft pages.

        ``ring_k`` are the chunk's latents at absolute positions when the ring store computed them;
        otherwise the tail's latents are computed here from the chunk's projected rows.
        """
        if planner is None:
            return
        cache = planner.manager.kv_cache_map.get(req_id)
        if cache is None or not cache.is_active:
            return
        slen = chunk_proj.shape[0]
        keep = dflash2_chunk_tail_start(first_pos, slen, planner.window) - first_pos
        stored = sum(latents.shape[0] for latents in page_latents)
        planner.add_chunk(req_id, first_pos, slen, row_base=stored - keep)
        if keep >= slen:
            return
        if ring_k is not None:
            page_latents.append(ring_k[keep:])
            return
        positions = torch.arange(
            first_pos + keep, first_pos + slen, dtype=torch.long, device=chunk_proj.device
        )
        latents, _ = draft_model.precompute_context_kv(chunk_proj[keep:], positions)
        page_latents.append(latents)

    def _seed_ring_at(self, draft_model, manager, req_id: int, slot: int, position: int) -> bool:
        """Seed ``slot``'s ring with the window the draft pages hold before ``position``.

        True when the request's draft history ends at ``position`` with a full window
        (``min(position, window)`` positions), which a reused prefix or a transferred prompt
        brings; the ring then holds rows ``[position - valid_length, position)``.
        """
        history = manager.get_draft_history(req_id)
        if history is None or history.position != position or history.valid_length <= 0:
            return False
        if history.valid_length != min(position, draft_model._window):
            logger.warning_once(
                f"DFlash2: request {req_id}'s draft pages hold {history.valid_length} of the "
                f"{min(position, draft_model._window)} positions before {position}; its drafter "
                "starts without them.",
                key="k31_dflash2_partial_draft_window",
            )
            return False
        if (
            not self._ctx_paged
            or self._ctx_block_tables is not None
            or not isinstance(self._ctx_kv_buf, torch.Tensor)
        ):
            logger.warning_once(
                "DFlash2: the drafter's context is not a private paged ring here; transferred and "
                "reused drafter context is not seeded.",
                key="k31_dflash2_no_ring_to_seed",
            )
            return False
        dflash2_seed_ring_from_pages(
            manager,
            self._draft_page_pools(draft_model, manager),
            self._ctx_kv_buf,
            self._ctx_page_table,
            self._ctx_ring_rows,
            self._ctx_page_size,
            [RingSeed(req_id, slot, history.valid_length, position)],
        )
        return True

    def _store_context_window(
        self, draft_model, spec_metadata, attn_metadata, captured_hs, manager, position_ids
    ) -> None:
        """A context-only drafter's prefill store: the latents of each chunk's last window
        positions, projected from their captured rows alone, into the draft pages."""
        if manager is None:
            raise RuntimeError(
                "A context-only DFlash2 drafter (a disaggregated context server) stores the "
                "prompt's context in the KV cache manager's draft pages, but the manager has none."
            )
        planner = self._draft_page_planner(manager, draft_model._window)
        host_pos = spec_metadata.host_position_ids
        kv_params = getattr(attn_metadata, "kv_cache_params", None)
        cached_tokens = getattr(kv_params, "num_cached_tokens_per_seq", None)
        offset = 0
        for i in range(attn_metadata.num_contexts):
            req_id = spec_metadata.request_ids[i]
            slen = int(attn_metadata._seq_lens[i])
            cache = manager.kv_cache_map.get(req_id)
            if slen > 0 and cache is not None and cache.is_active:
                first_pos = self._chunk_start(
                    i, offset, slen, host_pos, cached_tokens, position_ids
                )
                planner.add_chunk(req_id, first_pos, slen, row_base=offset)
            offset += slen
        writes = planner.writes(captured_hs.device)
        if writes is not None:
            projected = draft_model.project_target_hidden(captured_hs.index_select(0, writes.rows))
            freqs, indices = dflash2_rope_rows(
                draft_model, writes.host_positions, captured_hs.device
            )
            latents, _ = draft_model.precompute_context_kv(projected, indices, freqs=freqs)
            dflash2_store_draft_pages(
                self._draft_page_pools(draft_model, manager), latents, writes.pages, writes.offsets
            )
        self._draft_page_update = planner.history_update()

    def _assign_slot(self, req_id: int, reset: bool = False, updates: Optional[dict] = None):
        """The engine's slot assignment; a generation request's new slot (assigned while the step's
        inputs are prepared) is queued to be seeded from its draft pages before the forward."""
        is_new = req_id not in self._req_to_slot
        slot = super()._assign_slot(req_id, reset=reset, updates=updates)
        self._queue_draft_page_seed(req_id, is_new, slot, updates, reset)
        return slot

    def _map_step_slots(self, spec_metadata) -> None:
        """Map the step's requests to ring slots, on step metadata this worker has drafted from
        (``_dflash2_worker``) once its ring exists and while it keeps no unified draft cache.

        Requests that left the batch release their slots (requests paused in place keep theirs);
        each new generation request gets an empty slot (``_assign_slot``, which queues its
        draft-page seed); CUDA-graph padding and attention-DP dummy rows read the dummy slot. Slots
        are numbered from 0 and keep no base pages.
        """
        if (
            getattr(spec_metadata, "_dflash2_worker", None) is not self
            or not self._ctx_buf_inited
            or self._has_unified_draft_cache()
        ):
            return
        request_ids = spec_metadata.request_ids
        current = set(request_ids) | getattr(spec_metadata, "paused_in_place_ids", frozenset())
        updates = {}
        for req_id in list(self._req_to_slot):
            if req_id not in current:
                slot = self._req_to_slot.pop(req_id)
                updates[slot] = 0
                self._req_ctx_pos.pop(req_id, None)
                self._free_slots.append(slot)
        self._write_ctx_len(updates)
        num_contexts = max(0, len(request_ids) - spec_metadata.num_generations)
        for req_id in request_ids[num_contexts:]:
            if (
                req_id != ATTENTION_DP_DUMMY_REQUEST_ID
                and req_id < self._graph_dummy_id_floor
                and req_id not in self._req_to_slot
                and self._assign_slot(req_id) is None
            ):
                logger.warning(
                    f"DFlash: no free drafter slot for generation request {req_id}; "
                    "it shares the dummy slot and will draft poorly."
                )
        mapping = [
            self._dummy_slot
            if row >= num_contexts and req_id == ATTENTION_DP_DUMMY_REQUEST_ID
            else self._req_to_slot.get(req_id, self._dummy_slot)
            for row, req_id in enumerate(request_ids)
        ]
        self._batch_to_slot_buffer.update(torch.tensor(mapping, dtype=torch.long))

    def prepare_managed_draft_cache(
        self, draft_model, spec_metadata, attn_metadata, resource_manager
    ) -> None:
        """Before the forward (eager, capture or replay): the step's ring slots
        (``_map_step_slots``), the plain capture buffer, the proposal rows of generation requests
        arriving on a disaggregated server (``dflash2_invalidate_arrived_rows``), and the rings of
        generation requests that arrived with drafter context in their draft pages, seeded once."""
        del resource_manager
        from .dflash2_sampling import dflash2_invalidate_arrived_rows

        self._map_step_slots(spec_metadata)

        dflash2_invalidate_arrived_rows(self, spec_metadata, attn_metadata)
        request_ids, self._draft_page_seeds = self._draft_page_seeds, []
        if not request_ids or not self._ctx_buf_inited:
            return
        manager = dflash2_draft_pages_manager(getattr(attn_metadata, "kv_cache_manager", None))
        if manager is None:
            return
        updates = {}
        arrived_before = self._draft_page_arrived
        for req_id in request_ids:
            slot = self._req_to_slot.get(req_id)
            history = manager.get_draft_history(req_id)
            self._draft_page_queued += 1
            if history is None or history.valid_length <= 0:
                continue
            cache = manager.kv_cache_map.get(req_id)
            if cache is None or not cache.is_active or cache.history_length > history.position:
                # A request that already generated past its prompt end lost its slot (it left a
                # step, e.g. suspended for KV space): the window before the prompt end has left the
                # draft pages and is not its drafter's context any more.
                self._draft_page_late += 1
                logger.warning_once(
                    f"DFlash2: request {req_id} got a new drafter slot after it started generating "
                    "(it left a step, e.g. suspended for KV cache space); its drafter restarts "
                    "without context.",
                    key="k31_dflash2_late_slot",
                )
                continue
            self._draft_page_arrived += 1
            if (
                slot is not None
                and self._ctx_len_host[slot] == 0
                and self._seed_ring_at(draft_model, manager, req_id, slot, history.position)
            ):
                updates[slot] = history.position
                self._req_ctx_pos[req_id] = history.position
        self._write_ctx_len(updates)
        self._draft_page_seeded += len(updates)
        if any(
            _seed_count_logged(n) for n in range(arrived_before + 1, self._draft_page_arrived + 1)
        ):
            without = self._draft_page_queued - self._draft_page_arrived - self._draft_page_late
            logger.info(
                f"DFlash2: {self._draft_page_seeded} drafter ring(s) seeded from the draft pages for "
                f"the {self._draft_page_arrived} new generation request(s) that arrived with drafter "
                f"context in them ({without} arrived without; {self._draft_page_late} got a new slot "
                "after they started generating)"
            )

    def snapshot_managed_draft_history(self) -> None:
        """Queue the draft histories after this step's context chunks on their manager, which
        publishes them once the step's forward is done (the engine hook)."""
        update, self._draft_page_update = self._draft_page_update, None
        if update is not None:
            update.manager.queue_draft_history_update(update)

    def _check_no_real_generation(self, spec_metadata, attn_metadata, num_contexts, batch_size):
        """Reject a generation request on a context-only drafter.

        Only dummies reach a context server's generation rows: warmup and attention-DP padding
        requests (tracked by the KV cache manager) and CUDA-graph padding.
        """
        manager = getattr(attn_metadata, "kv_cache_manager", None)
        dummy_ids = getattr(manager, "_draft_dummy_request_ids", ())
        real = [
            request_id
            for request_id in spec_metadata.request_ids[num_contexts:batch_size]
            if request_id != ATTENTION_DP_DUMMY_REQUEST_ID
            and request_id < self._graph_dummy_id_floor
            and request_id not in dummy_ids
        ]
        if real:
            raise RuntimeError(
                f"Generation requests {real[:4]} reached the context-only DFlash2 drafter of a "
                "disaggregated context server (TRTLLM_DISAGG_ROLE=context), which never drafts."
            )

    def _queue_draft_page_seed(self, req_id: int, is_new: bool, slot, updates, reset: bool) -> None:
        """Queue a generation request's new slot (assigned while the step's inputs are prepared,
        i.e. without a prefill ``updates`` batch) to be seeded from its draft pages."""
        if is_new and slot is not None and updates is None and not reset:
            self._draft_page_seeds.append(req_id)


__all__ = [
    "DFlash2DraftPagesWorkerMixin",
    "DraftPagePlanner",
    "DraftPageWrites",
    "RingSeed",
    "dflash2_chunk_tail_start",
    "dflash2_attention_mode",
    "dflash2_context_kv_only",
    "dflash2_draft_page_pools",
    "dflash2_draft_pages_manager",
    "dflash2_draft_window",
    "dflash2_drafter_config",
    "dflash2_history_after_chunk",
    "dflash2_rope_rows",
    "dflash2_seed_ring_from_pages",
    "dflash2_sliding_window",
    "dflash2_store_draft_pages",
]
