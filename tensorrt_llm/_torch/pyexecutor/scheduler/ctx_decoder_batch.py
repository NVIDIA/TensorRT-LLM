# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoder-tail batching across the context passes of an attention-DP context server.

With bounded decoder replay, the layers from the replay boundary run only on the final window of
each prompt that ends in a pass, and every attention-DP rank runs them at the largest rank's rows.
They stream every local expert's weights once per pass whatever the rows, and a pass in which no
rank has final-window rows skips them. While every rank has at least one token budget of context
work queued beyond the pass, a pass whose final windows all fit ``HOLD_MAX_ROWS`` holds them to the
next pass: each finishing chunk is cut back to the start of its final window (in chunk units), or
its request is held whole; the freed budget goes to the next requests, no rank has decoder rows,
and the held windows run with the next pass's own. A request is held at most once, and requests
that need every context row are never held. Every rank plans its batch without allocating, and one
small all-gather per pass gives all ranks the same decision. Scheduling only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from tensorrt_llm.logger import logger

from ..ced_replay import requires_full_decoder_prefill

if TYPE_CHECKING:
    from .scheduler_v2 import KVCacheV2Scheduler

# The largest rank's final-window rows a pass may hold; the next pass runs them with its own.
HOLD_MAX_ROWS = 1536


def window_rows(pos: int, chunk: int, prompt_len: int, window: int) -> int:
    """Decoder rows of the chunk ``[pos, pos + chunk)``: its overlap with the final ``window``
    prompt positions."""
    return max(0, pos + chunk - max(pos, prompt_len - window))


def hold_chunk(pos: int, chunk: int, prompt_len: int, window: int, unit: int) -> int:
    """The chunk that keeps this pass out of the final window: cut back to the window start in
    chunk units, or 0 (hold the request whole) when less than a unit is left before it."""
    if window_rows(pos, chunk, prompt_len, window) == 0:
        return chunk
    room = prompt_len - window - pos
    cut = room // unit * unit if room > 0 else 0
    return min(cut, chunk) if cut >= unit else 0


def holdable(req) -> bool:
    """A request is held at most once, and never one that needs every context row or a remote
    tail."""
    return (
        not req.py_ctx_decoder_held
        and not requires_full_decoder_prefill(req)
        and req.py_csa2_remote_tail_mode is None
    )


@dataclass
class ContextPassPlan:
    """This rank's context phase: its final-window rows, those it may hold, and the context tokens
    queued beyond the pass."""

    rows: int = 0
    holdable_rows: int = 0
    backlog: int = 0


def hold_pass(gathered: Sequence[Sequence[int]], min_backlog: int) -> bool:
    """Every rank's decision from the gathered ``(active, rows, holdable rows, backlog, ...)``."""
    rows = [int(row[1]) for row in gathered]
    return (
        all(int(row[0]) == 1 for row in gathered)
        and 0 < max(rows) <= HOLD_MAX_ROWS
        and all(int(row[2]) == int(row[1]) for row in gathered)
        and all(int(row[3]) >= min_backlog for row in gathered)
    )


class CtxDecoderBatcher:
    """Per-executor state, used on the executor thread around ``schedule_request``. It serves a
    context server: the first pass in which a rank holds a request that is not context-only, or
    cannot hold at all, retires it on every rank."""

    def __init__(self, executor, scheduler: "KVCacheV2Scheduler", window: int):
        self.executor = executor
        self.scheduler = scheduler
        self.window = window
        self.retired = False

    def before_schedule(self, active_requests, inflight_ids) -> None:
        if self.retired:
            return
        # Holds empty a pass only when no chunk keeps decoder rows outside its final window.
        active = self.executor.model_engine.replays_final_windows_only()
        plan = (
            self.scheduler.plan_context_pass(active_requests, inflight_ids, self.window)
            if active
            else ContextPassPlan()
        )
        context_only = all(
            req.is_context_only_request for req in active_requests if not req.is_dummy
        )
        gathered = self.executor.dist.tp_allgather_int64(
            [int(active), plan.rows, plan.holdable_rows, plan.backlog, int(context_only)]
        ).tolist()
        if not all(int(row[4]) for row in gathered):
            self.retired = True
            logger.info("Context decoder-tail batching off: this server runs generation requests")
            return
        if not all(int(row[0]) for row in gathered):
            # The captured prefill graphs stay until the executor releases its resources.
            self.retired = True
            logger.info(
                "Context decoder-tail batching off: a prefill CUDA graph replays every chunk's "
                "decoder rows"
            )
            return
        hold = hold_pass(gathered, self.scheduler.max_num_tokens or 0)
        self.scheduler.set_ctx_decoder_hold(self.window if hold else 0)

    def after_schedule(self) -> None:
        self.scheduler.set_ctx_decoder_hold(0)


def install_ctx_decoder_batch(executor) -> Optional[CtxDecoderBatcher]:
    """Attach a batcher to a disaggregated attention-DP executor whose model replays its decoder
    over final windows and whose scheduler chunks context requests."""
    from ..model_engine import ModelEngine
    from .scheduler_v2 import ContextChunkingPolicy, KVCacheV2Scheduler

    scheduler, engine = executor.scheduler, executor.model_engine
    if not (
        isinstance(engine, ModelEngine)
        and isinstance(scheduler, KVCacheV2Scheduler)
        and executor.enable_attention_dp is True
        and executor.kv_cache_transceiver is not None
    ):
        return None
    model = getattr(engine.model, "model", None)
    window = getattr(model, "decoder_replay_window", 0)
    if not (
        executor.dist.tp_size > 1
        and executor.dist.pp_size == 1
        and scheduler.chunking_enabled
        and scheduler.chunking_policy != ContextChunkingPolicy.FORCE_CHUNK
        and getattr(model, "decoder_replay_split", None) is not None
        and isinstance(window, int)
        and window > 0
        and engine.spec_config is None
    ):
        return None
    logger.info(f"Context decoder-tail batching: holds up to {HOLD_MAX_ROWS} rows, window {window}")
    return CtxDecoderBatcher(executor, scheduler, window)
