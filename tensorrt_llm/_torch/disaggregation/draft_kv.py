# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whether the disagg transceiver ships the standalone drafter's context window.

A DFlash / DSpark drafter with a separate draft KV pool keeps the prompt's drafter K/V in that pool
(the context window its block decode attends to). Without this transfer a generation worker starts
every request from an empty window. The window travels as a second block set in the same session
and write as the target's KV.

The draft set ships only when all of these hold:

* the target model opts in through ``RuntimeHooks.disagg_draft_kv`` (off for every model by default);
* a separate draft KV cache manager exists (both sides: the peers refuse each other otherwise);
* ``TRTLLM_M31_DISABLE_DISAGG_DRAFT_KV`` is unset or false.

This module reads no model or manager state at import, so the environment contract is testable alone.
"""

import os
import threading
import time
from collections import Counter
from typing import List, Mapping, Optional, Tuple

DISABLE_DISAGG_DRAFT_KV_ENV = "TRTLLM_M31_DISABLE_DISAGG_DRAFT_KV"
# Test only: publish the context request's KV without waiting for its drafter-store event. This is the
# negative control of the ordering gate (it must break the draft-window oracle); never set it in serving.
TEST_SEND_BEFORE_EVENT_ENV = "TRTLLM_M31_F0A_TEST_SEND_BEFORE_EVENT"

_TRUE_VALUES = frozenset({"1", "true", "on", "yes"})
_FALSE_VALUES = frozenset({"", "0", "false", "off", "no"})


def parse_strict_bool_env(name: str, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Read a boolean environment variable under the strict value contract.

    ``1`` / ``true`` / ``on`` / ``yes`` (any case) are True; unset, empty, ``0`` / ``false`` /
    ``off`` / ``no`` are False; anything else raises ``ValueError`` naming the variable, so a typo
    never silently keeps (or drops) a feature.
    """
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of 1/true/on/yes or 0/false/off/no (or empty), got {raw!r}"
    )


def disagg_draft_kv_disabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_DISAGG_DRAFT_KV`` turns the draft-window transfer off."""
    return parse_strict_bool_env(DISABLE_DISAGG_DRAFT_KV_ENV, environ)


def send_before_draft_ctx_event(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the test-only knob skips the context side's drafter-store wait."""
    return parse_strict_bool_env(TEST_SEND_BEFORE_EVENT_ENV, environ)


def select_disagg_draft_kv_cache_manager(model: object, draft_kv_cache_manager: Optional[object]):
    """The draft KV cache manager the transceiver should ship, or None for the target-only path.

    ``model`` is the engine's model, whose ``runtime_hooks`` carry the opt-in. A model that opts in
    must run a drafter whose context lives in the managed pool, unpaired with the target's block
    reuse; anything else is refused here rather than mid-transfer.
    """
    if draft_kv_cache_manager is None:
        return None
    # Imported lazily: the hook module is light, but this module stays importable on its own.
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of

    if not runtime_hooks_of(model).disagg_draft_kv:
        return None
    if disagg_draft_kv_disabled():
        return None
    worker = getattr(model, "spec_worker", None)
    if worker is None or not hasattr(worker, "engage_disagg_draft_kv"):
        # A separate draft pool behind a worker without a paged drafter context (e.g. a one-model
        # Eagle3 layer): there is no context window to ship.
        return None
    if getattr(draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False):
        raise ValueError(
            "the disagg draft-window transfer requires an unpaired draft KV pool "
            "(enable_joint_kv_cache_reuse=False, i.e. TRTLLM_DRAFT_KEEP_JOINT_REUSE unset); "
            f"set {DISABLE_DISAGG_DRAFT_KV_ENV}=1 to serve without it"
        )
    return draft_kv_cache_manager


# ---------------------------------------------------------------------------------------------
# Per-request draft-window state and the fallback to an empty drafter window
# ---------------------------------------------------------------------------------------------

# Why a request's window is not shipped. The request itself never fails: it runs as if the
# transfer did not exist (the generation side's drafter starts from an empty window).
FALLBACK_NO_STORE_EVENT = "no_store_event"  # no drafter context store was recorded for it
FALLBACK_CTX_LEN_MISMATCH = "ctx_len_mismatch"  # the drafter did not store the whole prompt
FALLBACK_REUSE_BASE = "reuse_base"  # ctx_base outside the prompt's pages
FALLBACK_PAGE_MISMATCH = "page_mismatch"  # the context side lacks a window page
FALLBACK_NO_STATE = "no_ctx_state"  # generation side: no state arrived for the request
# The context TP ranks disagreed about a request (one shipped its slice, another did not), seen
# by the sender as a draft table on one side only, and by the generation ranks as a short draft
# write or an unusable state on some rank; every generation rank then falls back together.
FALLBACK_CTX_RANK_DIVERGENCE = "ctx_rank_divergence"
FALLBACK_RECEIVER_NOT_ASKED = "receiver_not_asked"  # context side shipped, the receiver asked none
FALLBACK_UNVERIFIED_WRITE = "unverified_write"  # the receive's byte attestation did not cover it
# The generation side receives synchronously (TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=1) with TP
# ranks that share requests: that path has no cross-rank agreement on a window's usability, so it
# receives none (target only) rather than let its ranks draft from different windows.
FALLBACK_SYNC_PATH_TP = "sync_path_tp"


def draft_receive_disabled_reason(mapping, synchronous_receive: bool) -> Optional[str]:
    """Why this rank must take no drafter window, or None.

    The synchronous receive decides a window's usability per rank with no consensus; TP ranks
    that share every request (TP > 1 without attention DP) could then seed different windows.
    """
    if synchronous_receive and mapping.tp_size > 1 and not mapping.enable_attention_dp:
        return FALLBACK_SYNC_PATH_TP
    return None


def encode_draft_ctx_state(ctx_len: int, ctx_base: int, valid: bool) -> List[int]:
    """The context side's drafter state for a context-first response's ``draft_ctx_state``.

    It rides its own DisaggregatedParams field, never ``opaque_state``: that one becomes the C++
    ContextPhaseParams' serialized DataTransceiverState, which parses and re-serializes the bytes.
    """
    return [int(ctx_len), int(ctx_base), 1 if valid else 0]


def decode_draft_ctx_state(state) -> Optional[Tuple[int, int, bool]]:
    """``(ctx_len, ctx_base, valid)`` from ``encode_draft_ctx_state``, or None for anything else."""
    if state is None or isinstance(state, (bytes, bytearray, str)):
        return None
    try:
        ctx_len, ctx_base, valid = (int(x) for x in state)
    except (TypeError, ValueError):
        return None
    if ctx_len < 0 or ctx_base < 0 or valid not in (0, 1):
        return None
    return ctx_len, ctx_base, bool(valid)


def empty_window_state(prompt_len: int, tokens_per_block: int) -> Tuple[int, int]:
    """Drafter ``(ctx_len, ctx_base)`` for a request whose window did not arrive.

    The generation side's draft pool holds only the pages behind the prompt end, so a drafter
    numbering from position 0 would address released pages. Basing the slot past every prompt
    page reads nothing and appends into the pages it holds: the drafter starts from an empty
    window, exactly as without the transfer.
    """
    return int(prompt_len), -(-int(prompt_len) // int(tokens_per_block))


class DraftKvCounters:
    """``draft_kv_sent`` / ``draft_kv_fallback{reason}`` for one side, logged periodically.

    One info line per ``interval_s`` (and on the first event) so the serving log carries the
    fallback rate; each fallback is also logged at debug level.
    """

    def __init__(self, role: str, interval_s: float = 60.0):
        self.role = role
        self.interval_s = interval_s
        self.sent = 0
        self.fallback: Counter = Counter()
        self._last_log: Optional[float] = None
        # The sender's worker threads count divergences too.
        self._lock = threading.Lock()

    def record_sent(self) -> None:
        with self._lock:
            self.sent += 1
        self._maybe_log()

    def record_fallback(self, reason: str, detail: str = "") -> None:
        with self._lock:
            first = self.fallback[reason] == 0
            self.fallback[reason] += 1
        from tensorrt_llm.logger import logger

        message = f"F0a draft window ({self.role}) falls back: reason={reason} {detail}"
        if first:
            # The first of each reason at info, with its detail; the rest at debug.
            logger.info(message)
        else:
            logger.debug(message)
        self._maybe_log()

    def summary(self) -> str:
        reasons = ",".join(f"{reason}={count}" for reason, count in sorted(self.fallback.items()))
        return (
            f"F0a draft window ({self.role}): draft_kv_sent={self.sent} "
            f"draft_kv_fallback={sum(self.fallback.values())} "
            f"draft_kv_fallback_by_reason={{{reasons}}}"
        )

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if self._last_log is not None and now - self._last_log < self.interval_s:
            return
        self._last_log = now
        from tensorrt_llm.logger import logger

        logger.info(self.summary())


# ---------------------------------------------------------------------------------------------
# Test only: draft-window dumps
# ---------------------------------------------------------------------------------------------

# A directory: each side dumps the window it shipped / received (see ``write_window_dump``).
# Unset is fully inert: nothing reads the pools and nothing is written.
DUMP_DIR_ENV = "TRTLLM_M31_F0A_DUMP_DIR"
DUMP_FORMAT = "f0a_g1d_v1"


def draft_dump_dir(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The draft-window dump directory, or None when the test-only knob is unset or empty."""
    env = os.environ if environ is None else environ
    return env.get(DUMP_DIR_ENV) or None


def _copy_to_host(dst, src_ptr: int, nbytes: int) -> None:
    import torch

    if torch.cuda.is_available():
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart

        (err,) = cudart.cudaMemcpy(
            dst.data_ptr(), src_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDefault
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"F0a dump: cudaMemcpy of {nbytes} B from {src_ptr:#x} failed: {err}"
            )
    else:
        import ctypes

        ctypes.memmove(dst.data_ptr(), src_ptr, nbytes)


def gather_window_pages(page_table, heads_per_rank: int, block_ids_per_group, ctx_len: int):
    """The window's held pages below ``ctx_len``, read through a draft block table.

    Returns ``(ordinals, layers)``: per layer group the block ordinals read (held and below
    ``ceil(ctx_len / tpb)``), and per layer ``{"layer_group", "global_layer_id", "k", "v"}``
    with ``uint8 [pages, heads_per_rank, tokens_per_block, row_bytes]`` each, the raw HND
    bytes of the K and V buffers (slot order) of those pages.
    """
    import numpy as np
    import torch

    tpb = int(page_table.tokens_per_block)
    num_pages = -(-int(ctx_len) // tpb)
    ordinals, layers = [], []
    for lg_idx, (group, block_ids) in enumerate(zip(page_table.layer_groups, block_ids_per_group)):
        ids = np.asarray(block_ids, dtype=np.int64)[:num_pages]
        held = np.flatnonzero(ids >= 0)
        ordinals.append(torch.as_tensor(held, dtype=torch.int64))
        local_to_global = {ll.local_layer_id: ll.global_layer_id for ll in group.local_layers}
        for view in group.pool_views:
            pool = page_table.pool_groups[group.pool_group_idx].pools[view.pool_idx]
            by_layer = {}
            for entry in view.buffer_entries:
                by_layer.setdefault(int(entry["local_layer_id"]), []).append(
                    (int(entry["offset"]), int(entry["size"]))
                )
            for local, buffers in sorted(by_layer.items(), key=lambda item: min(item[1])):
                buffers.sort()
                if len(buffers) != 2:
                    raise ValueError(f"F0a dump: layer {local} has {len(buffers)} buffers, not K+V")
                record = {"layer_group": lg_idx, "global_layer_id": int(local_to_global[local])}
                for name, (offset, size) in zip(("k", "v"), buffers):
                    row_bytes = size // (heads_per_rank * tpb)
                    out = torch.empty(
                        (len(held), heads_per_rank, tpb, row_bytes), dtype=torch.uint8
                    )
                    for i, ordinal in enumerate(held):
                        src = (
                            pool.base_address + int(ids[ordinal]) * pool.slot_stride_bytes + offset
                        )
                        _copy_to_host(out[i], src, size)
                    record[name] = out
                layers.append(record)
    return ordinals, layers


def write_window_dump(
    directory: str,
    side: str,
    rank: int,
    match_id: int,
    *,
    request_id: int,
    prompt_len: int,
    ctx_len: int,
    ctx_base: int,
    shipped: bool,
    reason: Optional[str] = None,
    page_table=None,
    heads_per_rank: int = 0,
    block_ids_per_group=None,
) -> str:
    """Write ``<directory>/<side>_<rank>_<match_id>.pt`` (atomically: a temp name, then rename).

    ``shipped`` False writes a marker without pages (a context-side fallback).
    """
    import torch

    payload = {
        "format": DUMP_FORMAT,
        "side": side,
        "rank": int(rank),
        "match_id": int(match_id),
        "request_id": int(request_id),
        "prompt_len": int(prompt_len),
        "ctx_len": int(ctx_len),
        "ctx_base": int(ctx_base),
        "shipped": bool(shipped),
        "reason": reason,
        "heads_per_rank": int(heads_per_rank),
        "tokens_per_block": None,
        "ordinals": [],
        "layers": [],
    }
    if shipped:
        payload["tokens_per_block"] = int(page_table.tokens_per_block)
        payload["ordinals"], payload["layers"] = gather_window_pages(
            page_table, heads_per_rank, block_ids_per_group, ctx_len
        )
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, f"{side}_{int(rank)}_{int(match_id)}.pt")
    temp = f"{final}.tmp.{os.getpid()}"
    torch.save(payload, temp)
    os.replace(temp, final)
    return final
