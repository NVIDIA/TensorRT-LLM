# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import time
from typing import Callable, Iterable, List, Optional, Tuple

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm.logger import logger

# A generation KV receive that has been in flight for longer than this stops
# counting toward the transfer window. The window bounds the transfer bandwidth
# and buffer a rank commits to at once; a receive that has made no progress for
# this long is consuming neither, it is wedged (a handshake the context side
# cannot answer, a lost completion) and will end through kv_transfer_timeout_ms.
# Keeping its blocks in the budget lets that one request park every later
# gen-init request on the rank: admission is FCFS, so the first candidate that
# no longer fits next to it defers everything behind it, and nothing moves until
# the timeout fires. The default sits above the slowest healthy receive seen
# (first contact between a context and a generation rank at cold start takes
# 10-20 s) and far below the timeout. 0 disables the exclusion.
STALE_TRANSFER_ENV = "TRTLLM_DISAGG_ADMISSION_STALE_TRANSFER_S"
DEFAULT_STALE_TRANSFER_S = 30.0


def stale_transfer_s_from_env() -> float:
    raw = os.environ.get(STALE_TRANSFER_ENV)
    if raw is None:
        return DEFAULT_STALE_TRANSFER_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            f"Ignoring {STALE_TRANSFER_ENV}={raw!r}: not a number; "
            f"using {DEFAULT_STALE_TRANSFER_S} s"
        )
        return DEFAULT_STALE_TRANSFER_S


@dataclasses.dataclass
class DisaggTransferAdmissionResult:
    admitted_requests: List[LlmRequest]
    active_transfer_blocks: int = 0
    admitted_transfer_blocks: int = 0
    deferred_request_count: int = 0
    limited_by_budget: bool = False
    # Receives in flight longer than the stale threshold: their blocks are
    # reported, not counted against the window, and the oldest one's age.
    stale_transfer_blocks: int = 0
    stale_transfer_oldest_s: float = 0.0

    def is_blocked_by_active_transfers(self) -> bool:
        return (
            self.limited_by_budget
            and not self.admitted_requests
            and self.active_transfer_blocks > 0
        )


class DisaggTransferAdmissionController:
    """FCFS admission gate for disaggregated generation KV transfers."""

    def __init__(
        self,
        max_tokens_in_buffer: Optional[int],
        tokens_per_block: Optional[int],
        stale_transfer_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_transfer_blocks = self._to_block_budget(max_tokens_in_buffer, tokens_per_block)
        self.tokens_per_block = tokens_per_block or 0
        # None reads the environment; the clock is injectable for tests and
        # must match the one that stamps py_kv_transfer_start_time.
        self.stale_transfer_s = (
            stale_transfer_s_from_env()
            if stale_transfer_s is None
            else max(0.0, float(stale_transfer_s))
        )
        self._clock = clock

    def enabled(self) -> bool:
        return self.max_transfer_blocks is not None

    @staticmethod
    def _to_block_budget(
        max_tokens_in_buffer: Optional[int], tokens_per_block: Optional[int]
    ) -> Optional[int]:
        if (
            max_tokens_in_buffer is None
            or max_tokens_in_buffer == 0
            or tokens_per_block is None
            or tokens_per_block <= 0
        ):
            return None
        return (max_tokens_in_buffer + tokens_per_block - 1) // tokens_per_block

    @staticmethod
    def _to_nonnegative_int(value) -> Optional[int]:
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            return None

    def _get_request_transfer_token_count(self, request: LlmRequest) -> int:
        for attr_name in ("total_input_len_cp", "py_prompt_len", "prompt_len"):
            token_count = self._to_nonnegative_int(getattr(request, attr_name, None))
            if token_count is not None:
                return token_count
        return 0

    def _estimate_request_blocks(self, request: LlmRequest) -> int:
        if self.tokens_per_block <= 0:
            return 0
        prompt_len = self._get_request_transfer_token_count(request)
        return (prompt_len + self.tokens_per_block - 1) // self.tokens_per_block

    def _estimate_requests_blocks(self, requests: Iterable[LlmRequest]) -> int:
        return sum(self._estimate_request_blocks(request) for request in requests)

    def _is_stale_transfer(self, request: LlmRequest, now: float) -> bool:
        if self.stale_transfer_s <= 0:
            return False
        started = getattr(request, "py_kv_transfer_start_time", None)
        # Only a real monotonic stamp is evidence; anything else (None, a test
        # double) keeps the request counted.
        if not isinstance(started, (int, float)) or isinstance(started, bool):
            return False
        return now - started > self.stale_transfer_s

    def _split_active_transfer_blocks(
        self, active_requests: Iterable[LlmRequest]
    ) -> Tuple[int, int, float]:
        """Blocks of the receives in flight as (counted, stale, oldest stale age in s).

        Only the counted part holds window budget; see STALE_TRANSFER_ENV.
        """
        now = self._clock()
        counted = stale = 0
        oldest_s = 0.0
        for request in active_requests:
            if not request.is_disagg_generation_transmission_in_progress:
                continue
            blocks = self._estimate_request_blocks(request)
            if self._is_stale_transfer(request, now):
                stale += blocks
                oldest_s = max(oldest_s, now - request.py_kv_transfer_start_time)
            else:
                counted += blocks
        return counted, stale, oldest_s

    def _estimate_active_transfer_blocks(self, active_requests: Iterable[LlmRequest]) -> int:
        return self._split_active_transfer_blocks(active_requests)[0]

    def select(
        self, active_requests: Iterable[LlmRequest], candidates: List[LlmRequest]
    ) -> DisaggTransferAdmissionResult:
        if not self.enabled():
            return DisaggTransferAdmissionResult(
                admitted_requests=list(candidates),
                active_transfer_blocks=self._estimate_active_transfer_blocks(active_requests),
                admitted_transfer_blocks=self._estimate_requests_blocks(candidates),
            )

        result = DisaggTransferAdmissionResult(admitted_requests=[])
        (
            result.active_transfer_blocks,
            result.stale_transfer_blocks,
            result.stale_transfer_oldest_s,
        ) = self._split_active_transfer_blocks(active_requests)
        if result.stale_transfer_blocks:
            logger.warning_once(
                "Disagg transfer admission: a generation KV receive has been in flight for "
                f"{result.stale_transfer_oldest_s:.0f} s (> {self.stale_transfer_s:.0f} s); "
                f"{result.stale_transfer_blocks} block(s) of such receives no longer hold the "
                "transfer window, so later requests on this rank are not parked behind them "
                "(the receives themselves end through kv_transfer_timeout_ms). Logged once.",
                key="disagg_admission_stale_transfer",
            )

        used_blocks = result.active_transfer_blocks
        max_transfer_blocks = self.max_transfer_blocks
        assert max_transfer_blocks is not None
        for request in candidates:
            request_blocks = self._estimate_request_blocks(request)
            fits_budget = used_blocks + request_blocks <= max_transfer_blocks
            admit_oversized_head = (
                not result.admitted_requests
                and result.active_transfer_blocks == 0
                and request_blocks > max_transfer_blocks
            )
            if not fits_budget and not admit_oversized_head:
                result.limited_by_budget = True
                break

            result.admitted_requests.append(request)
            used_blocks += request_blocks
            result.admitted_transfer_blocks += request_blocks

        result.deferred_request_count = len(candidates) - len(result.admitted_requests)
        return result
