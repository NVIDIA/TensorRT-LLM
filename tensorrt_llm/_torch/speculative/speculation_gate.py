from collections import deque
from typing import Optional, Tuple

from tensorrt_llm.logger import logger


class SpeculationGate:
    """
	Tracks rolling average of accepted draft tokens per iteration over the last N completed requests.
	Permanently disables speculation when average falls below a threshold.
	"""

    def __init__(self, window: int, threshold: float):
        self.window = window
        self.threshold = threshold
        self.acceptance_history: Deque[float] = deque()
        self.acceptance_sum: float = 0.0
        self.num_completed_for_acceptance = 0
        self.disabled = False
        logger.debug(
            f"[SpeculationGate] SpeculationGate initialized with window={self.window}, threshold={self.threshold}"
        )

    def reset(self) -> None:
        self.acceptance_history.clear()
        self.acceptance_sum = 0.0
        self.num_completed_for_acceptance = 0
        self.disabled = False

    def record_avg_decoded(
            self,
            avg_decoded_tokens_per_iter: float,
            request_id: Optional[int] = None) -> Tuple[bool, Optional[float]]:
        """
        Record a completed request's avg_decoded_tokens_per_iter.
        Returns (disabled_now, current_avg_accept) where disabled_now is True only when the call causes disable.
        """
        if self.disabled or self.window is None or self.window <= 0 or self.threshold is None:
            return False, None

        # Extra Guard: if caller passed None, skip updating the rolling stats
        if avg_decoded_tokens_per_iter is None:
            return False, None

        accepted_len = 0.0
        accepted_len = max(0.0, float(avg_decoded_tokens_per_iter) - 1.0)

        # Log per-request completion for debug
        if request_id is not None:
            logger.debug(
                f"[SpeculationGate] Request {request_id} completed: avg_decoded={avg_decoded_tokens_per_iter if avg_decoded_tokens_per_iter is not None else 'None'}, accepted_len={accepted_len:.3f}"
            )

        # O(1) rolling update
        self.acceptance_history.append(accepted_len)
        logger.debug(
            f"[SpeculationGate] Acceptance history: {self.acceptance_history}")
        self.acceptance_sum += accepted_len
        if len(self.acceptance_history) > self.window:
            removed = self.acceptance_history.popleft()
            self.acceptance_sum -= removed

        self.num_completed_for_acceptance += 1

        if self.num_completed_for_acceptance >= self.window:
            avg_accept = self.acceptance_sum / len(self.acceptance_history)
            if avg_accept < self.threshold:
                self.disabled = True
                logger.info(
                    f"[SpeculationGate] Speculative decoding disabled: rolling acceptance avg {avg_accept:.3f} < threshold {self.threshold} over last {self.window} requests"
                )
                return True, avg_accept
            else:
                # speculation is still enabled
                return False, avg_accept

        return False, None
