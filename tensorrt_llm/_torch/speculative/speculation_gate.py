from collections import deque
from typing import Deque, Optional, Tuple

from tensorrt_llm.logger import logger


class SpeculationGate:
    """
    Tracks a rolling average of true acceptance-rate samples over the last N
    speculation-enabled decoding iterations.

    Permanently disables speculation when the rolling average falls below the
    configured threshold.
    """

    def __init__(self, window: int, threshold: float):
        self.window = window
        self.threshold = threshold
        self.acceptance_rate_history: Deque[float] = deque()
        self.acceptance_rate_sum: float = 0.0
        self.num_recorded_samples = 0
        self.disabled = False
        logger.debug(
            f"[SpeculationGate] SpeculationGate initialized with window={self.window}, threshold={self.threshold}"
        )

    def reset(self) -> None:
        self.acceptance_rate_history.clear()
        self.acceptance_rate_sum = 0.0
        self.num_recorded_samples = 0
        self.disabled = False

    def record_acceptance_rate(
            self,
            acceptance_rate: float,
            sample_id: Optional[int] = None) -> Tuple[bool, Optional[float]]:
        """
        Record one speculation-enabled iteration's true acceptance rate.

        Returns (disabled_now, current_avg_acceptance_rate) where
        disabled_now is True only when this call causes permanent disable.
        """
        if self.disabled or self.window is None or self.window <= 0 or self.threshold is None:
            return False, None

        if acceptance_rate is None:
            return False, None

        acceptance_rate = float(acceptance_rate)
        if acceptance_rate < 0.0 or acceptance_rate > 1.0:
            raise ValueError("acceptance_rate must be in the range [0.0, 1.0], "
                             f"got {acceptance_rate}")

        if sample_id is not None:
            logger.debug(f"[SpeculationGate] Iteration {sample_id} recorded "
                         f"acceptance_rate={acceptance_rate:.3f}")

        # O(1) rolling update
        self.acceptance_rate_history.append(acceptance_rate)
        logger.debug(f"[SpeculationGate] Acceptance-rate history: "
                     f"{self.acceptance_rate_history}")
        self.acceptance_rate_sum += acceptance_rate
        if len(self.acceptance_rate_history) > self.window:
            removed = self.acceptance_rate_history.popleft()
            self.acceptance_rate_sum -= removed

        self.num_recorded_samples += 1

        if self.num_recorded_samples >= self.window:
            avg_acceptance_rate = (self.acceptance_rate_sum /
                                   len(self.acceptance_rate_history))
            if avg_acceptance_rate < self.threshold:
                self.disabled = True
                logger.info(
                    "[SpeculationGate] Speculative decoding disabled: "
                    f"rolling acceptance rate avg "
                    f"{avg_acceptance_rate:.3f} < threshold "
                    f"{self.threshold} over last {self.window} iterations")
                return True, avg_acceptance_rate
            return False, avg_acceptance_rate

        return False, None
