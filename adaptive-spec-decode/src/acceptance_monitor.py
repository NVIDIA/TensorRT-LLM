"""
Real-time acceptance rate monitor for speculative decoding.
Tracks per-request acceptance rates using exponential moving average.
"""
import time
from dataclasses import dataclass
from collections import deque

@dataclass
class AcceptanceStats:
    total_drafted: int = 0
    total_accepted: int = 0
    acceptance_rate: float = 0.0
    timestamp: float = 0.0

class AcceptanceMonitor:
    def __init__(self, ema_alpha=0.3, window_size=10, warmup_steps=3):
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self._ema_rate = 0.5
        self._history = deque(maxlen=window_size)
        self._step_count = 0
        self._total_drafted = 0
        self._total_accepted = 0

    def update(self, drafted, accepted):
        if drafted == 0:
            return self.current_stats()
        instant_rate = accepted / drafted
        self._ema_rate = (self.ema_alpha * instant_rate +
                         (1 - self.ema_alpha) * self._ema_rate)
        self._step_count += 1
        self._total_drafted += drafted
        self._total_accepted += accepted
        stats = AcceptanceStats(drafted, accepted, self._ema_rate, time.perf_counter())
        self._history.append(stats)
        return stats

    def current_stats(self):
        return AcceptanceStats(self._total_drafted, self._total_accepted,
                              self._ema_rate, time.perf_counter())

    @property
    def is_warmed_up(self):
        return self._step_count >= self.warmup_steps

    @property
    def acceptance_rate(self):
        return self._ema_rate

    @property
    def acceptance_trend(self):
        if len(self._history) < 3:
            return "unknown"
        recent = list(self._history)[-3:]
        if recent[-1].acceptance_rate > recent[0].acceptance_rate + 0.05:
            return "improving"
        elif recent[-1].acceptance_rate < recent[0].acceptance_rate - 0.05:
            return "degrading"
        return "stable"
