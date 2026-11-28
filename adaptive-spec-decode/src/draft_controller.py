"""
Adaptive Draft Length Controller.
Maps acceptance rate → optimal draft length.
"""
from dataclasses import dataclass
from .acceptance_monitor import AcceptanceMonitor

@dataclass
class DraftDecision:
    draft_length: int
    strategy: str
    confidence: float
    reason: str

class AdaptiveDraftController:
    AGGRESSIVE_THRESHOLD = 0.75
    MODERATE_THRESHOLD = 0.50
    CONSERVATIVE_THRESHOLD = 0.30

    DRAFT_LENGTHS = {
        "aggressive": 8,
        "moderate": 5,
        "conservative": 2,
        "vanilla": 0,
    }

    def __init__(self, monitor, min_draft=1, max_draft=10):
        self.monitor = monitor
        self.min_draft = min_draft
        self.max_draft = max_draft
        self._decision_history = []

    def decide(self):
        rate = self.monitor.acceptance_rate
        trend = self.monitor.acceptance_trend

        if not self.monitor.is_warmed_up:
            return DraftDecision(5, "warmup", 0.3,
                f"Warmup phase (step {self.monitor._step_count})")

        if rate >= self.AGGRESSIVE_THRESHOLD:
            strategy = "aggressive"
            draft_len = self.DRAFT_LENGTHS["aggressive"]
            if trend == "improving":
                draft_len = min(draft_len + 2, self.max_draft)
        elif rate >= self.MODERATE_THRESHOLD:
            strategy = "moderate"
            draft_len = self.DRAFT_LENGTHS["moderate"]
        elif rate >= self.CONSERVATIVE_THRESHOLD:
            strategy = "conservative"
            draft_len = self.DRAFT_LENGTHS["conservative"]
            if trend == "degrading":
                draft_len = max(draft_len - 1, self.min_draft)
        else:
            strategy = "vanilla"
            draft_len = 0

        if strategy != "vanilla":
            draft_len = max(self.min_draft, min(draft_len, self.max_draft))

        decision = DraftDecision(
            draft_length=draft_len,
            strategy=strategy,
            confidence=min(1.0, self.monitor._step_count / 10),
            reason=f"rate={rate:.2f}, trend={trend}, -> {strategy} (k={draft_len})"
        )
        self._decision_history.append(decision)
        return decision

    def get_report(self):
        if not self._decision_history:
            return {"total_decisions": 0}
        strategy_counts = {}
        for d in self._decision_history:
            strategy_counts[d.strategy] = strategy_counts.get(d.strategy, 0) + 1
        return {
            "total_decisions": len(self._decision_history),
            "strategy_distribution": strategy_counts,
            "avg_draft_length": sum(d.draft_length for d in self._decision_history) / len(self._decision_history),
            "final_acceptance_rate": self.monitor.acceptance_rate,
        }
