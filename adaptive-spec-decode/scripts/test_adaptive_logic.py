"""Unit tests for adaptive logic — runs on CPU, no GPU needed!"""
import sys
sys.path.insert(0, "/teamspace/studios/this_studio/adaptive-spec-decode")

from src.acceptance_monitor import AcceptanceMonitor
from src.draft_controller import AdaptiveDraftController

def test_monitor_ema():
    monitor = AcceptanceMonitor(ema_alpha=0.3, warmup_steps=2)
    monitor.update(drafted=10, accepted=9)
    assert monitor.acceptance_rate > 0.5, f"Expected > 0.5, got {monitor.acceptance_rate}"
    monitor.update(drafted=10, accepted=8)
    assert monitor.is_warmed_up, "Should be warmed up after 2 steps"
    for _ in range(5):
        monitor.update(drafted=10, accepted=2)
    assert monitor.acceptance_rate < 0.4, f"Expected < 0.4, got {monitor.acceptance_rate}"
    print("✅ test_monitor_ema PASSED")

def test_controller_strategies():
    monitor = AcceptanceMonitor(ema_alpha=1.0, warmup_steps=1)
    controller = AdaptiveDraftController(monitor)
    decision = controller.decide()
    assert decision.strategy == "warmup", f"Expected warmup, got {decision.strategy}"
    monitor.update(drafted=10, accepted=9)
    decision = controller.decide()
    assert decision.strategy == "aggressive", f"Expected aggressive, got {decision.strategy}"
    assert decision.draft_length >= 7, f"Expected >= 7, got {decision.draft_length}"
    for _ in range(5):
        monitor.update(drafted=10, accepted=1)
    decision = controller.decide()
    assert decision.strategy in ["conservative", "vanilla"], f"Expected conservative/vanilla, got {decision.strategy}"
    print("✅ test_controller_strategies PASSED")

def test_trend_detection():
    monitor = AcceptanceMonitor(ema_alpha=0.5, warmup_steps=1)
    monitor.update(drafted=10, accepted=3)
    monitor.update(drafted=10, accepted=5)
    monitor.update(drafted=10, accepted=8)
    assert monitor.acceptance_trend == "improving", f"Expected improving, got {monitor.acceptance_trend}"
    monitor2 = AcceptanceMonitor(ema_alpha=0.5, warmup_steps=1)
    monitor2.update(drafted=10, accepted=9)
    monitor2.update(drafted=10, accepted=6)
    monitor2.update(drafted=10, accepted=2)
    assert monitor2.acceptance_trend == "degrading", f"Expected degrading, got {monitor2.acceptance_trend}"
    print("✅ test_trend_detection PASSED")

def test_controller_report():
    monitor = AcceptanceMonitor(ema_alpha=0.5, warmup_steps=1)
    controller = AdaptiveDraftController(monitor)
    monitor.update(drafted=10, accepted=8)
    controller.decide()
    controller.decide()
    controller.decide()
    report = controller.get_report()
    assert report["total_decisions"] == 3
    assert "strategy_distribution" in report
    print("✅ test_controller_report PASSED")

if __name__ == "__main__":
    test_monitor_ema()
    test_controller_strategies()
    test_trend_detection()
    test_controller_report()
    print("\n🎉 ALL TESTS PASSED — Adaptive logic is correct!")
