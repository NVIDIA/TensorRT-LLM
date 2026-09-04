import os
import sys
import unittest

import pytest

from tensorrt_llm._torch.speculative.speculation_gate import SpeculationGate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.cpu_only
def test_returns_none_until_window_and_enabled_when_above_threshold():
    gate = SpeculationGate(window=3, threshold=0.5)

    disabled, avg = gate.record_acceptance_rate(0.8, sample_id=1)
    assert disabled is False and avg is None
    assert gate.disabled is False

    disabled, avg = gate.record_acceptance_rate(0.8, sample_id=2)
    assert disabled is False and avg is None
    assert gate.disabled is False

    disabled, avg = gate.record_acceptance_rate(0.8, sample_id=3)
    assert disabled is False
    assert avg == pytest.approx(0.8, rel=1e-6)
    assert gate.disabled is False


@pytest.mark.cpu_only
def test_disables_when_avg_below_threshold_and_stays_disabled():
    gate = SpeculationGate(window=3, threshold=0.3)

    gate.record_acceptance_rate(0.1)
    gate.record_acceptance_rate(0.2)

    disabled, avg = gate.record_acceptance_rate(0.3)
    assert disabled is True
    assert avg == pytest.approx(0.2, rel=1e-6)
    assert gate.disabled is True

    # Once disabled, subsequent calls do nothing and return (False, None)
    disabled, avg = gate.record_acceptance_rate(1.0)
    assert disabled is False and avg is None
    assert gate.disabled is True

    disabled, avg = gate.record_acceptance_rate(1.0)
    assert disabled is False and avg is None
    assert gate.disabled is True


@pytest.mark.cpu_only
def test_rolling_window_and_disable_on_drop():
    gate = SpeculationGate(window=3, threshold=0.7)

    # First three high-acceptance requests keep it enabled
    gate.record_acceptance_rate(0.9)
    gate.record_acceptance_rate(0.9)
    disabled, avg = gate.record_acceptance_rate(0.9)
    assert disabled is False
    assert avg == pytest.approx(0.9, rel=1e-6)
    assert gate.disabled is False

    # Fourth lower value enters window -> average drops below threshold -> disable
    disabled, avg = gate.record_acceptance_rate(0.2)
    assert disabled is True
    assert avg == pytest.approx((0.9 + 0.9 + 0.2) / 3.0, rel=1e-6)
    assert gate.disabled is True


if __name__ == "__main__":
    unittest.main()
