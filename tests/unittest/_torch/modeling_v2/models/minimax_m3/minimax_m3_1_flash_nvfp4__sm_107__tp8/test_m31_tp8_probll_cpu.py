# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP8_PROBLL (the one-request draft probabilities) on CPU: the switch, the TP8-only attach and the DRAFT_PROB1 hook.
The numerics checks (greedy parity with the stock LL chain, lg bounds, gather, q vs softmax, chi-square of the draws)
run at set-up on every rank of a TP8 decode worker (8 GPUs)."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

probll = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.draft_probs_one_request"
)
PROB1 = Path(probll.__file__).resolve().parents[1] / "kernels" / "m3ll" / "prob1" / "draft_prob1.py"


def test_default_on_with_kill_switch(monkeypatch):
    monkeypatch.delenv(probll.SWITCH, raising=False)
    assert probll.SWITCH == "TRTLLM_M31_DISABLE_TP8_PROBLL"
    assert importlib.reload(probll).ENABLED is True
    monkeypatch.setenv(probll.SWITCH, "1")
    assert importlib.reload(probll).ENABLED is False
    monkeypatch.setenv(probll.SWITCH, "maybe")
    with pytest.raises(ValueError, match=probll.SWITCH):
        importlib.reload(probll)
    monkeypatch.delenv(probll.SWITCH)
    importlib.reload(probll)


@pytest.mark.parametrize("tp", [2, 4])
def test_attach_is_tp8_only(tp):
    mapping = SimpleNamespace(tp_size=tp, tp_rank=0)
    assert probll.attach(None, mapping, None, None, None, 0, 200064, 7) is None


def test_killed_attach_takes_no_collective(monkeypatch):
    monkeypatch.setattr(probll, "ENABLED", False)
    monkeypatch.setattr(probll, "_agree", lambda *a: pytest.fail("collective while killed"))
    mapping = SimpleNamespace(tp_size=8, tp_rank=5)
    assert probll.attach(None, mapping, None, None, None, 0, 200064, 7) is None


def test_prob1_hook():
    src = PROB1.read_text()
    assert "from ....ops import draft_probs_one_request as _draft_probll" in src
    assert (
        "self._probll = _draft_probll.attach(self, mapping, comm, w1, w2_shard, voff, vocab, k)"
        in src
    )
    # The hook runs before the stock op and falls back to it when the one-request path declines.
    run = src.index("tokens = self._probll.run(")
    assert src.index("_op(self.op)(", run) > run
