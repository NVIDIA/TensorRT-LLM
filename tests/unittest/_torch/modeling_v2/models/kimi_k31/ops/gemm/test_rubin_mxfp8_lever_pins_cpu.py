# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of ``rubin_mxfp8_gemm``'s ``pinned`` flag (dep16's MXFP8 levers).

The entry runs with a recording autotuner and recording runners: unpinned, every GEMM asks the
family runner, as before; pinned, it asks the split-K-free runner and, at 81 to 96 rows on a
``PINNED_TACTICS`` weight shape, runs the pinned tactic; ``rubin_mxfp8_linear`` passes the flag.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm import (
    rubin_mxfp8_gemm,
    rubin_mxfp8_linear,
)

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="K3.1 Rubin (sm_107) entries; the CPU stage runs this file",
    ),
]

G = rubin_mxfp8_gemm


class _Runner:
    def __init__(self, name: str, log: list):
        self.name, self.log, self.tuning_config = name, log, f"{name}_config"

    def __call__(self, inputs, tactic):
        self.log.append(("run", self.name, tactic))
        return self.name


class _Tuner:
    def __init__(self, log: list):
        self.log = log

    def get(self):
        return self

    def choose_one(self, key, runners, config, inputs):
        (runner,) = runners
        self.log.append(("ask", key, runner.name, config))
        return runner, f"tuned:{runner.name}"


@pytest.fixture
def recorded(monkeypatch):
    log = []
    family, free = _Runner("family", log), _Runner("split_k_free", log)
    monkeypatch.setattr(G, "_runner", lambda: (_Tuner(log), family))
    monkeypatch.setattr(G, "_split_k_free_runner", lambda: free)
    monkeypatch.setattr(G, "_alpha", lambda device: torch.ones(()))
    return log


def _call(m: int, n: int, k: int, **kw):
    a = torch.zeros(m, k, dtype=torch.uint8)
    w = torch.zeros(n, k, dtype=torch.uint8)
    return G.rubin_mxfp8_gemm(a, w, torch.zeros(1), torch.zeros(1), **kw)


def test_unpinned_gemms_ask_the_family_runner(recorded):
    for m in (8, 96, 2954):
        assert _call(m, 7168, 12288) == "family"
    asks = [e for e in recorded if e[0] == "ask"]
    assert {e[2] for e in asks} == {"family"} and {e[1] for e in asks} == {G.TUNING_KEY}
    assert [e[2] for e in recorded if e[0] == "run"] == ["tuned:family"] * 3


@pytest.mark.parametrize("shape", sorted(G.PINNED_TACTICS))
def test_pinned_gemms_run_the_pin_at_the_verify_bucket_and_never_split_k(recorded, shape):
    n, k = shape
    for m in (G.PINNED_MIN_M - 1, G.PINNED_MIN_M, 90, G.PINNED_MAX_M, G.PINNED_MAX_M + 1):
        assert _call(m, n, k, pinned=True) == "split_k_free"
    asks = [e for e in recorded if e[0] == "ask"]
    assert len(asks) == 5 and {e[2] for e in asks} == {"split_k_free"}
    runs = [e[2] for e in recorded if e[0] == "run"]
    pin = G.PINNED_TACTICS[shape]
    assert runs == ["tuned:split_k_free", pin, pin, pin, "tuned:split_k_free"]
    assert pin[0] == "base" and pin[8] == 1


def test_a_pinned_gemm_off_the_table_takes_the_split_k_free_pick(recorded):
    assert (1536, 7168) not in G.PINNED_TACTICS
    assert _call(96, 1536, 7168, pinned=True) == "split_k_free"
    assert recorded[-1] == ("run", "split_k_free", "tuned:split_k_free")


def test_the_linear_entry_passes_the_flag(monkeypatch, recorded):
    quantized = []

    def quantize(x):  # rubin_mxfp8_quantize: swizzled scales, 32 values each
        quantized.append(tuple(x.shape))
        return torch.zeros(x.shape, dtype=torch.uint8), torch.zeros(1)

    monkeypatch.setattr(rubin_mxfp8_linear, "_quantize", lambda: quantize)
    w = torch.zeros(7168, 12288, dtype=torch.uint8)
    lin = rubin_mxfp8_linear.rubin_mxfp8_linear
    assert lin(torch.zeros(96, 12288), w, torch.zeros(1), pinned=True) == "split_k_free"
    assert recorded[-1] == ("run", "split_k_free", G.PINNED_TACTICS[(7168, 12288)])
    assert lin(torch.zeros(96, 12288), w, torch.zeros(1)) == "family"
    assert quantized == [(96, 12288)] * 2


def test_the_split_k_free_runner_caps_split_k_at_one():
    source = G.__loader__.get_source(G.__name__)
    assert "RubinMxfp8GemmRunner(output_dtype=torch.bfloat16, max_split_k=1)" in source
