# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The s2 head sampler's LM-head GEMM tactic and its arming decision (``kernels/s2_sampler/sampler.py``), on the host.

``sampler.py`` is loaded from its file into a stand-in package; the leaves it imports when called (the autotuner, the
low-M GEMM dispatcher, the split-K module at flashinfer's path, the logger, the TP all-gather and the context-K/V head
fork point) are stubbed for each test.
CPU torch only: no device, no built extension. The tests pin:

* with the s2 head installed, its 8-row GEMM runs the dispatcher's split-K runner at ``PINNED_TACTIC``, one of the
  bit-equivalent tactics, whatever the autotuner has cached for the shape (cuBLAS, the direct kernel, a split-K tactic
  in or outside the set, or no entry), and without asking the autotuner;
* the arming decision reads no autotuner entry: it waits only while the autotuner tunes, then arms, and refuses only
  for a shape the pinned tactic does not serve or a call that does not take the low-M dispatcher; one rank's refusal
  disarms every rank, once;
* whether the pinned tactic serves the shape is the verdict of ``validate_tactic`` of the split-K module at
  flashinfer's path (the one the runner launches through), not membership in the runner's autotune list: a list whose
  stage window excludes the pinned stage count (the served module's oversized shared-memory window) still arms when
  validation accepts the tactic, and a tactic validation rejects is refused with its reason;
* a refusal with the rejection-sampling areas reserved is logged as an error that names the fallback; without them it
  is a warning;
* the stock head is left alone: other row counts never reach the decision, and arming or running the s2 GEMM leaves
  the autotuner and the dispatcher's state as they were.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
_PACKAGE = "_musespark_open_tep_s2_pinned_gemm_under_test"

ROWS = 8
HIDDEN = 128
SHARD = 32
#: What a fresh autotuner cache has been seen to hold for the M = 8 LM-head shape, and members of the set.
SPLIT_K_OUTSIDE = ((128, 16, 1, 9), (128, 8, 1, 9), (64, 8, 1, 18), (128, 8, 2, 9))
SPLIT_K_INSIDE = ((128, 16, 1, 5), (128, 16, 1, 6), (128, 32, 1, 5))


class _Log:
    """The logger stub: every call as (level, message)."""

    def __init__(self) -> None:
        self.records: list = []

    def info(self, message, *args, **kwargs) -> None:
        self.records.append(("info", str(message)))

    def warning(self, message, *args, **kwargs) -> None:
        self.records.append(("warning", str(message)))

    def error(self, message, *args, **kwargs) -> None:
        self.records.append(("error", str(message)))

    def messages(self, level: str) -> list:
        return [message for kind, message in self.records if kind == level]


class _Runner:
    """A low-M runner: its valid tactics, and every call's tactic; a call writes ``a @ b_t`` into ``out``."""

    def __init__(self, name: str, tactics=()) -> None:
        self.name = name
        self.tactics = list(tactics)
        self.queried: list = []
        self.calls: list = []

    def get_valid_tactics(self, inputs, profile, **kwargs) -> list:
        self.queried.append(tuple(tuple(t.shape) for t in inputs))
        return list(self.tactics)

    def __call__(self, inputs, *, tactic=-1, bias=None, out=None):
        self.calls.append(tactic)
        out.copy_(inputs[0] @ inputs[1])
        return out


class _SplitKModule:
    """The split-K module at flashinfer's path: ``SplitKTactic`` and a ``validate_tactic`` with the module's stage
    bounds and public-row cap (``max_stages``: the stage cap its shared-memory capacity gives); every call recorded."""

    def __init__(self, max_stages: int = 20, max_m: int = 32) -> None:
        self.max_stages = max_stages
        self.max_m = max_m
        self.validated: list = []

    @staticmethod
    def SplitKTactic(mma_m, mma_n, split_k, ab_stages):  # noqa: N802 - the module's class name
        return (mma_m, mma_n, split_k, ab_stages)

    def validate_tactic(self, tactic, m, n, k, *, smem_capacity=None) -> None:
        self.validated.append((tuple(tactic), m, n, k))
        if not 1 <= tactic[3] <= self.max_stages:
            raise ValueError(f"ab_stages must be in [1, {self.max_stages}], got {tactic[3]}")
        if not 1 <= m <= self.max_m:
            raise ValueError(f"this low-M policy requires 1 <= M <= {self.max_m}, got {m}")


class _Dispatcher:
    """``LowMGemmDispatcher``'s state the s2 head reads: its runners, and the stock call's output buffers."""

    def __init__(self, split_k_tactics=SPLIT_K_INSIDE + SPLIT_K_OUTSIDE) -> None:
        self._prepared = True
        self._runner_no_bias = _Runner("split_k", split_k_tactics)
        self._cublas_no_bias = _Runner("cublas")
        self._direct = _Runner("direct")
        self._tuning_config = object()
        self._output_buffers: dict = {}

    @staticmethod
    def _is_candidate_shape(x, weight, bias) -> bool:
        return (
            bias is None
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_contiguous()
            and weight.is_contiguous()
            and x.shape[-1] == weight.shape[1]
        )


class _Tuner:
    """The autotuner: tuning or not, and the choice its cache holds for the shape (None: no entry). Every question
    asked of it is counted."""

    def __init__(self, dispatcher: _Dispatcher, cached=None, tuning: bool = False) -> None:
        self.is_tuning_mode = tuning
        self.cached = cached
        self.dispatcher = dispatcher
        self.asked = 0

    def choose_one(self, *args, **kwargs):
        self.asked += 1
        if self.cached is None:
            return self.dispatcher._cublas_no_bias, -1
        return self.cached

    @property
    def profiling_cache(self):
        self.asked += 1
        raise AssertionError("the s2 head read the autotuner's cache")


class _Group:
    """``Distributed.get``'s TP group: this rank's verdict, and the other ranks' (``others``)."""

    def __init__(self) -> None:
        self.others: list = []
        self.gathers = 0

    def tp_allgather(self, item, small_payload: bool = False) -> list:
        self.gathers += 1
        return [item] + list(self.others)


class _Env:
    """One test's stubs."""

    def __init__(
        self, monkeypatch, cached=None, tuning=False, active=True, split_k_tactics=None, splitk=None
    ) -> None:
        self.log = _Log()
        self.splitk = _SplitKModule() if splitk is None else splitk
        self.group = _Group()
        self.dispatcher = _Dispatcher() if split_k_tactics is None else _Dispatcher(split_k_tactics)
        self.tuner = _Tuner(self.dispatcher, cached, tuning)
        tuner = self.tuner
        group = self.group
        low_m = _module(
            "low_m_gemm",
            _DISPATCHER=_Dispatcher(),
            _AUTOTUNER_OP_NAME="trtllm::low_m_bf16_gemm",
            _should_apply_low_m_gemm=lambda x: active,
        )
        splitk_module = _module(
            "dense_bf16_gemm_sm100_splitk",
            SplitKTactic=self.splitk.SplitKTactic,
            validate_tactic=self.splitk.validate_tactic,
        )
        stubs = {
            "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk": splitk_module,
            "tensorrt_llm._torch.autotuner": _module(
                "autotuner", AutoTuner=types.SimpleNamespace(get=lambda: tuner)
            ),
            "tensorrt_llm._torch.modules": _module("modules", low_m_gemm=low_m),
            "tensorrt_llm._torch.modules.low_m_gemm": low_m,
            "tensorrt_llm.logger": _module("logger", logger=self.log),
            f"{_PACKAGE}.dispatch": _module(
                "dispatch", Distributed=types.SimpleNamespace(get=lambda mapping: group)
            ),
            f"{_PACKAGE}.kernels.split_sampler": _module("split_sampler"),
            f"{_PACKAGE}.kernels.split_sampler.target_head": _module(
                "target_head", context_kv_head_fork=lambda lm_head: (True, None)
            ),
        }
        for name, module in stubs.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(
            sys.modules[_PACKAGE], "dispatch", stubs[f"{_PACKAGE}.dispatch"], raising=False
        )
        monkeypatch.setattr(
            sys.modules[f"{_PACKAGE}.kernels"],
            "split_sampler",
            stubs[f"{_PACKAGE}.kernels.split_sampler"],
            raising=False,
        )
        stubs[f"{_PACKAGE}.kernels.split_sampler"].target_head = stubs[
            f"{_PACKAGE}.kernels.split_sampler.target_head"
        ]

    def head(self, world: int = 1, rs_areas: bool = False):
        """An ``S2Head`` on a ``[SHARD, HIDDEN]`` LM head and its LM-head stand-in."""
        generator = torch.Generator().manual_seed(world)
        weight = torch.randn(SHARD, HIDDEN, generator=generator).to(torch.bfloat16)
        lm_head = types.SimpleNamespace(
            weight=weight, _low_m_gemm_dispatcher=self.dispatcher, use_custom_cublas_mm=False
        )
        copy = torch.randn(SHARD, HIDDEN, generator=generator).to(torch.bfloat16)
        mapping = types.SimpleNamespace(tp_size=world, tp_rank=0)
        states = types.SimpleNamespace(rs_areas=rs_areas)
        return S2.S2Head(mapping, lm_head, copy, states, 0.0), lm_head


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _load():
    """``sampler.py`` as ``<package>.kernels.s2_sampler.sampler`` (its relative imports resolve in the package)."""
    for name, path in (
        (_PACKAGE, _TARGET),
        (f"{_PACKAGE}.kernels", _TARGET / "kernels"),
        (f"{_PACKAGE}.kernels.s2_sampler", _TARGET / "kernels" / "s2_sampler"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    name = f"{_PACKAGE}.kernels.s2_sampler.sampler"
    spec = importlib.util.spec_from_file_location(name, _TARGET / "kernels/s2_sampler/sampler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S2 = _load()


def _rows(n: int = ROWS) -> torch.Tensor:
    return torch.randn(n, HIDDEN, generator=torch.Generator().manual_seed(n)).to(torch.bfloat16)


def _cached(env: _Env, kind: str, tactic):
    runner = {
        "cublas": env.dispatcher._cublas_no_bias,
        "direct": env.dispatcher._direct,
        "split_k": env.dispatcher._runner_no_bias,
    }[kind]
    return runner, tactic


# ---- the pinned tactic ---------------------------------------------------------------------------------------------


def test_pinned_tactic_is_a_bit_equivalent_split_k_tactic():
    assert S2.PINNED_TACTIC in S2.BIT_EQUIVALENT_TACTICS
    assert len(S2.PINNED_TACTIC) == 4 and S2.PINNED_TACTIC[2] == 1


@pytest.mark.parametrize(
    "kind, tactic",
    [("cublas", -1), ("direct", 3)]
    + [("split_k", t) for t in SPLIT_K_OUTSIDE + SPLIT_K_INSIDE]
    + [(None, None)],
    ids=lambda v: str(v),
)
def test_s2_gemm_runs_the_pinned_tactic_whatever_the_autotuner_cached(monkeypatch, kind, tactic):
    env = _Env(monkeypatch)
    if kind is not None:
        env.tuner.cached = _cached(env, kind, tactic)
    head, lm_head = env.head()
    x = _rows()
    assert head.takes(x, lm_head)
    assert head.armed is True
    logits = head.produce(lm_head, x)
    split_k = env.dispatcher._runner_no_bias
    assert split_k.calls == [S2.PINNED_TACTIC]
    assert env.dispatcher._cublas_no_bias.calls == [] and env.dispatcher._direct.calls == []
    # The GEMM is on the s2 head's copy of the weight, into its local shard; the pending row is its scratch.
    assert torch.equal(head.local, x @ head.weight.t())
    assert logits is head.scratch
    assert env.tuner.asked == 0
    assert env.log.messages("info") and str(S2.PINNED_TACTIC) in env.log.messages("info")[0]


def test_every_s2_call_after_arming_runs_the_pinned_tactic(monkeypatch):
    env = _Env(monkeypatch, cached=None)
    head, lm_head = env.head()
    for step in range(3):
        env.tuner.cached = _cached(env, "split_k", SPLIT_K_OUTSIDE[step % len(SPLIT_K_OUTSIDE)])
        assert head.takes(_rows(), lm_head)
        head.produce(lm_head, _rows())
    assert env.dispatcher._runner_no_bias.calls == [S2.PINNED_TACTIC] * 3
    assert env.group.gathers == 0 and head.calls == 3 and env.tuner.asked == 0


# ---- the arming decision -------------------------------------------------------------------------------------------


def test_arming_waits_only_while_the_autotuner_tunes(monkeypatch):
    env = _Env(monkeypatch, tuning=True)
    head, lm_head = env.head()
    assert not head.takes(_rows(), lm_head)
    assert head.armed is None
    env.tuner.is_tuning_mode = False
    # No cached entry for the shape: the decision does not wait for one.
    assert head.takes(_rows(), lm_head)
    assert head.armed is True
    assert env.tuner.asked == 0
    assert env.splitk.validated == [(S2.PINNED_TACTIC, ROWS, SHARD, HIDDEN)]
    assert env.dispatcher._runner_no_bias.queried == []


def test_every_rank_arms_from_the_same_agreed_verdicts(monkeypatch):
    env = _Env(monkeypatch, cached=None)
    env.group.others = ["ok"] * 7
    head, lm_head = env.head(world=8)
    assert head.takes(_rows(), lm_head)
    assert head.takes(_rows(), lm_head)
    assert head.armed is True and env.group.gathers == 1


@pytest.mark.parametrize("waiting_rank", [1, 7])
def test_a_rank_still_tuning_keeps_every_rank_waiting(monkeypatch, waiting_rank):
    env = _Env(monkeypatch)
    env.group.others = ["ok"] * 7
    env.group.others[waiting_rank - 1] = "wait"
    head, lm_head = env.head(world=8)
    assert not head.takes(_rows(), lm_head)
    assert head.armed is None
    env.group.others = ["ok"] * 7
    assert head.takes(_rows(), lm_head)


def test_one_refusal_disarms_every_rank_once(monkeypatch):
    env = _Env(monkeypatch)
    env.group.others = ["ok"] * 6 + ["a refusal on rank 7"]
    head, lm_head = env.head(world=8)
    assert not head.takes(_rows(), lm_head)
    assert head.armed is False
    env.group.others = ["ok"] * 7
    assert not head.takes(_rows(), lm_head)
    assert env.group.gathers == 1
    assert env.dispatcher._runner_no_bias.calls == []
    (warning,) = env.log.messages("warning")
    assert "not armed" in warning and "a refusal on rank 7" in warning
    assert env.log.messages("error") == []


#: The autotune list of the served split-K module for the M = 8 LM-head shape, in kind: with the oversized shared
#: memory its long-K stage window is near a deeper cap, so no tactic at the pinned stage count is listed.
OVERSIZED_WINDOW = tuple(
    (mma_m, mma_n, 1, stages)
    for mma_m in (64, 128)
    for mma_n in (8, 16, 32)
    for stages in (16, 17, 18)
)


def test_the_pin_arms_when_the_served_autotune_window_excludes_its_stage_count(monkeypatch):
    env = _Env(monkeypatch, split_k_tactics=OVERSIZED_WINDOW, splitk=_SplitKModule(max_stages=20))
    assert S2.PINNED_TACTIC not in OVERSIZED_WINDOW
    head, lm_head = env.head(rs_areas=True)
    x = _rows()
    assert head.takes(x, lm_head)
    assert head.armed is True
    assert env.splitk.validated == [(S2.PINNED_TACTIC, ROWS, SHARD, HIDDEN)]
    assert env.dispatcher._runner_no_bias.queried == []
    head.produce(lm_head, x)
    assert env.dispatcher._runner_no_bias.calls == [S2.PINNED_TACTIC]
    assert env.log.messages("error") == [] and env.log.messages("warning") == []


def test_a_tactic_the_split_k_module_rejects_is_refused_with_its_reason(monkeypatch):
    # Listed by the runner, but the module's stage cap is below the pinned stage count: validation decides.
    env = _Env(monkeypatch, split_k_tactics=SPLIT_K_INSIDE, splitk=_SplitKModule(max_stages=4))
    head, lm_head = env.head()
    assert not head.takes(_rows(), lm_head)
    assert head.armed is False
    assert env.dispatcher._runner_no_bias.calls == []
    (warning,) = env.log.messages("warning")
    assert str(S2.PINNED_TACTIC) in warning and "cannot run" in warning
    assert "ab_stages must be in [1, 4], got 5" in warning


def test_a_shape_the_pinned_tactic_does_not_serve_is_refused(monkeypatch):
    env = _Env(monkeypatch, splitk=_SplitKModule(max_m=4))
    head, lm_head = env.head()
    assert not head.takes(_rows(), lm_head)
    assert head.armed is False
    (warning,) = env.log.messages("warning")
    assert str(S2.PINNED_TACTIC) in warning and f"M = {ROWS}" in warning


def test_a_call_off_the_low_m_dispatcher_is_refused_on_every_rank(monkeypatch):
    env = _Env(monkeypatch, active=False)
    env.group.others = ["ok"] * 7
    head, lm_head = env.head(world=8)
    assert not head.takes(_rows(), lm_head)
    assert head.armed is False and env.group.gathers == 1
    (warning,) = env.log.messages("warning")
    assert "RuntimeError" in warning and "low-M dispatcher" in warning


def test_a_refusal_with_rejection_sampling_areas_is_an_error(monkeypatch):
    env = _Env(monkeypatch, splitk=_SplitKModule(max_stages=4))
    head, lm_head = env.head(rs_areas=True)
    assert not head.takes(_rows(), lm_head)
    (error,) = env.log.messages("error")
    assert "not armed" in error and "rejection_sampler" in error and "stock rejection path" in error
    assert env.log.messages("warning") == []


def test_arming_with_rejection_sampling_areas_logs_no_error(monkeypatch):
    env = _Env(monkeypatch)
    head, lm_head = env.head(rs_areas=True)
    assert head.takes(_rows(), lm_head)
    assert env.log.messages("error") == [] and env.log.messages("warning") == []


# ---- the stock head ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(7, HIDDEN), (16, HIDDEN), (1, ROWS, HIDDEN)])
def test_other_calls_stay_on_the_stock_head_without_a_decision(monkeypatch, shape):
    env = _Env(monkeypatch)
    head, lm_head = env.head(world=8)
    x = torch.zeros(shape, dtype=torch.bfloat16)
    assert not head.takes(x, lm_head)
    assert head.armed is None and env.group.gathers == 0
    assert env.dispatcher._runner_no_bias.queried == [] and env.splitk.validated == []


def test_s2_leaves_the_autotuner_and_the_dispatcher_alone(monkeypatch):
    env = _Env(monkeypatch)
    env.tuner.cached = _cached(env, "cublas", -1)
    head, lm_head = env.head()
    buffers = dict(env.dispatcher._output_buffers)
    runners = (
        env.dispatcher._runner_no_bias,
        env.dispatcher._cublas_no_bias,
        env.dispatcher._direct,
    )
    assert head.takes(_rows(), lm_head)
    head.produce(lm_head, _rows())
    assert env.tuner.asked == 0 and env.tuner.cached == (env.dispatcher._cublas_no_bias, -1)
    assert env.dispatcher._output_buffers == buffers
    assert (
        env.dispatcher._runner_no_bias,
        env.dispatcher._cublas_no_bias,
        env.dispatcher._direct,
    ) == runners
    assert lm_head.weight is not head.weight
