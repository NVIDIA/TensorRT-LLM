# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode-step shell (``ull_shell.py``).

* :func:`ull_shell.adopt`: the model's ``ull_step_shell`` takes precedence over the
  environment switches below, which remain the shell of a model without one.
* ``TRTLLM_K3_ULL_PIN_EXECUTOR``: CPU choice and the thread affinities it sets.
* ``TRTLLM_K3_ULL_GIL_SWITCH_US``: the interpreter's thread switch interval.
* ``TRTLLM_K3_ULL_GRAPH_INPUT_ALIAS`` / ``TRTLLM_K3_ULL_SAMPLE_EXEC_STREAM``: the executor's
  default; setting them only logs that.
"""

import os
import sys
import threading
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import ull_shell
from tensorrt_llm._torch.pyexecutor.ull_shell import ExecutorThreadPin, StepShell, pin_executor_cpu


@pytest.fixture(autouse=True)
def _no_model_shell(monkeypatch):
    """Every test starts without a model shell (the environment switches apply)."""
    monkeypatch.setattr(ull_shell, "_SHELL", None)


# ---------------------------------------------------------------------------
# Adopting the model's shell
# ---------------------------------------------------------------------------


def test_adopt_takes_the_shell_a_model_asks_for(monkeypatch):
    from torch import nn

    fold = ull_shell.StepFold(False)
    monkeypatch.setattr(ull_shell, "STEP_FOLD", fold)
    monkeypatch.setattr(ull_shell, "_STEP_PREP_FROM_ENV", False)
    monkeypatch.delenv(ull_shell.STEP_FOLD_VERIFY_ENV, raising=False)

    class Plain(nn.Module):
        pass

    class Ull(nn.Module):
        ull_step_shell = StepShell(
            step_prep=True, step_fold=True, gil_switch_us=200, verify_commits=7
        )

    assert ull_shell.adopt(Plain()) is None and not fold.enabled
    assert not ull_shell.step_prep_enabled() and ull_shell.default_verify_commits() == 0
    wrapper = Plain()
    wrapper.inner = Ull()
    assert ull_shell.adopt(wrapper) == Ull.ull_step_shell
    assert ull_shell.shell() is Ull.ull_step_shell and ull_shell.step_prep_enabled()
    assert fold.enabled and fold.active and fold.verify == 7
    assert ull_shell.default_verify_commits() == 7
    # A model without a shell (e.g. a second engine) leaves the process's shell as it is.
    assert ull_shell.adopt(Plain()) is Ull.ull_step_shell and fold.enabled
    fold.suppress(True)
    assert fold.enabled and not fold.active
    fold.suppress(False)
    assert fold.active
    with pytest.raises(ValueError):
        StepShell(gil_switch_us=-5)


def test_shell_log_line_matches_k3():
    """Served-run gates grep the K3 line; verify_commits stays out of it."""
    shell = StepShell(
        step_prep=True, step_fold=True, pin_executor=True, gil_switch_us=200, verify_commits=200
    )
    line = f"ULL step shell: {shell}"
    assert line == (
        "ULL step shell: StepShell(step_prep=True, step_fold=True, pin_executor=True, "
        "gil_switch_us=200)"
    )


def test_shell_requests_the_fused_spec_head(monkeypatch):
    """``spec_head_fusion`` turns the fused decode-step head on like ``TLLM_SPEC_HEAD_FUSION=1``,
    stays out of the log line, and without a shell only the environment decides."""
    from tensorrt_llm._torch.speculative import spec_head_kernels

    monkeypatch.delenv(spec_head_kernels.SPEC_HEAD_FUSION_ENV_VAR, raising=False)
    assert not spec_head_kernels.spec_head_fusion_enabled()
    monkeypatch.setattr(ull_shell, "_SHELL", StepShell(step_prep=True))
    assert not spec_head_kernels.spec_head_fusion_enabled()
    shell = StepShell(step_prep=True, spec_head_fusion=True)
    monkeypatch.setattr(ull_shell, "_SHELL", shell)
    assert spec_head_kernels.spec_head_fusion_enabled()
    assert str(shell) == str(StepShell(step_prep=True))
    monkeypatch.setattr(ull_shell, "_SHELL", None)
    monkeypatch.setenv(spec_head_kernels.SPEC_HEAD_FUSION_ENV_VAR, "1")
    assert spec_head_kernels.spec_head_fusion_enabled()


def test_model_shell_overrides_environment(monkeypatch):
    original = sys.getswitchinterval()
    monkeypatch.setenv(ull_shell.PIN_EXECUTOR_ENV, "auto")
    monkeypatch.setenv(ull_shell.GIL_SWITCH_US_ENV, "300")
    monkeypatch.setattr(ull_shell, "_SHELL", StepShell(pin_executor=False, gil_switch_us=0))
    monkeypatch.setattr(ull_shell, "_smt_siblings", lambda cpu: {cpu})
    try:
        assert pin_executor_cpu({0, 1, 2}, 0) is None
        assert ull_shell.apply_gil_switch_interval() is None
        assert sys.getswitchinterval() == original
        monkeypatch.setattr(ull_shell, "_SHELL", StepShell(pin_executor=True, gil_switch_us=200))
        assert pin_executor_cpu({0, 1, 2}, 0) == 1
        assert ull_shell.apply_gil_switch_interval() == pytest.approx(200e-6, rel=0.05)
    finally:
        sys.setswitchinterval(original)


# ---------------------------------------------------------------------------
# Executor thread pinning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,available,gpu,expected",
    [
        ("", {0, 1, 2}, 0, None),
        ("0", {0, 1, 2}, 0, None),
        ("auto", {0, 1, 2}, 0, 1),
        ("auto", {4, 5, 6}, 0, 4),
        ("auto", {0}, 0, 0),
        # Two GPUs sharing one NUMA-wide set get different cores.
        ("auto", set(range(8)), 0, 1),
        ("auto", set(range(8)), 1, 2),
        # GPUs 2 / 3 on the second socket's set.
        ("auto", set(range(8, 16)), 2, 10),
        ("auto", set(range(8, 16)), 3, 11),
    ],
)
def test_pin_executor_cpu_choice(value, available, gpu, expected, monkeypatch):
    monkeypatch.setenv(ull_shell.PIN_EXECUTOR_ENV, value)
    monkeypatch.setattr(ull_shell, "_smt_siblings", lambda cpu: {cpu})
    assert pin_executor_cpu(available, gpu) == expected


def test_pin_executor_cpu_skips_smt_siblings(monkeypatch):
    # CPUs 0-3 are cores, 4-7 their SMT siblings (0/4, 1/5, ...).
    monkeypatch.setenv(ull_shell.PIN_EXECUTOR_ENV, "auto")
    monkeypatch.setattr(ull_shell, "_smt_siblings", lambda cpu: {cpu % 4, cpu % 4 + 4})
    assert [pin_executor_cpu(set(range(8)), g) for g in range(4)] == [1, 2, 3, 1]


def test_pin_executor_cpu_rejects_other_values(monkeypatch):
    monkeypatch.setenv(ull_shell.PIN_EXECUTOR_ENV, "5")
    with pytest.raises(ValueError):
        pin_executor_cpu({4, 5, 6}, 0)


def test_executor_thread_pin_off_touches_nothing(monkeypatch):
    monkeypatch.delenv(ull_shell.PIN_EXECUTOR_ENV, raising=False)

    def fail(*args):
        raise AssertionError("affinity queried with the switch off")

    monkeypatch.setattr(os, "sched_getaffinity", fail)
    monkeypatch.setattr(os, "sched_setaffinity", fail)
    assert ExecutorThreadPin.create() is None


def test_gil_switch_interval(monkeypatch):
    original = sys.getswitchinterval()
    try:
        monkeypatch.delenv(ull_shell.GIL_SWITCH_US_ENV, raising=False)
        assert ull_shell.apply_gil_switch_interval() is None
        assert sys.getswitchinterval() == original
        monkeypatch.setenv(ull_shell.GIL_SWITCH_US_ENV, "200")
        assert ull_shell.apply_gil_switch_interval() == pytest.approx(200e-6, rel=0.05)
        monkeypatch.setenv(ull_shell.GIL_SWITCH_US_ENV, "-5")
        with pytest.raises(ValueError):
            ull_shell.apply_gil_switch_interval()
    finally:
        sys.setswitchinterval(original)


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 4, reason="needs at least 4 CPUs")
def test_executor_thread_pin_affinities(monkeypatch):
    monkeypatch.setenv(ull_shell.PIN_EXECUTOR_ENV, "auto")
    process_cpus = os.sched_getaffinity(0)
    original = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            original[int(tid)] = os.sched_getaffinity(int(tid))
        except OSError:
            pass
    stop = threading.Event()
    bystander_ready = threading.Event()
    bystander_tid = []

    def bystander():
        bystander_tid.append(threading.get_native_id())
        bystander_ready.set()
        stop.wait(30)

    late_tid = []
    result = {}

    def executor():
        pin = ExecutorThreadPin.create()
        result["pin"] = pin
        result["own"] = os.sched_getaffinity(0)
        # A thread started later inherits the executor's single CPU until a refresh.
        late_ready = threading.Event()

        def late():
            late_tid.append(threading.get_native_id())
            late_ready.set()
            stop.wait(30)

        result["late_thread"] = threading.Thread(target=late, daemon=True)
        result["late_thread"].start()
        late_ready.wait(10)
        result["late_before"] = os.sched_getaffinity(late_tid[0])
        result["moved"] = pin.refresh()
        result["late_after"] = os.sched_getaffinity(late_tid[0])

    by = threading.Thread(target=bystander, daemon=True)
    by.start()
    bystander_ready.wait(10)
    ex = threading.Thread(target=executor)
    try:
        ex.start()
        ex.join(30)
        pin = result["pin"]
        assert pin is not None
        reserved = ull_shell._smt_siblings(pin.cpu)
        assert result["own"] == {pin.cpu}
        assert pin.cpu in process_cpus and pin.cpu != 0 or process_cpus == {0}
        others = os.sched_getaffinity(bystander_tid[0])
        assert not others & reserved and others <= process_cpus
        assert result["late_before"] == {pin.cpu}
        assert result["moved"] >= 1
        assert not result["late_after"] & reserved
    finally:
        stop.set()
        by.join(10)
        if "late_thread" in result:
            result["late_thread"].join(10)
        for tid, cpus in original.items():
            try:
                os.sched_setaffinity(tid, cpus)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Switches that select the executor's default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_default_on_switches_only_log(value, monkeypatch):
    logged = []
    monkeypatch.setattr(ull_shell, "logger", SimpleNamespace(info=logged.append))
    for name in (ull_shell.GRAPH_INPUT_ALIAS_ENV, ull_shell.SAMPLE_EXEC_STREAM_ENV):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    ull_shell.report_default_on_switches()
    if value is None:
        assert logged == []
    else:
        assert len(logged) == 2
        assert all(f"={value}: always on here" in line for line in logged)


@pytest.mark.parametrize(
    "spec,use_mrope,aliased",
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_decoder_graphs_alias_engine_inputs_for_speculative_engines(spec, use_mrope, aliased):
    """``TRTLLM_K3_ULL_GRAPH_INPUT_ALIAS``: the decoder runner gives a one-model speculative
    engine without MRoPE (the Kimi-K3 DSpark GEN) its own input buffers as graph inputs."""
    runner_mod = pytest.importorskip("tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner")
    input_ids, position_ids = object(), object()
    config = SimpleNamespace(
        is_encode_only=False,
        cuda_graph_config=object(),
        cuda_graph_padding_enabled=False,
        cuda_graph_batch_sizes=[1, 2, 3, 4],
        max_cuda_graph_batch_size=4,
        max_beam_width=1,
        spec_config=object() if spec else None,
        max_num_tokens=32,
        use_mrope=use_mrope,
        original_max_total_draft_tokens=7 if spec else 0,
        enable_attention_dp=False,
        max_batch_size=4,
        enable_in_graph_sampling=False,
        is_spec_decode=spec,
    )
    runner = SimpleNamespace(
        _config=config,
        _cuda_graph_mem_pool=None,
        _dynamic_draft_len_mapping=None,
        _mapping=None,
        _dist=None,
        _sparse_attention_config=None,
        _buffers=SimpleNamespace(input_ids_cuda=input_ids, position_ids_cuda=position_ids),
    )
    graph_config = runner_mod.DecoderRunner._cuda_graph_runner_config(runner)
    if aliased:
        assert graph_config.static_input_ids is input_ids
        assert graph_config.static_position_ids is position_ids
    else:
        assert graph_config.static_input_ids is None
        assert graph_config.static_position_ids is None


# ---------------------------------------------------------------------------
# Batched input commit gate (TRTLLM_K3_ULL_STEP_PREP admits FP4 MLA KV caches)
# ---------------------------------------------------------------------------


def _runner_module():
    return pytest.importorskip("tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner")


def _gate(input_commit, **overrides):
    runner_mod = _runner_module()
    spec_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: True), sa_config=None
    )
    runner = SimpleNamespace(
        _input_commit=input_commit,
        _config=SimpleNamespace(
            spec_config=spec_config, use_mrope=False, enable_attention_dp=False
        ),
        _use_beam_search=False,
        _lora=SimpleNamespace(model_config=None, cuda_graph_manager=None),
        _inputs=SimpleNamespace(guided_decoder=None),
        _mapping=SimpleNamespace(cp_config={}),
    )
    for name, value in overrides.items():
        setattr(runner, name, value)

    def gate(requests, md, **kwargs):
        args = dict(
            can_run_graph=True,
            promoted_context_request_ids=frozenset(),
            enable_spec_decode=True,
            is_dummy=False,
            ced_enabled=False,
            spec_tree_manager=None,
        )
        args.update(kwargs)
        return runner_mod.DecoderRunner._input_commit_for_step(runner, requests, md, **args)

    return gate


def _metadata(fp4_mla_state=None, flash_mla=False):
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

    md = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
    md.__dict__["enable_flash_mla"] = flash_mla
    md.__dict__["fp4_mla_state"] = fp4_mla_state
    return md


def _requests(*dummies, num_context=0):
    return SimpleNamespace(
        num_context_requests=num_context,
        num_generation_requests=len(dummies),
        generation_requests=[SimpleNamespace(is_dummy=d) for d in dummies],
    )


@pytest.mark.parametrize("step_prep", [False, True])
@pytest.mark.parametrize("fp4_mla", [False, True])
def test_input_commit_gate_admits_fp4_mla_with_step_prep(step_prep, fp4_mla, monkeypatch):
    runner_mod = _runner_module()
    monkeypatch.setattr(runner_mod, "step_prep_enabled", lambda: step_prep)
    commit = object()
    gate = _gate(commit)
    md = _metadata(fp4_mla_state=object() if fp4_mla else None)
    expected = commit if (not fp4_mla or step_prep) else None
    assert gate(_requests(False, False), md) is expected


def test_input_commit_gate_exclusions(monkeypatch):
    runner_mod = _runner_module()
    from tensorrt_llm.mapping import CpType

    monkeypatch.setattr(runner_mod, "step_prep_enabled", lambda: True)
    commit = object()
    md = _metadata(fp4_mla_state=object())
    batch = _requests(False, False)
    assert _gate(commit)(batch, md) is commit
    assert _gate(None)(batch, md) is None
    # Per-step exclusions.
    gate = _gate(commit)
    assert gate(batch, md, can_run_graph=False) is None
    assert gate(batch, md, is_dummy=True) is None
    assert gate(batch, md, ced_enabled=True) is None
    assert gate(batch, md, enable_spec_decode=False) is None
    assert gate(batch, md, spec_tree_manager=object()) is None
    assert gate(batch, md, promoted_context_request_ids=frozenset({3})) is None
    assert gate(_requests(False, True), md) is None
    assert gate(_requests(False, num_context=1), md) is None
    assert gate(_requests(), md) is None
    assert gate(batch, _metadata(flash_mla=True)) is None
    assert gate(batch, object()) is None
    # Engine-level exclusions.
    assert _gate(commit, _use_beam_search=True)(batch, md) is None
    assert _gate(commit, _inputs=SimpleNamespace(guided_decoder=object()))(batch, md) is None
    lora = SimpleNamespace(model_config=object(), cuda_graph_manager=None)
    assert _gate(commit, _lora=lora)(batch, md) is None
    helix = SimpleNamespace(cp_config={"cp_type": CpType.HELIX})
    assert _gate(commit, _mapping=helix)(batch, md) is commit
    ulysses = SimpleNamespace(cp_config={"cp_type": CpType.ULYSSES})
    assert _gate(commit, _mapping=ulysses)(batch, md) is None

    def config(one_engine=True, sa_config=None, **flags):
        mode = SimpleNamespace(use_one_engine=lambda: one_engine)
        return SimpleNamespace(
            spec_config=SimpleNamespace(spec_dec_mode=mode, sa_config=sa_config),
            use_mrope=flags.get("use_mrope", False),
            enable_attention_dp=flags.get("enable_attention_dp", False),
        )

    assert _gate(commit, _config=config())(batch, md) is commit
    assert _gate(commit, _config=config(one_engine=False))(batch, md) is None
    assert _gate(commit, _config=config(sa_config=object()))(batch, md) is None
    assert _gate(commit, _config=config(use_mrope=True))(batch, md) is None
    assert _gate(commit, _config=config(enable_attention_dp=True))(batch, md) is None


def _dsa_metadata(cls=None):
    pytest.importorskip("tensorrt_llm._torch.attention.backends.sparse.dsa")
    if cls is None:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_metadata import (  # noqa: E501
            RubinUllDSAMetadata as cls,
        )

    md = cls.__new__(cls)
    md.__dict__["enable_flash_mla"] = False
    md.__dict__["fp4_mla_state"] = None
    return md


@pytest.mark.parametrize("step_prep", [False, True])
def test_input_commit_gate_admits_dsa_metadata_with_step_prep(step_prep, monkeypatch):
    """The Kimi K3.1 ULL generation target's DSA metadata (``RubinUllDSAMetadata``) declares
    ``stages_input_commit``; only the shell's step_prep admits it. The engine's DSA metadata does
    not declare it, so every other DSA model keeps the per-copy path."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm.mapping import CpType

    runner_mod = _runner_module()
    monkeypatch.setattr(runner_mod, "step_prep_enabled", lambda: step_prep)
    commit = object()
    helix = SimpleNamespace(cp_config={"cp_type": CpType.HELIX})
    md = _dsa_metadata()
    expected = commit if step_prep else None
    assert _gate(commit, _mapping=helix)(_requests(False), md) is expected

    engine = _dsa_metadata(DSAtrtllmAttentionMetadata)
    assert _gate(commit, _mapping=helix)(_requests(False), engine) is None

    # A subclass that does not declare it itself stays on the per-copy path.
    class Derived(type(md)):
        pass

    derived = Derived.__new__(Derived)
    derived.__dict__.update(md.__dict__)
    assert _gate(commit, _mapping=helix)(_requests(False), derived) is None

    class Other(TrtllmAttentionMetadata):
        pass

    other = Other.__new__(Other)
    other.__dict__.update(md.__dict__)
    assert _gate(commit, _mapping=helix)(_requests(False), other) is None
