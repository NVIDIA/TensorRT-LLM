# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PyExecutor construction tests for DWDP: manager ownership and the dense DWDP launch gate."""

import os
import re
from contextlib import contextmanager
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.modules.dwdp import vmm
from tensorrt_llm._torch.modules.dwdp.dense import config as dense_config
from tensorrt_llm._torch.modules.dwdp.dense.config import (
    DENSE_ENV,
    DENSE_MIN_BYTES_ENV,
    DENSE_SEGMENTS_ENV,
    DENSE_STRICT_ENV,
)
from tensorrt_llm._torch.modules.dwdp.dense.gate import validate_dense_gate
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager, set_global_dwdp_manager
from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    DwdpConfig,
    MTPDecodingConfig,
    PrefillCudaGraphBackend,
    SleepConfig,
    TorchCompileConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_HANDLE_TYPE = vmm.cuda.CUmemAllocationHandleType


@contextmanager
def _restore_global_dwdp_manager():
    """Keep the process-global manager from leaking into later tests."""
    previous = get_global_dwdp_manager()
    try:
        yield
    finally:
        set_global_dwdp_manager(previous)


def test_create_py_executor_rolls_back_new_dwdp_manager_on_error():
    with _restore_global_dwdp_manager():
        set_global_dwdp_manager(None)
        manager = MagicMock()
        manager.__exit__.side_effect = lambda *_args: set_global_dwdp_manager(None)

        def fail_after_registration(**_kwargs):
            set_global_dwdp_manager(manager)
            raise RuntimeError("construction failed")

        with patch.object(
            py_executor_creator,
            "_create_py_executor_impl",
            side_effect=fail_after_registration,
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                py_executor_creator.create_py_executor(llm_args=MagicMock())

        manager.__exit__.assert_called_once_with(None, None, None)
        assert get_global_dwdp_manager() is None


def test_create_py_executor_leaves_a_pre_existing_dwdp_manager_alone():
    """The wrapper only tears down a manager it created, not one it inherited."""
    with _restore_global_dwdp_manager():
        existing = MagicMock()
        set_global_dwdp_manager(existing)

        def fail_without_registering(**_kwargs):
            raise RuntimeError("construction failed")

        with patch.object(
            py_executor_creator,
            "_create_py_executor_impl",
            side_effect=fail_without_registering,
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                py_executor_creator.create_py_executor(llm_args=MagicMock())

        existing.__exit__.assert_not_called()
        assert get_global_dwdp_manager() is existing


# ----------------------------------------------------------------------
# Dense DWDP launch gate
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_dense_switches(monkeypatch):
    """Start every test without ``TRTLLM_DWDP_DENSE*`` variables; tests set the ones they need."""
    for name in [name for name in os.environ if name.startswith(DENSE_ENV)]:
        monkeypatch.delenv(name)


@pytest.fixture
def fabric_host(monkeypatch):
    """Make this host share FABRIC handles, whatever its architecture."""
    monkeypatch.setattr(vmm, "peer_handle_type", lambda: _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC)


def _dense_launch(**overrides) -> TorchLlmArgs:
    """A dense DWDP context-worker launch, with ``overrides`` applied without validation."""
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        dwdp_config=DwdpConfig(
            dwdp_size=16, num_groups=1, num_experts_per_worker=56, num_prefetch_experts=56
        ),
    )
    return args.model_copy(update=overrides)


def _mapping(tp_size: int = 1, pp_size: int = 1, cp_size: int = 1) -> Mapping:
    return Mapping(
        world_size=tp_size * pp_size * cp_size,
        rank=0,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
    )


class _Violation(NamedTuple):
    """A launch that violates exactly one requirement of the dense DWDP gate."""

    name: str
    requirement: str
    """What the error must say the launch requires."""
    overrides: dict
    """``llm_args`` fields that differ from the accepted launch."""
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    posix_handles: bool = False
    segments: str | None = None
    """``TRTLLM_DWDP_DENSE_SEGMENTS``; None leaves it unset."""


#: One violation per requirement, in the order the gate checks them.
_VIOLATIONS = [
    _Violation("switches", "valid TRTLLM_DWDP_DENSE_* switches", {}, segments="atn"),
    _Violation("dwdp_config", "dwdp_config with dwdp_size > 1", {"dwdp_config": None}),
    _Violation(
        "cuda_graph_config", "cuda_graph_config=None", {"cuda_graph_config": CudaGraphConfig()}
    ),
    _Violation(
        "torch_compile_config",
        "torch_compile_config=None",
        {"torch_compile_config": TorchCompileConfig()},
    ),
    _Violation(
        "prefill_cuda_graph_backend",
        "prefill_cuda_graph_backend='disabled'",
        {"prefill_cuda_graph_backend": PrefillCudaGraphBackend.BREAKABLE},
    ),
    _Violation("sleep_config", "sleep_config=None", {"sleep_config": SleepConfig()}),
    _Violation("lora_config", "lora_config=None", {"lora_config": LoraConfig()}),
    _Violation(
        "speculative_config",
        "speculative_config=None",
        {"speculative_config": MTPDecodingConfig(max_draft_len=1)},
    ),
    _Violation("enable_ugpu", "enable_ugpu=False", {"enable_ugpu": True}),
    _Violation(
        "disable_overlap_scheduler",
        "disable_overlap_scheduler=True",
        {"disable_overlap_scheduler": False},
    ),
    _Violation("tp_size", "tp_size=1", {}, tp_size=2),
    _Violation("pp_size", "pp_size=1", {}, pp_size=2),
    _Violation("cp_size", "cp_size=1", {}, cp_size=2),
    _Violation("peer_handle_type", "FABRIC peer handles", {}, posix_handles=True),
]


def _check(
    monkeypatch,
    overrides: dict,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    posix_handles: bool = False,
    segments: str | None = None,
) -> None:
    """Run the gate on the accepted launch changed by ``overrides``, the sizes, handles, switches."""
    handle_type = (
        _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        if posix_handles
        else _HANDLE_TYPE.CU_MEM_HANDLE_TYPE_FABRIC
    )
    monkeypatch.setattr(vmm, "peer_handle_type", lambda: handle_type)
    if segments is None:
        monkeypatch.delenv(DENSE_SEGMENTS_ENV, raising=False)
    else:
        monkeypatch.setenv(DENSE_SEGMENTS_ENV, segments)
    validate_dense_gate(_dense_launch(**overrides), _mapping(tp_size, pp_size, cp_size))


def test_dense_gate_accepts_a_dense_dwdp_context_worker(fabric_host):
    validate_dense_gate(_dense_launch(), _mapping())


@pytest.mark.parametrize("violation", _VIOLATIONS, ids=[v.name for v in _VIOLATIONS])
def test_dense_gate_names_the_violated_requirement(monkeypatch, violation):
    with pytest.raises(ValueError, match=re.escape(f"requires {violation.requirement}")) as error:
        _check(
            monkeypatch,
            violation.overrides,
            violation.tp_size,
            violation.pp_size,
            violation.cp_size,
            violation.posix_handles,
            violation.segments,
        )
    assert DENSE_ENV in str(error.value)


@pytest.mark.parametrize(
    "dwdp_config, actual",
    [(None, "dwdp_config is not set"), (DwdpConfig(dwdp_size=1), "dwdp_size=1")],
)
def test_dense_gate_requires_a_dwdp_group(fabric_host, dwdp_config, actual):
    with pytest.raises(ValueError, match=re.escape(f"dwdp_size > 1, but {actual}")):
        validate_dense_gate(_dense_launch(dwdp_config=dwdp_config), _mapping())


def test_dense_gate_names_the_peer_handle_type_it_found(monkeypatch):
    with pytest.raises(ValueError, match="CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR"):
        _check(monkeypatch, {}, posix_handles=True)


def test_dense_gate_reports_the_first_violation(monkeypatch):
    """With every requirement from the k-th on violated, the error names the k-th."""
    for first, violation in enumerate(_VIOLATIONS):
        overrides, posix_handles, segments = {}, False, None
        tp_size = pp_size = cp_size = 1
        for later in _VIOLATIONS[first:]:
            overrides.update(later.overrides)
            tp_size = max(tp_size, later.tp_size)
            pp_size = max(pp_size, later.pp_size)
            cp_size = max(cp_size, later.cp_size)
            posix_handles = posix_handles or later.posix_handles
            segments = later.segments if later.segments is not None else segments
        with pytest.raises(ValueError, match=re.escape(f"requires {violation.requirement}")):
            _check(monkeypatch, overrides, tp_size, pp_size, cp_size, posix_handles, segments)


@pytest.mark.parametrize("segments", ["atn", "attention", "moe", "attn,mlp", "attn;moe", ","])
def test_dense_gate_rejects_a_bad_segment_list_before_the_load(monkeypatch, segments):
    """The one switch whose bad values raise is read at the gate, before the weights load."""
    requirement = re.escape("requires valid TRTLLM_DWDP_DENSE_* switches")
    with pytest.raises(ValueError, match=requirement) as error:
        _check(monkeypatch, {}, segments=segments)
    assert f"{DENSE_SEGMENTS_ENV}={segments!r}" in str(error.value)


@pytest.mark.parametrize("segments", ["attn", "attn,moe", " MOE , attn ", "attn,attn", ""])
def test_dense_gate_accepts_every_valid_segment_list(monkeypatch, segments):
    _check(monkeypatch, {}, segments=segments)


def test_dense_gate_reads_the_switches_it_ignores_without_failing(monkeypatch, fabric_host):
    """Bad booleans and integers keep their defaults; the gate already logs their warnings."""
    warnings = []
    monkeypatch.setattr(dense_config, "logger", SimpleNamespace(warning=warnings.append))
    monkeypatch.setenv(DENSE_STRICT_ENV, "maybe")
    monkeypatch.setenv(DENSE_MIN_BYTES_ENV, "1MB")

    validate_dense_gate(_dense_launch(), _mapping())

    assert len(warnings) == 2
    assert any(f"{DENSE_STRICT_ENV}='maybe'" in message for message in warnings)
    assert any(f"{DENSE_MIN_BYTES_ENV}='1MB'" in message for message in warnings)


@pytest.mark.parametrize("value", [None, "0", "false", "off"])
def test_dense_gate_is_not_consulted_when_dense_is_not_requested(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(DENSE_ENV, raising=False)
    else:
        monkeypatch.setenv(DENSE_ENV, value)
    gate = MagicMock()
    monkeypatch.setattr(py_executor_creator, "validate_dense_gate", gate)

    py_executor_creator._check_dense_dwdp_launch(_dense_launch(), _mapping())

    gate.assert_not_called()


def test_unsupported_launch_passes_when_dense_is_not_requested(monkeypatch):
    """Without TRTLLM_DWDP_DENSE the launch is the DWDP checks' business only."""
    monkeypatch.delenv(DENSE_ENV, raising=False)
    unsupported = TorchLlmArgs(model="/tmp/dummy_model", skip_tokenizer_init=True)

    py_executor_creator._check_dense_dwdp_launch(unsupported, _mapping(tp_size=2))


class _Stop(Exception):
    """Ends ``_create_py_executor_impl`` at the DWDP manager or at the first step after the DWDP block."""


def _executor_llm_args(**overrides) -> SimpleNamespace:
    """The ``llm_args`` fields ``_create_py_executor_impl`` reads up to the DWDP manager."""
    fields = dict(
        garbage_collection_gen0_threshold=0,
        lora_config=None,
        kv_connector_config=None,
        scheduler_config=SimpleNamespace(),
        peft_cache_config=None,
        kv_cache_config=SimpleNamespace(
            enable_block_reuse=False, enable_partial_reuse=False, tokens_per_block=32
        ),
        decoding_config=None,
        guided_decoding_backend=None,
        mm_encoder_only=False,
        enable_chunked_prefill=False,
        get_runtime_sizes=lambda: (1, 128, 128, 4),
        sparse_attention_config=None,
        attn_backend="TRTLLM",
        speculative_config=None,
        enable_ugpu=False,
        disable_overlap_scheduler=True,
        parallel_config=SimpleNamespace(to_mapping=lambda: None),
        dwdp_config=DwdpConfig(
            dwdp_size=16, num_groups=1, num_experts_per_worker=56, num_prefetch_experts=56
        ),
        sleep_config=None,
        cache_transceiver_config=None,
        cuda_graph_config=None,
        torch_compile_config=None,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _create_executor(monkeypatch, llm_args: SimpleNamespace, log: list[str]) -> None:
    """Run ``_create_py_executor_impl`` until it builds the DWDP manager or the memory monitor.

    Logs the gate, the manager construction and the memory monitor (the first step after the
    DWDP block) into ``log``; the manager and the monitor raise :class:`_Stop`.
    """
    monkeypatch.setattr(
        py_executor_creator,
        "_load_config_and_create_checkpoint_loader",
        lambda args, _checkpoint_dir: (args, None),
    )
    monkeypatch.setattr(
        py_executor_creator, "_get_mapping", lambda _mapping: Mapping(world_size=1, rank=0)
    )
    monkeypatch.setattr(py_executor_creator, "dwdp_global_rank", lambda: 3)
    monkeypatch.setattr(
        py_executor_creator.Distributed, "get", staticmethod(lambda _mapping: MagicMock())
    )

    def gate(args, mapping):
        log.append("validate_dense_gate")
        validate_dense_gate(args, mapping)

    def stop(step):
        def construct(*_args, **_kwargs):
            log.append(step)
            raise _Stop(step)

        return construct

    monkeypatch.setattr(py_executor_creator, "validate_dense_gate", gate)
    monkeypatch.setattr(py_executor_creator, "DwdpManager", stop("DwdpManager"))
    monkeypatch.setattr(py_executor_creator, "_ExecutorMemoryMonitor", stop("memory monitor"))
    py_executor_creator._create_py_executor_impl(llm_args=llm_args)


def test_executor_creation_runs_the_dense_gate_before_the_dwdp_manager(monkeypatch, fabric_host):
    monkeypatch.setenv(DENSE_ENV, "1")
    log = []

    with pytest.raises(_Stop):
        _create_executor(monkeypatch, _executor_llm_args(), log)

    assert log == ["validate_dense_gate", "DwdpManager"]


@pytest.mark.parametrize(
    "overrides, requirement",
    [
        ({"cuda_graph_config": object()}, "cuda_graph_config=None"),
        ({"dwdp_config": None}, "dwdp_config with dwdp_size > 1"),
    ],
    ids=["cuda_graph_config", "no_dwdp_config"],
)
def test_executor_creation_rejects_a_dense_launch_before_the_dwdp_manager(
    monkeypatch, fabric_host, overrides, requirement
):
    monkeypatch.setenv(DENSE_ENV, "1")
    log = []

    with pytest.raises(ValueError, match=re.escape(f"requires {requirement}")):
        _create_executor(monkeypatch, _executor_llm_args(**overrides), log)

    assert log == ["validate_dense_gate"]


def test_executor_creation_rejects_a_bad_segment_list_before_the_dwdp_manager(
    monkeypatch, fabric_host
):
    monkeypatch.setenv(DENSE_ENV, "1")
    monkeypatch.setenv(DENSE_SEGMENTS_ENV, "atn")
    log = []

    with pytest.raises(ValueError, match=re.escape(f"{DENSE_SEGMENTS_ENV}='atn'")):
        _create_executor(monkeypatch, _executor_llm_args(), log)

    assert log == ["validate_dense_gate"]


def test_executor_creation_ignores_a_bad_segment_list_when_dense_is_off(monkeypatch):
    """Dense off: nothing reads ``TRTLLM_DWDP_DENSE_SEGMENTS``, so a bad value breaks nothing."""
    monkeypatch.setenv(DENSE_SEGMENTS_ENV, "bogus")
    log = []

    with pytest.raises(_Stop):
        _create_executor(monkeypatch, _executor_llm_args(), log)

    assert log == ["DwdpManager"]


@pytest.mark.parametrize("with_dwdp", [False, True], ids=["no_dwdp", "dwdp"])
def test_executor_creation_skips_the_dense_gate_when_not_requested(monkeypatch, with_dwdp):
    monkeypatch.delenv(DENSE_ENV, raising=False)
    overrides = {"cuda_graph_config": object()}
    if not with_dwdp:
        overrides["dwdp_config"] = None
    log = []

    with pytest.raises(_Stop):
        _create_executor(monkeypatch, _executor_llm_args(**overrides), log)

    assert log == ["DwdpManager" if with_dwdp else "memory monitor"]


def test_dense_gate_runs_when_dense_is_requested(monkeypatch, fabric_host):
    monkeypatch.setenv(DENSE_ENV, "1")
    args, mapping = _dense_launch(), _mapping()
    gate = MagicMock()
    monkeypatch.setattr(py_executor_creator, "validate_dense_gate", gate)

    py_executor_creator._check_dense_dwdp_launch(args, mapping)

    gate.assert_called_once_with(args, mapping)
    monkeypatch.setattr(py_executor_creator, "validate_dense_gate", validate_dense_gate)
    with pytest.raises(ValueError, match=re.escape("requires tp_size=1")):
        py_executor_creator._check_dense_dwdp_launch(args, _mapping(tp_size=2))
