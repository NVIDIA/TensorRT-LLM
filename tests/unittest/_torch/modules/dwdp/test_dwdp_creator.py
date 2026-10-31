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
"""PyExecutor construction ownership tests for DWDP."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import NoReturn
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.modules.dwdp import fabric_gather
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    DenseDwdpManager,
    get_global_dense_dwdp_manager,
    set_global_dense_dwdp_manager,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode
from tensorrt_llm._torch.pyexecutor import py_executor_creator
from tensorrt_llm._torch.pyexecutor.dwdp import get_global_dwdp_manager, set_global_dwdp_manager
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantAlgo

__extra_import_path__ = ["../../pyexecutor"]
from test_py_executor_creator_mla_cache_reuse_sync import (  # noqa: E402
    _DummyCalibrator,
    _DummyModelEngine,
    _make_llm_args,
)

pytestmark = pytest.mark.cpu_only


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


@contextmanager
def _restore_global_dense_dwdp_manager() -> Iterator[None]:
    """Keep the process-global dense manager from leaking into later tests."""
    previous = get_global_dense_dwdp_manager()
    try:
        yield
    finally:
        set_global_dense_dwdp_manager(previous)


def test_create_py_executor_rolls_back_a_new_standalone_dense_manager_on_error() -> None:
    """A standalone dense manager registered by the failed construction is exited."""
    with _restore_global_dwdp_manager(), _restore_global_dense_dwdp_manager():
        set_global_dwdp_manager(None)
        set_global_dense_dwdp_manager(None)
        manager = MagicMock()
        manager.__exit__.side_effect = lambda *_args: set_global_dense_dwdp_manager(None)

        def fail_after_registration(**_kwargs: object) -> NoReturn:
            set_global_dense_dwdp_manager(manager)
            raise RuntimeError("construction failed")

        with patch.object(
            py_executor_creator,
            "_create_py_executor_impl",
            side_effect=fail_after_registration,
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                py_executor_creator.create_py_executor(llm_args=MagicMock())

        manager.__exit__.assert_called_once_with(None, None, None)
        assert get_global_dense_dwdp_manager() is None


def test_create_py_executor_leaves_a_pre_existing_dense_manager_alone() -> None:
    """The wrapper only exits a dense manager it created, not one it inherited."""
    with _restore_global_dense_dwdp_manager():
        existing = MagicMock()
        set_global_dense_dwdp_manager(existing)

        def fail_without_registering(**_kwargs: object) -> NoReturn:
            raise RuntimeError("construction failed")

        with patch.object(
            py_executor_creator,
            "_create_py_executor_impl",
            side_effect=fail_without_registering,
        ):
            with pytest.raises(RuntimeError, match="construction failed"):
                py_executor_creator.create_py_executor(llm_args=MagicMock())

        existing.__exit__.assert_not_called()
        assert get_global_dense_dwdp_manager() is existing


# ----------------------------------------------------------------------
# The serving phase gets the DWDP manager
# ----------------------------------------------------------------------
# ``_create_py_executor_impl`` runs to its end over the stand-ins of
# ``test_py_executor_creator_mla_cache_reuse_sync.py``, with stand-in DWDP managers, a stand-in
# dense DWDP gate, which the creator runs only when ``dwdp_config.enable_dense`` asks for it, and a
# PyExecutor built without ``__init__``: its ``release_resources``, which ends the profiling phase
# of the KV-cache estimation, logs the managers it would exit, and its ``rebind_resources`` binds
# those of the serving phase.

#: Creator dependencies replaced by no-ops.
_NOOPS = ("instantiate_sampler", "get_spec_resource_manager", "get_spec_drafter")
_NOOPS += ("_adjust_torch_mem_fraction", "log_memory_usage")


def _standalone_dense_manager(
    mode: DenseMode, *, mapping: Mapping, config: DenseConfig
) -> DenseDwdpManager:
    """A real standalone manager, of an attention-DP instance whatever the creator's stand-in
    mapping."""
    assert mode is DenseMode.STANDALONE
    instance = Mapping(world_size=2, rank=0, tp_size=2, enable_attention_dp=True)
    return DenseDwdpManager(mode, mapping=instance, config=config)


def _create(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    kv_cache: str,
    *,
    real_dense_manager: bool = False,
    copy_mode: str | None = None,
) -> tuple[list, object]:
    """Run ``create_py_executor`` with DWDP in ``mode``, start a forward of the executor it returns.

    Args:
        mode: ``"expert"`` (expert DWDP alone), ``"attached"`` (dense DWDP attached to it) or
            ``"standalone"`` (dense DWDP alone).
        kv_cache: ``"estimated"``, ``"not_estimated"`` or ``"none"`` (a model without one).
        real_dense_manager: Standalone, the creator builds a real ``DenseDwdpManager``, whose
            setup only logs; the model engine logs the SM reserve it is built with.
        copy_mode: Standalone, ``dwdp_config.copy_mode``.

    Returns:
        The call log and the one DWDP manager created.
    """
    log, managers = [], []

    def manager(*_args: object, **_kwargs: object) -> SimpleNamespace:
        """Stands in for ``DwdpManager`` and ``DenseDwdpManager``."""
        managers.append(
            SimpleNamespace(
                __enter__=lambda: None,
                setup=lambda _model: log.append("setup"),
                start_forward=lambda: log.append("start_forward"),
                prefetch_first_layers=lambda: log.append("prefetch_first_layers"),
            )
        )
        return managers[-1]

    def kv_cache_creator(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            try_prepare_estimation=lambda: kv_cache == "estimated",
            build_managers=lambda _resources, estimating: log.append(
                ("build_managers", estimating)
            ),
            # Ends the profiling phase, as KvCacheCreator.configure_kv_cache_capacity does.
            configure_kv_cache_capacity=lambda executor: executor.release_resources(),
            teardown_managers=lambda _resources: None,
        )

    def create_py_executor_instance(**kwargs: object) -> PyExecutor:
        given = kwargs["dwdp_manager"], kwargs["dense_dwdp_manager"]
        log.append(("executor", *given))
        executor = object.__new__(PyExecutor)  # with what the creator and _prime_dwdp read
        executor._metrics, executor.start_worker = {}, lambda: None
        executor.peft_cache_config = kwargs["peft_cache_config"]
        executor.dwdp_manager, executor.dense_dwdp_manager = given

        def release_resources() -> None:
            """Log the managers ``PyExecutor.release_resources`` would exit."""
            log.append(("release", executor.dwdp_manager, executor.dense_dwdp_manager))
            executor.dwdp_manager = executor.dense_dwdp_manager = None

        def rebind_resources(resources: SimpleNamespace) -> None:
            """Bind the serving phase's managers, as ``_initialize_execution`` does."""
            log.append("rebind")
            executor.dwdp_manager = resources.dwdp_manager
            executor.dense_dwdp_manager = resources.dense_dwdp_manager

        executor.release_resources = release_resources
        executor.rebind_resources = rebind_resources
        executor.shutdown = lambda: log.append("shutdown")
        return executor

    def create_py_executor_resources(**kwargs: object) -> SimpleNamespace:
        given = kwargs["dwdp_manager"], kwargs["dense_dwdp_manager"]
        log.append(("serving resources", *given))
        return SimpleNamespace(dwdp_manager=given[0], dense_dwdp_manager=given[1])

    def model_engine(attn_runtime_features: object, **_kwargs: object) -> _DummyModelEngine:
        if real_dense_manager:
            log.append(("model built with an SM reserve of", sm_reserve()))
        engine = _DummyModelEngine(
            attn_runtime_features=attn_runtime_features, kv_cache_quant_algo=QuantAlgo.NO_QUANT
        )
        engine.model.model_config.is_generation = kv_cache != "none"
        return engine

    dense_mode = {"attached": DenseMode.ATTACHED, "standalone": DenseMode.STANDALONE}.get(mode)
    fakes = dict.fromkeys(_NOOPS, lambda *_args, **_kwargs: None)
    fakes.update(
        validate_dense_gate=lambda *_args: log.append("gate") or dense_mode,
        DwdpManager=manager,
        DenseDwdpManager=_standalone_dense_manager if real_dense_manager else manager,
        KvCacheCreator=kv_cache_creator,
        create_py_executor_instance=create_py_executor_instance,
        create_py_executor_resources=create_py_executor_resources,
        ModelEngine=model_engine,
        _load_config_and_create_checkpoint_loader=lambda llm_args, _dir: (llm_args, None),
        _get_mapping=lambda _: Mapping(),
        dwdp_global_rank=lambda: 0,
        get_calibrator=_DummyCalibrator,
        is_mla=lambda _: True,
        is_hybrid_linear=lambda _: False,
        get_sm_version=lambda: 100,
    )
    for name, fake in fakes.items():
        monkeypatch.setattr(py_executor_creator, name, fake)
    monkeypatch.setattr(py_executor_creator.Distributed, "get", staticmethod(lambda _: None))
    cuda = dict(
        mem_get_info=lambda: (2 << 30, 4 << 30),
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda: None,
        memory_stats=lambda: {"allocated_bytes.all.current": 0},
        Stream=lambda: SimpleNamespace(cuda_stream=123),
    )
    for name, fake in cuda.items():
        monkeypatch.setattr(py_executor_creator.torch.cuda, name, fake)

    llm_args = _make_llm_args()
    if mode == "standalone":
        llm_args.dwdp_config = DwdpConfig(
            enable_experts=False, enable_dense=True, copy_mode=copy_mode
        )
    else:
        llm_args.dwdp_config = DwdpConfig(
            enable_dense=mode == "attached",
            dwdp_size=2,
            num_experts_per_worker=4,
            num_prefetch_experts=4,
        )
    if real_dense_manager:
        monkeypatch.setattr(DenseDwdpManager, "setup", lambda _self, _model: log.append("setup"))
    executor = py_executor_creator.create_py_executor(llm_args=llm_args, checkpoint_dir=None)
    executor._prime_dwdp()
    (manager,) = [executor.dense_dwdp_manager] if real_dense_manager else managers
    return log, manager


@pytest.mark.parametrize("kv_cache", ["estimated", "not_estimated", "none"])
@pytest.mark.parametrize("mode", ["expert", "attached", "standalone"])
def test_only_the_serving_phase_gets_the_manager(
    monkeypatch: pytest.MonkeyPatch, mode: str, kv_cache: str
) -> None:
    """The manager is set up once; only the serving phase gets it and starts its forwards.

    With the KV-cache estimation, the executor is built for profiling without the manager, so
    ``release_resources`` exits none, and ``rebind_resources`` hands it to the serving phase.
    Without the estimation the executor serves from its construction and gets it then, and dense
    DWDP is set up before ``build_managers`` sizes the final KV cache, which then gets the memory
    dense DWDP frees; expert DWDP alone is set up after it.
    """
    log, manager = _create(monkeypatch, mode, kv_cache)

    given = (None, manager) if mode == "standalone" else (manager, None)
    final_kv_cache = ("build_managers", False)
    setup_first = mode != "expert"
    expected = {
        "estimated": [
            ("build_managers", True),
            "setup",
            ("executor", None, None),
            ("release", None, None),
            final_kv_cache,
            ("serving resources", *given),
            "rebind",
        ],
        "not_estimated": ["setup", final_kv_cache] if setup_first else [final_kv_cache, "setup"],
        "none": ["setup"],
    }[kv_cache]
    if kv_cache != "estimated":
        expected.append(("executor", *given))
    start = "prefetch_first_layers" if mode == "standalone" else "start_forward"
    assert log == ["gate"] * (mode != "expert") + expected + [start]


def test_a_standalone_gather_reserves_its_sms_before_the_model_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kernels that size their persistent grids when the model is built (the KDA prefill) leave
    the SM gather's resident SMs out; exiting the manager drops the reserve.  The gather comes
    from dwdp_config.copy_mode."""
    monkeypatch.delenv("TRTLLM_CUTEDSL_SM_RESERVE", raising=False)
    monkeypatch.setenv("TRTLLM_DWDP_GATHER_BLOCKS", "30")
    monkeypatch.setattr(fabric_gather, "THREADS", 1024)
    with _restore_global_dense_dwdp_manager():
        log, manager = _create(
            monkeypatch, "standalone", "not_estimated", real_dense_manager=True, copy_mode="gather"
        )
        try:
            assert ("model built with an SM reserve of", 30) in log
            assert log.index("setup") > log.index(("model built with an SM reserve of", 30))
            assert get_global_dense_dwdp_manager() is manager and sm_reserve() == 30
        finally:
            manager.__exit__(None, None, None)
    assert sm_reserve() == 0
