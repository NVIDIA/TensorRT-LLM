# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the Kimi K3 strict FMHA test's remote-worker configuration."""

import ast
import os
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_common.grouped_test_utils import preserve_worker_env

pytestmark = pytest.mark.cpu_only

_FMHA_ENV = "TLLM_FMHA_LIBS"
_KIMI_TEST = Path(__file__).resolve().parents[2] / "integration/defs/accuracy/test_kimi3.py"


def _load_test_case(class_name: str, namespace: dict[str, object]) -> type:
    """Load the real test methods without importing the GPU integration harness."""
    tree = ast.parse(_KIMI_TEST.read_text())
    case = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    case.decorator_list = []
    for method in case.body:
        if isinstance(method, ast.FunctionDef):
            if method.name in ("test_prims_ts", "test_prims_ts_fp8_tep8"):
                assert "pytest.mark.private_mpi_session" in (
                    ast.unparse(decorator) for decorator in method.decorator_list
                )
            method.decorator_list = []
    namespace.update(
        pytest=pytest,
        LlmapiAccuracyTestHarness=object,
        llm_models_root=lambda: "/unused-models",
        KvCacheConfig=SimpleNamespace,
        DSparkDecodingConfig=SimpleNamespace,
        CudaGraphConfig=SimpleNamespace,
        MambaStateConfig=SimpleNamespace,
        MoeConfig=SimpleNamespace,
        QuantAlgo=SimpleNamespace(MIXED_PRECISION="mixed", FP8="fp8"),
    )
    exec(compile(ast.Module(body=[case], type_ignores=[]), str(_KIMI_TEST), "exec"), namespace)
    return namespace[class_name]


class _RemoteWorkers:
    """Pre-existing rank environments that do not inherit subsequent parent changes."""

    def __init__(self, n_workers: int) -> None:
        self.environments = [
            {} if value is None else {_FMHA_ENV: value}
            for value in (None, "", "fallback", "+prims_ts") * (n_workers // 4)
        ]
        self.rank = 0

    def submit_sync(self, callback: Callable[..., object], *args: object) -> list[object]:
        # RemoteTask and the MPI transport use standard pickle, not cloudpickle.
        callback, args = pickle.loads(pickle.dumps((callback, args)))
        results = []
        for self.rank, environment in enumerate(self.environments):
            with patch.dict(os.environ, environment, clear=True):
                try:
                    results.append(callback(*args))
                finally:
                    environment.clear()
                    environment.update(os.environ)
        return list(reversed(results))

    def selections(self) -> list[str | None]:
        return [environment.get(_FMHA_ENV) for environment in self.environments]


@pytest.mark.parametrize("failure", [None, "constructor", "evaluation"])
@pytest.mark.parametrize("variant", ["dspark", "bf16-kv", "fp8-kv"])
def test_strict_selection_reaches_existing_workers_and_is_restored(
    monkeypatch: pytest.MonkeyPatch, failure: str | None, variant: str
) -> None:
    n_workers = 8 if variant == "dspark" else 16
    workers = _RemoteWorkers(n_workers)
    original = workers.selections()
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(
            MPI=SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_rank=lambda: workers.rank))
        ),
    )
    monkeypatch.setenv(_FMHA_ENV, "fallback")
    observed = []
    strict_run = False

    class FakeLLM:
        def __init__(self, _model: str, **kwargs: object) -> None:
            self.strict = strict_run
            assert kwargs["tensor_parallel_size"] == n_workers
            kv_dtype = getattr(kwargs["kv_cache_config"], "dtype", "auto")
            if self.strict:
                assert kv_dtype == ("auto" if variant == "bf16-kv" else "fp8")
            overrides = kwargs.get("env_overrides") or {}
            # worker_main receives only the serialized LLM args, not the
            # pytest process's current environment.
            for environment in workers.environments:
                environment.update(overrides)
            observed.append(workers.selections())
            assert observed[-1] == (["prims_ts"] * len(original) if self.strict else original)
            os.environ.update(overrides)
            if self.strict and failure == "constructor":
                raise RuntimeError("constructor failed")
            self.args = SimpleNamespace(
                kv_cache_config=kwargs["kv_cache_config"],
                quant_config=SimpleNamespace(
                    quant_algo="mixed", kv_cache_quant_algo="fp8" if kv_dtype == "fp8" else None
                ),
            )

        def __enter__(self) -> "FakeLLM":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    def evaluate(llm: FakeLLM) -> None:
        if llm.strict and failure == "evaluation":
            raise RuntimeError("evaluation failed")

    def create_session(size: int) -> _RemoteWorkers:
        assert size == n_workers
        return workers

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.executor.utils",
        SimpleNamespace(
            create_mpi_comm_session=create_session,
            get_spawn_proxy_process_env=lambda: True,
        ),
    )
    case_type = _load_test_case(
        "TestKimiK3DSpark" if variant == "dspark" else "TestKimiK3",
        dict(
            LLM=FakeLLM,
            GSM8K=lambda _model: SimpleNamespace(evaluate=evaluate),
            assert_acceptance_length_for_llm=lambda *_args: None,
            assert_guided_decoding_regex=lambda *_args: None,
            assert_kv_cache_reuse_for_llm=lambda *_args: None,
            preserve_worker_env=preserve_worker_env,
        ),
    )
    case = case_type()
    case._assert_checkpoint_routing = lambda: None
    case._assert_resolved_args = lambda *_args: None
    case._assert_attention_dp = lambda *_args: None
    case._assert_logits_processor = lambda *_args: None

    def run_baseline() -> None:
        if variant == "dspark":
            case.test_gsm8k_tep8()
        else:
            with monkeypatch.context() as baseline_patch:
                case.test_w4a16_mxfp4("baseline", baseline_patch)

    def run_strict(strict_patch: pytest.MonkeyPatch) -> None:
        if variant == "dspark":
            case.test_prims_ts_fp8_tep8(strict_patch)
        else:
            case.test_prims_ts(variant == "fp8-kv", strict_patch)

    run_baseline()
    strict_run = True
    with monkeypatch.context() as strict_patch:
        if failure:
            with pytest.raises(RuntimeError, match=f"{failure} failed"):
                run_strict(strict_patch)
        else:
            run_strict(strict_patch)
    assert os.environ[_FMHA_ENV] == "fallback"
    assert workers.selections() == original
    strict_run = False
    run_baseline()
    assert observed == [original, ["prims_ts"] * len(original), original]
