# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for VisualGen's client latency metric adapter."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
_PERF_DIR = Path(__file__).resolve().parents[3] / "tests/integration/defs/perf"
_Modules = tuple[types.ModuleType, types.ModuleType]


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> _Modules:
    """Load the real adapter and harness, isolating GPU/environment imports only."""
    stubs = {
        "torch": {},
        "defs": {"__path__": []},
        "defs.perf": {"__path__": []},
        "defs.conftest": {"get_llm_root": lambda: "/unused", "llm_models_root": lambda: "/unused"},
        "defs.trt_test_alternative": {"print_info": lambda *args, **kwargs: None},
        "defs.perf.perf_regression_utils": {
            "process_and_upload_test_results": lambda **kwargs: pytest.fail("Unexpected upload")
        },
        "tensorrt_llm.commands.utils": {"get_visual_gen_num_gpus": lambda config: 1},
        "tensorrt_llm._utils": {
            "get_free_port": lambda: 0,
            "get_sm_version": lambda: pytest.fail("Unexpected GPU capability query"),
        },
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    loaded = []
    for name in ("visual_gen_perf_utils", "test_visual_gen_perf_sanity"):
        fullname = f"defs.perf.{name}"
        spec = importlib.util.spec_from_file_location(fullname, _PERF_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded[0], loaded[1]


def _result(backend: str) -> dict:
    def stats(value: float) -> dict:
        return {
            "mean": value,
            "median": value * 1.1,
            "percentiles": {"p90": value * 1.5, "p99": value * 2},
        }

    result = {
        "backend": backend,
        "model": "test-model",
        "config": {"num_gpus": 1},
        "total_requests": 1,
        "completed": 1,
        "request_throughput": 2,
        "e2e_latency": stats(0.94),
        "timings": {"server_gen": stats(0.5)},
    }
    if backend == "openai-videos":
        result["gen_latency"] = stats(0.8)
    return result


def _validate(harness: types.ModuleType, result: dict) -> None:
    harness.VisualGenPerfSanityTestConfig._validate_benchmark_result(
        None,
        model_path="test-model",
        expected_num_gpus=1,
        result_data=result,
        client_config={"backend": result.get("backend")},
        result_path="test-only.json",
    )


@pytest.mark.parametrize(
    ("backend", "generation"),
    [("openai-videos", 0.8), ("openai-images", 0.94), ("openai-image-edits", 0.94)],
)
def test_client_metrics_keep_existing_database_fields(
    modules: _Modules, backend: str, generation: float
) -> None:
    producer, harness = modules
    result = _result(backend)
    _validate(harness, result)
    entry = producer.build_visual_gen_db_entry(
        gpu_type="b200",
        model_name="test-model",
        server_name="test-server",
        server_config={},
        client_config={"backend": backend, "num_inference_steps": 50, "max_concurrency": 1},
        result_data=result,
    )
    for statistic, scale in (("mean", 1), ("median", 1.1), ("p90", 1.5), ("p99", 2)):
        assert entry[f"d_{statistic}_generation"] == pytest.approx(generation * scale)
        assert entry[f"d_{statistic}_latency"] == pytest.approx(0.94 * scale)
    assert set(producer.get_visual_gen_match_keys()) <= entry.keys()
    assert "s_generation_timing_source" not in entry


def test_video_without_generation_does_not_fall_back_to_e2e(modules: _Modules) -> None:
    producer, harness = modules
    result = _result("openai-videos")
    del result["gen_latency"]
    with pytest.raises(ValueError, match="Missing.*gen_latency"):
        _validate(harness, result)
    with pytest.raises(ValueError, match="Missing.*gen_latency"):
        producer.extract_visual_gen_metrics(result)


@pytest.mark.parametrize(
    ("backend", "field"),
    [
        ("openai-videos", "gen_latency"),
        ("openai-videos", "e2e_latency"),
        ("openai-images", "e2e_latency"),
    ],
)
@pytest.mark.parametrize("invalid", [float("nan"), 0, -1, "missing_p99"])
def test_invalid_client_series_is_rejected(
    modules: _Modules, backend: str, field: str, invalid: float | str
) -> None:
    _, harness = modules
    result = _result(backend)
    if invalid == "missing_p99":
        del result[field]["percentiles"]["p99"]
    else:
        result[field]["median"] = invalid
    with pytest.raises(ValueError, match=field):
        _validate(harness, result)


def test_missing_backend_reports_missing_key(modules: _Modules) -> None:
    _, harness = modules
    result = _result("openai-images")
    del result["backend"]
    with pytest.raises(ValueError, match="Missing keys.*backend"):
        _validate(harness, result)
