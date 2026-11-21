# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest import mock

import pynvml
import pytest

from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.reporting import PerfItemTuple, ReportUtility, StatsKeeper
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats
from tensorrt_llm.bench.tuning import heuristics
from tensorrt_llm.bench.tuning import utils as tuning_utils
from tensorrt_llm.bench.tuning.dataclasses import NemotronHybridConfig
from tensorrt_llm.bench.utils import VALID_QUANT_ALGOS
from tensorrt_llm.bench.utils.data import (
    DatasetFormatError,
    create_dataset_from_stream,
    initialize_tokenizer,
)
from tensorrt_llm.llmapi import CudaGraphConfig
from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


class _FakeTokenizer:
    """Minimal tokenizer stub for testing create_dataset_from_stream."""

    def __call__(self, text, **kwargs):
        ids = list(range(len(text.split())))
        return {"input_ids": ids}

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


def _benchmark_runtime_config(backend: str) -> RuntimeConfig:
    return RuntimeConfig(
        model="model",
        sw_version="test",
        settings_config={"max_batch_size": 8, "max_num_tokens": 128},
        mapping={
            "pp_size": 1,
            "tp_size": 1,
            "gpus_per_node": 1,
            "moe_ep_size": 1,
            "moe_cluster_size": 1,
        },
        decoding_config={},
        performance_options={
            "pytorch_config": {
                "cuda_graph_config": {"batch_sizes": [1, 2]},
                "disable_overlap_scheduler": True,
            },
        },
        backend=backend,
    )


def test_runtime_config_omits_legacy_trt_args():
    llm_args = _benchmark_runtime_config("pytorch").get_llm_args()

    assert "extended_runtime_perf_knob_config" not in llm_args
    assert "batching_type" not in llm_args


def test_runtime_config_preserves_pytorch_performance_options():
    llm_args = _benchmark_runtime_config("pytorch").get_llm_args()

    assert llm_args["cuda_graph_config"] == {"batch_sizes": [1, 2]}
    assert llm_args["disable_overlap_scheduler"] is True


@pytest.mark.parametrize("capacity", [1, 3, 17, 65])
@pytest.mark.parametrize(
    "graph_options",
    [None, {}, {"enable_padding": True}, {"batch_sizes": []}, {"batch_sizes": None}],
    ids=["unspecified", "empty", "padding", "empty-sizes", "unset-sizes"],
)
def test_runtime_graph_defaults_follow_effective_capacity(
    capacity: int, graph_options: dict | None
) -> None:
    config = _benchmark_runtime_config("pytorch")
    config.settings_config.max_batch_size = capacity
    options = config.performance_options.pytorch_config
    if graph_options is None:
        del options["cuda_graph_config"]
    else:
        options["cuda_graph_config"] = deepcopy(graph_options)

    llm_args = config.get_llm_args()
    graph = CudaGraphConfig(**llm_args["cuda_graph_config"])

    assert graph.max_batch_size == capacity, "Graph defaults must follow runtime capacity"
    assert graph.batch_sizes
    assert graph.batch_sizes[-1] == capacity
    assert all(0 < size <= capacity for size in graph.batch_sizes)
    if graph_options and "enable_padding" in graph_options:
        assert graph.enable_padding is graph_options["enable_padding"]


@pytest.mark.parametrize("capacity", [3, 17, 65])
@pytest.mark.parametrize("explicit_kind", ["maximum", "sizes", "both"])
def test_runtime_graph_preserves_explicit_capture_domain(capacity: int, explicit_kind: str) -> None:
    config = _benchmark_runtime_config("pytorch")
    config.settings_config.max_batch_size = capacity
    capture_maximum = capacity + 2
    sizes = [capture_maximum, 1, capacity]
    graph_options = {"enable_padding": True}
    if explicit_kind in {"maximum", "both"}:
        graph_options["max_batch_size"] = capture_maximum
    if explicit_kind in {"sizes", "both"}:
        graph_options["batch_sizes"] = sizes
    config.performance_options.pytorch_config["cuda_graph_config"] = graph_options

    llm_args = config.get_llm_args()
    graph = CudaGraphConfig(**llm_args["cuda_graph_config"])

    assert graph.max_batch_size == capture_maximum, "Explicit graph maximum must take precedence"
    assert graph.enable_padding is True
    if explicit_kind in {"sizes", "both"}:
        assert graph.batch_sizes == sorted(sizes), "Explicit capture sizes must be preserved"
    else:
        assert graph.batch_sizes[-1] == capture_maximum
        assert all(0 < size <= capture_maximum for size in graph.batch_sizes)


def test_runtime_graph_none_disables_capture() -> None:
    config = _benchmark_runtime_config("pytorch")
    config.performance_options.pytorch_config["cuda_graph_config"] = None

    assert config.get_llm_args()["cuda_graph_config"] is None


def test_runtime_graph_zero_retains_public_default_sentinel() -> None:
    config = _benchmark_runtime_config("pytorch")
    config.performance_options.pytorch_config["cuda_graph_config"] = {"max_batch_size": 0}

    graph_options = config.get_llm_args()["cuda_graph_config"]
    assert graph_options["max_batch_size"] == 0, "Zero is an explicit public default sentinel"
    graph = CudaGraphConfig(**graph_options)
    assert graph.max_batch_size > 0
    assert graph.batch_sizes == CudaGraphConfig(max_batch_size=0).batch_sizes


def test_runtime_graph_defaults_do_not_mutate_repeated_calls() -> None:
    config = _benchmark_runtime_config("pytorch")
    config.performance_options.pytorch_config["cuda_graph_config"] = {"enable_padding": True}
    original_options = deepcopy(config.performance_options.pytorch_config)
    capacities = [3, 17, 5]
    graphs = []

    for capacity in capacities:
        config.settings_config.max_batch_size = capacity
        graph_options = config.get_llm_args()["cuda_graph_config"]
        graphs.append(CudaGraphConfig(**graph_options))
        assert config.performance_options.pytorch_config == original_options, (
            "Deriving graph defaults must not mutate reusable performance options"
        )

    assert [graph.max_batch_size for graph in graphs] == capacities
    assert [graph.batch_sizes[-1] for graph in graphs] == capacities


@pytest.mark.parametrize("explicit_capacity", [False, True])
@pytest.mark.parametrize("graph_source", ["performance", "yaml"])
def test_runtime_graph_defaults_follow_yaml_cli_capacity_priority(
    tmp_path: Path, explicit_capacity: bool, graph_source: str
) -> None:
    config = _benchmark_runtime_config("pytorch")
    cli_capacity, yaml_capacity = 5, 19
    config.settings_config.max_batch_size = cli_capacity
    config.performance_options.pytorch_config["cuda_graph_config"] = {"enable_padding": True}
    yaml_text = f"max_batch_size: {yaml_capacity}\n"
    if graph_source == "yaml":
        yaml_text += "cuda_graph_config:\n  enable_padding: false\n"
    options_path = tmp_path / "runtime.yaml"
    options_path.write_text(yaml_text)
    config.extra_llm_api_options = str(options_path)
    config.explicit_cli_keys = {"max_batch_size"} if explicit_capacity else set()
    expected_capacity = cli_capacity if explicit_capacity else yaml_capacity

    llm_args = config.get_llm_args()
    graph = CudaGraphConfig(**llm_args["cuda_graph_config"])

    assert llm_args["max_batch_size"] == expected_capacity
    assert graph.max_batch_size == expected_capacity, (
        "Graph defaults must follow the final capacity"
    )
    assert graph.batch_sizes[-1] == expected_capacity
    assert graph.enable_padding is (graph_source == "performance")


@pytest.mark.parametrize("explicit_graph", [False, True])
def test_runtime_graph_yaml_cli_graph_priority(tmp_path: Path, explicit_graph: bool) -> None:
    config = _benchmark_runtime_config("pytorch")
    cli_sizes, yaml_sizes = [1, 3], [2, 7]
    config.performance_options.pytorch_config["cuda_graph_config"] = {"batch_sizes": cli_sizes}
    options_path = tmp_path / "graphs.yaml"
    options_path.write_text(f"cuda_graph_config:\n  batch_sizes: {yaml_sizes}\n")
    config.extra_llm_api_options = str(options_path)
    config.explicit_cli_keys = {"cuda_graph_config"} if explicit_graph else set()
    expected_sizes = cli_sizes if explicit_graph else yaml_sizes

    graph = CudaGraphConfig(**config.get_llm_args()["cuda_graph_config"])

    assert graph.batch_sizes == expected_sizes
    assert graph.max_batch_size == max(expected_sizes)


def test_format_startup_metrics() -> None:
    output = ReportUtility._format_startup_metrics(
        {
            "executor_startup_seconds": 0.5,
            "startup_collection_complete": False,
            "model_loader": {
                "checkpoint_preparation_seconds": 1.25,
                "checkpoint_loader_kind": "HfCheckpointLoader",
                "weight_loading": {
                    "total_model_loading_seconds": 2.5,
                },
            },
        }
    )

    assert "= STARTUP METRICS\n" in output
    assert "executor_startup_seconds: 0.5000\n" in output
    assert "startup_collection_complete: False\n" in output
    assert "model_loader.checkpoint_preparation_seconds: 1.2500\n" in output
    assert "model_loader.checkpoint_loader_kind: HfCheckpointLoader\n" in output
    assert "model_loader.weight_loading.total_model_loading_seconds: 2.5000\n" in output
    assert ReportUtility._format_startup_metrics({}) == ""


def test_empty_stream_raises_dataset_format_error():
    tokenizer = _FakeTokenizer()
    empty_stream = io.StringIO("")

    with pytest.raises(DatasetFormatError, match="No data was read from the dataset stream"):
        create_dataset_from_stream(tokenizer, empty_stream)


def _lora_dataset_stream(lora_request) -> io.StringIO:
    line = {"task_id": 1, "prompt": "hello world", "output_tokens": 8, "lora_request": lora_request}
    return io.StringIO(json.dumps(line) + "\n")


@pytest.mark.parametrize(
    "lora_request, message",
    [
        ({}, "missing required key(s) ['lora_name', 'lora_int_id']"),
        ({"lora_path": "/tmp/a"}, "missing required key(s) ['lora_name', 'lora_int_id']"),
        ({"lora_name": "adapter"}, "missing required key(s) ['lora_int_id']"),
        ({"lora_int_id": 1}, "missing required key(s) ['lora_name']"),
        ("adapter", "must be a JSON object, got str"),
        (["adapter", 1], "must be a JSON object, got list"),
    ],
)
def test_malformed_lora_request_raises_dataset_format_error(lora_request, message):
    with pytest.raises(DatasetFormatError) as exc_info:
        create_dataset_from_stream(_FakeTokenizer(), _lora_dataset_stream(lora_request))
    assert message in str(exc_info.value)
    # The offending request is named so it can be found in a large dataset.
    assert "'task_id': 1" in str(exc_info.value)
    assert f"'lora_request': {lora_request!r}" in str(exc_info.value)


def test_lora_request_is_parsed():
    _, requests = create_dataset_from_stream(
        _FakeTokenizer(), _lora_dataset_stream({"lora_name": "adapter", "lora_int_id": 3})
    )
    assert requests[0].lora_request.lora_name == "adapter"
    assert requests[0].lora_request.lora_int_id == 3


def test_initialize_tokenizer_routes_through_transformers_tokenizer():
    """``initialize_tokenizer`` must call ``TransformersTokenizer.from_pretrained``.

    Routing through ``TransformersTokenizer`` is what lets ``trtllm-bench``
    inherit the post-load fixes (e.g. ``maybe_fix_byte_level_tokenizer``,
    which prevents DeepSeek-V3 from loading with a Metaspace pre-tokenizer
    that silently strips spaces -- "hello world" -> "helloworld" -- on
    transformers >= 5.x). Without this routing the bench would see a
    different tokenizer than the rest of TRT-LLM uses.
    """
    inner = mock.MagicMock()
    inner.pad_token_id = 0  # already set => add_special_tokens not invoked
    wrapper = mock.MagicMock(tokenizer=inner)

    with mock.patch(
        "tensorrt_llm.bench.utils.data.TransformersTokenizer.from_pretrained", return_value=wrapper
    ) as routed:
        out = initialize_tokenizer("dummy/model")

    routed.assert_called_once_with("dummy/model", padding_side="left", trust_remote_code=True)
    # Bench code uses the raw HF tokenizer (calls __call__, encode,
    # add_special_tokens on it), so the wrapper must be peeled off.
    assert out is inner


def test_int8_not_offered_as_bench_quant_choice() -> None:
    # INT8 is unsupported by the build path, so it must not be advertised as a
    # trtllm-bench --quantization choice (issue #7091).
    assert f"{QuantAlgo.INT8}" not in get_args(VALID_QUANT_ALGOS)


@pytest.mark.parametrize(
    "values, weights, expected_average",
    [
        # No weights -> plain arithmetic mean (backward compatible).
        ([1.0, 2.0, 3.0, 4.0], None, 2.5),
        # Weighted mean: sum(w * v) / sum(w).
        ([0.5, 1.0], [1.0, 3.0], 0.875),
        # The heavy (large-weight) value pulls the average toward itself
        # instead of being equally weighted (0.75 -> 0.63).
        ([0.9, 0.6], [10, 90], 0.63),
        # Uniform weights reproduce the unweighted mean.
        ([2.0, 4.0, 6.0, 8.0], [5.0, 5.0, 5.0, 5.0], 5.0),
        # Zero total weight must not divide by zero; fall back to plain mean.
        ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 2.0),
    ],
)
def test_weighted_average(values, weights, expected_average):
    stats = PercentileStats.from_iterable(values, weights=weights)
    assert stats.average == pytest.approx(expected_average)


def test_weights_only_affect_average():
    """Weights adjust the average but never the percentiles/min/max."""
    values = [1.0, 2.0, 3.0, 4.0]
    weighted = PercentileStats.from_iterable(values, weights=[1.0, 100.0, 1.0, 1.0])
    unweighted = PercentileStats.from_iterable(values)
    assert weighted.average != pytest.approx(unweighted.average)
    assert (
        weighted.p50,
        weighted.p90,
        weighted.p95,
        weighted.p99,
        weighted.minimum,
        weighted.maximum,
    ) == (
        unweighted.p50,
        unweighted.p90,
        unweighted.p95,
        unweighted.p99,
        unweighted.minimum,
        unweighted.maximum,
    )


def test_mismatched_weights_length_raises():
    """zip(strict=True) rejects mismatched values/weights lengths."""
    with pytest.raises(ValueError):
        PercentileStats.from_iterable([1.0, 2.0, 3.0], weights=[1.0, 2.0])


def _perf_item(rid, num_out, decoding_iteration):
    """Build a final PerfItemTuple for a single request.

    ``num_total_output_tokens`` is ``len(tokens)``, so ``tokens`` is sized to
    ``num_out``; ``decoding_iteration`` maps onto ``RequestRecord.decode_iteration``.
    """
    return PerfItemTuple(
        start_timestamp=0,
        end_timestamp=1000,
        request_id=rid,
        num_input_tokens=8,
        response_is_final=True,
        error=False,
        tokens=list(range(num_out)),
        decoding_iteration=decoding_iteration,
        time_on_first_token=100,
    )


def test_acceptance_length_weighted_by_iterations():
    """AR/AL averages are weighted by per-request decoding iterations.

    AL_i = num_total_output_tokens / (decode_iteration + 1), weighted by the
    iteration count (decode_iteration + 1). A short request has high AL but few
    iterations; a long request has low AL but many iterations, so it dominates.
    """
    keeper = StatsKeeper()
    # short request: AL = 10 / 2 = 5.0 over 2 iterations
    keeper.register_request_perf_item(_perf_item(0, num_out=10, decoding_iteration=1))
    # long request: AL = 300 / 200 = 1.5 over 200 iterations
    keeper.register_request_perf_item(_perf_item(1, num_out=300, decoding_iteration=199))

    stats = keeper.generate_statistics_summary(max_draft_tokens=3, batch_size=1)

    # iteration-weighted mean: weights are (decode_iteration + 1).
    weighted = (2 * 5.0 + 200 * 1.5) / (2 + 200)
    assert stats.acceptance_length_percentiles.average == pytest.approx(weighted)

    # Design intent: under iteration weighting the per-request AL average
    # equals the globally-computed acceptance_length
    # (= sum(output_tokens) / total_decoding_iterations). This would fail under
    # output-token weighting.
    assert stats.acceptance_length_percentiles.average == pytest.approx(stats.acceptance_length)


def _patch_nvml(monkeypatch, get_memory_info):
    """Replace the NVML calls and logger used by get_device_memory; record init/shutdown."""
    calls = []
    nvml = tuning_utils.pynvml
    monkeypatch.setattr(nvml, "nvmlInit", lambda: calls.append("init"))
    monkeypatch.setattr(nvml, "nvmlShutdown", lambda: calls.append("shutdown"))
    monkeypatch.setattr(nvml, "nvmlDeviceGetHandleByIndex", lambda index: f"handle{index}")
    monkeypatch.setattr(nvml, "nvmlDeviceGetMemoryInfo", get_memory_info)
    bench_logger = mock.Mock()
    monkeypatch.setattr(tuning_utils, "logger", bench_logger, raising=False)
    # Ignore the cgroup of the machine running the tests.
    monkeypatch.setattr(tuning_utils, "_CGROUP_MEMORY_LIMIT_FILES", ())
    return calls, bench_logger


def _raise_nvml(error_code):
    def get_memory_info(handle):
        # NVMLError.__new__ returns the matching subclass, e.g. NVMLError_NotSupported.
        raise pynvml.NVMLError(error_code)

    return get_memory_info


def test_get_device_memory_reports_nvml_total(monkeypatch):
    calls, bench_logger = _patch_nvml(
        monkeypatch, lambda handle: SimpleNamespace(total=80 * 1024**3, free=0, used=0)
    )
    with mock.patch("psutil.virtual_memory") as virtual_memory:
        assert tuning_utils.get_device_memory() == pytest.approx(80.0)
    virtual_memory.assert_not_called()
    bench_logger.warning.assert_not_called()
    assert calls == ["init", "shutdown"]


def test_get_device_memory_falls_back_when_nvml_memory_not_supported(monkeypatch):
    """Unified-memory GPUs such as DGX Spark (GB10) do not report memory through NVML."""
    calls, bench_logger = _patch_nvml(monkeypatch, _raise_nvml(pynvml.NVML_ERROR_NOT_SUPPORTED))
    with mock.patch(
        "psutil.virtual_memory", return_value=SimpleNamespace(total=128 * 1024**3)
    ) as virtual_memory:
        assert tuning_utils.get_device_memory() == pytest.approx(128.0)
    virtual_memory.assert_called_once_with()
    bench_logger.warning.assert_called_once()
    assert "128.00 GiB" in bench_logger.warning.call_args.args[0]
    assert calls == ["init", "shutdown"]


def test_get_device_memory_shuts_down_nvml_on_other_errors(monkeypatch):
    calls, bench_logger = _patch_nvml(monkeypatch, _raise_nvml(pynvml.NVML_ERROR_UNKNOWN))
    with pytest.raises(pynvml.NVMLError_Unknown):
        tuning_utils.get_device_memory()
    bench_logger.warning.assert_not_called()
    assert calls == ["init", "shutdown"]


@pytest.mark.parametrize(
    ("cgroup_value", "expected_gib"),
    [
        ("34359738368\n", 32.0),  # cgroup v2 limit below system memory
        ("max\n", 128.0),  # cgroup v2 without a limit
        ("9223372036854771712\n", 128.0),  # cgroup v1 "unlimited" sentinel
    ],
)
def test_get_device_memory_fallback_respects_cgroup_limit(
    monkeypatch, tmp_path, cgroup_value, expected_gib
):
    """Inside a container, the fallback must not report more than the container may use."""
    _patch_nvml(monkeypatch, _raise_nvml(pynvml.NVML_ERROR_NOT_SUPPORTED))
    limit_file = tmp_path / "memory.max"
    limit_file.write_text(cgroup_value)
    monkeypatch.setattr(
        tuning_utils, "_CGROUP_MEMORY_LIMIT_FILES", (str(tmp_path / "missing"), str(limit_file))
    )
    with mock.patch("psutil.virtual_memory", return_value=SimpleNamespace(total=128 * 1024**3)):
        assert tuning_utils.get_device_memory() == pytest.approx(expected_gib)


def _nemotron_nano_12b_v2_bench_config(mamba_ssm_cache_dtype, dtype):
    """trtllm-bench model config with the NVIDIA-Nemotron-Nano-12B-v2 geometry."""
    # Pydantic accepts ssm_state_size and conv_kernel as validation aliases.
    # pyrefly: ignore [missing-argument]
    return NemotronHybridConfig(
        name="nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        model_type="nemotron_h",
        param_count=12_310_001_152,
        hybrid_override_pattern="M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M*-M-M-M-M-",
        num_hidden_layers=62,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_size=5120,
        head_dim=128,
        ssm_state_size=128,
        conv_kernel=4,
        mamba_num_heads=128,
        mamba_head_dim=80,
        n_groups=8,
        dtype=dtype,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
    )


def test_calc_engine_setting_sizes_unset_ssm_cache_dtype_from_model_dtype(monkeypatch):
    """An unset mamba_ssm_cache_dtype is sized like the model dtype."""
    monkeypatch.setattr(heuristics, "get_device_memory", lambda: 80.0)

    def engine_setting(mamba_ssm_cache_dtype, dtype):
        return heuristics.calc_engine_setting(
            _nemotron_nano_12b_v2_bench_config(mamba_ssm_cache_dtype, dtype),
            QuantConfig(),
            tp_size=1,
            pp_size=1,
            target_input_len=128,
            target_output_len=128,
        )

    bf16_setting = engine_setting("bfloat16", "bfloat16")
    fp32_setting = engine_setting("float32", "float32")
    assert fp32_setting[0] < bf16_setting[0]
    for unset in (None, "auto"):
        assert engine_setting(unset, "bfloat16") == bf16_setting
        assert engine_setting(unset, "float32") == fp32_setting
