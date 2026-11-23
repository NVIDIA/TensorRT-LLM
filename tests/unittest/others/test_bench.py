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
import torch
from safetensors.torch import save_file

from tensorrt_llm.bench.benchmark import get_general_cli_options
from tensorrt_llm.bench.benchmark.utils.general import get_settings
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment, DatasetMetadata
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


def _llm_api_options(tmp_path, **llm_args):
    options = tmp_path / "extra_llm_api_options.yaml"
    options.write_text("".join(f"{k}: {v}\n" for k, v in llm_args.items()))
    return str(options)


@pytest.mark.parametrize(
    "checkpoint_format, model_path, downloads",
    [("mistral", None, True), ("mistral", "local", False), ("HF", None, False)],
)
def test_cli_options_download_a_mistral_native_model_name_before_reading_it(
    tmp_path, checkpoint_format, model_path, downloads
):
    """A model name in a Mistral-native format is read from its snapshot, as the LLM API reads it.

    The checkpoint path is set before the model type, the tuning heuristic or the report read
    the checkpoint; a local checkpoint and the Hugging Face format are not downloaded.
    """
    snapshot = tmp_path / "snapshot"
    local = tmp_path / "local"
    env = BenchmarkEnvironment(
        model="org/model",
        checkpoint_path=local if model_path else None,
        workspace=tmp_path,
        revision="rev",
    )
    params = {}
    if checkpoint_format is not None:
        params["extra_llm_api_options"] = _llm_api_options(
            tmp_path, checkpoint_format=checkpoint_format
        )
    with mock.patch(
        "tensorrt_llm.bench.benchmark.download_hf_model", return_value=snapshot
    ) as download:
        options = get_general_cli_options(params, env)
    if downloads:
        download.assert_called_once_with("org/model", "rev")
        expected = snapshot
    else:
        download.assert_not_called()
        expected = local if model_path else None
    assert env.checkpoint_path == expected
    assert options.model_path == expected
    assert options.checkpoint_path == (expected or "org/model")


# A small dense Mistral-native model: 2 layers, 4 query and 2 KV heads of 64.
_MISTRAL_NATIVE_PARAMS = {
    "dim": 256,
    "n_layers": 2,
    "head_dim": 64,
    "hidden_dim": 512,
    "n_heads": 4,
    "n_kv_heads": 2,
    "norm_eps": 1e-5,
    "vocab_size": 1024,
    "rope_theta": 1e6,
}
# The Hugging Face config.json of another model, to tell which of the two files was read.
_HF_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 1024,
    "torch_dtype": "bfloat16",
}


def _weights(*shapes):
    return {f"w{i}": torch.zeros(shape, dtype=torch.bfloat16) for i, shape in enumerate(shapes)}


def _mistral_native_checkpoint(path, *, weights="consolidated", hf_config=False):
    """Write params.json and, by `weights`, its weights.

    `weights` is "consolidated", "sharded" (consolidated shards and their index),
    "unindexed_shards" or None.
    """
    path.mkdir()
    (path / "params.json").write_text(json.dumps(_MISTRAL_NATIVE_PARAMS))
    if weights == "consolidated":
        save_file(_weights((1024, 256), (256,)), path / "consolidated.safetensors")
    elif weights in ("sharded", "unindexed_shards"):
        weight_map = {}
        for shard, (name, tensor) in enumerate(_weights((1024, 256), (256,)).items(), 1):
            file_name = f"consolidated-{shard:05d}-of-00002.safetensors"
            save_file({name: tensor}, path / file_name)
            weight_map[name] = file_name
        if weights == "sharded":
            (path / "consolidated.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )
    if hf_config:
        (path / "config.json").write_text(json.dumps(_HF_CONFIG))
        save_file(_weights((1024, 128)), path / "model.safetensors")
    return path


@pytest.mark.parametrize(
    "checkpoint_format,model_type", [(None, "llama"), ("mistral_large_3", "mistral_common")]
)
def test_cli_options_read_the_model_type_of_the_checkpoint_format(
    tmp_path, checkpoint_format, model_type
):
    """A Mistral-native checkpoint's model type comes from params.json, even beside a config.json.

    The model type selects the multimodal prompt rendering.
    """
    checkpoint = _mistral_native_checkpoint(tmp_path / "checkpoint", hf_config=True)
    env = BenchmarkEnvironment(model="m", checkpoint_path=checkpoint, workspace=tmp_path)
    params = {}
    if checkpoint_format is not None:
        params["extra_llm_api_options"] = _llm_api_options(
            tmp_path, checkpoint_format=checkpoint_format
        )
    options = get_general_cli_options(params, env)
    assert options.checkpoint_format == checkpoint_format
    assert options.model_type == model_type


def _tune(tmp_path, checkpoint, checkpoint_format="mistral"):
    """Run get_settings without explicit limits; return the engine-setting arguments."""
    params = {
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "extra_llm_api_options": _llm_api_options(
            tmp_path, checkpoint_format=checkpoint_format, enable_chunked_prefill="false"
        ),
    }
    metadata = SimpleNamespace(max_isl=4096, avg_isl=1024, avg_osl=256)
    with mock.patch(
        "tensorrt_llm.bench.benchmark.utils.general.get_benchmark_engine_settings",
        return_value=(16, 32768),
    ) as engine_settings:
        settings = get_settings(params, metadata, "mistral-native", checkpoint)
    assert settings["settings_config"]["max_batch_size"] == 16
    assert settings["settings_config"]["max_num_tokens"] == 32768
    model_config, quant_config = engine_settings.call_args.args[:2]
    return model_config, quant_config


@pytest.mark.parametrize(
    "weights,hf_config,checkpoint_format",
    [
        ("consolidated", False, "mistral"),
        ("sharded", False, "mistral"),
        ("unindexed_shards", False, "mistral"),
        # params.json wins over a config.json beside it.
        ("consolidated", True, "mistral_large_3"),
    ],
)
def test_tuning_reads_a_mistral_native_checkpoint_from_its_native_files(
    tmp_path, weights, hf_config, checkpoint_format
):
    """The tuning heuristic reads params.json and the consolidated weights, as the runtime does."""
    checkpoint = _mistral_native_checkpoint(
        tmp_path / "checkpoint", weights=weights, hf_config=hf_config
    )
    model_config, quant_config = _tune(tmp_path, checkpoint, checkpoint_format)
    assert model_config.model_type == "mistral_common"
    assert model_config.num_hidden_layers == 2
    assert model_config.num_attention_heads == 4
    assert model_config.num_key_value_heads == 2
    assert model_config.head_size == 64
    assert model_config.dtype == "bfloat16"
    assert not model_config.is_mla()
    assert model_config.param_count == 1024 * 256 + 256
    assert model_config.checkpoint_size_in_gb == (1024 * 256 + 256) * 2 / 1024**3
    assert quant_config.quant_algo is None


def test_an_image_dataset_of_a_mistral_native_checkpoint_renders_with_its_tekken_file(tmp_path):
    """An image dataset renders with the Tekken file, without --custom_tokenizer.

    trtllm-bench reads a Mistral-native checkpoint's tokenizer through mistral-common, as its
    input processor does, though the directory ships a Hugging Face tokenizer too; each image
    goes at its [IMG] marker.
    """
    import base64
    import io
    import shutil

    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from PIL import Image
    from test_common.llm_data import llm_models_root
    from test_common.mistral_tekken import TEKKEN_CHECKPOINT_FILES, write_hf_tokenizer

    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    pytest.importorskip("cv2")  # mistral-common's image encoder resizes with OpenCV
    root = llm_models_root()
    tekken = None if root is None else root / TEKKEN_CHECKPOINT_FILES["v15"]
    if tekken is None or not tekken.is_file():
        pytest.skip("needs a Tekken file with an image encoder under LLM_MODELS_ROOT")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    shutil.copy(tekken, checkpoint / "tekken.json")
    write_hf_tokenizer(checkpoint)
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "mistral3"}))
    image_path = tmp_path / "red.png"
    Image.new("RGB", (64, 48), (200, 30, 30)).save(image_path)

    assert not isinstance(initialize_tokenizer(str(checkpoint)), MistralTokenizer)
    tokenizer = initialize_tokenizer(str(checkpoint), checkpoint_format="mistral")
    assert isinstance(tokenizer, MistralTokenizer)
    line = {
        "task_id": 0,
        "prompt": "Describe [IMG] briefly.",
        "media_paths": [str(image_path)],
        "output_tokens": 4,
    }
    _, requests = create_dataset_from_stream(
        tokenizer,
        io.StringIO(json.dumps(line) + "\n"),
        model_dir=str(checkpoint),
        model_type="mistral_common",
        modality="image",
    )
    buffer = io.BytesIO()
    Image.open(image_path).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    content = [
        {"type": "text", "text": "Describe "},
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": " briefly."},
    ]
    expected = tokenizer.mistral.encode_chat_completion(
        ChatCompletionRequest.from_openai(messages=[{"role": "user", "content": content}])
    ).tokens
    assert requests[0].prompt["prompt_token_ids"] == expected


def test_tuning_reads_config_json_when_the_native_files_cannot_be_read(tmp_path):
    checkpoint = _mistral_native_checkpoint(tmp_path / "checkpoint", weights=None, hf_config=True)
    model_config, quant_config = _tune(tmp_path, checkpoint)
    assert model_config.model_type == "llama"
    assert model_config.num_hidden_layers == 3
    assert model_config.param_count == 1024 * 128
    assert quant_config.quant_algo is None


def test_tuning_of_an_unreadable_mistral_native_checkpoint_needs_explicit_limits(tmp_path):
    """Neither the native files nor a config.json describe the checkpoint."""
    checkpoint = _mistral_native_checkpoint(tmp_path / "checkpoint", weights=None)
    with pytest.raises(ValueError, match="pass --max_batch_size and --max_num_tokens"):
        _tune(tmp_path, checkpoint)


def _fp8_mistral_native_checkpoint(path, *, hf_config):
    """A Mistral-native checkpoint whose FP8 quantization only params.json declares.

    The quantization a report shows tells whether it read params.json or config.json.
    """
    checkpoint = _mistral_native_checkpoint(path, hf_config=hf_config)
    params = json.loads((checkpoint / "params.json").read_text())
    params["quantization"] = {"qformat_weight": "fp8_e4m3", "qscheme_act": "TENSOR"}
    (checkpoint / "params.json").write_text(json.dumps(params))
    return checkpoint


def _engine_report(checkpoint, checkpoint_format):
    """Return the engine section of the report of a one-request run on `checkpoint`."""
    keeper = StatsKeeper()
    keeper.register_request_perf_item(_perf_item(0, num_out=4, decoding_iteration=3))
    stats = PercentileStats.from_iterable([8])
    metadata = DatasetMetadata(
        isl_stats=stats, osl_stats=stats, seq_len_stats=stats, num_requests=1
    )
    runtime_config = _benchmark_runtime_config("pytorch").model_copy(
        update={
            "model": "mistral-native",
            "model_path": checkpoint,
            "mapping": {"pp_size": 1, "tp_size": 1, "moe_ep_size": 1, "world_size": 1},
        }
    )
    llm_args = {"kv_cache_config": {"dtype": "auto"}, "checkpoint_format": checkpoint_format}
    report = ReportUtility(keeper, metadata, runtime_config, mock.Mock(), llm_args)
    with mock.patch.object(ReportUtility, "_query_gpu_info", return_value=None):
        return report.get_statistics_dict()["engine"]


@pytest.mark.parametrize(
    "hf_config,checkpoint_format", [(False, "mistral"), (True, "mistral_large_3")]
)
def test_report_reads_a_mistral_native_checkpoint_from_its_native_files(
    tmp_path, hf_config, checkpoint_format
):
    """The report reads the checkpoint in the format the LLM loaded, here params.json."""
    checkpoint = _fp8_mistral_native_checkpoint(tmp_path / "checkpoint", hf_config=hf_config)
    engine = _engine_report(checkpoint, checkpoint_format)
    assert engine["dtype"] == "bfloat16"
    assert engine["quantization"] == QuantAlgo.FP8_BLOCK_SCALES
    assert engine["kv_cache_dtype"] is None


def test_report_reads_config_json_in_the_hugging_face_format(tmp_path):
    checkpoint = _fp8_mistral_native_checkpoint(tmp_path / "checkpoint", hf_config=True)
    engine = _engine_report(checkpoint, None)
    assert engine["dtype"] == "bfloat16"
    assert engine["quantization"] is None
