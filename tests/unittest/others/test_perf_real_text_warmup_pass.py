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
"""A real-text serve case's warmup pass must use other prompts, or none, never fail the case.

test_perf.py builds the pass's prompt file from the articles after the measured ones.
Drives MultiMetricPerfTest's dataset generator with a finite fake source and tokenizer.
"""

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import List

import pytest

pytestmark = pytest.mark.cpu_only

_TESTS_DIR = Path(__file__).resolve().parents[2]
_DEFS_DIR = _TESTS_DIR / "integration" / "defs"
# A real-text spec-dec model: 4 measured requests, waves of 2.
_TEST_ID = (
    "nemotron_3.5_lightning_30b_nvfp4_mtp-serve-pytorch-streaming-float4-maxbs:2-maxnt:8192"
    "-input_output_len:128,16-reqs:4-con:2"
)
_INPUT_LEN, _OUTPUT_LEN = 128, 16
_NUM_MEASURED = 4


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _package(name: str, path: Path = None) -> types.ModuleType:
    package = _module(name)
    package.__path__ = [str(path)] if path else []
    return package


class _CharTokenizer:
    """One token per character, so decoded prompts show which article they came from."""

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


class _Source:
    """A cnn_dailymail stand-in with a fixed list of articles; counts its loads."""

    def __init__(self, articles: List[str], error_on_load: int = 0, error: Exception = None):
        self.articles = articles
        self.loads = 0
        self.error_on_load = error_on_load
        self.error = error

    def load_dataset(self, *args, **kwargs):
        self.loads += 1
        if self.loads == self.error_on_load:
            raise self.error
        return [{"article": article} for article in self.articles]


@pytest.fixture
def perf(monkeypatch):
    """test_perf.py with the imports that need a CI test environment stubbed.

    Returns the module and the list its print_warning appends to.
    """
    warnings: List[str] = []
    noop = lambda *args, **kwargs: None  # noqa: E731
    stubs = {
        "defs": _package("defs", _DEFS_DIR),
        "defs.perf": _package("defs.perf", _DEFS_DIR / "perf"),
        "defs.trt_test_alternative": _module(
            "defs.trt_test_alternative",
            is_linux=lambda: True,
            is_windows=lambda: False,
            print_info=noop,
            print_warning=warnings.append,
            print_error=noop,
            Popen=subprocess.Popen,
            cleanup_process_tree=noop,
        ),
        "defs.conftest": _module(
            "defs.conftest",
            get_device_count=lambda: 8,
            llm_models_root=lambda: "/models",
            trt_environment=None,
        ),
        "defs.common": _module("defs.common", get_trt_llm_lib_dir=lambda venv: ""),
        "defs.local_venv": _module("defs.local_venv", PythonVenvRunnerImpl=object),
        "defs.test_list_parser": _module("defs.test_list_parser", parse_test_list=noop),
        "defs.perf.gpu_clock_lock": _module(
            "defs.perf.gpu_clock_lock",
            GPUClockLock=object,
            InvalidGPUMonitoringResultError=RuntimeError,
        ),
        "defs.perf.data_export": _module(
            "defs.perf.data_export",
            GPU_MONITORING_FORMAT_KEYS=[],
            write_csv=noop,
            write_gpu_monitoring_no_test_results=noop,
            write_yaml=noop,
        ),
        "defs.perf.session_data_writer": _module(
            "defs.perf.session_data_writer", SessionDataWriter=object
        ),
        "tensorrt_llm": _package("tensorrt_llm"),
        "tensorrt_llm.llmapi": _package("tensorrt_llm.llmapi"),
        "tensorrt_llm.llmapi.mpi_session": _module(
            "tensorrt_llm.llmapi.mpi_session", get_mpi_world_size=lambda: 1
        ),
        "transformers": _module(
            "transformers",
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _CharTokenizer()),
        ),
    }
    # Set aside any copy imported under the real packages, so the import below binds
    # to the stubs.
    for name in [n for n in sys.modules if n.startswith("defs.perf.") or n == "allowed_configs"]:
        monkeypatch.delitem(sys.modules, name)
    loaded_before = set(sys.modules)
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    # test_perf.py imports allowed_configs from its own directory (__extra_import_path__).
    monkeypatch.syspath_prepend(str(_DEFS_DIR / "perf"))
    module = importlib.import_module("defs.perf.test_perf")
    yield module, warnings
    for name in set(sys.modules) - loaded_before:
        if name.split(".")[0] in ("defs", "allowed_configs"):
            sys.modules.pop(name, None)


def _make_test(perf_module, tmp_path: Path):
    test = perf_module.MultiMetricPerfTest(f"perf/test_perf.py::test_perf[{_TEST_ID}]")
    test.set_runtime_configs(str(tmp_path), str(tmp_path), str(tmp_path), str(tmp_path / "cache"))
    return test


def _use_source(monkeypatch, source: _Source) -> None:
    monkeypatch.setitem(
        sys.modules, "datasets", _module("datasets", load_dataset=source.load_dataset)
    )


def _prompts(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    return [json.loads(line)["input"]["messages"][1]["content"] for line in lines]


def _measured_dataset(perf_module, test, tmp_path: Path) -> Path:
    measured = tmp_path / "dataset_custom.jsonl"
    test.generate_trtllm_custom_dataset(str(measured), _INPUT_LEN, _OUTPUT_LEN, "cnn_dailymail")
    return measured


def test_warmup_pass_is_skipped_when_the_source_has_no_articles_to_spare(
    perf, tmp_path, monkeypatch
) -> None:
    perf_module, warnings = perf
    # Exactly the measured request count of usable articles; the empty one is skipped.
    _use_source(monkeypatch, _Source([f"article {i}" for i in range(_NUM_MEASURED)] + [""]))
    test = _make_test(perf_module, tmp_path)
    measured = _measured_dataset(perf_module, test, tmp_path)
    measured_before = measured.read_text()

    warmup_pass = test._get_serve_warmup_pass(str(tmp_path), _INPUT_LEN, _OUTPUT_LEN, ["client"])

    assert warmup_pass is None
    assert any(warning.startswith("Warmup pass: none") for warning in warnings), warnings
    assert measured.read_text() == measured_before
    assert len(_prompts(measured)) == _NUM_MEASURED


def test_warmup_pass_reads_the_articles_after_the_measured_ones(
    perf, tmp_path, monkeypatch
) -> None:
    perf_module, warnings = perf
    _use_source(monkeypatch, _Source([f"article {i}" for i in range(_NUM_MEASURED + 2)]))
    test = _make_test(perf_module, tmp_path)
    measured = _measured_dataset(perf_module, test, tmp_path)

    warmup_pass = test._get_serve_warmup_pass(str(tmp_path), _INPUT_LEN, _OUTPUT_LEN, ["client"])

    assert warmup_pass is not None and warmup_pass.num_requests == 2
    warmup_file = Path(warmup_pass.cmd[warmup_pass.cmd.index("--dataset-path") + 1])
    warmup_prompts = _prompts(warmup_file)
    assert len(warmup_prompts) == 2
    assert not set(warmup_prompts) & set(_prompts(measured))
    assert all(f"article {i}" in p for i, p in zip(range(_NUM_MEASURED, 6), warmup_prompts))
    assert not warnings


@pytest.mark.parametrize(
    "error", [OSError("cnn_dailymail shard unreadable"), ValueError("unknown split")]
)
def test_other_dataset_errors_still_fail_the_case(perf, tmp_path, monkeypatch, error) -> None:
    perf_module, _ = perf
    # The measured file loads the source first; loading it again for the pass fails.
    source = _Source([f"article {i}" for i in range(8)], error_on_load=2, error=error)
    _use_source(monkeypatch, source)
    test = _make_test(perf_module, tmp_path)
    _measured_dataset(perf_module, test, tmp_path)

    with pytest.raises(type(error)) as excinfo:
        test._get_serve_warmup_pass(str(tmp_path), _INPUT_LEN, _OUTPUT_LEN, ["client"])
    assert excinfo.value is error
