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
"""CPU unit tests for the llmapi Triton backend helpers
(triton_backend/all_models/llmapi/tensorrt_llm/1/helpers.py).

The helpers module imports triton_python_backend_utils, which only exists
inside a Triton Python-backend process; a minimal stub is injected so the
pure-python request/config parsing logic can be tested without a Triton
server or GPU.
"""

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest


class _StubTritonModelException(Exception):
    pass


class _StubTensor:

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def as_numpy(self) -> np.ndarray:
        return self._array

    def is_cpu(self) -> bool:
        return True


class _StubRequest:
    """Request stub: maps input-tensor names to numpy arrays."""

    def __init__(self, tensors: dict | None = None) -> None:
        self._tensors = tensors or {}

    def get_tensor(self, name: str) -> "_StubTensor | None":
        value = self._tensors.get(name)
        return None if value is None else _StubTensor(value)


_MODEL_DIR_HOLDER = {"dir": ""}

_LLMAPI_MODEL_DIR = (Path(__file__).resolve().parents[3] / "triton_backend" /
                     "all_models" / "llmapi" / "tensorrt_llm" / "1")


def _load_by_path(name: str, path: Path) -> types.ModuleType:
    if "triton_python_backend_utils" not in sys.modules:
        stub = types.ModuleType("triton_python_backend_utils")
        stub.TritonModelException = _StubTritonModelException
        stub.get_input_tensor_by_name = (
            lambda request, name: request.get_tensor(name))
        stub.get_model_dir = lambda: _MODEL_DIR_HOLDER["dir"]
        stub.Logger = types.SimpleNamespace(
            log_warning=lambda *_args, **_kwargs: None,
            log_info=lambda *_args, **_kwargs: None,
            log_error=lambda *_args, **_kwargs: None)
        sys.modules["triton_python_backend_utils"] = stub

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_by_path("llmapi_triton_helpers",
                        _LLMAPI_MODEL_DIR / "helpers.py")

_TritonModelException = sys.modules[
    "triton_python_backend_utils"].TritonModelException


class FakeProcessor:
    """Stands in for a tensorrt_llm.sampling_params.LogitsProcessor."""

    def __call__(self, req_id: object, logits: object, ids: object,
                 stream_ptr: object, client_id: object) -> None:
        return None


_FAKE_MODULE_NAME = "_llmapi_helpers_test_processors"


@pytest.fixture(autouse=True)
def fake_processor_module() -> Iterator[types.ModuleType]:
    module = types.ModuleType(_FAKE_MODULE_NAME)
    module.FakeProcessor = FakeProcessor
    module.processor_instance = FakeProcessor()
    module.nested = types.SimpleNamespace(processor=FakeProcessor())
    module.not_callable = object()
    sys.modules[_FAKE_MODULE_NAME] = module
    yield module
    del sys.modules[_FAKE_MODULE_NAME]


def test_load_logits_post_processors_empty() -> None:
    assert helpers.load_logits_post_processors(None) == {}
    assert helpers.load_logits_post_processors({}) == {}


def test_load_logits_post_processors_class_and_instance() -> None:
    processors = helpers.load_logits_post_processors({
        "from_class":
        f"{_FAKE_MODULE_NAME}:FakeProcessor",
        "from_instance":
        f"{_FAKE_MODULE_NAME}:processor_instance",
        "from_nested":
        f"{_FAKE_MODULE_NAME}:nested.processor",
    })
    assert sorted(processors) == ["from_class", "from_instance", "from_nested"]
    # classes are instantiated once at load time
    assert isinstance(processors["from_class"], FakeProcessor)
    assert all(callable(p) for p in processors.values())


@pytest.mark.parametrize(
    "specs, match",
    [
        (["not", "a", "dict"], "must be a mapping"),
        ({
            "p": "no_colon_spec"
        }, "module.path:attribute"),
        ({
            "p": 42
        }, "module.path:attribute"),
        ({
            "p": "definitely_missing_module_xyz:attr"
        }, "cannot import"),
        ({
            "p": ":attr_only"
        }, "module.path:attribute"),
        ({
            "p": f"{_FAKE_MODULE_NAME}:"
        }, "module.path:attribute"),
        ({
            "p": f"{_FAKE_MODULE_NAME}:nested..processor"
        }, "module.path:attribute"),
        ({
            "p": f"{_FAKE_MODULE_NAME}:missing_attr"
        }, "no attribute"),
        ({
            "p": f"{_FAKE_MODULE_NAME}:not_callable"
        }, "non-callable"),
    ],
)
def test_load_logits_post_processors_errors(specs: object, match: str) -> None:
    with pytest.raises(_TritonModelException, match=match):
        helpers.load_logits_post_processors(specs)


def _name_request(name: bytes) -> _StubRequest:
    return _StubRequest({
        "sampling_param_logits_post_processor_name":
        np.asarray([name], dtype=object),
    })


def test_get_logits_post_processor_from_request_absent() -> None:
    processors = {"biaser": FakeProcessor()}
    assert helpers.get_logits_post_processor_from_request(
        _StubRequest(), processors) is None
    # empty name behaves like absent
    assert helpers.get_logits_post_processor_from_request(
        _name_request(b""), processors) is None


def test_get_logits_post_processor_from_request_lookup() -> None:
    processor = FakeProcessor()
    resolved = helpers.get_logits_post_processor_from_request(
        _name_request(b"biaser"), {"biaser": processor})
    assert resolved is processor


def test_get_logits_post_processor_from_request_unknown_name() -> None:
    with pytest.raises(_TritonModelException, match="Unknown logits"):
        helpers.get_logits_post_processor_from_request(
            _name_request(b"missing"), {"biaser": FakeProcessor()})
    # helpful message when nothing is configured at all
    with pytest.raises(_TritonModelException, match="none configured"):
        helpers.get_logits_post_processor_from_request(
            _name_request(b"missing"), {})


def test_model_yaml_plumbing(tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute the exact model.py initialize() composition: build the
    processor registry from model.yaml and keep the section out of the LLM
    engine args."""
    # model.py does `from helpers import ...`; monkeypatch restores both the
    # alias and the model-dir holder after the test.
    monkeypatch.setitem(sys.modules, "helpers", helpers)
    model = _load_by_path("llmapi_triton_model", _LLMAPI_MODEL_DIR / "model.py")
    monkeypatch.setitem(_MODEL_DIR_HOLDER, "dir", str(tmp_path))

    (tmp_path / "model.yaml").write_text(
        "model: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
        "backend: pytorch\n"
        "logits_post_processors:\n"
        f"  biaser: {_FAKE_MODULE_NAME}:FakeProcessor\n"
        "triton_config: {max_batch_size: 0, decoupled: false}\n")

    registry = helpers.load_logits_post_processors(
        model.get_model_config("model.yaml",
                               include_keys=["logits_post_processors"
                                             ]).get("logits_post_processors"))
    assert sorted(registry) == ["biaser"]
    assert isinstance(registry["biaser"], FakeProcessor)

    engine_args = model.get_model_config(
        "model.yaml", exclude_keys=["triton_config", "logits_post_processors"])
    assert "logits_post_processors" not in engine_args
    assert "triton_config" not in engine_args
    assert engine_args["model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # absent section degrades to an empty registry
    (tmp_path / "model.yaml").write_text(
        "model: x\ntriton_config: {max_batch_size: 0, decoupled: false}\n")
    registry = helpers.load_logits_post_processors(
        model.get_model_config("model.yaml",
                               include_keys=["logits_post_processors"
                                             ]).get("logits_post_processors"))
    assert registry == {}
