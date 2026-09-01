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
from pathlib import Path

import numpy as np
import pytest


class _StubTritonModelException(Exception):
    pass


class _StubTensor:

    def __init__(self, array):
        self._array = array

    def as_numpy(self):
        return self._array

    def is_cpu(self):
        return True


class _StubRequest:
    """Request stub: maps input-tensor names to numpy arrays."""

    def __init__(self, tensors=None):
        self._tensors = tensors or {}

    def get_tensor(self, name):
        value = self._tensors.get(name)
        return None if value is None else _StubTensor(value)


def _load_helpers():
    if "triton_python_backend_utils" not in sys.modules:
        stub = types.ModuleType("triton_python_backend_utils")
        stub.TritonModelException = _StubTritonModelException
        stub.get_input_tensor_by_name = (
            lambda request, name: request.get_tensor(name))
        stub.Logger = types.SimpleNamespace(
            log_warning=lambda *_args, **_kwargs: None)
        sys.modules["triton_python_backend_utils"] = stub

    helpers_path = (Path(__file__).resolve().parents[3] / "triton_backend" /
                    "all_models" / "llmapi" / "tensorrt_llm" / "1" /
                    "helpers.py")
    spec = importlib.util.spec_from_file_location("llmapi_triton_helpers",
                                                  helpers_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()

_TritonModelException = sys.modules[
    "triton_python_backend_utils"].TritonModelException


class FakeProcessor:
    """Stands in for a tensorrt_llm.sampling_params.LogitsProcessor."""

    def __call__(self, req_id, logits, ids, stream_ptr, client_id):
        return None


_FAKE_MODULE_NAME = "_llmapi_helpers_test_processors"


@pytest.fixture(autouse=True)
def fake_processor_module():
    module = types.ModuleType(_FAKE_MODULE_NAME)
    module.FakeProcessor = FakeProcessor
    module.processor_instance = FakeProcessor()
    module.nested = types.SimpleNamespace(processor=FakeProcessor())
    module.not_callable = object()
    sys.modules[_FAKE_MODULE_NAME] = module
    yield module
    del sys.modules[_FAKE_MODULE_NAME]


def test_load_logits_post_processors_empty():
    assert helpers.load_logits_post_processors(None) == {}
    assert helpers.load_logits_post_processors({}) == {}


def test_load_logits_post_processors_class_and_instance():
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
            "p": f"{_FAKE_MODULE_NAME}:missing_attr"
        }, "no attribute"),
        ({
            "p": f"{_FAKE_MODULE_NAME}:not_callable"
        }, "non-callable"),
    ],
)
def test_load_logits_post_processors_errors(specs, match):
    with pytest.raises(_TritonModelException, match=match):
        helpers.load_logits_post_processors(specs)


def _name_request(name):
    return _StubRequest({
        "sampling_param_logits_post_processor_name":
        np.asarray([name], dtype=object),
    })


def test_get_logits_post_processor_from_request_absent():
    processors = {"biaser": FakeProcessor()}
    assert helpers.get_logits_post_processor_from_request(
        _StubRequest(), processors) is None
    # empty name behaves like absent
    assert helpers.get_logits_post_processor_from_request(
        _name_request(b""), processors) is None


def test_get_logits_post_processor_from_request_lookup():
    processor = FakeProcessor()
    resolved = helpers.get_logits_post_processor_from_request(
        _name_request(b"biaser"), {"biaser": processor})
    assert resolved is processor


def test_get_logits_post_processor_from_request_unknown_name():
    with pytest.raises(_TritonModelException, match="Unknown logits"):
        helpers.get_logits_post_processor_from_request(
            _name_request(b"missing"), {"biaser": FakeProcessor()})
    # helpful message when nothing is configured at all
    with pytest.raises(_TritonModelException, match="none configured"):
        helpers.get_logits_post_processor_from_request(
            _name_request(b"missing"), {})
