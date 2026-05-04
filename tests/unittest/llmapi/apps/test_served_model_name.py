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
"""Unit tests for multi-name / alias handling in OpenAIServer."""

import json
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from tensorrt_llm.commands.serve import _resolve_served_model_names, _split_served_model_names
from tensorrt_llm.serve.openai_server import OpenAIServer


def test_normalize_single_name():
    assert OpenAIServer._normalize_model_names(["m"]) == ("m", ["m"])


def test_normalize_dedup_preserves_order():
    primary, aliases = OpenAIServer._normalize_model_names(("primary", "a1", "primary", "a2", "a1"))
    assert primary == "primary"
    assert aliases == ["primary", "a1", "a2"]


def test_normalize_directory_path_uses_basename(tmp_path):
    """Primary resolves to basename, but the original path stays a valid alias."""
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    primary, aliases = OpenAIServer._normalize_model_names([str(model_dir), "alias"])
    assert primary == "ckpt"
    assert aliases == ["ckpt", str(model_dir), "alias"]


def test_normalize_directory_path_dedups_basename_and_path(tmp_path):
    """User-provided 'ckpt' alias after a dir path dedups to a single basename."""
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    _, aliases = OpenAIServer._normalize_model_names([str(model_dir), "ckpt", "alias"])
    assert aliases == ["ckpt", str(model_dir), "alias"]


def test_normalize_file_path_not_treated_as_dir(tmp_path):
    """A primary that exists as a FILE keeps its full path (no basename rewrite)."""
    model_file = tmp_path / "ckpt.bin"
    model_file.write_text("")
    primary, aliases = OpenAIServer._normalize_model_names([str(model_file)])
    assert primary == str(model_file)
    assert aliases == [str(model_file)]


def test_normalize_accepts_bare_str():
    """Defensive: a plain str caller is treated as a single-name list."""
    assert OpenAIServer._normalize_model_names("m") == ("m", ["m"])


def test_normalize_empty_raises():
    """Passing an empty sequence (or all-empty-string sequence) must raise."""
    with pytest.raises(ValueError, match="at least one non-empty"):
        OpenAIServer._normalize_model_names([])

    with pytest.raises(ValueError, match="at least one non-empty"):
        OpenAIServer._normalize_model_names(["", ""])


def test_is_model_supported_accepts_original_directory_path(tmp_path):
    """A client that knows the path the server was launched with still gets through."""
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    primary, aliases = OpenAIServer._normalize_model_names([str(model_dir)])
    server = _make_server(primary, aliases)
    assert server._is_model_supported(str(model_dir)) is True
    assert server._is_model_supported("ckpt") is True


def _make_server(primary, aliases):
    server = OpenAIServer.__new__(OpenAIServer)
    server.model = primary
    server.served_model_names = aliases
    return server


@pytest.mark.parametrize("name", ["primary", "alias1", "alias2"])
def test_is_model_supported_known(name):
    server = _make_server("primary", ["primary", "alias1", "alias2"])
    assert server._is_model_supported(name) is True


@pytest.mark.parametrize("name", [None, ""])
def test_is_model_supported_empty_is_ok(name):
    """vLLM-parity: empty/None client-supplied model is treated as valid."""
    server = _make_server("primary", ["primary", "alias1"])
    assert server._is_model_supported(name) is True


def test_is_model_supported_unknown():
    server = _make_server("primary", ["primary", "alias1"])
    assert server._is_model_supported("not-an-alias") is False


def test_check_model_accepts_known_alias():
    server = _make_server("primary", ["primary", "alias1"])
    request = SimpleNamespace(model="alias1")
    assert server._check_model(request) is None


def test_check_model_rejects_unknown_with_404():
    server = _make_server("primary", ["primary", "alias1"])
    request = SimpleNamespace(model="not-an-alias")
    response = server._check_model(request)
    assert response is not None
    assert response.status_code == HTTPStatus.NOT_FOUND
    body = json.loads(response.body)
    assert body["type"] == "NotFoundError"
    assert "not-an-alias" in body["message"]


@pytest.mark.parametrize(
    "flag,expected",
    [
        (None, ["/p/m"]),
        ((), ["/p/m"]),
        (("foo",), ["foo"]),
        (("foo", "bar"), ["foo", "bar"]),
        (("foo", "", "bar"), ["foo", "bar"]),
        (("foo", "bar", "foo"), ["foo", "bar"]),
    ],
)
def test_resolve_served_model_names_cli(flag, expected):
    assert _resolve_served_model_names(flag, {"model": "/p/m"}) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ((), ()),
        (("a",), ("a",)),
        (("a", "b", "c"), ("a", "b", "c")),  # repeated flags stay as-is
        (("a,b,c",), ("a", "b", "c")),  # comma-separated single flag
        (("a,b", "c"), ("a", "b", "c")),  # mixed
        (("a , b , c",), ("a", "b", "c")),  # whitespace around commas
        (("a,,b",), ("a", "b")),  # empty pieces dropped
        (("", "a"), ("a",)),  # empty flag dropped
    ],
)
def test_split_served_model_names(raw, expected):
    assert _split_served_model_names(None, None, raw) == expected
