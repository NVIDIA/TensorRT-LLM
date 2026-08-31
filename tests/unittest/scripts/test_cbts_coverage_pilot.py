#!/usr/bin/env python3
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

import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType, TracebackType
from typing import NoReturn, TypeAlias, Union

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PILOT_PATH = REPO_ROOT / "jenkins" / "scripts" / "cbts" / "coverage_pilot.py"
PR_API_URL = "https://api.github.com/repos/NVIDIA/TensorRT-LLM/pulls/123"
JSONValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JSONValue"],
    dict[str, "JSONValue"],
]


@pytest.fixture(scope="module")
def pilot_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coverage_pilot", PILOT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload: JSONValue) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


@pytest.mark.parametrize(
    ("pr_info", "expected"),
    (
        ({"user": {"login": " Pilot-User "}}, ("Pilot-User", "author resolved")),
        ({"user": {}}, ("", "PR API response has no author login")),
        ({}, ("", "PR API response has no user")),
        ([], ("", "PR API response is not an object")),
    ),
)
def test_extract_pr_author(
    pilot_module: ModuleType,
    pr_info: JSONValue,
    expected: tuple[str, str],
) -> None:
    assert pilot_module.extract_pr_author(pr_info) == expected


def test_fetch_pr_author_uses_token(
    pilot_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(request: urllib.request.Request, timeout: int) -> _Response:
        assert request.full_url == PR_API_URL
        assert request.get_header("Authorization") == "Bearer token"
        assert timeout == 15
        return _Response({"user": {"login": "pilot-user"}})

    monkeypatch.setattr(pilot_module.urllib.request, "urlopen", urlopen)

    assert pilot_module.fetch_pr_author(PR_API_URL, token="token") == (
        "pilot-user",
        "author resolved",
    )


def test_fetch_pr_author_rejects_untrusted_url(
    pilot_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_urlopen(
        _request: urllib.request.Request,
        _timeout: int,
    ) -> NoReturn:
        raise AssertionError("untrusted URLs must not be requested")

    monkeypatch.setattr(pilot_module.urllib.request, "urlopen", unexpected_urlopen)

    login, reason = pilot_module.fetch_pr_author("https://example.com/pulls/123", token="token")
    assert not login
    assert reason == "missing or unexpected GitHub PR API URL"


def test_fetch_pr_author_fails_closed_on_api_error(
    pilot_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(_request: urllib.request.Request, *, timeout: int) -> NoReturn:
        assert timeout == 15
        raise urllib.error.URLError("unavailable")

    monkeypatch.setattr(pilot_module.urllib.request, "urlopen", urlopen)

    login, reason = pilot_module.fetch_pr_author(PR_API_URL, token="token")
    assert not login
    assert reason.startswith("PR author lookup failed:")


def test_main_reads_bot_trigger_payload(
    pilot_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    trigger_phrase = json.dumps({"github_pr_api_url": PR_API_URL})
    monkeypatch.setenv("gitlabTriggerPhrase", trigger_phrase)
    monkeypatch.setenv("GITHUB_API_TOKEN", "token")

    def fetch_pr_author(
        pr_api_url: str,
        *,
        token: str,
    ) -> tuple[str, str]:
        assert pr_api_url == PR_API_URL
        assert token == "token"
        return "pilot-user", "author resolved"

    monkeypatch.setattr(pilot_module, "fetch_pr_author", fetch_pr_author)

    assert pilot_module.main([]) == 0
    captured = capfd.readouterr()
    assert captured.out == "pilot-user\n"
    assert "pr_author=pilot-user, reason=author resolved" in captured.err
