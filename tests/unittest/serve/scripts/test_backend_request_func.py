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
"""Tests for the trust_env decision of the benchmark client sessions."""

import urllib.request

import pytest

from tensorrt_llm.serve.scripts.backend_request_func import env_changes_request

pytestmark = pytest.mark.cpu_only

PROXY_VARS = ["http_proxy", "https_proxy", "all_proxy", "no_proxy"]
URL = "http://127.0.0.1:8000/v1/completions"


@pytest.fixture
def netrc_path(monkeypatch, tmp_path):
    """Clear the proxy variables and point NETRC at a file that does not exist yet."""
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    # On macOS and Windows, getproxies() also reads the system proxy.
    monkeypatch.setattr(urllib.request, "getproxies", urllib.request.getproxies_environment)
    monkeypatch.setattr(urllib.request, "proxy_bypass", urllib.request.proxy_bypass_environment)
    path = tmp_path / "netrc"
    monkeypatch.setenv("NETRC", str(path))
    return path


def test_no_proxy_and_no_netrc(netrc_path):
    assert not env_changes_request(URL)


def test_proxy_for_the_scheme(netrc_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    assert env_changes_request("http://10.0.0.5:8000/v1/completions")
    assert not env_changes_request("https://10.0.0.5:8000/v1/completions")


def test_no_proxy_excludes_the_host(netrc_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    assert not env_changes_request(URL)


def test_netrc_entry_for_the_host(netrc_path):
    netrc_path.write_text("machine bench.internal login user password secret\n")
    assert env_changes_request("http://bench.internal:8000/v1/completions")
    assert not env_changes_request(URL)
