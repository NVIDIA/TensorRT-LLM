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
"""Resolve the PR author for the Groovy-owned CBTS coverage pilot policy."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping

import click

_GITHUB_TOKEN_ENV = "GITHUB_API_TOKEN"
_TRIGGER_PHRASE_ENV = "gitlabTriggerPhrase"
_PR_API_URL_RE = re.compile(r"https://api\.github\.com/repos/NVIDIA/TensorRT-LLM/pulls/\d+\Z")
_REQUEST_TIMEOUT_SECONDS = 15


def pr_api_url_from_trigger_phrase(trigger_phrase: str) -> str:
    """Return the bot-provided GitHub PR API URL, or an empty string."""
    if not trigger_phrase:
        return ""
    try:
        payload = json.loads(trigger_phrase)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, Mapping):
        return ""
    value = payload.get("github_pr_api_url")
    return value.strip() if isinstance(value, str) else ""


def extract_pr_author(pr_info: object) -> tuple[str, str]:
    """Return ``(normalized_login, reason)`` for one PR response."""
    if not isinstance(pr_info, Mapping):
        return "", "PR API response is not an object"
    user = pr_info.get("user")
    if not isinstance(user, Mapping):
        return "", "PR API response has no user"
    raw_login = user.get("login")
    login = raw_login.strip() if isinstance(raw_login, str) else ""
    if not login:
        return "", "PR API response has no author login"
    return login, "author resolved"


def fetch_pr_author(
    pr_api_url: str,
    token: str = "",
) -> tuple[str, str]:
    """Fetch one trusted GitHub PR endpoint and return its author."""
    if not _PR_API_URL_RE.fullmatch(pr_api_url):
        return "", "missing or unexpected GitHub PR API URL"

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "tensorrt-llm-cbts-pilot",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(pr_api_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            pr_info = json.loads(response.read())
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ) as error:
        return "", f"PR author lookup failed: {error}"
    return extract_pr_author(pr_info)


@click.command("pilot", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--pr-api-url",
    default=None,
    help="GitHub PR API URL; defaults to the bot trigger payload.",
)
def main(pr_api_url):
    """Resolve the PR author for the Groovy-owned CBTS coverage pilot policy."""
    if pr_api_url is None:
        pr_api_url = pr_api_url_from_trigger_phrase(os.environ.get(_TRIGGER_PHRASE_ENV, ""))
    login, reason = fetch_pr_author(
        pr_api_url,
        token=os.environ.get(_GITHUB_TOKEN_ENV, ""),
    )
    print(
        f"CBTS coverage pilot author lookup: pr_author={login or 'unknown'}, reason={reason}",
        file=sys.stderr,
    )
    # Jenkins consumes stdout; diagnostics stay on stderr in the console log.
    print(login)


if __name__ == "__main__":
    main()
