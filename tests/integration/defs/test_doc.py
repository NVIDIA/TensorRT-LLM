# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import concurrent.futures
import os
import re
from collections import defaultdict
from urllib.parse import urlparse

import pytest
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

requests.packages.urllib3.disable_warnings(
    requests.packages.urllib3.exceptions.InsecureRequestWarning
)

# Markdown discovery filters. The walker prunes any directory whose name is in
# SKIP_DIR_NAMES or starts with a prefix in SKIP_DIR_PREFIXES, and drops any
# file in SKIP_FILENAMES (e.g., auto-generated attribution files).
SKIP_DIR_NAMES = {"3rdparty", "_deps", "build", "include", "node_modules", ".git"}
SKIP_DIR_PREFIXES = (".venv", "venv")
SKIP_FILENAMES = {
    "ATTRIBUTIONS-Python.md",
    "ATTRIBUTIONS-CPP-x86_64.md",
    "ATTRIBUTIONS-CPP-aarch64.md",
}

# URLs that return 404 at HTTP level but are valid in a browser
# (e.g., GitHub Pages sites using JS redirects)
EXCEPTION_URLS = [
    "https://nvidia.github.io/",
]

HTML_LINK_PATTERN = re.compile(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"')


def _get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _extract_markdown_links(text):
    """Extract markdown links handling nested parentheses."""
    links = []
    i = 0
    while i < len(text):
        start_bracket = text.find("[", i)
        if start_bracket == -1:
            break
        close_bracket = text.find("]", start_bracket)
        if close_bracket == -1 or close_bracket + 1 >= len(text) or text[close_bracket + 1] != "(":
            i = start_bracket + 1
            continue

        open_paren = close_bracket + 1
        depth = 1
        j = open_paren + 1
        close_paren = -1
        while j < len(text) and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = j
            j += 1

        if close_paren != -1:
            url = text[open_paren + 1 : close_paren]
            links.append(url)
            i = close_paren + 1
        else:
            i = open_paren + 1
    return links


def _clean_url(url):
    if url.startswith("<") and url.endswith(">"):
        url = url[1:-1]
    open_count = url.count("(")
    close_count = url.count(")")
    if open_count != close_count:
        if close_count > open_count and url.endswith(")"):
            while close_count > open_count and url.endswith(")"):
                url = url[:-1]
                close_count -= 1
    while url and url[-1] in ".,;:'\"]":
        url = url[:-1]
    return url.strip()


def _find_markdown_files(root_dir):
    markdown_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Prune in-place so os.walk doesn't descend into skipped dirs.
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(SKIP_DIR_PREFIXES)
        ]
        for filename in filenames:
            if filename.lower().endswith(".md"):
                if filename in SKIP_FILENAMES:
                    continue
                markdown_files.append(os.path.join(dirpath, filename))
    return markdown_files


def _extract_urls(file_path):
    """Extract and normalize URLs from a markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")

    url_info_list = []

    for line_num, line in enumerate(lines, 1):
        for url in _extract_markdown_links(line):
            url_info_list.append((_clean_url(url), line_num))
        for match in HTML_LINK_PATTERN.finditer(line):
            url_info_list.append((_clean_url(match.group(1)), line_num))

    normalized = []
    for url, line_num in url_info_list:
        if url.startswith("www."):
            url = "https://" + url
        if not url.startswith(("http://", "https://")):
            continue
        normalized.append((url, line_num))
    return normalized


def _check_url(url_info):
    """Return (is_valid, url, line_num, reason)."""
    url, line_num = url_info

    if url in EXCEPTION_URLS:
        return True, url, line_num, "Known exception URL (skipped validation)"

    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        return False, url, line_num, "Invalid URL format"
    if parsed.netloc in ("localhost",) or parsed.netloc.startswith("127.0.0."):
        return True, url, line_num, "local"
    if "drive.google.com" in parsed.netloc:
        return True, url, line_num, "Google Drive (auth required)"
    if parsed.netloc == "github.com" and ("/blob/" in parsed.path or "/tree/" in parsed.path):
        return True, url, line_num, "GitHub repo-internal ref"

    session = _get_session()
    try:
        resp = session.head(url, timeout=10, allow_redirects=True, verify=False)
        if resp.status_code == 404:
            resp = session.get(url, timeout=10, allow_redirects=True, verify=False, stream=True)
            resp.close()
        if resp.status_code == 404:
            return False, url, line_num, "404 Not Found"
        return True, url, line_num, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        if "Connection" in str(e):
            return True, url, line_num, "connection issue (transient)"
        return False, url, line_num, str(e)
    except Exception as e:
        return False, url, line_num, f"Error: {e}"


def test_url_validity(llm_root):
    """Scan all markdown files in the repo and assert no URLs return 404."""
    md_files = _find_markdown_files(llm_root)
    assert md_files, f"No markdown files found under {llm_root}"

    all_urls = []
    for md_file in md_files:
        for url, line_num in _extract_urls(md_file):
            all_urls.append((url, line_num, md_file))

    if not all_urls:
        pytest.skip("No URLs found in any markdown file")

    # De-duplicate URLs (check each unique URL once, keep all locations for reporting)
    unique_urls = {}
    for url, line_num, md_file in all_urls:
        if url not in unique_urls:
            unique_urls[url] = []
        unique_urls[url].append((md_file, line_num))

    url_items = [(url, 0) for url in unique_urls]  # line_num=0 placeholder

    invalid = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_check_url, item): item for item in url_items}
        for future in concurrent.futures.as_completed(futures):
            is_valid, url, _, reason = future.result()
            if not is_valid:
                for md_file, line_num in unique_urls[url]:
                    invalid.append((md_file, line_num, url, reason))

    if invalid:
        invalid.sort()
        by_file = defaultdict(list)
        for md_file, line_num, url, reason in invalid:
            by_file[md_file].append((line_num, url, reason))
        report_lines = [f"Found {len(invalid)} invalid URL(s) in {len(by_file)} file(s):"]
        for md_file, entries in sorted(by_file.items()):
            report_lines.append(f"{md_file}:")
            for line_num, url, reason in entries:
                report_lines.append(f"  L{line_num} [{reason}] {url}")
        pytest.fail("\n".join(report_lines))
