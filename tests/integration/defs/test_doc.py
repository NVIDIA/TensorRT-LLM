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
from urllib.parse import unquote, urlparse

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
TRTLLM_GITHUB_PATH_PATTERN = re.compile(
    r"^/nvidia/tensorrt-llm/(?P<link_type>blob|tree)/"
    r"(?P<git_ref>[^/]+)/(?P<repo_path>.+)$",
    re.IGNORECASE,
)


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
            links.append((url, start_bracket))
            i = close_paren + 1
        else:
            i = open_paren + 1
    return links


def _is_in_inline_code(text, position):
    """Return whether a position is enclosed by a Markdown backtick span."""
    offset = 0
    while offset < position:
        if text[offset] != "`":
            offset += 1
            continue

        delimiter_end = offset
        while delimiter_end < len(text) and text[delimiter_end] == "`":
            delimiter_end += 1
        delimiter = text[offset:delimiter_end]
        closing_offset = text.find(delimiter, delimiter_end)
        if closing_offset == -1:
            return delimiter_end <= position
        if delimiter_end <= position < closing_offset:
            return True
        offset = closing_offset + len(delimiter)
    return False


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


def _clean_markdown_destination(destination):
    """Remove an optional Markdown link title from a destination."""
    destination = destination.strip()
    if destination.startswith("<"):
        closing_bracket = destination.find(">")
        if closing_bracket != -1:
            destination = destination[1:closing_bracket]
    else:
        destination = re.split(r"(?<!\\)\s", destination, maxsplit=1)[0]
        destination = destination.replace(r"\ ", " ")
    return destination.strip()


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


def _extract_links(file_path):
    """Extract and normalize link destinations from a markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")

    url_info_list = []

    code_fence = None
    for line_num, line in enumerate(lines, 1):
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            fence = fence_match.group(1)
            if code_fence is None:
                code_fence = fence
            elif (
                fence[0] == code_fence[0]
                and len(fence) >= len(code_fence)
                and not line[fence_match.end() :].strip()
            ):
                code_fence = None
            continue

        for url, column in _extract_markdown_links(line):
            url_info_list.append(
                (
                    _clean_markdown_destination(url),
                    line_num,
                    code_fence is not None or _is_in_inline_code(line, column),
                )
            )
        for match in HTML_LINK_PATTERN.finditer(line):
            url_info_list.append(
                (
                    _clean_url(match.group(1)),
                    line_num,
                    code_fence is not None or _is_in_inline_code(line, match.start()),
                )
            )

    normalized = []
    for url, line_num, is_in_code in url_info_list:
        if url.startswith("www."):
            url = "https://" + url
        normalized.append((url, line_num, is_in_code))
    return normalized


def _check_relative_repository_link(url_info, source_file, root_dir):
    """Validate a relative link against the repository worktree."""
    url, line_num = url_info
    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc or not parsed.path or os.path.isabs(parsed.path):
        return None

    root_dir = os.path.abspath(root_dir)
    local_path = os.path.abspath(os.path.join(os.path.dirname(source_file), unquote(parsed.path)))
    if os.path.commonpath((root_dir, local_path)) != root_dir:
        return False, url, line_num, "Path escapes the TensorRT-LLM repository"

    candidate_paths = [local_path]
    if not os.path.splitext(local_path)[1]:
        candidate_paths.extend((f"{local_path}.md", f"{local_path}.rst"))
    is_valid = any(os.path.exists(candidate_path) for candidate_path in candidate_paths)
    reason = f"Repository path {'exists' if is_valid else 'not found'}: {local_path}"
    return is_valid, url, line_num, reason


def _check_url(url_info, root_dir):
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
    github_path = parsed.path.lower()
    if (
        parsed.netloc.lower() == "github.com"
        and github_path.startswith("/nvidia/tensorrt-llm/")
        and ("/blob/" in github_path or "/tree/" in github_path)
    ):
        path_match = TRTLLM_GITHUB_PATH_PATTERN.fullmatch(parsed.path)
        if path_match and path_match.group("git_ref") == "main":
            link_type = path_match.group("link_type").lower()
            repo_path = unquote(path_match.group("repo_path"))
            local_path = os.path.abspath(os.path.join(root_dir, repo_path))
            root_dir = os.path.abspath(root_dir)
            if os.path.commonpath((root_dir, local_path)) != root_dir:
                return False, url, line_num, "Path escapes the TensorRT-LLM repository"

            if link_type == "blob":
                is_valid = os.path.isfile(local_path)
                target_type = "file"
            else:
                is_valid = os.path.isdir(local_path)
                target_type = "directory"
            reason = f"TensorRT-LLM {target_type} {'exists' if is_valid else 'not found'} locally"
            return is_valid, url, line_num, reason
        return True, url, line_num, "TensorRT-LLM repo-internal ref"

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


def _fail_on_invalid_links(invalid):
    if not invalid:
        return

    invalid.sort()
    file_count = len({md_file for md_file, _, _, _ in invalid})
    report_lines = [f"Found {len(invalid)} invalid link(s) in {file_count} file(s):"]
    for index, (md_file, line_num, url, reason) in enumerate(invalid, 1):
        report_lines.append(f"{index}. {os.path.abspath(md_file)}:{line_num} [{reason}] {url}")
    pytest.fail("\n".join(report_lines))


def test_http_url_validity(llm_root):
    """Scan all markdown files and validate HTTP URLs."""
    md_files = _find_markdown_files(llm_root)
    assert md_files, f"No markdown files found under {llm_root}"

    # Normalize before comparison so variants (trailing slash, query, fragment,
    # case in scheme/host) all match.
    def _normalize(u):
        p = urlparse(u)
        return (p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"))

    skip_urls = {
        _normalize("https://github.com/NVIDIA/llm-compiler"),
    }
    all_urls = []
    for md_file in md_files:
        for url, line_num, _ in _extract_links(md_file):
            parsed = urlparse(url)
            if parsed.scheme.lower() not in ("http", "https"):
                continue
            if _normalize(url) in skip_urls:
                continue
            all_urls.append((url, line_num, md_file))

    if not all_urls:
        pytest.skip("No HTTP URLs found in any markdown file")

    # De-duplicate URLs (check each unique URL once, keep all locations for reporting)
    unique_urls = {}
    for url, line_num, md_file in all_urls:
        if url not in unique_urls:
            unique_urls[url] = []
        unique_urls[url].append((md_file, line_num))

    url_items = [(url, 0) for url in unique_urls]  # line_num=0 placeholder

    invalid = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_check_url, item, llm_root): item for item in url_items}
        for future in concurrent.futures.as_completed(futures):
            is_valid, url, _, reason = future.result()
            if not is_valid:
                for md_file, line_num in unique_urls[url]:
                    invalid.append((md_file, line_num, url, reason))

    _fail_on_invalid_links(invalid)


def test_relative_path_validity(llm_root):
    """Scan all markdown files and validate relative repository paths."""
    md_files = _find_markdown_files(llm_root)
    assert md_files, f"No markdown files found under {llm_root}"

    invalid = []
    has_relative_path = False
    for md_file in md_files:
        for url, line_num, is_in_code in _extract_links(md_file):
            if is_in_code:
                continue
            result = _check_relative_repository_link((url, line_num), md_file, llm_root)
            if result is None:
                continue
            has_relative_path = True
            is_valid, _, _, reason = result
            if not is_valid:
                invalid.append((md_file, line_num, url, reason))

    if not has_relative_path:
        pytest.skip("No relative repository paths found in any markdown file")

    _fail_on_invalid_links(invalid)
