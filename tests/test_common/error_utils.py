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

import io
import os
import re
from collections.abc import Iterable

ERROR_KEYWORDS = [
    "RuntimeError",
    "out of memory",
    "ValueError",
    "FileNotFoundError",
    "ConnectionRefusedError",
    "ClientConnectorError",
    "CancelledError",
    "TimeoutError",
    "PMI2_Init failed to initialize",
    "OSError",
]
# Autotuner warmup intentionally probes tactics that can OOM and logs e.g.
# "[Autotuner] Single-pair run failed ... CUDA out of memory ...". Only that
# specific marker+OOM combination is benign: an autotuner-prefixed line with
# any other error keyword (e.g. "[Autotuner] RuntimeError: ...") is a real
# failure and must still be reported.
AUTOTUNER_MARKER = "[Autotuner]"
AUTOTUNER_BENIGN_TEXTS = [
    "out of memory",
]
ENDPOINT_RETRY_MARKER = "is not ready, with exception:"
PIP_TIMEOUT_RETRY = re.compile(
    r"\s*WARNING: Retrying \(Retry\([^)]*\)\) after connection broken by "
    r"'(?:Read|Connect)TimeoutError\(.*\)': \S+\s*"
)


def is_benign_line(line: str) -> bool:
    """True for lines expected during a HEALTHY run despite an error keyword."""
    if ENDPOINT_RETRY_MARKER in line:
        return True
    # pip has scheduled another download attempt. Terminal exceptions and
    # warning lines containing another error keyword must still fail the scan.
    if PIP_TIMEOUT_RETRY.fullmatch(line):
        return not any(keyword in line for keyword in ERROR_KEYWORDS if keyword != "TimeoutError")
    return AUTOTUNER_MARKER in line and any(text in line for text in AUTOTUNER_BENIGN_TEXTS)


SLURM_LOG_TAIL_LINES = 200  # Number of lines to print from slurm job logs
ERROR_CONTEXT_LINES = 100  # Number of lines to print before and after error line


def _error_lines(lines: Iterable[str], first_line_idx: int = 1) -> list[tuple[int, str]]:
    """(line number, line) of each line with an error keyword that is not benign."""
    error_lines = []
    for line_idx, line in enumerate(lines, start=first_line_idx):
        if is_benign_line(line):
            continue
        for keyword in ERROR_KEYWORDS:
            if keyword in line:
                error_lines.append((line_idx, line.strip()))
                break  # Only add line once even if multiple keywords match
    return error_lines


def check_error(file_path: str) -> list[tuple[int, str]]:
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", errors="replace") as f:
        return _error_lines(f)


class LogScanner:
    """Scans growing logs for error keywords, reading only what a log gained since its last scan.

    Reading a whole log at every poll makes each poll cost the size of the log, and on Lustre the
    reader's locks hold up the writer on another node: the srun that collects the ranks' stdout,
    and with it every rank whose stdout pipe is full. A log that shrank is scanned from its start.
    """

    def __init__(self) -> None:
        # log -> (bytes scanned, lines scanned): only complete lines count as scanned.
        self._scanned: dict[str, tuple[int, int]] = {}

    def scan(self, file_path: str) -> list[tuple[int, str]]:
        try:
            size = os.path.getsize(file_path)
        except OSError:
            return []
        offset, num_lines = self._scanned.get(file_path, (0, 0))
        if size < offset:
            offset, num_lines = 0, 0
        with open(file_path, "rb") as f:
            f.seek(offset)
            data = f.read(size - offset)
        # A last line without its newline is checked now, and read again once it is complete.
        complete = data.rfind(b"\n") + 1
        lines = _text_lines(data[:complete])
        error_lines = _error_lines(lines, num_lines + 1)
        if complete < len(data):
            error_lines += _error_lines(_text_lines(data[complete:]), num_lines + len(lines) + 1)
        self._scanned[file_path] = (offset + complete, num_lines + len(lines))
        return error_lines


def _text_lines(data: bytes) -> list[str]:
    """``data``'s lines as reading the file in text mode splits and decodes them."""
    with io.TextIOWrapper(io.BytesIO(data), errors="replace") as text:
        return list(text)


def report_error(
    error_msg: str | Exception, log_files: list[str], tail_lines: int = SLURM_LOG_TAIL_LINES
) -> None:
    # Convert Exception to string if needed
    if isinstance(error_msg, Exception):
        error_msg_str = str(error_msg)
    else:
        error_msg_str = error_msg

    messages = [error_msg_str]

    for log_file in log_files:
        if not os.path.exists(log_file):
            messages.append(f"Failed to read {log_file}: Path doesn't exist")
            continue

        all_lines = None
        error_lines = []
        try:
            with open(log_file, "r", errors="replace") as f:
                all_lines = f.readlines()
            # Scan the buffered lines (iterating the exhausted file handle
            # after readlines() would yield nothing).
            for line_idx, line in enumerate(all_lines, start=1):
                if is_benign_line(line):
                    continue
                for keyword in ERROR_KEYWORDS:
                    if keyword in line:
                        error_lines.append((line_idx, line.strip()))
                        break
        except Exception as e:
            all_lines = None
            error_lines = []
            messages.append(f"Failed to read {log_file}: {e}")

        if error_lines:
            error_lines_str = ", ".join(
                [f"Error line {line_idx}: {line_str}" for line_idx, line_str in error_lines]
            )
            messages.append(error_lines_str)
            # Find the first error line number for context
            first_idx = min(line_idx for line_idx, _ in error_lines)
            if all_lines is not None:
                # Use all_lines for error context
                start_idx = max(0, first_idx - ERROR_CONTEXT_LINES - 1)
                end_idx = min(len(all_lines), first_idx + ERROR_CONTEXT_LINES)
                context_lines = all_lines[start_idx:end_idx]
                messages.append("".join(context_lines))
        if all_lines is not None:
            # ALWAYS include the end of the log, even when a keyword matched
            # above: the first keyword hit may be benign noise while the
            # actual fatal error sits at the end of the file.
            tail_content = "".join(all_lines[-tail_lines:]) if all_lines else "(empty)"
            messages.append(f"--- {log_file} [last {tail_lines} lines] ---")
            messages.append(tail_content)

    raise RuntimeError("\n".join(messages))
