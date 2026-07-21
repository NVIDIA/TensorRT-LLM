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
"""Start S3 FD capture before pytest imports the initial conftests."""

import argparse
import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

import pytest

from test_common import s3_output

_CAPTURE_STATE_ATTRIBUTE = "_s3_output_early_capture_state"


@dataclass
class _EarlyCaptureState:
    capture: s3_output.SessionCapture
    claimed: bool = False

    def cleanup(self) -> None:
        self.capture.stop()
        if not self.claimed:
            self.capture.remove_files()


def _is_xdist_worker(config: pytest.Config) -> bool:
    return bool(os.environ.get("PYTEST_XDIST_WORKER")) or hasattr(config, "workerinput")


def _is_xdist_controller(config: pytest.Config) -> bool:
    if _is_xdist_worker(config):
        return False
    namespace = getattr(config, "known_args_namespace", config.option)
    num_processes = getattr(namespace, "numprocesses", None)
    if num_processes in (None, 0, "0"):
        num_processes = getattr(config.option, "numprocesses", None)
    return num_processes not in (None, 0, "0")


def _parse_early_capture_options(
    early_config: pytest.Config, args: list[str]
) -> tuple[str | None, str | None, str, str]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-dir", "-O")
    parser.add_argument("--s3-upload-path")
    parser.add_argument("--s3-capture-mode")
    parsed, _ = parser.parse_known_args(args)

    namespace = early_config.known_args_namespace
    output_path = cast(
        str | None,
        getattr(namespace, "output_dir", None) or parsed.output_dir,
    )
    upload_path = cast(
        str | None,
        getattr(namespace, "s3_upload_path", None)
        or parsed.s3_upload_path
        or os.environ.get("S3_UPLOAD_PATH"),
    )
    capture_mode = cast(
        str,
        getattr(namespace, "s3_capture_mode", None) or parsed.s3_capture_mode or "session",
    )
    pytest_capture = cast(str, getattr(namespace, "capture", "fd"))
    return output_path, upload_path, capture_mode, pytest_capture


def _capture_state(config: pytest.Config) -> _EarlyCaptureState | None:
    return cast(
        _EarlyCaptureState | None,
        getattr(config, _CAPTURE_STATE_ATTRIBUTE, None),
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_load_initial_conftests(
    early_config: pytest.Config, args: list[str]
) -> Generator[None, object, object]:
    output_path, upload_path, capture_mode, pytest_capture = _parse_early_capture_options(
        early_config, args
    )
    if (
        _capture_state(early_config) is not None
        or _is_xdist_controller(early_config)
        or not output_path
        or not upload_path
        or capture_mode != "session"
        or pytest_capture != "no"
    ):
        return (yield)

    capture = s3_output.SessionCapture(output_path)
    capture.start()
    state = _EarlyCaptureState(capture)
    setattr(early_config, _CAPTURE_STATE_ATTRIBUTE, state)
    early_config.add_cleanup(state.cleanup)
    try:
        return (yield)
    finally:
        # MPI runtimes initialized while conftests load retain the spool FD.
        # Restore the parent so pytest configuration output stays visible.
        capture.suspend_parent()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    if _is_xdist_controller(config):
        return
    state = _capture_state(config)
    session_capture = state.capture if state is not None else None
    plugin = s3_output.register_plugin(config, session_capture=session_capture)
    if plugin is not None and state is not None:
        state.claimed = True
