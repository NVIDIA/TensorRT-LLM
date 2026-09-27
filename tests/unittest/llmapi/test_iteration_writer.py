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

from unittest.mock import MagicMock, call, patch

import pytest

from tensorrt_llm.bench.benchmark.utils.processes import IterationWriter

pytestmark = pytest.mark.cpu_only


def test_iteration_writer_creates_parent_before_start(tmp_path):
    log_path = tmp_path / "missing" / "nested" / "iterations.jsonl"
    process = MagicMock()

    def assert_parent_exists():
        assert log_path.parent.is_dir()
        assert log_path.is_file()

    process.start.side_effect = assert_parent_exists
    process.is_alive.return_value = False

    with patch("tensorrt_llm.bench.benchmark.utils.processes.Process", return_value=process):
        writer = IterationWriter(log_path)
        with writer.capture():
            pass

    process.start.assert_called_once()


def test_iteration_writer_bounds_process_shutdown(tmp_path):
    process = MagicMock()
    process.is_alive.side_effect = [True, True]

    with patch("tensorrt_llm.bench.benchmark.utils.processes.Process", return_value=process):
        writer = IterationWriter(tmp_path / "iterations.jsonl")
        with writer.capture():
            pass

    assert process.join.call_args_list == [call(timeout=5), call(timeout=5), call()]
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
