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
"""Test execution for TensorRT-LLM CI stages, driven by jenkins/scripts/run_tests.py.

Modules:
    render: split the test-db list into this shard's regular and isolated tests.
    commands: build the pytest commands for isolated runs and reruns.
    execution: run pytest, rerun eligible failures, and record each run's outcome.
    results: count, summarize and merge the JUnit XMLs.
    console: banners and phase summaries in the stage log.

The modules import ``test_rerun`` from jenkins/scripts, so that directory must be
on ``sys.path``; it is when run_tests.py is the entry point.
"""


class SetupError(Exception):
    """The stage could not run its tests (collection failed or found nothing)."""
