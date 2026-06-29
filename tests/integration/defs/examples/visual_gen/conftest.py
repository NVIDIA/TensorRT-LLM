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

"""Shared fixtures for VisualGen example integration tests."""

import pytest
from defs.trt_test_alternative import check_call


@pytest.fixture(scope="session")
def _visual_gen_deps(llm_venv):
    """Install av, diffusers, and ffmpeg once per VisualGen test session."""
    llm_venv.run_cmd(["-m", "pip", "install", "av"])
    llm_venv.run_cmd(["-m", "pip", "install", "diffusers>=0.37.0"])
    check_call(["apt-get", "update", "-y"], shell=False)
    check_call(["apt-get", "install", "-y", "ffmpeg"], shell=False)
