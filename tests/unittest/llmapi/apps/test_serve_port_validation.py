# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest

from tensorrt_llm.commands.serve import launch_server


def test_launch_server_rejects_nonpositive_port_without_disagg() -> None:
    # --port is user input; a non-positive port without a disagg cluster
    # config must raise a clear ValueError (survives `python -O`) rather than
    # an AssertionError, before any bind/model load.
    with pytest.raises(ValueError, match="Port must be specified"):
        launch_server(
            host="localhost",
            port=0,
            llm_args={"backend": "pytorch", "model": "dummy-model"},
            disagg_cluster_config=None,
        )
