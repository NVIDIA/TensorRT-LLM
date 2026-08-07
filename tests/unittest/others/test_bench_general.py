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
from types import SimpleNamespace

from tensorrt_llm.bench.benchmark.utils.general import get_settings
from tensorrt_llm.version import __version__


def test_get_settings_reports_installed_version() -> None:
    # Providing max_batch_size + max_num_tokens skips the model-loading path,
    # so this exercises get_settings without a model / GPU. Previously the
    # sw_version was hardcoded to "1.2" (issue #11560).
    params = {
        "max_batch_size": 8,
        "max_num_tokens": 1024,
        "pp": 1,
        "tp": 1,
        "ep": 1,
        "cluster_size": 1,
        "gpus_per_node": 8,
        "backend": "pytorch",
    }
    settings = get_settings(params, SimpleNamespace(), model="dummy-model", model_path=None)
    assert settings["sw_version"] == __version__
