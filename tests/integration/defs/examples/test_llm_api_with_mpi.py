# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from defs.common import venv_mpi_check_call
from defs.conftest import llm_models_root


def test_llm_api_single_gpu_with_mpirun(llmapi_example_root, llm_venv):
    qwen_model_root = os.path.join(llm_models_root(), "Qwen3", "Qwen3-0.6B")
    src_dst_dict = {
        qwen_model_root: f"{llm_venv.get_working_directory()}/Qwen3/Qwen3-0.6B",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    summary_cmd = [f"{llmapi_example_root}/quickstart_example.py"]

    venv_mpi_check_call(llm_venv, ["mpirun", "-n", "1", "--allow-run-as-root"],
                        summary_cmd)
