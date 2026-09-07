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
# Confirm that the default backend is changed

import os

from defs.common import venv_check_output

from ..conftest import llm_models_root

model_path = os.path.join(
    llm_models_root(),
    "Qwen3.5-4B",
)


class TestLlmDefaultBackend:
    """
    Check that the default backend is PyTorch for v1.0 breaking change
    """

    def test_llm_args_type_default(self, llm_root, llm_venv):
        # Keep the complete example code here
        from tensorrt_llm.llmapi import LLM, KvCacheConfig, TorchLlmArgs

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
        llm = LLM(model=model_path, kv_cache_config=kv_cache_config)

        # The default backend should be PyTorch
        assert llm.args.backend == "pytorch"
        assert isinstance(llm.args, TorchLlmArgs)

        for output in llm.generate(["Hello, world!"]):
            print(output)

    def test_llm_args_logging(self, llm_root, llm_venv):
        # It should print the backend in the log
        script_path = os.path.join(os.path.dirname(__file__),
                                   "_run_llmapi_llm.py")
        print(f"script_path: {script_path}")

        # Test with pytorch backend
        pytorch_cmd = [
            script_path, "--model_dir", model_path, "--backend", "pytorch"
        ]

        pytorch_output = venv_check_output(llm_venv, pytorch_cmd)

        # Check that pytorch backend keyword appears in logs
        assert "Using LLM with PyTorch backend" in pytorch_output, f"Expected 'pytorch' in logs, got: {pytorch_output}"
