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

import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run_import_check(script: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in [REPO_ROOT, env.get("PYTHONPATH", "")] if path
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        timeout=120,
    )


def test_top_level_import_does_not_eagerly_import_visual_gen():
    result = _run_import_check("""
        import sys
        import tensorrt_llm
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig

        assert "tensorrt_llm.visual_gen" not in sys.modules
        assert tensorrt_llm.KvCacheConfig is KvCacheConfig

        _ = tensorrt_llm.VisualGenParams
        assert "tensorrt_llm.visual_gen" in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_serve_import_does_not_eagerly_import_server_modules():
    result = _run_import_check("""
        import sys
        import tensorrt_llm.serve as serve

        assert "tensorrt_llm.serve.openai_server" not in sys.modules
        assert "tensorrt_llm.serve.openai_disagg_server" not in sys.modules

        _ = serve.OpenAIServer
        assert "tensorrt_llm.serve.openai_server" in sys.modules
        _ = serve.OpenAIDisaggServer
        assert "tensorrt_llm.serve.openai_disagg_server" in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_moe_communication_factory_does_not_eagerly_import_flashinfer_strategy():
    result = _run_import_check("""
        import sys
        import tensorrt_llm
        import tensorrt_llm._torch.modules.fused_moe.communication.communication_factory
        import tensorrt_llm._torch.modules.fused_moe.moe_scheduler

        assert (
            "tensorrt_llm._torch.modules.fused_moe.communication."
            "nvlink_two_sided_flashinfer"
        ) not in sys.modules
    """)
    assert result.returncode == 0, result.stderr
