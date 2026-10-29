# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QA checks for packaged TRTLLM-Gen FP8 sparse MLA on SM100."""

import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from .runner_interface import PythonRunnerInterface


@pytest.mark.parametrize("allocator", ["default", "expandable"])
def test_fp8_sparse_mla_fallback(
    llm_root: str,
    llm_venv: PythonRunnerInterface,
    output_dir: str,
    allocator: str,
) -> None:
    """Run the existing FP8 sparse-MLA matrix with isolated allocator settings."""
    report = Path(output_dir) / f"results-sparse-mla-fallback-{allocator}.xml"
    test_file = Path(llm_root) / "tests/unittest/_torch/attention/sparse/dsa/test_dsa_sparse_mla.py"
    env = {
        "TLLM_FMHA_LIBS": "fallback",
        "PYTORCH_ALLOC_CONF": f"expandable_segments:{allocator == 'expandable'}",
        "PYTORCH_CUDA_ALLOC_CONF": f"expandable_segments:{allocator == 'expandable'}",
    }
    # Set allocator options before the child imports torch or initializes CUDA.
    bootstrap = textwrap.dedent("""\
        import os
        import sys
        import pytest
        import torch
        from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
        from tensorrt_llm._torch.attention.backends.fmha.registry import get_enabled_fmha_lib_classes

        print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA runtime: {torch.version.cuda}", flush=True)
        print(f"TLLM_FMHA_LIBS={os.environ['TLLM_FMHA_LIBS']}; "
              f"PYTORCH_ALLOC_CONF={os.environ['PYTORCH_ALLOC_CONF']}", flush=True)
        assert torch.cuda.get_device_capability(0) == (10, 0), "Requires SM100"
        assert get_enabled_fmha_lib_classes() == [FallbackFmha]
        raise SystemExit(pytest.main(sys.argv[1:]))
        """)
    llm_venv.run_cmd(
        [
            "-c",
            bootstrap,
            f"{test_file}::test_sparse_attention_mla",
            "-k",
            "float8_e4m3fn",
            "-vv",
            "-s",
            "--tb=short",
            f"--junitxml={report}",
        ],
        env=env,
        timeout=1800,
    )
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert len(cases) == 6, f"Expected six FP8 cases, got {len(cases)}"
    for case in cases:
        assert case.find("skipped") is None, f"Skipped: {case.attrib}"
        assert case.find("failure") is None, f"Failed: {case.attrib}"
        assert case.find("error") is None, f"Errored: {case.attrib}"
