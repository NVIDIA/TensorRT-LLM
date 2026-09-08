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

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_PATH = _REPO_ROOT / "docs/source/models/supported-models.md"
_INTEGRATION_DEFS = _REPO_ROOT / "tests/integration/defs"
_TEST_LISTS = _REPO_ROOT / "tests/integration/test_lists"

# One end-to-end carrier per architecture is enough here. Backend, constraint
# type, parallelism, and speculative-decoding combinations remain covered by
# the dedicated guided-decoding suites instead of being repeated per model.
_ARCHITECTURE_TESTS = {
    "DeepseekV3ForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestDeepSeekV3Lite::"
        "test_guided_decoding[xgrammar-mtp_nextn=0]"
    ),
    "DeepseekV32ForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestDeepSeekV32::test_fp8_blockscale[baseline]"
    ),
    "GlmMoeDsaForCausalLM": "accuracy/test_glm52.py::TestGLM52NVFP4::test_runtime",
    "DeepseekV4ForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestDeepSeekV4Flash::test_auto_dtype"
    ),
    "Glm4MoeForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestGlm4MoeGuidedDecoding::test_guided_decoding"
    ),
    "Qwen3MoeForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestQwen3_30B_A3B::test_fp8[latency-torch_compile=False]"
    ),
    "Qwen3NextForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestQwen3NextInstruct::test_bf16_4gpu[tep4]"
    ),
    "Qwen3_5MoeForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestQwen3_5_35B_A3B::test_bf16[tp1-CUTLASS]"
    ),
    "Qwen4ExpForCausalLM": (
        "accuracy/test_llm_api_pytorch.py::TestQwen3_8_Flash_Next::test_bf16_tp2_cutlass"
    ),
    "Llama4ForConditionalGeneration": (
        "accuracy/test_llm_api_pytorch.py::TestLlama4GuidedDecoding::test_guided_decoding"
    ),
    "GptOssForCausalLM": ("accuracy/test_llm_api_pytorch.py::TestGPTOSS::test_guided_decoding"),
    "KimiK3ForConditionalGeneration": (
        "accuracy/test_kimi3.py::TestKimiK3::test_w4a16_mxfp4[baseline]"
    ),
    "Glm4MoeLiteForCausalLM": (
        "accuracy/test_llm_api_autodeploy.py::TestGLM4Flash::test_guided_decoding"
    ),
    "NemotronHForCausalLM": ("accuracy/test_llm_api_pytorch.py::TestNemotronV3Nano::test_fp8"),
    "Gemma4ForConditionalGeneration": (
        "accuracy/test_llm_api_pytorch_multimodal.py::TestGemma4_26B_A4B::test_nvfp4_no_mtp"
    ),
    "Gemma4UnifiedForConditionalGeneration": (
        "accuracy/test_llm_api_pytorch_multimodal.py::TestGemma4Unified12B::test_guided_decoding"
    ),
    "Step3p7ForConditionalGeneration": (
        "accuracy/test_llm_api_pytorch.py::TestStep3p7GuidedDecoding::test_guided_decoding"
    ),
    "MiniMaxM3SparseForConditionalGeneration": (
        "accuracy/test_llm_api_pytorch.py::TestMiniMaxM3::test_mxfp8[use_msa=False]"
    ),
}

# Qwen4 stays Untested in the matrix until its GB300 carrier passes in CI. The
# carrier remains scheduled so the status can be promoted without adding a new
# test later.
_PENDING_CI_VALIDATION = {"Qwen4ExpForCausalLM"}


def _guided_decoding_statuses() -> dict[str, str]:
    lines = _MATRIX_PATH.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("| Model Architecture/Feature")
    )
    headers = [cell.strip() for cell in lines[header_index].split("|")[1:-1]]
    guided_decoding_index = headers.index("Guided Decoding")

    statuses = {}
    for line in lines[header_index + 2 :]:
        if not line.startswith("| `"):
            break
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        architecture_match = re.match(r"`([^`]+)`", cells[0])
        assert architecture_match is not None
        statuses[architecture_match.group(1)] = cells[guided_decoding_index]
    return statuses


def _assert_node_exists(nodeid: str) -> None:
    parts = nodeid.split("::")
    source_path = _INTEGRATION_DEFS / parts[0]
    assert source_path.is_file(), f"Missing test source for {nodeid}"

    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_name = parts[1]
    method_name = parts[2].split("[", maxsplit=1)[0]
    test_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    assert test_class is not None, f"Missing class for {nodeid}"
    test_method = next(
        (
            node
            for node in test_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        ),
        None,
    )
    assert test_method is not None, f"Missing test method for {nodeid}"
    method_source = ast.get_source_segment(source, test_method)
    coverage_markers = (
        "assert_guided_decoding_regex",
        "guided_decoding_backend",
        "cover_guided_decoding=True",
    )
    assert any(marker in method_source for marker in coverage_markers), (
        f"Carrier method has no guided-decoding coverage: {nodeid}"
    )


def _scheduled_tests() -> set[str]:
    scheduled_tests = set()
    for path in _TEST_LISTS.rglob("*"):
        if path.name == "waives.txt" or path.suffix not in {".txt", ".yaml", ".yml"}:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.partition("#")[0].strip()
            if entry.startswith("- "):
                entry = entry.removeprefix("- ").strip()
            if entry:
                scheduled_tests.add(entry.split(maxsplit=1)[0])
    return scheduled_tests


def _unconditionally_waived_tests() -> set[str]:
    waives_path = _TEST_LISTS / "waives.txt"
    waived_tests = set()
    for line in waives_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "full:")):
            continue
        waived_tests.add(line.partition(" SKIP")[0])
    return waived_tests


def test_guided_decoding_support_has_scheduled_architecture_coverage():
    statuses = _guided_decoding_statuses()
    documented_as_supported = {
        architecture for architecture, status in statuses.items() if status.startswith("Yes")
    }
    assert _PENDING_CI_VALIDATION <= set(_ARCHITECTURE_TESTS)
    assert documented_as_supported == set(_ARCHITECTURE_TESTS) - _PENDING_CI_VALIDATION
    for architecture in _PENDING_CI_VALIDATION:
        assert statuses[architecture] == "Untested"

    scheduled_tests = _scheduled_tests()
    unconditionally_waived_tests = _unconditionally_waived_tests()
    for nodeid in _ARCHITECTURE_TESTS.values():
        _assert_node_exists(nodeid)
        assert nodeid in scheduled_tests, f"Guided-decoding carrier is not scheduled: {nodeid}"
        assert nodeid not in unconditionally_waived_tests, (
            f"Guided-decoding carrier is unconditionally waived: {nodeid}"
        )
