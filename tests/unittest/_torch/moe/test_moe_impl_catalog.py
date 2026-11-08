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
"""What is generated from the MoE implementation registry stays in step with it."""

from pathlib import Path

import pytest

from tensorrt_llm._torch.moe.fused_moe.impl_catalog import (
    CATALOG_BEGIN,
    CATALOG_END,
    splice_moe_impl_catalog,
)
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import moe_impl_request

pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GUIDE = _REPO_ROOT / "docs/source/developer-guide/moe-perf-guide.md"


def test_the_perf_guide_embeds_the_current_catalog():
    document = _GUIDE.read_text(encoding="utf-8")
    assert splice_moe_impl_catalog(document) == document, (
        "The MoE implementation catalog in moe-perf-guide.md is stale; run "
        "python3 scripts/generate_moe_impl_catalog.py"
    )


@pytest.mark.parametrize(
    "document",
    [
        "no markers",
        f"{CATALOG_BEGIN}\n{CATALOG_BEGIN}\n{CATALOG_END}",
        f"{CATALOG_END}\n{CATALOG_BEGIN}",
    ],
    ids=["missing", "duplicated", "reversed"],
)
def test_splicing_refuses_a_document_without_one_ordered_block(document):
    with pytest.raises(ValueError, match="marker|must follow"):
        splice_moe_impl_catalog(document)


@pytest.mark.parametrize(
    "tactic, quant, families",
    [
        ("trtllm_gen", "nvfp4", {"TRTLLM": ("trtllm.trtllm_gen.fused_moe.nvfp4",)}),
        ("cutedsl", "nvfp4", {"CUTEDSL", "CUTEDSL_FC12", "DENSEGEMM", "MEGAMOE_CUTEDSL"}),
        ("grouped_gemm.nvfp4", "nvfp4", {"CUTLASS", "CUTEDSL"}),
        ("cutlass.grouped_gemm.nvfp4", "nvfp4", {"CUTLASS"}),
        ("mega_moe", "nvfp4", {"MEGAMOE_CUTEDSL": ("trtllm.cutedsl.mega_moe.nvfp4",)}),
        (
            "mega_moe",
            "w4a8_mxfp4_mxfp8",
            {"MEGAMOE_DEEPGEMM": ("deepgemm.cuda.mega_moe.w4a8_mxfp4_mxfp8",)},
        ),
    ],
)
def test_the_perf_guide_examples_resolve_as_it_says(tactic, quant, families):
    """The tactic section of moe-perf-guide.md quotes these resolutions; update it with them."""
    resolved = moe_impl_request(tactic=tactic).narrowed_to({quant}).families()
    if isinstance(families, dict):
        assert resolved == families
    else:
        assert set(resolved) == families
