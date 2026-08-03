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

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "model_file",
    [
        "modeling_deepseekv3.py",
        "modeling_glm.py",
        "modeling_exaone_moe.py",
    ],
)
def test_pre_mlp_nvfp4_fusion_guards_unquantized_dense_mlp(model_file: str) -> None:
    source = (_REPO_ROOT / "tensorrt_llm" / "_torch" / "models" / model_file).read_text()
    start = source.index("    def forward_mlp")
    end = source.index("        hidden_states = self.mlp", start)
    pre_mlp_branch = source[start:end]

    guard = "if self.mlp.gate_up_proj.has_nvfp4:"
    scale_access = "scale=self.mlp.gate_up_proj.input_scale"

    assert guard in pre_mlp_branch
    assert pre_mlp_branch.index(guard) < pre_mlp_branch.index(scale_access)
