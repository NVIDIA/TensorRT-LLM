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
"""Identity rows of VisualGen names re-exported from moved serve modules.

No kept shim has a VisualGen path, so this table holds interface rows only
(``warns=False``, ``ban_old_path=False``, ``retire_in="not retired"``): names a
package facade keeps re-exporting as its own interface, checked for identity
only. Lives under a ``visual_gen/`` path so that VisualGen-only pull requests
still run it; the machinery's self-tests are in
``tests/unittest/serve/test_shim_identity.py``.
"""

import pytest
from test_common.serve.shims import ShimRow, check_identity

pytestmark = pytest.mark.cpu_only

# The VisualGen request and response models live in
# ``tensorrt_llm.serve.api.visual_gen.protocol``; ``tensorrt_llm.serve.openai_protocol``
# keeps re-exporting them lazily as part of its own interface, for good: no warning, no
# ban on the old path. These rows check identity only.
VISUAL_GEN_SHIM_ROWS: tuple = (
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "ImageGenerationRequest",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "ImageEditRequest",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "ImageObject",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "ImageGenerationResponse",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "MediaReferenceItem",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "VideoGenerationRequest",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "VideoJob",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
    ShimRow(
        "tensorrt_llm.serve.openai_protocol",
        "VideoJobList",
        "tensorrt_llm.serve.api.visual_gen.protocol",
        retire_in="not retired",
        ban_old_path=False,
        warns=False,
    ),
)


def test_visual_gen_shim_identity():
    assert check_identity(VISUAL_GEN_SHIM_ROWS) == []


def test_visual_gen_rows_are_interface_rows():
    kept_shims = [
        f"{row.old_module}.{row.name}"
        for row in VISUAL_GEN_SHIM_ROWS
        if row.warns or row.ban_old_path or row.retire_in != "not retired"
    ]
    assert kept_shims == [], (
        f"{kept_shims} are kept-shim rows, but this table checks identity only. A kept shim "
        "also needs its FutureWarning and old-path ban checked: call check_warnings and "
        "find_old_path_imports from test_common.serve.shims on this table here, and delete "
        "this test."
    )
