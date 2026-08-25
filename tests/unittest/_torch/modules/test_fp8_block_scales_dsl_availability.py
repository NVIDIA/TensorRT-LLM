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
"""FP8 block-scales GEMM selection must honour cutlass-DSL availability.

``cute_dsl_fp8_gemm_blackwell`` is registered inside ``cute_dsl_custom_ops``'s
``if IS_CUTLASS_DSL_AVAILABLE:`` block, so on a build without the optional
cutlass DSL the op does not exist. ``FP8BlockScalesLinearMethod`` used to
dispatch to it based on ``use_cute_dsl_blockscaling_mm`` / ``disable_deep_gemm``
and the GPU arch alone, which raises

    AttributeError: '_OpNamespace' 'trtllm' object has no attribute
                    'cute_dsl_fp8_gemm_blackwell'

at the call site. See nvbug 6644645.

The predicate is driven directly over the full truth table, including both
availability values, so the tests are non-vacuous whether or not the DSL is
installed in the running environment -- exercising only the ambient value would
pass trivially in CI, where ``nvidia-cutlass-dsl`` is an unconditional
requirements.txt entry. No GPU is needed: the predicate is pure.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.modules import linear as linear_mod
from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod

# (use_cute_dsl_blockscaling_mm, disable_deep_gemm, dsl_available, expected).
# Either flag alone requests the CuteDSL GEMM, so each must be gated; and
# availability alone must never opt a caller in. Expectations are spelled out
# rather than recomputed from the flags, which would just restate the predicate.
CASES = [
    pytest.param(True, False, True, True, id="requested_via_use_cute_dsl-available"),
    pytest.param(False, True, True, True, id="requested_via_disable_deep_gemm-available"),
    pytest.param(True, True, True, True, id="requested_via_both-available"),
    pytest.param(True, False, False, False, id="requested_via_use_cute_dsl-unavailable"),
    pytest.param(False, True, False, False, id="requested_via_disable_deep_gemm-unavailable"),
    pytest.param(True, True, False, False, id="requested_via_both-unavailable"),
    pytest.param(False, False, True, False, id="not_requested-available"),
    pytest.param(False, False, False, False, id="not_requested-unavailable"),
]


@pytest.mark.parametrize("use_cute_dsl, disable_deep_gemm, dsl_available, expected", CASES)
def test_cute_dsl_gemm_is_selected_only_when_requested_and_available(
    monkeypatch: pytest.MonkeyPatch,
    use_cute_dsl: bool,
    disable_deep_gemm: bool,
    dsl_available: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(linear_mod, "IS_CUTLASS_DSL_AVAILABLE", dsl_available)
    # Stand-in for Linear exposing only the two flags the predicate reads.
    module = SimpleNamespace(
        use_cute_dsl_blockscaling_mm=use_cute_dsl,
        disable_deep_gemm=disable_deep_gemm,
    )

    assert FP8BlockScalesLinearMethod._use_cute_dsl_gemm(module) is expected, (
        "the CuteDSL FP8 GEMM must be selected exactly when a flag requests it and the "
        "cutlass DSL is installed: the op is registered only under "
        "IS_CUTLASS_DSL_AVAILABLE, so selecting it otherwise raises AttributeError at the "
        "call site, while skipping it when available needlessly withholds the kernel"
    )
