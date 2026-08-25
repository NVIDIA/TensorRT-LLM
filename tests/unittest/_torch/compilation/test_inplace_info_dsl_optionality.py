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
"""``inplace_info()`` must not require the optional cutlass DSL.

The ``cute_dsl_*`` ops are registered inside ``cute_dsl_custom_ops``'s
``if IS_CUTLASS_DSL_AVAILABLE:`` block, so without the DSL they do not exist and
``torch.ops.trtllm.<op>.default`` raises ``AttributeError``. ``inplace_info()``
is reached from ``compilation.backend.Backend``, built for *every*
``torch.compile`` run with no cute-DSL opt-in, so naming those ops
unconditionally lets an absent *optional* dependency break torch.compile
outright. See nvbug 6644645.

The tests pin the contract in both directions so they are non-vacuous whether or
not the DSL is installed here -- ``nvidia-cutlass-dsl`` is an unconditional
requirements.txt entry, so CI always has it and a bare ``inplace_info()`` call
could never regress there.
"""

import pytest
import torch

from tensorrt_llm._torch.compilation.utils import get_optional_trtllm_op, inplace_info

# Registered only under the provider's IS_CUTLASS_DSL_AVAILABLE guard. All four
# annotate the same mutated argument.
DSL_GATED_INPLACE_OPS = (
    "cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell",
    "cute_dsl_fp8_bmm_blackwell",
    "cute_dsl_bf16_bmm_blackwell",
    "cute_dsl_bf16_gemm_blackwell",
)
DSL_GATED_MUTATES_ARGS = {1: "output"}


def test_inplace_info_builds_without_the_cutlass_dsl(monkeypatch):
    """The map must build even when every DSL-gated op is missing.

    The absence is simulated rather than assumed, so this fails on the pre-fix
    revision -- which reads ``torch.ops.trtllm.<op>.default`` directly -- even in
    a CI environment that does have the DSL installed. Shadowing the name on the
    ``torch.ops.trtllm`` namespace object is what makes that robust: resolving an
    op caches it there, so patching ``_OpNamespace.__getattr__`` instead would be
    silently bypassed once anything in the session has already touched these ops.
    """
    for op_name in DSL_GATED_INPLACE_OPS:
        monkeypatch.setattr(torch.ops.trtllm, op_name, object(), raising=False)
        assert get_optional_trtllm_op(op_name) is None, "absence not simulated"

    inplace_map = inplace_info()

    assert inplace_map, "inplace_info() must still return the non-DSL entries"
    # Non-optional entries are unaffected by DSL availability.
    assert torch.ops.trtllm.fused_qk_norm_rope.default in inplace_map


@pytest.mark.parametrize("op_name", DSL_GATED_INPLACE_OPS)
def test_dsl_gated_op_keeps_its_entry_when_available(op_name):
    """When the DSL *is* installed the entry must survive unchanged.

    Guards against fixing the AttributeError by simply dropping these ops, which
    would silently lose the in-place annotation and let the copy-removal and
    multi-stream passes mis-handle a mutating cute-DSL kernel.
    """
    op = get_optional_trtllm_op(op_name)
    if op is None:
        pytest.skip(f"{op_name} is not registered; cutlass DSL is absent")

    assert inplace_info().get(op) == DSL_GATED_MUTATES_ARGS, (
        f"{op_name} is registered but its inplace_info() entry is missing or "
        "changed; the copy-removal and multi-stream passes would not know it "
        "mutates its output argument"
    )
