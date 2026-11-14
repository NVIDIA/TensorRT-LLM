# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp4ep4 target's MoE block (``moe.py``) on one GPU (the checks are ``_kimi_k3_moe_checks``): the gate's
routing check host-side; ``pack_mxfp4_expert`` against the built-in trtllm-gen expert loader at several ranks of
moe TP 4; and one rank's block: its expert window and intermediate share, chunked routed experts equal to their
chunks, and a finite forward that repeats bit for bit."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_moe_checks import (  # noqa: E402
    check_block,
    check_gate_checks_routing,
    check_pack_matches_builtin,
    sm_100f,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        moe as target,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

TARGET = "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4"
needs_sm_100f = pytest.mark.skipif(
    not sm_100f(),
    reason="the trtllm-gen MXFP4 SiTU cubins and the fused route + quant need SM100 or SM103",
)


def test_gate_checks_routing():
    check_gate_checks_routing(target)


@needs_sm_100f
def test_pack_matches_builtin_loader():
    check_pack_matches_builtin(target, 4, (0, 1, 2, 3))


@needs_sm_100f
def test_block():
    check_block(target, TARGET)
