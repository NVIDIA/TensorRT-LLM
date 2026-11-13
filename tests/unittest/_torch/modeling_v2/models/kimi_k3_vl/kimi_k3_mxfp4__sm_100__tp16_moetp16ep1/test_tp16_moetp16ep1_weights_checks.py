# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target's checkpoint load (``weights.py``), host-side: it rejects the configurations it does not
take (helix, attention DP, pipeline parallelism, FP8 attention, routed experts other than packed MXFP4) with
the reason; the KDA and MLA fused rows read each key through its own cut into its row range (the checks are
``_kimi_k3_loader_parity``)."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_loader_parity import (  # noqa: E402
    REJECTIONS,
    check_fused_rows,
    check_rejects,
    check_rejects_nvfp4_experts,
    check_row_cuts,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        weights,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("case", sorted(REJECTIONS))
def test_rejects(case):
    check_rejects(weights, case)


def test_rejects_nvfp4_experts():
    check_rejects_nvfp4_experts(weights)


def test_fused_rows_land_in_their_row_ranges():
    check_fused_rows(weights)


def test_row_cuts_per_key():
    check_row_cuts(weights)
