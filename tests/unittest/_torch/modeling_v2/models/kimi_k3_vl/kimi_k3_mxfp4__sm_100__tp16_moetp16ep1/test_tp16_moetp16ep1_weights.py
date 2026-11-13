# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target's checkpoint load (``weights.py``) on one GPU, against the built-in Kimi K3 model loaded
by the built-in loader: on a scaled-down model at three ranks' shares, every parameter and buffer both hold, before and
after the post-load, and every derived weight hold the same bits; the load reads the keys the built-in plan names, and
fails on a missing, unknown or FP8 key (the checks are ``_kimi_k3_loader_parity``)."""

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_loader_parity import (  # noqa: E402
    LAYOUTS,
    check_load_matches_builtin,
    check_load_rejects_missing_and_unknown_keys,
    sm_100,
)

pytest.importorskip("fla")  # the built-in KDA module the comparison loads into

TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1"
try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E501
        modeling,
        weights,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

TARGET_CLS = modeling.ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1

pytestmark = pytest.mark.skipif(
    not sm_100(), reason="the routed-expert relayout and the KDA derived weights need SM 10.x"
)


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_load_matches_builtin_streamed(layout, tmp_path, monkeypatch):
    check_load_matches_builtin(
        TARGET,
        weights,
        TARGET_CLS,
        layout,
        True,
        tmp_path,
        monkeypatch,
    )


def test_load_matches_builtin_without_index(tmp_path, monkeypatch):
    check_load_matches_builtin(
        TARGET,
        weights,
        TARGET_CLS,
        "tp4_moetp2ep2_r3",
        False,
        tmp_path,
        monkeypatch,
    )


def test_load_rejects_missing_unknown_and_fp8_keys(tmp_path, monkeypatch):
    check_load_rejects_missing_and_unknown_keys(TARGET, weights, TARGET_CLS, tmp_path, monkeypatch)
