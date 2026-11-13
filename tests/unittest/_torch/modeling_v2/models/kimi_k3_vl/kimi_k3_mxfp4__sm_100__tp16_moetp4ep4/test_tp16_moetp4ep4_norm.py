# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp4ep4 target's ``norm.py``, on one GPU: its RMSNorm gives the built-in RMSNorm's bits, and its stream
overlap runs both calls (the checks are ``_kimi_k3_norm``)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget

__extra_import_path__ = [".."]
from _kimi_k3_norm import (  # noqa: E402
    WIDTHS,
    check_rms_norm_fp32_matches_builtin,
    check_rms_norm_matches_builtin,
    check_rms_norm_with_residual_matches_builtin,
    check_run_in_parallel,
    parametrize_rms_norm,
)

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
        norm,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@parametrize_rms_norm()
def test_rms_norm_matches_builtin(rows, width):
    check_rms_norm_matches_builtin(norm, rows, width)


@parametrize_rms_norm()
def test_rms_norm_with_residual_matches_builtin(rows, width):
    check_rms_norm_with_residual_matches_builtin(norm, rows, width)


@pytest.mark.parametrize("width", WIDTHS)
def test_rms_norm_fp32_matches_builtin(width):
    check_rms_norm_fp32_matches_builtin(norm, width)


def test_run_in_parallel():
    check_run_in_parallel(norm)
