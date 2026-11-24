# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``modeling.module_loading_mode``, the value of the ``K31_MODULE_LOADING`` line each rank logs at
target construction: the driver's module loading mode, ``LAZY`` or ``EAGER``, which follows
``CUDA_MODULE_LOADING`` when the process set it before the driver initialized."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"


def test_module_loading_mode_is_the_drivers() -> None:
    from importlib import import_module

    torch.cuda.init()
    mode = import_module(f"{_PKG}.modeling").module_loading_mode()
    assert mode in ("LAZY", "EAGER")
    requested = os.environ.get("CUDA_MODULE_LOADING", "").upper()
    if requested in ("LAZY", "EAGER"):
        assert mode == requested
