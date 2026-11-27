# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile coverage for the balanced Rubin GVR main kernel."""

from importlib import import_module

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the balanced Rubin GVR kernel requires SM107",
)


def test_balanced_gvr_main_kernel_compiles_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    gvr = import_module(
        "tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa."
        "glm_5_2_nvfp4__sm_107__high_throughput.kernels.attention.gvr_balanced"
    )
    compile_kernel = gvr.cute.compile
    traced_kernels = []

    def record_compile(kernel, *args, **kwargs):
        assert isinstance(kernel, gvr.GvrMainKernel)
        assert len(args) == 21
        traced_kernels.append(kernel)
        return compile_kernel(kernel, *args, **kwargs)

    monkeypatch.setattr(gvr, "_COMPILE_CACHE", {})
    monkeypatch.setattr(gvr.cute, "compile", record_compile)
    rows = 56
    next_n = 8
    specialization = (1024, 8, 1, 256, 2, True, False, next_n, 0, 3, 2)

    compiled = gvr.get_compiled(specialization, hf=True)

    assert gvr.get_compiled(specialization, hf=True) is compiled
    assert len(traced_kernels) == 1
    kernel = traced_kernels[0]
    assert kernel.kern is not None
    assert (kernel.blk, kernel.u, kernel.minb, kernel.nbs, kernel.kpt) == specialization[:5]
    assert kernel.split and kernel.varlen and kernel.balanced and kernel.hf
    assert (kernel.next_n, kernel.cr_shift, kernel.r_const, kernel.nj) == (
        next_n,
        0,
        3,
        (rows + 31) // 32,
    )
