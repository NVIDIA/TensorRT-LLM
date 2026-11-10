# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-arch KDA kernel selection in ``KDAKernelDispatch`` (host-side, no GPU)."""

import pytest

pytest.importorskip("fla")

from tensorrt_llm._torch.modules.kimi_kda import _kda_kernels  # noqa: E402

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("sm_version", "decode_path", "prefill_and_verify_path"),
    [
        (89, "fla", "fla"),
        (90, "optimized", "fla"),
        (100, "optimized", "optimized"),
        (103, "optimized", "optimized"),
        (107, "optimized", "optimized"),
        (120, "fla", "fla"),
    ],
)
def test_kernel_paths_per_arch(monkeypatch, sm_version, decode_path, prefill_and_verify_path):
    monkeypatch.setattr(_kda_kernels, "get_kda_sm_version", lambda: sm_version)
    monkeypatch.setattr(_kda_kernels, "is_intree_prefill_available", lambda: True)
    monkeypatch.setattr(_kda_kernels, "is_intree_mtp_available", lambda: True)

    dispatch = _kda_kernels.KDAKernelDispatch()

    assert dispatch.decode_kernel_path == decode_path
    assert dispatch.prefill_kernel_path == prefill_and_verify_path
    assert dispatch.verify_kernel_path == prefill_and_verify_path


def test_decode_opt_out_keeps_fla_on_hopper(monkeypatch):
    monkeypatch.setattr(_kda_kernels, "get_kda_sm_version", lambda: 90)

    dispatch = _kda_kernels.KDAKernelDispatch(use_optimized_decode=False)

    assert dispatch.decode_kernel_path == "fla"
