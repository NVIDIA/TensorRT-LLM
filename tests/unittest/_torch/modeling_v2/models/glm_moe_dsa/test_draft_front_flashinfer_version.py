# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


def test_pinned_flashinfer_is_verified_for_draft_front() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    match = re.search(
        r"(?m)^flashinfer-python==([^\s;#]+)", (repo_root / "requirements.txt").read_text()
    )
    assert match is not None, "requirements.txt must pin flashinfer-python exactly"
    pinned_version = match.group(1)
    norm_path = (
        repo_root
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/glm_moe_dsa"
        / "glm_5_2_nvfp4__sm_103__tp8/kernels/draft_front/native_norm.py"
    )
    spec = importlib.util.spec_from_file_location("draft_front_native_norm", norm_path)
    assert spec is not None and spec.loader is not None
    native_norm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_norm)
    assert pinned_version in native_norm.VERIFIED_FLASHINFER_VERSIONS, (
        f"FlashInfer {pinned_version} is not verified for draft_front. "
        "Re-run the draft-front bit-exactness check against FlashInfer's CuTe RMSNorm "
        "before extending VERIFIED_FLASHINFER_VERSIONS."
    )
