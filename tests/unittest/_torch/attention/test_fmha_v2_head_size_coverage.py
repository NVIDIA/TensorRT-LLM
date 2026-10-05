# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static coverage for the Pixtral head-size-104 FMHA v2 kernel."""

import importlib.util
import pathlib
import tempfile
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.cpu_only

_SETUP_PY = pathlib.Path(__file__).resolve().parents[4] / "cpp" / "kernels" / "fmha_v2" / "setup.py"


@pytest.fixture(scope="module")
def kernel_specs() -> tuple[ModuleType, list[Any]]:
    """Return the generator module and specs using the wheel's generation flags."""
    if not _SETUP_PY.is_file():
        pytest.skip(f"fmha_v2 generator not present at {_SETUP_PY}")

    spec = importlib.util.spec_from_file_location("fmha_v2_setup", _SETUP_PY)
    module = importlib.util.module_from_spec(spec)
    captured = []

    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as scratch:
        patch.chdir(scratch)
        patch.setenv("ENABLE_SM89_QMMA", "1")
        patch.setenv("ENABLE_HMMA_FP32", "1")
        patch.setenv("ENABLE_SM100", "1")
        patch.setenv("ENABLE_SM120", "1")
        patch.setenv("GENERATE_CUBIN", "1")
        patch.setenv("GENERATE_CU_TRTLLM", "true")
        patch.setenv("SCHEDULING_MODE", "1")
        patch.delenv("DISABLE_SKIP_SOFTMAX", raising=False)
        spec.loader.exec_module(module)
        module.generate_files = captured.extend
        module.enumerate_kernels()

    assert captured, "enumerate_kernels did not reach generate_files"
    return module, [kspec for kspec, *_ in captured]


@pytest.mark.parametrize("sm", [90, 100, 120])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_pixtral_head_size_has_a_padding_mask_kernel(
    kernel_specs: tuple[ModuleType, list[Any]],
    sm: int,
    dtype: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate the packed-QKV padding-mask kernels Pixtral requests."""
    module, specs = kernel_specs
    matching = [
        kspec
        for kspec in specs
        if kspec.sm == sm
        and kspec.head_size == 104
        and kspec.dtype == dtype
        and kspec.input_layout == module.InputLayout.PACKED_QKV
        and kspec.flash_attention
        and kspec.warp_specialization == (sm == 90)
        and (sm != 120 or not kspec.tiled)
        and not kspec.enable_skip_softmax
    ]
    assert matching, f"No SM{sm} {dtype} packed-QKV FMHA v2 kernel is generated for head size 104"

    monkeypatch.setenv("GENERATE_CUBIN", "1")
    assert any(module.selected_mask_types(kspec)[0] == "1" for kspec in matching), (
        f"Every SM{sm} {dtype} head-size-104 kernel has the padding mask disabled"
    )


def test_pixtral_padding_mask_excludes_other_variants(
    kernel_specs: tuple[ModuleType, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep head-104 padding limited to supported architectures and precision modes."""
    module, specs = kernel_specs
    packed = [
        kspec
        for kspec in specs
        if kspec.head_size == 104
        and kspec.input_layout == module.InputLayout.PACKED_QKV
        and kspec.flash_attention
    ]
    added_arches = [kspec for kspec in packed if kspec.sm in (90, 120)]
    excluded = {
        "fp16_fp32": [kspec for kspec in added_arches if kspec.dtype == "fp16_fp32"],
        "fp8": [kspec for kspec in added_arches if kspec.dtype.startswith("e4m3")],
        "skip_softmax": [kspec for kspec in added_arches if kspec.enable_skip_softmax],
        "sm90_non_warp_specialized": [
            kspec for kspec in added_arches if kspec.sm == 90 and not kspec.warp_specialization
        ],
        "sm120_tiled": [
            kspec
            for kspec in added_arches
            if kspec.sm == 120 and kspec.tiled and kspec.dtype in ("fp16", "bf16")
        ],
        "other_sms": [kspec for kspec in packed if kspec.sm not in (90, 100, 120)],
    }
    monkeypatch.setenv("GENERATE_CUBIN", "1")
    for variant, candidates in excluded.items():
        assert candidates, f"No {variant} specifications were generated to check"
        assert all(module.selected_mask_types(kspec)[0] == "0" for kspec in candidates), (
            f"An excluded {variant} head-size-104 kernel has the padding mask enabled"
        )
