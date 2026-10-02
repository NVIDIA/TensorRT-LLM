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
def sm100_specs() -> tuple[ModuleType, list[Any]]:
    """Return the generator module and its SM100 specs without writing files."""
    if not _SETUP_PY.is_file():
        pytest.skip(f"fmha_v2 generator not present at {_SETUP_PY}")

    spec = importlib.util.spec_from_file_location("fmha_v2_setup", _SETUP_PY)
    module = importlib.util.module_from_spec(spec)
    captured = []

    with pytest.MonkeyPatch.context() as patch, tempfile.TemporaryDirectory() as scratch:
        patch.chdir(scratch)
        patch.setenv("ENABLE_SM100", "1")
        patch.setenv("GENERATE_CUBIN", "1")
        spec.loader.exec_module(module)
        module.generate_files = captured.extend
        module.enumerate_kernels()

    assert captured, "enumerate_kernels did not reach generate_files"
    return module, [kspec for kspec, *_ in captured if kspec.sm == 100]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_sm100_pixtral_head_size_has_a_padding_mask_kernel(
    sm100_specs: tuple[ModuleType, list[Any]], dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generate the packed-QKV padding-mask kernels Pixtral requests."""
    module, specs = sm100_specs
    matching = [
        kspec
        for kspec in specs
        if kspec.head_size == 104
        and kspec.dtype == dtype
        and kspec.input_layout == module.InputLayout.PACKED_QKV
        and kspec.flash_attention
    ]
    assert matching, f"No SM100 {dtype} packed-QKV FMHA v2 kernel is generated for head size 104"

    monkeypatch.setenv("GENERATE_CUBIN", "1")
    assert any(module.selected_mask_types(kspec)[0] == "1" for kspec in matching), (
        "Every SM100 head-size-104 kernel has the padding mask disabled"
    )
