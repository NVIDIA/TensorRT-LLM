# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fmha_sm100 through KernelBuilder: a call signature's variants listed by replaying the
package's plan and run, and built where and under the lock the package's own lookup uses."""

import pytest
import torch

from tensorrt_llm._torch import kernel_builder
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_kernel_builds
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
    require_msa_module,
)


@pytest.fixture
def fmha_sm100(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 (Blackwell) required")
    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) not importable")
    monkeypatch.setattr(kernel_builder.KernelBuilder, "_instance", kernel_builder.KernelBuilder())
    module = require_msa_module()
    if msa_kernel_builds._package is None:
        pytest.skip("TLLM_KERNEL_BUILDER=0")
    return module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_listing_replays_every_query_length_without_compiling(fmha_sm100, monkeypatch, dtype):
    from fmha_sm100 import jit

    def no_compile(name, params):
        raise AssertionError(f"listing compiled {name}")

    monkeypatch.setattr(jit._variant_manager, "_compile_only", no_compile)
    page_size = head_dim = 128
    num_heads, num_kv_heads, qo_len = 8, 1, 100
    q = torch.zeros(qo_len, num_heads, head_dim, device="cuda").to(dtype)
    k = torch.zeros(4, num_kv_heads, page_size, head_dim, device="cuda").to(dtype)
    kv_indices = torch.arange(4, dtype=torch.int32, device="cuda")
    options = dict(num_kv_heads=num_kv_heads, page_size=page_size, num_kv_splits=1, causal=True)
    builds = msa_kernel_builds._package._list(
        q, k, k, (num_heads,), options, {"kv_indices": kv_indices, "output_maxscore": False}
    )
    names = [name for name, _ in builds]
    assert len(names) == len(set(names))
    # A query tile of 128 rows and one of 256 cover at least two variants.
    assert len(names) >= 2
    # Workers compile in parallel: the package's lazily loaded templates are loaded already.
    assert jit._variant_manager._inst_template is not None
    assert jit._variant_manager._run_template is not None


def test_kernels_are_built_once_where_the_package_loads_them(fmha_sm100, monkeypatch, tmp_path):
    from fmha_sm100 import jit

    compiled = []

    def compile_variant(name, params, cache_dir):
        compiled.append(name)
        (cache_dir / f"{name}.so").write_bytes(b"")

    def compile_plan(cache_dir):
        compiled.append("plan")
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "fmha_sm100_plan.so").write_bytes(b"")

    monkeypatch.setattr(jit, "_built_dir", lambda name, library: tmp_path / name)
    monkeypatch.setattr(jit._variant_manager, "_compile_only", compile_variant)
    monkeypatch.setattr(jit, "_do_compile_plan", compile_plan)
    package = msa_kernel_builds._package
    assert package._build("v", {}) is True
    assert package._build("v", {}) is False
    assert package.build_module("plan", "fmha_sm100_plan.so", "_do_compile_plan") is True
    assert package.build_module("plan", "fmha_sm100_plan.so", "_do_compile_plan") is False
    assert compiled == ["v", "plan"]
