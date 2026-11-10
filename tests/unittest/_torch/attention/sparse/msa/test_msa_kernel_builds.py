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


def test_a_rehearsal_run_lists_and_launches_nothing(fmha_sm100, monkeypatch):
    from fmha_sm100 import jit

    def no_compile(name, params):
        raise AssertionError(f"compiled {name}")

    monkeypatch.setattr(jit._variant_manager, "_compile_only", no_compile)
    builder = kernel_builder.KernelBuilder.get()
    submitted = []
    monkeypatch.setattr(builder, "submit", lambda op, key, build: submitted.append(key) or True)
    page_size = head_dim = 128
    num_heads, num_kv_heads, qo_len = 8, 1, 100
    q = torch.ones(qo_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.ones(4, num_kv_heads, page_size, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_indices = torch.arange(4, dtype=torch.int32, device="cuda")
    lens = torch.tensor([qo_len], dtype=torch.int32)
    out = torch.full_like(q, float("nan"))
    with builder.rehearsal():
        plan = fmha_sm100.fmha_sm100_plan(
            lens,
            lens.clone(),
            num_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            num_kv_splits=1,
            causal=True,
        )
        fmha_sm100.fmha_sm100(q, k, k, plan, kv_indices=kv_indices, out=out, output_maxscore=False)
    # The run listed its variants, compiled none, and left zeros where its output goes.
    modules = {name for name, _, _ in msa_kernel_builds._MODULES}
    assert len([key for key in submitted if key not in modules]) >= 2
    assert torch.equal(out, torch.zeros_like(out))


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
