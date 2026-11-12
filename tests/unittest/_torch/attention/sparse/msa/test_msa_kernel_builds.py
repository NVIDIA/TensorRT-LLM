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


@pytest.fixture
def no_compile(monkeypatch):
    from fmha_sm100 import jit

    def compile_variant(name, params):
        raise AssertionError(f"listing compiled {name}")

    monkeypatch.setattr(jit._variant_manager, "_compile_only", compile_variant)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_listing_replays_the_query_lengths_up_to_the_call_s(fmha_sm100, no_compile, dtype):
    from fmha_sm100 import jit

    page_size = head_dim = 128
    num_heads, num_kv_heads, qo_len = 8, 1, 100
    q = torch.zeros(qo_len, num_heads, head_dim, device="cuda").to(dtype)
    k = torch.zeros(4, num_kv_heads, page_size, head_dim, device="cuda").to(dtype)
    kv_indices = torch.arange(4, dtype=torch.int32, device="cuda")
    options = dict(num_kv_heads=num_kv_heads, page_size=page_size, causal=True)
    run_options = {"kv_indices": kv_indices, "output_maxscore": False}

    def variants(longest):
        shapes = msa_kernel_builds._CONFIG.shapes({"qo_len": longest})
        builds = msa_kernel_builds._package._list(
            q, k, k, (num_heads,), options, 1, run_options, shapes
        )
        names = [name for name, _ in builds]
        assert len(names) == len(set(names))
        return set(names)

    # A call up to 256 rows reaches the 128- and 256-row tiles; one of 8 rows reaches fewer.
    assert len(variants(256)) >= 2
    assert variants(8) < variants(256)
    # Workers compile in parallel: the package's lazily loaded templates are loaded already.
    assert jit._variant_manager._inst_template is not None
    assert jit._variant_manager._run_template is not None


def test_the_indexer_maximum_score_pass_also_lists_its_split_variants(fmha_sm100, no_compile):
    page_size = head_dim = 128
    num_index_heads, qo_len = 4, 128
    q = torch.zeros(qo_len, num_index_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 1, page_size, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_indices = torch.arange(1, dtype=torch.int32, device="cuda")
    options = dict(
        page_size=page_size,
        causal=True,
        num_qo_heads=num_index_heads,
        num_kv_heads=1,
        output_maxscore=True,
    )
    run_options = {"kv_indices": kv_indices, "output_o": False, "output_maxscore": True}

    def variants(config, observed):
        builds = msa_kernel_builds._package._list(
            q, k, k, (), options, 1, run_options, config.shapes(observed)
        )
        return {name for name, _ in builds}

    unsplit = variants(msa_kernel_builds._CONFIG, {"qo_len": qo_len})
    swept = variants(msa_kernel_builds._PROXY_CONFIG, {"qo_len": qo_len, "num_kv_splits": 1})
    assert unsplit < swept


def test_a_rehearsal_run_lists_and_launches_nothing(fmha_sm100, no_compile, monkeypatch):
    builder = kernel_builder.KernelBuilder.get()
    submitted = []
    monkeypatch.setattr(
        builder, "submit", lambda op, key, build, uses_gpu=False: submitted.append(key) or True
    )
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
    modules.add(msa_kernel_builds._K2Q_CSR_MODULE)
    assert len([key for key in submitted if key not in modules]) >= 2
    assert torch.equal(out, torch.zeros_like(out))


def test_an_msa_model_queues_the_modules_its_calls_load(fmha_sm100, monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    builder = kernel_builder.KernelBuilder.get()
    submitted = []
    monkeypatch.setattr(
        builder, "submit", lambda op, key, build, uses_gpu=False: submitted.append(key) or True
    )
    monkeypatch.setattr(msa_kernel_builds._package, "modules_submitted", False)
    config = SimpleNamespace(
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(implementation="msa")
    )
    msa_kernel_builds.queue_builds(config)
    expected = {name for name, _, _ in msa_kernel_builds._MODULES}
    expected.add(msa_kernel_builds._K2Q_CSR_MODULE)
    if IS_FLASHINFER_AVAILABLE:
        expected.add("trtllm_gen_fmha")
    assert set(submitted) == expected


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
