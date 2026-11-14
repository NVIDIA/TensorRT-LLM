# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The opt-in ahead-of-time build of the minimax_m3 family's vendored MSA kernels.

With BUILD_MSA_AOT set in the build the family's csrc/msa_aot renders the variant sources
(generate_fmha_variants.py) and compiles them, the plan / reduction / sparse_topk modules, the k2q CSR builder and the
E4M3X prefill into tensorrt_llm/libs/msa_fmha_sm100, and the vendored packages load them from there instead of
compiling at run time. Without that build (the default) the GPU tests skip: everything compiles at run time.
"""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[7]
_FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
_FAM_DIR = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
_GENERATOR = _FAM_DIR / "csrc/msa_aot/generate_fmha_variants.py"
_MSA = f"{_FAM}.kernels.msa_kernels.fmha_sm100"
_PAGE = 128


def _generator():
    spec = importlib.util.spec_from_file_location("generate_fmha_variants", _GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _GENERATOR.is_file(), reason="needs the source tree")
def test_generated_variant_sources_are_current():
    pytest.importorskip("jinja2")
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def _prebuilt_or_skip():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM10x only")
    jit = importlib.import_module(f"{_MSA}.jit")
    if not jit._PREBUILT_DIR.is_dir():
        pytest.skip("no ahead-of-time MSA modules (the default run-time compile build)")
    if jit._proxy_modes() != (False, False):
        pytest.skip("the ahead-of-time set holds the SM107 proxy modes")
    return jit


@pytest.mark.skipif(not _GENERATOR.is_file(), reason="needs the source tree")
def test_every_listed_variant_is_prebuilt():
    jit = _prebuilt_or_skip()
    for variant in _generator().VARIANTS:
        assert (jit._PREBUILT_DIR / variant / f"{variant}.so").is_file(), variant
    for module in (
        "plan/fmha_sm100_plan.so",
        "reduction/fmha_sm100_reduction.so",
        "sparse_topk/sparse_topk_select.so",
        "k2q/sparse_build_k2q_csr_m31_ext.so",
    ):
        assert (jit._PREBUILT_DIR / module).is_file(), module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads", [1, 2])
def test_m31_proxy_shapes_never_compile(dtype, heads, monkeypatch):
    """The MSA indexer proxy pass over the query rows the targets prebuild (1..129) and beyond, with and without KV
    splits, loads ahead-of-time modules only: every run-time compile takes a file lock first."""
    jit = _prebuilt_or_skip()
    fmha = importlib.import_module(_MSA)

    def _no_compile(lock_path):
        raise AssertionError(f"run-time compile under {lock_path}")

    monkeypatch.setattr(jit, "_acquire_file_lock", _no_compile)
    g = torch.Generator(device="cuda").manual_seed(heads)
    for q in (1, 8, 32, 33, 64, 65, 128, 129, 200):
        for splits in (1, 4) if q <= 128 else (1,):
            kv = [5000 + q, 20000 + q]
            pages = [-(-x // _PAGE) for x in kv]
            idx_q = torch.randn(2 * q, heads, 128, generator=g, device="cuda").to(dtype)
            idx_k = torch.randn(sum(pages), 1, _PAGE, 128, generator=g, device="cuda").to(dtype)
            kv_indices = torch.randperm(sum(pages), generator=g, device="cuda").to(torch.int32)
            qo = torch.full((2,), q, dtype=torch.int32)
            kvl = torch.tensor(kv, dtype=torch.int32)
            plan = fmha.fmha_sm100_plan(
                qo,
                kvl,
                heads,
                num_kv_heads=1,
                qo_offset=kvl - qo,
                page_size=_PAGE,
                output_maxscore=True,
                causal=True,
                num_kv_splits=splits,
            )
            _, max_score = fmha.fmha_sm100(
                idx_q,
                idx_k,
                idx_k,
                plan,
                kv_indices=kv_indices,
                output_o=False,
                output_maxscore=True,
                sm_scale=0.088,
            )
            assert torch.isfinite(max_score).any()
    torch.cuda.synchronize()
    assert jit._variant_manager._loaded


def test_k2q_builder_is_prebuilt():
    jit = _prebuilt_or_skip()
    sparse = importlib.import_module(f"{_MSA}.sparse")
    k2q = importlib.import_module(f"{_MSA}.cute.src.sm100.build_k2q_csr")
    assert (
        Path(k2q._ext.__file__).resolve()
        == (jit._PREBUILT_DIR / "k2q" / "sparse_build_k2q_csr_m31_ext.so").resolve()
    )
    q, length = 1024, 9000
    pages = -(-length // _PAGE)
    g = torch.Generator(device="cuda").manual_seed(3)
    q2k = torch.sort(
        torch.randint(0, pages, (2, q, 16), generator=g, device="cuda", dtype=torch.int32), -1
    ).values
    cu_q = torch.tensor([0, q], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    row_ptr, _, _ = sparse.build_k2q_csr(
        q2k.contiguous(),
        cu_q,
        cu_kv,
        _PAGE,
        total_k=length,
        max_seqlen_k=length,
        max_seqlen_q=q,
        total_rows=pages,
        qhead_per_kv=16,
        return_schedule=True,
        schedule_order="lpt",
    )
    torch.cuda.synchronize()
    steps = row_ptr[:, 1:] - row_ptr[:, :-1]
    assert bool((steps >= 0).all()) and 0 < int(row_ptr[0, -1]) <= q * 16


def test_e4m3x_prefill_and_index_reader_are_vendored():
    """The E4M3X sparse-prefill forward loads its ahead-of-time module; the Q8KV4 index reader imports in-tree."""
    _prebuilt_or_skip()
    bf16x = importlib.import_module(f"{_FAM}.kernels.msa_kernels.prefill_bf16x.jit")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the ahead-of-time E4M3X forward is the sm_107a module")
    if not (bf16x._aot_dir()).is_dir():
        pytest.skip("the ahead-of-time build had no CUDA 13.4 / cutlass_msa for the E4M3X forward")
    assert bf16x.prebuilt_available()
    reader = importlib.import_module(
        f"{_FAM}.kernels.msa_kernels.fmha_sm100_q8kv8.q8_indexer_interface"
    )
    assert "/msa_kernels/fmha_sm100_q8kv8/" in reader.__file__
    assert hasattr(reader, "BatchDecodeIndexerQ8KV4Wrapper")
