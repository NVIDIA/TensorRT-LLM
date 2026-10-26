# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vendored MSA fmha_sm100 modules the MiniMax-M3.1 targets call are built ahead of time.

cpp/tensorrt_llm/kernels/minimaxM3Msa renders the variant sources (generate_fmha_variants.py) and compiles them, the
plan / reduction / sparse_topk modules and the k2q CSR builder into tensorrt_llm/libs/msa_fmha_sm100. On that path
nothing compiles at run time: a compile logs "msa fmha JIT fallback variant=...".
"""

import importlib
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[6]
_GENERATOR = _REPO / "cpp/tensorrt_llm/kernels/minimaxM3Msa/generate_fmha_variants.py"
_MSA = "tensorrt_llm._torch.attention.backends.fmha.msa_kernels.fmha_sm100"
_PAGE = 128


def _import(name: str):
    """Import a vendored MSA module the way require_msa_module does: the CuTe DSL shim first."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        _install_msa_cutlass_compatibility,
    )

    _install_msa_cutlass_compatibility()
    return importlib.import_module(name)


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
    jit = _import(f"{_MSA}.jit")
    if not jit._PREBUILT_DIR.is_dir():
        pytest.skip("no prebuilt MSA modules (a build without SM10x architectures)")
    if jit._proxy_modes() != (False, False):
        pytest.skip("the prebuilt set holds the SM107 proxy modes")
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
        "k2q/sparse_build_k2q_csr_ext.so",
    ):
        assert (jit._PREBUILT_DIR / module).is_file(), module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads", [1, 2])
def test_m31_proxy_shapes_never_compile(dtype, heads, caplog):
    """The MSA indexer proxy pass (msa_indexer._proxy_max_score) over the query rows the targets prebuild (1..129) and
    beyond, with and without KV splits, loads prebuilt modules only."""
    jit = _prebuilt_or_skip()
    fmha = _import(_MSA)
    caplog.set_level(logging.WARNING)
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
    assert "msa fmha JIT fallback" not in caplog.text
    assert jit._variant_manager._loaded


def test_k2q_builder_is_prebuilt(caplog):
    _prebuilt_or_skip()
    caplog.set_level(logging.WARNING)
    sparse = _import(f"{_MSA}.sparse")
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
    assert "msa fmha JIT fallback" not in caplog.text


def test_e4m3x_prefill_and_index_reader_are_vendored():
    """The E4M3X sparse-prefill forward loads its prebuilt module; the NVFP4 index-K reader imports in-tree."""
    _prebuilt_or_skip()
    bf16x = _import("tensorrt_llm._torch.attention.backends.fmha.msa_kernels.prefill_bf16x.jit")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the prebuilt E4M3X forward is the sm_107a module")
    assert bf16x.prebuilt_available()
    reader = _import(
        "tensorrt_llm._torch.attention.backends.fmha.msa_kernels.fmha_sm100_q8kv8.q8_indexer_interface"
    )
    assert "/msa_kernels/fmha_sm100_q8kv8/" in reader.__file__
    assert hasattr(reader, "BatchDecodeIndexerQ8KV4Wrapper")
