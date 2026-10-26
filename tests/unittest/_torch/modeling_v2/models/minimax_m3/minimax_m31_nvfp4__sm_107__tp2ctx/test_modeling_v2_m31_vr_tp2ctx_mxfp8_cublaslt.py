# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_MXFP8_CUBLASLT (M2, sm_107 pins): the tp2ctx prefill MXFP8 GEMMs run on cuBLASLt with a pinned algorithm
(the op from the build, or from the libs overlay).

The fork's pin table is checked as data: every recorded configuration has splitk_num 1 and no reduction scheme (one
fp32 K-order accumulation per output row, so the bits are the same at any M a band serves), and each (N, K)'s bands
tile [1024, 16384] without gaps or overlap. The GPU checks (bitwise vs every CUTLASS tactic at the band edges, 2^+-12
stress data, determinism, graph == eager) live in the gate kit's MB oracle; in-model, each band runs a first-use
bitwise self-check. CPU.
"""

import ast
from pathlib import Path

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

_TARGET = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2ctx"
_SITES = {(6144, 4096), (12288, 6144), (6144, 6144), (3072, 6144), (6144, 1536)}


def _pins():
    tree = ast.parse((_TARGET / "modeling.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "_MXFP8_CUBLASLT_PINS"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("modeling.py has no _MXFP8_CUBLASLT_PINS")


def test_every_pin_is_row_local():
    for (n, k), bands in _pins().items():
        for lo, hi, config in bands:
            assert len(config) == 9, (n, k, lo, hi)
            algo_id, tile, stages, splitk, reduction = config[:5]
            assert splitk == 1 and reduction == 0, (
                f"N={n} K={k} M in [{lo}, {hi}] would split K: {config}"
            )


def test_bands_tile_the_prefill_rows_per_shape():
    pins = _pins()
    assert set(pins) == _SITES
    for (n, k), bands in pins.items():
        edges = sorted((lo, hi) for lo, hi, _ in bands)
        assert edges[0][0] == 1024 and edges[-1][1] == 16384, (n, k, edges)
        for (_, hi), (lo, _) in zip(edges, edges[1:]):
            assert lo == hi + 1, f"N={n} K={k}: bands {edges} leave a gap or overlap"


def test_the_linears_try_the_pinned_gemm_first_with_a_kill_switch():
    src = (_TARGET / "modeling.py").read_text()
    assert '_MXFP8_CUBLASLT = not m31_disabled("TP2CTX_MXFP8_CUBLASLT")' in src
    assert "pinned = _mxfp8_cublaslt_gemm(self, module, input, prequant, bias)" in src
    # A band is taken only after the first-use bitwise self-check (cublasLtMatmulAlgoCheck rejects every pin).
    assert "why = _mxfp8_cublaslt_self_check(module, band)" in src
    assert "minimax_m31_mxfp8_cublaslt_algo_check" not in src
