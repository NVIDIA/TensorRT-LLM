# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""The family's FP8-output MXFP8 GEMM copies three methods of main's Rubin dense GEMM; they differ from
main's only by the FP8 output.

``kernels/k3r_gemm/mxfp8_fp8_out_gemm.K3Sm107MXFP8Fp8OutGemmKernel`` overrides ``kernel_impl`` (alpha in
BF16 and the FP8 epilogue store for an E4M3 C) and ``_setup_attributes`` (the ``epi_tile_n`` epilogue
subtile) of ``_torch/kernels/rubin/dense_blockscaled_gemm_persistent.Sm107BlockScaledPersistentDenseGemmKernel``;
``ops/gemm/_mxfp8_gemm_fp8_out.K3MXFP8Fp8OutRubinRunner.forward`` is
``custom_ops/cute_dsl_custom_ops.CuteDSLBlockScaledRubinLinear.forward`` with the untuned FP8 tactic, the
10-element tactic, the E4M3 C pointer, the cache key and the family kernel classes. Each copy is compared
with main's method as normalised source (``ast.unparse``, docstring dropped): the line diff must be
exactly the recorded one, so a change to main's method fails here until the copy follows it. Parsing
only: no GPU, no imports.
"""

import ast
import difflib
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[8]
TORCH = ROOT / "tensorrt_llm/_torch"
FAMILY = TORCH / "_experimental/modeling_v2/models/kimi_k3_vl"
MAIN_KERNEL = (
    TORCH / "kernels/rubin/dense_blockscaled_gemm_persistent.py",
    "Sm107BlockScaledPersistentDenseGemmKernel",
)
FAMILY_KERNEL = (FAMILY / "kernels/k3r_gemm/mxfp8_fp8_out_gemm.py", "K3Sm107MXFP8Fp8OutGemmKernel")
MAIN_RUNNER = (TORCH / "custom_ops/cute_dsl_custom_ops.py", "CuteDSLBlockScaledRubinLinear")
FAMILY_RUNNER = (FAMILY / "ops/gemm/_mxfp8_gemm_fp8_out.py", "K3MXFP8Fp8OutRubinRunner")

EXPECTED = {
    "kernel_impl": [
        "-    alpha_value = alpha[0].to(self.c_dtype)",
        "+    alpha_dtype = self.c_dtype",
        "+    if cutlass.const_expr(self.c_dtype == cutlass.Float8E4M3FN):",
        "+        alpha_dtype = cutlass.BFloat16",
        "+    alpha_value = alpha[0].to(alpha_dtype)",
        "-            acc_consumer_state = utils.gemm.sm100.epilogue_tma_store(self, tidx, warp_idx, tma_atom_c, tCtAcc, sC, tCgC, epi_tile, epi_tiles_executed, lambda x: epilogue_op(alpha_value * x), mma_tile_coord_mnl, acc_consumer_state, acc_pipeline, c_pipeline)",
        "+            epilogue_store = utils.gemm.sm100.epilogue_tma_store",
        "+            if cutlass.const_expr(self.c_dtype == cutlass.Float8E4M3FN):",
        "+                epilogue_store = epilogue_tma_store_fp8",
        "+            acc_consumer_state = epilogue_store(self, tidx, warp_idx, tma_atom_c, tCtAcc, sC, tCgC, epi_tile, epi_tiles_executed, lambda x: epilogue_op(alpha_value * x), mma_tile_coord_mnl, acc_consumer_state, acc_pipeline, c_pipeline)",
    ],
    "_setup_attributes": [
        "+    if self.epi_tile_n_override is not None:",
        "+        cta_tile_n = self.cta_tile_shape_mnk[1]",
        "+        if self.cta_tile_shape_mnk[0] < 128 or cta_tile_n % self.epi_tile_n_override != 0:",
        "+            raise ValueError(f'Epilogue subtile N={self.epi_tile_n_override} needs a CTA tile with M >= 128 and N divisible by it, got {self.cta_tile_shape_mnk[:2]}.')",
        "+        self.epi_tile = (cute.make_layout(cute.size(self.epi_tile[0])), cute.make_layout(self.epi_tile_n_override))",
    ],
    "forward": [
        "+    if not (isinstance(tactic, tuple) and len(tactic) > 0):",
        "+        untuned = self.untuned_default_tactic(inputs[0].shape[0], inputs[1].shape[0], inputs[0].shape[1])",
        "+        if untuned is not None:",
        "+            tactic = untuned",
        "+    epi_tile_n = None",
        "-            if len(tactic) not in {6, 7, 8, 9}:",
        "-                raise ValueError('CuteDSL SM107 base tactics must use Boolean use_prefetch with an optional scheduler mode and raster order, followed by split_k')",
        "+            if len(tactic) not in {6, 7, 8, 9, 10}:",
        "+                raise ValueError('CuteDSL SM107 base tactics must use Boolean use_prefetch with an optional scheduler mode, raster order, split_k and epilogue subtile N')",
        "-            if len(tuning_options) == 3:",
        "+            if len(tuning_options) >= 3:",
        "+            if len(tuning_options) == 4:",
        "+                epi_tile_n = tuning_options[3]",
        "-    c_ptr = self.make_cute_dsl_global_pointer(c_tensor, cutlass.BFloat16, 16)",
        "+    c_ptr = self.make_cute_dsl_global_pointer(c_tensor, self.c_dtype, 16)",
        "-    cache_key = (kernel_variant, sf_vec_size, mma_tiler_mnk, mma_inst_shape, preferred_cluster_shape_mn, cluster_shape_mn, swap_ab, use_prefetch, raster_order, scheduler_mode, split_k, locality_domain_half_gemm, self.use_tvm_ffi, max_active_preferred_clusters, max_active_clusters)",
        "+    cache_key = (self.output_dtype, kernel_variant, sf_vec_size, mma_tiler_mnk, mma_inst_shape, preferred_cluster_shape_mn, cluster_shape_mn, swap_ab, use_prefetch, raster_order, scheduler_mode, split_k, locality_domain_half_gemm, self.use_tvm_ffi, max_active_preferred_clusters, max_active_clusters, epi_tile_n)",
        "-            c_ptr = self.make_cute_dsl_global_pointer(c_tensor, cutlass.BFloat16, 16)",
        "+            c_ptr = self.make_cute_dsl_global_pointer(c_tensor, self.c_dtype, 16)",
        "-            gemm = Sm107BlockScaledPersistentDenseGemmMixedClustersKernel(sf_vec_size, mma_inst_shape, mma_tiler_mnk, preferred_cluster_shape_mn, cluster_shape_mn, None if use_prefetch else 0, raster_order=raster_order, scheduler_type=scheduler_type)",
        "+            gemm = self.__class__.mixed_kernel_class(sf_vec_size, mma_inst_shape, mma_tiler_mnk, preferred_cluster_shape_mn, cluster_shape_mn, None if use_prefetch else 0, raster_order=raster_order, scheduler_type=scheduler_type)",
        "-            gemm = self.__class__.kernel_class(sf_vec_size, mma_inst_shape, mma_tiler_mnk, cluster_shape_mn, None if use_prefetch else 0, raster_order=raster_order, scheduler_type=scheduler_type, split_k=split_k)",
        "+            gemm = self.__class__.kernel_class(sf_vec_size, mma_inst_shape, mma_tiler_mnk, cluster_shape_mn, None if use_prefetch else 0, raster_order=raster_order, scheduler_type=scheduler_type, split_k=split_k, epi_tile_n=epi_tile_n)",
    ],
}

COPIES = [
    ("kernel_impl", MAIN_KERNEL, FAMILY_KERNEL),
    ("_setup_attributes", MAIN_KERNEL, FAMILY_KERNEL),
    ("forward", MAIN_RUNNER, FAMILY_RUNNER),
]


def _method(where, name):
    path, class_name = where
    tree = ast.parse(path.read_text(), filename=str(path))
    (cls,) = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name]
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        fn.body = body[1:]
    return ast.unparse(fn).splitlines()


@pytest.mark.parametrize("name, main, family", COPIES, ids=[c[0] for c in COPIES])
def test_copy_differs_from_main_only_by_the_fp8_output(name, main, family):
    diff = [
        line
        for line in difflib.unified_diff(
            _method(main, name), _method(family, name), lineterm="", n=0
        )
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    assert diff == EXPECTED[name]
