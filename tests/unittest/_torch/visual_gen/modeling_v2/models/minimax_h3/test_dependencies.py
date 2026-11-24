# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[7]
_TORCH = "tensorrt_llm._torch"
_VG = f"{_TORCH}.visual_gen"
_V2 = f"{_VG}.modeling_v2"
_H3 = f"{_V2}.models.minimax_h3"
_CATALOGS = (f"{_TORCH}.modeling_v2.catalog", f"{_V2}.catalog")
_RUNTIME = {
    "checkpoints.WeightLoader",
    "config.DiffusionModelConfig",
    "config.DiffusionPipelineConfig",
    "config.create_attention_metadata_state",
    "cuda_graph_runner.resolved_extra_key",
    "mapping.VisualGenMapping",
    "models.modeling.BaseDiffusionModel",
    "output.CudaPhaseTimer",
    "output.PipelineOutput",
    "pipeline.BasePipeline",
    "pipeline.ExtraParamSchema",
    "pipeline.RefSlotSpec",
    "pipeline.RoleSpec",
    "pipeline_registry.PipelineComponent",
    "sleep.PipelineSleepManager",
    "sleep.SleepRestoreMode",
}
_PRIMTS = f"{_TORCH}.attention.backends.prims_ts"
_VG_CUTE = f"{_VG}.cute_dsl_kernels"
_SPARSE_KERNEL = f"{_VG_CUTE}.blackwell.video_sparse_attention"
_H3_PROVIDERS = {
    "components/linear.py": {
        f"{_TORCH}.modules.low_m_gemm.low_m_gemm_dispatcher",
        f"{_TORCH}.modules.low_m_gemm.low_m_gemm_takes_rows",
    },
    "ops/quantized_gemm.py": {
        f"{_TORCH}.kernels.blackwell.fp8_block_scale_low_m_gemm.max_m_for",
        "tensorrt_llm.quantization.utils.fp8_utils.per_token_quant_and_transform",
        "tensorrt_llm.quantization.utils.fp8_utils.unpack_col_major_tma_aligned_packed_tensor",
    },
}
_CATALOG_PROVIDERS = {
    "attention/sol_attention.py": {f"{_VG_CUTE}.blackwell.sol_attn.sol_attn"},
    "attention/sol_predictor.py": {
        f"{_VG}.attention.backends.sparse.sol.kernels.visual_gen_sol_predictor"
    },
    "attention/block_sparse_attention.py": {
        f"{_SPARSE_KERNEL}.CUTE_AVAILABLE",
        f"{_SPARSE_KERNEL}.block_sparse_attn_from_indices_cute",
    },
    "attention/_cute_fmha_launcher.py": {
        f"{_VG_CUTE}.blackwell.attention.BlackwellFusedMultiHeadAttentionForward",
        f"{_VG_CUTE}.blackwell.attention.BlackwellFusedMultiHeadBlockScaledAttentionForward",
        f"{_VG_CUTE}.helpers.fmha_helpers.MaskEnum",
    },
    "attention/prims_dense_attention.py": {
        f"{_PRIMTS}.BatchPrefillTSWrapper",
        f"{_PRIMTS}.VCAttentionConfig",
        f"{_PRIMTS}.vc_attention.quantization.vc_quantize",
        f"{_PRIMTS}.vc_attention.quantization.vc_quantize_fp8",
        f"{_PRIMTS}.vc_attention.quantization.vc_quantize_repair",
        f"{_PRIMTS}.vc_attention.quantization.vc_token_permutation",
    },
    "attention/prims_routed_attention.py": {
        f"{_PRIMTS}.BlockSparseTSWrapper",
        f"{_PRIMTS}.SageAttentionConfig",
        f"{_PRIMTS}.SageAttentionParams",
        f"{_PRIMTS}._block_sparse.config._validate_block_sparse_static_profile",
        f"{_PRIMTS}.sage.config._validate_int8_qk_device",
        f"{_TORCH}.attention.kernels.sage_quant.sage_quantize",
    },
}


def _imports(path: Path) -> Iterator[tuple[int, str]]:
    package = ".".join(path.parent.relative_to(_REPO).parts)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = resolve_name("." * node.level + (node.module or ""), package)
            for alias in node.names:
                yield node.lineno, f"{module}.{alias.name}"


def test_v2_model_and_catalog_import_boundaries() -> None:
    runtime_symbols = {f"{_VG}.{name}" for name in _RUNTIME}
    shared = (*_CATALOGS, f"{_VG}.runtime")
    violations = []
    for package in (_H3, f"{_V2}.catalog"):
        roots = (*shared, _H3) if package == _H3 else shared
        root = _REPO.joinpath(*package.split("."))
        assert root.is_dir(), f"Missing source package: {package}"
        providers = _H3_PROVIDERS if package == _H3 else _CATALOG_PROVIDERS
        for path in sorted(root.rglob("*.py")):
            allowed = runtime_symbols.union(providers.get(path.relative_to(root).as_posix(), ()))
            for lineno, name in _imports(path):
                if not (name == "tensorrt_llm" or name.startswith("tensorrt_llm.")):
                    continue
                if name in allowed or any(
                    name == root or name.startswith(root + ".") for root in roots
                ):
                    continue
                violations.append(f"{path.relative_to(_REPO)}:{lineno} imports {name}")
    assert not violations, "\n".join(violations)
