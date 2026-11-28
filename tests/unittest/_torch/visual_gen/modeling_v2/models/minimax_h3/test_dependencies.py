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
_VISUAL_GEN_SERVICES = {
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
    "pipeline_registry.register_pipeline",
    "sleep.PipelineSleepManager",
    "sleep.SleepRestoreMode",
}
_SHARED_SERVICES = {f"{_VG}.{name}" for name in _VISUAL_GEN_SERVICES} | {
    f"{_TORCH}.autotuner.AutoTuner",
    f"{_TORCH}.autotuner.DynamicTensorSpec",
    f"{_TORCH}.autotuner.TunableRunner",
    f"{_TORCH}.autotuner.TuningConfig",
    f"{_TORCH}.cute_dsl_utils.IS_CUTLASS_DSL_AVAILABLE",
    f"{_TORCH}.cute_dsl_utils.IS_CUTLASS_DSL_RUBIN_AVAILABLE",
    f"{_TORCH}.distributed.AllReduce",
    f"{_TORCH}.flashinfer_utils.IS_FLASHINFER_AVAILABLE",
    f"{_TORCH}.utils.get_last_power_of_2_num_tokens_buckets",
    f"{_TORCH}.utils.get_model_extra_attrs",
    f"{_TORCH}.utils.last_positive_power_of_2",
    f"{_V2}._native.load_native_op",
    "tensorrt_llm._utils.get_sm_version",
    "tensorrt_llm._utils.is_sm_100f",
    "tensorrt_llm._utils.mpi_disabled",
    "tensorrt_llm.bindings.internal.thop.BufferKind",
    "tensorrt_llm.bindings.ipc_nvls_supported",
    "tensorrt_llm.functional.AllReduceStrategy",
    "tensorrt_llm.inputs.utils.load_image",
    "tensorrt_llm.logger.logger",
    "tensorrt_llm.mapping.Mapping",
    "tensorrt_llm.mapping.MpiTopology",
    "tensorrt_llm.models.modeling_utils.QuantConfig",
    "tensorrt_llm.quantization.mode.QuantAlgo",
    "tensorrt_llm.visual_gen.args.AttentionConfig",
    "tensorrt_llm.visual_gen.args.QuantAttentionConfig",
    "tensorrt_llm.visual_gen.params.VisualGenParams",
    "tensorrt_llm.visual_gen.sparse_attention.SkipSoftmaxAttentionConfig",
    "tensorrt_llm.visual_gen.sparse_attention.SolAttentionConfig",
    "tensorrt_llm.visual_gen.sparse_attention.VideoSparseAttentionConfig",
}
_PRIMTS = f"{_TORCH}.attention.backends.prims_ts"
_VG_CUTE = f"{_VG}.cute_dsl_kernels"
_SPARSE_KERNEL = f"{_VG_CUTE}.blackwell.video_sparse_attention"
_H3_PROVIDERS = {
    "components/trtllm_attention_prims.py": {
        f"{_TORCH}.attention.backends.fmha.registry.FMHA_LIBS",
        f"{_TORCH}.attention.backends.fmha.registry.get_enabled_fmha_lib_classes",
    },
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
    "attention/flash_attention_4.py": {
        f"{_TORCH}.cute_dsl_utils.load_flash_attn_cute_forward",
    },
    "attention/trtllm_context_attention.py": {"tensorrt_llm.bindings.internal.thop.run_context"},
    "gemm/gate_up_swiglu_quack.py": {f"{_TORCH}.cute_dsl_utils.install_cutlass_dsl_compatibility"},
    "attention/sol_attention.py": {f"{_VG_CUTE}.blackwell.sol_attn.sol_attn"},
    "attention/deco_attention.py": {f"{_VG_CUTE}.blackwell.deco_attention"},
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
    violations = []
    for package in (_H3, f"{_V2}.catalog"):
        roots = (*_CATALOGS, _H3) if package == _H3 else _CATALOGS
        root = _REPO.joinpath(*package.split("."))
        assert root.is_dir(), f"Missing source package: {package}"
        providers = _H3_PROVIDERS if package == _H3 else _CATALOG_PROVIDERS
        for path in sorted(root.rglob("*.py")):
            allowed = _SHARED_SERVICES.union(providers.get(path.relative_to(root).as_posix(), ()))
            for lineno, name in _imports(path):
                if not (name == "tensorrt_llm" or name.startswith("tensorrt_llm.")):
                    continue
                if name in allowed or any(
                    name == root or name.startswith(root + ".") for root in roots
                ):
                    continue
                violations.append(f"{path.relative_to(_REPO)}:{lineno} imports {name}")
    assert not violations, "\n".join(violations)
