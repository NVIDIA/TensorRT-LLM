# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return the process-wide override registries a model may fill at construction to their defaults.

Some models install replacements of shared kernels and helpers (samplers, speculative glue, logits
widening, autotuner bundles) into module-level registries when they are built. Those registries
outlive the model, so the engine resets them before it builds a model: a model that installs
nothing then runs the stock paths even when an earlier model in this process installed some.
"""


def reset_process_overrides() -> None:
    """Set every model override registry back to its default (no override)."""
    from tensorrt_llm._torch import flashinfer_autotune_cache
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.models import modeling_dflash
    from tensorrt_llm._torch.modules import logits_processor
    from tensorrt_llm._torch.speculative import dspark
    from tensorrt_llm._torch.speculative import interface as spec_interface

    from .perf_metrics_manager import disarm_gpu_timestamps
    from .sampler.ops import flashinfer as flashinfer_ops

    spec_interface.set_cheap_padding_rows(False)
    spec_interface.set_chain_speculative_sampling_override(None)
    spec_interface.set_chain_speculative_sampling_slot_override(None)
    spec_interface.set_padding_rows_fill_override(None)
    flashinfer_ops.set_softmax_override(None)
    flashinfer_ops.set_top_p_renorm_override(None)
    flashinfer_ops.set_sampling_from_probs_override(None)
    dspark.set_ancestral_glue_override(None)
    dspark.set_ancestral_block_override(None)
    logits_processor.set_logits_widen_override(None)
    modeling_dflash.set_dflash_merge_future_override(None)
    flashinfer_autotune_cache.use_bundled_file(None)
    AutoTuner.get().bundle_reporter = None
    disarm_gpu_timestamps()


def arm_moe_flashinfer_fp4_quantize(model) -> int:
    """Opt the TRT-LLM MoE op backends of ``model``'s modules in to FlashInfer NVFP4 quantization
    (RuntimeHooks.moe_flashinfer_fp4_quantize); returns how many took it. Each MoE leaf holds its
    own backend instance, so another model in the process keeps the TRT op."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm.logger import logger

    armed = 0
    for module in model.modules():
        backend = getattr(module, "op_backend", None)
        if isinstance(backend, TRTLLMOpBackend) and backend.enable_flashinfer_fp4_quantize():
            armed += 1
    logger.info(f"MoE NVFP4 activation quantization through FlashInfer: {armed} MoE backends")
    return armed
