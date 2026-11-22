# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

import torch

from tensorrt_llm._errors import ConfigInputError, UnsupportedError
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import (
    HfCheckpointLoader, checkpoint_io_policy_metadata)
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
    HfWeightLoader
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.weight_sharing import (
    ArtifactIdentity, PostTransformConfigIdentity,
    PostTransformQualificationDecision, SourceIdentity)
from tensorrt_llm._utils import (ENABLE_MULTI_DEVICE, get_sm_version,
                                 local_mpi_comm, mpi_disabled,
                                 str_dtype_to_torch)
from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig,
                                          ExecutorMemoryType,
                                          ModelExpressConfig, MultimodalConfig,
                                          PrefillCudaGraphBackend,
                                          SparseAttentionConfig, TorchLlmArgs)
from tensorrt_llm.llmapi.llm_utils import (_resolve_kv_cache_manager_v2_auto,
                                           _resolve_transceiver_runtime_auto,
                                           apply_model_defaults_to_llm_args)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.quantization.utils.fp4_utils import float4_e2m1x2

from ...llmapi.llm_args import CheckpointIOPolicy, LoadFormat
from ..kernel_builder import KernelBuilder
from ..locality_domain.policy import LocalityDomainPolicy
from ..memory.monitor import claims as monitor_claims
from ..memory.monitor import names as monitor_names
from ..mmap_utils import (page_cache_bytes, release_file_page_cache,
                          wait_for_page_migration_to_settle)
from ..model_config import ModelConfig
from ..modeling_v2 import modeling_v2_resolve_on_all_ranks
from ..models import AutoModelForCausalLM
from ..models.checkpoints import memory_init, model_hooks
from ..models.checkpoints.base_weight_mapper import BaseWeightMapper
from ..models.checkpoints.checkpoint_catalog import CheckpointCatalog
from ..models.checkpoints.gms import model_loading as gms_loading
from ..models.checkpoints.mx import model_loading as mx_loading
from ..models.checkpoints.weight_load_plan import WeightLoadPlan
from ..models.modeling_utils import (DecoderModelForCausalLM,
                                     ModelLoaderMetricNames,
                                     get_registered_model_class, timing_metric)
from ..modules.embedding import Embedding, LMHead
from ..modules.low_m_gemm import LOW_M_GEMM_ACTIVE, prepare_low_m_gemm
from ..moe.fused_moe.moe_load_balancer import (BaseMoeLoadBalancer,
                                               maybe_create_moe_load_balancer)
from ..virtual_memory import RestoreMode
from .config_utils import (is_hybrid_linear, is_minimax_m3, is_mla,
                           resolve_auto_ssm_cache_dtype,
                           supports_fp4_mla_attention, uses_fp4_mla_attention,
                           validate_kimi_kda_state_dtype)

_KV_CACHE_MAP = {
    "fp8": QuantAlgo.FP8.value,
    "fp8_ds_mla": QuantAlgo.FP8.value,
    "nvfp4": QuantAlgo.NVFP4.value,
    "auto": "auto",
    # Mixed-precision KV: the first/last tokens and attention use the second
    # dtype. float16/bfloat16 leave the KV cache unquantized and must equal
    # the model dtype (checked in validate_and_set_kv_cache_quant).
    "mixed_nvfp4_fp8": QuantAlgo.FP8.value,
    "mixed_nvfp4_float16": None,
    "mixed_nvfp4_bfloat16": None,
}
_VALID_KV_CACHE_DTYPES = ("fp8", "fp8_ds_mla", "nvfp4", "auto",
                          "mixed_nvfp4_fp8", "mixed_nvfp4_float16",
                          "mixed_nvfp4_bfloat16")
_SHADOW_WEIGHT_LOAD_PLAN_ENV = "TRTLLM_SHADOW_WEIGHT_LOAD_PLAN"
_WEIGHT_LOAD_THREADS_ENV = "TLLM_WEIGHT_LOAD_THREADS"

# TorchLlmArgs fields that only the model classes listing them in
# `model_specific_llm_args` implement.
_MODEL_SPECIFIC_LLM_ARGS = ("replicate_embedding",
                            "mnnvl_allreduce_fp8_payload")


def _reject_unsupported_model_llm_args(llm_args: TorchLlmArgs,
                                       model_cls: type) -> None:
    supported = getattr(model_cls, "model_specific_llm_args", frozenset())
    for name in _MODEL_SPECIFIC_LLM_ARGS:
        if getattr(llm_args, name) and name not in supported:
            raise ValueError(f"{name} is not supported by {model_cls.__name__}")


def _validate_encoder_attention_backend_support(
        model_cls: type | None, mm_config: MultimodalConfig | None) -> None:
    """Reject unsupported encoder selections before allocating model weights."""
    if mm_config is None or not mm_config.encoder_attention_backend:
        return
    supported = getattr(model_cls, "supported_encoder_attention_backends", {})
    model_name = model_cls.__name__ if model_cls is not None else "The model"
    for encoder, backend in mm_config.encoder_attention_backend.items():
        if not supported or encoder in supported and not supported[encoder]:
            raise ValueError(
                f"The {encoder} encoder's attention backend cannot be changed "
                f"for {model_name}")
        if encoder not in supported:
            raise ValueError(f"Unknown encoder {encoder!r} for {model_name}; "
                             f"available encoders: {sorted(supported)}")
        if backend not in supported[encoder]:
            raise ValueError(
                f"The {encoder} encoder does not support attention backend {backend!r}; "
                f"supported backends: {sorted(supported[encoder])}")
        if backend == "FLASHINFER":
            from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE

            if not IS_FLASHINFER_AVAILABLE:
                raise ValueError(
                    f"The {encoder} encoder requested FLASHINFER, but FlashInfer is unavailable"
                )


def _validate_encoder_attention_backend_enabled(
        mm_config: MultimodalConfig | None,
        pretrained_config: object,
        *,
        disable_mm_encoder: bool = False) -> None:
    """Reject selections for encoders that the effective model config omits."""
    if mm_config is None or not mm_config.encoder_attention_backend:
        return
    if disable_mm_encoder:
        # Text-only runs, such as trtllm-bench without --modality, reuse
        # serving configs. A skipped encoder never applies the selection, and
        # models that ignore disable_mm_encoder still build it with the setting.
        logger.info(
            "multimodal_config.encoder_attention_backend is not applied "
            "to encoders skipped by disable_mm_encoder=True")
        return
    for encoder in mm_config.encoder_attention_backend:
        if getattr(pretrained_config, f"{encoder}_config", None) is None:
            raise ValueError(
                f"The {encoder} encoder is not enabled in the model config")


def _validate_and_adjust_mamba_snapshot_config(config: ModelConfig,
                                               llm_args: TorchLlmArgs) -> None:
    """Validate snapshot reuse after the model and V2 setting are resolved."""
    if not is_hybrid_linear(config.pretrained_config):
        return

    kv_cache_config = llm_args.kv_cache_config
    state_config = kv_cache_config.mamba_state_config
    has_additional_snapshots = bool(
        state_config.additional_snapshot_offsets_from_start
        or state_config.additional_snapshot_offsets_from_end)
    if (has_additional_snapshots
            and kv_cache_config.use_kv_cache_manager_v2 is not True):
        raise ConfigInputError(
            "Mamba additional snapshot offsets require "
            "kv_cache_config.use_kv_cache_manager_v2=True after resolving "
            "the model configuration.")

    has_periodic_snapshots = state_config.periodic_snapshot_interval > 0
    if (kv_cache_config.enable_block_reuse and not has_periodic_snapshots
            and not has_additional_snapshots):
        logger.warning(
            "Disabling KV cache block reuse for the hybrid Mamba model "
            "because no Mamba state snapshot policy is configured. Set "
            "kv_cache_config.mamba_state_config.periodic_snapshot_interval "
            "to a positive value or provide additional snapshot offsets to "
            "enable block reuse.")
        kv_cache_config.enable_block_reuse = False


def effective_mamba_ssm_cache_dtype(config: ModelConfig,
                                    mamba_ssm_cache_dtype: str) -> torch.dtype:
    """The recurrent-state cache dtype the engine runs with."""
    if mamba_ssm_cache_dtype == "auto":
        return resolve_auto_ssm_cache_dtype(config.pretrained_config,
                                            config.torch_dtype)
    return str_dtype_to_torch(mamba_ssm_cache_dtype)


def validate_and_set_mamba_ssm_cache_dtype(
        config: ModelConfig,
        mamba_ssm_cache_dtype: str,
        mamba_ssm_stochastic_rounding: bool = False,
        mamba_ssm_philox_rounds: int = 10) -> None:
    mamba_ssm_cache_dtype = effective_mamba_ssm_cache_dtype(
        config, mamba_ssm_cache_dtype)
    validate_kimi_kda_state_dtype(config.pretrained_config,
                                  mamba_ssm_cache_dtype,
                                  mamba_ssm_stochastic_rounding)

    config.quant_config.mamba_ssm_cache_dtype = mamba_ssm_cache_dtype
    config.quant_config.mamba_ssm_stochastic_rounding = mamba_ssm_stochastic_rounding
    config.quant_config.mamba_ssm_philox_rounds = mamba_ssm_philox_rounds


def validate_and_set_kv_cache_quant(model_config: ModelConfig,
                                    pyt_kv_cache_dtype: str) -> None:
    logger.info(
        f'Validating KV Cache config against kv_cache_dtype="{pyt_kv_cache_dtype}"'
    )
    # Quantization from hf_quant_config.json
    kv_cache_quant = model_config.quant_config.kv_cache_quant_algo
    # PyTorch configuration quantization
    valid_pyt_quant = bool(pyt_kv_cache_dtype in _VALID_KV_CACHE_DTYPES)
    mapped_pyt_quant = _KV_CACHE_MAP.get(pyt_kv_cache_dtype, None)
    unquantized_mixed_precision_high_dtype = mapped_pyt_quant is None
    sparse_attention_config = model_config.sparse_attention_config
    if (pyt_kv_cache_dtype.startswith("mixed_")
            and sparse_attention_config is not None
            and sparse_attention_config.algorithm in ("dsa", "deepseek_v4")):
        # These backends keep the history in their own pool, in the mixed-precision low
        # format; a drafter inherits this KV dtype from the model, as for every KV dtype.
        # TODO: keep drafters that are not MTP in the mixed-precision high dtype on every backend.
        mapped_pyt_quant = _KV_CACHE_MAP[pyt_kv_cache_dtype.split("_")[1]]

    effective_kv_cache_quant = (kv_cache_quant if pyt_kv_cache_dtype == "auto"
                                else mapped_pyt_quant)
    # MiniMax-M3 MSA keeps NVFP4 only on its sparse layers, which MSA and
    # Triton serve; its dense and draft layers stay FP8 for trtllm-gen.
    sparse_config = model_config.sparse_attention_config
    uses_minimax_m3_msa = (is_minimax_m3(sparse_config)
                           and sparse_config.implementation == "msa")
    # DSA never hands trtllm-gen its NVFP4 latent cache: attention reads FP8
    # rows gathered from it, or PrimTS reads its records in place.
    uses_dsa_nvfp4_latent = (sparse_config is not None
                             and sparse_config.algorithm == "dsa"
                             and not pyt_kv_cache_dtype.startswith("mixed_"))
    if (effective_kv_cache_quant in (QuantAlgo.NVFP4, QuantAlgo.NVFP4.value)
            and not supports_fp4_mla_attention(model_config)
            and not getattr(getattr(model_config, "modeling_v2_target", None),
                            "supports_nvfp4_kv_cache", False)
            and not uses_minimax_m3_msa and not uses_dsa_nvfp4_latent
            and torch.cuda.is_available() and get_sm_version() == 107):
        if pyt_kv_cache_dtype.startswith("mixed_"):
            raise ValueError(
                f'kv_cache_config.dtype="{pyt_kv_cache_dtype}" keeps the history '
                "in NVFP4, which sparse MLA does not support on SM107.")
        logger.warning(
            "NVFP4 KV cache is not supported by trtllm-gen on SM107; "
            "using FP8 KV cache instead.")
        model_config.quant_config.kv_cache_quant_algo = QuantAlgo.FP8.value
        if model_config.quant_config_dict is not None:
            for layer_quant_config in model_config.quant_config_dict.values():
                layer_quant_config.kv_cache_quant_algo = QuantAlgo.FP8.value
        return

    # If we're letting the checkpoint dictate the quant with auto, simply
    # return and do not modify the checkpoint.
    if pyt_kv_cache_dtype == "auto":
        logger.info(
            f'KV cache quantization set to "{pyt_kv_cache_dtype}". Using '
            "checkpoint KV quantization.")
        return

    # If we have an invalid quantization, simply raise an exception.
    if not valid_pyt_quant:
        raise ValueError(
            "Overriding KV cache quantization with an invalid type "
            f'"llm_args.KvCacheConfig.dtype="{pyt_kv_cache_dtype}" '
            f'Accepted types are "{_VALID_KV_CACHE_DTYPES}".')

    # An unquantized mixed-precision high dtype has to be the model dtype; KVCacheManagerV2
    # would otherwise reject it after the weights are loaded.
    if unquantized_mixed_precision_high_dtype:
        mixed_precision_high_dtype = str_dtype_to_torch(
            pyt_kv_cache_dtype.rsplit("_", 1)[1])
        if model_config.torch_dtype != mixed_precision_high_dtype:
            raise ValueError(
                f'llm_args.KvCacheConfig.dtype="{pyt_kv_cache_dtype}" keeps '
                f"the first/last tokens in {mixed_precision_high_dtype}, but the model dtype "
                f"is {model_config.torch_dtype}.")

    # If we get to this point we have a valid explicit quantization setting.
    # Explicit kv_cache_dtype requests force override the checkpoint setting.
    if kv_cache_quant is not None and mapped_pyt_quant != kv_cache_quant:
        logger.warning("Overriding checkpoint KV cache quantization "
                       f'"{kv_cache_quant}" with llm_args.KvCacheConfig.dtype='
                       f'"{pyt_kv_cache_dtype}".')

    # Apply explicit override from kv_cache_config.dtype.
    model_config.quant_config.kv_cache_quant_algo = mapped_pyt_quant
    # MIXED_PRECISION checkpoints carry per-layer QuantConfigs in
    # quant_config_dict; modules built from them (e.g. attention) must agree
    # with the global config on the KV element size, otherwise the KV pool is
    # allocated with the overridden dtype while attention layers read/write
    # with the checkpoint dtype -> out-of-bounds access.
    if model_config.quant_config_dict is not None:
        for layer_quant_config in model_config.quant_config_dict.values():
            layer_quant_config.kv_cache_quant_algo = mapped_pyt_quant


def validate_fp4_mla_config(model_config: ModelConfig,
                            llm_args: TorchLlmArgs) -> None:
    """Validate FP4 MLA before model construction and KV-cache allocation."""
    if not (model_config.sparse_attention_config is None
            and is_mla(model_config.pretrained_config)
            and model_config.quant_config.quant_mode.has_fp4_kv_cache()):
        return
    if not supports_fp4_mla_attention(model_config):
        raise UnsupportedError(
            "Dense FP4 MLA requires the TRTLLM attention backend.")
    from ..attention.backends.fp4_mla.cache_manager import \
        is_fp4_mla_context_only_cache
    from ..attention.backends.fp4_mla.config import validate_fp4_mla_backend
    from ._util import is_disagg_enabled

    sm_version = get_sm_version()
    mapping = model_config.mapping
    context_only = is_fp4_mla_context_only_cache(
        is_disagg_enabled(llm_args.cache_transceiver_config))
    validate_fp4_mla_backend(sm_version,
                             context_only=context_only,
                             helix=mapping.has_cp_helix())
    if mapping.cp_size != 1 and not mapping.has_cp_helix():
        raise UnsupportedError(
            "FP4 MLA supports context parallelism only through Helix.")
    if llm_args.kv_cache_config.use_kv_cache_manager_v2 is False:
        raise ConfigInputError("FP4 MLA requires use_kv_cache_manager_v2=True.")

    spec_config = model_config.spec_config
    if spec_config is not None and spec_config.spec_dec_mode.use_one_engine():
        spec_dec_mode = spec_config.spec_dec_mode
        # A standalone drafter (DFlash, DSpark) keeps its layers in its own
        # BF16 draft pool, outside the FP4 MLA layout. Without that pool it
        # falls back to a context arena that is dense in max_seq_len.
        if not (spec_dec_mode.is_dflash() or spec_dec_mode.is_dspark()):
            # HP/V-scale side pools are not switched by draft_kv_cache_context.
            # Set this before models and workers consult the shared predicate.
            spec_config._allow_separate_draft_kv_cache = False


def validate_encoder_decoder_kv_cache_config(model_config: ModelConfig,
                                             kv_cache_config) -> None:
    """Validate encoder-decoder KV-cache requirements for the PyTorch runtime.

    Both V1 (``KVCacheManager``, default and production target) and V2
    (``KVCacheManagerV2``, additive secondary path) are supported for
    encoder-decoder models.  Both paths require ``cross_kv_cache_fraction``
    so the cross-attention pool can be sized.
    """
    if model_config.is_encoder_decoder:
        if kv_cache_config.cross_kv_cache_fraction is None:
            raise ConfigInputError(
                "Encoder-decoder models require kv_cache_config.cross_kv_cache_fraction to be set."
            )
        if (kv_cache_config.use_kv_cache_manager_v2 is True
                and kv_cache_config.enable_block_reuse):
            # Both self- and cross-attention K/V depend on the full encoder
            # input, which is not part of V2's prefix reuse identity.
            logger.warning(
                "Disabling KV cache block reuse for encoder-decoder models "
                "with KV cache manager V2: reuse requires the full encoder "
                "input identity.")
            kv_cache_config.enable_block_reuse = False
        return

    if kv_cache_config.cross_kv_cache_fraction is not None:
        raise ConfigInputError(
            "kv_cache_config.cross_kv_cache_fraction should only be set for encoder-decoder models."
        )


def validate_encoder_decoder_tp_scope(model_config: ModelConfig) -> None:
    """Validate initially supported encoder-decoder TP combinations."""
    if not model_config.is_encoder_decoder:
        return

    mapping = model_config.mapping
    if mapping.enable_attention_dp:
        raise UnsupportedError(
            "Encoder-decoder models do not support attention DP yet. "
            "Set enable_attention_dp=False.")

    if mapping.cp_size > 1:
        raise UnsupportedError(
            "Encoder-decoder models do not support context parallelism yet. "
            "Set context_parallel_size=1.")

    if mapping.pp_size > 1:
        raise UnsupportedError(
            "Encoder-decoder models do not support pipeline parallelism yet. "
            "Set pipeline_parallel_size=1.")

    if mapping.tp_size > 1 and model_config.attn_backend != "TRTLLM":
        raise UnsupportedError(
            "Encoder-decoder tensor parallelism currently supports "
            f"attn_backend='TRTLLM' only, but got "
            f"attn_backend='{model_config.attn_backend}'.")


def validate_lm_head_output_dtype(model: torch.nn.Module,
                                  requested_dtype: str) -> None:
    """Reject an explicit output dtype that the constructed LM heads ignore."""
    if requested_dtype == "auto":
        return

    expected_dtype = str_dtype_to_torch(requested_dtype)
    found_lm_head = False
    for name, module in model.named_modules():
        # Embedding inherits LMHead but does not project hidden states to logits.
        if not isinstance(module, LMHead) or isinstance(module, Embedding):
            continue
        found_lm_head = True
        if module.output_dtype != expected_dtype:
            raise ValueError(
                f"lm_head_output_dtype={requested_dtype!r} is not supported "
                f"by {type(model).__name__}: {name or '<root>'}.output_dtype "
                f"is {module.output_dtype}, expected {expected_dtype}.")

    if not found_lm_head:
        raise ValueError(
            f"lm_head_output_dtype={requested_dtype!r} is not supported by "
            f"{type(model).__name__}: no LMHead was constructed.")


# Bounds for uniform dummy values of the 1-byte types torch cannot sample
# directly. They are not the widest possible, only small enough to avoid NaNs.
# FP4 bytes are 2 packed values: up to 64, the high 4 bits stay 0.
_DUMMY_QUANTIZED_BOUNDS = {
    torch.float8_e4m3fn: (-3.0, 3.0),
    torch.int8: (-3.0, 3.0),
    float4_e2m1x2: (0, 64),
}


def _fill_dummy_quantized(tensor: torch.Tensor,
                          generator: torch.Generator) -> None:
    """Uniform values of a type in `_DUMMY_QUANTIZED_BOUNDS`, sampled in FP16 and cast."""
    low, high = _DUMMY_QUANTIZED_BOUNDS[tensor.dtype]
    values = torch.empty(tensor.shape,
                         dtype=torch.float16,
                         device=tensor.device)
    tensor.copy_(
        values.uniform_(low, high, generator=generator).to(tensor.dtype))


def runner_rehearses(model: torch.nn.Module, llm_args: TorchLlmArgs) -> bool:
    """Whether the runner ``model`` starts with runs the KernelBuilder rehearsal's
    forwards: a decoder runner with context warmups (``resolve_runner_type``),
    not an encoder-decoder or multimodal-encoder one. Encode-only and pooling
    (not generation) models run on encoder executors, without a decoder runner.
    """
    if llm_args.encode_only or not model.model_config.is_generation:
        return False
    from .engine.runners import DecoderRunner, resolve_runner_type

    runner_cls = resolve_runner_type(model, llm_args)
    return (issubclass(runner_cls, DecoderRunner)
            and runner_cls._context_warmups_supported)


def fill_rehearsal_values(tensors: Iterable[torch.Tensor],
                          low: float = -1e-3,
                          high: float = 1e-3,
                          seed: int = 0) -> None:
    """Values a forward can run on, for tensors that have none of their own.

    Floating and quantized tensors get dummy values as `initialize_dummy_weights` gives
    them; any other tensor gets zeros, so an index stays in range.
    """
    for tensor in tensors:
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)
        if tensor.dtype in _DUMMY_QUANTIZED_BOUNDS:
            _fill_dummy_quantized(tensor, generator)
        elif torch.is_floating_point(tensor) and tensor.element_size() >= 2:
            tensor.uniform_(low, high, generator=generator)
        else:
            tensor.zero_()


def _meta_tensor_slots(
        model: torch.nn.Module) -> list[tuple[torch.nn.Module, str]]:
    """The modules of ``model`` and the names of their parameters and buffers
    that hold a meta tensor: a tensor with no value of its own."""
    return [(module, name) for module in model.modules()
            for tensors in (module._parameters, module._buffers)
            for name, tensor in tensors.items()
            if tensor is not None and tensor.is_meta]


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    This is similar to this function in SGLang with a few changes:
    https://github.com/sgl-project/sglang/blob/e074e76b31d4fff13e87a455dbc3acdaa92c537a/python/sglang/srt/model_loader/weight_utils.py#L577
    This method is used to initialize weights with dummy values for testing
    models without checkpoints. Unquantized (FP16/BF16/etc) values are generated
    from a uniform distribution over the interval (low, high).
    For some quantized types (FP8/NVFP4), torch has no built-in way to generate random values.
    We simply generate values uniformly across an interval that has been empirically verified
    to not generate NaNs/inf for these.
    """

    # Calibration scalars (input_scale / inv_input_scale / kv_scales /
    # inv_kv_scales / alpha / scalar_alpha) must keep their create_weights
    # default (typically 1.0). Randomizing them breaks FP8 attention output
    # and KV cache scaling for any `load_format="dummy"` + IPC update_weights
    # flow when the checkpoint doesn't ship calibrated values (e.g., HF
    # FineGrainedFP8, which uses dynamic activation quantization by design).
    _SKIP_NAME_SUFFIXES = (
        ".input_scale",
        ".inv_input_scale",
        ".kv_scales",
        ".inv_kv_scales",
        ".alpha",
        ".scalar_alpha",
    )

    for _name, param in model.state_dict().items():
        if any(_name.endswith(_s) for _s in _SKIP_NAME_SUFFIXES):
            continue
        generator = torch.Generator(device=param.data.device)
        generator.manual_seed(seed)
        dtype = param.data.dtype

        if param.data.element_size() < 2:
            if dtype not in _DUMMY_QUANTIZED_BOUNDS:
                raise NotImplementedError(f"Unknown quantized type: {dtype}.")
            _fill_dummy_quantized(param.data, generator)

        # Note: no need to to mess with int32 params, these are probably
        # constants and not weights.
        elif torch.is_floating_point(param):
            param.uniform_(low, high, generator=generator)

    initialize_dummy_states(model, low=low, high=high, seed=seed)


def initialize_dummy_states(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 0,
) -> None:
    """Let each module build, on dummy weights in place, what its load builds
    beyond the checkpoint's tensors (``initialize_dummy_state``)."""
    # Non-persistent buffers are intentionally absent from state_dict(). Let
    # modules that need such buffers during dummy execution initialize them
    # explicitly after the regular state has been populated.
    for module in model.modules():
        initialize_dummy_state = getattr(module, "initialize_dummy_state", None)
        if initialize_dummy_state is not None:
            initialize_dummy_state(low=low, high=high, seed=seed)


def _checkpoint_io_incompatibility_reason(
    checkpoint_loader: Optional[BaseCheckpointLoader],
    checkpoint_format: CheckpointFormat | str | None,
    load_format: LoadFormat,
    requested_policy: CheckpointIOPolicy,
    partial_model_loading: bool,
) -> Optional[str]:
    """Return the first unmet requirement for optimized checkpoint I/O.

    This function is called only when requested_policy is not CheckpointIOPolicy.NATIVE.
    """
    if checkpoint_loader is not None:
        return "an explicit checkpoint loader was provided"
    if checkpoint_format != CheckpointFormat.HF:
        return (
            f"checkpoint_io_policy={requested_policy.value!r} requires checkpoint_format='HF' "
            f"(received '{checkpoint_format}')")

    if load_format != LoadFormat.AUTO:
        return (
            f"checkpoint_io_policy={requested_policy.value!r} requires load_format='auto' "
            f"(received {load_format.name!r})")
    if partial_model_loading:
        return "partial model loading was requested"

    from tensorrt_llm._torch.models.modeling_utils import (
        CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING, get_checkpoint_weight_loader)

    if (CHECKPOINT_LOADER_FORMAT_DEFAULT_MAPPING.get(CheckpointFormat.HF)
            is not HfCheckpointLoader):
        return "the registered HF checkpoint loader is not the built-in loader"
    if get_checkpoint_weight_loader(CheckpointFormat.HF) is not HfWeightLoader:
        return "the registered HF weight loader is not the built-in loader"
    return None


def _resolve_checkpoint_io_policy(
    checkpoint_loader: Optional[BaseCheckpointLoader],
    checkpoint_format: CheckpointFormat | str | None,
    load_format: LoadFormat,
    requested_policy: CheckpointIOPolicy,
    partial_model_loading: bool,
) -> tuple[CheckpointIOPolicy, Optional[str]]:
    """Return the policy to use and the reason for a fallback, if any."""

    if requested_policy == CheckpointIOPolicy.NATIVE:
        return CheckpointIOPolicy.NATIVE, None

    # requested_policy can be AUTO or RANK_STRIPED_READ_AHEAD:
    reason = _checkpoint_io_incompatibility_reason(checkpoint_loader,
                                                   checkpoint_format,
                                                   load_format,
                                                   requested_policy,
                                                   partial_model_loading)

    if reason is None:
        # Compatible: AUTO and RANK_STRIPED_READ_AHEAD both use read-ahead.
        return CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD, None

    # Not compatible: fall back to native I/O.
    message = ("Checkpoint I/O policy resolved before loading: "
               f"requested={requested_policy}, "
               f"selected={CheckpointIOPolicy.NATIVE}, "
               f"reason={reason}.")
    if requested_policy == CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD:
        logger.warning(message)
    else:
        logger.info(message)
    return CheckpointIOPolicy.NATIVE, reason


def construct_checkpoint_loader(
    checkpoint_loader: Optional[BaseCheckpointLoader],
    checkpoint_format: CheckpointFormat | str | None,
    *,
    mx_config: Optional[ModelExpressConfig] = None,
    checkpoint_io_policy: CheckpointIOPolicy = CheckpointIOPolicy.NATIVE,
    load_format: LoadFormat = LoadFormat.AUTO,
    partial_model_loading: bool = False,
) -> BaseCheckpointLoader:
    # Just in case someone wrongly passes a str instead of a CheckpointIOPolicy.
    checkpoint_io_policy = CheckpointIOPolicy(checkpoint_io_policy)
    requested_checkpoint_io_policy = checkpoint_io_policy
    checkpoint_io_policy, selection_fallback_reason = \
        _resolve_checkpoint_io_policy(
            checkpoint_loader,
            checkpoint_format,
            load_format,
            requested_checkpoint_io_policy,
            partial_model_loading,
        )
    from tensorrt_llm._torch.models.modeling_utils import (
        get_checkpoint_weight_loader, get_config_loader)

    if checkpoint_loader is not None:
        checkpoint_weight_loader = checkpoint_loader.weight_loader
        if type(checkpoint_weight_loader) is HfWeightLoader:
            checkpoint_weight_loader._configure_checkpoint_io_policy(
                checkpoint_io_policy,
                requested_checkpoint_io_policy=requested_checkpoint_io_policy,
                selection_fallback_reason=selection_fallback_reason,
            )
        return checkpoint_loader

    checkpoint_weight_loader_cls = get_checkpoint_weight_loader(
        checkpoint_format)
    if checkpoint_weight_loader_cls is HfWeightLoader:
        checkpoint_weight_loader = checkpoint_weight_loader_cls(
            checkpoint_io_policy=checkpoint_io_policy,
            requested_checkpoint_io_policy=requested_checkpoint_io_policy,
            selection_fallback_reason=selection_fallback_reason,
            partial_model_loading=partial_model_loading,
        )
    else:
        checkpoint_weight_loader = checkpoint_weight_loader_cls()
    config_loader = get_config_loader(checkpoint_format)()

    # Pass extra kwargs for format-specific loaders (e.g. MX).
    extra_kwargs: dict = {}
    if checkpoint_format == CheckpointFormat.MX:
        if mx_config is not None:
            extra_kwargs["mx_server_url"] = mx_config.server_url

    checkpoint_loader = BaseCheckpointLoader.get(
        checkpoint_format=checkpoint_format,
        weight_loader=checkpoint_weight_loader,
        weight_mapper=None,
        config_loader=config_loader,
        **extra_kwargs)

    return checkpoint_loader


def _weight_load_num_threads() -> Optional[int]:
    """Threads ``TLLM_WEIGHT_LOAD_THREADS`` asks for, capped at usable CPUs."""
    value = os.environ.get(_WEIGHT_LOAD_THREADS_ENV)
    if value is None or not value.strip():
        return None
    try:
        requested = int(value)
    except ValueError as error:
        raise ConfigInputError(f"{_WEIGHT_LOAD_THREADS_ENV} must be a positive "
                               f"integer, got {value!r}") from error
    if requested <= 0:
        raise ConfigInputError(f"{_WEIGHT_LOAD_THREADS_ENV} must be a positive "
                               f"integer, got {value!r}")
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return min(requested, available)


@contextmanager
def _weight_load_threads(requested: Optional[int]) -> Iterator[None]:
    """Run this thread's load on ``requested`` torch threads.

    None keeps the process's count. Only raises; restores on exit and on error.
    """
    previous = torch.get_num_threads()
    if requested is None or requested <= previous:
        yield
        return
    # The native backend sets one count per process, once; a raise would stay.
    if "ATen parallel backend: OpenMP" not in torch.__config__.parallel_info():
        logger.warning(f"Ignoring {_WEIGHT_LOAD_THREADS_ENV}: only torch's "
                       "OpenMP backend can restore the thread count")
        yield
        return
    logger.info(f"Weight loading uses {requested} torch threads (was "
                f"{previous}), as {_WEIGHT_LOAD_THREADS_ENV} asks")
    try:
        # The first call also sizes torch's pthreadpool for good; keep its size.
        torch.set_num_threads(previous)
        torch.set_num_threads(requested)
        # New threads start from torch's process-wide count; set it back from
        # another thread so this one keeps the raise and loader pools do not.
        setter = threading.Thread(target=torch.set_num_threads,
                                  args=(previous, ))
        setter.start()
        setter.join()
        yield
    finally:
        torch.set_num_threads(previous)


def _inspect_shadow_weight_load_plan(
        checkpoint_loader: Any, checkpoint_dir: str, weight_mapper: Any,
        **kwargs: Any) -> tuple[CheckpointCatalog, WeightLoadPlan] | None:
    """Build and validate advisory checkpoint metadata without changing loading."""
    enabled = os.environ.get(_SHADOW_WEIGHT_LOAD_PLAN_ENV,
                             "0").lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None

    build_catalog = getattr(checkpoint_loader, "build_checkpoint_catalog", None)
    build_plan = getattr(weight_mapper, "build_weight_load_plan", None)
    if build_catalog is None or build_plan is None:
        return None

    try:
        catalog = build_catalog(checkpoint_dir, **kwargs)
        if catalog is None:
            return None
        plan = build_plan(catalog)
        plan.validate_against(catalog)
    except Exception as error:
        # Shadow inspection is strictly advisory. In particular, it must not
        # turn a successful native load into a startup failure.
        logger.warning("Checkpoint shadow plan inspection failed; continuing "
                       "with the unchanged loader: "
                       f"{type(error).__name__}: {error}")
        return None

    logger.info(
        "Checkpoint shadow plan validated: "
        f"catalog_id={catalog.catalog_id}, plan_id={plan.plan_id}, "
        f"coverage={plan.coverage.value}, ordering={plan.ordering.value}, "
        f"objects={len(catalog.objects)}, "
        f"tensors={len(catalog.tensors)}, demands={len(plan.demands)}.")
    return catalog, plan


def _load_weights_strips_compile_wrapper(model: torch.nn.Module) -> bool:
    """Whether ``model.load_weights`` tolerates a torch.compile wrapper in the tree.

    The generic loaders (``modeling_utils._load_weights_impl`` / ``_v2``) strip
    the wrapper's ``_orig_mod`` path component before matching checkpoint
    keys, so a model that inherits ``DecoderModelForCausalLM.load_weights``
    is safe. A model with its own ``load_weights`` walk is not known to be,
    unless its class sets ``load_weights_strips_compile_wrapper = True``
    (for overrides that only pre-process and delegate to the generic path).
    """
    cls = type(model)
    if getattr(cls, "load_weights_strips_compile_wrapper", False):
        return True
    from tensorrt_llm._torch.models.modeling_utils import \
        DecoderModelForCausalLM
    return getattr(cls, "load_weights",
                   None) is DecoderModelForCausalLM.load_weights


def _assert_no_compile_wrapper_in_paths(model: torch.nn.Module) -> None:
    """Fail loudly if a torch.compile wrapper would make this reload a no-op.

    Loaders that match checkpoint tensors to modules by dotted path over
    ``named_modules()`` see ``._orig_mod.`` in every path under a compiled
    scope and match nothing; because RL refit passes
    ``allow_partial_loading=True`` the compiled subtree is then skipped in
    silence and keeps its pre-refit weights. The generic loaders strip the
    component themselves (``strip_torch_compile_wrapper``); this guard covers
    models with their own ``load_weights`` walk (see
    ``_load_weights_strips_compile_wrapper``). Callers unwrap first via
    ``model_engine.unwrap_compiled_model_for_refit()``.
    """
    # Escape hatch, for reproducing the pre-guard behaviour on purpose.
    if os.environ.get("TLLM_REFIT_SKIP_WRAPPER_CHECK") == "1":
        return
    for name, _ in model.named_parameters():
        if "._orig_mod." in name or name.startswith("_orig_mod."):
            raise RuntimeError(
                "Refit would silently load nothing: a torch.compile wrapper is still "
                f"installed, so parameter paths carry '_orig_mod' (e.g. {name!r}) while "
                "checkpoint keys do not, and this model's load_weights does not strip "
                "it. Unwrap before loading weights (e.g. "
                "model_engine.unwrap_compiled_model_for_refit()), or set "
                "load_weights_strips_compile_wrapper = True on a model class whose "
                "load_weights delegates to the generic loaders.")


def _check_compile_wrapper_before_reload(model: torch.nn.Module) -> None:
    if not _load_weights_strips_compile_wrapper(model):
        _assert_no_compile_wrapper_in_paths(model)


def _reject_dense_dwdp_update(model: torch.nn.Module, action: str) -> None:
    """Refuse to rewrite the weights of a model that dense DWDP sharded.

    Dense DWDP repoints the dense weights of the MoE decoder layers into staging
    slots that several layers share and that the next gather overwrites, so
    weights written through the modules would be lost. Its setup marks the
    model and its decoder with ``_dense_dwdp_active``; the mark stays after the
    dense DWDP teardown, which leaves zero-size tensors behind.

    Args:
        model: The model about to be updated.
        action: What the caller was about to do, for the error message.

    Raises:
        RuntimeError: The model or its decoder (``model.model``) is marked.
    """
    decoder = getattr(model, "model", None)
    if getattr(model, "_dense_dwdp_active", False) or getattr(
            decoder, "_dense_dwdp_active", False):
        raise RuntimeError(
            f"Cannot {action}: dense DWDP (dwdp_config.enable_dense) moved "
            "this model's dense weights into staging slots shared across "
            "layers, so they cannot be rewritten in place. Launch without "
            "dwdp_config.enable_dense to update weights.")


def _holds_its_bytes(tensor: torch.Tensor) -> bool:
    """Whether ``tensor``'s storage still holds its elements: post-load processing
    can release a parameter's memory in place (``untyped_storage().resize_(0)``)."""
    return (tensor.untyped_storage().nbytes()
            >= (tensor.storage_offset() + tensor.numel()) *
            tensor.element_size())


class _ModuleTree:
    """Each module's submodules, attributes and parameters, to put the module tree
    back as it was.

    Putting back a submodule slot is not enough: assigning a module to an attribute
    (e.g. a draft's `embed_tokens = None` tied to the target's) moves the name from
    the attributes to the submodules.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._modules = [(module, dict(module._modules), {
            name: value
            for name, value in vars(module).items()
            if name not in ("_parameters", "_buffers", "_modules")
        }) for module in model.modules()]
        # Each parameter slot and its layout; the slots of one parameter share a key.
        self._parameters = [(module, name, id(param), param.shape, param.dtype,
                             param.device, param.requires_grad)
                            for module in model.modules()
                            for name, param in module._parameters.items()
                            if param is not None]
        # Each parameter's tensor as created, which keeps its memory allocated
        # while post-load processing replaces the parameter (restore_parameters).
        self._tensors = {
            id(param): param.detach()
            for module in model.modules()
            for param in module._parameters.values() if param is not None
        }

    def restore(self) -> None:
        """Put back the submodules, and the attributes as they were.

        An attribute that was None is None again: what a forward or post-load
        processing set it to since was built on dummy weights or a KV cache of
        the rehearsal, or refers to a module this drops. The checkpoint load or
        the first forward sets it again.
        """
        for module, children, attributes in self._modules:
            module._modules.clear()
            module._modules.update(children)
            for name, value in attributes.items():
                if value is None:
                    module.__dict__[name] = None
                else:
                    module.__dict__.setdefault(name, value)

    def restore_parameters(self) -> None:
        """Give each parameter slot its layout back, for the checkpoint to load into.

        A parameter that post-load processing released or replaced with another
        layout gets its tensor as created back: the checkpoint loads into the
        memory it was created in, as a load without the rehearsal does, so that
        post-load processing frees the same memory and leaves the allocator's
        cache as fragmented as it does then. One whose memory was released in
        place is created again, empty. Slots that shared a parameter share one
        again.
        """
        kept: dict[int, torch.nn.Parameter] = {}
        for module, name, key, shape, dtype, device, _ in self._parameters:
            param = module._parameters.get(name)
            if (param is not None and param.shape == shape
                    and param.dtype == dtype and param.device == device
                    and _holds_its_bytes(param)):
                kept.setdefault(key, param)
        for module, name, key, shape, dtype, device, requires_grad in self._parameters:
            if key not in kept:
                tensor = self._tensors.get(key)
                if tensor is None or not _holds_its_bytes(tensor):
                    tensor = torch.empty(shape, dtype=dtype, device=device)
                kept[key] = torch.nn.Parameter(tensor,
                                               requires_grad=requires_grad)
            module.__dict__.pop(name, None)
            module._parameters[name] = kept[key]
        # The parameters hold the tensors they need now.
        self._tensors.clear()


@dataclass
class _DeferredCheckpoint:
    """A checkpoint load `ModelLoader.load` left for `load_deferred_checkpoint`:
    the inputs of its `LoadFormat.AUTO` read and the module tree as created."""
    model_config: ModelConfig
    checkpoint_dir: str
    checkpoint_loader: BaseCheckpointLoader
    loads_draft_weights: bool
    source_identity: Optional[SourceIdentity]
    post_transform_qualification: PostTransformQualificationDecision
    tree: _ModuleTree


_CheckpointStartupMetadata = dict[str,
                                  str | dict[str, str | bool | None] | None]


def _checkpoint_startup_metadata(
    checkpoint_loader: BaseCheckpointLoader,
    load_format: LoadFormat,
) -> _CheckpointStartupMetadata:
    """Describe the concrete checkpoint source selected for this load."""
    weight_loader = getattr(checkpoint_loader, "weight_loader", None)
    source_kind = getattr(checkpoint_loader, "checkpoint_format", None)
    return {
        "checkpoint_loader_kind":
        type(checkpoint_loader).__name__,
        "checkpoint_weight_loader_kind":
        (type(weight_loader).__name__
         if weight_loader is not None else "unknown"),
        "checkpoint_source_kind":
        str(source_kind or "unknown"),
        "load_format":
        load_format.name.lower(),
        # Explicitly unknown until a supported weight load has finished.
        "checkpoint_io_policy":
        None,
    }


def _check_post_init_model_compat(
        model: DecoderModelForCausalLM, load_format: LoadFormat,
        checkpoint_loader: BaseCheckpointLoader) -> None:
    """Raise error if the initialized LLM model cannot load the checkpoint.
    """
    requires_regular_checkpoint_loading = any(
        getattr(module, "_requires_regular_checkpoint_loading", False)
        for module in model.modules())
    # Pinned-host parameters must be materialized and filled in place. The
    # regular AUTO load copies the read weights into the existing parameters,
    # which keeps their stable addresses. AUTO may still resolve to MX, so
    # check the resolved format too.
    if requires_regular_checkpoint_loading and (
            load_format != LoadFormat.AUTO
            or checkpoint_loader.checkpoint_format == CheckpointFormat.MX):
        raise UnsupportedError(
            "Host-resident model weights currently require the regular "
            "checkpoint loading of load_format=AUTO; DUMMY, VISION_ONLY, GMS, "
            "and MX loaders cannot materialize them")


def _load_checkpoint_into_model(
    checkpoint_loader: BaseCheckpointLoader,
    checkpoint_dir: str,
    *,
    model: DecoderModelForCausalLM,
    is_draft_model: bool,
    mapping: Mapping,
    weight_mapper: Optional[BaseWeightMapper],
    extra_load_weights_kwargs: Optional[dict[str, Any]],
    metrics: dict[str, float],
) -> dict[str, str | bool | None] | None:
    """Load weights from one main or draft checkpoint, under auto load format.

    If enabled and compatible, use rank-striped read-ahead IO policy for a
    faster load. Otherwise, use the native loading solution by calling
    `checkpoint_loader.load_weights()`.

    Args:
        checkpoint_loader: checkpoint loader of this load
        checkpoint_dir: checkpoint to read, e.g. the target checkpoint
            directory or ``spec_config.speculative_model``
        model: model that holds the target and, if any, the draft model
        is_draft_model: whether this is the separate draft checkpoint
        mapping: parallel mapping of this rank
        weight_mapper: initialized weight mapper for this checkpoint
        extra_load_weights_kwargs: extra kwargs for the read, e.g.
            ``source_identity`` and the MX kwargs for the target, or the
            draft's ``key_prefixes``, outside which no key is copied; or None
        metrics: model loader timing metrics, updated in place

    Returns:
        The checkpoint I/O policy that read this checkpoint, or None when the
        built-in HF weight loader did not read it from storage.
    """
    load_weights_kwargs: dict[str, Any] = {"mapping": mapping}
    if is_draft_model:
        preparation_metric = ModelLoaderMetricNames.DRAFT_CHECKPOINT_PREPARATION_SECONDS
        population_metric = ModelLoaderMetricNames.DRAFT_WEIGHT_POPULATION_SECONDS
        finalization_metric = ModelLoaderMetricNames.DRAFT_CHECKPOINT_FINALIZATION_SECONDS
        model_load_weights = model.load_draft_weights
    else:
        preparation_metric = ModelLoaderMetricNames.CHECKPOINT_PREPARATION_SECONDS
        population_metric = ModelLoaderMetricNames.WEIGHT_POPULATION_SECONDS
        finalization_metric = ModelLoaderMetricNames.CHECKPOINT_FINALIZATION_SECONDS
        model_load_weights = model.load_weights
        # Only the target read gets the model, so a context-only worker reads
        # the target checkpoint lazily.
        load_weights_kwargs["model"] = model
    load_weights_kwargs.update(extra_load_weights_kwargs or {})
    key_prefixes = load_weights_kwargs.get("key_prefixes")

    # If possible, use rank-striped read-ahead IO policy to faster reading:
    # - Must be HfCheckpointLoader, not its derived class
    # - weight_loader must be HfWeightLoader, not its derived class
    # - checkpoint_io_policy is RANK_STRIPED_READ_AHEAD
    # - no context only lazy loading
    if (type(checkpoint_loader) is HfCheckpointLoader
            and type(checkpoint_loader.weight_loader) is HfWeightLoader
            and checkpoint_loader.weight_loader.checkpoint_io_policy
            == CheckpointIOPolicy.RANK_STRIPED_READ_AHEAD
            and not checkpoint_loader.uses_context_only_lazy_loading(
                load_weights_kwargs.get("model"))):
        # read ahead, and copy the weights inside the weight loader's session
        weight_loader = checkpoint_loader.weight_loader
        with weight_loader.open_weight_session(
                checkpoint_dir, **load_weights_kwargs) as weights:
            _copy_weights_into_model(
                model_load_weights,
                model_hooks.select_draft_weights(weights, key_prefixes),
                weight_mapper, population_metric.value, metrics)
        # The weight loader's session records its own timings.
        io_status = weight_loader.last_checkpoint_io_status
        metrics[preparation_metric.value] = io_status.preparation_seconds
        metrics[finalization_metric.value] = io_status.finalization_seconds
    else:
        # Call ``checkpoint_loader.load_weights()`` to open weight tensors as
        # a map (str -> tensor) first before calling LLM model class'
        # ``load_weights()`` function to load them into GPU.

        # Situations that go into this branch:
        # - Hf checkpoint loader with native IO policy or context-only lazy loading
        # - MX or Mistral checkpoint loader
        # - 3rd-party custom checkpoint loader or weight loader
        with timing_metric(preparation_metric.value, metrics):
            weights = checkpoint_loader.load_weights(checkpoint_dir,
                                                     **load_weights_kwargs)
        # When using MX, it may have already loaded the weights in
        # ``load_weights()``. So it can skip the weight copy below.
        # MX indicate weights to be preloaded successfully by returning
        # `weights` as empty. It does not yet support draft model.
        if (weights or is_draft_model
                or not checkpoint_loader.is_weights_preloaded()):
            _copy_weights_into_model(
                model_load_weights,
                model_hooks.select_draft_weights(weights, key_prefixes),
                weight_mapper, population_metric.value, metrics)
        metrics.setdefault(finalization_metric.value, 0.0)

    _inspect_shadow_weight_load_plan(checkpoint_loader, checkpoint_dir,
                                     weight_mapper, **load_weights_kwargs)
    # Take the policy after the weight session finished, which can turn
    # read-ahead into native I/O, and before a draft read with the same loader
    # resets it.
    return checkpoint_io_policy_metadata(checkpoint_loader)


def _copy_weights_into_model(model_load_weights: Callable[..., None],
                             weights: dict[str, Any],
                             weight_mapper: Optional[BaseWeightMapper],
                             population_metric_name: str,
                             metrics: dict[str, float]) -> None:
    """Copy read checkpoint weights into the model and time the copy."""
    with timing_metric(population_metric_name, metrics):
        model_hooks.call_load_weights(model_load_weights, weights,
                                      weight_mapper)
        if torch.cuda.is_available():  # CPU guard
            torch.cuda.synchronize()


def load_model_from_auto_load_format(
    checkpoint_loader: BaseCheckpointLoader,
    checkpoint_dir: str,
    *,
    model_config: ModelConfig,
    model: DecoderModelForCausalLM,
    mapping: Mapping,
    llm_args: TorchLlmArgs,
    spec_config: Optional[DecodingBaseConfig],
    source_identity: Optional[SourceIdentity],
    qualification: PostTransformQualificationDecision,
    metrics: dict[str, float],
    startup_metadata: _CheckpointStartupMetadata,
) -> tuple[BaseWeightMapper, bool]:
    """Load the target and any separate draft checkpoint for ``LoadFormat.AUTO``.

    The model is already built and has CUDA memory; post-load hooks run
    afterwards in `ModelLoader.load()`.

    Built-in and custom checkpoint loaders, including custom weight loaders,
    read the checkpoint here.

    Args:
        checkpoint_loader: checkpoint loader of this load
        checkpoint_dir: checkpoint directory of the target model
        model_config: config the model was built from
        model: model to fill
        mapping: parallel mapping of this rank
        llm_args: LLM args of this load
        spec_config: speculative decoding config, or None
        source_identity: identity of this rank's model for MX, or None
        qualification: MX post-transform profile decision for this load
        metrics: model loader timing metrics, updated in place
        startup_metadata: model loader startup metadata; the checkpoint I/O
            policy of the target and draft reads is recorded in place

    Returns:
        The target weight mapper, and whether the checkpoint loader wrote
        the target weights into the model directly (MX P2P).
    """
    # load target weights
    weights_checkpoint_dir = (model.llm_checkpoint_dir if hasattr(
        model, 'llm_checkpoint_dir') else checkpoint_dir)
    # Generic loaders ignore source_identity; MXCheckpointLoader pops it.
    target_load_weights_kwargs: dict[str, Any] = {
        "source_identity": source_identity
    }
    if checkpoint_loader.checkpoint_format == CheckpointFormat.MX:
        target_load_weights_kwargs.update(
            mx_loading.get_mx_load_weights_kwargs(model_config=model_config,
                                                  llm_args=llm_args,
                                                  qualification=qualification))
    weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
        model, model_config)
    startup_metadata["checkpoint_io_policy"] = _load_checkpoint_into_model(
        checkpoint_loader,
        weights_checkpoint_dir,
        model=model,
        is_draft_model=False,
        mapping=mapping,
        weight_mapper=weight_mapper,
        extra_load_weights_kwargs=target_load_weights_kwargs,
        metrics=metrics)
    # Read before the draft load: an MX draft read without MX state resets
    # this flag.
    weights_preloaded = checkpoint_loader.is_weights_preloaded()

    # load separate draft weights
    if spec_config is not None and spec_config.needs_separate_draft_weights:
        draft_weight_mapper = checkpoint_loader.get_initialized_draft_weight_mapper(
            model, weight_mapper)
        draft_load_kwargs = model_hooks.draft_checkpoint_load_kwargs(model)
        startup_metadata[
            "draft_checkpoint_io_policy"] = _load_checkpoint_into_model(
                checkpoint_loader,
                spec_config.speculative_model,
                model=model,
                is_draft_model=True,
                mapping=mapping,
                weight_mapper=draft_weight_mapper,
                extra_load_weights_kwargs=draft_load_kwargs,
                metrics=metrics)
    return weight_mapper, weights_preloaded


def captured_cuda_graph_batch_sizes(
        llm_args: TorchLlmArgs, max_num_tokens: int,
        spec_config: Optional[DecodingBaseConfig]) -> Tuple[int, ...]:
    """The decode CUDA-graph batch sizes the engine captures
    (``engine.cuda_graph.plan_cuda_graphs``): the configured ones within
    max_batch_size and max_num_tokens // tokens per request; () without
    CUDA graphs. Sizes above them never run, so buffers sized by the largest
    graph batch (``ModelConfig.cuda_graph_batch_sizes``) stop there."""
    config = llm_args.cuda_graph_config
    if config is None or not config.batch_sizes:
        return ()
    from .engine.cuda_graph import filter_cuda_graph_batch_sizes
    max_batch_size = (llm_args.max_batch_size if llm_args.max_batch_size
                      is not None else max(config.batch_sizes))
    tokens_per_request = (spec_config.tokens_per_gen_step
                          if spec_config is not None else 1)
    return tuple(
        filter_cuda_graph_batch_sizes(list(config.batch_sizes), max_batch_size,
                                      max_num_tokens, tokens_per_request,
                                      config.enable_padding))


_RELEASE_CHECKPOINT_PAGE_CACHE_ENV = "TRTLLM_RELEASE_CHECKPOINT_PAGE_CACHE"
_CHECKPOINT_PAGE_CACHE_RELEASE_SECONDS = "checkpoint_page_cache_release_seconds"
# Smaller files can free too little memory to set off compaction worth
# waiting for.
_MIN_ADVISED_BYTES_TO_SETTLE = 1 << 30


def _drop_page_cache_of_dirs(checkpoint_dirs: list[str]) -> None:
    """Drop the page cache of the top-level files in ``checkpoint_dirs``.

    Then waits for the page migration that follows to settle, unless the files
    total less than 1 GiB. The wait depends on the files' size, not on how
    much of them was cached: the drop in the host's page cache, which the log
    reports, also counts what other processes read meanwhile.
    """
    try:
        files = []
        for checkpoint_dir in checkpoint_dirs:
            if not os.path.isdir(checkpoint_dir):
                continue
            with os.scandir(checkpoint_dir) as entries:
                files += [entry.path for entry in entries if entry.is_file()]
    except OSError as error:
        logger.warning(f"Skipping the checkpoint page-cache release: {error}")
        return
    start = time.monotonic()
    cached_before = page_cache_bytes()
    advised_bytes = release_file_page_cache(files)
    cached_after = page_cache_bytes()
    release_seconds = time.monotonic() - start
    if advised_bytes >= _MIN_ADVISED_BYTES_TO_SETTLE:
        settle = (f"page migration settled after "
                  f"{wait_for_page_migration_to_settle():.1f} s")
    else:
        settle = "no wait for page migration"
    shrank = ("by an unknown amount"
              if cached_before is None or cached_after is None else
              f"by {(cached_before - cached_after) / 2**30:.1f} GiB")
    logger.info(
        f"Advised DONTNEED on {len(files)} checkpoint files "
        f"({advised_bytes / 2**30:.1f} GiB) in {release_seconds:.1f} s; "
        f"the host's page cache shrank {shrank}; {settle}")


def _release_checkpoint_page_cache(
        checkpoint_dirs: list[str]) -> Optional[float]:
    """Drop the page cache of the checkpoint files this server read on this node.

    Opt-in with ``TRTLLM_RELEASE_CHECKPOINT_PAGE_CACHE=1``. The weights are on the
    GPU at this point. Left cached, the files are dropped later, at a time the
    process does not choose, and the kernel's compaction of the freed memory then
    stalls the server's host threads while it serves. Dropping them here, and
    waiting for the page migration that follows to settle, moves both into
    startup. A later restart reads the checkpoint from storage again.

    Every rank of this server calls it after loading, with the directories it
    read files from (top-level files only). Once all of them have finished,
    rank 0 of each node drops the files of every directory any rank on its node
    read; the others wait for it. Other servers on the node that read the same
    files are not coordinated with.

    Returns the seconds this rank spent here, or None if the release is off.
    """
    if os.environ.get(_RELEASE_CHECKPOINT_PAGE_CACHE_ENV, "0") != "1":
        return None
    if ENABLE_MULTI_DEVICE and mpi_disabled():
        logger.warning_once(
            f"{_RELEASE_CHECKPOINT_PAGE_CACHE_ENV}=1 is ignored without MPI, which "
            "tells when every rank on the node has finished reading.",
            key=_RELEASE_CHECKPOINT_PAGE_CACHE_ENV)
        return None
    start = time.monotonic()
    checkpoint_dirs = sorted({d for d in checkpoint_dirs if d})
    node_comm = None
    is_dropper = True
    if ENABLE_MULTI_DEVICE:
        # Returns once every rank on the node has finished reading.
        node_comm = local_mpi_comm()
        checkpoint_dirs = sorted(
            set().union(*node_comm.allgather(checkpoint_dirs)))
        is_dropper = node_comm.Get_rank() == 0
    try:
        if is_dropper and checkpoint_dirs:
            _drop_page_cache_of_dirs(checkpoint_dirs)
    finally:
        if node_comm is not None:
            node_comm.Barrier()
    return time.monotonic() - start


class ModelLoader:
    """
    Handles the loading, configuration, and weight initialization of a PyTorch model.
    This class isolates model loading logic from the main execution engine.
    """

    # Loaders built without __init__ keep the process's torch thread count.
    _weight_load_num_threads: Optional[int] = None

    def __init__(self,
                 llm_args: TorchLlmArgs,
                 mapping: Mapping,
                 spec_config: Optional[DecodingBaseConfig],
                 sparse_attention_config: Optional["SparseAttentionConfig"],
                 max_num_tokens: int,
                 max_seq_len: Optional[int],
                 lora_config: Optional[LoraConfig] = None,
                 model_weights_memory_tag: Optional[ExecutorMemoryType] = None,
                 model_weights_restore_mode: Optional[RestoreMode] = None):
        """
        Initializes the ModelLoader.

        Args:
            llm_args: Configuration for the PyTorch backend.
            mapping: The distributed mapping configuration.
            spec_config: Configuration for speculative decoding.
            max_num_tokens: The maximum number of tokens the engine will handle.
            max_seq_len: The maximum sequence length.
            lora_config: Configuration for LoRA.
            model_weights_memory_tag: When set, parameter allocations during
                `load()` are placed under a separate virtual-memory tag so
                they can be released/materialized independently of buffers.
            model_weights_restore_mode: RestoreMode for the model weights
                virtual-memory scope.
        """
        self.llm_args = llm_args
        self.mapping = mapping
        self.spec_config = spec_config
        self.sparse_attention_config = sparse_attention_config
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.lora_config = lora_config
        self.model_weights_memory_tag = model_weights_memory_tag
        self.model_weights_restore_mode = model_weights_restore_mode
        self._weight_load_num_threads = _weight_load_num_threads()
        self.weight_mapper = None
        self._weight_pool_proxy = None
        self._gms_backend = None
        self._checkpoint_loader: Optional[BaseCheckpointLoader] = None
        self._deferred: Optional[_DeferredCheckpoint] = None
        # Mostly weight loading and processing time metrics, updated when load() is called.
        self._metrics: dict[str, float] = {}
        self._startup_metadata: _CheckpointStartupMetadata = {}

    @property
    def metrics(self) -> dict[str, float]:
        """Return weight loading and processing time metrics."""
        return self._metrics

    @property
    def startup_metadata(self) -> _CheckpointStartupMetadata:
        """Return checkpoint-source metadata captured by the latest load."""
        return self._startup_metadata

    @staticmethod
    def load_config_and_apply_defaults(
            checkpoint_dir: str, llm_args: TorchLlmArgs,
            checkpoint_loader: BaseCheckpointLoader) -> TorchLlmArgs:
        """Load model config and apply model-specific defaults to llm_args."""
        return ModelLoader.load_config_with_defaults(checkpoint_dir, llm_args,
                                                     checkpoint_loader)[0]

    @staticmethod
    def load_config_with_defaults(
        checkpoint_dir: str, llm_args: TorchLlmArgs,
        checkpoint_loader: BaseCheckpointLoader
    ) -> tuple[TorchLlmArgs, Optional[ModelConfig]]:
        """Load the model config and apply its model-specific defaults to llm_args.

        The config has ``model_kwargs`` applied, as model loading applies them.

        Args:
            checkpoint_dir: The checkpoint to read the config from.
            llm_args: The arguments the defaults are applied to, in place.
            checkpoint_loader: The loader that reads the config.

        Returns:
            ``llm_args``, and the loaded config, which is None without a checkpoint loader.
        """
        if checkpoint_loader is None:
            # No config to resolve a model class from; still resolve the
            # "auto" sentinel so it never leaks past config loading.
            _resolve_transceiver_runtime_auto(llm_args)
            _resolve_kv_cache_manager_v2_auto(llm_args)
            return llm_args, None

        config_kwargs = {
            'trust_remote_code': llm_args.trust_remote_code,
            'mm_encoder_only': llm_args.mm_encoder_only,
        }
        if llm_args.model_kwargs is not None:
            config_kwargs['model_kwargs'] = llm_args.model_kwargs
        if llm_args.parallel_config:
            config_kwargs['mapping'] = llm_args.parallel_config.to_mapping()

        if llm_args.speculative_config:
            from tensorrt_llm._torch.speculative.utils import \
                resolve_mtp_checkpoint_source

            resolve_mtp_checkpoint_source(llm_args.speculative_config,
                                          checkpoint_dir)
            config_kwargs['spec_config'] = llm_args.speculative_config

        config = checkpoint_loader.load_config(checkpoint_dir, **config_kwargs)

        model_cls = AutoModelForCausalLM._resolve_class(config)
        if model_cls is not None:
            _reject_unsupported_model_llm_args(llm_args, model_cls)
        original_kv_cache_manager_setting = (
            llm_args.kv_cache_config.use_kv_cache_manager_v2)

        # Preferences follow the checkpoint's original architecture:
        # _resolve_class may rewrite it to an execution class (e.g. the EAGLE3
        # prefixing above), which must not drop the target model's
        # preferences.
        preference_cls = model_cls
        architectures = getattr(config.pretrained_config, 'architectures', None)
        if architectures:
            preference_cls = get_registered_model_class(
                architectures[0]) or model_cls

        _validate_encoder_attention_backend_support(preference_cls,
                                                    llm_args.multimodal_config)

        # model_cls is None when the architecture is unknown/unsupported.
        model_defaults = {}
        if model_cls and hasattr(model_cls, 'get_model_defaults'):
            model_defaults = model_cls.get_model_defaults(llm_args) or {}
            if model_defaults:
                applied_defaults = apply_model_defaults_to_llm_args(
                    llm_args, model_defaults)
                if applied_defaults:
                    logger.info(
                        f"Applied model defaults for {model_cls.__name__}: {applied_defaults}"
                    )

        if llm_args.speculative_config is not None:
            from tensorrt_llm._torch.speculative import \
                update_spec_config_from_model_config
            from tensorrt_llm._torch.speculative.utils import \
                resolve_mtp_checkpoint_source

            # Model defaults reconstruct nested Pydantic configs and drop private / runtime fields,
            # so resolve the checkpoint source again before restoring derived MTP state.
            resolve_mtp_checkpoint_source(llm_args.speculative_config,
                                          checkpoint_dir)
            update_spec_config_from_model_config(llm_args.speculative_config,
                                                 config.pretrained_config,
                                                 preference_cls)

        # Resolve "auto" sentinel values after model defaults are applied.
        _resolve_transceiver_runtime_auto(llm_args, preference_cls,
                                          config.pretrained_config)
        if (original_kv_cache_manager_setting == "auto" and
            (llm_args.kv_cache_config.dtype == "nvfp4" or
             (llm_args.kv_cache_config.dtype == "auto"
              and config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4))):
            # Resolve FP4 MLA before the generic model preference turns auto
            # into False. Keep an explicit False distinguishable and reject it
            # during FP4 validation instead of silently overriding the user.
            fp4_mla_config = replace(
                config,
                attn_backend=llm_args.attn_backend,
                sparse_attention_config=llm_args.sparse_attention_config,
                quant_config=copy.deepcopy(config.quant_config),
                quant_config_dict=copy.deepcopy(config.quant_config_dict),
            )
            if supports_fp4_mla_attention(fp4_mla_config):
                validate_and_set_kv_cache_quant(fp4_mla_config,
                                                llm_args.kv_cache_config.dtype)
                if uses_fp4_mla_attention(fp4_mla_config):
                    llm_args.kv_cache_config.use_kv_cache_manager_v2 = True
        _resolve_kv_cache_manager_v2_auto(llm_args, preference_cls,
                                          config.pretrained_config)
        _validate_and_adjust_mamba_snapshot_config(config, llm_args)
        if original_kv_cache_manager_setting == "auto":
            logger.info(
                "Resolved use_kv_cache_manager_v2='auto' to %s for %s",
                llm_args.kv_cache_config.use_kv_cache_manager_v2,
                model_cls.__name__
                if model_cls is not None else "unknown model")

        return llm_args, config

    @staticmethod
    def _needs_source_identity(checkpoint_loader: BaseCheckpointLoader,
                               load_format: LoadFormat) -> bool:
        """Whether this load path can consume a SourceIdentity gate.

        Keep ordinary loading a strict no-op: SourceIdentity is currently used
        only by MX (pre-transfer fallback gate) and GMS (pre-materialize
        strict gate), so default HF/AUTO paths should not even build one.
        """
        return load_format == LoadFormat.GMS or checkpoint_loader.checkpoint_format == CheckpointFormat.MX

    def _megamoe_fuse_shared_expert(
            self, checkpoint_loader: BaseCheckpointLoader) -> bool:
        """The moe_config switch, off when the weights are shared: the fused copy is per process."""
        requested = self.llm_args.moe_config.megamoe_fuse_shared_expert
        if requested and self._needs_source_identity(checkpoint_loader,
                                                     self.llm_args.load_format):
            logger.warning(
                "megamoe_fuse_shared_expert is off: shared weights (GMS / MX) "
                "keep the separate shared expert")
            return False
        return requested

    @staticmethod
    def _build_source_identity(
        config: ModelConfig,
        model: DecoderModelForCausalLM,
        *,
        checkpoint_dir: str,
        model_name: str,
        transform_abi_id: Optional[str],
        fallback_on_artifact_error: bool,
    ) -> Optional[SourceIdentity]:
        """Build the local identity without weakening artifact validation.

        Artifact construction remains fail-closed. MX may convert an artifact
        error into an unavailable local identity so its compatibility gate
        falls back to disk; GMS propagates the error because it has no fallback.
        """
        try:
            artifact_identity = ArtifactIdentity.from_checkpoint(checkpoint_dir)
        except (OSError, RuntimeError, ValueError) as error:
            if not fallback_on_artifact_error:
                raise
            logger.warning(
                "Unable to build checkpoint artifact identity for MX checkpoint "
                f"{checkpoint_dir}; falling back to regular checkpoint loading: {error}"
            )
            return None

        return SourceIdentity.from_model_config(
            config,
            model,
            artifact_identity=artifact_identity,
            model_name=model_name,
            transform_abi_id=transform_abi_id,
        )

    def load(
        self,
        checkpoint_dir: str,
        checkpoint_loader: BaseCheckpointLoader,
        defer_checkpoint: bool = False,
    ) -> tuple[DecoderModelForCausalLM, BaseMoeLoadBalancer | None]:
        """
        Loads the model, its weights, and applies necessary configurations.

        Args:
            checkpoint_dir: The directory of the model checkpoint.
            checkpoint_loader: The loader object for model checkpoints.
            defer_checkpoint: Where the checkpoint can load later as it would
                now (`_can_defer_checkpoint`) and the KernelBuilder rehearsal
                is on, return the model with dummy weights instead;
                `has_deferred_checkpoint` then tells, and
                `load_deferred_checkpoint` loads the checkpoint.

        Returns:
            The loaded and initialized PyTorch model.
        """
        # Reset before the total timer starts: the timer writes into this dict.
        self._metrics = {}
        self._deferred = None
        # Directories the checkpoint load reads files from, in _timed_load or,
        # deferred, in load_deferred_checkpoint.
        self._checkpoint_dirs_read: list[str] = []
        with timing_metric(
                ModelLoaderMetricNames.TOTAL_MODEL_LOADING_SECONDS.value,
                self._metrics):
            model, moe_load_balancer = self._timed_load(
                checkpoint_dir,
                checkpoint_loader,
                defer_checkpoint=defer_checkpoint)
        # A deferred load finishes once load_deferred_checkpoint has read the
        # checkpoint.
        if self._deferred is None:
            self._finish_load()
        # Attribute the parameters the model ended up with, whichever load
        # path produced them (a deferred load keeps the same modules). Only a
        # weak reference; read when reporting.
        monitor_claims.claim_module_memory(model, monitor_names.WEIGHTS)
        return model, moe_load_balancer

    def _timed_load(
        self,
        checkpoint_dir: str,
        checkpoint_loader: BaseCheckpointLoader,
        defer_checkpoint: bool = False,
    ) -> tuple[DecoderModelForCausalLM, BaseMoeLoadBalancer | None]:
        """Do all the work of `load()`, which times it as the total loading time.

        A deferred checkpoint load stops after the dummy weights
        (`_load_rehearsal_weights`).
        """
        self._checkpoint_loader = checkpoint_loader
        config: ModelConfig = self._load_and_validate_config(
            checkpoint_dir, checkpoint_loader)
        load_format = self.llm_args.load_format
        gms_loading.check_gms_locality_domain_compat(load_format, config)
        self._startup_metadata = _checkpoint_startup_metadata(
            checkpoint_loader, load_format)

        # Some model constructors normalize or rewrite config fields. Capture the
        # original model type and architecture specified in the config here
        original_config_identity = PostTransformConfigIdentity.from_model_config(
            config)

        # build the model; MoE layers register with the active load balancer
        # while they are built, so only construction needs its block
        with maybe_create_moe_load_balancer(config,
                                            self.mapping) as moe_load_balancer:
            # Some constructors adjust the config they are built from, so the
            # model's config gets its own name.
            model, model_config, is_meta_init = memory_init.create_model_with_meta_tensors(
                config)

        validate_lm_head_output_dtype(model, self.llm_args.lm_head_output_dtype)

        # raise error when the built model cannot load the checkpoint as is
        _check_post_init_model_compat(model, load_format, checkpoint_loader)

        loads_draft_weights = (self.spec_config is not None and
                               self.spec_config.needs_separate_draft_weights)
        speculative_mode = mx_loading.get_speculative_mode_name(
            self.spec_config)
        post_transform_qualification = mx_loading.qualify_post_transform_profile(
            model,
            config_identity=original_config_identity,
            speculative_mode=speculative_mode,
            loads_draft_weights=loads_draft_weights)

        # Where the checkpoint load is deferred, the tensors that hold no value
        # of their own until it loads get dummy values instead.
        rehearsal_slots: list[tuple[torch.nn.Module, str]] = []
        if defer_checkpoint and self._can_defer_checkpoint(
                model,
                checkpoint_loader,
                is_meta_init=is_meta_init,
                moe_load_balancer=moe_load_balancer):
            rehearsal_slots = _meta_tensor_slots(model)

        # GMS allocates its own memory later
        if load_format != LoadFormat.GMS:
            weight_pool = memory_init.allocate_model_on_cuda(
                model,
                is_meta_init=is_meta_init,
                load_format=load_format,
                model_weights_memory_tag=self.model_weights_memory_tag,
                model_weights_restore_mode=self.model_weights_restore_mode)
            if weight_pool is not None:
                self._weight_pool_proxy = weight_pool

        source_identity: Optional[SourceIdentity] = None
        if self._needs_source_identity(checkpoint_loader, load_format):
            # Receiver's local SourceIdentity, built once from the final
            # ModelConfig and the freshly constructed module. The module's
            # realized parameter/buffer (shape, dtype) map is the layout
            # ground truth; building it here (post-construction,
            # pre-weight-load) gives producer and consumer a common,
            # comparable lifecycle point.
            source_identity = self._build_source_identity(
                model_config,
                model,
                checkpoint_dir=checkpoint_dir,
                model_name=str(
                    getattr(self.llm_args, "model", None) or checkpoint_dir),
                transform_abi_id=(
                    post_transform_qualification.transform_abi_id),
                fallback_on_artifact_error=(load_format != LoadFormat.GMS and
                                            checkpoint_loader.checkpoint_format
                                            == CheckpointFormat.MX),
            )

        if rehearsal_slots:
            self._deferred = self._load_rehearsal_weights(
                model,
                rehearsal_slots,
                model_config=model_config,
                checkpoint_dir=checkpoint_dir,
                checkpoint_loader=checkpoint_loader,
                loads_draft_weights=loads_draft_weights,
                source_identity=source_identity,
                post_transform_qualification=post_transform_qualification)
            return model, moe_load_balancer

        weights_preloaded = False
        if load_format == LoadFormat.AUTO:
            weights_preloaded = self._load_checkpoint_weights(
                checkpoint_loader,
                checkpoint_dir,
                model=model,
                model_config=model_config,
                loads_draft_weights=loads_draft_weights,
                source_identity=source_identity,
                post_transform_qualification=post_transform_qualification)

        elif load_format == LoadFormat.GMS:
            gms_result = gms_loading.load_gms_weights(
                model,
                gms_config=self.llm_args.gms_config,
                mapping=self.mapping,
                llm_args=self.llm_args,
                checkpoint_dir=checkpoint_dir,
                checkpoint_loader=checkpoint_loader,
                model_config=model_config,
                is_meta_init=is_meta_init,
                loads_draft_weights=loads_draft_weights,
                speculative_model=(self.spec_config.speculative_model
                                   if loads_draft_weights else None),
                qualification=post_transform_qualification,
                source_identity=source_identity,
                metrics=self._metrics,
                startup_metadata=self._startup_metadata)
            self._gms_backend = gms_result.gms_backend
            weights_preloaded = gms_result.weights_preloaded
            if gms_result.weight_mapper is not None:
                self.weight_mapper = gms_result.weight_mapper
            # A writer reads the checkpoint unless its loader preloaded the
            # weights, and reads a separate draft; a reader reads neither.
            gms_writer = (gms_result.weight_mapper is not None
                          or gms_result.weights_preloaded)
            if gms_result.weight_mapper is not None:
                self._checkpoint_dirs_read.append(
                    getattr(model, "llm_checkpoint_dir", checkpoint_dir))
            if gms_writer and loads_draft_weights:
                self._checkpoint_dirs_read.append(
                    self.spec_config.speculative_model)

        elif load_format == LoadFormat.DUMMY:
            self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                model, model_config)
            initialize_dummy_weights(model)
            if loads_draft_weights:
                model.draft_model.load_weights_from_target_model(model)

        elif load_format == LoadFormat.VISION_ONLY:
            # Vision weights are already loaded within the model.
            logger.info(
                "LoadFormat.VISION_ONLY: skipping weight loading; using preloaded vision weights."
            )

        else:
            raise NotImplementedError(
                f"No load support for load format: {load_format}")

        self._post_load(
            checkpoint_loader,
            model,
            checkpoint_dir=checkpoint_dir,
            load_format=load_format,
            weights_preloaded=weights_preloaded,
            post_transform_qualification=post_transform_qualification,
            source_identity=source_identity,
            moe_load_balancer=moe_load_balancer)
        return model, moe_load_balancer

    def _can_defer_checkpoint(
        self,
        model: DecoderModelForCausalLM,
        checkpoint_loader: BaseCheckpointLoader,
        *,
        is_meta_init: bool,
        moe_load_balancer: BaseMoeLoadBalancer | None,
    ) -> bool:
        """Whether the checkpoint can load after a KernelBuilder rehearsal on
        dummy weights as it would load now, and the rehearsal runs.

        Only where the AUTO loader of a checkpoint that is not MX fills tensors
        materialized from meta outside a weights memory tag, nothing else
        holds them (MoE load balancer, pinned host copies), the module tree is
        the Modeling V1 one the refit hooks cover, and every module of it can
        reload its weights. And only for a model whose dummy weights build
        what its checkpoint load builds, where its runner rehearses.
        """
        if not (self.llm_args.load_format == LoadFormat.AUTO
                and checkpoint_loader.checkpoint_format != CheckpointFormat.MX
                and is_meta_init and self.model_weights_memory_tag is None
                and moe_load_balancer is None and getattr(
                    model.model_config, "modeling_v2_target", None) is None):
            return False
        if not all(
                getattr(module, "supports_weight_reload", True) and not getattr(
                    module, "_requires_regular_checkpoint_loading", False)
                for module in model.modules()):
            return False
        return (KernelBuilder.get().rehearse
                and getattr(model, "rehearses_on_dummy_weights", False)
                and runner_rehearses(model, self.llm_args))

    def _load_rehearsal_weights(
        self,
        model: DecoderModelForCausalLM,
        rehearsal_slots: list[tuple[torch.nn.Module, str]],
        *,
        model_config: ModelConfig,
        checkpoint_dir: str,
        checkpoint_loader: BaseCheckpointLoader,
        loads_draft_weights: bool,
        source_identity: Optional[SourceIdentity],
        post_transform_qualification: PostTransformQualificationDecision,
    ) -> _DeferredCheckpoint:
        """What `LoadFormat.DUMMY` loads, into the tensors that hold no value
        (``rehearsal_slots``), and the post-load hooks on these dummy weights.

        Returns what `load_deferred_checkpoint` needs to load the checkpoint:
        the module tree as created and the inputs of the load.
        """
        logger.info(
            "Loading dummy weights for a KernelBuilder rehearsal; the checkpoint "
            "loads after it.")
        deferred = _DeferredCheckpoint(
            model_config=model_config,
            checkpoint_dir=checkpoint_dir,
            checkpoint_loader=checkpoint_loader,
            loads_draft_weights=loads_draft_weights,
            source_identity=source_identity,
            post_transform_qualification=post_transform_qualification,
            tree=_ModuleTree(model),
        )
        with timing_metric(
                ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.value,
                self._metrics):
            # Read from the slots after the allocation; no reference outlives
            # the fill, so what the post-load hooks release is freed.
            fill_rehearsal_values(
                getattr(module, name).detach()
                for module, name in rehearsal_slots)
            initialize_dummy_states(model)
            if loads_draft_weights:
                model.draft_model.load_weights_from_target_model(model)
            model_hooks.walk_full_post_load(model)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if LOW_M_GEMM_ACTIVE:
                prepare_low_m_gemm(model)
        return deferred

    def _load_checkpoint_weights(
        self,
        checkpoint_loader: BaseCheckpointLoader,
        checkpoint_dir: str,
        *,
        model: DecoderModelForCausalLM,
        model_config: ModelConfig,
        loads_draft_weights: bool,
        source_identity: Optional[SourceIdentity],
        post_transform_qualification: PostTransformQualificationDecision,
    ) -> bool:
        """The `LoadFormat.AUTO` read of the target and any separate draft
        checkpoint; the directories it reads files from are recorded for the
        page-cache release.

        Returns:
            Whether the checkpoint loader wrote the target weights into the
            model directly.
        """
        # read the checkpoint with the checkpoint loader, built-in or custom
        with _weight_load_threads(self._weight_load_num_threads):
            self.weight_mapper, weights_preloaded = load_model_from_auto_load_format(
                checkpoint_loader,
                checkpoint_dir,
                model_config=model_config,
                model=model,
                mapping=self.mapping,
                llm_args=self.llm_args,
                spec_config=self.spec_config,
                source_identity=source_identity,
                qualification=post_transform_qualification,
                metrics=self._metrics,
                startup_metadata=self._startup_metadata)
        if not weights_preloaded:
            self._checkpoint_dirs_read.append(
                getattr(model, "llm_checkpoint_dir", checkpoint_dir))
        if loads_draft_weights:
            self._checkpoint_dirs_read.append(
                self.spec_config.speculative_model)
        return weights_preloaded

    def _post_load(
        self,
        checkpoint_loader: BaseCheckpointLoader,
        model: DecoderModelForCausalLM,
        *,
        checkpoint_dir: str,
        load_format: LoadFormat,
        weights_preloaded: bool,
        post_transform_qualification: PostTransformQualificationDecision,
        source_identity: Optional[SourceIdentity],
        moe_load_balancer: BaseMoeLoadBalancer | None,
    ) -> None:
        """The post-load hooks and publish step, the MoE load balancer's
        finalization, the release of the memory they freed, the final weight
        manifest and the low-M GEMM dispatch."""
        with timing_metric(
                ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.value,
                self._metrics):
            # GMS RW and RO have already run the post_load_* hooks
            # themselves. RW handles them inside `mem_pool_scope` so the
            # committed pool reflects the post-post_load layout; RO runs
            # `setup_aliases()` before `materialize_module` to wire aliases
            # prior to zero-copy mapping, then refreshes derived state after
            # real GMS tensors are bound.
            if load_format != LoadFormat.GMS:
                checkpoint_loader.post_load_apply(
                    model, weights_preloaded=weights_preloaded)
                mx_loading.run_post_load_walks_and_publish(
                    checkpoint_loader,
                    model,
                    checkpoint_dir=checkpoint_dir,
                    weights_preloaded=weights_preloaded,
                    qualification=post_transform_qualification,
                    source_identity=source_identity)

            # TODO(GMS-MOE-LB): when the (MoE, GMS) combination is enabled,
            # `register_weight_slots_after_to_cuda` and `finalize_model`
            # must run INSIDE `mem_pool_scope` and BEFORE `finalize_write`
            # so MoE allocations become part of the committed layout that
            # RO peers receive. Today they run outside the pool and after
            # commit, which would silently produce a broken MoE routing
            # state on RO peers — that combination is REJECTED at config
            # time by `TorchLlmArgs.validate_gms_moe_compat` in
            # `tensorrt_llm/llmapi/llm_args.py`. When implementing the
            # follow-up, drop the validator gate AFTER moving these calls
            # into the pool scope.
            if isinstance(moe_load_balancer, BaseMoeLoadBalancer):
                moe_load_balancer.register_weight_slots_after_to_cuda()
                logger.info("moe_load_balancer finalizing model...")
                moe_load_balancer.finalize_model()
                logger.info("moe_load_balancer finalize model done")

            if torch.cuda.is_available():  # CPU guard
                torch.cuda.synchronize()
            # Reclaim segments freed during per-module post_load_weights (e.g.
            # MegaMoE _transform_main_weights releases ~5-6 GiB of redundant
            # weight Parameters via `.data = empty(0)` that PyTorch's caching
            # allocator otherwise holds onto). Returning them to the driver
            # gives downstream stages (KV cache estimation, attention workspace
            # alloc, autotuner warmup symmetric-fabric setup) full visibility
            # of free HBM. Single one-shot call after all modules are
            # finalized; per-layer empty_cache here is unsafe because it can
            # perturb NVLink barrier synchronization in multi-rank DG init.
            torch.cuda.empty_cache()

            manifest_result = mx_loading.dump_final_weight_manifest(
                checkpoint_loader,
                model,
                weights_preloaded=weights_preloaded,
                mapping=self.mapping,
                load_format=self.llm_args.load_format)
            if manifest_result is not None:
                self._metrics[ModelLoaderMetricNames.WEIGHT_MANIFEST_SECONDS.
                              value] = (manifest_result.build_seconds +
                                        manifest_result.write_seconds)

            if LOW_M_GEMM_ACTIVE:
                prepare_low_m_gemm(model)

    def _finish_load(self) -> None:
        """Drop the page cache of the checkpoint files the load read, and log
        the loading metrics."""
        release_seconds = _release_checkpoint_page_cache(
            self._checkpoint_dirs_read)
        if release_seconds is not None:
            self._metrics[
                _CHECKPOINT_PAGE_CACHE_RELEASE_SECONDS] = release_seconds

        metrics = ", ".join(f"{name}={value:.4f}"
                            for name, value in self._metrics.items())
        logger.info(
            f"Model loading metrics for rank {self.mapping.rank}: {metrics}")

    @property
    def has_deferred_checkpoint(self) -> bool:
        """True after `load(defer_checkpoint=True)` returned a model with dummy
        weights, until `load_deferred_checkpoint` loads the checkpoint."""
        return self._deferred is not None

    def load_deferred_checkpoint(self, model: DecoderModelForCausalLM) -> None:
        """Load the checkpoint a `load(defer_checkpoint=True)` left out.

        The model goes back to the state `load` had it in before it put dummy
        weights in: the module tree as created, and each parameter in its
        creation layout, through `pre_reload_weights` as for a weight update
        and otherwise created again. The rest of the load then runs as `load`
        runs it, the page-cache release of the files it reads included.

        Args:
            model: The model `load` returned.
        """
        deferred, self._deferred = self._deferred, None
        assert deferred is not None, "no checkpoint load was deferred"
        qualification = deferred.post_transform_qualification
        with timing_metric(
                ModelLoaderMetricNames.TOTAL_MODEL_LOADING_SECONDS.value,
                self._metrics):
            torch.cuda.synchronize()
            deferred.tree.restore()
            for module in model.modules():
                if hasattr(module, "pre_reload_weights") and not getattr(
                        module, "_weights_removed", False):
                    module.pre_reload_weights()
            deferred.tree.restore_parameters()
            model_hooks.reset_weights_transformed(model)
            weights_preloaded = self._load_checkpoint_weights(
                deferred.checkpoint_loader,
                deferred.checkpoint_dir,
                model=model,
                model_config=deferred.model_config,
                loads_draft_weights=deferred.loads_draft_weights,
                source_identity=deferred.source_identity,
                post_transform_qualification=qualification)
            self._post_load(deferred.checkpoint_loader,
                            model,
                            checkpoint_dir=deferred.checkpoint_dir,
                            load_format=LoadFormat.AUTO,
                            weights_preloaded=weights_preloaded,
                            post_transform_qualification=qualification,
                            source_identity=deferred.source_identity,
                            moe_load_balancer=None)
        self._finish_load()

    def reload(self,
               model: DecoderModelForCausalLM,
               weights: dict,
               allow_partial_loading: bool = False) -> None:
        """Reload model weights without running post-load hooks.

        Reload is used by incremental update paths that may provide only a
        partial set of replacement weights. Full reloads reset transform guards
        before rebinding fresh weights. Partial reloads keep existing transform
        guards intact because untouched modules may already contain transformed
        live weights. The owner of the update lifecycle is responsible for
        running post-load processing once all bytes are present. Weight
        manifests are not written here; see
        `mx_loading.dump_final_weight_manifest`.

        A torch.compile wrapper left in the tree is fine for models that load
        through the generic loaders (they strip the wrapper's ``_orig_mod``
        path component, ``modeling_utils.strip_torch_compile_wrapper``); a
        model with its own ``load_weights`` walk must be unwrapped first, and
        ``_check_compile_wrapper_before_reload`` fails loudly otherwise.

        Args:
            model: Model instance receiving the replacement weights.
            weights: Checkpoint weights to pass to `model.load_weights`.
            allow_partial_loading: Whether missing replacement weights are
                allowed by models that support partial loading.

        Returns:
            None.

        Raises:
            RuntimeError: Dense DWDP sharded the model's dense weights, or the
                weight mapper was not initialized.
        """
        _reject_dense_dwdp_update(model, "reload weights")
        if self.weight_mapper is None:
            raise RuntimeError(
                "Cannot reload weights: weight_mapper was not initialized. "
                "This can happen when the initial load used GMS, MX P2P, or "
                "VISION_ONLY, which bypass the standard weight mapping path.")
        _check_compile_wrapper_before_reload(model)
        if not allow_partial_loading:
            model_hooks.reset_weights_transformed(model)
        with _weight_load_threads(self._weight_load_num_threads):
            model_hooks.call_load_weights(
                model.load_weights,
                weights,
                self.weight_mapper,
                allow_partial_loading=allow_partial_loading)
        if torch.cuda.is_available():  # CPU guard
            torch.cuda.current_stream().synchronize()

    def begin_update_weights(self) -> None:
        """Start an incremental update session on the persistent mapper.

        Raises:
            RuntimeError: The weight mapper was not initialized, or dense DWDP
                sharded the dense weights of the model it is bound to.
        """
        if self.weight_mapper is None:
            raise RuntimeError(
                "Cannot update weights: weight_mapper was not initialized")
        _reject_dense_dwdp_update(self.weight_mapper.model, "update weights")
        self.weight_mapper.begin_update_weights()

    def finalize_update_weights(self) -> None:
        """Validate and close the persistent mapper update session."""
        if self.weight_mapper is None:
            raise RuntimeError(
                "Cannot update weights: weight_mapper was not initialized")
        self.weight_mapper.finalize_update_weights()

    def abort_update_weights(self) -> None:
        """Discard persistent mapper state after a failed update session."""
        if self.weight_mapper is not None:
            self.weight_mapper.abort_update_weights()

    def cleanup(self) -> None:
        """Release backend resources acquired during :meth:`load`.

        This releases the optional GMS client and the active checkpoint
        loader. Releasing GMS disconnects from the daemon and evicts the
        per-tag client registry entry; weights remain alive on-device for
        any other process holding an RO lock on the same `tag`.

        A second call is a no-op because both handles are dropped. Cleanup is
        best effort so shutdown and destructor paths do not propagate backend
        failures.
        """
        if self._gms_backend is not None:
            gms_backend = self._gms_backend
            self._gms_backend = None
            # GMSBackend.cleanup() logs and swallows its own errors.
            gms_backend.cleanup()
        if self._checkpoint_loader is not None:
            try:
                self._checkpoint_loader.cleanup()
            except Exception as exc:  # noqa: BLE001 - shutdown must remain best effort
                logger.warning(f"Failed to clean up checkpoint loader "
                               f"{self._checkpoint_loader!r}: {exc!r}")
            finally:
                self._checkpoint_loader = None

    def _load_and_validate_config(
            self, checkpoint_dir: str,
            checkpoint_loader: BaseCheckpointLoader) -> ModelConfig:
        """Loads and validates the model configuration."""
        from tensorrt_llm._torch.speculative.utils import (
            resolve_mtp_checkpoint_source, update_spec_config_from_model_config,
            uses_mtp_head_checkpoint)

        resolve_mtp_checkpoint_source(self.spec_config, checkpoint_dir)

        # Mirror use_fine_grained_sync into env var so C++ getEnvUseFineGrainedSync()
        # sees it at kernel-option construction time. Never write "0": that would
        # clobber an externally exported TLLM_USE_FINE_GRAINED_SYNC=1.
        if self.llm_args.use_fine_grained_sync:
            os.environ["TLLM_USE_FINE_GRAINED_SYNC"] = "1"
        load_config_kwargs = dict(
            checkpoint_dir=checkpoint_dir,
            trust_remote_code=self.llm_args.trust_remote_code,
            mapping=self.mapping,
            enable_min_latency=self.llm_args.enable_min_latency,
            use_cuda_graph=self.llm_args.cuda_graph_config is not None,
            force_dynamic_quantization=self.llm_args.force_dynamic_quantization,
            spec_config=self.spec_config,
            sparse_attention_config=self.sparse_attention_config,
            kv_cache_compression_config=(
                self.llm_args.kv_cache_compression_config),
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            moe_max_num_tokens=self.llm_args.moe_config.max_num_tokens,
            moe_load_balancer=self.llm_args.moe_config.load_balancer,
            lora_config=self.lora_config,
            allreduce_strategy=self.llm_args.allreduce_strategy,
            mnnvl_allreduce_fp8_payload=self.llm_args.
            mnnvl_allreduce_fp8_payload,
            replicate_embedding=self.llm_args.replicate_embedding,
            cuda_graph_batch_sizes=captured_cuda_graph_batch_sizes(
                self.llm_args, self.max_num_tokens, self.spec_config),
            max_batch_size=self.llm_args.max_batch_size,
            mm_encoder_only=self.llm_args.mm_encoder_only,
            disable_mm_encoder=self.llm_args.disable_mm_encoder,
            attn_backend=self.llm_args.attn_backend,
            skip_correction_threshold=(
                self.llm_args.mla_skip_correction_threshold
                if self.llm_args.enable_mla_skip_correction else 0.0),
            moe_backend=self.llm_args.moe_config.backend,
            moe_tactic=self.llm_args.moe_config.tactic,
            moe_disable_finalize_fusion=self.llm_args.moe_config.
            disable_finalize_fusion,
            megamoe_fuse_shared_expert=self._megamoe_fuse_shared_expert(
                checkpoint_loader),
            use_low_precision_moe_combine=self.llm_args.moe_config.
            use_low_precision_moe_combine,
            nvfp4_gemm_allowed_backends=self.llm_args.nvfp4_gemm_config.
            allowed_backends,
            use_cute_dsl_blockscaling_mm=self.llm_args.
            use_cute_dsl_blockscaling_mm,
            use_cute_dsl_blockscaling_bmm=self.llm_args.
            use_cute_dsl_blockscaling_bmm,
            video_pruning_rate=self.llm_args.multimodal_config.
            video_pruning_rate,
            multimodal_config=self.llm_args.multimodal_config,
            use_cute_dsl_bf16_bmm=self.llm_args.use_cute_dsl_bf16_bmm,
            use_cute_dsl_bf16_gemm=self.llm_args.use_cute_dsl_bf16_gemm,
            lm_head_output_dtype=(torch.float32
                                  if self.llm_args.lm_head_output_dtype
                                  == "float32" else None),
            locality_domain_policy=LocalityDomainPolicy(
                enabled=self.llm_args.enable_locality_domains),
        )

        # Only pass model_kwargs if it's explicitly set (not None)
        if self.llm_args.model_kwargs is not None:
            load_config_kwargs['model_kwargs'] = self.llm_args.model_kwargs

        config = checkpoint_loader.load_config(**load_config_kwargs)
        # Check after model_kwargs, which can remove an encoder.
        _validate_encoder_attention_backend_enabled(
            self.llm_args.multimodal_config,
            config.pretrained_config,
            disable_mm_encoder=self.llm_args.disable_mm_encoder)

        # Decided here, after `load_config_and_apply_defaults` resolved the
        # built-in class and applied its model defaults to `llm_args`: a
        # modeling_v2 target reads the arguments as the engine will run with
        # them and never overrides those defaults.
        config = replace(config,
                         modeling_v2_target=modeling_v2_resolve_on_all_ranks(
                             config, self.llm_args))

        if uses_mtp_head_checkpoint(self.spec_config):
            # `load_config_and_apply_defaults` already ran this, but against a
            # config object it then discards. The MTP heads' structure fields
            # (head count, block pattern) come from `speculative_model` and
            # have to reach the config the model is actually built from.
            update_spec_config_from_model_config(self.spec_config,
                                                 config.pretrained_config)

        # Store nvfp4 config in extra_attrs for Linear layer access
        config.extra_attrs[
            'nvfp4_gemm_allowed_backends'] = config.nvfp4_gemm_allowed_backends
        config.extra_attrs[
            'kv_cache_dtype'] = self.llm_args.kv_cache_config.dtype
        config.extra_attrs["bounded_replay_on_generation"] = (
            self.llm_args.bounded_replay_on_generation)
        config.extra_attrs["breakable_prefill_cuda_graph"] = (
            self.llm_args.prefill_cuda_graph_backend ==
            PrefillCudaGraphBackend.BREAKABLE)
        # Store allreduce pre-allocation config for AllReduce module access.
        # Use get_text_config() so VLM wrapper configs (e.g. KimiK2VLConfig,
        # KimiK25Config) that store the text config under .text_config are
        # handled transparently.  For flat configs get_text_config() returns
        # self, so this is safe for all config types.  Still guard with
        # try/except for configs that lack hidden_size entirely.
        try:
            config.extra_attrs[
                'allreduce_max_num_tokens'] = config.max_num_tokens
            config.extra_attrs[
                'allreduce_hidden_size'] = config.pretrained_config.get_text_config(
                ).hidden_size
            config.extra_attrs[
                'allreduce_dtype'] = config.pretrained_config.torch_dtype
        except AttributeError as e:
            logger.warning(
                f"Could not read allreduce pre-allocation config from "
                f"{type(config.pretrained_config).__name__}: {e}. "
                f"AllReduce pre-allocation will be skipped.")

        validate_encoder_decoder_tp_scope(config)
        validate_encoder_decoder_kv_cache_config(config,
                                                 self.llm_args.kv_cache_config)
        validate_and_set_kv_cache_quant(config,
                                        self.llm_args.kv_cache_config.dtype)
        validate_fp4_mla_config(config, self.llm_args)
        validate_and_set_mamba_ssm_cache_dtype(
            config, self.llm_args.kv_cache_config.mamba_ssm_cache_dtype,
            self.llm_args.kv_cache_config.mamba_ssm_stochastic_rounding,
            self.llm_args.kv_cache_config.mamba_ssm_philox_rounds)

        # Allow overriding the number of layers via environment variable
        # Note: This is kept for backward compatibility, but model_kwargs is preferred
        num_layers_override = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM",
                                                 "0"))
        if num_layers_override > 0:
            logger.warning(
                f"TLLM_OVERRIDE_LAYER_NUM is deprecated. Use model_kwargs instead: "
                f"model_kwargs={{'num_hidden_layers': {num_layers_override}}}")
            config.pretrained_config.num_hidden_layers = num_layers_override
            for sub_config in ["text_config", "vision_config"]:
                if hasattr(config.pretrained_config, sub_config):
                    getattr(config.pretrained_config,
                            sub_config).num_hidden_layers = num_layers_override

        # Shared-weights vanilla MTP: build extra MTP layer instances beyond
        # what the checkpoint provides (one ckpt MTP layer, multiple draft
        # tokens, one KV cache per draft position) by sharing the single
        # ckpt MTP layer's weights via mod-indexing in
        # DeepseekV3WeightLoader. We expand
        # pretrained_config.num_nextn_predict_layers to max_draft_len before
        # model construction and preserve the original ckpt count as
        # `_ckpt_num_nextn_predict_layers` for downstream mod-indexing.
        #
        # NOTE: this is a very special MTP mode that has not been used in
        # any real-world workload to date; only DeepSeek has indicated they
        # want to keep the path alive for their model. We therefore only
        # support it on DeepSeek model_types for now. Other MTP-capable
        # model families don't need this mode -- when their users request
        # vanilla with max_draft_len > ckpt count, the natural
        # `min(max_draft_len, ckpt_nextn)` clamp inside MTPForCausalLM
        # silently caps the draft length to ckpt_nextn, which is the
        # expected behavior for them.
        _DEEPSEEK_MTP_MODEL_TYPES = {"deepseek_v3", "deepseek_v32"}
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
        spec_config = self.spec_config
        if (isinstance(spec_config, MTPDecodingConfig)
                and spec_config.use_mtp_vanilla
                and spec_config.max_draft_len is not None
                and getattr(config.pretrained_config, 'model_type',
                            None) in _DEEPSEEK_MTP_MODEL_TYPES
                and getattr(config.pretrained_config,
                            'num_nextn_predict_layers', None)):
            ckpt_nextn = config.pretrained_config.num_nextn_predict_layers
            if spec_config.max_draft_len > ckpt_nextn:
                config.pretrained_config._ckpt_num_nextn_predict_layers = ckpt_nextn
                config.pretrained_config.num_nextn_predict_layers = \
                    spec_config.max_draft_len
                logger.warning(
                    f"MTP vanilla: expanding num_nextn_predict_layers from "
                    f"{ckpt_nextn} to {spec_config.max_draft_len} to match "
                    f"max_draft_len. Extra MTP layer instances will share "
                    f"checkpoint weights via mod-indexing.")
        return config
