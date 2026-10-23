# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from dataclasses import replace
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import torch

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
    BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import \
    HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
    HfWeightLoader
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.weight_sharing import (
    ArtifactIdentity, PostTransformConfigIdentity,
    PostTransformQualificationDecision, SourceIdentity)
from tensorrt_llm._utils import get_sm_version, str_dtype_to_torch
from tensorrt_llm.checkpoint_format import CheckpointFormat
from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig,
                                          ExecutorMemoryType,
                                          ModelExpressConfig,
                                          SparseAttentionConfig, TorchLlmArgs)
from tensorrt_llm.llmapi.llm_utils import (_resolve_kv_cache_manager_v2_auto,
                                           _resolve_transceiver_runtime_auto,
                                           apply_model_defaults_to_llm_args)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.quantization.utils.fp4_utils import float4_e2m1x2

from ...llmapi.llm_args import CheckpointIOPolicy, LoadFormat
from .._experimental.modeling_v2 import modeling_v2_resolve_on_all_ranks
from ..locality_domain.policy import LocalityDomainPolicy
from ..model_config import ModelConfig
from ..models import AutoModelForCausalLM
from ..models.checkpoints import memory_init, model_hooks
from ..models.checkpoints.checkpoint_catalog import CheckpointCatalog
from ..models.checkpoints.gms import model_loading as gms_loading
from ..models.checkpoints.mx import model_loading as mx_loading
from ..models.checkpoints.weight_load_plan import WeightLoadPlan
from ..models.modeling_utils import (DecoderModelForCausalLM,
                                     get_registered_model_class, timing_metric)
from ..modules.embedding import Embedding, LMHead
from ..modules.low_m_gemm import LOW_M_GEMM_ACTIVE, prepare_low_m_gemm
from ..moe.fused_moe.moe_load_balancer import (MoeLoadBalancer,
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
    # Mixed-precision KV: the retained tokens and attention use the second
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
        raise ValueError(
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

    effective_kv_cache_quant = (kv_cache_quant if pyt_kv_cache_dtype == "auto"
                                else mapped_pyt_quant)
    # MiniMax-M3 MSA keeps NVFP4 only on its sparse layers, which MSA and
    # Triton serve; its dense and draft layers stay FP8 for trtllm-gen.
    sparse_config = model_config.sparse_attention_config
    uses_minimax_m3_msa = (is_minimax_m3(sparse_config)
                           and sparse_config.implementation == "msa")
    if (effective_kv_cache_quant in (QuantAlgo.NVFP4, QuantAlgo.NVFP4.value)
            and not supports_fp4_mla_attention(model_config)
            and not uses_minimax_m3_msa and torch.cuda.is_available()
            and get_sm_version() == 107):
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

    # An unquantized retained dtype has to be the model dtype; KVCacheManagerV2
    # would otherwise reject it after the weights are loaded.
    if mapped_pyt_quant is None:
        retained_dtype = str_dtype_to_torch(
            pyt_kv_cache_dtype.rsplit("_", 1)[1])
        if model_config.torch_dtype != retained_dtype:
            raise ValueError(
                f'llm_args.KvCacheConfig.dtype="{pyt_kv_cache_dtype}" keeps '
                f"the retained tokens in {retained_dtype}, but the model dtype "
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
        raise ValueError("Dense FP4 MLA requires the TRTLLM attention backend; "
                         "hybrid linear attention is not supported.")
    if model_config.mapping.cp_size != 1:
        raise ValueError("FP4 MLA does not support context parallelism.")
    if llm_args.kv_cache_config.use_kv_cache_manager_v2 is False:
        raise ValueError("FP4 MLA requires use_kv_cache_manager_v2=True.")
    if llm_args.enable_chunked_prefill:
        raise ValueError(
            "FP4 MLA does not support chunked prefill or cached context; "
            "set enable_chunked_prefill=False.")

    spec_config = model_config.spec_config
    if spec_config is not None and spec_config.spec_dec_mode.use_one_engine():
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
            raise ValueError(
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
        raise ValueError(
            "kv_cache_config.cross_kv_cache_fraction should only be set for encoder-decoder models."
        )


def validate_encoder_decoder_tp_scope(model_config: ModelConfig) -> None:
    """Validate initially supported encoder-decoder TP combinations."""
    if not model_config.is_encoder_decoder:
        return

    mapping = model_config.mapping
    if mapping.enable_attention_dp:
        raise ValueError(
            "Encoder-decoder models do not support attention DP yet. "
            "Set enable_attention_dp=False.")

    if mapping.cp_size > 1:
        raise ValueError(
            "Encoder-decoder models do not support context parallelism yet. "
            "Set context_parallel_size=1.")

    if mapping.pp_size > 1:
        raise ValueError(
            "Encoder-decoder models do not support pipeline parallelism yet. "
            "Set pipeline_parallel_size=1.")

    if mapping.tp_size > 1 and model_config.attn_backend != "TRTLLM":
        raise ValueError(
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

    def _get_random_min_max(dtype: torch.dtype) -> Tuple[int, int]:
        """Return safe (min, max) bounds for uniform sampling of `dtype`."""
        # These values are not necessarily the largest possible min/max,
        # they need to be small enough to avoid NaNs.
        if dtype in (torch.float8_e4m3fn, torch.int8):
            return (-3.0, 3.0)

        elif dtype == float4_e2m1x2:
            # These correspond to bits of 2 packed FP4 values.
            # Because we only go up to 64, the high 4 bits will
            # always be 0. But this is fine - we just need values
            # that won't generate NaNs.
            return (0, 64)

        else:
            raise NotImplementedError(f"Unknown quantized type: {dtype}.")

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
            # We need to do a cast/round since torch doesn't have uniform_
            # support for these dtypes.
            tmp_param = torch.empty(param.data.shape,
                                    dtype=torch.float16,
                                    device=param.data.device)

            quant_min, quant_max = _get_random_min_max(dtype)
            tmp_param = tmp_param.uniform_(quant_min,
                                           quant_max,
                                           generator=generator)

            param.data.copy_(tmp_param.to(dtype))

        # Note: no need to to mess with int32 params, these are probably
        # constants and not weights.
        elif torch.is_floating_point(param):
            param.uniform_(low, high, generator=generator)

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


class ModelLoaderMetricNames(Enum):
    TOTAL_MODEL_LOADING_SECONDS = "total_model_loading_seconds"
    CHECKPOINT_PREPARATION_SECONDS = "checkpoint_preparation_seconds"
    WEIGHT_POPULATION_SECONDS = "weight_population_seconds"
    CHECKPOINT_FINALIZATION_SECONDS = "checkpoint_finalization_seconds"
    DRAFT_CHECKPOINT_PREPARATION_SECONDS = (
        "draft_checkpoint_preparation_seconds")
    DRAFT_WEIGHT_POPULATION_SECONDS = "draft_weight_population_seconds"
    DRAFT_CHECKPOINT_FINALIZATION_SECONDS = (
        "draft_checkpoint_finalization_seconds")
    POST_LOAD_PROCESSING_SECONDS = "post_load_processing_seconds"
    WEIGHT_MANIFEST_SECONDS = "weight_manifest_seconds"


def _checkpoint_startup_metadata(
    checkpoint_loader: BaseCheckpointLoader,
    load_format: LoadFormat,
) -> dict[str, str]:
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
    }


def _check_post_init_model_compat(
        model: DecoderModelForCausalLM, load_format: LoadFormat,
        checkpoint_loader: BaseCheckpointLoader) -> None:
    """Raise error if the initialized LLM model cannot load the checkpoint.
    """
    requires_standard_hf_loading = any(
        getattr(module, "_requires_standard_hf_loading", False)
        for module in model.modules())
    # Pinned-host parameters must be materialized and filled in place.
    # AUTO uses the HF mapper that preserves their stable addresses.
    if requires_standard_hf_loading and (load_format != LoadFormat.AUTO
                                         or checkpoint_loader.checkpoint_format
                                         != CheckpointFormat.HF):
        raise ValueError(
            "Host-resident model weights currently require the standard "
            "load_format=AUTO and checkpoint_format=HF")


def _load_checkpoint_into_model(
    checkpoint_loader: BaseCheckpointLoader,
    checkpoint_dir: str,
    *,
    config: ModelConfig,
    model: DecoderModelForCausalLM,
    is_draft_model: bool,
    mapping: Mapping,
    weight_mapper: Any,
    extra_load_weights_kwargs: Optional[dict[str, Any]],
    metrics: dict[str, float],
) -> None:
    """Read one checkpoint and copy its weights into the model.

    Used once for the target checkpoint and once for the separate draft
    checkpoint. HF-family checkpoint loaders (HF, MX, Mistral) read inside
    their weight session, which reads once or with rank-striped read-ahead
    and times its own start and finish. Other checkpoint loaders read once
    with ``load_weights``.

    Args:
        checkpoint_loader: checkpoint loader of this load
        checkpoint_dir: checkpoint to read, e.g. the target checkpoint
            directory or ``spec_config.speculative_model``
        config: validated model config of this load, before the model was
            built; picks native read or read-ahead
        model: model that holds the target and, if any, the draft model
        is_draft_model: whether this is the separate draft checkpoint
        mapping: parallel mapping of this rank
        weight_mapper: initialized weight mapper for this checkpoint
        extra_load_weights_kwargs: extra kwargs for the read, e.g.
            ``source_identity`` and the MX kwargs for the target; or None
        metrics: model loader timing metrics, updated in place
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

    if (isinstance(checkpoint_loader, HfCheckpointLoader)
            and isinstance(checkpoint_loader.weight_loader, HfWeightLoader)):
        # read the checkpoint and copy the weights inside the HF session
        with checkpoint_loader.open_weight_session(
                checkpoint_dir, config=config,
                **load_weights_kwargs) as weights:
            _copy_weights_into_model(model_load_weights, weights, weight_mapper,
                                     population_metric.value, metrics)
        io_status = checkpoint_loader.weight_loader.last_checkpoint_io_status
        metrics[preparation_metric.value] = io_status.preparation_seconds
        metrics[finalization_metric.value] = io_status.finalization_seconds
    else:
        # Other checkpoint loaders read once, with nothing to finalize.
        # Custom CheckpointLoader and custom WeightLoader goes here:
        with timing_metric(preparation_metric.value, metrics):
            weights = checkpoint_loader.load_weights(checkpoint_dir,
                                                     **load_weights_kwargs)
        _copy_weights_into_model(model_load_weights, weights, weight_mapper,
                                 population_metric.value, metrics)
        metrics.setdefault(finalization_metric.value, 0.0)

    _inspect_shadow_weight_load_plan(checkpoint_loader, checkpoint_dir,
                                     weight_mapper, **load_weights_kwargs)


def _copy_weights_into_model(model_load_weights: Callable, weights: dict,
                             weight_mapper: Any, population_metric_name: str,
                             metrics: dict[str, float]) -> None:
    """Copy read checkpoint weights into the model and time the copy."""
    # MX P2P returns no weights after writing them into the model.
    if not weights:
        return
    with timing_metric(population_metric_name, metrics):
        model_hooks.call_load_weights(model_load_weights, weights,
                                      weight_mapper)
        if torch.cuda.is_available():  # CPU guard
            torch.cuda.synchronize()


def load_model_from_auto_load_format(
    checkpoint_loader: BaseCheckpointLoader,
    checkpoint_dir: str,
    *,
    config: ModelConfig,
    model_config: ModelConfig,
    model: DecoderModelForCausalLM,
    mapping: Mapping,
    llm_args: TorchLlmArgs,
    spec_config: Optional[DecodingBaseConfig],
    source_identity: Optional[SourceIdentity],
    qualification: PostTransformQualificationDecision,
    metrics: dict[str, float],
) -> tuple[Any, bool]:
    """Load the target and any separate draft checkpoint for ``LoadFormat.AUTO``.

    The model is already built and has CUDA memory; post-load hooks run
    afterwards in `ModelLoader.load()`.

    Custom CheckpointLoader and custom WeightLoader currently only support
    ``LoadFormat.AUTO`` and go through this function to load weights.

    Args:
        checkpoint_loader: checkpoint loader of this load
        checkpoint_dir: checkpoint directory of the target model
        config: validated model config, before the model was built
        model_config: config the model was built from
        model: model to fill
        mapping: parallel mapping of this rank
        llm_args: LLM args of this load
        spec_config: speculative decoding config, or None
        source_identity: identity of this rank's model for MX, or None
        qualification: MX post-transform profile decision for this load
        metrics: model loader timing metrics, updated in place

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
    _load_checkpoint_into_model(
        checkpoint_loader,
        weights_checkpoint_dir,
        config=config,
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
        _load_checkpoint_into_model(checkpoint_loader,
                                    spec_config.speculative_model,
                                    config=config,
                                    model=model,
                                    is_draft_model=True,
                                    mapping=mapping,
                                    weight_mapper=draft_weight_mapper,
                                    extra_load_weights_kwargs=None,
                                    metrics=metrics)
    return weight_mapper, weights_preloaded


class ModelLoader:
    """
    Handles the loading, configuration, and weight initialization of a PyTorch model.
    This class isolates model loading logic from the main execution engine.
    """

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
        self.weight_mapper = None
        self._weight_pool_proxy = None
        self._gms_backend = None
        self._checkpoint_loader: Optional[BaseCheckpointLoader] = None
        # Mostly weight loading and processing time metrics, updated when load() is called.
        self._metrics: dict[str, float] = {}
        self._startup_metadata: dict[str, str] = {}

    @property
    def metrics(self) -> dict[str, float]:
        """Return weight loading and processing time metrics."""
        return self._metrics

    @property
    def startup_metadata(self) -> dict[str, str]:
        """Return checkpoint-source metadata captured by the latest load."""
        return self._startup_metadata

    @staticmethod
    def load_config_and_apply_defaults(
            checkpoint_dir: str, llm_args: TorchLlmArgs,
            checkpoint_loader: BaseCheckpointLoader) -> TorchLlmArgs:
        """Load model config and apply model-specific defaults to llm_args."""
        if checkpoint_loader is None:
            # No config to resolve a model class from; still resolve the
            # "auto" sentinel so it never leaks past config loading.
            _resolve_transceiver_runtime_auto(llm_args)
            _resolve_kv_cache_manager_v2_auto(llm_args)
            return llm_args

        config_kwargs = {
            'trust_remote_code': llm_args.trust_remote_code,
            'mm_encoder_only': llm_args.mm_encoder_only,
        }
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

        return llm_args

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
    ) -> tuple[DecoderModelForCausalLM, MoeLoadBalancer | None]:
        """
        Loads the model, its weights, and applies necessary configurations.

        Args:
            checkpoint_dir: The directory of the model checkpoint.
            checkpoint_loader: The loader object for model checkpoints.

        Returns:
            The loaded and initialized PyTorch model.
        """
        # Reset before the total timer starts: the timer writes into this dict.
        self._metrics = {}
        with timing_metric(
                ModelLoaderMetricNames.TOTAL_MODEL_LOADING_SECONDS.value,
                self._metrics):
            model, moe_load_balancer = self._timed_load(checkpoint_dir,
                                                        checkpoint_loader)

        metrics = ", ".join(f"{name}={value:.4f}"
                            for name, value in self._metrics.items())
        logger.info(
            f"Model loading metrics for rank {self.mapping.rank}: {metrics}")
        return model, moe_load_balancer

    def _timed_load(
        self,
        checkpoint_dir: str,
        checkpoint_loader: BaseCheckpointLoader,
    ) -> tuple[DecoderModelForCausalLM, MoeLoadBalancer | None]:
        """Do all the work of `load()`, which times it as the total loading time."""
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
        if load_format == LoadFormat.GMS or checkpoint_loader.checkpoint_format == CheckpointFormat.MX:
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

        weights_preloaded = False
        if load_format == LoadFormat.AUTO:
            # Custom CheckpointLoader and custom WeightLoader currently only support
            # ``LoadFormat.AUTO`` and go through this function to load weights.
            self.weight_mapper, weights_preloaded = load_model_from_auto_load_format(
                checkpoint_loader,
                checkpoint_dir,
                config=config,
                model_config=model_config,
                model=model,
                mapping=self.mapping,
                llm_args=self.llm_args,
                spec_config=self.spec_config,
                source_identity=source_identity,
                qualification=post_transform_qualification,
                metrics=self._metrics)

        elif load_format == LoadFormat.GMS:
            gms_result = gms_loading.load_gms_weights(
                model,
                gms_config=self.llm_args.gms_config,
                mapping=self.mapping,
                llm_args=self.llm_args,
                checkpoint_dir=checkpoint_dir,
                checkpoint_loader=checkpoint_loader,
                config=model_config,
                is_meta_init=is_meta_init,
                loads_draft_weights=loads_draft_weights,
                speculative_model=(self.spec_config.speculative_model
                                   if loads_draft_weights else None),
                qualification=post_transform_qualification,
                source_identity=source_identity,
                metrics=self._metrics)
            self._gms_backend = gms_result.gms_backend
            weights_preloaded = gms_result.weights_preloaded
            if gms_result.weight_mapper is not None:
                self.weight_mapper = gms_result.weight_mapper

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
            if isinstance(moe_load_balancer, MoeLoadBalancer):
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
        return model, moe_load_balancer

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
        """
        if self.weight_mapper is None:
            raise RuntimeError(
                "Cannot reload weights: weight_mapper was not initialized. "
                "This can happen when the initial load used GMS, MX P2P, or "
                "VISION_ONLY, which bypass the standard weight mapping path.")
        _check_compile_wrapper_before_reload(model)
        if not allow_partial_loading:
            model_hooks.reset_weights_transformed(model)
        model_hooks.call_load_weights(
            model.load_weights,
            weights,
            self.weight_mapper,
            allow_partial_loading=allow_partial_loading)
        if torch.cuda.is_available():  # CPU guard
            torch.cuda.current_stream().synchronize()

    def begin_update_weights(self) -> None:
        """Start an incremental update session on the persistent mapper."""
        if self.weight_mapper is None:
            raise RuntimeError(
                "Cannot update weights: weight_mapper was not initialized")
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
            gms_loading.release_gms_backend(gms_backend)
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
            mm_encoder_only=self.llm_args.mm_encoder_only,
            disable_mm_encoder=self.llm_args.disable_mm_encoder,
            attn_backend=self.llm_args.attn_backend,
            skip_correction_threshold=(
                self.llm_args.mla_skip_correction_threshold
                if self.llm_args.enable_mla_skip_correction else 0.0),
            moe_backend=self.llm_args.moe_config.backend,
            moe_disable_finalize_fusion=self.llm_args.moe_config.
            disable_finalize_fusion,
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
