# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import os
import traceback
import warnings
from dataclasses import replace
from typing import Optional

import torch

from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.weight_sharing import (WeightManifestWriteResult,
                                                maybe_write_weight_manifest)
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

from ...llmapi.llm_args import CheckpointIOPolicy, LoadFormat
from .._experimental.modeling_v2 import modeling_v2_resolve
from ..locality_domain.policy import LocalityDomainPolicy
from ..model_config import ModelConfig
from ..models import AutoModelForCausalLM
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_utils import (DecoderModelForCausalLM, MetaInitMode,
                                     get_registered_model_class, timing_metric)
from ..modules.low_m_gemm import LOW_M_GEMM_ACTIVE, prepare_low_m_gemm
from ..moe.fused_moe.moe_load_balancer import (MoeLoadBalancer,
                                               maybe_create_moe_load_balancer)
from ..virtual_memory import RestoreMode
from ..virtual_memory import scope as virtual_memory_scope
from .config_utils import (is_hybrid_linear, is_mla,
                           resolve_auto_ssm_cache_dtype,
                           supports_fp4_mla_attention, uses_fp4_mla_attention,
                           validate_kimi_kda_state_dtype)
from .weight_load_strategies import (ModelLoaderMetricNames, WeightLoadContext,
                                     WeightLoadStrategy, model_hooks,
                                     select_weight_load_strategy)

_KV_CACHE_MAP = {
    "fp8": QuantAlgo.FP8.value,
    "fp8_ds_mla": QuantAlgo.FP8.value,
    "nvfp4": QuantAlgo.NVFP4.value,
    "auto": "auto"
}
_VALID_KV_CACHE_DTYPES = ("fp8", "fp8_ds_mla", "nvfp4", "auto")
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


def validate_and_set_mamba_ssm_cache_dtype(
        config: ModelConfig,
        mamba_ssm_cache_dtype: str,
        mamba_ssm_stochastic_rounding: bool = False,
        mamba_ssm_philox_rounds: int = 10) -> None:
    if mamba_ssm_cache_dtype == "auto":
        mamba_ssm_cache_dtype = resolve_auto_ssm_cache_dtype(
            config.pretrained_config, config.torch_dtype)
    else:
        mamba_ssm_cache_dtype = str_dtype_to_torch(mamba_ssm_cache_dtype)
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
    if (effective_kv_cache_quant in (QuantAlgo.NVFP4, QuantAlgo.NVFP4.value)
            and not supports_fp4_mla_attention(model_config)
            and torch.cuda.is_available() and get_sm_version() == 107):
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


def get_rank_model_storage(model):
    total_bytes = 0
    for _, param in model.named_parameters():
        if param.device.type == 'cuda' and param.device.index == torch.cuda.current_device(
        ):
            total_bytes += param.element_size() * param.nelement()
    for _, buf in model.named_buffers():
        if buf.device.type == 'cuda' and buf.device.index == torch.cuda.current_device(
        ):
            total_bytes += buf.element_size() * buf.nelement()
    return total_bytes


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

    from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import \
        HfCheckpointLoader
    from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
        HfWeightLoader
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
    """Return the policy to use and the reason for a fallback, if any.

    CheckpointIOPolicy.MODEL_STREAMER does not support fallback: Will raise ValueError.
    """

    if requested_policy == CheckpointIOPolicy.NATIVE:
        return CheckpointIOPolicy.NATIVE, None

    # requested_policy can be AUTO, MODEL_STREAMER or RANK_STRIPED_READ_AHEAD:
    reason = _checkpoint_io_incompatibility_reason(checkpoint_loader,
                                                   checkpoint_format,
                                                   load_format,
                                                   requested_policy,
                                                   partial_model_loading)

    if requested_policy == CheckpointIOPolicy.MODEL_STREAMER:
        if reason is not None:
            raise ValueError(
                f"checkpoint_io_policy='model_streamer' is unsupported: {reason}"
            )
        return CheckpointIOPolicy.MODEL_STREAMER, None

    # request_policy is AUTO or RANK_STRIPED_READ_AHEAD
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
    from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import \
        BaseCheckpointLoader
    from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import \
        HfWeightLoader
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

    if checkpoint_io_policy == CheckpointIOPolicy.MODEL_STREAMER:
        # `ModelStreamerLoadStrategy` streams the weights into the model
        # itself. The HF loader still supplies the config and the weight
        # mapper, and its reader stays native and unused.
        checkpoint_io_policy = CheckpointIOPolicy.NATIVE
        requested_checkpoint_io_policy = CheckpointIOPolicy.NATIVE

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


def _apply_to_buffers_only(model: torch.nn.Module, fn):
    """Apply *fn* to every buffer in *model*, skipping parameters.
    """
    for module in model.modules():
        for key, buf in module._buffers.items():
            if buf is not None:
                module._buffers[key] = fn(buf)


def _checkpoint_startup_metadata(
    checkpoint_loader: BaseCheckpointLoader,
    load_format: LoadFormat,
    weight_load_strategy: WeightLoadStrategy,
) -> dict[str, str]:
    """Describe the concrete checkpoint source selected for this load.

    ``weight_load_strategy_kind`` names the `WeightLoadStrategy`, for example
    ``"ModelStreamerLoadStrategy"``, because several strategies share the same
    checkpoint loader.
    """
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
        "weight_load_strategy_kind":
        type(weight_load_strategy).__name__,
    }


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
        self._weight_load_strategy: Optional[WeightLoadStrategy] = None
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

    def load(
        self,
        checkpoint_dir: str,
        checkpoint_loader: BaseCheckpointLoader,
    ) -> tuple[DecoderModelForCausalLM, MoeLoadBalancer | None]:
        """
        Loads the model, its weights, and applies necessary configurations.

        The steps that depend on where the weights come from (checkpoint
        files, MX, GMS, dummy values) run on the load strategy that
        `select_weight_load_strategy()` picks; `WeightLoadStrategy`
        describes them.

        Args:
            checkpoint_dir: The directory of the model checkpoint.
            checkpoint_loader: The loader object for model checkpoints.

        Returns:
            The loaded and initialized PyTorch model.
        """
        self._metrics = {}
        self._checkpoint_loader = checkpoint_loader
        config = self._load_and_validate_config(checkpoint_dir,
                                                checkpoint_loader)
        load_format = self.llm_args.load_format
        weight_load_strategy = select_weight_load_strategy(
            WeightLoadContext(
                checkpoint_dir=checkpoint_dir,
                checkpoint_loader=checkpoint_loader,
                llm_args=self.llm_args,
                mapping=self.mapping,
                spec_config=self.spec_config,
                metrics=self._metrics,
            ))
        self._weight_load_strategy = weight_load_strategy
        weight_load_strategy.check_config(config)
        self._startup_metadata = _checkpoint_startup_metadata(
            checkpoint_loader, load_format, weight_load_strategy)

        with timing_metric(
                ModelLoaderMetricNames.TOTAL_MODEL_LOADING_SECONDS.value,
                self._metrics), maybe_create_moe_load_balancer(
                    config, self.mapping) as moe_load_balancer:
            model, config, is_meta_init = self._construct_model(config)
            self._check_host_resident_weights(model, weight_load_strategy)
            weight_load_strategy.prepare(model, config)

            allocate_weights = functools.partial(self._allocate_weights,
                                                 model,
                                                 is_meta_init=is_meta_init)
            with weight_load_strategy.weight_storage(model, allocate_weights):
                rank_model_storage = get_rank_model_storage(model)
                logger.info(
                    f"Use {rank_model_storage / (1024**3):.2f} GB for model weights."
                )
                weights_preloaded = weight_load_strategy.load_weights(
                    model, config)
                self.weight_mapper = weight_load_strategy.weight_mapper
                if weight_load_strategy.context.loads_draft_weights:
                    weight_load_strategy.load_draft_weights(model)
                with timing_metric(
                        ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.
                        value, self._metrics):
                    weight_load_strategy.finalize_weights(
                        model, weights_preloaded)
                    weight_load_strategy.publish_weights(
                        model, weights_preloaded)

            with timing_metric(
                    ModelLoaderMetricNames.POST_LOAD_PROCESSING_SECONDS.value,
                    self._metrics):
                # TODO(GMS-MOE-LB): when the (MoE, GMS) combination is enabled,
                # `register_weight_slots_after_to_cuda` and `finalize_model`
                # must run INSIDE `weight_storage()` (the GMS writer's
                # `mem_pool_scope`) and BEFORE `finalize_write` so MoE
                # allocations become part of the committed layout that RO
                # peers receive. Today they run outside the pool and after
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

        manifest_result = self._dump_final_weight_manifest(
            checkpoint_loader, model, weights_preloaded=weights_preloaded)
        if manifest_result is not None:
            self._metrics[ModelLoaderMetricNames.WEIGHT_MANIFEST_SECONDS.
                          value] = (manifest_result.build_seconds +
                                    manifest_result.write_seconds)

        metrics = ", ".join(f"{name}={value:.4f}"
                            for name, value in self._metrics.items())
        logger.info(
            f"Model loading metrics for rank {self.mapping.rank}: {metrics}")
        if LOW_M_GEMM_ACTIVE:
            prepare_low_m_gemm(model)
        return model, moe_load_balancer

    @staticmethod
    def _construct_model(
        config: ModelConfig
    ) -> tuple[DecoderModelForCausalLM, ModelConfig, bool]:
        """Build the model, on the meta device when the model allows it.

        Meta-device construction allocates no memory for parameters, so
        `_allocate_weights()` can place them later, for example in a GMS pool
        or a virtual-memory scope. A model that fails to build on the meta
        device is built normally instead.

        Returns:
            The model, the config it was built from, and whether it was built
            on the meta device.
        """
        try:
            # config will be modified in-place for some models, like Qwen2
            config_copy = copy.deepcopy(config)
            with MetaInitMode():
                model = AutoModelForCausalLM.from_config(config_copy)
            return model, config_copy, True
        except Exception:
            logger.info(
                f"Fallback to regular model init: {traceback.format_exc(limit=10)}"
            )
            return AutoModelForCausalLM.from_config(config), config, False

    @staticmethod
    def _check_host_resident_weights(
            model: DecoderModelForCausalLM,
            weight_load_strategy: WeightLoadStrategy) -> None:
        """Reject a strategy that cannot fill weights in pinned host memory.

        Raises:
            ValueError: If a module sets ``_requires_standard_hf_loading`` and
                the load strategy does not support it.
        """
        # Pinned-host parameters must be materialized and filled in place.
        # Only the standard checkpoint strategy uses the HF mapper that
        # preserves their stable addresses.
        requires_standard_hf_loading = any(
            getattr(module, "_requires_standard_hf_loading", False)
            for module in model.modules())
        if (requires_standard_hf_loading
                and not weight_load_strategy.supports_host_resident_weights):
            raise ValueError(
                "Host-resident model weights currently require the standard "
                "Hugging Face AUTO loader; DUMMY, VISION_ONLY, GMS, MX, and "
                "Model Streamer loaders cannot materialize them")

    def _allocate_weights(self, model: DecoderModelForCausalLM, *,
                          is_meta_init: bool, use_memory_tag: bool) -> None:
        """Give every parameter and buffer CUDA memory before weights load.

        `WeightLoadStrategy.weight_storage()` calls this, so a strategy
        decides where the memory comes from; the GMS writer calls it inside
        its shared pool.

        1. With ``use_memory_tag`` and a ``model_weights_memory_tag``, buffers
           go to the current virtual-memory scope and parameters to the
           dedicated weights scope, so they can be released and restored
           independently of buffers.
        2. Otherwise, tensors still on the meta device (from meta-device
           construction) get empty CUDA storage.
        3. ``model.to("cuda")`` moves anything still on the CPU.

        Args:
            model: The freshly constructed model.
            is_meta_init: Whether `_construct_model()` built it on the meta
                device.
            use_memory_tag: Whether to honor ``model_weights_memory_tag``.
        """
        load_format = self.llm_args.load_format
        memo: dict[torch.Tensor, torch.Tensor] = {}

        if use_memory_tag and self.model_weights_memory_tag is not None:
            # Allocate buffers to the outer virtual_memory_scope,
            # but parameters (weights) to the dedicated inner virtual_memory_scope.

            def allocate_buffer_on_cuda(t: torch.Tensor):
                if t not in memo:
                    if t.device == torch.device('meta'):
                        cuda_t = torch.empty_like(t, device='cuda')
                    else:
                        cuda_t = t.cuda()
                    memo[t] = cuda_t
                    memo[cuda_t] = cuda_t
                return memo[t]

            _apply_to_buffers_only(model, allocate_buffer_on_cuda)

            need_initialized_weights = load_format not in (LoadFormat.AUTO,
                                                           LoadFormat.DUMMY)

            def allocate_weights_on_cuda(t: torch.Tensor):
                if t not in memo:
                    cuda_t = torch.empty_like(t, device='cuda')
                    if t.device != torch.device('meta') and (
                            need_initialized_weights or is_meta_init):
                        if t.is_cuda:
                            memory_type_map = {
                                ExecutorMemoryType.MODEL_WEIGHTS_MAIN:
                                ExecutorMemoryType.MODEL_ENGINE_MAIN,
                                ExecutorMemoryType.MODEL_WEIGHTS_DRAFT:
                                ExecutorMemoryType.MODEL_ENGINE_DRAFT,
                            }

                            warnings.warn(
                                f"A weight tensor of shape {t.shape} is already allocated on CUDA device before "
                                f"the weight allocation stage. This will cause extra CUDA memory usage in the "
                                f"'{memory_type_map[self.model_weights_memory_tag]}' scope."
                            )
                        cuda_t.copy_(t)
                    memo[t] = cuda_t
                    memo[cuda_t] = cuda_t
                return memo[t]

            with virtual_memory_scope(self.model_weights_memory_tag,
                                      self.model_weights_restore_mode) as pool:
                model._apply(allocate_weights_on_cuda)
            self._weight_pool_proxy = pool
        elif is_meta_init:

            def init_meta_tensor(t: torch.Tensor):
                if t.device != torch.device('meta'):
                    return t

                if t not in memo:
                    memo[t] = torch.empty_like(t, device='cuda')
                return memo[t]

            model._apply(init_meta_tensor)

        # Ensure everything is at least on CUDA
        # No-op if worked as expected
        model.to("cuda")
        memo.clear()

    def _dump_final_weight_manifest(
            self, checkpoint_loader: BaseCheckpointLoader,
            model: DecoderModelForCausalLM, *,
            weights_preloaded: bool) -> Optional[WeightManifestWriteResult]:
        """Write the env-gated final-state weight manifest for this rank.

        This is a no-op unless `MX_WEIGHT_MANIFEST_DIR` is set. It runs after
        every post-load hook and after MoE load-balancer finalization, and
        before engine warmup, so the manifest describes the final post-load
        state that all roles of the MX qualification harness share.
        `reload()` is deliberately not covered: incremental weight updates own
        their own lifecycle.
        """
        return maybe_write_weight_manifest(
            model,
            family="final",
            rank=self.mapping.rank,
            context={
                "boundary": "model_loader_load_end",
                "checkpoint_format": checkpoint_loader.checkpoint_format,
                "weights_preloaded": bool(weights_preloaded),
                "load_format": str(self.llm_args.load_format),
                "tp_rank": self.mapping.tp_rank,
                "pp_rank": self.mapping.pp_rank,
                "world_size": self.mapping.world_size,
                "model_class": type(model).__name__,
            })

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
        manifests are not written here; see `_dump_final_weight_manifest`.

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
                "This can happen when the initial load used a GMS reader or "
                "VISION_ONLY, which bypass the standard weight mapping path.")
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

        This releases the load strategy (for GMS, its client; see
        `GmsLoadStrategy.cleanup()`) and the active checkpoint loader.

        A second call is a no-op because both handles are dropped. Cleanup is
        best effort so shutdown and destructor paths do not propagate backend
        failures.
        """
        if self._weight_load_strategy is not None:
            weight_load_strategy = self._weight_load_strategy
            self._weight_load_strategy = None
            weight_load_strategy.cleanup()
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
                         modeling_v2_target=modeling_v2_resolve(
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
