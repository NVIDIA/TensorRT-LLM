# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import transformers

from .._utils import global_mpi_rank, local_mpi_rank, mpi_rank
# yapf: disable
from ..bindings.executor import (BatchingType, CapacitySchedulerPolicy,
                                 ContextChunkingPolicy, ExecutorConfig,
                                 KvCacheRetentionConfig)
# yapf: enable
from ..logger import logger
from ..models.modeling_utils import QuantAlgo, QuantConfig
from ..models.quant_config_utils import \
    update_quant_config_from_compressed_tensors
from ..quantization.modelopt_config import (is_modelopt_quant_config,
                                            read_modelopt_quant_config,
                                            warn_if_inline_diverges)
# yapf: disable
from .llm_args import (CalibConfig, CudaGraphConfig, DecodeCudaGraphConfig,
                       DraftTargetDecodingConfig, Eagle3DecodingConfig,
                       EagleDecodingConfig, EncodeCudaGraphConfig,
                       KvCacheConfig, LlmArgs, LookaheadDecodingConfig,
                       MedusaDecodingConfig, MTPDecodingConfig,
                       NGramDecodingConfig, SchedulerConfig, TorchLlmArgs,
                       UserProvidedDecodingConfig, _ModelWrapper,
                       _ParallelConfig, update_llm_args_with_extra_dict,
                       update_llm_args_with_extra_options)
# yapf: enable
from .mpi_session import MpiSession
from .tokenizer import TransformersTokenizer, load_hf_tokenizer
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import download_hf_model, print_traceback_on_error


class ModelLoader:
    """The ModelLoader is used to build an end-to-end model for a single-gpu.

    It accepts model name or a local model dir, and will download the model if necessary.
    """

    def __init__(self,
                 llm_args: LlmArgs,
                 workspace: Optional[str | tempfile.TemporaryDirectory] = None,
                 llm_build_stats: Optional["LlmBuildStats"] = None):
        self.llm_args = llm_args
        self._workspace = workspace or tempfile.TemporaryDirectory()
        self.llm_build_stats = llm_build_stats or LlmBuildStats()

        self.model_obj = _ModelWrapper(self.llm_args.model)
        self.speculative_model_obj = _ModelWrapper(
            self.llm_args.speculative_model
        ) if self.llm_args.speculative_model is not None else None

        self.rank = mpi_rank()
        self.global_rank = global_mpi_rank()
        self.mapping = llm_args.parallel_config.to_mapping()

        # For model from hub, the _model_dir is None, and will updated once downloaded
        self._model_dir: Optional[
            Path] = self.model_obj.model_dir if self.model_obj.is_local_model else None

        self._speculative_model_dir: Optional[
            Path] = self.speculative_model_obj.model_dir if self.speculative_model_obj is not None and self.speculative_model_obj.is_local_model else None

    def _apply_modelopt_quant_config(self, hf_quant_config: Dict[str, Any],
                                     explicit_kv_cache_quant_algo) -> None:
        """Apply a normalized modelopt ``quantization`` inner dict onto ``self.llm_args.quant_config``.

        Pops the well-known fields, validates them, then forwards any
        remaining ``QuantConfig`` fields (e.g. AWQ ``has_zero_point`` /
        ``pre_quant_scale``) via setattr.
        """
        quant_config = self.llm_args.quant_config
        hf_quant_algo = hf_quant_config.pop("quant_algo", None)
        if hf_quant_algo is None:
            raise ValueError("Pre-quantized checkpoint must have quant_algo.")
        hf_quant_algo = QuantAlgo(hf_quant_algo)
        if quant_config.quant_algo is None:
            logger.info(
                f"Setting quant_algo={hf_quant_algo} from HF quant config.")
            quant_config.quant_algo = hf_quant_algo
        elif quant_config.quant_algo != hf_quant_algo:
            raise ValueError(
                f"Specified quant_algo={quant_config.quant_algo}, conflicting with quant_algo={hf_quant_algo} from HF quant config."
            )

        hf_kv_cache_quant_algo = hf_quant_config.pop("kv_cache_quant_algo",
                                                     None)
        # modelopt hf_quant_config.json may spell "no KV-cache quantization" as
        # JSON null OR the string "none"/"null" (the Inkling NVFP4 checkpoint
        # uses ``"kv_cache_quant_algo": "none"``); both mean an unquantized KV
        # cache -> None, not QuantAlgo("none") (which is not a member). Mirrors
        # the same normalization in ModelConfig.load_modelopt_quant_config.
        if isinstance(hf_kv_cache_quant_algo, str) and \
                hf_kv_cache_quant_algo.strip().lower() in ("none", "null", ""):
            hf_kv_cache_quant_algo = None
        if hf_kv_cache_quant_algo is not None:
            hf_kv_cache_quant_algo = QuantAlgo(hf_kv_cache_quant_algo)
            if explicit_kv_cache_quant_algo is not None:
                if explicit_kv_cache_quant_algo != hf_kv_cache_quant_algo:
                    logger.warning(
                        f"Overriding checkpoint kv_cache_quant_algo={hf_kv_cache_quant_algo} with explicit kv_cache_config.dtype={explicit_kv_cache_quant_algo}."
                    )
                quant_config.kv_cache_quant_algo = explicit_kv_cache_quant_algo
            elif quant_config.kv_cache_quant_algo is None:
                logger.info(
                    f"Setting kv_cache_quant_algo={hf_kv_cache_quant_algo} from HF quant config."
                )
                quant_config.kv_cache_quant_algo = hf_kv_cache_quant_algo
            elif quant_config.kv_cache_quant_algo != hf_kv_cache_quant_algo:
                raise ValueError(
                    f"Specified kv_cache_quant_algo={quant_config.kv_cache_quant_algo}, conflicting with kv_cache_quant_algo={hf_kv_cache_quant_algo} from HF quant config."
                )
        else:
            if quant_config.kv_cache_quant_algo not in [
                    None, QuantAlgo.FP8, QuantAlgo.NVFP4
            ]:
                raise ValueError(
                    f"Only kv_cache_quant_algo={QuantAlgo.FP8} or {QuantAlgo.NVFP4} is allowed for pre-quantized checkpoint, got {quant_config.kv_cache_quant_algo}."
                )

        # quantized_layers is handled separately (e.g. via LayerQuantConfig
        # in PretrainedConfig for TRT, or _torch/model_config.py for PyTorch)
        hf_quant_config.pop("quantized_layers", None)

        quant_config_fields = set(quant_config.model_fields.keys())
        for key, value in hf_quant_config.items():
            if key not in quant_config_fields:
                logger.warning(
                    f"Ignoring unknown field '{key}' from HF quant config (not a QuantConfig field)."
                )
                continue
            logger.info(
                f"Setting {key}={str(value)[:100]}{'...' if len(str(value)) > 100 else ''} from HF quant config."
            )
            setattr(quant_config, key, value)
        self.llm_args.quant_config = quant_config

    def _update_from_hf_quant_config(self) -> bool:
        """Update quant_config from the config file of pre-quantized HF checkpoint.

        Returns:
            prequantized (bool): Whether the checkpoint is pre-quantized.
        """
        kv_cache_dtype = self.llm_args.kv_cache_config.dtype
        explicit_kv_cache_quant_algo = {
            "fp8": QuantAlgo.FP8,
            "nvfp4": QuantAlgo.NVFP4,
        }.get(kv_cache_dtype)
        requires_global_quant_config_fallback = False

        hf_quant_config_path = f"{self._model_dir}/hf_quant_config.json"
        if os.path.exists(hf_quant_config_path):
            logger.info(
                f"Found {hf_quant_config_path}, pre-quantized checkpoint is used."
            )
            with open(hf_quant_config_path, "r") as f:
                normalized = read_modelopt_quant_config(json.load(f))
            # Cross-check against inline config.json.quantization_config if any.
            # Done before _apply_modelopt_quant_config since the apply step
            # mutates ``normalized`` via ``.pop()``.
            try:
                with open(f"{self._model_dir}/config.json", "r") as f:
                    warn_if_inline_diverges(
                        normalized,
                        json.load(f).get("quantization_config"),
                        source_file="hf_quant_config.json",
                    )
            except FileNotFoundError:
                pass
            if normalized.get("quant_algo") is None:
                if normalized.get("quantized_layers") is not None:
                    requires_global_quant_config_fallback = True
                    logger.info(
                        "hf_quant_config.json does not set a global quant_algo; "
                        "falling back to config.json or model_kwargs for global "
                        "quantization.")
                else:
                    raise ValueError(
                        "Pre-quantized checkpoint must have quant_algo.")
            else:
                self._apply_modelopt_quant_config(normalized,
                                                  explicit_kv_cache_quant_algo)
                return True

        hf_config_path = f"{self._model_dir}/config.json"
        hf_quant_config = None
        if os.path.exists(hf_config_path):
            with open(hf_config_path, "r") as f:
                hf_config = json.load(f)
                hf_quant_config = hf_config.get("quantization_config", None)
                if hf_quant_config is not None:
                    logger.info(
                        f"Found quantization_config field in {hf_config_path}, pre-quantized checkpoint is used."
                    )
        if self.llm_args.model_kwargs is not None and "quantization_config" in self.llm_args.model_kwargs:
            logger.info(
                f"Update hf_quant_config from model_kwargs: quantization_config={self.llm_args.model_kwargs['quantization_config']} (previous value: {hf_quant_config})"
            )
            hf_quant_config = self.llm_args.model_kwargs["quantization_config"]
        elif hf_quant_config is not None:
            logger.info(
                f"Use quantization_config from {hf_config_path}: quantization_config={hf_quant_config}"
            )

        if requires_global_quant_config_fallback and hf_quant_config is None:
            raise ValueError(
                "hf_quant_config.json does not set a global quant_algo and no "
                "quantization_config fallback was found.")

        if hf_quant_config is not None:
            if is_modelopt_quant_config(hf_quant_config):
                self._apply_modelopt_quant_config(
                    read_modelopt_quant_config(hf_quant_config),
                    explicit_kv_cache_quant_algo)
                return True
            quant_config = self.llm_args.quant_config
            # DeepSeek V3 FP8 ckpt
            if hf_quant_config.get("quant_method") == "fp8":
                if hf_quant_config.get("weight_block_size") is not None:
                    quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                    quant_config.group_size = hf_quant_config[
                        "weight_block_size"][0]
                    quant_config.exclude_modules = ["*eh_proj"]
                else:
                    # Ministral 3 static quant
                    quant_config.quant_algo = QuantAlgo.FP8
            elif hf_quant_config.get("quant_method") == "mxfp4":
                from .._torch.model_config import ModelConfig
                quant_config.quant_algo = ModelConfig.get_mxfp4_quant_algo(
                    self.llm_args.moe_config.backend)
                quant_config.group_size = 32
                quant_config.exclude_modules = [
                    'block.*.attn.out', 'block.*.mlp.gate', 'block.*.attn.qkv',
                    'embedding', 'unembedding'
                ]
            # MXFP8 checkpoints (e4m3 weights + UE8M0 1x32 block scales, dynamic
            # MXFP8 acts).
            elif hf_quant_config.get("quant_method") == "mxfp8":
                quant_config.quant_algo = QuantAlgo.MXFP8
                block_size = hf_quant_config.get("weight_block_size", [1, 32])
                # MXFP8 uses 1x32 blocks along the K dim; group_size is the K
                # block (32).
                assert tuple(block_size) == (1, 32), (
                    f"MXFP8 only supports weight_block_size=[1,32], got {block_size}"
                )
                quant_config.group_size = block_size[1]

                # Layers the producer left in BF16.
                ignored = hf_quant_config.get("ignored_layers", [])
                hf_exclude_modules = hf_quant_config.get(
                    'modules_to_not_convert', None)
                if hf_exclude_modules is not None:
                    quant_config.exclude_modules = list(
                        dict.fromkeys(hf_exclude_modules + ignored))
                else:
                    quant_config.exclude_modules = list(ignored)
            # NOTE: This is for llm-compressor's quantized checkpoints.
            elif hf_quant_config.get("quant_method") == "compressed-tensors":
                update_quant_config_from_compressed_tensors(
                    quant_config, hf_quant_config)
            elif hf_quant_config.get("quant_method") == "nvfp4":
                quant_config.quant_algo = QuantAlgo.NVFP4
                group_size = hf_quant_config.get("group_size", 16)
                assert group_size == 16, "NVFP4 only supports group_size=16"
                quant_config.group_size = group_size
                default_exclude = ['*.mlp.gate', 'lm_head']

                hf_exclude_modules = hf_quant_config.get(
                    'modules_to_not_convert', None)
                if hf_exclude_modules is not None:
                    quant_config.exclude_modules = list(
                        dict.fromkeys(hf_exclude_modules + default_exclude))
                else:
                    quant_config.exclude_modules = default_exclude
            elif hf_quant_config.get("quant_method") is None:
                # quantization_config present but quant_method is null → not
                # pre-quantized; tell caller so calibration is not skipped.
                return False
            else:
                raise NotImplementedError(
                    f"Unsupported quantization_config: {hf_quant_config}.")

            return True

        return False

    @staticmethod
    def load_hf_tokenizer(model_dir,
                          trust_remote_code: bool = True,
                          use_fast: bool = True,
                          **kwargs) -> Optional[TransformersTokenizer]:
        if (tokenizer := load_hf_tokenizer(model_dir, trust_remote_code,
                                           use_fast, **kwargs)) is not None:
            return tokenizer
        else:
            logger.warning(f"Failed to load tokenizer from {model_dir}")
            return None

    @staticmethod
    def load_hf_generation_config(
            model_dir, **kwargs) -> Optional[transformers.GenerationConfig]:
        try:
            return transformers.GenerationConfig.from_pretrained(
                model_dir, **kwargs)
        except Exception as e:
            logger.warning(
                f"Failed to load hf generation config from {model_dir}, encountered error: {e}"
            )
            return None

    @staticmethod
    def load_hf_model_config(
            model_dir,
            trust_remote_code: bool = True,
            **kwargs) -> Optional[transformers.PretrainedConfig]:
        try:
            # Route via load_pretrained_config so model_types registered in
            # TRT-LLM's _CONFIG_REGISTRY (e.g. deepseek_v32 / kimi_k2 /
            # glm_moe_dsa) are dispatched to their TRT-LLM-local config class
            # and get the same compat handling as the engine's own config load
            # (e.g. dropping GLM-MoE-DSA's unsupported layer_types); it falls
            # back to AutoConfig for everything else. Calling AutoConfig /
            # PretrainedConfig.from_pretrained directly here would instead hit
            # transformers' validate_layer_type and return None.
            from tensorrt_llm._torch.pyexecutor.config_utils import \
                load_pretrained_config
            return load_pretrained_config(model_dir,
                                          trust_remote_code=trust_remote_code,
                                          **kwargs)
        except Exception as e:
            logger.warning(
                f"Failed to load hf model config from {model_dir}, encountered error: {e}"
            )
            return None


class CachedModelLoader:
    """The CachedModelLoader is used to build the model in both single or multi-gpu, with optional caching.
    """

    def __init__(
        self,
        llm_args: LlmArgs,
        llm_build_stats: weakref.ReferenceType["LlmBuildStats"],
        mpi_session: Optional[MpiSession] = None,
        workspace: Optional[str] = None,
    ):
        self.llm_args = llm_args
        self.mpi_session = mpi_session
        self._workspace = workspace or tempfile.TemporaryDirectory()
        self.llm_build_stats = llm_build_stats

        # This is used for build cache. To compute the cache key, a local HF model is required, it could be download
        # from HF model hub, so this helps to hold the path.
        self._hf_model_dir: Optional[Path] = None

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name) if isinstance(
            self._workspace, tempfile.TemporaryDirectory) else Path(
                self._workspace)

    def _submit_to_all_workers(
        self,
        task: Callable[..., Any],
        *args,
        **kwargs,
    ) -> List[Any]:
        if self.llm_args.parallel_config.is_multi_gpu:
            return self.mpi_session.submit_sync(task, *args, **kwargs)
        else:
            return [task(*args, **kwargs)]

    def _download_hf_model_if_needed(self,
                                     model_obj: _ModelWrapper,
                                     revision: Optional[str] = None) -> Path:
        """Download a model from HF hub if needed.

        Also updates the model_obj.model_dir with the local model dir.
        """
        if model_obj.is_hub_model:
            model_dirs = self._submit_to_all_workers(
                CachedModelLoader._node_download_hf_model,
                model=model_obj.model_name,
                revision=revision)
            model_dir = next((d for d in model_dirs if d is not None), None)
            model_obj.model_dir = model_dir
            return model_dir
        return model_obj.model_dir

    def __call__(self) -> Tuple[Path, Union[Path, None]]:

        # Download speculative model from HuggingFace if needed (all backends)
        if (self.llm_args.speculative_config is not None and
                self.llm_args.speculative_config.speculative_model is not None):
            spec_model_obj = _ModelWrapper(
                self.llm_args.speculative_config.speculative_model)
            spec_model_dir = self._download_hf_model_if_needed(spec_model_obj)
            self.llm_args.speculative_config.speculative_model = spec_model_dir

        # AutoDeploy doesn't use ModelLoader
        if self.llm_args.backend == "_autodeploy":
            return None, ""

        self._hf_model_dir = None
        self.model_loader = ModelLoader(self.llm_args)

        if self.llm_args.backend not in ["pytorch", "_autodeploy"]:
            raise ValueError(
                f'backend {self.llm_args.backend} is not supported.')

        self._hf_model_dir = self._download_hf_model_if_needed(
            self.model_loader.model_obj, revision=self.llm_args.revision)

        if self.llm_args.quant_config.quant_algo is not None:
            logger.warning(
                "QuantConfig for pytorch backend is ignored. You can load "
                "quantized model with hf_quant_config.json directly.")
        # Currently, this is to make updated quant_config visible by llm.args.quant_config
        # TODO: Unify the logics with those in tensorrt_llm/_torch/model_config.py
        self.model_loader._update_from_hf_quant_config()

        return None, self._hf_model_dir

    @print_traceback_on_error
    @staticmethod
    def _node_download_hf_model(
        model: str,
        revision: Optional[str] = None,
    ) -> Optional[Path]:
        if local_mpi_rank() == 0:
            return download_hf_model(model, revision)
        else:
            return None


@dataclass
class LlmBuildStats:
    """LlmBuildStats is the statistics for the LLM model building."""
    # Whether the cache is hit for the engine
    cache_hitted: bool = False
    cache_info: Optional[str] = None

    model_from_hf_hub: bool = False

    local_model_dir: Optional[Path] = None

    # The path to the trt-llm engine
    engine_dir: Optional[Path] = None

    # The build steps information, including the step name and the latency in seconds.
    build_steps_info: List[Tuple[str, float]] = field(default_factory=list)


__all__ = [
    'LlmArgs',
    'LlmBuildStats',
    'ModelLoader',
    '_ParallelConfig',
    '_ModelWrapper',
    'BatchingType',
    'ExecutorConfig',
    'SchedulerConfig',
    'KvCacheRetentionConfig',
    'LookaheadDecodingConfig',
    'MedusaDecodingConfig',
    'MTPDecodingConfig',
    'NGramDecodingConfig',
    'DraftTargetDecodingConfig',
    'UserProvidedDecodingConfig',
    'ContextChunkingPolicy',
    'CapacitySchedulerPolicy',
    'QuantConfig',
    'CalibConfig',
    'CudaGraphConfig',
    'DecodeCudaGraphConfig',
    'EncodeCudaGraphConfig',
    'KvCacheConfig',
    'CachedModelLoader',
    'EagleDecodingConfig',
    'Eagle3DecodingConfig',
    'update_llm_args_with_extra_dict',
    'update_llm_args_with_extra_options',
    'apply_model_defaults_to_llm_args',
]


def _deep_merge(base: Dict[str, Any], *overlays: Dict[str,
                                                      Any]) -> Dict[str, Any]:
    """Deep merge multiple dictionaries with right-side precedence."""
    result = base.copy()

    for overlay in overlays:
        if not overlay:
            continue

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(
                    value, dict):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value

    return result


def apply_model_defaults_to_llm_args(
        llm_args: 'TorchLlmArgs',
        model_defaults_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply model defaults to a Pydantic LlmArgs instance.

    Returns the defaults that were actually applied.
    """
    if not model_defaults_dict:
        return {}

    # Key-presence check: any value form (dict, pydantic object, None) is
    # rejected — the deep-merge could materialize a transceiver config in
    # aggregated mode or silently enable a disabled one.
    if "cache_transceiver_config" in model_defaults_dict:
        raise ValueError(
            "Model defaults must not contain 'cache_transceiver_config': the "
            "deep-merge could materialize or silently enable a transceiver "
            "config the user did not turn on. Declare a runtime preference "
            "via get_preferred_transceiver_runtime() instead.")

    user_overrides = llm_args.model_dump(exclude_unset=True)
    base_state = llm_args.model_dump()
    merged_state = _deep_merge(base_state, model_defaults_dict, user_overrides)

    new_args = llm_args.__class__(**merged_state)

    for field_name in llm_args.model_fields:
        setattr(llm_args, field_name, getattr(new_args, field_name))

    def _compute_applied(defaults: Dict[str, Any],
                         overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively compute applied defaults."""
        applied = {}
        for key, default_value in defaults.items():
            if isinstance(default_value, dict):
                user_override = overrides.get(key, {})
                if isinstance(user_override, dict):
                    nested_applied = _compute_applied(default_value,
                                                      user_override)
                    if nested_applied:
                        applied[key] = nested_applied
                elif key not in overrides:
                    applied[key] = default_value
            else:
                if key not in overrides:
                    applied[key] = default_value
        return applied

    return _compute_applied(model_defaults_dict, user_overrides)


def _resolve_kv_cache_manager_v2_auto(
        llm_args: 'TorchLlmArgs', model_defaults_dict: Dict[str, Any]) -> bool:
    """Resolve the KV cache manager auto setting after model defaults are applied."""
    setting = llm_args.kv_cache_config.use_kv_cache_manager_v2
    if setting != "auto":
        return setting

    kv_cache_defaults = model_defaults_dict.get("kv_cache_config", {})
    model_default = (kv_cache_defaults.get("use_kv_cache_manager_v2", False)
                     if isinstance(kv_cache_defaults, dict) else False)
    if model_default == "auto":
        model_default = False
    if not isinstance(model_default, bool):
        raise ValueError(
            "Model default kv_cache_config.use_kv_cache_manager_v2 must be "
            f"True, False, or 'auto', got {model_default!r}.")

    llm_args.kv_cache_config.use_kv_cache_manager_v2 = model_default
    return model_default


def _resolve_transceiver_runtime_auto(llm_args: 'TorchLlmArgs',
                                      model_cls: Optional[type] = None,
                                      pretrained_config: Any = None) -> None:
    """Resolve the 'auto' sentinel in cache_transceiver_config.transceiver_runtime.

    Semantics:
    - Disagg disabled (config is None or backend is None): no-op. The model
      preference must never materialize or alter a transceiver config that the
      user did not enable.
    - Explicit user value ('CPP'/'PYTHON'/None): left untouched.
    - 'auto': adopt ``model_cls.get_preferred_transceiver_runtime()`` when the
      effective backend supports it (the Python transceiver requires NIXL);
      otherwise fall back to None (C++ transceiver).

    ``pretrained_config`` is forwarded to the hook so implementation classes
    shared by several architectures can differentiate per checkpoint.
    """
    cfg = llm_args.cache_transceiver_config
    if cfg is None or cfg.backend is None:
        return
    if cfg.transceiver_runtime != "auto":
        return

    preferred = None
    if model_cls is not None:
        get_preferred = getattr(model_cls, 'get_preferred_transceiver_runtime',
                                None)
        if get_preferred is not None:
            preferred = get_preferred(pretrained_config)
    if preferred not in (None, "CPP", "PYTHON"):
        raise ValueError(
            f"{model_cls.__name__}.get_preferred_transceiver_runtime() must "
            f"return 'CPP', 'PYTHON', or None, got {preferred!r}.")

    effective_backend, _ = cfg._resolve_default_backend()
    if preferred == "PYTHON" and effective_backend != "NIXL":
        logger.info(
            f"Model prefers the Python transceiver, but backend "
            f"{effective_backend} does not support it; falling back to the "
            f"C++ transceiver.")
        preferred = None

    cfg.transceiver_runtime = preferred
    logger.info(
        f"Resolved transceiver_runtime='auto' to {preferred!r} for "
        f"{model_cls.__name__ if model_cls is not None else 'unknown model'}.")
