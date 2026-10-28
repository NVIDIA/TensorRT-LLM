# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import transformers

from .._utils import global_mpi_rank, local_mpi_rank, mpi_rank
# yapf: disable
from ..bindings.executor import (BatchingType, CapacitySchedulerPolicy,
                                 ContextChunkingPolicy, KvCacheRetentionConfig)
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
                       DFlashDecodingConfig, DraftTargetDecodingConfig,
                       Eagle3DecodingConfig, EagleDecodingConfig,
                       EncodeCudaGraphConfig, KvCacheConfig, KVEventsConfig,
                       LlmArgs, MTPDecodingConfig, NGramDecodingConfig,
                       SchedulerConfig, TorchLlmArgs,
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

    def __init__(self, llm_args: LlmArgs):
        self.llm_args = llm_args

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
            "fp8_ds_mla": QuantAlgo.FP8,
            "nvfp4": QuantAlgo.NVFP4,
            "mixed_nvfp4_fp8": QuantAlgo.FP8,
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
    def load_hf_generation_config_dict(
            model_dir: Union[str, Path]) -> Dict[str, Any]:
        """Load only values explicitly present in generation_config.json."""
        generation_config_path = Path(model_dir) / "generation_config.json"
        try:
            with open(generation_config_path, "r") as config_file:
                generation_config = json.load(config_file)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"Failed to load generation config values from {generation_config_path}, encountered error: {e}"
            )
            return {}

        if not isinstance(generation_config, dict):
            logger.warning(
                f"Ignoring generation config from {generation_config_path}: expected a JSON object."
            )
            return {}
        return generation_config

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

    def __init__(self,
                 llm_args: LlmArgs,
                 mpi_session: Optional[MpiSession] = None,
                 is_attached_frontend: bool = False):
        self.llm_args = llm_args
        self.mpi_session = mpi_session
        self._is_attached_frontend = is_attached_frontend
        self._hf_model_dir: Optional[Path] = None

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
            if self._is_attached_frontend:
                # Attached frontends have no MPI session; resolve the cached
                # snapshot locally, independent of their CUDA device index.
                model_dir = download_hf_model(model_obj.model_name, revision)
            else:
                model_dirs = self._submit_to_all_workers(
                    CachedModelLoader._node_download_hf_model,
                    model=model_obj.model_name,
                    revision=revision)
                model_dir = next((d for d in model_dirs if d is not None), None)
            model_obj.model_dir = model_dir
            return model_dir
        return model_obj.model_dir

    def __call__(self) -> Optional[Path]:

        # Download speculative model from HuggingFace if needed (all backends)
        if (self.llm_args.speculative_config is not None and
                self.llm_args.speculative_config.speculative_model is not None):
            spec_model_obj = _ModelWrapper(
                self.llm_args.speculative_config.speculative_model)
            was_hub_model = spec_model_obj.is_hub_model
            spec_model_dir = self._download_hf_model_if_needed(spec_model_obj)
            self.llm_args.speculative_config.speculative_model = spec_model_dir
            # Some speculative configs read defaults out of the draft
            # checkpoint's config.json (e.g. DFlash's target_layer_ids). A hub
            # repo id had no readable config.json at validation time; a local
            # path was already resolved back then.
            if was_hub_model:
                resolve_from_checkpoint = getattr(
                    self.llm_args.speculative_config, 'resolve_from_checkpoint',
                    None)
                if resolve_from_checkpoint is not None:
                    resolve_from_checkpoint()
                # The DFlash buffer-fit check reads the drafter geometry from
                # the same config.json, so for a hub repo id the validation
                # pass only covered the token budget. Re-run it now that the
                # checkpoint is local; a local path was fully checked at
                # validation time and is not re-run.
                if (isinstance(self.llm_args, TorchLlmArgs)
                        and isinstance(self.llm_args.speculative_config,
                                       DFlashDecodingConfig)):
                    self.llm_args._validate_dflash_ctx_budget()

        self._hf_model_dir = None
        self.model_loader = ModelLoader(self.llm_args)

        if self.llm_args.backend != "pytorch":
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

        # Issue #11083 — context-aware defaults (two-stage loading, stage 1):
        # read the on-disk config.json (pure Python, no GPU), resolve the
        # model class, and apply model-specific default overrides to llm_args
        # **before** weight loading. This surfaces model-aware defaults early
        # enough for LLM construction, telemetry, and user inspection of
        # llm.args. The same resolution runs a second time inside
        # ModelLoader.load_config_and_apply_defaults during executor creation,
        # but that richer pass is GPU-dependent.
        if self.llm_args.backend == "pytorch":
            apply_early_model_defaults(self._hf_model_dir, self.llm_args)

        return self._hf_model_dir

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


__all__ = [
    'LlmArgs',
    'ModelLoader',
    '_ParallelConfig',
    '_ModelWrapper',
    'BatchingType',
    'SchedulerConfig',
    'KvCacheRetentionConfig',
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
    'KVEventsConfig',
    'CachedModelLoader',
    'EagleDecodingConfig',
    'Eagle3DecodingConfig',
    'update_llm_args_with_extra_dict',
    'update_llm_args_with_extra_options',
    'apply_model_defaults_to_llm_args',
    'apply_early_model_defaults',
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


def _capture_pydantic_fields_set(model: Any) -> Dict[int, set]:
    """Recursively snapshot ``__pydantic_fields_set__`` for a Pydantic model tree.

    ``__pydantic_fields_set__`` is a live mutable ``set`` that Pydantic mutates
    in place on every ``setattr``.  Capturing with ``set(...)`` keeps a stable
    snapshot; a bare reference would be silently mutated by the subsequent
    field assignments in :func:`apply_model_defaults_to_llm_args`.

    Returns a mapping ``{id(model): set_copy}`` consumable by
    :func:`_restore_pydantic_fields_set`.
    """
    from pydantic import BaseModel
    snapshots: Dict[int, set] = {}

    def _walk(node: Any) -> None:
        if not isinstance(node, BaseModel):
            return
        snapshots[id(node)] = set(node.__pydantic_fields_set__)
        for child in node.__dict__.values():
            _walk(child)

    _walk(model)
    return snapshots


def _restore_pydantic_fields_set(model: Any,
                                 snapshots: Dict[int, set]) -> None:
    """Restore ``__pydantic_fields_set__`` from a :func:`_capture_pydantic_fields_set` snapshot.

    Operates **in-place** on the existing live set via ``.clear()`` +
    ``.update()``.  Reassigning a fresh set would break Pydantic's internal
    handler — later ``setattr`` calls rely on ``__pydantic_fields_set__``
    supporting ``.add``.

    Relies on :func:`apply_model_defaults_to_llm_args` preserving object
    identity for nested Pydantic models, so ``id(node)`` matches between
    capture and restore.
    """
    from pydantic import BaseModel

    def _walk(node: Any) -> None:
        if not isinstance(node, BaseModel):
            return
        if id(node) in snapshots:
            node.__pydantic_fields_set__.clear()
            node.__pydantic_fields_set__.update(snapshots[id(node)])
        for child in node.__dict__.values():
            _walk(child)

    _walk(model)


def _deep_copy_values_to(target: Any, source: Any) -> None:
    """Copy field values from *source* to *target*, preserving *target*'s identity.

    When both sides are Pydantic ``BaseModel`` instances, each field is set
    individually so nested model objects retain their ``id`` — this is
    essential for :func:`_restore_pydantic_fields_set` to match snapshots.
    """
    from pydantic import BaseModel
    if isinstance(target, BaseModel) and isinstance(source, BaseModel):
        for field_name in target.model_fields:
            src_val = getattr(source, field_name)
            dst_val = getattr(target, field_name)
            if isinstance(src_val, BaseModel) and isinstance(
                    dst_val, BaseModel):
                _deep_copy_values_to(dst_val, src_val)
            else:
                setattr(target, field_name, src_val)


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

    # Snapshot the original "explicitly set" field state before any in-place
    # mutation.  Naive setattr below marks every touched field as "user-set"
    # from Pydantic's perspective, which would break downstream
    # ``model_dump(exclude_unset=True)`` calls and block later defaults
    # passes from overriding values the user never explicitly provided.
    # We capture recursively (including nested Pydantic models) and restore
    # after mutation — but only because ``_deep_copy_values_to`` preserves
    # object identity for nested models.
    original_fields_set = _capture_pydantic_fields_set(llm_args)

    new_args = llm_args.__class__(**merged_state)

    # Copy merged values back into llm_args.  When both sides hold a
    # BaseModel (e.g. a user-set nested config), do a deep field-by-field
    # copy so the target object retains its identity — otherwise the
    # recursive fields_set snapshot would be orphaned and couldn't be
    # restored.
    from pydantic import BaseModel
    for field_name in type(llm_args).model_fields:
        src_val = getattr(new_args, field_name)
        dst_val = getattr(llm_args, field_name)
        if isinstance(src_val, BaseModel) and isinstance(dst_val, BaseModel):
            _deep_copy_values_to(dst_val, src_val)
        else:
            setattr(llm_args, field_name, src_val)

    _restore_pydantic_fields_set(llm_args, original_fields_set)

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


def apply_early_model_defaults(model_dir: Union[str, Path],
                               llm_args: 'TorchLlmArgs') -> Dict[str, Any]:
    """Read config.json and apply model-specific LLM API defaults before weight loading.

    This is the two-stage loading step described in Issue #11083: we read only
    the on-disk ``config.json`` (a pure-Python, no-GPU operation), identify the
    model architecture, and ask the resolved model class for its
    :meth:`get_model_defaults` overrides. The overrides are merged into
    ``llm_args`` without touching any field the user explicitly set.

    The same resolution happens a second time inside
    :meth:`ModelLoader.load_config_and_apply_defaults` during executor creation.
    That second pass operates on a richer ``ModelConfig`` produced by the
    checkpoint loader (spec config, quant config, etc.) and also resolves
    ``"auto"`` sentinel values. This helper is intentionally lean: it does
    **not** replicate any of that. Its job is merely to surface model-aware
    defaults early enough that :class:`LLM` construction, telemetry, and any
    user inspection of ``llm.args`` see the same defaults the executor will.

    Safe to call with no GPU. Returns the applied-defaults dict (empty if no
    model class matched or no defaults were produced). Never raises for
    missing/unknown config — falls back silently so a malformed or unsupported
    checkpoint cannot crash LLM construction.
    """
    if not isinstance(llm_args, TorchLlmArgs):
        return {}

    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        logger.debug(
            f"apply_early_model_defaults: no config.json at {config_path}; "
            "skipping early defaults.")
        return {}

    # Pure-Python config read — no transformers, no torch, no GPU required.
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            hf_config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(
            f"apply_early_model_defaults: failed to read {config_path}: {exc}; "
            "skipping early defaults.")
        return {}

    # Validate parsed config is a JSON object (not e.g. a bare array).
    if not isinstance(hf_config, dict):
        logger.debug(
            f"apply_early_model_defaults: config.json did not parse as a "
            f"JSON object (got {type(hf_config).__name__}); "
            "skipping early defaults.")
        return {}

    # Validate architectures is a nonempty list of strings.
    architectures = hf_config.get("architectures")
    if not (isinstance(architectures, list) and architectures
            and all(isinstance(a, str) for a in architectures)):
        model_type = hf_config.get("model_type") or "(unknown)"
        logger.debug(
            f"apply_early_model_defaults: config.json has no valid "
            f"'architectures' list (model_type={model_type}); "
            "cannot resolve a model class for early defaults.")
        return {}

    # Lazy import: the model registry lives under _torch.models and is only
    # populated on demand. Importing it here does not pull in CUDA.
    try:
        from tensorrt_llm._torch.models.modeling_utils import \
            get_registered_model_class
    except Exception as exc:
        logger.debug(
            f"apply_early_model_defaults: cannot import model registry: {exc}; "
            "skipping early defaults.")
        return {}

    model_cls = get_registered_model_class(architectures[0])
    if model_cls is None:
        logger.debug(
            f"apply_early_model_defaults: no registered class for "
            f"architecture {architectures[0]!r}; skipping early defaults.")
        return {}

    if not hasattr(model_cls, "get_model_defaults"):
        return {}

    try:
        model_defaults = model_cls.get_model_defaults(llm_args) or {}
    except Exception as exc:
        logger.debug(
            f"apply_early_model_defaults: {model_cls.__name__}.get_model_defaults "
            f"raised {exc}; skipping early defaults.")
        return {}

    if not model_defaults:
        return {}

    try:
        applied = apply_model_defaults_to_llm_args(llm_args, model_defaults)
    except Exception as exc:
        logger.debug(
            f"apply_early_model_defaults: applying defaults for "
            f"{model_cls.__name__} raised {exc}; skipping early defaults.")
        return {}

    if applied:
        logger.info(
            f"[early] Applied model defaults for {model_cls.__name__}: "
            f"{applied}")
    return applied


def _resolve_kv_cache_manager_v2_auto(llm_args: 'TorchLlmArgs',
                                      model_cls: Optional[type] = None,
                                      pretrained_config: Any = None) -> bool:
    """Resolve the KV cache manager auto setting from the model preference.

    A model preference for V2 is demoted to V1 for routes V2 cannot serve; a
    boolean value otherwise wins, including the default ``True``.
    Disaggregated serving with V2 requires the Python transceiver with NIXL,
    so ``auto`` falls back to V1 for any other route. The transceiver runtime
    auto setting must be resolved first.

    The fallback only reaches models whose manager class is selected by
    ``use_kv_cache_manager_v2``. Models routed to a V2 manager unconditionally
    -- sparse attention picks its class from the algorithm alone -- keep a V2
    manager after the demotion, so the arm does not protect them.

    A model whose ``requires_kv_cache_manager_v2`` hook returns True has no
    supported V1 route: the disaggregated-serving arm raises instead of
    demoting, and an explicit ``use_kv_cache_manager_v2=False`` is rejected.
    """
    model_name = getattr(model_cls, "__name__", "the model")
    requires_v2 = False
    if model_cls is not None:
        get_required = getattr(model_cls, 'requires_kv_cache_manager_v2', None)
        if get_required is not None:
            requires_v2 = bool(get_required(pretrained_config))

    setting = llm_args.kv_cache_config.use_kv_cache_manager_v2
    if setting != "auto":
        if requires_v2 and setting is False:
            raise ValueError(
                f"{model_name} requires KV cache manager v2; "
                "kv_cache_config.use_kv_cache_manager_v2=False is not supported "
                "for this model.")
        return setting

    preferred_version = None
    if model_cls is not None:
        get_preferred = getattr(model_cls,
                                'get_preferred_kv_cache_manager_version', None)
        if get_preferred is not None:
            preferred_version = get_preferred(pretrained_config)
    if preferred_version not in (None, "V1", "V2"):
        raise ValueError(
            f"{model_cls.__name__}.get_preferred_kv_cache_manager_version() "
            f"must return 'V1', 'V2', or None, got {preferred_version!r}.")

    use_v2 = preferred_version == "V2" or requires_v2

    transceiver_config = llm_args.cache_transceiver_config
    if (use_v2 and transceiver_config is not None
            and transceiver_config.backend is not None):
        effective_backend, _ = transceiver_config._resolve_default_backend()
        runtime = transceiver_config.transceiver_runtime
        if effective_backend != "NIXL" or runtime != "PYTHON":
            if requires_v2:
                raise ValueError(
                    f"{model_name} requires KV cache manager v2, which "
                    "disaggregated serving keeps only on backend='NIXL' with "
                    "transceiver_runtime='PYTHON'; this run uses "
                    f"transceiver_runtime={runtime!r} with "
                    f"backend={effective_backend!r}. Configure the cache "
                    "transceiver for NIXL with the Python runtime.")
            logger.info(
                "KV cache manager V2 is the model preference, but disaggregated "
                "serving uses transceiver_runtime=%r with backend=%r; "
                "falling back to V1.", runtime, effective_backend)
            use_v2 = False

    if (not use_v2 and llm_args.kv_cache_config.block_reuse_config.
            swa_endpoint_rewind_tokens > 0):
        raise ValueError(
            "block_reuse_config.swa_endpoint_rewind_tokens requires KV cache "
            "manager v2, but use_kv_cache_manager_v2='auto' resolved to V1. "
            "Set kv_cache_config.use_kv_cache_manager_v2=True explicitly.")
    if (not use_v2 and
            llm_args.kv_cache_config.block_reuse_config.enable_branch_snapshot):
        raise ValueError(
            "block_reuse_config.enable_branch_snapshot requires KV cache "
            "manager v2, but use_kv_cache_manager_v2='auto' resolved to V1. "
            "Set kv_cache_config.use_kv_cache_manager_v2=True explicitly.")
    llm_args.kv_cache_config.use_kv_cache_manager_v2 = use_v2
    return use_v2


def _transceiver_python_fallback_reason(
        llm_args: 'TorchLlmArgs') -> Optional[str]:
    """Why 'auto' should not default to the Python transceiver, or None.

    Checks only the constraints that are decidable from
    ``cache_transceiver_config`` itself (backend and timeout). Any other
    incompatibility (e.g. non-helix context parallelism) is NOT resolved
    here and fails loudly at transceiver creation instead. Only consulted
    when the model expressed no runtime preference: a model preferring
    'PYTHON' may genuinely require it (e.g. recurrent-state transfer), so it
    is never silently rerouted — transceiver creation raises with an
    actionable message instead.
    """
    cfg = llm_args.cache_transceiver_config
    effective_backend, _ = cfg._resolve_default_backend()
    if effective_backend != "NIXL":
        return (f"backend {effective_backend!r} (the Python transceiver "
                "requires NIXL)")
    if cfg.kv_transfer_timeout_ms is None:
        return ("kv_transfer_timeout_ms=None (the Python transceiver "
                "requires a finite timeout)")
    # Deliberately reads only cache_transceiver_config: external callers
    # (e.g. the perf-sanity cache-transceiver precheck) invoke the resolver
    # with a lightweight stand-in object, and per-server fields beyond this
    # config could resolve differently on ctx and gen servers (e.g. context
    # parallelism, where only the gen side runs helix). Conditions the
    # Python transceiver cannot serve (such as non-helix CP) fail loudly at
    # transceiver creation instead.
    return None


def _resolve_transceiver_runtime_auto(llm_args: 'TorchLlmArgs',
                                      model_cls: Optional[type] = None,
                                      pretrained_config: Any = None) -> None:
    """Resolve the 'auto' sentinel in cache_transceiver_config.transceiver_runtime.

    Semantics:
    - Disagg disabled (config is None or backend is None): no-op. Resolution
      must never materialize or alter a transceiver config that the user did
      not enable.
    - Explicit user value ('CPP'/'PYTHON'/None): left untouched.
    - 'auto': a model preference from
      ``model_cls.get_preferred_transceiver_runtime()`` ('CPP' or 'PYTHON')
      is adopted verbatim — never rerouted, so unsupported configurations
      surface as transceiver-creation errors. Without a preference, default
      to the Python (V2) transceiver, falling back to None (C++ transceiver)
      for configurations it does not support (non-NIXL backend or an
      infinite kv_transfer_timeout_ms).

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

    resolved = preferred if preferred is not None else "PYTHON"

    # Fallbacks apply only to the no-preference default: an explicit model
    # preference is adopted verbatim (see _transceiver_python_fallback_reason).
    if preferred is None:
        fallback_reason = _transceiver_python_fallback_reason(llm_args)
        if fallback_reason is not None:
            logger.info(
                f"Falling back to the C++ transceiver: {fallback_reason}.")
            resolved = None

    cfg.transceiver_runtime = resolved
    logger.info(
        f"Resolved transceiver_runtime='auto' to {resolved!r} for "
        f"{model_cls.__name__ if model_cls is not None else 'unknown model'}.")
