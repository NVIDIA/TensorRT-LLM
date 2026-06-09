# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Internal VisualGen pipeline and model configuration helpers."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    CacheBackendName,
    CacheConfig,
    CacheDiTConfig,
    CompilationConfig,
    CudaGraphConfig,
    ParallelConfig,
    TeaCacheConfig,
    TorchCompileConfig,
    VisualGenArgs,
)
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig
from tensorrt_llm.visual_gen.sparse_attention import (
    auto_detect_sparse_attention_config as _auto_detect_sparse_attention_config,
)
from tensorrt_llm.visual_gen.sparse_attention import (
    auto_detect_sparse_yaml as _auto_detect_sparse_yaml,
)
from tensorrt_llm.visual_gen.sparse_attention import (
    load_sparse_config_from_yaml as _load_sparse_config_from_yaml,
)

# =============================================================================
# Utilities
# =============================================================================


def discover_pipeline_components(checkpoint_path: Path) -> Dict[str, Path]:
    """
    Discover components from diffusers pipeline's model_index.json.

    Returns dict mapping component name to config.json path.
    """
    model_index_path = checkpoint_path / "model_index.json"
    if not model_index_path.exists():
        return {}

    with open(model_index_path) as f:
        model_index = json.load(f)

    components = {}
    for key, value in model_index.items():
        if key.startswith("_") or value is None:
            continue
        config_path = checkpoint_path / key / "config.json"
        if config_path.exists():
            components[key] = config_path

    return components


def create_attention_metadata_state() -> Dict[str, Any]:
    """Create model-scoped attention metadata state for TRTLLM visual-gen backend."""
    return {"metadata": None, "capacity": (0, 0)}


def _model_config_value(value: Any, *, deep_copy: bool = True) -> Any:
    """Return a value for a per-component DiffusionModelConfig."""
    if value is None:
        return None
    if not deep_copy:
        return value
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    return deepcopy(value)


class _VisualGenConfigBase(BaseModel):
    """Base for internal VisualGen configs that carry runtime objects."""

    # Pydantic reserves `model_config` for class-level settings. This is not
    # a VisualGen model config; it lets fields hold objects such as Mapping.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DiffusionModelConfig(_VisualGenConfigBase):
    """Internal config for one TRT-LLM VisualGen model component."""

    component_name: Optional[str] = None
    pretrained_config: Optional[Any] = None
    mapping: Mapping = PydanticField(default_factory=Mapping)
    skip_create_weights_in_init: bool = False
    force_dynamic_quantization: bool = False
    allreduce_strategy: AllReduceStrategy = PydanticField(default=AllReduceStrategy.NCCL)
    extra_attrs: Dict = PydanticField(default_factory=dict)

    # Unified parallelism mapping copied from the owning pipeline config.
    visual_gen_mapping: Optional[Any] = None  # VisualGenMapping (lazy import)

    dynamic_weight_quant: bool = False

    # Shared runtime configs copied from the owning pipeline config.
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    # Per-layer quant (from load_diffusion_quant_config layer_quant_config; None until mixed-precision parsing exists)
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    compilation: CompilationConfig = PydanticField(default_factory=CompilationConfig)
    torch_compile: TorchCompileConfig = PydanticField(default_factory=TorchCompileConfig)
    cuda_graph: CudaGraphConfig = PydanticField(default_factory=CudaGraphConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    attention_metadata_state: Optional[Dict[str, Any]] = None
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    cache: Optional[CacheConfig] = None

    # Observability — flat field mirrors VisualGenArgs.enable_layerwise_nvtx_marker.
    enable_layerwise_nvtx_marker: bool = False

    @property
    def cache_backend(self) -> Optional[CacheBackendName]:
        return self.cache.cache_backend if self.cache is not None else None  # type: ignore[return-value]

    @property
    def teacache(self) -> Optional[TeaCacheConfig]:
        return self.cache if isinstance(self.cache, TeaCacheConfig) else None

    @property
    def cache_dit(self) -> Optional[CacheDiTConfig]:
        return self.cache if isinstance(self.cache, CacheDiTConfig) else None

    @property
    def torch_dtype(self) -> "torch.dtype":
        """Get the torch dtype of the model (default: bfloat16)."""
        return torch.bfloat16

    def get_quant_config(self, name: Optional[str] = None) -> QuantConfig:
        """Get quantization config for a layer or global. Resembles LLM ModelConfig.get_quant_config."""
        if name is None or self.quant_config_dict is None:
            return self.quant_config
        if name in self.quant_config_dict:
            return self.quant_config_dict[name]
        return self.quant_config


# =============================================================================
# DiffusionPipelineConfig - Internal pipeline configuration (merged/parsed)
# =============================================================================


class DiffusionPipelineConfig(_VisualGenConfigBase):
    """Internal config for an entire VisualGen pipeline.

    This is created by PipelineLoader from VisualGenArgs + checkpoint and owns
    pipeline/runtime state plus one DiffusionModelConfig per model component.
    """

    model_configs: Dict[str, DiffusionModelConfig] = PydanticField(default_factory=dict)
    mapping: Mapping = PydanticField(default_factory=Mapping)
    skip_create_weights_in_init: bool = False
    force_dynamic_quantization: bool = False
    allreduce_strategy: AllReduceStrategy = PydanticField(default=AllReduceStrategy.NCCL)
    extra_attrs: Dict = PydanticField(default_factory=dict)

    # Unified parallelism mapping (populated by setup_visual_gen_mapping)
    visual_gen_mapping: Optional[Any] = None  # VisualGenMapping (lazy import)

    dynamic_weight_quant: bool = False

    # Sub-configs from VisualGenArgs (merged during from_pretrained)
    quant_config: QuantConfig = PydanticField(default_factory=QuantConfig)
    # Per-layer quant (from load_diffusion_quant_config layer_quant_config; None until mixed-precision parsing exists)
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None
    compilation: CompilationConfig = PydanticField(default_factory=CompilationConfig)
    torch_compile: TorchCompileConfig = PydanticField(default_factory=TorchCompileConfig)
    cuda_graph: CudaGraphConfig = PydanticField(default_factory=CudaGraphConfig)
    attention: AttentionConfig = PydanticField(default_factory=AttentionConfig)
    attention_metadata_state: Optional[Dict[str, Any]] = None
    parallel: ParallelConfig = PydanticField(default_factory=ParallelConfig)
    cache: Optional[CacheConfig] = None

    # Observability — flat field mirrors VisualGenArgs.enable_layerwise_nvtx_marker.
    enable_layerwise_nvtx_marker: bool = False

    @property
    def primary_model_config(self) -> DiffusionModelConfig:
        return self.model_configs["transformer"]

    @property
    def primary_pretrained_config(self) -> Any:
        return self.primary_model_config.pretrained_config

    @property
    def cache_backend(self) -> Optional[CacheBackendName]:
        return self.cache.cache_backend if self.cache is not None else None  # type: ignore[return-value]

    @property
    def teacache(self) -> Optional[TeaCacheConfig]:
        return self.cache if isinstance(self.cache, TeaCacheConfig) else None

    @property
    def cache_dit(self) -> Optional[CacheDiTConfig]:
        return self.cache if isinstance(self.cache, CacheDiTConfig) else None

    @property
    def torch_dtype(self) -> "torch.dtype":
        """Get the torch dtype of the model (default: bfloat16)."""
        return torch.bfloat16

    def get_quant_config(self, name: Optional[str] = None) -> QuantConfig:
        """Get quantization config for a layer or global. Resembles LLM ModelConfig.get_quant_config."""
        if name is None or self.quant_config_dict is None:
            return self.quant_config
        if name in self.quant_config_dict:
            return self.quant_config_dict[name]
        return self.quant_config

    def _make_model_config(
        self,
        component_name: str,
        model_pretrained_config: Any,
    ) -> DiffusionModelConfig:
        return DiffusionModelConfig(
            component_name=_model_config_value(component_name),
            pretrained_config=_model_config_value(model_pretrained_config),
            # Topology mappings carry distributed process-group handles.
            mapping=_model_config_value(self.mapping, deep_copy=False),
            skip_create_weights_in_init=_model_config_value(self.skip_create_weights_in_init),
            force_dynamic_quantization=_model_config_value(self.force_dynamic_quantization),
            allreduce_strategy=_model_config_value(self.allreduce_strategy),
            extra_attrs=_model_config_value(self.extra_attrs),
            # Topology mappings carry distributed process-group handles.
            visual_gen_mapping=_model_config_value(self.visual_gen_mapping, deep_copy=False),
            dynamic_weight_quant=_model_config_value(self.dynamic_weight_quant),
            quant_config=_model_config_value(self.quant_config),
            quant_config_dict=_model_config_value(self.quant_config_dict),
            compilation=_model_config_value(self.compilation),
            torch_compile=_model_config_value(self.torch_compile),
            cuda_graph=_model_config_value(self.cuda_graph),
            attention=_model_config_value(self.attention),
            attention_metadata_state=_model_config_value(self.attention_metadata_state),
            parallel=_model_config_value(self.parallel),
            cache=_model_config_value(self.cache),
            enable_layerwise_nvtx_marker=_model_config_value(self.enable_layerwise_nvtx_marker),
        )

    @staticmethod
    def load_diffusion_quant_config(
        quant_config_dict: dict,
    ) -> Tuple[QuantConfig, Optional[Dict], bool, bool]:
        """
        Parse quantization config in ModelOpt format.

        Returns: (quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant)
            - quant_config: Global QuantConfig
            - layer_quant_config: Per-layer config dict (None if not using mixed precision)
            - dynamic_weight_quant: Whether to quantize weights at load time
            - dynamic_activation_quant: Whether to quantize activations dynamically
        """
        quant_algo_str = quant_config_dict.get("quant_algo")
        quant_algo = None
        if quant_algo_str:
            algo_map = {
                "FP8": QuantAlgo.FP8,
                "FP8_BLOCK_SCALES": QuantAlgo.FP8_BLOCK_SCALES,
                "NVFP4": QuantAlgo.NVFP4,
                "W4A16_AWQ": QuantAlgo.W4A16_AWQ,
                "W4A8_AWQ": QuantAlgo.W4A8_AWQ,
                "W8A8_SQ_PER_CHANNEL": QuantAlgo.W8A8_SQ_PER_CHANNEL,
            }
            quant_algo = algo_map.get(quant_algo_str)
            if quant_algo is None:
                raise ValueError(f"Unknown quant_algo: {quant_algo_str}")

        # Parse group_size and dynamic flags from config_groups
        group_size = None
        dynamic_weight_quant = False
        dynamic_activation_quant = False
        for group_config in quant_config_dict.get("config_groups", {}).values():
            weights_config = group_config.get("weights", {})
            activations_config = group_config.get("input_activations", {})
            dynamic_weight_quant = weights_config.get("dynamic", False)
            dynamic_activation_quant = activations_config.get("dynamic", False)
            # Extract group_size from weights config (e.g., NVFP4: group_size=16)
            if group_size is None:
                group_size = weights_config.get("group_size")
            break

        # Set defaults based on quant_algo if group_size not specified
        if group_size is None:
            if quant_algo in (QuantAlgo.NVFP4,):
                group_size = 16  # NVFP4 default
            elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
                group_size = 128  # FP8 blockwise default

        # Auto-enable dynamic weight quantization if quant_algo is specified
        # but no explicit config_groups setting is present.
        # This allows simple configs like {"quant_algo": "FP8"} to work.
        if quant_algo is not None and not quant_config_dict.get("config_groups"):
            dynamic_weight_quant = quant_config_dict.get("dynamic", True)
            # NVFP4 requires dynamic activation quantization when using dynamic mode
            # since input_scale is not calibrated
            if quant_algo == QuantAlgo.NVFP4 and dynamic_weight_quant:
                dynamic_activation_quant = True

        quant_config = QuantConfig(
            quant_algo=quant_algo,
            group_size=group_size,
            exclude_modules=quant_config_dict.get("ignore"),
        )

        # TODO: Per-layer config (None for now - future: parse mixed precision settings)
        layer_quant_config = None

        return quant_config, layer_quant_config, dynamic_weight_quant, dynamic_activation_quant

    @staticmethod
    def _convert_quantization_metadata(
        qmeta: Dict,
        tensor_keys: List[str],
    ) -> Dict:
        """
        TODO: Consider refactor this to be a utility functions.
        Convert per-layer ``_quantization_metadata`` to ModelOpt format.

        Some checkpoints (e.g. HuggingFace-quantized FP8) embed per-layer
        quantization info as::

            {"format_version": "1.0",
             "layers": {"model.diffusion_model.block.attn.to_q": {"format": "float8_e4m3fn"}, ...}}

        This converts it to the ModelOpt-compatible dict that
        :meth:`load_diffusion_quant_config` understands::

            {"quant_algo": "FP8",
             "config_groups": {"default": {"weights": {"dynamic": false}, ...}},
             "ignore": ["proj_in", "proj_out", ...]}
        """
        _FORMAT_TO_ALGO = {
            "float8_e4m3fn": "FP8",
        }

        layers = qmeta.get("layers", {})
        if not layers:
            return {}

        formats = {info.get("format") for info in layers.values()}
        if len(formats) != 1:
            logger.warning(f"_quantization_metadata has mixed formats {formats}; skipping")
            return {}

        fmt = formats.pop()
        quant_algo = _FORMAT_TO_ALGO.get(fmt)
        if quant_algo is None:
            logger.warning(f"_quantization_metadata format '{fmt}' is not supported; skipping")
            return {}

        quantized_layers = set(layers.keys())

        # Build ignore list: weight-bearing layers NOT in the quantized set.
        # Tensor keys ending with ".weight" (but not ".weight_scale") indicate
        # layers that own learnable weights.
        non_quantized = []
        for key in tensor_keys:
            if key.endswith(".weight") and not key.endswith("_scale.weight"):
                layer_name = key[: -len(".weight")]
                if layer_name not in quantized_layers:
                    non_quantized.append(layer_name)

        result = {
            "quant_algo": quant_algo,
            "config_groups": {
                "default": {
                    "weights": {"dynamic": False},
                    "input_activations": {"dynamic": False},
                }
            },
            "ignore": sorted(non_quantized),
        }
        logger.info(
            f"Converted _quantization_metadata: algo={quant_algo}, "
            f"{len(quantized_layers)} quantized layers, "
            f"{len(non_quantized)} excluded layers"
        )
        return result

    @classmethod
    def _try_load_safetensors_config(cls, checkpoint_path: Path) -> Optional[Dict]:
        """Try to read embedded config from a single-safetensors checkpoint.

        Accepts either a directory (globs for ``*.safetensors``) or a direct
        path to a ``.safetensors`` file.

        Returns the full config dict if found, ``None`` otherwise.
        """
        try:
            import safetensors.torch
        except ImportError:
            return None

        if checkpoint_path.is_file() and checkpoint_path.suffix == ".safetensors":
            sft_files = [checkpoint_path]
        else:
            sft_files = sorted(checkpoint_path.glob("*.safetensors"))

        if not sft_files:
            return None

        try:
            with safetensors.torch.safe_open(str(sft_files[0]), framework="pt") as f:
                meta = f.metadata()
                if meta and "config" in meta:
                    config = json.loads(meta["config"])
                    if "quantization_config" in meta:
                        config["quantization_config"] = json.loads(meta["quantization_config"])
                    elif "_quantization_metadata" in meta:
                        qmeta = json.loads(meta["_quantization_metadata"])
                        converted = cls._convert_quantization_metadata(qmeta, list(f.keys()))
                        if converted:
                            config["quantization_config"] = converted
                    return config
        except Exception:
            pass
        return None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        args: Optional["VisualGenArgs"] = None,
        **kwargs,
    ) -> "DiffusionPipelineConfig":
        """
        Load config from pretrained checkpoint.

        Called by PipelineLoader with VisualGenArgs:
            config = DiffusionPipelineConfig.from_pretrained(
                checkpoint_dir=args.model,
                args=args,
            )

        Supports two checkpoint formats:
        * **Diffusers directory layout** -- ``model_index.json`` with
          component sub-directories each containing ``config.json``.
        * **Single-safetensors** -- no ``model_index.json``; config embedded
          in the safetensors metadata header under a ``"config"`` key.  The
          transformer section is extracted as ``pretrained_config`` and the
          full dict is stored in ``extra_attrs["monolithic_safetensors_config"]``
          for use by component configurators.

        Args:
            checkpoint_dir: Path to checkpoint
            args: VisualGenArgs containing user config
                - (compilation, torch_compile, cuda_graph, pipeline, attention, parallel, teacache,
                   cache_backend, cache_dit)
            **kwargs: Additional config options (e.g., mapping)
        """
        kwargs.pop("trust_remote_code", None)

        # Extract sub-configs from args or use defaults.
        compilation_cfg = args.compilation_config if args else CompilationConfig()
        torch_compile_cfg = args.torch_compile_config if args else TorchCompileConfig()
        cuda_graph_cfg = args.cuda_graph_config if args else CudaGraphConfig()
        attention_cfg = args.attention_config if args else AttentionConfig()
        parallel_cfg = args.parallel_config if args else ParallelConfig()
        cache_cfg = args.cache_config if args else None
        enable_layerwise_nvtx_marker = bool(args.enable_layerwise_nvtx_marker) if args else False

        from .pipeline_registry import PipelineComponent

        component = PipelineComponent.TRANSFORMER
        checkpoint_path = Path(checkpoint_dir)
        extra_attrs: Dict[str, Any] = {}

        # LTX-2 stage-2 paths (spatial_upsampler_path, distilled_lora_path)
        # are surfaced to the LTX2 pipeline consumer via extra_attrs. The
        # resolved pipeline_config kwarg comes from PipelineLoader after
        # registry validation; when from_pretrained is called directly
        # (mostly in unit tests), fall back to the raw VisualGenArgs dict.
        resolved_pipeline_config = kwargs.pop("pipeline_config", None)
        if resolved_pipeline_config is None:
            resolved_pipeline_config = dict(args.pipeline_config) if args else {}
        for key in ("spatial_upsampler_path", "distilled_lora_path"):
            value = resolved_pipeline_config.get(key)
            if value:
                extra_attrs[key] = value

        # Discover pipeline components (diffusers layout)
        components = discover_pipeline_components(checkpoint_path)
        component_config_dicts: Dict[str, Dict[str, Any]] = {}

        if components:
            # ---------- Diffusers directory layout ----------
            if component not in components:
                raise ValueError(
                    f"Component '{component}' not found. Available: {list(components.keys())}"
                )
            config_path = components[component]
            if not config_path.exists():
                raise ValueError(f"Config not found at {config_path}")

            for component_name, component_config_path in components.items():
                with open(component_config_path) as f:
                    component_config_dicts[component_name] = json.load(f)

            config_dict = component_config_dicts[component]
            pretrained_config = SimpleNamespace(**config_dict)

            # Ensure _name_or_path is set so TeaCache coefficient matching works.
            if not getattr(pretrained_config, "_name_or_path", None):
                pretrained_config._name_or_path = str(checkpoint_path)

            model_index_path = checkpoint_path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path) as f:
                    model_index = json.load(f)
                if "boundary_ratio" in model_index and "transformer_2" in model_index:
                    transformer_2_spec = model_index.get("transformer_2")
                    if transformer_2_spec and transformer_2_spec[0] is not None:
                        pretrained_config.boundary_ratio = model_index["boundary_ratio"]
                if "expand_timesteps" in model_index:
                    pretrained_config.expand_timesteps = bool(model_index["expand_timesteps"])
        else:
            # ---------- Single safetensors ----------
            native_config = cls._try_load_safetensors_config(checkpoint_path)

            if native_config is not None:
                transformer_dict = native_config.get("transformer", {})
                component_config_dicts["transformer"] = transformer_dict
                transformer_2_dict = native_config.get("transformer_2")
                if isinstance(transformer_2_dict, dict):
                    component_config_dicts["transformer_2"] = transformer_2_dict
                pretrained_config = SimpleNamespace(**transformer_dict)
                if not getattr(pretrained_config, "_name_or_path", None):
                    pretrained_config._name_or_path = str(checkpoint_path)
                extra_attrs["monolithic_safetensors_config"] = native_config

                # quantization_config lives as a separate safetensors metadata
                # key, not inside the transformer section. Propagate it so
                # the quant-config resolution below can pick it up.
                if "quantization_config" in native_config:
                    qc = native_config["quantization_config"]
                    # ModelOpt prefixes module names with the wrapped model
                    # attribute (e.g. "velocity_model.proj_out"). Strip that
                    # wrapper prefix so the ignore list matches TRT-LLM names.
                    _MODELOPT_WRAPPER_PREFIXES = (
                        "model.diffusion_model.",
                        "velocity_model.",
                        "denoiser.",
                        "unet.",
                        "dit.",
                    )
                    if "ignore" in qc and qc["ignore"]:
                        cleaned = []
                        for entry in qc["ignore"]:
                            for wp in _MODELOPT_WRAPPER_PREFIXES:
                                if entry.startswith(wp):
                                    entry = entry[len(wp) :]
                                    break
                            cleaned.append(entry)
                        qc["ignore"] = cleaned
                    pretrained_config.quantization_config = qc
            else:
                raise ValueError(
                    f"Config not found at {checkpoint_dir}. "
                    "Expected model_index.json (diffusers) or "
                    "safetensors with embedded config metadata."
                )

        # Load sparse attention calibration. ModelOpt artifacts carry the
        # calibration formula and disabled-layer/component map; user config
        # contributes only public knobs such as target_sparsity.
        yaml_sparse = None
        yaml_path = attention_cfg.sparse_config_path
        if yaml_path is not None:
            yaml_sparse = _load_sparse_config_from_yaml(yaml_path)
            if yaml_sparse is not None:
                logger.info(f"Loaded sparse config from {yaml_path}")

        if yaml_sparse is None:
            yaml_sparse = _auto_detect_sparse_yaml(str(checkpoint_path))
            if yaml_sparse is not None:
                logger.info("Auto-detected sparse config YAML from checkpoint")

        if yaml_sparse is None:
            ckpt_dict = vars(pretrained_config) if pretrained_config else {}
            yaml_sparse = _auto_detect_sparse_attention_config(ckpt_dict)
            if yaml_sparse is not None:
                formula = yaml_sparse._formula
                if formula is not None:
                    logger.info(
                        "Auto-detected sparse config from config.json "
                        f"(formula: {formula.formula!r}, coefficients: {formula.coefficients})"
                    )
                else:
                    logger.info("Auto-detected sparse config from config.json")

        if yaml_sparse is not None:
            user_cfg = attention_cfg.sparse_attention_config
            if user_cfg is not None and isinstance(user_cfg, SkipSoftmaxAttentionConfig):
                attention_cfg = attention_cfg.model_copy(
                    update={"sparse_attention_config": yaml_sparse._with_public_overrides(user_cfg)}
                )
            else:
                attention_cfg = attention_cfg.model_copy(
                    update={"sparse_attention_config": yaml_sparse}
                )

        # Resolve quant_config. A user dict containing ``quant_algo``
        # is parsed via ``load_diffusion_quant_config``; a user dict
        # without ``quant_algo`` (including ``{}``) falls through to the
        # checkpoint's embedded ``quantization_config``.
        user_quant = args.quant_config if args else None
        if isinstance(user_quant, dict) and user_quant.get("quant_algo") is not None:
            quant_config, quant_config_dict, dynamic_weight_quant, dynamic_activation_quant = (
                cls.load_diffusion_quant_config(user_quant)
            )
        elif isinstance(user_quant, QuantConfig) and user_quant.quant_algo is not None:
            quant_config = user_quant
            quant_config_dict = None
            dynamic_weight_quant = False
            dynamic_activation_quant = False
        else:
            quant_config = QuantConfig()
            quant_config_dict = None
            dynamic_weight_quant = False
            dynamic_activation_quant = False
            quant_dict = getattr(pretrained_config, "quantization_config", None)
            if isinstance(quant_dict, dict):
                quant_config, quant_config_dict, dynamic_weight_quant, dynamic_activation_quant = (
                    cls.load_diffusion_quant_config(quant_dict)
                )

        # Enable tunable FP4 quantize for visual gen: larger activation
        # tensors (full image/video latents) amortize the AutoTuner overhead.
        if quant_config.quant_algo == QuantAlgo.NVFP4:
            from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod

            NVFP4LinearMethod.use_tunable_quantize = True

        attention_metadata_state = (
            create_attention_metadata_state() if attention_cfg.backend == "TRTLLM" else None
        )

        pipeline_config = cls(
            quant_config=quant_config,
            quant_config_dict=quant_config_dict,
            dynamic_weight_quant=dynamic_weight_quant,
            force_dynamic_quantization=dynamic_activation_quant,
            # Sub-configs from VisualGenArgs
            compilation=compilation_cfg,
            torch_compile=torch_compile_cfg,
            cuda_graph=cuda_graph_cfg,
            attention=attention_cfg,
            attention_metadata_state=attention_metadata_state,
            parallel=parallel_cfg,
            cache=cache_cfg,
            enable_layerwise_nvtx_marker=enable_layerwise_nvtx_marker,
            skip_create_weights_in_init=True,
            extra_attrs=extra_attrs,
            **kwargs,
        )

        for component_name, config_dict in component_config_dicts.items():
            if component_name == component:
                component_pretrained_config = pretrained_config
            else:
                component_pretrained_config = SimpleNamespace(**config_dict)
                if not getattr(component_pretrained_config, "_name_or_path", None):
                    component_pretrained_config._name_or_path = getattr(
                        pretrained_config, "_name_or_path", ""
                    )
            pipeline_config.model_configs[component_name] = pipeline_config._make_model_config(
                component_name,
                component_pretrained_config,
            )

        if not pipeline_config.model_configs:
            pipeline_config.model_configs["transformer"] = pipeline_config._make_model_config(
                "transformer",
                pretrained_config,
            )

        return pipeline_config
