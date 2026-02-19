"""Interface to initialize and load HF models."""

import json
import os
import re
import types
from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import safetensors.torch
import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_in_model
from accelerate.utils import modeling
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HFValidationError, filter_repo_objects, validate_repo_id
from PIL import Image
from torch._prims_common import DeviceLikeType
from torch.export import Dim
from torch.fx import GraphModule
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from ..utils._config import deep_merge_dicts
from ..utils.logger import ad_logger
from .factory import (
    DynamicShape,
    FullModelExportInfo,
    ModelFactory,
    ModelFactoryRegistry,
    ShardingConfigSource,
    SubModuleExportInfo,
)
from .quant_config_reader import QuantConfigReader, autodetect_quant_config_reader


@contextmanager
def hf_load_state_dict_with_device(device: DeviceLikeType):
    """Patch HF loading utilities according to our needs.

    Following patches are applied:
        1. load_state_dict to use provided device. NOTE (lucaslie): this function is called by
           ``load_checkpoint_in_model``. We provide the device map here as a patch instead of going
           through ``load_checkpoint_in_model``. This is because otherwise
           ``load_checkpoint_in_model`` will execute its own state_dict loading logic instead of
           calling ``nn.Module.load_state_dict``. However, we rely on the state dict loading hooks
           in ``nn.Module.load_state_dict`` to correctly load the weights. By providing the device
           map here, we can ensure that ``load_checkpoint_in_model`` will call
           ``nn.Module.load_state_dict``.
        2. change logging level of logger to ERROR to avoid logging warnings from HF state_dict
           loading for missing/unexpected keys (happens for MoE expert-sharded layers for example).
    """
    # save the original load_state_dict method
    original_load_state_dict = modeling.load_state_dict

    # save the original logger level
    original_logger_level = modeling.logger.level

    # Define and apply the patched version
    def load_state_dict_with_device(checkpoint_file, device_map=None):
        return original_load_state_dict(checkpoint_file, device_map={"": device})

    # Apply the patch
    modeling.load_state_dict = load_state_dict_with_device

    # Change the logger level to ERROR
    modeling.logger.setLevel("ERROR")

    try:
        yield
    finally:
        # Restore the original method, even if an exception occurred
        modeling.load_state_dict = original_load_state_dict
        # Restore the original logger level
        modeling.logger.setLevel(original_logger_level)


# TODO (lucaslie): continue working on the base class
class AutoModelFactory(ModelFactory):
    @property
    @abstractmethod
    def automodel_cls(self) -> Type[_BaseAutoModelClass]:
        """Get the AutoModel class for calling from_pretrained and from_config."""


@ModelFactoryRegistry.register("AutoModelForCausalLM")
class AutoModelForCausalLMFactory(AutoModelFactory):
    _tokenizer_defaults = {
        "legacy": False,
        "padding_side": "left",
        "truncation_side": "left",
        "trust_remote_code": True,
        "use_fast": True,
    }

    _model_defaults = {
        "use_cache": False,
    }

    # The below maps from a model's config class definition's name (str) to the alternative `AutoModelForCausalLM`
    # implementation we would like to use.
    _custom_model_mapping: Dict[str, Type[AutoModelForCausalLM]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quant_config_reader: QuantConfigReader | None = None
        # Ingest defaults for tokenizer and model kwargs
        self.tokenizer_kwargs = deep_merge_dicts(self._tokenizer_defaults, self.tokenizer_kwargs)
        self.model_kwargs = deep_merge_dicts(
            self._model_defaults,
            self.model_kwargs or {},
        )

        # set sharding config source to huggingface
        self._sharding_config["source"] = ShardingConfigSource.HUGGINGFACE

        # Some models' transformers implementation has changed in between when safetensors were produced
        # and / or uploaded to HuggingFace hub. When building the model, we will try to determine whether
        # a mapping of the parameter names exists and hold that information in this attribute.
        self._checkpoint_conversion_mapping: Optional[Dict[str, str]] = None

    @property
    def automodel_cls(self) -> Type[_BaseAutoModelClass]:
        return AutoModelForCausalLM

    @property
    def vocab_size_padded(self) -> Optional[int]:
        model_config, _ = self._get_model_config()
        return getattr(model_config, "vocab_size", None)

    def _recursive_update_config(
        self, config: PretrainedConfig, update_dict: Dict[str, Any]
    ) -> Tuple[PretrainedConfig, Dict[str, Any]]:
        """
        Deep-merge a PretrainedConfig object with values from update_dict.

        Args:
            config: PretrainedConfig object to update
            update_dict: Dictionary with values to update in the config

        Returns:
            A tuple of (updated_config, nested_unused_kwargs) where nested_unused_kwargs captures
            any keys from update_dict that could not be applied to config, preserving nesting.
        """
        nested_unused_kwargs: Dict[str, Any] = {}

        for key, value_new in update_dict.items():
            # Check if the key exists in config
            if not hasattr(config, key):
                nested_unused_kwargs[key] = value_new
                continue

            target_value = getattr(config, key)

            # Handle nested PretrainedConfig objects...
            if isinstance(value_new, dict) and isinstance(target_value, PretrainedConfig):
                # Recursively update nested configs
                updated_value, child_unused = self._recursive_update_config(target_value, value_new)
                setattr(config, key, updated_value)
                if child_unused:
                    nested_unused_kwargs[key] = child_unused
            elif (
                key in ["torch_dtype", "dtype"]
                and isinstance(value_new, str)
                and value_new != "auto"
            ):
                # check special handling of torch_dtype (DEPRECATED!) and dtype key to ensure we
                # use the correct torch.dtype object instead of a string.
                dtype = getattr(torch, value_new)
                assert isinstance(dtype, torch.dtype), f"Invalid {dtype=}"
                setattr(config, key, dtype)
            else:
                # Direct update for simple values
                setattr(config, key, value_new)

        return config, nested_unused_kwargs

    def _get_model_config(self) -> Tuple[PretrainedConfig, Dict[str, Any]]:
        # prefetch the model once without weights
        self.prefetch_checkpoint(skip_loading_weights=True)

        # NOTE (lucaslie): HF doesn't recursively update nested PreTrainedConfig objects. Instead,
        # the entire subconfig will be overwritten.
        # we want to recursively update model_config from model_kwargs here.
        model_config, unused_kwargs = AutoConfig.from_pretrained(
            self.model, return_unused_kwargs=True, trust_remote_code=True
        )
        model_config, nested_unused_kwargs = self._recursive_update_config(
            model_config, self.model_kwargs
        )
        # merge nested unused kwargs into HF's unused kwargs (preserve nesting)
        merged_unused = deep_merge_dicts(unused_kwargs, nested_unused_kwargs)
        return model_config, merged_unused

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        """Build the model on the desired device."""
        model_config, unused_kwargs = self._get_model_config()

        config_cls_name = type(model_config).__name__
        custom_model_cls = self._custom_model_mapping.get(config_cls_name, None)
        with (init_empty_weights if device == "meta" else nullcontext)():
            if custom_model_cls is not None:
                # `_from_config` has some behavior we would like to use where possible. It is
                # defined in the `PreTrainedModel` mixin.
                ad_logger.info(f"Using custom model implementation {custom_model_cls}")
                if not hasattr(custom_model_cls, "_from_config"):
                    raise ValueError(
                        f"`{custom_model_cls.__name__}` must have a `_from_config` class method. "
                        "Consider inheriting from `PreTrainedModel`."
                    )
                model = custom_model_cls._from_config(model_config, **unused_kwargs)
            else:
                model = self.automodel_cls.from_config(
                    model_config,
                    **{
                        "trust_remote_code": True,
                        **unused_kwargs,
                    },
                )

        if device == "meta":
            # post-init --> this must be called explicitly for HF models the way we initialize them
            # since this "gets lost" with the init_empty_weights context manager.
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        # if present, initialize sharding config.
        self._set_sharding_config(model.config)
        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)

        model.eval()

        if self._quant_config_reader is not None:
            model = self._quant_config_reader.post_process_model(model, model_config)

        return model

    def _set_sharding_config(self, model_config: PretrainedConfig):
        """Set the sharding config for the model."""
        if hasattr(model_config, "base_model_tp_plan"):
            self._sharding_config["tp_plan"] = model_config.base_model_tp_plan

    def get_quant_config(self) -> Dict:
        """Returns the quantization config for this model or an empty dict if not quantized."""
        if self._quant_config_reader is not None:
            return self._quant_config_reader.get_config()
        return {}

    def get_cache_config_updates(self):
        """Return kv cache dtype updates."""
        if not self._quant_config_reader:
            return {}

        kv_cache_dtype = self._quant_config_reader.get_config().get("kv_cache_dtype", "auto")
        assert kv_cache_dtype in ("fp8", "auto"), (
            f"Unsupported dtype: {kv_cache_dtype}. Only fp8 and auto are supported."
        )
        return {"dtype": kv_cache_dtype}

    def init_tokenizer(self) -> Optional[Any]:
        """Initialize the tokenizer—either a custom name or the model's default."""
        if self.tokenizer is None:
            return None
        return AutoTokenizer.from_pretrained(self.tokenizer, **self.tokenizer_kwargs)

    def build_and_load_model(self, device: DeviceLikeType) -> nn.Module:
        """Automatically build the model from_pretrained and load the weights.

        Args:
            device: The device to build the model on.

        Returns:
            The built model.

        If we skip weight loading, we will fall back to the build_model+load_or_random_init methods.
        NOTE that there is NO sharding when skip_loading_weights is True.
        """
        # only this way can we skip downloading/loading weights
        if self.skip_loading_weights or "cuda" not in str(device):
            ad_logger.info("Falling back to build_model+load_or_random_init methods.")
            model = self.build_model("meta")
            self.load_or_random_init(model, device)
            return model

        # full joint loading of weights and model
        self.prefetch_checkpoint(force=True)  # ensuring weights are downloaded
        model_config, unused_kwargs = self._get_model_config()
        model = self.automodel_cls.from_pretrained(
            self.model,
            config=model_config,
            **{
                "trust_remote_code": True,
                "tp_plan": "auto",
                **unused_kwargs,
                "dtype": "auto",  # takes precedence over unused_kwargs!
            },
        )
        model.eval()
        return model

    @staticmethod
    def _get_ignore_patterns(repo_id: str, skip_prefetch_weights: bool) -> List[str]:
        """Get the ignore patterns for the HF repo."""
        ignore_patterns = ["*.pt", "*.pth"]
        bin_pattern = "*.bin*"
        safetensors_pattern = "*.safetensors*"
        if skip_prefetch_weights:
            ignore_patterns.extend([bin_pattern, safetensors_pattern])
            return ignore_patterns

        # now check if we can identify safetensors or bin ignore patterns from the repo
        # make this totally fail-safe...
        try:
            validate_repo_id(repo_id)
            api = HfApi()
            repo_info = api.repo_info(repo_id)

            # check files in the repo
            files = [f.rfilename for f in repo_info.siblings]

            # if we have safetensors we can ignore all bin files
            if list(filter_repo_objects(items=files, allow_patterns=[safetensors_pattern])):
                ignore_patterns.append(bin_pattern)
        except:  # noqa: E722
            pass

        return ignore_patterns

    def _get_checkpoint_file(self, checkpoint: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """Get the most relevant checkpoint file from the HF checkpoint directory.

        Args:
            checkpoint: The path to the checkpoint directory or file.

        Returns:
            The path to the most relevant checkpoint file.

        Following order of precedence for the checkpoint file:

            1. checkpoint is a file
            2. checkpoint dir contains SAFE_WEIGHTS_INDEX_NAME
            3. checkpoint dir contains SAFE_WEIGHTS_NAME
            4. checkpoint dir contains WEIGHTS_INDEX_NAME
            5. checkpoint dir contains WEIGHTS_NAME

        """

        if os.path.isfile(checkpoint):
            return checkpoint

        # Verify that checkpoint is a directory
        if not os.path.isdir(checkpoint):
            raise ValueError(
                f"Checkpoint path '{checkpoint}' is neither a file nor a directory, or does not exist."
            )

        # now check which is the most relevant checkpoint file
        safe_weights_index_path = os.path.join(checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.isfile(safe_weights_index_path):
            return safe_weights_index_path

        safe_weights_path = os.path.join(checkpoint, SAFE_WEIGHTS_NAME)
        if os.path.isfile(safe_weights_path):
            return safe_weights_path

        weights_index_path = os.path.join(checkpoint, WEIGHTS_INDEX_NAME)
        if os.path.isfile(weights_index_path):
            return weights_index_path

        weights_path = os.path.join(checkpoint, WEIGHTS_NAME)
        if os.path.isfile(weights_path):
            return weights_path

        raise ValueError(
            f"Could not find any model weights in {checkpoint}. "
            f"Expected one of the following files: {SAFE_WEIGHTS_INDEX_NAME}, {SAFE_WEIGHTS_NAME}, "
            f"{WEIGHTS_INDEX_NAME}, or {WEIGHTS_NAME}."
        )

    def _prefetch_checkpoint(self, model_name_or_path: str, skip_prefetch_weights: bool) -> str:
        """Prefetch checkpoint from a HF repo if needed.

        We support the native HF/torch formats in the following order of precedence:

            1. safetensors
            2. pytorch_model.bin
        """
        # check if it's a repo id and if so download the repo
        is_hf_repo = True
        try:
            validate_repo_id(model_name_or_path)
        except HFValidationError:
            is_hf_repo = False
        if is_hf_repo:
            ad_logger.info("Pre-fetching checkpoint directory from HF repo.")
            ignore_patterns = self._get_ignore_patterns(model_name_or_path, skip_prefetch_weights)
            fetched_dir = snapshot_download(model_name_or_path, ignore_patterns=ignore_patterns)
        else:
            fetched_dir = model_name_or_path

        # at this point it should be a directory (either the original one or the download dir)
        assert os.path.isdir(fetched_dir), f"Checkpoint path {fetched_dir} is not a directory."

        self._load_quantization_config(fetched_dir)

        return fetched_dir

    def _load_checkpoint(
        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
    ):
        """Load the checkpoint into the model."""
        # identify the most relevant checkpoint file
        ckpt_file = self._get_checkpoint_file(self.model)

        load_handle = model.register_load_state_dict_pre_hook(self._remap_param_names_load_hook)
        # Ensure it's the first one.
        model._load_state_dict_pre_hooks.move_to_end(key=load_handle.id, last=False)

        get_handle = model.register_state_dict_post_hook(
            _StateDictParamNameConverter(self._checkpoint_conversion_mapping)
        )
        # Ensure it's the first one.
        model._state_dict_hooks.move_to_end(key=get_handle.id, last=False)

        try:
            if disable_preload:
                # Load checkpoint directly to GPU using accelerate's load_checkpoint_in_model (no CPU preload)
                ad_logger.info(
                    "disable_preload=True: Using accelerate's load_checkpoint_in_model (no CPU preload)"
                )
                with hf_load_state_dict_with_device(device):
                    load_checkpoint_in_model(model, checkpoint=ckpt_file, full_state_dict=False)
            else:
                # Preload checkpoint files to CPU
                ad_logger.info("Preloading checkpoint files to CPU")
                self._load_checkpoint_with_preload(model, ckpt_file, device)
        finally:
            load_handle.remove()
            get_handle.remove()

    def _load_checkpoint_with_preload(
        self, model: nn.Module, ckpt_file: str, device: DeviceLikeType
    ):
        all_weights = self._load_full_checkpoint_to_cpu(ckpt_file)

        ad_logger.info(f"Loading weights into model (device: {device})...")
        model.load_state_dict(all_weights, strict=False)

        ad_logger.info("Checkpoint loading completed")

    def _load_full_checkpoint_to_cpu(self, checkpoint: str) -> dict:
        """Load the full checkpoint to CPU memory.

        Args:
            checkpoint: Can be:
                - a path to a file containing a whole model state dict
                - a path to a `.json` file containing the index to a sharded checkpoint
                - a path to a folder containing a unique `.index.json` file and the shards
                - a path to a folder containing a unique pytorch_model.bin or model.safetensors
        """
        checkpoint_files = None
        index_filename = None

        # Fast path: Direct .index.json file (most common case for sharded checkpoints)
        if os.path.isfile(checkpoint):
            if checkpoint.endswith(".index.json"):
                index_filename = checkpoint
            else:
                checkpoint_files = [checkpoint]
        elif os.path.isdir(checkpoint):
            # Check if the whole state dict is present (priority order matches accelerate)
            potential_state_bin = [f for f in os.listdir(checkpoint) if f == WEIGHTS_NAME]
            potential_state_safetensor = [
                f for f in os.listdir(checkpoint) if f == SAFE_WEIGHTS_NAME
            ]

            # Case 1: pytorch_model.bin (WEIGHTS_NAME)
            if len(potential_state_bin) == 1:
                checkpoint_files = [os.path.join(checkpoint, potential_state_bin[0])]
            # Case 2: model.safetensors (SAFE_WEIGHTS_NAME)
            elif len(potential_state_safetensor) == 1:
                checkpoint_files = [os.path.join(checkpoint, potential_state_safetensor[0])]
            else:
                # Case 3: Otherwise check for sharded checkpoints
                potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
                if len(potential_index) == 0:
                    raise ValueError(
                        f"{checkpoint} is not a folder containing a `.index.json` file or a "
                        f"{WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
                    )
                elif len(potential_index) == 1:
                    index_filename = os.path.join(checkpoint, potential_index[0])
                else:
                    raise ValueError(
                        f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                    )
        else:
            raise ValueError(
                f"`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
                f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got "
                f"{checkpoint}."
            )

        # Load checkpoint files from index if needed
        if index_filename is not None:
            checkpoint_folder = os.path.dirname(index_filename)
            with open(index_filename, "r") as f:
                index = json.load(f)

            if "weight_map" in index:
                index = index["weight_map"]
            checkpoint_files = list(set(index.values()))
            checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

        # Load all weights
        all_weights = {}
        for checkpoint_file in checkpoint_files:
            ad_logger.info(f"Loading weight file: {checkpoint_file}")
            if checkpoint_file.endswith(".safetensors"):
                file_weights = safetensors.torch.load_file(checkpoint_file, device="cpu")
            elif checkpoint_file.endswith((".bin", ".pth")):
                file_weights = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            else:
                raise ValueError(f"Unsupported checkpoint format: {checkpoint_file}")

            all_weights.update(file_weights)

        return all_weights

    def _load_quantization_config(self, fetched_dir: str):
        """Load the quantization config from the model directory if not done already."""
        if self._quant_config_reader is not None:
            return
        result = autodetect_quant_config_reader(fetched_dir)
        if result is None:
            return
        reader, extra_model_kwargs = result
        if reader is not None:
            self._quant_config_reader = reader
            self.model_kwargs = deep_merge_dicts(self.model_kwargs, extra_model_kwargs)

    def _remap_param_names_load_hook(self, model, state_dict, *args, **kwargs) -> None:
        """Hook to handle potential param name conversions.

        Some models' transformers implementation can change in between when safetensors were produced
        and / or uploaded to HuggingFace hub. This hook applies the mapping (when present) to reflect
        these differences.
        """
        conversion_mapping = self._checkpoint_conversion_mapping
        if conversion_mapping:
            keys_to_process = list(state_dict.keys())
            for key in keys_to_process:
                new_key = key
                for pattern, replacement in conversion_mapping.items():
                    new_key = re.sub(pattern, replacement, new_key)

                if new_key != key:
                    state_dict[new_key] = state_dict.pop(key)

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        return [FullModelExportInfo()]

    @classmethod
    def register_custom_model_cls(
        cls, config_cls_name: str, custom_model_cls: Type[AutoModelForCausalLM]
    ) -> None:
        """Register a custom model implementation.

        This is useful when the default `AutoModelForCausalLM` is not the one we want to use. For
        example, when the model's code is in a HuggingFace repo that is out of date, or has
        dependencies that TensorRT-LLM does not have, etc.

        Args:
            config_cls_name: This should be the model's config class definition's `__name__` attribute.
            custom_model_cls: The `AutoModelForCausalLM` implementation that should be used for
                `model_type`.
        """
        cls._custom_model_mapping[config_cls_name] = custom_model_cls

    def __init_subclass__(cls, **kwargs):
        """Hook when child classes are defined."""
        cls._custom_model_mapping = {}


class _StateDictParamNameConverter:
    """Helper class for applying param name conversions to a state dict.

    The reason this is a class instead of a method of factory like `_remap_param_names_load_hook`
    is because PyTorch tries to set an `_from_public_api` attribute on hooks, and bound instance
    methods cannot have attributes set on them without major hacks.
    """

    def __init__(self, conversion_mapping: Optional[Dict[str, str]]):
        conversion_mapping = conversion_mapping or {}

        # NOTE: most of the code in this class is forked from `PreTrainedModel.save_pretrained`.
        reverse_key_mapping = {v: k for k, v in conversion_mapping.items()}
        self._mapping = reverse_key_mapping

    def __call__(self, module, state_dict, *args, **kwargs) -> None:
        """Hook to handle potential param name conversions.

        For the same reasons as the `load` hook, we define one to for `state_dict`. This is to silence
        potentially misleading warnings about certain parameter names not being used, because the
        `accelerate` library's logic for determining which keys are unexpected bases it on the keys
        in the `module.state_dict()` return value, not on what `module.load_state_dict()` returns.
        """
        if self._mapping:
            keys_to_process = list(state_dict.keys())
            for key in keys_to_process:
                new_key = key
                for pattern, replacement in self._mapping.items():
                    replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                    replacement = re.sub(r"\(.*\)", "", replacement)
                    new_key, n_replace = re.subn(pattern, replacement, key)
                    # Early exit of the loop
                    if n_replace > 0:
                        break

                if new_key != key:
                    state_dict[new_key] = state_dict.pop(key)


class TextModelExportInfo(SubModuleExportInfo):
    """An export configuration for the text model portion of a VLM."""

    def post_process(self, sub_mod: nn.Module, sub_gm: GraphModule):
        """Post-process the subgraph module and make sure the embedding remains available."""
        # make sure get_input_embeddings function is available in the graph module
        embed_tokens = sub_mod.get_input_embeddings()
        sub_gm.get_input_embeddings = types.MethodType(
            sub_mod.get_input_embeddings.__func__, sub_gm
        )

        # retrieve+replicate expected submodule hierarchy for where the embedding module is located
        for embed_name, subsubmod in sub_mod.named_modules():
            if subsubmod is embed_tokens:
                break
        else:
            raise RuntimeError(
                "Could not find embedding module in model. Expected embedding module to be a "
                "submodule of the text submodule."
            )
        sub_gm.set_submodule(embed_name, embed_tokens)

        # add a dummy node to the graph for making the embedding module impure --> impure nodes
        # won't be deleted from the graph during cleanup and this way we ensure that the embedding
        # module is not deleted from the GraphModule either.
        # TODO (lucaslie): is there a better way to make the embedding module "sticky"?
        n_embed_tokens = sub_gm.graph.get_attr(f"{embed_name}.weight")
        sub_gm.graph.call_function(
            torch._assert, args=(n_embed_tokens, "Avoid embedding getting deleted from graph.")
        )

    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        batch_size_dynamic = Dim.DYNAMIC
        seq_len_dynamic = Dim.DYNAMIC
        return {
            "input_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
            "inputs_embeds": {0: batch_size_dynamic, 1: seq_len_dynamic},
            "position_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
        }

    @classmethod
    def from_autoinferred(cls, model: nn.Module) -> "TextModelExportInfo":
        """Create an export configuration from the model by auto-inferring the text submodule.

        model:
            The full model (AutoModelForImageTextToText)

        Returns:
            An export configuration for the text submodule with the right submodule key.

        The text submodule is being auto-discovered by looking at the first submodule that contains
        the ``text_config`` instead of the full config object.
        """
        # retrieve expected text_config class
        text_config_cls = type(model.config.text_config)

        # heuristic to identify the text submodule
        submodule_key = None
        for name, submodule in model.named_modules():
            if isinstance(getattr(submodule, "config", None), text_config_cls):
                submodule_key = name
                break

        if submodule_key is None:
            raise ValueError(
                "Could not find text submodule in model. Expected text submodule to have a config "
                f"object of type {text_config_cls}."
            )

        return cls(submodule_key)


@ModelFactoryRegistry.register("AutoModelForImageTextToText")
class AutoModelForImageTextToTextFactory(AutoModelForCausalLMFactory):
    _model_defaults = {
        "text_config": {
            "use_cache": False,
        },
    }

    def _set_sharding_config(self, model_config: PretrainedConfig):
        """Override the sharding config for the model with text_config."""
        super()._set_sharding_config(model_config)

        if hasattr(model_config, "text_config"):
            text_config = model_config.text_config
            if hasattr(text_config, "base_model_tp_plan"):
                self._sharding_config["tp_plan"] = text_config.base_model_tp_plan

    @property
    def automodel_cls(self) -> Type[_BaseAutoModelClass]:
        return AutoModelForImageTextToText

    def init_tokenizer(self) -> Optional[Any]:
        """Initialize the tokenizer—either a custom name or the model's default."""
        processor = self.init_processor()
        if processor is None:
            return None
        return processor.tokenizer

    def init_processor(self) -> Optional[Any]:
        """Initialize the processor for the model."""
        if self.tokenizer is None:
            return None
        return AutoProcessor.from_pretrained(self.tokenizer, **self.tokenizer_kwargs)

    # NOTE: for now we only export text_model - hence using the default example_inputs is
    # sufficient. Leaving the logic below for future reference as a special function called
    # `get_example_inputs_with_images`. It's also used in unit tests at the moment.
    def get_example_inputs_with_images(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of example inputs for the model with images."""

        def _prep_seq(text, img1, img2):
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img1},
                        {"type": "image", "image": img2},
                        {"type": "text", "text": text},
                    ],
                }
            ]

        # Create a batch of conversations (batch_size = 2).
        # Note that we explicitly use 2 images in the examples to avoid potential shape specialization(s)
        # in `torch.compile` / `torch.export`.
        batch_messages = [
            _prep_seq(
                "Describe what you see in the two images and their differences.",
                Image.new("RGB", self._example_image_dims, color=(128, 128, 128)),
                Image.new("RGB", self._example_image_dims, color=(64, 64, 64)),
            ),
            _prep_seq(
                "What are the main differences between these two images?",
                Image.new("RGB", self._example_image_dims, color=(255, 0, 0)),
                Image.new("RGB", self._example_image_dims, color=(0, 255, 0)),
            ),
        ]

        processor = self.init_processor()
        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            return_attention_mask=False,
        )

        # We should have no need for the attention mask, and it can actually cause issues in
        # downstream code.
        inputs.pop("attention_mask", None)

        # NOTES:
        # 1. `inputs` is dict-like, but not a dict (hence the dict unpacking below).
        # 2. Although `get_extra_inputs` allows implementations to specify "extra inputs", the example
        #    values still need to be returned by `get_example_inputs`.
        return {**inputs}

    @property
    def _example_image_dims(self) -> Tuple[int, int]:
        # Some specializations (children) of this class may override this if their models have
        # assumptions on the image dimensions. For example, they may have a lower bound due to
        # the patch size they use.
        return (64, 64)

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        return [TextModelExportInfo.from_autoinferred(model)]
