"""Interface to initialize and load HF models."""

import os
import types
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_in_model
from accelerate.utils import modeling
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HFValidationError, filter_repo_objects, validate_repo_id
from torch._prims_common import DeviceLikeType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from ..custom_ops.attention_interface import CacheConfig
from ..utils._config import deep_merge_dicts
from ..utils.logger import ad_logger
from .factory import ModelFactory, ModelFactoryRegistry, ShardingConfigSource
from .quant_config_reader import QuantConfigReader, QuantConfigReaderRegistry


@contextmanager
def hf_load_state_dict_with_device(device: DeviceLikeType):
    """Patch HF load_state_dict to use provided device.

    NOTE (lucaslie): this function is called by ``load_checkpoint_in_model``. We provide the device
    map here as a patch instead of going through ``load_checkpoint_in_model``. This is because
    otherwise ``load_checkpoint_in_model`` will execute its own state_dict loading logic instead of
    calling ``nn.Module.load_state_dict``. However, we rely on the state dict loading hooks in
    ``nn.Module.load_state_dict`` to correctly load the weights. By providing the device map here,
    we can ensure that ``load_checkpoint_in_model`` will call ``nn.Module.load_state_dict``.
    """
    # save the original load_state_dict method
    original_load_state_dict = modeling.load_state_dict

    # Define and apply the patched version
    def load_state_dict_with_device(checkpoint_file, device_map=None):
        return original_load_state_dict(checkpoint_file, device_map={"": device})

    # Apply the patch
    modeling.load_state_dict = load_state_dict_with_device

    try:
        yield
    finally:
        # Restore the original method, even if an exception occurred
        modeling.load_state_dict = original_load_state_dict


@ModelFactoryRegistry.register("AutoModelForCausalLM")
class AutoModelForCausalLMFactory(ModelFactory):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._quant_config_reader: QuantConfigReader | None = None
        # Ingest defaults for tokenizer and model kwargs
        self.tokenizer_kwargs = deep_merge_dicts(self._tokenizer_defaults, self.tokenizer_kwargs)
        self.model_kwargs = deep_merge_dicts(
            self._model_defaults,
            self.model_kwargs,
        )

        # special handling for torch_dtype in model_kwargs since HF does not correctly update
        # torch_dtype string to an actual torch.dtype object (only with default)
        if "torch_dtype" in self.model_kwargs:
            dtype = self.model_kwargs["torch_dtype"]
            if isinstance(dtype, str):
                dtype = getattr(torch, self.model_kwargs["torch_dtype"])
            assert isinstance(dtype, torch.dtype), f"Invalid dtype: {dtype}"
            self.model_kwargs["torch_dtype"] = dtype

        # set sharding config source to huggingface
        self._sharding_config["source"] = ShardingConfigSource.HUGGINGFACE

    @property
    def autoconfig_from_pretrained(self):
        return AutoConfig.from_pretrained

    @property
    def autotokenizer_from_pretrained(self):
        return AutoTokenizer.from_pretrained

    # TODO (@lucaslie): Do we ever want to switch to from_pretrained?
    @property
    def automodel_from_config(self):
        return AutoModelForCausalLM.from_config

    @staticmethod
    def _simple_forward(model: nn.Module, input_ids: torch.Tensor, position_ids: torch.Tensor):
        """A simple forward pass for the model to functionalize the args.

        This follows the standard function signature as expected by factory.py.
        """
        return type(model).forward(model, input_ids=input_ids, position_ids=position_ids)

    def _recursive_update_config(self, config: PretrainedConfig, update_dict: Dict[str, Any]):
        """
        Deep-merge a PretrainedConfig object with values from update_dict.

        Args:
            config: PretrainedConfig object to update
            update_dict: Dictionary with values to update in the config

        Returns:
            The updated PretrainedConfig object
        """
        for key, value_new in update_dict.items():
            # Check if the key exists in config
            if not hasattr(config, key):
                continue

            target_value = getattr(config, key)

            # Handle nested PretrainedConfig objects...
            if isinstance(value_new, dict) and isinstance(target_value, PretrainedConfig):
                # Recursively update nested configs
                updated_value = self._recursive_update_config(target_value, value_new)
                setattr(config, key, updated_value)
            else:
                # Direct update for simple values
                setattr(config, key, value_new)

        return config

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        """Build the model on the desired device."""

        # NOTE (lucaslie): HF doesn't recursively update nested PreTrainedConfig objects. Instead,
        # the entire subconfig will be overwritten.
        # we want to recursively update model_config from model_kwargs here.
        model_config = self.autoconfig_from_pretrained(self.model, trust_remote_code=True)
        model_config = self._recursive_update_config(model_config, self.model_kwargs)

        with (init_empty_weights if device == "meta" else nullcontext)():
            model = self.automodel_from_config(model_config, trust_remote_code=True)

        # post-init --> this must be called explicitly for HF models the way we initialize them since
        # this "gets lost" with the init_empty_weights context manager.
        if hasattr(model, "post_init"):
            model.post_init()

        # if present, initialize sharding config. We need head_dim for colwise sharding.
        self._set_sharding_config(model.config)

        # patch forward method
        model.original_forward = model.forward
        model.forward = types.MethodType(self._simple_forward, model)

        model.eval()

        return model

    def _set_sharding_config(self, model_config: PretrainedConfig):
        """Set the sharding config for the model."""
        self._sharding_config["head_dim"] = 1
        if hasattr(model_config, "base_model_tp_plan"):
            self._sharding_config["tp_plan"] = model_config.base_model_tp_plan
        if hasattr(model_config, "head_dim") and model_config.head_dim is not None:
            self._sharding_config["head_dim"] = model_config.head_dim
        elif hasattr(model_config, "hidden_size") and hasattr(model_config, "num_attention_heads"):
            self._sharding_config["head_dim"] = (
                model_config.hidden_size // model_config.num_attention_heads
            )
        if hasattr(model_config, "num_hidden_layers"):
            self._sharding_config["num_hidden_layers"] = model_config.num_hidden_layers

    def get_quant_config(self) -> Dict:
        """Returns the quantization config for this model or an empty dict if not quantized."""
        if self._quant_config_reader is not None:
            return self._quant_config_reader.get_config()
        return {}

    def get_cache_config(self):
        """Return kv cache dtype configuration."""
        if not self._quant_config_reader:
            return CacheConfig(dtype=None)

        kv_cache_dtype = self._quant_config_reader.get_config().get("kv_cache_dtype")
        torch_dtype = torch.float8_e4m3fn if kv_cache_dtype == "float8_e4m3fn" else None
        assert torch_dtype in (torch.float8_e4m3fn, None), (
            f"Unsupported dtype: {torch_dtype}. Only torch.float8_e4m3fn is supported."
        )

        return CacheConfig(dtype=torch_dtype)

    def init_tokenizer(self) -> Optional[Any]:
        """Initialize the tokenizer—either a custom name or the model's default."""
        if self.tokenizer is None:
            return None
        return self.autotokenizer_from_pretrained(self.tokenizer, **self.tokenizer_kwargs)

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

    def _load_checkpoint(self, model: nn.Module, device: DeviceLikeType):
        """Load the checkpoint into the model."""
        # identify the most relevant checkpoint file
        ckpt_file = self._get_checkpoint_file(self.model)
        # reuse the load checkpoint utility from accelerate
        with hf_load_state_dict_with_device(device):
            # Set `full_state_dict=False` to skip Accelerate's FSDP weight sync logic.
            # Internally, load_checkpoint_in_model → set_model_state_dict → _load_model_state_dict,
            # which collects local model params, syncs weights from checkpoint, and applies them via
            # model.load_state_dict.
            # This sync step can interfere with load_hooks by mixing raw checkpoint weights and
            # model-transformed weights,leading to unexpected key mismatches or format issues.
            load_checkpoint_in_model(model, checkpoint=ckpt_file, full_state_dict=False)

    def _load_quantization_config(self, fetched_dir: str):
        """Load the quantization config from the model directory if not done already."""
        if self._quant_config_reader is not None:
            return
        # TODO: specified by user or auto-detect
        reader_cls = QuantConfigReaderRegistry.get("modelopt")
        result = reader_cls.from_file(fetched_dir)
        if result is None:
            return
        reader, extra_model_kwargs = result

        if reader is not None:
            self._quant_config_reader = reader
            self.model_kwargs = deep_merge_dicts(self.model_kwargs, extra_model_kwargs)


@ModelFactoryRegistry.register("AutoModelForImageTextToText")
class AutoModelForImageTextToTextFactory(AutoModelForCausalLMFactory):
    _model_defaults = {
        "use_cache": False,
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
            if hasattr(text_config, "head_dim"):
                self._sharding_config["head_dim"] = text_config.head_dim
            if hasattr(text_config, "num_hidden_layers"):
                self._sharding_config["num_hidden_layers"] = text_config.num_hidden_layers

    @property
    def automodel_from_config(self):
        return AutoModelForImageTextToText.from_config
