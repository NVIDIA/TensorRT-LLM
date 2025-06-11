"""Interface to initialize and load HF models."""

import json
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
from ..utils.logger import ad_logger
from .factory import ModelFactory, ModelFactoryRegistry


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
    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer_kwargs.setdefault("trust_remote_code", True)
        self._quant_config = None

        # NEVER use cache
        self.model_kwargs["use_cache"] = False

        # special handling for torch_dtype in model_kwargs since HF does not correctly update
        # torch_dtype string to an actual torch.dtype object (only with default)
        if "torch_dtype" in self.model_kwargs:
            dtype = self.model_kwargs["torch_dtype"]
            if isinstance(dtype, str):
                dtype = getattr(torch, self.model_kwargs["torch_dtype"])
            assert isinstance(dtype, torch.dtype), f"Invalid dtype: {dtype}"
            self.model_kwargs["torch_dtype"] = dtype

        # prefetch the model+checkpoint
        self.prefetch_checkpoint()
        # load the quantization config
        self._load_quantization_config()

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
        Recursively update a PretrainedConfig object with values from update_dict.

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

    def build_model(self, device: DeviceLikeType) -> nn.Module:
        """Build the model on the desired device."""
        # We only support fp16 to fp4 conversion.
        if self._quant_config and self._quant_config.get("quant_algo", None) == "NVFP4":
            self.model_kwargs["torch_dtype"] = torch.half

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

        # patch forward method
        model.forward = types.MethodType(self._simple_forward, model)

        model.eval()
        return model

    def get_quant_config(self) -> Dict:
        return self._quant_config or {}

    def get_cache_config(self):
        """Setup cache information based on quantization information."""
        if self._quant_config is not None and "kv_cache_quant_algo" in self._quant_config.keys():
            kv_cache_format = self._quant_config.get("kv_cache_quant_algo", None)
            if kv_cache_format is not None:
                assert kv_cache_format == "FP8", (
                    f"KV cache quantization format {kv_cache_format} is not supported."
                )
            kv_cache_dtype = torch.float8_e4m3fn if kv_cache_format is not None else None
        else:
            kv_cache_dtype = None
        return CacheConfig(dtype=kv_cache_dtype)

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

        return fetched_dir

    def _load_checkpoint(self, model: nn.Module, device: DeviceLikeType):
        """Load the checkpoint into the model."""
        # check if we skip loading weights
        if self.skip_loading_weights:
            return

        # prefetch if needed
        self.prefetch_checkpoint()

        # identify the most relevant checkpoint file
        ckpt_file = self._get_checkpoint_file(self.model)
        # reuse the load checkpoint utility from accelerate
        with hf_load_state_dict_with_device(device):
            load_checkpoint_in_model(model, checkpoint=ckpt_file)

    def _load_quantization_config(self):
        assert self.model
        hf_quant_config_file = os.path.join(self.model, "hf_quant_config.json")
        if os.path.exists(hf_quant_config_file):
            with open(hf_quant_config_file, "r") as file:
                quantization_config = json.load(file)
                assert quantization_config.get("producer", {}).get("name", None) == "modelopt", (
                    "Only support modelopt quantized checkpoint"
                )
                self._quant_config = quantization_config.get("quantization", {})

                # We do not quantize lm_head.
                if "exclude_modules" not in self._quant_config:
                    self._quant_config["exclude_modules"] = ["lm_head"]


@ModelFactoryRegistry.register("AutoModelForImageTextToText")
class AutoModelForImageTextToTextFactory(AutoModelForCausalLMFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # additional heuristic to propagate "important keys"
        # TODO (lucaslie): WAR until we have better support on dashboard to control model_kwargs
        keys_to_propagate = [
            "num_hidden_layers",
            "max_position_embeddings",
            "use_cache",
            "torch_dtype",
        ]
        self.model_kwargs["text_config"] = self.model_kwargs.get("text_config", {})
        for key in keys_to_propagate:
            if key in self.model_kwargs:
                self.model_kwargs["text_config"][key] = self.model_kwargs[key]

    @property
    def automodel_from_config(self):
        return AutoModelForImageTextToText.from_config
