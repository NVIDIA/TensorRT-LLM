"""Interface to initialize and load HF models."""

import json
import os
import types
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import modeling
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from torch._prims_common import DeviceLikeType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from ..custom_ops.attention_interface import CacheConfig
from ..utils.logger import ad_logger
from .factory import ModelFactory, ModelFactoryRegistry


@contextmanager
def load_state_dict_with_assign():
    """
    Context manager that temporarily patches torch.nn.Module.load_state_dict
    to use assign=True, which directly replaces parameters in the model with those
    from the loaded state_dict, maintaining their data type and device placement.
    """
    # Save the original load_state_dict method
    original_load_state_dict = torch.nn.Module.load_state_dict

    # Define and apply the patched version
    def load_state_dict_with_assign(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        return original_load_state_dict(self, state_dict, strict=strict, assign=True)

    # Apply the patch
    torch.nn.Module.load_state_dict = load_state_dict_with_assign

    try:
        # Allow the context body to execute
        yield
    finally:
        # Restore the original method, even if an exception occurred
        torch.nn.Module.load_state_dict = original_load_state_dict


@contextmanager
def hf_load_state_dict_with_device(device: DeviceLikeType):
    """Patch HF load_state_dict to use provided device."""
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
        self._quant_config = None

        # heuristic to disable use_cache
        self.model_kwargs["use_cache"] = False

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
        """Initialize the tokenizer for the model."""
        if not self.model:
            return None
        return self.autotokenizer_from_pretrained(self.model, **self.tokenizer_kwargs)

    def prefetch_checkpoint(self):
        """Prefetch checkpoint from a HF repo if needed."""
        # already prefetched
        if self._prefetched_path:
            return

        # check if it's a repo id and if so download the repo
        is_hf_repo = True
        try:
            validate_repo_id(self.model)
        except HFValidationError:
            is_hf_repo = False
        if is_hf_repo:
            # we don't expect to use bin files or pt/pth checkpoint files (they are quite large)
            ignore_patterns = ["*.bin", "*.pt", "*.pth"]
            # we will also ignore the .safetensors files if we skip loading weights
            if self.skip_loading_weights:
                ignore_patterns.append("*.safetensors")
            ad_logger.info("Pre-fetching checkpoint directory from HF repo.")
            fetched_dir = snapshot_download(self.model, ignore_patterns=ignore_patterns)
        else:
            fetched_dir = self.model

        # at this point it should be a directory (either the original one or the download dir)
        assert os.path.isdir(fetched_dir), f"Checkpoint path {fetched_dir} is not a directory."

        self._prefetched_path = fetched_dir

    def _load_checkpoint(self, model: nn.Module, device: DeviceLikeType):
        """Load the checkpoint into the model."""
        # check if we skip loading weights
        if self.skip_loading_weights:
            return

        # prefetch if needed
        self.prefetch_checkpoint()

        # reuse the load checkpoint utility from accelerate
        with load_state_dict_with_assign(), hf_load_state_dict_with_device(device):
            load_checkpoint_and_dispatch(model, checkpoint=self.model)

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
