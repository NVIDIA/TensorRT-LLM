"""Interface to initialize and load HF models."""

import json
import os
import types
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from torch._prims_common import DeviceLikeType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict

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
    def load_state_dict_with_assign(*args, **kwargs):
        return original_load_state_dict(*args, **kwargs, assign=True)

    # Apply the patch
    torch.nn.Module.load_state_dict = load_state_dict_with_assign

    try:
        # Allow the context body to execute
        yield
    finally:
        # Restore the original method, even if an exception occurred
        torch.nn.Module.load_state_dict = original_load_state_dict


def _to_maybe_empty(model: nn.Module, device: DeviceLikeType):
    """A mix of ``model.to(device)`` and ``model.to_empty(device)``.

    If a parameter is already initialized, then we will call `to()` on it. Otherwise, we will
    initialize it with an empty tensor on the given device.

    """
    model._apply(
        lambda t: torch.empty_like(t, device=device)
        if t.device == torch.device("meta")
        else t.to(device)
    )


@ModelFactoryRegistry.register("hf")
class HFFactory(ModelFactory):
    def __init__(
        self,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_kwargs = model_kwargs or {}
        self.model_kwargs["use_cache"] = False
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self._quant_config = None

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

    def build_model(self, device: DeviceLikeType) -> nn.Module:
        """Build the model on the desired device."""
        # We only support fp16 to fp4 conversion.
        if self._quant_config and self._quant_config.get("quant_algo", None) == "NVFP4":
            self.model_kwargs["torch_dtype"] = torch.half

        model_config = self.autoconfig_from_pretrained(
            self.model, trust_remote_code=True, **self.model_kwargs
        )

        with (init_empty_weights if device == "meta" else nullcontext)():
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(model_config.torch_dtype)
            model = self.automodel_from_config(model_config, trust_remote_code=True)
            torch.set_default_dtype(default_dtype)

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
            ignore_patterns = ["**pytorch_model*.bin*", "**.pt", "**.pth"]
            # we will also ignore the .safetensors files if we skip loading weights
            if self.skip_loading_weights:
                ignore_patterns.append("**safetensors")
            ad_logger.info("Pre-fetching checkpoint directory from HF repo.")
            fetched_dir = snapshot_download(self.model, ignore_patterns=ignore_patterns)
        else:
            fetched_dir = self.model

        # at this point it should be a directory (either the original one or the download dir)
        assert os.path.isdir(fetched_dir), f"Checkpoint path {fetched_dir} is not a directory."

        self._prefetched_path = fetched_dir

    def _load_checkpoint(self, model, **kwargs):
        """Load the checkpoint into the model."""
        # check if we skip loading weights
        if self.skip_loading_weights:
            return

        # prefetch if needed
        self.prefetch_checkpoint()

        ckpt_path = self.model

        # sharded checkpoint
        if os.path.isfile(os.path.join(ckpt_path, "model.safetensors.index.json")):
            _to_maybe_empty(model, device="cpu")
            with load_state_dict_with_assign():
                load_sharded_checkpoint(model, ckpt_path, strict=False)
            return

        # look for a single file in the directory ending with .safetensors or .pt/.pth
        safetensors_files = [f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
        torch_files = [f for f in os.listdir(ckpt_path) if f.endswith((".pt", ".pth"))]
        if len(safetensors_files) > 1:
            raise ValueError(f"Multiple .safetensors files in {ckpt_path}: {safetensors_files}")
        elif len(safetensors_files) == 1:
            state_dict = load_state_dict(os.path.join(ckpt_path, safetensors_files[0]))
        elif len(torch_files) > 1:
            raise ValueError(f"Multiple .pt/.pth files found in {ckpt_path}: {torch_files}")
        elif len(torch_files) == 1:
            state_dict = torch.load(os.path.join(ckpt_path, torch_files[0]), **kwargs)
        else:
            raise ValueError(f"No checkpoint found in {ckpt_path}")

        with load_state_dict_with_assign():
            model.load_state_dict(state_dict, strict=False)

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
