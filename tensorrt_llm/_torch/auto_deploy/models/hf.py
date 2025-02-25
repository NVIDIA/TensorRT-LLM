"""Interface to initialize and load HF models."""

import json
import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from torch._prims_common import DeviceLikeType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict

from ..utils.logger import ad_logger
from .factory import ModelFactory, ModelFactoryRegistry


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
        self.quantization = None

        if self.ckpt_path:
            # prefetch if needed
            self.prefetch_checkpoint()

            self._load_quantization_config()

    def build_model(self, device: DeviceLikeType) -> nn.Module:
        """Build the model on the desired device."""
        # We only support fp16 to fp4 conversion.
        if self.quantization and self.quantization.get("quant_algo", None) == "NVFP4":
            self.model_kwargs["torch_dtype"] = torch.half

        model_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=True, **self.model_kwargs
        )
        get_model_from_config = AutoModelForCausalLM.from_config

        with (init_empty_weights if device == "meta" else nullcontext)():
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(model_config.torch_dtype)
            model = get_model_from_config(model_config, trust_remote_code=True)
            torch.set_default_dtype(default_dtype)

        # post-init --> this must be called explicitly for HF models the way we initialize them since
        # this "gets lost" with the init_empty_weights context manager.
        if hasattr(model, "post_init"):
            model.post_init()

        model.eval()
        return model

    def get_quantization(self) -> Optional[Dict]:
        return self.quantization

    def get_positional_encoding_config(self, model: nn.Module) -> Optional[Dict[str, Any]]:
        """Return the positional encoding configuration for the model."""
        return {
            "rope_theta": getattr(getattr(model, "config", {}), "rope_theta", None),
        }

    def init_tokenizer(self) -> Optional[Any]:
        """Initialize the tokenizer for the model."""
        if not self.ckpt_path:
            return None
        return AutoTokenizer.from_pretrained(self.ckpt_path, **self.tokenizer_kwargs)

    def prefetch_checkpoint(self):
        """Prefetch checkpoint from a HF repo if needed."""
        # already prefetched
        if self._prefetched_path:
            return
        # nothing to fetch since no ckpt_path
        if not self._ckpt_path:
            return

        # check if it's a repo id and if so download the repo
        is_hf_repo = True
        try:
            validate_repo_id(self._ckpt_path)
        except HFValidationError:
            is_hf_repo = False
        if is_hf_repo:
            # we don't expect to use bin files in this context and they are quite large.
            ad_logger.info("Pre-fetching checkpoint directory from HF repo.")
            fetched_dir = snapshot_download(
                self._ckpt_path, ignore_patterns=["**pytorch_model*.bin*", "**.pt"]
            )
        else:
            fetched_dir = self._ckpt_path

        # at this point it should be a directory (either the original one or the download dir)
        assert os.path.isdir(fetched_dir), f"Checkpoint path {fetched_dir} is not a directory."

        self._prefetched_path = fetched_dir

    def _load_checkpoint(self, model, **kwargs):
        """Load the checkpoint into the model."""
        # check for ckpt_path
        if not self.ckpt_path:
            return

        # prefetch if needed
        self.prefetch_checkpoint()

        ckpt_path = self.ckpt_path

        # sharded checkpoint
        if os.path.isfile(os.path.join(ckpt_path, "model.safetensors.index.json")):
            _to_maybe_empty(model, device="cpu")
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

        model.load_state_dict(state_dict, strict=False, assign=True)

    def _load_quantization_config(self):
        assert self.ckpt_path
        hf_quant_config_file = os.path.join(self.ckpt_path, "hf_quant_config.json")
        if os.path.exists(hf_quant_config_file):
            with open(hf_quant_config_file, "r") as file:
                quantization_config = json.load(file)
                assert quantization_config.get("producer", {}).get("name", None) == "modelopt", (
                    "Only support modelopt quantized checkpoint"
                )
                self.quantization = quantization_config.get("quantization", {})

                # We do not quantize lm_head.
                if "exclude_modules" not in self.quantization:
                    self.quantization["exclude_modules"] = ["lm_head"]
