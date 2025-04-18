"""Interface to initialize and load HF models."""

import json
import os
import types
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from torch._prims_common import DeviceLikeType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    load_sharded_checkpoint,
    load_state_dict,
)

from ..custom_ops.attention_interface import CacheConfig, PositionalEmbeddingConfig
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


# a patch for sdpa_attention_forward to appear as single node in the graph
# Save the original function before we override it
original_sdpa_attention_forward = sdpa_attention_forward


@torch.library.custom_op("sdpa::attention", mutates_args=())
def sdpa_attention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    num_key_value_groups: Optional[int] = None,
) -> torch.Tensor:
    """A wrapper for the SDPA attention operation to appear as a single node in the graph."""
    # Create a temporary module with the num_key_value_groups attribute for the original function
    temp_module = nn.Module()
    if num_key_value_groups is not None:
        temp_module.num_key_value_groups = num_key_value_groups

    output, _ = original_sdpa_attention_forward(
        module=temp_module,
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
    )
    return output


@sdpa_attention_op.register_fake
def sdpa_attention_op_fake(
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    num_key_value_groups=None,
):
    """Fake implementation of SDPA attention."""
    return torch.empty_like(query).transpose(1, 2).contiguous()


def wrapped_sdpa_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    """Wrapper function for sdpa_attention_forward that appears as a single node in the graph."""
    # Extract num_key_value_groups from the module if available
    num_key_value_groups = getattr(module, "num_key_value_groups", None)

    # Call our custom op, ignoring other kwargs for simplicity
    return torch.ops.sdpa.attention(
        query, key, value, attention_mask, dropout, scaling, is_causal, num_key_value_groups
    ), None


# TODO (lliebenwein): let's check if we can get rid of this patch and just use a pattern matcher instead...
# TODO: there seems to be two patterns: repeat_kv from HF's sdpa and repeat_interleave Llama-stack
# TODO: we can also build in an escape hatch if we fail to detect repeat_kv and just stick with the
# qkv we found...
sdpa_attention_forward = wrapped_sdpa_attention_forward

# also update the attention interface
ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward


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
        self._pos_embd_config = None

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
        """A simple forward pass for the model to functionalize the args."""
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

        # Store positional embedding config
        hf_config: Dict[str, Any] = getattr(model, "config", object())

        # TODO: let's see how we can support more Rope variants in the future
        self._pos_embd_config = PositionalEmbeddingConfig(
            mode="rope" if hasattr(hf_config, "rope_theta") else None,
            rope_theta=getattr(hf_config, "rope_theta", 0.0),
            rope_scale=1.0,
        )

        # patch forward method
        model.forward = types.MethodType(self._simple_forward, model)

        model.eval()
        return model

    def get_quant_config(self) -> Dict:
        return self._quant_config or {}

    def get_positional_embedding_config(self):
        """Return the positional encoding configuration for the model."""
        assert self._pos_embd_config is not None, "Please call build_model first."

        return self._pos_embd_config

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
