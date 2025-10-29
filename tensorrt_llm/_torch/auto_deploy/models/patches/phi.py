"""
Patch RotaryEmbedding implementations in Phi3/Phi4 models for torch.export compatibility.

This module fixes two issues that break model export and state dict reloading:

1. Phi3RotaryEmbedding:
   Registers `inv_freq` as an empty buffer, leading to missing keys when loading the state dict.

2. Phi3LongRoPEScaledRotaryEmbedding:
   Dynamically initializes `short_factor` and `long_factor` during forward passes, which is incompatible
   with torch.export.

These patches move the initialization of `inv_freq` and `short_factor` into the class constructor and simplify LongRoPE
to always use the `short_factor` path (removing dynamic updates).
"""

import math
import re
import types

import torch
from transformers import AutoModelForCausalLM

_from_config_original = AutoModelForCausalLM.from_config


# Additional steps in init phase to calculate and register inv_freq
# Needed by microsoft/Phi-3-mini-128k-instruct, microsoft/Phi-3-medium-128k-instruct,
# microsoft/Phi-3-mini-4k-instruct, microsoft/Phi-3-medium-4k-instruct
def _patched_phi3_emb_init(
    self,
):
    dev = torch.device("cpu")
    inv = 1.0 / (
        self.base
        ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=dev).float() / self.dim)
    )
    self._buffers.pop("inv_freq", None)
    self.register_buffer("inv_freq", inv, persistent=False)


# copied from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py#L122
# with an additional line: self.inv_freq = self.inv_freq.to(x.device)
@torch.no_grad()
def _patch_phi3_emb_forward(self, x, position_ids, seq_len=None):
    self.inv_freq = self.inv_freq.to(x.device)
    # x: [bs, num_attention_heads, seq_len, head_size]
    if self.inv_freq is None:
        self.inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
        )
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 since bfloat16 loses precision on long contexts
    # See https://github.com/huggingface/transformers/pull/29285
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Additional steps in init phase to calculate and register ext_factors
# Needed by microsoft/Phi-3-mini-128k-instruct, microsoft/Phi-3-medium-128k-instruct
def _patched_phi3_long_emb_init(
    self,
):
    _patched_phi3_emb_init(self)
    self.ext_factors = torch.tensor(
        self.short_factor, dtype=torch.float32, device=torch.device("cpu")
    )


# Copied from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py#L151
# with ext_factors calculation logic simplified
# Needed by microsoft/Phi-3-mini-128k-instruct, microsoft/Phi-3-medium-128k-instruct
@torch.no_grad()
def _patch_phi3_long_emb_forward(self, x, position_ids, seq_len=None):
    self.ext_factors = self.ext_factors.to(x.device)

    inv_freq_shape = (
        torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim
    )
    self.inv_freq = 1.0 / (self.ext_factors * self.base**inv_freq_shape)

    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    # Force float32 since bfloat16 loses precision on long contexts
    # See https://github.com/huggingface/transformers/pull/29285
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )

        cos = emb.cos() * scaling_factor
        sin = emb.sin() * scaling_factor
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Simplify https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/modeling_phi3.py#L383
# needed by microsoft/Phi-4-mini-instruct
def _patch_phi4_long_rope_update(self, position_ids, device):
    self.original_inv_freq = self.original_inv_freq.to(device)
    self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)


# Remove the @dynamic_rope_update decorator from transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding.forward
# needed by microsoft/Phi-4-mini-reasoning
@torch.no_grad()
def _patch_phi3_emb_with_decorator_forward(self, x, position_ids):
    inv_freq_expanded = (
        self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = (
        x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    if re.search(r"Phi-4-mini-instruct", getattr(config, "_name_or_path", "")):
        for _, module in model.named_modules():
            name = type(module).__name__
            if name == "Phi3RotaryEmbedding":
                module._longrope_frequency_update = types.MethodType(
                    _patch_phi4_long_rope_update, module
                )
        return model
    if re.search(r"Phi-4-mini-reasoning", getattr(config, "_name_or_path", "")):
        for _, module in model.named_modules():
            name = type(module).__name__
            if name == "Phi3RotaryEmbedding":
                module.forward = types.MethodType(_patch_phi3_emb_with_decorator_forward, module)
        return model
    # handling: microsoft/Phi-3-mini-128k-instruct, microsoft/Phi-3-medium-128k-instruct
    # microsoft/Phi-3-mini-4k-instruct, microsoft/Phi-3-medium-4k-instruct
    # microsoft/Phi-3.5-mini-instruct
    pattern = (
        r"Phi-(?:3|3\.5)-"
        r"(?:mini|medium)"
    )
    if not re.search(pattern, getattr(config, "_name_or_path", "")):
        return model
    for _, module in model.named_modules():
        name = type(module).__name__
        if name == "Phi3RotaryEmbedding":
            _patched_phi3_emb_init(module)
            module.forward = types.MethodType(_patch_phi3_emb_forward, module)
        elif name in ("Phi3LongRoPEScaledRotaryEmbedding", "Phi3SuScaledRotaryEmbedding"):
            _patched_phi3_long_emb_init(module)
            module.forward = types.MethodType(_patch_phi3_long_emb_forward, module)
    return model


# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
