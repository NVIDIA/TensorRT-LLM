import contextlib
import importlib.util
import sys
import types
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM

from tensorrt_llm._torch.auto_deploy.models.patches.bamba import _bamba_mixer_torch_forward


# Forked from:
# https://github.com/state-spaces/mamba/blob/6b32be06d026e170b3fdaf3ae6282c5a6ff57b06/mamba_ssm/ops/triton/layernorm_gated.py
# NOTES:
# 1. At time of writing (09/25/2025), the nano nemotron v2 modeling code expects `mamba_ssm`
#    to be installed so as to be able to make use of its grouped gated RMS norm operation.
#    We therefore replace it with one that uses einops + pytorch.
def _rms_norm_ref(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True
):
    dtype = x.dtype
    # N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


# The original implementation looks at `cache_position[0]` to decide what to do which does not
# play well with export. Plus, we do not want it to be updated anyway.
def _nemotron_h_model_update_mamba_mask(self, attention_mask, cache_position):
    return None


def _nemotron_h_model_update_causal_mask(self, attention_mask, input_tensor, cache_position):
    # Force attention to use causal mode without explicit masks
    return None


def _nemotron_h_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    device = hidden_states.device
    with contextlib.ExitStack() as stack:
        if device.type == "cuda":
            stack.enter_context(torch.cuda.stream(torch.cuda.default_stream(device)))
        # * Use torch.cuda.stream() to avoid NaN issues when using multiple GPUs
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(
                hidden_states, cache_params=cache_params, cache_position=cache_position
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(hidden_states, cache_position=cache_position)
            hidden_states = hidden_states[0]
        elif self.block_type in ["mlp", "moe"]:
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        hidden_states = residual + hidden_states
        return hidden_states


# Note: we assume experts have no bias for now
def _nemotron_h_moe_forward(self, hidden_states: torch.Tensor):
    """
    Uses NemotronH router (returns indices, weights) and dispatches through auto_deploy::torch_moe
    with act_fn='relu2'. Falls back to original forward if any expert has bias.
    """

    residuals = hidden_states
    orig_shape = hidden_states.shape
    topk_indices, topk_weights = self.gate(hidden_states)
    x_flat = hidden_states.view(-1, hidden_states.shape[-1])

    out_flat = torch.ops.auto_deploy.torch_moe(
        x_flat,
        topk_indices,
        topk_weights,
        w1_weight=[e.up_proj.weight for e in self.experts],
        w2_weight=[e.down_proj.weight for e in self.experts],
        w3_weight=[],
        act_fn="relu2",
        mlp_style="mlp",
    )

    out = out_flat.view(*orig_shape)
    out = out + self.shared_experts(residuals)
    return out


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, List[Tuple[str, Callable]]] = {
    "NemotronHMamba2Mixer": [("forward", _bamba_mixer_torch_forward)],
    "NemotronHModel": [
        ("_update_causal_mask", _nemotron_h_model_update_causal_mask),
        ("_update_mamba_mask", _nemotron_h_model_update_mamba_mask),
    ],
    "NemotronHBlock": [("forward", _nemotron_h_block_forward)],
    "NemotronHMOE": [("forward", _nemotron_h_moe_forward)],
}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if (module_name := type(module).__name__) in CUSTOM_MODULE_PATCHES.keys():
            patches = CUSTOM_MODULE_PATCHES[module_name]
            for method_name, method_patch in patches:
                setattr(module, method_name, types.MethodType(method_patch, module))

    return model


# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched

# TODO: figure out how this can be incorporated into the export patch system
# Only patch if the module isn't available
_mamba_ssm_module = "mamba_ssm"
_mamba_ssm_submodule = f"{_mamba_ssm_module}.ops.triton.layernorm_gated"
if importlib.util.find_spec(_mamba_ssm_module) is None:
    stub_mod = types.ModuleType(_mamba_ssm_submodule)
    stub_mod.rmsnorm_fn = _rms_norm_ref
    sys.modules[_mamba_ssm_submodule] = stub_mod
