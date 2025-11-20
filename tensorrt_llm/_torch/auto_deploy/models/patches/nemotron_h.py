import contextlib
import importlib.util
import sys
import types
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig, AutoModelForCausalLM

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


def _nemotron_h_topk_router_forward(self, hidden_states):
    """
    Forward pass for NemotronHTopkRouter using the optimized noaux_tc_op kernel.

    This replaces the original forward method which used pure PyTorch operations
    with a fused CUDA kernel that performs:
    1. Sigmoid activation of logits
    2. Group-based expert selection
    3. Top-k selection within selected groups
    4. Normalized weight computation
    """
    hidden_states = hidden_states.view(-1, self.config.hidden_size)
    router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

    # Use the fused noaux_tc_op kernel which applies sigmoid internally
    # and performs group-based top-k selection with normalization
    topk_weights, topk_indices = torch.ops.trtllm.noaux_tc_op(
        router_logits,
        self.e_score_correction_bias,
        self.n_group,
        self.topk_group,
        self.top_k,
        self.routed_scaling_factor,
    )

    return topk_indices, topk_weights


# Note: we assume experts have no bias for now
def _nemotron_h_moe_forward(self, hidden_states: torch.Tensor):
    """
    Uses NemotronH router (returns indices, weights) and dispatches through auto_deploy::torch_moe
    with act_fn='relu2'. Handles both latent MOE and direct MOE architectures.
    """

    residuals = hidden_states
    orig_shape = hidden_states.shape
    topk_indices, topk_weights = self.gate(hidden_states)
    x_flat = hidden_states.view(-1, hidden_states.shape[-1])

    # Check if this is a latent MOE (has fc1_latent_proj and fc2_latent_proj)
    has_latent_proj = hasattr(self, "fc1_latent_proj") and hasattr(self, "fc2_latent_proj")

    if has_latent_proj:
        # Latent MOE: project to latent space before routing
        x_flat = self.fc1_latent_proj(x_flat)

    # Route through experts (operates in latent space if latent MOE, full space otherwise)
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

    if has_latent_proj:
        # Latent MOE: project back from latent space
        out_flat = self.fc2_latent_proj(out_flat)

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
    "NemotronHTopkRouter": [("forward", _nemotron_h_topk_router_forward)],
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

_config_from_pretrained_original = AutoConfig.from_pretrained
_nemotron_h_base_model_tp_plan = {
    # mamba SSM layer
    "in_proj": "mamba",
    "out_proj": "rowwise",
    # attention layer
    "q_proj": "colwise",
    "k_proj": "colwise",
    "v_proj": "colwise",
    "o_proj": "rowwise",
    # NOTE: consider not sharding shared experts and/or
    # latent projections at all, keeping them replicated.
    # To do so, comment out the corresponding entries.
    # moe layer: SHARED experts
    "up_proj": "colwise",
    "down_proj": "rowwise",
    # MoLE: latent projections: simple shard
    "fc1_latent_proj": "gather",
    "fc2_latent_proj": "gather",
}


def get_config_from_pretrained_patched(*args, **kwargs):
    ret = _config_from_pretrained_original(*args, **kwargs)
    config = ret[0] if isinstance(ret, tuple) else ret
    # heuristic to check if it's a NemotronH MoE Model
    model_type = getattr(config, "model_type", None)
    num_moe_layers = getattr(config, "layers_block_type", []).count("moe")
    if model_type == "nemotron_h" and num_moe_layers > 0:
        config.base_model_tp_plan = _nemotron_h_base_model_tp_plan
    return (config, *ret[1:]) if isinstance(ret, tuple) else config


# TODO: figure out how this can be incorporated into the export patch system
AutoConfig.from_pretrained = get_config_from_pretrained_patched

# TODO: figure out how this can be incorporated into the export patch system
# Only patch if the module isn't available
_mamba_ssm_module = "mamba_ssm"
_mamba_ssm_submodule = f"{_mamba_ssm_module}.ops.triton.layernorm_gated"
if importlib.util.find_spec(_mamba_ssm_module) is None:
    stub_mod = types.ModuleType(_mamba_ssm_submodule)
    stub_mod.rmsnorm_fn = _rms_norm_ref
    sys.modules[_mamba_ssm_submodule] = stub_mod
