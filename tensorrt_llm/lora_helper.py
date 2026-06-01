# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections.abc import Callable, Iterable
from typing import Dict, List, Literal, Optional

from pydantic import Field

from tensorrt_llm.llmapi.utils import StrictBaseModel

logger = logging.getLogger(__name__)

# Routed-expert MoE LoRA modules, each mapped to the one kernel flag that marks
# its residual-stream side shared across experts (A for the up-projections, B
# for the down-projection). The slot names are counterintuitive: moe_h_to_4h is
# gate_proj (w1, silu side) and uses the `gated` slot; moe_gate is up_proj (w3,
# linear side) and uses `fc1`. Must match `slot_to_kernel` in
# `fused_moe_cutlass._extract_moe_lora_tensors`.
MOE_MODULE_SHARED_FLAG: Dict[str, str] = {
    "moe_h_to_4h": "gated_shared_a",
    "moe_gate": "fc1_shared_a",
    "moe_4h_to_h": "fc2_shared_b",
}


def all_false_moe_shared_flags() -> Dict[str, bool]:
    """Return a fresh kernel flag dict, one key per module, all set to False.

    The default for adapters with no shared side. Only each module's residual
    side is representable, so the mirror flags (fc1_shared_b, fc2_shared_a,
    gated_shared_b) are always false and omitted.
    """
    return {flag: False for flag in MOE_MODULE_SHARED_FLAG.values()}


def moe_shared_sides_to_kernel_flags(
        shared_sides: Dict[str, tuple]) -> Dict[str, bool]:
    """Translate per-module shared sides into the kernel flag dict.

    Only each module's residual-stream side (see `MOE_MODULE_SHARED_FLAG`) can be
    shared; a detected non-canonical sharing (or an unknown module) is ignored.

    Args:
        shared_sides: Mapping from MoE LoRA module name to its (shared_a,
            shared_b) pair, as detected by the LoRA loader.

    Returns:
        The kernel flag dict (see `all_false_moe_shared_flags`), each flag True iff that
        module's residual side is shared across experts.
    """
    flags = all_false_moe_shared_flags()
    for module_name, (shared_a, shared_b) in shared_sides.items():
        flag_name = MOE_MODULE_SHARED_FLAG.get(module_name)
        if flag_name is None:
            continue
        # Only the module's canonical residual side has a kernel flag, so we
        # honor just that side (selected by the flag's `_a`/`_b` suffix). If
        # both sides were detected shared, the non-canonical one collapses away
        # here; reading it per-expert stays correct since its slices are equal.
        if shared_a and shared_b:
            logger.debug(
                "MoE LoRA module '%s': both A and B sides detected shared; "
                "honoring only the canonical side (%s).", module_name,
                flag_name)
        shared = shared_a if flag_name.endswith("_a") else shared_b
        if shared:
            flags[flag_name] = True
    return flags


def merge_moe_shared_flags_for_batch(
    active_uids: Iterable[str],
    get_flags: Callable[[str], Dict[str, bool]],
) -> Optional[Dict[str, bool]]:
    """Merge per-adapter MoE shared-outer flags for one fused-MoE call.

    The op applies one global flag set per call, so a side is marked shared only
    when every active adapter shares it (the intersection of the per-uid flags).
    Where adapters disagree, that side falls back to the per-expert read, which
    stays correct for all of them because shared sides are replicated per expert
    in the LoRA cache. This lets a batch freely mix shared-outer and per-expert
    adapters.

    Args:
        active_uids: LoRA task ids present in the current batch.
        get_flags: Callable returning the kernel flag dict for a uid.

    Returns:
        The flag dict to set as `lora_params['moe_shared_flags']`, or None
        when there are no active uids or no side is shared by all of them.
    """
    uids = list(active_uids)
    if not uids:
        return None
    merged: Optional[Dict[str, bool]] = None
    for uid in uids:
        flags = get_flags(uid)
        if merged is None:
            merged = dict(flags)
        else:
            for key in merged:
                merged[key] = merged[key] and flags.get(key, False)
    assert merged is not None
    return merged if any(merged.values()) else None


def get_missing_qkv_modules_from_lora_modules(
        lora_target_modules: List[str]) -> List[str]:
    """Get missing QKV modules from LoRA target modules.

    In current design, q_lora_params, k_lora_params and v_lora_params should be all enabled or
    all disabled at the same time. However, some lora checkpoints (e.g. BART) only contain two of them,
    so we use zero tensor to fill the missing ones.
    """
    missing_qkv_modules = []
    if any(x in lora_target_modules for x in ["attn_q", "attn_k", "attn_v"]):
        for lora_module in ["attn_q", "attn_k", "attn_v"]:
            if lora_module not in lora_target_modules:
                missing_qkv_modules.append(lora_module)
    if any(x in lora_target_modules
           for x in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]):
        for lora_module in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]:
            if lora_module not in lora_target_modules:
                missing_qkv_modules.append(lora_module)
    return missing_qkv_modules


def get_default_trtllm_modules_to_hf_modules():
    """Get default mapping from TensorRT-LLM module names to HuggingFace module names."""
    return {
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_qkv": "qkv_proj",
        "attn_dense": "o_proj",
        "mlp_h_to_4h": "gate_proj",
        "mlp_4h_to_h": "down_proj",
        "mlp_gate": "up_proj",
        "mlp_gate_up": "gate_up_proj",
        "moe_h_to_4h": "w1",
        "moe_4h_to_h": "w2",
        "moe_gate": "w3",
        "moe_router": "gate",
        "shared_expert_h_to_4h": "shared_expert.gate_proj",
        "shared_expert_4h_to_h": "shared_expert.down_proj",
        "shared_expert_gate": "shared_expert.up_proj",
        "mamba_in_proj": "in_proj",
        "mamba_out_proj": "out_proj",
        "moe_latent_fc1": "fc1_latent_proj",
        "moe_latent_fc2": "fc2_latent_proj",
    }


def use_lora(
    model,
    lora_config: "LoraConfig",
    trtllm_modules_to_hf_modules: Optional[Dict[str, str]] = None,
):
    """Use LoRA with the given model and configuration.

    This function is a wrapper that delegates to the appropriate loading function
    based on the LoRA checkpoint source.
    """
    if lora_config.lora_ckpt_source == "nemo":
        from .lora_manager import load_nemo_lora
        load_nemo_lora(model, lora_config)
    elif lora_config.lora_ckpt_source == "hf":
        from .lora_manager import load_hf_lora
        load_hf_lora(model, lora_config, trtllm_modules_to_hf_modules)
    else:
        raise ValueError(
            f"Unsupported lora_ckpt_source: {lora_config.lora_ckpt_source}")


class LoraConfig(StrictBaseModel):
    lora_dir: List[str] = Field(
        default_factory=list,
        description="List of directories containing LoRA adapter checkpoints. "
        "For HuggingFace format, expects adapter_model.bin/.safetensors and adapter_config.json. "
        "For NeMo format, expects .nemo files. If empty, lora_target_modules must be provided."
    )
    lora_ckpt_source: Literal["hf", "nemo"] = Field(
        default="hf",
        description=
        "Checkpoint format: 'hf' for HuggingFace PEFT format, 'nemo' for NeMo format. "
        "NeMo only supports attn_qkv module.")
    max_lora_rank: int = Field(
        default=64,
        description=
        "Maximum LoRA rank across all adapters, used to pre-allocate workspace memory. "
        "Set to the actual max rank of your adapters to reduce memory usage.")
    lora_target_modules: List[str] = Field(
        default_factory=list,
        description="TensorRT-LLM module names where LoRA is applied "
        "(e.g., ['attn_q', 'attn_k', 'attn_v'], ['attn_qkv'], ['mlp_gate', 'mlp_up']). "
        "If empty and lora_dir is provided, inferred from checkpoint.")
    trtllm_modules_to_hf_modules: Dict[str, str] = Field(
        default_factory=dict,
        description=
        "Mapping from TensorRT-LLM module names to HuggingFace module names "
        "(e.g., {'attn_q': 'q_proj'}). If empty, uses default mappings.")
    max_loras: Optional[int] = Field(
        default=None,
        description=
        "Maximum number of LoRA adapters that can be loaded simultaneously in GPU memory. "
        "Controls device-side LoRA cache size.")
    max_cpu_loras: Optional[int] = Field(
        default=None,
        description=
        "Maximum number of LoRA adapters that can be stored in CPU memory. "
        "Controls host-side cache for prefetching adapters before moving to GPU."
    )
    swap_gate_up_proj_lora_b_weight: bool = Field(
        default=True,
        description=
        "Whether to swap gate/up projection order in fused gate_up_proj LoRA B weights. "
        "Set to False for models like Phi-4-MM that use a different weight order."
    )

    @property
    def missing_qkv_modules(self) -> List[str]:
        return get_missing_qkv_modules_from_lora_modules(
            self.lora_target_modules)
