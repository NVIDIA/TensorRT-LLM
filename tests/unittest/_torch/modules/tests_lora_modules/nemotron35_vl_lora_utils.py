# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Synthetic LoRA adapters for the Nemotron 3.5 VL checkpoints.

No Nemotron VL LoRA adapter is published, so multi-LoRA validation fabricates
its own. This differs from the builder in ``test_nemotron_h_lora_sanity.py`` in
three ways that the 3.5 checkpoints force:

* dimensions live under a nested ``llm_config``, not at the top level;
* the layer plan is the ``layers_block_type`` list (``mamba`` / ``moe`` /
  ``attention``), not the ``hybrid_override_pattern`` string;
* weights are keyed with an optional ``language_model.`` prefix, which is what
  an adapter actually trained against the VL wrapper would carry.

Both key layouts are expected to parse; :func:`assert_adapter_keys_parse`
is the cheap CPU check that they do, because
:func:`tensorrt_llm._torch.peft.lora.loaders.iterate_hf_lora` warns and skips an
unrecognized module rather than failing, which would otherwise leave an adapter
silently inert on the GPU.
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file

# Per block type, the HF projection names an adapter targets. Keys cover both
# spellings of the layer plan: the 3.5 checkpoints' `layers_block_type` words
# and the older `hybrid_override_pattern` letters.
_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MAMBA_PROJECTIONS = ("in_proj", "out_proj")
_MOE_PROJECTIONS = (
    "shared_experts.up_proj",
    "shared_experts.down_proj",
    "fc1_latent_proj",
    "fc2_latent_proj",
)

# Three vocabularies name the same 88-layer plan, and a Nemotron 3.5 checkpoint
# may use any of them:
#
# * `full_attention` / `linear_attention` -- the current Nemotron-H names, used
#   by the NVFP4 exports;
# * `attention` / `mamba` -- the legacy names, used by the bf16 SourceOfTruth;
# * `*` / `M` / `E` -- the `hybrid_override_pattern` letters on older
#   checkpoints, which is also what the runtime dispatches on at
#   `modeling_nemotron_h.py:702`.
#
# The current-to-legacy mapping is the checkpoint's own, from
# `_nemotron_h_compatible_config` in its `configuration_nemotron_h_omni.py`:
# `{"linear_attention": "mamba", "full_attention": "attention"}`. Both bf16 and
# NVFP4 resolve to the same 40 mamba / 40 moe / 8 attention split, so an adapter
# built from either checkpoint targets the same layers.
_BLOCK_ALIASES = {
    "attention": "attention",
    "full_attention": "attention",
    "*": "attention",
    "mamba": "mamba",
    "linear_attention": "mamba",
    "M": "mamba",
    "moe": "moe",
    "E": "moe",
    # A dense-MLP block carries neither shared experts nor latent projections,
    # so nothing is emitted for it.
    "mlp": "mlp",
    "-": "mlp",
}

# The projections that sit on the mixer submodule rather than on the mlp one.
_MIXER_PROJECTIONS = frozenset(_ATTENTION_PROJECTIONS + _MAMBA_PROJECTIONS)

# Matches `NemotronHForCausalLM.lora_config()`; duplicated here so the builder
# can be used without importing the model (and thus torch CUDA init) on a CPU box.
NEMOTRON_H_HF_MODULES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",
        "out_proj",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
        "fc1_latent_proj",
        "fc2_latent_proj",
    }
)


def read_llm_config(base_model_path: str) -> dict:
    """Return the language-model config, whether or not it is VL-nested.

    The VL checkpoints nest the decoder config under ``llm_config``; the
    text-only Super checkpoint puts the same fields at the top level. Accepting
    both lets one builder serve both, which is what makes a text-only control
    run possible.
    """
    with open(os.path.join(base_model_path, "config.json")) as f:
        config = json.load(f)
    return config.get("llm_config", config)


def layer_plan(llm_config: dict) -> List[str]:
    """Normalize either layer-plan spelling into a list of block names."""
    raw: Optional[Sequence[str]] = llm_config.get("layers_block_type")
    if raw is None:
        raw = llm_config.get("hybrid_override_pattern")
    if raw is None:
        raise KeyError(
            "config declares neither 'layers_block_type' nor 'hybrid_override_pattern'; "
            "cannot determine which layer takes which LoRA modules"
        )
    plan = []
    for block in raw:
        if block not in _BLOCK_ALIASES:
            raise KeyError(f"unrecognized layer block type {block!r}")
        plan.append(_BLOCK_ALIASES[block])
    return plan


def projection_dims(llm_config: dict) -> Dict[str, Tuple[int, int]]:
    """(in_features, out_features) for every projection an adapter may target."""
    hidden = llm_config["hidden_size"]
    q_dim = llm_config["num_attention_heads"] * llm_config["head_dim"]
    kv_dim = llm_config["num_key_value_heads"] * llm_config["head_dim"]
    shared_intermediate = llm_config["moe_shared_expert_intermediate_size"]
    latent = llm_config["moe_latent_size"]

    # Mamba2: in_proj emits z, x, B, C and dt concatenated; out_proj maps the
    # inner dimension back to the residual stream.
    d_inner = llm_config["mamba_head_dim"] * llm_config["mamba_num_heads"]
    d_in_proj = (
        2 * d_inner
        + 2 * llm_config["n_groups"] * llm_config["ssm_state_size"]
        + llm_config["mamba_num_heads"]
    )

    return {
        "q_proj": (hidden, q_dim),
        "k_proj": (hidden, kv_dim),
        "v_proj": (hidden, kv_dim),
        "o_proj": (q_dim, hidden),
        "in_proj": (hidden, d_in_proj),
        "out_proj": (d_inner, hidden),
        "shared_experts.up_proj": (hidden, shared_intermediate),
        "shared_experts.down_proj": (shared_intermediate, hidden),
        "fc1_latent_proj": (hidden, latent),
        "fc2_latent_proj": (latent, hidden),
    }


def _projections_for(block: str) -> Tuple[str, ...]:
    if block == "attention":
        return _ATTENTION_PROJECTIONS
    if block == "mamba":
        return _MAMBA_PROJECTIONS
    if block == "moe":
        return _MOE_PROJECTIONS
    return ()


def create_nemotron35_lora_adapter(
    output_dir: str,
    base_model_path: str,
    *,
    lora_rank: int = 8,
    seed: int = 0,
    key_prefix: str = "",
    std: float = 0.02,
    layer_indices: Optional[Sequence[int]] = None,
) -> str:
    """Write a synthetic HF LoRA adapter for a Nemotron 3.5 (VL) checkpoint.

    Args:
        output_dir: directory to create; receives ``adapter_config.json`` and
            ``adapter_model.safetensors``.
        base_model_path: checkpoint whose ``config.json`` supplies dimensions.
        lora_rank: adapter rank. ``lora_alpha`` is set to ``2 * lora_rank`` so
            that the ``alpha / r`` scaling applied at
            ``_torch/peft/lora/manager.py:815`` stays constant as rank varies.
            Without that, a high-rank adapter perturbs the logits less than a
            low-rank one and a divergence assertion turns rank-dependent.
        seed: distinguishes adapters. Two adapters differing only by seed must
            produce different output, which is the multi-LoRA assertion.
        key_prefix: ``"language_model."`` to mimic an adapter trained against
            the VL wrapper; ``""`` for one trained against the bare decoder.
        std: both ``lora_A`` and ``lora_B`` are drawn at this scale. Measured on
            Nemotron 3.5 Super VL: 0.02 leaves greedy output byte-identical to
            base even with the adapter demonstrably applied, so it reads as
            "LoRA was not applied"; 0.2 moves every token on both the text and
            the image path. Callers asserting divergence should pass 0.2. The
            sanity-test scale (A 0.01, B 0.001) is weaker still.
        layer_indices: restrict to these layers. Defaults to every layer.

    Returns:
        ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)
    llm_config = read_llm_config(base_model_path)
    plan = layer_plan(llm_config)
    dims = projection_dims(llm_config)
    generator = torch.Generator().manual_seed(seed)

    selected = set(range(len(plan))) if layer_indices is None else set(layer_indices)

    def randn(rows: int, cols: int) -> torch.Tensor:
        weight = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
        return (weight * std).to(torch.bfloat16)

    weights: Dict[str, torch.Tensor] = {}
    targeted = set()
    for layer_idx, block in enumerate(plan):
        if layer_idx not in selected:
            continue
        for projection in _projections_for(block):
            in_dim, out_dim = dims[projection]
            submodule = "mixer" if projection in _MIXER_PROJECTIONS else "mlp"
            key = (
                f"base_model.model.{key_prefix}backbone.layers.{layer_idx}.{submodule}.{projection}"
            )
            weights[f"{key}.lora_A.weight"] = randn(lora_rank, in_dim)
            weights[f"{key}.lora_B.weight"] = randn(out_dim, lora_rank)
            targeted.add(projection)

    if not weights:
        raise ValueError(
            f"no LoRA weights produced for layers {sorted(selected)}; "
            "the selected layers carry no adaptable projection"
        )

    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name_or_path": base_model_path,
                "bias": "none",
                "peft_type": "LORA",
                "r": int(lora_rank),
                "lora_alpha": float(2 * lora_rank),
                "target_modules": sorted(targeted),
                "task_type": "CAUSAL_LM",
                "use_rslora": False,
            },
            f,
        )
    return output_dir


def create_vision_only_lora_adapter(
    output_dir: str,
    base_model_path: str,
    *,
    lora_rank: int = 8,
    seed: int = 0,
) -> str:
    """Write an adapter that targets only vision-tower projections.

    No VLM in the repo routes ``lora_params`` to a vision encoder, so this
    adapter must be inert. It exists as the negative control: a test that only
    ever shows adapters changing the output cannot distinguish "LoRA works"
    from "any adapter perturbs everything".
    """
    os.makedirs(output_dir, exist_ok=True)
    llm_config = read_llm_config(base_model_path)
    with open(os.path.join(base_model_path, "config.json")) as f:
        vision_config = json.load(f).get("vision_config", {})
    vit_hidden = vision_config.get("hidden_size") or llm_config["hidden_size"]
    generator = torch.Generator().manual_seed(seed)

    weights = {}
    for layer_idx in range(2):
        key = f"base_model.model.vision_model.encoder.layers.{layer_idx}.attn.qkv"
        weights[f"{key}.lora_A.weight"] = (
            torch.randn(lora_rank, vit_hidden, generator=generator, dtype=torch.float32) * 0.02
        ).to(torch.bfloat16)
        weights[f"{key}.lora_B.weight"] = (
            torch.randn(3 * vit_hidden, lora_rank, generator=generator, dtype=torch.float32) * 0.02
        ).to(torch.bfloat16)

    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name_or_path": base_model_path,
                "bias": "none",
                "peft_type": "LORA",
                "r": int(lora_rank),
                "lora_alpha": float(2 * lora_rank),
                "target_modules": ["qkv"],
                "task_type": "CAUSAL_LM",
                "use_rslora": False,
            },
            f,
        )
    return output_dir


def assert_adapter_keys_parse(adapter_dir: str, hf_modules=NEMOTRON_H_HF_MODULES) -> Dict:
    """Fail if any adapter key is dropped by the HF LoRA loader.

    ``iterate_hf_lora`` warns and continues on an unrecognized module name, so
    a typo'd key produces an adapter that loads clean and does nothing. This
    asserts every ``lora_A`` key reached a module, which is the difference
    between "the adapter had no effect" and "the adapter was never applied".
    """
    from tensorrt_llm._torch.peft.lora.loaders import iterate_hf_lora

    weights = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    consumed = []

    def collect(layer_idx, hf_module, expert_idx, inout_or_mag, tensor):
        consumed.append((layer_idx, hf_module, inout_or_mag))

    parsed = iterate_hf_lora(collect, weights, set(hf_modules))

    expected = len(weights)
    if len(consumed) != expected:
        skipped = sorted(
            {
                key
                for key in weights
                if not any(f".{module}.lora_" in key for _, module, _ in consumed)
            }
        )
        raise AssertionError(
            f"{expected - len(consumed)} of {expected} LoRA weights were dropped by "
            f"iterate_hf_lora; the adapter would load clean and stay inert. "
            f"First dropped keys: {skipped[:5]}"
        )
    return parsed
