# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Helpers for the offline sharding-IR equivalence test.

The test takes a path to a sharding-IR-aware modeling file (e.g.
``models/custom/modeling_deepseek.py``, ``modeling_qwen3.py``, or any
``modeling_<name>.py`` whose canonical
implementation uses the sharding-IR path) and builds a tiny variant of that
model -- few layers, small hidden size, random weights -- without touching
the filesystem or LLM_MODELS_ROOT.

The path is the *only* user input: everything else (Python module name,
``*ForCausalLM`` class, HF config class) is derived from the file by walking
``AutoModelForCausalLMFactory._custom_model_mapping`` (populated when the
modeling module is imported). No assumption is made about the filename --
post-#13478 the IR-aware implementation is the canonical version for
deepseek/nemotron_h/qwen3/qwen3_5_moe (no ``_ir`` suffix); the helpers also
work for any other modeling file that opts into the sharding-IR path.

The tiny config is a single universal ``tiny_kwargs`` dict applied with
``setattr`` (so fields not used by a given model are silent no-ops), plus a
hasattr-driven feature-detection pass for the residue that depends on
``num_hidden_layers``.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class IRModelSpec:
    """Wires a modeling-file path to the classes the test will instantiate.

    All four fields are derived from the path by :func:`spec_from_modeling_file`;
    this class is just a typed bag of the derived values.
    """

    config_module: str
    """Python path of the HF ``configuration_*.py`` module owning the config."""

    config_cls: str
    """Class name of the HF config (e.g. ``Qwen3Config``)."""

    modeling_module: str
    """Python path of the sharding-IR modeling module."""

    modeling_cls: str
    """Class name of the ``*ForCausalLM`` to instantiate."""


# -----------------------------------------------------------------------------
# Tiny-config building blocks
# -----------------------------------------------------------------------------

# Single shared kitchen-sink dict. Covers fields that have a single sensible
# scalar default across all currently-onboarded sharding-IR model families
# (dense, GQA, MoE, MLA, SSM/Mamba). Applied with ``setattr`` (not constructor
# kwargs), so fields not read by a given config are harmless no-ops -- no
# per-family gating needed.
_TINY_KWARGS_UNIVERSAL: Dict[str, Any] = {
    # 4 layers so the rotation (mamba, attention, moe, mamba) covers every
    # block family AND a uniform-scale bug at layer N gets re-normalized into
    # a non-uniform-shape error at layer N+1 -- otherwise the final
    # ``norm_f`` RMSNorm before ``lm_head`` is scale-invariant and hides
    # uniform-scaling bugs (e.g. a missing all_reduce after a rowwise linear).
    "num_hidden_layers": 4,
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "vocab_size": 64,
    "max_position_embeddings": 256,
    "rope_theta": 1000000.0,
    # ``rope_scaling`` is intentionally NOT set here: in transformers 5.x,
    # ``PretrainedConfig.__setattr__`` aliases ``rope_scaling`` <->
    # ``rope_parameters`` and on composite configs (e.g.
    # ``Qwen3_5MoeConfig``) propagates the value into ``text_config``,
    # so a universal ``rope_scaling = None`` would also nuke
    # ``text_config.rope_parameters`` and break models that read it
    # (e.g. ``Qwen3_5MoeTextRotaryEmbedding``). DeepSeek-V3 needs a
    # ``factor`` key in its ``rope_scaling`` dict to clear
    # ``DeepSeekV3Attention.__init__``; that family-specific patch is
    # handled in ``_apply_layer_count_dependent_quirks`` below, gated
    # on the DeepSeek MLA marker so it does not touch other families.
    # MoE -- deepseek requires num_experts % n_group == 0 (n_group defaults to 8)
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_local_experts": 8,
    # MoE intermediate sizes must stay divisible by the NVFP4 block size (16)
    # AFTER sharding so the ``--sharding-ir-quant nvfp4`` path's on-the-fly
    # ``fp4_quantize`` load hook (needs ``in_features % 16 == 0``) doesn't trip
    # ``Expected k % sfVecSize == 0``. Routed experts are MoE-TP-sharded by
    # ``moe_tp_size`` (<=2 here): 32 -> 16 stays valid. The NemotronH shared
    # expert (``moe_shared_expert_intermediate_size``) is a dense MLP TP-sharded
    # by the full ``tp_size`` (<=4 under ``tep``): 64 -> 16 stays valid. bf16
    # runs are unaffected (just wider experts).
    "moe_intermediate_size": 32,
    "moe_shared_expert_intermediate_size": 64,
    "first_k_dense_replace": 0,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    # DeepSeek-V3 alternates dense and MoE blocks every ``moe_layer_freq``
    # layers (``layer_idx % moe_layer_freq == 0`` -> MoE). Real configs ship
    # ``moe_layer_freq = 1`` (every layer is MoE) but ``DeepseekV3Config()``
    # leaves the attribute unset, which trips ``DeepSeekV3DecoderLayer``
    # before sharding ever runs.
    "moe_layer_freq": 1,
    # MLA (DeepSeek-V3)
    "q_lora_rank": 8,
    "kv_lora_rank": 8,
    "qk_rope_head_dim": 8,
    "qk_nope_head_dim": 8,
    "v_head_dim": 8,
    # SSM (NemotronH / Mamba)
    "ssm_state_size": 8,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    # GDN / linear attention (Qwen3.5-MoE delta block) -- defaults are huge
    # (value_dim = 32 * 128 = 4096), so shrink to per-tp_size friendly sizes.
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
}


def _balanced_layers_block_type(num_layers: int) -> list:
    """Construct a ``num_layers``-long list rotating over the supported block types.

    Used to patch hybrid SSM configs that store per-layer block types as a
    list (e.g. ``NemotronHConfig.layers_block_type``). The rotation covers
    ``"mamba"`` (SSM), ``"attention"`` (self-attn), and ``"moe"`` (mixture of
    experts) so all three families' branches get exercised by the sharding-IR
    equivalence test, including the MoE path inside hybrid models. With
    ``num_layers >= 3`` the resulting pattern contains at least one of each.
    """
    rotation = ("mamba", "attention", "moe")
    return [rotation[i % len(rotation)] for i in range(num_layers)]


def _is_writable_attr(obj: Any, name: str) -> bool:
    """``True`` iff ``name`` is a real settable attribute on ``obj``.

    Filters out getter-only ``property`` descriptors (e.g.
    ``NemotronHConfig.hybrid_override_pattern`` is a backward-compat property
    derived from the settable ``layers_block_type`` list, not a settable
    field itself). Plain instance attributes and properties with a setter
    return ``True``.
    """
    cls_attr = getattr(type(obj), name, None)
    if isinstance(cls_attr, property):
        return cls_attr.fset is not None
    return hasattr(obj, name)


def _apply_layer_count_dependent_quirks(config: Any, num_layers: int) -> None:
    """Patch config fields whose valid value depends on ``num_layers``.

    The kitchen-sink :data:`_TINY_KWARGS_UNIVERSAL` covers fields that have a
    single sensible scalar default across all model families. This function
    handles the residue: fields whose value has to *match* ``num_layers`` or
    encode a per-layer layout (e.g. hybrid Mamba/Attention interleaving
    expressed as a per-layer character string or list).

    *This is a heuristic, not an exhaustive check.* Each branch was added in
    response to a real model surfacing a layer-count-dependent field during
    bring-up. When a future model surfaces a new such field, add a new
    ``hasattr`` branch here -- dispatch is by *feature presence on the config
    object*, not by the modeling-file name. That way the patch fires for any
    model that exposes the same field, independent of file naming or family.

    Currently handled:

    * ``layers_block_type`` -- per-layer list of block-type strings
      (``"mamba"`` / ``"attention"`` / ``"moe"``). Hybrid SSM models validate
      ``len(layers_block_type) == num_hidden_layers``, so the default list
      shipped with a full-size config rejects a 4-layer override. We replace
      it with a list that rotates over all three block types so that each
      family branch (SSM, attention, MoE) is exercised by the test. (Some
      configs expose a derived ``hybrid_override_pattern`` getter on top of
      this list; that getter is *not* settable, so we patch the underlying
      list.)

    *To extend*: add a new branch of the form::

        if _is_writable_attr(config, "<field>"):
            config.<field> = <value derived from num_layers>

    and document the new case in the list above. Use
    :func:`_is_writable_attr` rather than bare ``hasattr`` so getter-only
    ``property`` attributes are skipped (they raise ``AttributeError`` on
    assignment).
    """
    if _is_writable_attr(config, "layers_block_type"):
        config.layers_block_type = _balanced_layers_block_type(num_layers)


def _apply_per_family_quirks(config: Any) -> None:
    """Patch config fields whose default is broken for a specific family.

    These are NOT layer-count dependent and NOT safe to put in
    :data:`_TINY_KWARGS_UNIVERSAL`, because the universal kwargs are
    applied via ``setattr`` to every model's config, and some keys
    (notably ``rope_scaling``) trigger aliasing / sub-config
    propagation inside ``PretrainedConfig.__setattr__`` that would
    break other families. Each branch dispatches on a feature marker
    on the config object (never on the filename) so it fires for any
    config that exposes the same field, independent of family naming.

    **Ordering invariant**: must be called on a *pristine* config --
    i.e. before :data:`_TINY_KWARGS_UNIVERSAL` is applied -- because
    the universal kwargs ``setattr`` many family-specific keys
    (``kv_lora_rank``, ``q_lora_rank``, ``ssm_state_size``, ...) onto
    every config to keep one tiny-kwargs dict universal. After the
    universal pass every config looks like every family, so
    feature-presence detection here would over-match.

    Currently handled:

    * ``kv_lora_rank`` -- present iff this is a DeepSeek-V3 / MLA
      family config. ``modeling_deepseek.DeepSeekV3Attention.__init__``
      reads ``config.rope_scaling["factor"]`` unconditionally when
      ``rope_scaling is not None``. In production deployments the real
      DeepSeek-V3 checkpoint ships a full yarn dict that includes
      ``factor``, but a default-constructed ``DeepseekV3Config`` in
      transformers 5.x sets ``rope_scaling = {"rope_type": "default"}``
      without the yarn keys, so the lookup raises ``KeyError: 'factor'``
      before any sharding work runs. We override with a minimal dict
      that (a) provides ``factor`` to clear the unconditional lookup
      and (b) keeps ``rope_type = "default"`` so ``_init_rope`` routes
      through the vanilla rotary branch (correct stimulus for a sharding
      equivalence test; yarn extrapolation behaviour is out of scope).
    """
    if _is_writable_attr(config, "kv_lora_rank"):
        config.rope_scaling = {"rope_type": "default", "factor": 1.0}


# -----------------------------------------------------------------------------
# Spec derivation from a modeling-file path
# -----------------------------------------------------------------------------


def _path_to_dotted_module(path: str, custom_models_package: str) -> str:
    """Convert a modeling-file path to its Python dotted module name.

    Accepts:

    * An absolute path ending in ``models/custom/modeling_x.py``.
    * A relative path ending in ``models/custom/modeling_x.py``.
    * A bare module short name such as ``modeling_x``.

    ``custom_models_package`` comes from the loaded factory's module, so the
    result targets either bundled AutoDeploy or standalone ``llmc`` without a
    filesystem-root or top-level-package assumption. The conversion is purely
    syntactic -- no filename pattern is required.
    """
    candidate = Path(path.replace("\\", "/"))
    if len(candidate.parts) > 1:
        if candidate.suffix != ".py" or candidate.parent.name != "custom":
            raise ValueError(f"Path {path!r} must identify a Python file under models/custom.")
        if candidate.parent.parent.name != "models":
            raise ValueError(f"Path {path!r} must identify a Python file under models/custom.")
    module_name = candidate.stem
    return f"{custom_models_package}.{module_name}"


def _resolve_config_cls_from_transformers_registry(config_cls_name: str) -> Any:
    """Look up an HF config class by its ``__name__`` via the transformers registry.

    Iterates ``transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES``
    (a ``model_type -> config_class_name`` dict) to find the model_type whose
    config name matches, then resolves the (lazy) entry via ``CONFIG_MAPPING``.
    Returns ``None`` if the class isn't in the upstream transformers registry.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING, CONFIG_MAPPING_NAMES

    for model_type, name in CONFIG_MAPPING_NAMES.items():
        if name == config_cls_name:
            try:
                return CONFIG_MAPPING[model_type]
            except KeyError:
                return None
    return None


def spec_from_modeling_file(path: str) -> IRModelSpec:
    """Derive a full :class:`IRModelSpec` from any modeling-file path.

    Makes no assumption about filename -- works for canonical
    ``modeling_qwen3.py`` (post-#13478), legacy ``modeling_<name>_ir.py`` if
    any still exist, or any future ``modeling_<name>.py``. Importing the
    module triggers its self-registration via
    ``AutoModelForCausalLMFactory.register_custom_model_cls("<ConfigName>", <ModelCls>)``;
    we then look up the registered ``*ForCausalLM`` class and resolve its
    HF config class.

    The HF config class is resolved in this order:

    1. ``model_cls.config_class`` -- the HF convention; preferred.
    2. The modeling module's own globals, looked up by the registered config
       class *name* (the registration key) -- works for IR files that import
       their config class at the top.
    3. The upstream transformers registry
       (``transformers.models.auto.configuration_auto.CONFIG_MAPPING``) --
       works for any config class transformers knows about, including IR
       files that only reference the config class by string in their
       registration call.
    """
    # Deferred import: pulls in the active AutoDeploy package and so must run
    # after the caller has done any necessary sys.path / env-var setup.
    from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

    models_package = AutoModelForCausalLMFactory.__module__.rsplit(".", 1)[0]
    module_name = _path_to_dotted_module(path, f"{models_package}.custom")
    mod = importlib.import_module(module_name)

    # Walk AutoModelForCausalLMFactory AND every subclass. Each subclass gets
    # its own _custom_model_mapping via __init_subclass__ (see hf.py:668), so a
    # model registered against a specialized factory (e.g. NemotronFlashForCausalLMFactory,
    # EagleDrafterFactory) is invisible to a search on the base factory alone.
    factories = [AutoModelForCausalLMFactory]
    pending = [AutoModelForCausalLMFactory]
    seen = {AutoModelForCausalLMFactory}
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                pending.append(sub)
                factories.append(sub)

    candidates = []
    for factory in factories:
        candidates.extend(
            (cfg_name, cls)
            for cfg_name, cls in factory._custom_model_mapping.items()
            if cls.__module__ == module_name and cls.__name__.endswith("ForCausalLM")
        )
    if not candidates:
        raise RuntimeError(
            f"No '*ForCausalLM' class registered from {module_name!r}. "
            "Ensure the modeling file ends with a "
            "'<AutoModelForCausalLMFactory or subclass>.register_custom_model_cls(...)' "
            "call for a 'ForCausalLM' class."
        )
    config_cls_name, model_cls = candidates[0]

    config_cls = getattr(model_cls, "config_class", None)
    if config_cls is None:
        config_cls = getattr(mod, config_cls_name, None)
    if config_cls is None:
        config_cls = _resolve_config_cls_from_transformers_registry(config_cls_name)
    if config_cls is None:
        raise RuntimeError(
            f"Could not resolve config class {config_cls_name!r} for "
            f"{model_cls.__module__}.{model_cls.__name__}. Tried "
            f"`model_cls.config_class`, the modeling module's globals, and "
            f"the transformers CONFIG_MAPPING. Either set "
            f"`{model_cls.__name__}.config_class = {config_cls_name}` in the "
            f"modeling file, or import {config_cls_name} at the module top."
        )

    return IRModelSpec(
        config_module=config_cls.__module__,
        config_cls=config_cls.__name__,
        modeling_module=module_name,
        modeling_cls=model_cls.__name__,
    )


# -----------------------------------------------------------------------------
# Tiny model build + forward helpers
# -----------------------------------------------------------------------------


def build_tiny_config(cfg_cls: type) -> Any:
    """Default-construct ``cfg_cls`` and patch it down to the tiny test shape.

    Shared by :func:`build_ir_model` (base modeling files) and
    :func:`build_eagle_draft_model` (Eagle drafter base configs). Applies the
    per-family quirks on the pristine config first, then the universal tiny
    kwargs, then the layer-count-dependent quirks -- see :func:`build_ir_model`
    for why that ordering matters.
    """
    config = cfg_cls()
    _apply_per_family_quirks(config)
    for k, v in _TINY_KWARGS_UNIVERSAL.items():
        setattr(config, k, v)
    _apply_layer_count_dependent_quirks(config, _TINY_KWARGS_UNIVERSAL["num_hidden_layers"])
    return config


def build_ir_model(spec: IRModelSpec, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Programmatically build the IR-onboarded model with a tiny config.

    Does not touch the filesystem and does not require LLM_MODELS_ROOT. The
    universal :data:`_TINY_KWARGS_UNIVERSAL` is applied with ``setattr`` on a
    default-constructed config; this works for fields not accepted as
    constructor kwargs in the installed transformers version. Family-specific
    field defaults that the universal kwargs cannot safely cover are patched
    via :func:`_apply_per_family_quirks`, which runs **before** the universal
    kwargs are applied so that its feature-presence dispatch reads the
    pristine config (the universal kwargs ``setattr`` many family-specific
    keys onto every config, which would otherwise cause spurious matches).
    Layer-count dependent fields are then patched via
    :func:`_apply_layer_count_dependent_quirks`.
    """
    cfg_module = importlib.import_module(spec.config_module)
    cfg_cls = getattr(cfg_module, spec.config_cls)
    config = build_tiny_config(cfg_cls)

    modeling_module = importlib.import_module(spec.modeling_module)
    model_cls = getattr(modeling_module, spec.modeling_cls)
    model = model_cls(config).to(device=device, dtype=dtype).eval()
    return model


# -----------------------------------------------------------------------------
# Eagle draft-model build + forward helpers
#
# The Eagle draft is not a standalone registered modeling file: it is built by
# ``EagleDrafterFactory`` from a base model config wrapped in ``EagleConfig``.
# For the sharding-IR equivalence test we don't need the target or any
# speculative-decoding machinery -- the draft GraphModule is just numbers in /
# numbers out, so we build the draft on a tiny config and feed it random
# ``(inputs_embeds, position_ids, hidden_states)`` exactly as the exported draft
# GM expects (see ``DraftModelExportInfo._init_dynamic_shape_lookup`` and
# ``EagleDrafterForCausalLM.forward``). Sharding correctness is a property of
# the math, independent of whether the activations are "real".
# -----------------------------------------------------------------------------

# model_type -> (base-config module, base-config class). The base config is
# built tiny, then wrapped via ``EagleConfig.from_base_config(base, model_type)``.
# ``llama`` resolves directly from transformers; ``nemotron_h`` has no bundled
# config class (see modeling_nemotron_h.py:631), so it is resolved through the
# transformers registry by name.
_EAGLE_BASE_CONFIG: Dict[str, Tuple[str, str]] = {
    "llama": ("transformers.models.llama.configuration_llama", "LlamaConfig"),
}


def _resolve_eagle_base_config_cls(model_type: str) -> type:
    """Return the base-model config class for an Eagle draft ``model_type``."""
    if model_type in _EAGLE_BASE_CONFIG:
        mod_name, cls_name = _EAGLE_BASE_CONFIG[model_type]
        return getattr(importlib.import_module(mod_name), cls_name)
    # Fallback: resolve by the conventional config class name via the
    # transformers registry (handles nemotron_h, whose config class is not
    # bundled locally).
    guessed = "".join(part.capitalize() for part in model_type.split("_")) + "Config"
    cfg_cls = _resolve_config_cls_from_transformers_registry(guessed)
    if cfg_cls is None:
        raise RuntimeError(
            f"Could not resolve a base config class for Eagle model_type "
            f"{model_type!r} (tried _EAGLE_BASE_CONFIG and transformers "
            f"registry name {guessed!r})."
        )
    return cfg_cls


def build_eagle_draft_model(model_type: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Build a tiny ``EagleDrafterForCausalLM`` for the given base ``model_type``.

    Mirrors ``EagleDrafterFactory._build_model`` (models/eagle.py): build a tiny
    base config, wrap it via ``EagleConfig.from_base_config``, then instantiate
    ``EagleDrafterForCausalLM``. No checkpoint / filesystem access.
    """
    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
        EagleConfig,
        EagleDrafterForCausalLM,
    )

    base_cfg = build_tiny_config(_resolve_eagle_base_config_cls(model_type))
    base_cfg.model_type = model_type
    eagle_cfg = EagleConfig.from_base_config(base_cfg, model_type)
    model = EagleDrafterForCausalLM(eagle_cfg).to(device=device, dtype=dtype).eval()
    return model


def build_random_draft_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
    std: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Deterministic ``{inputs_embeds, position_ids, hidden_states}`` for the draft.

    All float tensors share one generator so both equivalence sides see
    identical inputs. ``hidden_states`` is the (already fc-compressed) target
    hidden state the draft consumes -- shape ``[B, S, hidden_size]``.

    ``std`` is small (matched to the weight-init std) on purpose. The draft
    layer computes ``residual + attn(norm(...))``: the RMSNorm makes the
    attention/MLP *input* scale-invariant, so their output magnitude is set by
    the weights (~init std), while the residual is the raw ``hidden_states``.
    If ``hidden_states`` were unit-scale (``randn``) the residual would dwarf
    the sharded attn/MLP contribution and mask sharding bugs (a broken
    all_reduce would only perturb the small term). Scaling the inputs to the
    weight-init scale keeps the residual and the sharded contribution
    comparable so the negative-control (``SHARDING_IR_SABOTAGE=1``) actually
    trips.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    inputs_embeds = (
        torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, generator=gen)
        * std
    )
    hidden_states = (
        torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, generator=gen)
        * std
    )
    position_ids = (
        torch.arange(seq_len, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
        .contiguous()
    )
    return {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "hidden_states": hidden_states,
    }


def extract_draft_output(out: Any) -> torch.Tensor:
    """Pull the comparison tensor out of an ``Eagle3DraftOutput``.

    Prefers ``last_hidden_state`` (always populated for both Llama and
    NemotronH drafts); falls back to ``logits`` / ``norm_hidden_state``. Also
    accepts a raw tensor or tuple (post-export GM output).
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    for attr in ("last_hidden_state", "logits", "norm_hidden_state"):
        val = getattr(out, attr, None)
        if val is not None:
            return val
    raise TypeError(f"Cannot extract a draft output tensor from {type(out)}")


def extract_logits(out: Any) -> torch.Tensor:
    """Pull the logits tensor out of a model forward result.

    Accepts a raw tensor (post torch.export typically yields a tuple), a tuple
    or list with logits at position 0, or an HF ``ModelOutput`` with a
    ``.logits`` attribute.
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(f"Cannot extract logits from forward output of type {type(out)}")


def build_random_prefill_inputs(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic ``(input_ids, position_ids)`` for the equivalence prefill."""
    gen = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long, generator=gen
    )
    position_ids = (
        torch.arange(seq_len, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
        .contiguous()
    )
    return input_ids, position_ids


def random_init_with_seed(model: nn.Module, seed: int, std: float = 0.02) -> None:
    """Re-initialize model parameters in-place with deterministic random values.

    Uses a CPU ``Generator`` to avoid relying on rank-dependent CUDA RNG state;
    every rank that calls this with the same ``seed`` ends up with bit-identical
    weights, which is what the equivalence test relies on. Random draws are
    always done at fp32 and cast to the parameter dtype (fp8/bf16 don't have
    a ``normal_kernel_cpu`` implementation).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            flat = torch.empty(p.numel(), dtype=torch.float32).normal_(0.0, std, generator=gen)
            p.data.copy_(flat.view_as(p).to(dtype=p.dtype, device=p.device))
        for b in model.buffers():
            if b.dtype.is_floating_point:
                flat = torch.empty(b.numel(), dtype=torch.float32).normal_(0.0, std, generator=gen)
                b.data.copy_(flat.view_as(b).to(dtype=b.dtype, device=b.device))
