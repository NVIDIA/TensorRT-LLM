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
``tensorrt_llm/_torch/auto_deploy/models/custom/modeling_deepseek.py``,
``modeling_qwen3.py``, or any ``modeling_<name>.py`` whose canonical
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
    "moe_intermediate_size": 16,
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


def fix_moe_routers_deterministic(model) -> int:
    """Force MoE routers to deterministically pick experts ``[0..top_k-1]``.

    Walks ``model.named_modules`` looking for modules whose class name contains
    ``Router`` or ``Gate`` (e.g. ``Qwen3_5MoeTopKRouter``, ``DeepSeekV3MoEGate``,
    ``NemotronHTopkRouter``) with a 2-d ``weight`` of shape
    ``(num_experts, hidden_size)`` and ``num_experts <= 64``. For each match:

      * ``weight[i, :] = (num_experts - i) / sqrt(H)``  -- small but monotonic
        decreasing coefficient over experts, keeps the linear projection in the
        export graph.
      * ``e_score_correction_bias[i] = (num_experts - i) * 100``  (if present)
        -- the grouped-top-k path in DeepSeek-V3 / Nemotron-H reads this bias,
        so making it large and monotonic dominates the routing decision.
      * For Qwen3.5-MoE routers (no built-in bias), the router's ``forward`` is
        replaced by a version that adds a strong monotonic logit bias before
        softmax / top-k. The bias dominates the linear contribution from
        ``hidden_states``, so top-k always picks experts ``[0..top_k-1]``
        regardless of input -- killing the cross-precision routing flips that
        would otherwise mask sharding bugs vs reduction-order noise.

    Router weights themselves are not TP-sharded, so both the unsharded and
    sharded forwards see the same router output for a given token. Returns the
    count of fixed router modules.
    """
    import types

    import torch
    import torch.nn.functional as F

    def _patched_qwen_router_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight) + self._test_router_bias
        routing_weights = F.softmax(router_logits, dtype=torch.float, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights.to(hidden_states.dtype), selected_experts

    fixed = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if "Router" not in cls and "Gate" not in cls:
                continue
            if not hasattr(mod, "weight"):
                continue
            w = mod.weight
            if w.ndim != 2:
                continue
            num_experts, hidden = w.shape
            if num_experts > 64:
                continue  # not a router (probably a FFN projection)
            coeffs_fp32 = torch.arange(num_experts, 0, -1, dtype=torch.float32, device=w.device)
            new_w = (
                (coeffs_fp32 / (hidden**0.5)).unsqueeze(1).expand(num_experts, hidden).to(w.dtype)
            )
            w.data.copy_(new_w)
            if hasattr(mod, "e_score_correction_bias"):
                # Used by DeepSeek-V3 / Nemotron-H grouped top-k path (noaux_tc_op).
                b = mod.e_score_correction_bias
                b.data.copy_((coeffs_fp32 * 100.0).to(b.dtype))
            elif cls == "Qwen3_5MoeTopKRouter":
                # Qwen3.5-MoE has no built-in bias; install one and patch forward.
                bias = (coeffs_fp32 * 100.0).to(w.dtype)
                if not hasattr(mod, "_test_router_bias"):
                    mod.register_buffer("_test_router_bias", bias)
                else:
                    mod._test_router_bias.data.copy_(bias)
                mod.forward = types.MethodType(_patched_qwen_router_forward, mod)
            fixed += 1
    return fixed


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


def _path_to_dotted_module(path: str) -> str:
    """Convert a modeling-file path to its Python dotted module name.

    Accepts:

    * An absolute path: ``/.../tensorrt_llm/_torch/auto_deploy/models/custom/modeling_x.py``
    * A path relative to cwd or repo root: ``tensorrt_llm/_torch/.../modeling_x.py``
    * A bare module short name: ``modeling_x`` (resolved under
      ``tensorrt_llm._torch.auto_deploy.models.custom``)

    The conversion is purely syntactic -- no filename pattern is required.
    """
    if "/" not in path and not path.endswith(".py"):
        return f"tensorrt_llm._torch.auto_deploy.models.custom.{path}"

    p = Path(path).resolve() if not Path(path).is_absolute() else Path(path)
    p = p.with_suffix("")
    parts = p.parts
    if "tensorrt_llm" not in parts:
        raise ValueError(f"Path {path!r} does not contain a 'tensorrt_llm' package root anchor.")
    idx = parts.index("tensorrt_llm")
    return ".".join(parts[idx:])


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
    module_name = _path_to_dotted_module(path)
    mod = importlib.import_module(module_name)

    # Deferred import: pulls in tensorrt_llm and so must run *after* the
    # caller has done any necessary sys.path / env-var setup.
    from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

    candidates = [
        (cfg_name, cls)
        for cfg_name, cls in AutoModelForCausalLMFactory._custom_model_mapping.items()
        if cls.__module__ == module_name and cls.__name__.endswith("ForCausalLM")
    ]
    if not candidates:
        raise RuntimeError(
            f"No '*ForCausalLM' class registered from {module_name!r}. "
            "Ensure the modeling file ends with a "
            "'AutoModelForCausalLMFactory.register_custom_model_cls(...)' "
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
    config = cfg_cls()
    _apply_per_family_quirks(config)
    for k, v in _TINY_KWARGS_UNIVERSAL.items():
        setattr(config, k, v)
    _apply_layer_count_dependent_quirks(config, _TINY_KWARGS_UNIVERSAL["num_hidden_layers"])

    modeling_module = importlib.import_module(spec.modeling_module)
    model_cls = getattr(modeling_module, spec.modeling_cls)
    model = model_cls(config).to(device=device, dtype=dtype).eval()
    return model


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
