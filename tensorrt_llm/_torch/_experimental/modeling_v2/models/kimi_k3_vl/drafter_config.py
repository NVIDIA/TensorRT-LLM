# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which standalone DSpark drafter checkpoints the tp16_moetp4ep4 target builds, read from the drafter's config, the
mapping and the requested attention backend.

The target's drafters (``drafter.py`` GQA, ``drafter_mla.py`` MLA) and the family's ``routing.py`` apply the same
functions: each drafter constructor raises where one of them refuses, and ``route`` declines the deployment before it
returns the target, so ``modeling_v2="auto"`` falls back to Modeling V1 and ``"require"`` reports ``NoTarget``. The
module sits at the family level so that ``route`` reads it without importing a target package (whose ``__init__.py``
loads the family's native library). It imports no kernels and no ops; the build probes below import flashinfer and
read the TensorRT-LLM bindings when they are called.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Optional, Tuple

# The head switches and their spellings (RadixArk ships ``enable_confidence_head`` top-level).
_HEAD_KEY_ALIASES = {"use_confidence_head": ("use_confidence_head", "enable_confidence_head")}

#: The ``attention_backend`` names `K3DSparkDrafter` takes; it declines FA4 among them
#: (`gqa_attention_backend_declines`).
GQA_ATTENTION_BACKENDS = ("AUTO", "TRTLLM", "VANILLA", "FA4")

#: The block decodes `K3MLADSparkDrafter` runs; AUTO picks `MLA_DEFAULT_ATTENTION_BACKEND` and does not fall back.
MLA_ATTENTION_BACKENDS = ("VANILLA", "TRTLLM", "CUTEDSL")
MLA_DEFAULT_ATTENTION_BACKEND = "CUTEDSL"

# The TensorRT-LLM bindings the TRTLLM-Gen paged attention's workspace sizing and preprocessing need.
_TRTLLM_GEN_NANOBIND_OPS = (
    "get_trtllm_gen_context_workspace_layout",
    "get_trtllm_gen_generation_workspace_layout",
    "trtllm_gen_context_preprocess",
    "trtllm_gen_context_postprocess",
    "trtllm_gen_generation_preprocess",
    "build_trtllm_gen_kv_cache_metadata",
)


def head_config(pretrained_config, key):
    """One DSpark head switch from ``dspark_config``, ``dflash_config``, ``dspark_<key>`` or ``<key>`` (in that order,
    for each spelling of ``key``); None where the drafter declares it nowhere."""
    dspark_cfg = getattr(pretrained_config, "dspark_config", None) or {}
    dflash_cfg = getattr(pretrained_config, "dflash_config", None) or {}
    for name in _HEAD_KEY_ALIASES.get(key, (key,)):
        for value in (
            dspark_cfg.get(name),
            dflash_cfg.get(name),
            getattr(pretrained_config, f"dspark_{name}", None),
            getattr(pretrained_config, name, None),
        ):
            if value is not None:
                return value
    return None


def is_mla(pretrained_config) -> bool:
    """Whether the drafter config is MLA-shaped (``kv_lora_rank`` and ``qk_rope_head_dim``): `K3MLADSparkDrafter`
    builds it, else `K3DSparkDrafter`. The built-in DSpark drafter dispatches on the same test."""
    return bool(
        getattr(pretrained_config, "kv_lora_rank", None)
        and getattr(pretrained_config, "qk_rope_head_dim", None)
    )


def non_vanilla_markov_head(pretrained_config) -> Optional[str]:
    """The Markov head type where the drafter declares a Markov head (``markov_rank`` above 0) that is not vanilla, else
    None. Both drafters run only the vanilla head."""
    markov_rank = int(head_config(pretrained_config, "markov_rank") or 0)
    head_type = str(head_config(pretrained_config, "markov_head_type") or "vanilla").lower()
    if markov_rank > 0 and head_type != "vanilla":
        return head_type
    return None


def is_causal(pretrained_config) -> bool:
    """Whether the drafter declares a causal block. Both drafters attend non-causally within the block."""
    return bool(head_config(pretrained_config, "causal"))


def logit_transform(dflash_config: dict) -> Tuple[float, Optional[float]]:
    """The ``(multiplier, softcap)`` a drafter's ``dflash_config`` declares; ``(1.0, None)`` for none. Both drafters
    read it at construction.

    Raises:
        ValueError: a multiplier that is not positive and finite, or a softcap that is negative or not finite.
    """
    multiplier = dflash_config.get("output_multiplier")
    multiplier = 1.0 if multiplier is None else float(multiplier)
    if not (math.isfinite(multiplier) and multiplier > 0):
        raise ValueError(
            f"dflash_config.output_multiplier must be positive and finite, got {multiplier}"
        )
    softcap = dflash_config.get("final_logit_softcapping")
    softcap = 0.0 if softcap is None else float(softcap)
    if not (math.isfinite(softcap) and softcap >= 0):
        raise ValueError(
            f"dflash_config.final_logit_softcapping must be non-negative and finite, got {softcap}"
        )
    return multiplier, softcap or None


def logit_transform_declines(pretrained_config) -> Optional[str]:
    """Why neither drafter takes this drafter's logit transform, or None (`logit_transform` raises ``ValueError`` in
    the constructors)."""
    try:
        logit_transform(getattr(pretrained_config, "dflash_config", None) or {})
    except ValueError as error:
        return str(error)
    return None


def requested_backend_unavailable(drafter: str, backend: str, reason: str) -> str:
    """The refusal of an explicitly requested ``backend`` this build cannot run (``ValueError`` in the constructors)."""
    return f"attention_backend={backend!r} was requested but it is unavailable for {drafter}: {reason}."


# ----------------------------------------------------------------------------------------------------------------------
# The GQA drafter (K3DSparkDrafter).
# ----------------------------------------------------------------------------------------------------------------------


def gqa_declines(pretrained_config) -> Optional[str]:
    """Why `K3DSparkDrafter` does not build this standalone drafter config, or None (``NoTarget`` in the constructor).

    It builds a Qwen3-shaped GQA backbone: q / k / v of ``head_dim`` heads with q / k RMSNorm, full-rotary
    RoPE (plain or YaRN), every layer full non-causal attention, a bias-free SiLU MLP, and the DSpark heads.
    Everything else the built-in DSpark drafters serve (an MLA backbone, sliding windows, DFlash 2, other backbones)
    is declined."""
    cfg = pretrained_config
    if is_mla(cfg):
        return "the drafter is MLA-shaped"
    if getattr(cfg, "model_type", None) != "qwen3":
        return f"the drafter backbone is {getattr(cfg, 'model_type', None)!r}, not a Qwen3-shaped GQA backbone"
    dflash_config = getattr(cfg, "dflash_config", None) or {}
    if any(dflash_config.get(key) for key in ("use_swa", "conv_kernel_size", "selector_rank")):
        return "the drafter declares sliding windows or DFlash 2 features"
    layer_types = getattr(cfg, "layer_types", None) or []
    if any(t != "full_attention" for t in layer_types) or getattr(cfg, "use_sliding_window", False):
        return "the drafter has sliding-window layers"
    if getattr(cfg, "sliding_window", None) is not None and getattr(
        cfg, "use_sliding_window", True
    ):
        return "the drafter declares a sliding window"
    if is_causal(cfg):
        return "the drafter is causal"
    if getattr(cfg, "attention_bias", False) or getattr(cfg, "mlp_bias", False):
        return "the drafter's projections have biases"
    if getattr(cfg, "hidden_act", "silu") != "silu":
        return f"the drafter's MLP activation is {cfg.hidden_act}"
    if getattr(cfg, "partial_rotary_factor", 1.0) != 1.0:
        return "the drafter's RoPE is partial-rotary"
    head_type = non_vanilla_markov_head(cfg)
    if head_type is not None:
        return f"the drafter's Markov head is {head_type!r}, not vanilla"
    return None


def gqa_tp_declines(pretrained_config, mapping) -> Optional[str]:
    """Why `K3DSparkDrafter` does not split this drafter over ``mapping``'s TP group, or None (``NoTarget`` in the
    constructor): its query heads, KV heads and MLP width must split evenly (TP 1 under attention DP)."""
    tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
    for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        count = getattr(pretrained_config, name, None)
        if count is not None and count % tp_size:
            return f"the drafter's {name} ({count}) does not split evenly over TP {tp_size}"
    return None


def gqa_unknown_attention_backend(requested: str) -> Optional[str]:
    """Why `K3DSparkDrafter` does not take the ``attention_backend`` name ``requested``, or None (``ValueError`` in
    the constructor): CUTEDSL and any name outside `GQA_ATTENTION_BACKENDS`."""
    if requested not in GQA_ATTENTION_BACKENDS:
        return (
            "K3DSparkDrafter attention backend must be one of ['VANILLA', 'TRTLLM', 'FA4'], got "
            f"{requested!r}."
        )
    return None


def gqa_attention_backend_declines(requested: str) -> Optional[str]:
    """Why `K3DSparkDrafter` does not run its block attention on ``attention_backend`` ``requested``, or None
    (``NoTarget`` in the constructor): FA4. ``DSparkDecodingConfig.attention_backend`` cannot ask for it (its values
    are AUTO, VANILLA, TRTLLM, CUTEDSL), so only a direct construction reaches it."""
    if requested == "FA4":
        return "the drafter runs its block attention on TRTLLM or VANILLA; attention_backend is FA4"
    return None


def trtllm_gen_unavailability_reason(sm: int) -> Optional[str]:
    """Why this build cannot run `K3DSparkDrafter`'s TRTLLM block attention (flashinfer TRTLLM-Gen) on SM ``sm``, or
    None: flashinfer missing, a TensorRT-LLM binding the kernel's workspace needs missing, or not SM100 / SM103."""
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
    from tensorrt_llm._utils import is_sm_100f

    if not IS_FLASHINFER_AVAILABLE:
        return "flashinfer is not installed"
    from tensorrt_llm.bindings.internal import thop

    missing_ops = [op for op in _TRTLLM_GEN_NANOBIND_OPS if not hasattr(thop, op)]
    if missing_ops:
        return "missing fused nanobind ops: " + ", ".join(missing_ops)
    if not is_sm_100f(sm):
        return f"requires SM100 or SM103, got SM{sm}"
    return None


def gqa_attention_backend_refusal(requested: str, sm: int) -> Optional[str]:
    """Why `K3DSparkDrafter` refuses ``attention_backend`` ``requested`` on SM ``sm``, or None, in the constructor's
    order: an unknown name, FA4, an explicit TRTLLM this build cannot run. AUTO falls back to VANILLA where TRTLLM
    is unavailable; VANILLA's flash-attn is probed only by the constructor."""
    reason = gqa_unknown_attention_backend(requested) or gqa_attention_backend_declines(requested)
    if reason is None and requested == "TRTLLM":
        unavailable = trtllm_gen_unavailability_reason(sm)
        if unavailable is not None:
            reason = requested_backend_unavailable("K3DSparkDrafter", requested, unavailable)
    return reason


# ----------------------------------------------------------------------------------------------------------------------
# The MLA drafter (K3MLADSparkDrafter).
# ----------------------------------------------------------------------------------------------------------------------


def mla_declines(pretrained_config) -> Optional[str]:
    """Why `K3MLADSparkDrafter` does not build this standalone drafter config, or None (``NoTarget`` in the
    constructor): it builds an MLA backbone (``kv_lora_rank`` and ``qk_rope_head_dim``)."""
    if not is_mla(pretrained_config):
        return "the drafter is not MLA-shaped"
    return None


def mla_cp_declines(mapping) -> Optional[str]:
    """Why `K3MLADSparkDrafter` does not run on ``mapping``, or None (``NotImplementedError`` in the constructor):
    context parallelism."""
    if mapping.cp_size > 1:
        return f"The MLA DSpark drafter does not support context parallelism (cp_size={mapping.cp_size})."
    return None


def mla_tp_declines(pretrained_config, mapping) -> Optional[str]:
    """Why `K3MLADSparkDrafter` does not split this drafter over ``mapping``'s TP group, or None (``ValueError`` in the
    constructor): its heads and MLP width must split evenly (TP 1 under attention DP)."""
    tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
    for name in ("num_attention_heads", "intermediate_size"):
        count = getattr(pretrained_config, name, None)
        if count is not None and count % tp_size:
            return f"DSpark MLA drafter has {name}={count}, not divisible by tp_size {tp_size}."
    return None


def mla_unknown_attention_backend(backend: str) -> Optional[str]:
    """Why `K3MLADSparkDrafter` does not run the block decode ``backend`` (AUTO resolved to
    `MLA_DEFAULT_ATTENTION_BACKEND`), or None (``ValueError`` in the constructor)."""
    if backend not in MLA_ATTENTION_BACKENDS:
        return (
            f"K3MLADSparkDrafter attention backend must be one of {list(MLA_ATTENTION_BACKENDS)}, got "
            f"{backend!r}."
        )
    return None


@lru_cache(maxsize=1)
def mla_paged_decode_kernel():
    """flashinfer's absorbed-MLA paged decode (``flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla``), the kernel
    of `K3MLADSparkDrafter`'s TRTLLM block decode, or None where it is not importable."""
    try:
        import flashinfer

        return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
    except (ImportError, AttributeError):
        return None


def mla_paged_decode_unavailability_reason() -> Optional[str]:
    """Why this build cannot run `K3MLADSparkDrafter`'s TRTLLM block decode, or None."""
    if mla_paged_decode_kernel() is None:
        return "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla is not importable"
    return None


def mla_attention_backend_refusal(requested: str) -> Optional[str]:
    """Why `K3MLADSparkDrafter` refuses ``attention_backend`` ``requested``, or None, as far as this module can tell:
    an unknown name, or an explicit TRTLLM without flashinfer's MLA decode. CUTEDSL (AUTO's pick) needs the CuTe DSL
    MLA decode registered with per-token KV bounds; only the constructor probes it, through the target's op."""
    backend = MLA_DEFAULT_ATTENTION_BACKEND if requested == "AUTO" else requested
    reason = mla_unknown_attention_backend(backend)
    if reason is None and backend == "TRTLLM":
        unavailable = mla_paged_decode_unavailability_reason()
        if unavailable is not None:
            reason = requested_backend_unavailable("K3MLADSparkDrafter", backend, unavailable)
    return reason


def mla_markov_head_declines(pretrained_config) -> Optional[str]:
    """Why `K3MLADSparkDrafter` does not run this drafter's Markov head, or None (``ValueError`` in the
    constructor)."""
    head_type = non_vanilla_markov_head(pretrained_config)
    if head_type is not None:
        return (
            f"DSpark drafter declares markov_head_type='{head_type}'; only 'vanilla' is supported."
        )
    return None


def mla_causal_declines(pretrained_config) -> Optional[str]:
    """Why `K3MLADSparkDrafter` does not run this drafter's block, or None (``ValueError`` in the constructor): a
    causal block."""
    if is_causal(pretrained_config):
        return "DSpark drafter sets causal=true; the block decode is non-causal."
    return None


def mla_rope_scaling(pretrained_config) -> dict:
    """The drafter's RoPE settings: ``rope_parameters``, else ``rope_scaling``, else none."""
    return (
        getattr(pretrained_config, "rope_parameters", None)
        or getattr(pretrained_config, "rope_scaling", None)
        or {}
    )


def mla_rope_type(scaling: dict) -> str:
    """The RoPE type of ``scaling`` (`mla_rope_scaling`)."""
    return scaling.get("rope_type") or scaling.get("type") or "default"


def mla_rope_declines(pretrained_config) -> Optional[str]:
    """Why `K3MLADSparkDrafter` does not build this drafter's RoPE, or None: a RoPE other than YaRN (``ValueError`` in
    the constructor) or a YaRN without its ``factor`` or ``original_max_position_embeddings`` (``KeyError``)."""
    scaling = mla_rope_scaling(pretrained_config)
    rope_type = mla_rope_type(scaling)
    if rope_type != "yarn":
        return (
            f"MLA DSpark drafter declares rope_type={rope_type!r}; only 'yarn' is supported (the block decode "
            "builds its own YaRN table)."
        )
    for key in ("factor", "original_max_position_embeddings"):
        if key not in scaling:
            return f"MLA DSpark drafter's YaRN RoPE declares no {key!r}"
    return None


# ----------------------------------------------------------------------------------------------------------------------
# The deployment.
# ----------------------------------------------------------------------------------------------------------------------


def declines(pretrained_config, mapping, attention_backend: str, sm: int) -> Optional[str]:
    """Why neither drafter builds this standalone drafter config on ``mapping`` with ``attention_backend`` on SM
    ``sm``, or None: every refusal of the drafter that builds it, in its constructor's order, except the probes only
    the constructor runs (the GQA drafter's flash-attn for VANILLA, the MLA drafter's CuTe DSL decode for CUTEDSL)."""
    cfg = pretrained_config
    if is_mla(cfg):
        return (
            mla_declines(cfg)
            or mla_cp_declines(mapping)
            or mla_tp_declines(cfg, mapping)
            or mla_attention_backend_refusal(attention_backend)
            or logit_transform_declines(cfg)
            or mla_markov_head_declines(cfg)
            or mla_causal_declines(cfg)
            or mla_rope_declines(cfg)
        )
    return (
        gqa_declines(cfg)
        or gqa_tp_declines(cfg, mapping)
        or gqa_attention_backend_refusal(attention_backend, sm)
        or logit_transform_declines(cfg)
    )


def external_drafter_config_kwargs(model_config, spec_config) -> dict:
    """``ModelConfig.from_pretrained`` kwargs for a standalone drafter checkpoint: the execution-layout properties it
    inherits from the target engine (the standalone case of the built-in
    ``modeling_speculative.external_drafter_config_kwargs``: no EPLB namespace is shared)."""
    requested_moe_backend = getattr(spec_config, "moe_backend", None)
    return dict(
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=model_config.moe_backend
        if requested_moe_backend is None
        else requested_moe_backend,
        mapping=model_config.mapping.mapping_for_full_rows(),
        spec_config=None,
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
        use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
        max_seq_len=model_config.max_seq_len,
    )


def load_pretrained_config(checkpoint_dir, trust_remote_code: bool = False, **kwargs) -> Any:
    """The ``pretrained_config`` that ``ModelConfig.from_pretrained(checkpoint_dir, trust_remote_code, **kwargs)``
    builds, read the same way (the engine's config loader under the remote-code lock, the dtype filled in, a text
    sub-config mirrored onto the top level) without parsing the checkpoint's quantization or reading weights."""
    from tensorrt_llm._torch.model_config import _mirror_text_subconfig_attrs, hf_remote_code_lock
    from tensorrt_llm._torch.pyexecutor.config_utils import (
        load_pretrained_config as load_engine_config,
    )

    with hf_remote_code_lock():
        pretrained_config = load_engine_config(
            checkpoint_dir, trust_remote_code=trust_remote_code, **kwargs
        )
    dtype = getattr(pretrained_config, "dtype", None)
    if dtype is None:
        dtype = getattr(getattr(pretrained_config, "text_config", None), "dtype", None)
    pretrained_config.torch_dtype = dtype
    _mirror_text_subconfig_attrs(pretrained_config)
    return pretrained_config


def standalone_drafter_declines(model_config, spec_config, sm: int) -> Optional[str]:
    """Why the tp16_moetp4ep4 target does not build the standalone DSpark drafter of ``spec_config`` on SM ``sm``, or
    None: its checkpoint config, loaded as the target's ``spec_shell.build`` loads it, read by `declines` with the
    mapping and attention backend the drafter is built with. A config that cannot be loaded is declined, so under
    ``modeling_v2="auto"`` Modeling V1 reports the load error itself."""
    kwargs = external_drafter_config_kwargs(model_config, spec_config)
    try:
        pretrained_config = load_pretrained_config(spec_config.speculative_model, **kwargs)
    except Exception as error:  # noqa: BLE001 - any load failure declines; Modeling V1 reports it.
        return (
            f"the drafter config at {spec_config.speculative_model!r} cannot be loaded "
            f"({type(error).__name__}: {error})"
        )
    return declines(pretrained_config, kwargs["mapping"], spec_config.attention_backend, sm)
