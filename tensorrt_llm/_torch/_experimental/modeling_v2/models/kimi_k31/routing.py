# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a KimiK31ForConditionalGeneration config lands. Read this file and you know.

Kimi K3.1 Pro (Moonshot's MXFP4 checkpoint: routed experts MXFP4, attention, shared expert and
vision BF16; a DSA indexer on every MLA layer) has a general target and two SM 10.3 perf targets.

``kimi_k31_pro_mxfp4__sm_10x__default`` is the general functional target on sm_103 (GB300, B300)
and sm_107 (Rubin). It derives its layout from the engine's ``Mapping`` and serves attention TP
(1, 2, 4, 8, 16 or 32 ranks, experts split by any MoE TP x EP that divides the 96 heads and 896
experts) and attention DP (2, 4, 8, 16 or 32 ranks, EP-only), as context, generation or
aggregated servers, and Helix context parallelism (TP 1 without attention DP, CP over 2, 4, 8, 16 or
32 ranks, the experts split over the TP x CP group, block reuse disabled; functional only, as the
engine's Helix support runs it), with the TRTLLM attention backend, an FP8 latent KV cache
(``kv_cache_config.dtype='fp8'``) or the checkpoint's BF16 latents (``auto``: the checkpoint declares no
KV-cache scheme), FP8 index keys, a position-free DSA indexer (``dsa_config``
``index_rope_dim`` 0) on the GVR / CUDA radix Top-K (no ``use_cute_dsl_topk``), an FP32 KDA recurrent
state, and DFlash2 speculative decoding (one-engine DFlash with the DFlash2 MLA drafter, its context
in the drafter's own dtype) or none. The text config must have the layer layout the targets build:
every layer KDA or MLA, every MLA layer a DSA layer, attention-residual blocks, and the latent routed
MoE with its SiTu betas, whose experts load as MXFP4 (not excluded from quantization). Everything
else raises ``NoTarget`` with the reason, below in the order checked.

On sm_103 with an FP8 latent KV cache the perf targets take the layouts they are written for, under the general
target's rules.
The suffix names the parallelism: ``tep`` is attention TP with MoE EP > 1 (optionally x MoE TP), ``dep``
is attention DP with MoE EP. ``kimi_k31_pro_mxfp4__sm_103__tep`` takes the attention TP layouts in
``_SM103_TEP_LAYOUTS`` (TEP8 with EP8, MoE TP2 x EP4 or MoE TP4 x EP2, TEP16 with MoE TP4 x EP4, TEP32
with MoE TP4 x EP8) and ``kimi_k31_pro_mxfp4__sm_103__dep`` the attention DP layout in
``_SM103_DEP_LAYOUTS`` (DEP16), where
it also admits two EP load balancer configurations (``moe_config.load_balancer``): a static expert
placement (one slot per expert and ``layer_updates_per_iter`` 0, as
``examples/wide_ep/ep_load_balancer/generate_eplb_config.py`` writes it), and per-layer dynamic EPLB
(``mode: per_layer_dynamic``, ``num_slots`` above the 896 experts and a multiple of the 16 EP ranks:
helper slots the target refills on every routed call; :func:`per_layer_dynamic_mismatch`). Every other
admitted deployment, every sm_107 one, every one with BF16 latents, and any deployment with
``TLLM_K31_DEFAULT_TARGET=1`` on its ranks runs the general target, which takes no load balancer.

``TLLM_K31_DEFAULT_TARGET`` (``1`` / ``true`` / ``yes`` / ``on``, off by default) routes every deployment to the general
target over the sm_103 perf target its GPU, layout and KV cache would select: an A/B against the general forward.
Every rank routes, so the variable has to reach every rank: set it before launch or pass it through
``env_overrides``. ``route`` logs the target it picks at INFO.

The ModelOpt NVFP4 export (routed experts NVFP4, shared experts and KDA projections MXFP8, the rest
BF16) runs on SM100, SM103, and SM107 through ``kimi_k31_pro_nvfp4__sm_10x__default``. It independently
selects FP8/NVFP4 main cache, FP8/FP4 index keys and FP32/BF16 KDA state, with TRTLLM, CUTEDSL or
MEGAMOE_CUTEDSL experts. The target owns its admission checks; its draft history uses the unified
cache in aggregated and disaggregated serving.

On sm_107, the export (``config.modelopt_nvfp4_mismatch``) runs on four specialized targets,
``kimi_k31_pro_nvfp4__sm_107__{cep,dep16,helix8,tep8}``. Each is one layout of one recipe: an FP8
latent KV cache, FP8 index keys, an FP32 KDA state and DFlash2 with an FP8 drafter history. They
take the layouts, server roles, MoE backends and draft lengths of ``_SPECIALIZED`` and nothing
else; only a context server may keep just the drafter history (``context_kv_only``). The dep16
target also serves the dep12 layout (attention DP and MoE EP over 12 ranks) on a context server:
the same recipe and kernels, with the all-to-all's expert windows (75 experts on ranks 0-7, 74 on
ranks 8-11). The configurable default never takes dep12 (its attention DP runs powers of two).
It serves the dep8 layout (attention DP and MoE EP over 8 ranks, 112 experts per rank) on a context
server the same way; a dep8 context server takes the dep16 target, not the configurable default.
Their drafter history lives in the unified draft cache of KV cache manager V2, so they declare no
per-slot drafter context (:func:`dflash_context_bytes_per_slot`). The server role is the one
``trtllm-serve --server_role`` passes as ``TRTLLM_DISAGG_ROLE`` (:func:`disagg_role`); routing
records it in ``model_config.extra_attrs`` (``DISAGG_ROLE_ATTR``), and beside it the lever
variables the target checks or logs (``LEVER_ENV_ATTR``, :func:`_lever_env`); the loader
carries both to the config the target is built from. The general target and the sm_103 perf
targets never take the export, and ``TLLM_K31_DEFAULT_TARGET`` does not apply to it.

``kimi_k31_pro_nvfp4__sm_107__cep`` is the context-parallel context server of N ranks, N in
``_CEP_SIZES`` (layout ``cep{N}``: TP 1 and Ulysses context parallelism N, i.e.
``context_parallel_size: N`` with ``cp_config: {cp_type: ULYSSES}``, without attention DP, MoE EP
N; N is bound at construction from the ``Mapping``). The Mapping alone tells it from tep8 (TP 8). A
cep-shaped layout of another N is refused with why (:func:`_cep_refusal`). It runs the MoE backend
the arguments name, ``CUTEDSL`` (the all-to-all around the CuteDSL experts) or ``MEGAMOE_CUTEDSL``
(MegaMoE). Its CuteDSL context servers, and
dep16's, are the specialized servers that take an EP load balancer: per-layer dynamic EPLB
(``moe_config.load_balancer`` with ``mode: per_layer_dynamic`` and ``num_slots`` above the 896
experts, a multiple of the EP ranks; helper slots refilled on every routed call;
:func:`per_layer_dynamic_mismatch`, ``_DYNAMIC_EPLB``), off unless configured.


The configurable NVFP4 default serves the remaining supported deployments on all three
architectures, including image input and BF16 draft history. The specialized SM107 targets serve
text only (``disable_mm_encoder``), except the dep16 target's context servers (dep16, dep12 and
dep8, ``_VISION_CELLS``), which encode images with the MoonViT tower when ``disable_mm_encoder`` is
false; any other specialized layout and role with images goes to the default target, which
refuses Ulysses CP, so a cep server with images has no target. All of them require explicit FP8
history; ``auto`` keeps BF16 history in the default target.

The ULL generation server on sm_107 (Helix context parallelism over 16 ranks, MoE TP 16 / EP 1,
FP8 latent KV, FP8 index keys, FP32 KDA state, DFlash2 at max_draft_len 7) goes to
``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1`` (:func:`_route_ull_generation`), whose recipe
module (``ops/common/rubin_ull_features.py``) validates the recipe and selects the ``ull16``
profile. It serves generation only; its context servers are the dep16 and tep8 DFlash2 context
targets, and image prompts need a dep16 one (the context server encodes them).

``TLLM_K31_HELIX16_WIDE`` (``1`` / ``true`` / ``yes`` / ``on``; off by default or ``0`` /
``false`` / ``no`` / ``off``; anything else refused) sends that ULL layout to its wide-batch fork
``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` on the same ``ull16`` profile, which also admits
max_draft_len 3, 4 and 5 (``rubin_ull_features.decide_wide``). It must reach every rank, like
``TLLM_K31_DEFAULT_TARGET``. Routing records the fork's switches ``TLLM_K31_HELIX16_WIDE_TAIL``
/ ``_FRONT`` in ``model_config.extra_attrs`` (:func:`helix16_wide_options`).

Everything else raises ``NoTarget`` with the reason, below in the order checked.
``modeling_v2="off"`` has no built-in K3.1 implementation to fall back to.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping as _Mapping
from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, dflash_context_for, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

#: Selects the general target over the sm_103 perf targets (see the module docstring).
DEFAULT_TARGET_ENV = "TLLM_K31_DEFAULT_TARGET"


def default_target_requested() -> bool:
    """Whether ``TLLM_K31_DEFAULT_TARGET`` asks for the general target."""
    return os.environ.get(DEFAULT_TARGET_ENV, "").strip().lower() in ("1", "true", "yes", "on")


#: Selects the ULL generation layout's experimental wide-batch target (see the module docstring).
HELIX16_WIDE_ENV = "TLLM_K31_HELIX16_WIDE"


def helix16_wide_requested() -> bool:
    """Whether ``TLLM_K31_HELIX16_WIDE`` picks the wide-batch fork (:func:`helix16_wide_switch`)."""
    return helix16_wide_switch(HELIX16_WIDE_ENV, False)


#: The wide-batch target's boolean switches' values (any case, outer spaces ignored): on, off.
HELIX16_WIDE_ON = ("1", "true", "yes", "on")
HELIX16_WIDE_OFF = ("0", "false", "no", "off")


def helix16_wide_switch(name: str, default: bool) -> bool:
    """The boolean switch ``name``: ``HELIX16_WIDE_ON`` / ``HELIX16_WIDE_OFF``, ``default`` when
    unset or empty; any other value raises ``ValueError`` (a typo never runs at the default)."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value not in HELIX16_WIDE_ON + HELIX16_WIDE_OFF:
        allowed = HELIX16_WIDE_ON + HELIX16_WIDE_OFF
        raise ValueError(f"{name}={value!r}: one of {allowed}, or unset")
    return value in HELIX16_WIDE_ON


#: ``model_config.extra_attrs`` key of routing's record of the switches; the target reads no env.
HELIX16_WIDE_ATTR = "kimi_k31_helix16_wide"
#: The wide-batch target's two switches, each value's first entry its default: ``tail``
#: (``TLLM_K31_HELIX16_WIDE_TAIL``, 5-64 rows) ``tuned`` or ``rows``; ``front``
#: (``TLLM_K31_HELIX16_WIDE_FRONT``, above 64 rows) ``tmem_merged`` or ``once`` (the target's
#: ``WideMoeOptions``).
HELIX16_WIDE_TAIL_ENV = "TLLM_K31_HELIX16_WIDE_TAIL"
HELIX16_WIDE_TAIL = ("tuned", "rows")
HELIX16_WIDE_FRONT_ENV = "TLLM_K31_HELIX16_WIDE_FRONT"
HELIX16_WIDE_FRONT = ("tmem_merged", "once")


def helix16_wide_options() -> dict:
    """``{"tail": .., "front": ..}``: each variable's value or its default; another value raises."""
    options = {}
    for key, env, allowed in (
        ("tail", HELIX16_WIDE_TAIL_ENV, HELIX16_WIDE_TAIL),
        ("front", HELIX16_WIDE_FRONT_ENV, HELIX16_WIDE_FRONT),
    ):
        value = os.environ.get(env, "").strip().lower() or allowed[0]
        if value not in allowed:
            raise ValueError(f"{env}={value!r}: one of {allowed}")
        options[key] = value
    return options


# (hidden, routed experts, experts per token, attention heads, q_lora_rank, kv_lora_rank, vocab).
# The layer count is left out so that truncated test checkpoints of the same shape route here
# (ungated). Shape-sniffing also matches fine-tunes of the same shape; the accuracy gate pins
# identity.
_SHAPE = (7168, 896, 16, 96, 1536, 512, 163840)
_SHAPE_FIELDS = (
    "hidden_size",
    "num_experts",
    "num_experts_per_token",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "vocab_size",
)
# GPUs the target runs on, and the ones it refuses with a reason.
_COMPUTE_CAPABILITIES = ((10, 3), (10, 7))
_NOT_VALIDATED = {(10, 0): "sm_100 is not validated yet"}
# Rubin: the GPU of the specialized targets.
_RUBIN = (10, 7)
# The ULL layout: Rubin, and GB300 for architecture-independent correctness checks.
_ULL_COMPUTE_CAPABILITIES = ((10, 3), (10, 7))
# kv_cache_config.dtype values that name a cache layout outside the target (FP8 and BF16 latents).
_KV_LAYOUTS_NOT_IN_TARGET = ("nvfp4", "fp8_ds_mla")
# Attention TP sizes (no attention DP): divisors of the 96 MLA / KDA / drafter heads that also
# divide the drafter MLP.
_ATTENTION_TP_SIZES = (1, 2, 4, 8, 16, 32)
# Attention DP sizes (EP-only MoE; 896 % size == 0 and a power of two).
_ATTENTION_DP_SIZES = (2, 4, 8, 16, 32)
_NUM_HEADS = 96
_NUM_EXPERTS = 896
# Page sizes of the paged indexer logits (DeepGEMM block_kv).
_TOKENS_PER_BLOCK = (32, 64, 128)
# The GPU and the layouts, as (attention DP, tp_size, moe_tp, moe_ep), the perf targets take.
_PERF_COMPUTE_CAPABILITY = (10, 3)
_SM103_TEP_LAYOUTS = frozenset(
    {
        (False, 8, 1, 8),  # TEP8, EP8 (the default split)
        (False, 8, 2, 4),  # TEP8, MoE TP2 x EP4 (less routed-expert skew across ranks)
        (False, 8, 4, 2),  # TEP8, MoE TP4 x EP2
        (False, 16, 4, 4),  # TEP16, MoE TP4 x EP4
        (False, 32, 4, 8),  # TEP32, MoE TP4 x EP8
    }
)
_SM103_DEP_LAYOUTS = frozenset({(True, 16, 1, 16)})  # DEP16, EP16
# Attention projections the target builds on an MLA / a KDA layer, all BF16.
_MLA_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "g_proj", "o_proj")
_KDA_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
# Module-name spellings of a text layer: per-layer quant entries use the first three; exclusions
# may also be written under the wrapper's ``language_model.layers.`` prefix.
_LAYER_PREFIXES = ("language_model.model.", "model.", "")
_EXCLUDED_LAYER_PREFIXES = _LAYER_PREFIXES + ("language_model.",)
# KDA recurrent-state dtypes the targets admit: "auto" resolves to FP32 for Kimi.
_KDA_STATE_DTYPES = ("auto", "float32")
# Intermediate width of each routed expert (moe_intermediate_size).
_ROUTED_EXPERT_WIDTH = 3072
# TRTLLM_FORCE_COMM_METHOD values the routed MoE's attention-DP exchange takes.
_EXCHANGES = ("NVLINK_ONE_SIDED", "ALLGATHER")
# Module-name spellings of a layer's routed experts and the text-model prefixes the targets ask the
# exclusion check about, after mapping the wrapper's ``language_model.`` exclusions onto the text model.
_ROUTED_EXPERT_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")
_ROUTED_EXPERT_PREFIXES = ("language_model.model.", "model.")
_LANGUAGE_MODEL_PREFIX = "language_model."
# speculative_config.context_kv_dtype values the DFlash2 drafter takes: its context stays in its own dtype.
_DRAFT_CONTEXT_KV_DTYPES = (None, "auto")

#: The variable that carries the server role (``trtllm-serve --server_role``).
DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
#: The ``model_config.extra_attrs`` key where routing records the server role of the sm_107 target
#: it returns; the target fixes its role at construction from it and refuses a config without
#: it.
DISAGG_ROLE_ATTR = "kimi_k31_disagg_role"
#: The ``model_config.extra_attrs`` key of routing's record, for a specialized target, of
#: ``LEVER_ENV`` / ``LEVER_ENV_PREFIXES`` as the workers see them (:func:`_lever_env`); the
#: family README says which a server role checks and which it logs.
LEVER_ENV_ATTR = "kimi_k31_lever_env"
LEVER_ENV_PREFIXES = (
    "TRTLLM_K3_FC12_",
    "TRTLLM_CUTEDSL_FC2_",
    "TRTLLM_NVLINK_ONE_SIDED_A2A_",
    "MEGAMOE_",
)
LEVER_ENV = (
    "TLLM_ROUTING_DISABLE_COOP_LAUNCH",
    "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET",
    "TLLM_MOE_ASYNC_INPUT_QUANT",
)
# The context-parallel sizes N of the cep target: a Ulysses CP N executor whose ranks each run 1 / N
# of the batch's rows through every layer (the MLA and KDA layers context-parallel, the MoE EP N). N
# divides the 96 KDA heads and the 896 experts, and its KV fans into the tep8 generation server.
_CEP_SIZES = (8, 16)
# The specialized targets' layouts, as (world_size, tp, cp, CP type (_cp_kind), attention DP,
# moe_tp, moe_ep). dep12 and dep8 run on the dep16 target (:func:`_route_modelopt`).
_SPECIALIZED_LAYOUTS = {
    **{f"cep{n}": (n, 1, n, "ulysses", False, 1, n) for n in _CEP_SIZES},
    "dep16": (16, 16, 1, None, True, 1, 16),
    "dep12": (12, 12, 1, None, True, 1, 12),
    "dep8": (8, 8, 1, None, True, 1, 8),
    "helix8": (8, 1, 8, "helix", False, 2, 4),
    "tep8": (8, 8, 1, None, False, 1, 8),
}
# The draft lengths (max_draft_len) a specialized server takes, one per server: the target verifies
# a block of max_draft_len + 1 rows per request on every step. A generation server's first block
# holds the draft the context server in front of it hands over, so the two run the same length.
_DRAFT_LENS = (3, 4, 5, 6, 7)
# Why no specialized target takes a longer draft (the refusal names it).
_LONGER_DRAFT = (
    "a draft of k fills 2k + 1 rows of the KDA verify kernels' 16-row chunk (the k replayed drafts "
    "and the k + 1 new rows) and k + 1 rows of one indexer-logits atom (32 index heads a row, at "
    "most 256 columns), so k stops at 7"
)
# (layout, server role) -> (MoE backends, draft lengths) of the specialized targets. A target of
# one backend builds it whatever moe_config asks, so the arguments must name it; the cep context
# server builds the one of its two the arguments name (the all-to-all around the CuteDSL experts,
# or MegaMoE). An aggregated server takes the generation role.
_SPECIALIZED = {
    **{(f"cep{n}", "context"): (("CUTEDSL", "MEGAMOE_CUTEDSL"), _DRAFT_LENS) for n in _CEP_SIZES},
    ("dep16", "context"): (("CUTEDSL",), _DRAFT_LENS),
    ("dep16", "generation"): (("MEGAMOE_CUTEDSL",), _DRAFT_LENS),
    ("dep12", "context"): (("CUTEDSL",), _DRAFT_LENS),
    ("dep8", "context"): (("CUTEDSL",), _DRAFT_LENS),
    ("helix8", "generation"): (("CUTEDSL_FC12",), _DRAFT_LENS),
    ("tep8", "context"): (("CUTEDSL",), _DRAFT_LENS),
    ("tep8", "generation"): (("CUTEDSL_FC12",), _DRAFT_LENS),
}
# The (layout, server role) cells that take image input (disable_mm_encoder false): the dep16
# target's context servers, which build the MoonViT tower and its projector (its Dims.vision).
# Every other cell serves text only.
_VISION_CELLS = frozenset({("dep16", "context"), ("dep12", "context"), ("dep8", "context")})
# The (layout, server role) cells whose target takes per-layer dynamic EPLB
# (moe_config.load_balancer mode per_layer_dynamic): the context servers' CuteDSL experts behind
# the one-sided all-to-all (cep and dep16; a cep server under MegaMoE takes none).
_DYNAMIC_EPLB = frozenset({*((f"cep{n}", "context") for n in _CEP_SIZES), ("dep16", "context")})
# The specialized targets' KV pages (their get_model_defaults).
_SPECIALIZED_TOKENS_PER_BLOCK = 64
_SPECIALIZED_RECIPE = (
    "the sm_107 specialized targets run the ModelOpt NVFP4 export with an FP8 latent KV cache "
    "(kv_cache_config.dtype 'fp8'), FP8 index keys, an FP32 KDA state "
    "(mamba_ssm_cache_dtype 'float32' or 'auto') and DFlash2 speculative decoding with the "
    "targets' FP8 drafter history (speculative_config.context_kv_dtype 'auto' or 'fp8')"
)
_LM_HEAD_TP_IN_ADP = (
    "the DFlash2 candidate selector reads full-vocabulary draft logits: no enable_lm_head_tp_in_adp"
)


def _mxfp4_experts(pretrained_config) -> bool:
    quant = getattr(pretrained_config, "quantization_config", None)
    if quant is None:
        text = getattr(pretrained_config, "text_config", None)
        quant = getattr(text, "quantization_config", None)
    if not isinstance(quant, dict):
        quant = getattr(quant, "to_dict", lambda: {})()
    return quant.get("format") == "mxfp4-pack-quantized"


def _quantized_attention(model_config: ModelConfig, text) -> Optional[str]:
    """Why an attention projection resolves to a quantized recipe, or None.

    Resolved as the target resolves each projection: an exclusion from quantization keeps it
    BF16, else a per-layer entry names its recipe, else the global one (none under
    MIXED_PRECISION). The checkpoint's attention is BF16; the target has no FP8 block-scaled
    (or other quantized) attention projections or MLA absorption weights.
    """
    quant = model_config.quant_config
    declared = model_config.quant_config_dict or {}
    global_algo = getattr(quant, "quant_algo", None)
    if global_algo == QuantAlgo.MIXED_PRECISION:
        global_algo = None
    if global_algo is None and not declared:
        return None
    layer_kinds = getattr(text, "linear_attn_config", None) or {}
    mla_layers = set(layer_kinds.get("full_attn_layers", ()))
    kda_layers = set(layer_kinds.get("kda_layers", ()))
    for layer_idx in range(getattr(text, "num_hidden_layers", 0) or 0):
        # linear_attn_config numbers layers from 1.
        if layer_idx + 1 in mla_layers:
            projections = _MLA_PROJECTIONS
        elif layer_idx + 1 in kda_layers:
            projections = _KDA_PROJECTIONS
        else:
            continue
        for projection in projections:
            module = f"layers.{layer_idx}.self_attn.{projection}"
            if quant is not None and any(
                quant.is_module_excluded_from_quantization(prefix + module)
                for prefix in _EXCLUDED_LAYER_PREFIXES
            ):
                continue
            entries = [declared[p + module] for p in _LAYER_PREFIXES if p + module in declared]
            algo = entries[0].quant_algo if entries else global_algo
            if algo is not None:
                return (
                    f"attention projection model.{module} is quantized as {algo}: the Kimi K3.1 "
                    "target loads BF16 attention projections, as the checkpoint ships them (no "
                    "FP8 block-scaled MLA absorption); leave the attention projections out of "
                    "quantization, as the checkpoint's quantization_config does "
                    "(ignore 're:.*self_attn.*')"
                )
    return None


def _moe_layers(text) -> Tuple[int, ...]:
    """The 0-based layers the targets build a routed MoE on (the rest run the dense MLP)."""
    if getattr(text, "num_experts", None) is None:
        return ()
    first_moe = getattr(text, "first_k_dense_replace", 0) or 0
    freq = getattr(text, "moe_layer_freq", 1) or 1
    return tuple(
        layer
        for layer in range(getattr(text, "num_hidden_layers", 0) or 0)
        if layer >= first_moe and layer % freq == 0
    )


def _layer_layout_mismatch(text) -> Optional[str]:
    """Why the text config's layers are not the ones the targets build, or None.

    The targets' constructors require: every layer exactly one of KDA / MLA (``linear_attn_config``,
    numbered from 1), a DSA indexer on every MLA layer (the MLA serves DSA sparse attention only), a
    positive ``attn_res_block_size``, and on the routed-MoE layers the latent MoE
    (``routed_expert_hidden_size``, ``latent_moe_use_norm``) with positive SiTu betas, of which
    ``activation_situ_linear_beta`` must be set (``activation_situ_beta`` defaults to 1).
    """
    fix = "serve the Kimi K3.1 checkpoint's own config.json"
    kinds = getattr(text, "linear_attn_config", None)
    if not isinstance(kinds, dict):
        return f"the text config has no linear_attn_config (KDA / MLA layer kinds); {fix}"
    kda_layers = set(kinds.get("kda_layers") or ())
    mla_layers = set(kinds.get("full_attn_layers") or ())
    for layer in range(1, (getattr(text, "num_hidden_layers", 0) or 0) + 1):
        if (layer in kda_layers) == (layer in mla_layers):
            return (
                f"layer {layer} must be exactly one of linear_attn_config kda_layers / "
                f"full_attn_layers (the Kimi K3.1 layers are KDA or MLA); {fix}"
            )
    dsa_layers = set(text.dsa_config.get("dsa_layers") or ())
    dense = sorted(
        layer
        for layer in mla_layers
        if layer <= (getattr(text, "num_hidden_layers", 0) or 0) and layer not in dsa_layers
    )
    if dense:
        return (
            f"MLA layers {dense} are not in dsa_config.dsa_layers: the Kimi K3.1 MLA serves DSA "
            "sparse attention only; list every full_attn_layers entry in dsa_config.dsa_layers"
        )
    block = getattr(text, "attn_res_block_size", None)
    if not isinstance(block, int) or block <= 0:
        return (
            f"attn_res_block_size {block!r}: the Kimi K3.1 layers run the attention residual over "
            f"blocks of a positive number of layers; {fix}"
        )
    if not _moe_layers(text):
        return None
    if getattr(text, "routed_expert_hidden_size", None) is None:
        return (
            "the Kimi K3.1 routed experts are a latent MoE: the text config needs "
            f"routed_expert_hidden_size; {fix}"
        )
    if not getattr(text, "latent_moe_use_norm", False):
        return (
            "the Kimi K3.1 latent MoE normalizes its latent input: the text config needs "
            f"latent_moe_use_norm true; {fix}"
        )
    situ_beta = getattr(text, "activation_situ_beta", None)
    situ_linear_beta = getattr(text, "activation_situ_linear_beta", None)
    if situ_linear_beta is None:
        return (
            "the Kimi K3.1 routed experts' SiTu activation needs activation_situ_linear_beta "
            f"(the fused kernels have no identity linear branch); {fix}"
        )
    if (1.0 if situ_beta is None else situ_beta) <= 0 or situ_linear_beta <= 0:
        return (
            f"SiTu betas {situ_beta!r} and {situ_linear_beta!r}: the Kimi K3.1 routed experts take "
            f"positive betas; {fix}"
        )
    return None


def _text_quant_config(quant):
    """``quant`` with its exclusions mapped onto the text model, as the targets' wrapper maps them
    (``language_model.X`` to ``X``, ``language_model.layers.`` to ``model.layers.``)."""
    if quant is None or not quant.exclude_modules:
        return quant
    mapped = []
    for module in quant.exclude_modules:
        if module.startswith(_LANGUAGE_MODEL_PREFIX):
            module = module[len(_LANGUAGE_MODEL_PREFIX) :]
            if module.startswith("layers."):
                module = "model." + module
        mapped.append(module)
    quant = copy.copy(quant)
    quant.exclude_modules = mapped
    return quant


def _unquantized_routed_experts(model_config: ModelConfig, text) -> Optional[str]:
    """Why a routed-MoE layer's experts would load unquantized, or None.

    The targets read an exclusion from quantization as unquantized routed experts, and the loader has
    no checkpoint layout for those: the routed experts load only as MXFP4."""
    quant = _text_quant_config(model_config.quant_config)
    if quant is None:
        return None
    for layer in _moe_layers(text):
        for prefix in _ROUTED_EXPERT_PREFIXES:
            for suffix in _ROUTED_EXPERT_SUFFIXES:
                module = f"{prefix}layers.{layer}.{suffix}"
                if quant.is_module_excluded_from_quantization(module):
                    return (
                        f"routed experts {module} are excluded from quantization: the Kimi K3.1 "
                        "target loads MXFP4 routed experts only; drop the exclusion "
                        "(quantization_config ignore / exclude_modules) that names them"
                    )
    return None


def _draft_checkpoint_config(spec_config) -> Optional[dict]:
    """The drafter checkpoint's config.json, or None when it is not a readable local file."""
    path = os.path.join(str(getattr(spec_config, "speculative_model", None) or ""), "config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _dflash2_mla_mismatch(spec_config) -> Optional[str]:
    """Why ``spec_config`` is not one-engine DFlash with a DFlash2 MLA drafter, or None.

    Two-engine modes are refused: the engine treats their draft tokens by the attention backend's class
    (``extend_ctx`` / ``attention_need_spec_dec_mode``), and the K3.1 backend declaration is not the engine's
    TRT-LLM backend."""
    mode = spec_config.spec_dec_mode
    if not (mode.is_dflash() and mode.use_one_engine()):
        return (
            "the Kimi K3.1 target runs DFlash2 speculative decoding (one-engine DFlash with the "
            f"DFlash2 MLA drafter) or none, got {mode.name}"
        )
    variant = getattr(spec_config, "spec_dec_variant", None)
    if variant != 2:
        return (
            "the Kimi K3.1 target's DFlash drafter is a DFlash2 checkpoint (spec_dec_variant 2), "
            f"got spec_dec_variant {variant!r}; point speculative_model at the DFlash2 MLA drafter"
        )
    draft = _draft_checkpoint_config(spec_config)
    if draft is not None:
        attention_mode = (draft.get("dflash_config") or {}).get("attention_mode")
        if attention_mode != "mla":
            return (
                "the Kimi K3.1 target's DFlash2 drafter is the MLA one "
                f"(dflash_config.attention_mode 'mla'), got {attention_mode!r}"
            )
        if _declares_dspark_heads(draft):
            return (
                "the Kimi K3.1 target's DFlash2 drafter runs no DSpark heads, but the drafter declares them "
                "(projector_type 'dspark', shift_label, a confidence head or a Markov head)"
            )
    return None


def _declares_dspark_heads(draft: dict) -> bool:
    """Whether a drafter config.json asks for the DSpark head set, in any of the spellings DSpark checkpoints use
    (``dspark_config``, ``dflash_config``, top level, ``dspark_``-prefixed top level)."""

    def setting(*names):
        for name in names:
            for value in (
                (draft.get("dspark_config") or {}).get(name),
                (draft.get("dflash_config") or {}).get(name),
                draft.get(f"dspark_{name}"),
                draft.get(name),
            ):
                if value is not None:
                    return value
        return None

    return bool(
        str(setting("projector_type") or "").lower() == "dspark"
        or setting("shift_label")
        or setting("use_confidence_head", "enable_confidence_head")
        or int(setting("markov_rank") or 0) > 0
    )


def _moe_split(mapping) -> Tuple[int, int]:
    """(moe_tp, moe_ep) the target builds: the user's split, else EP-only (over TP x CP under Helix)."""
    if getattr(mapping, "moe_tp_ep_user_specified", False):
        return mapping.moe_tp_size, mapping.moe_ep_size
    return 1, mapping.tp_size * mapping.cp_size


def _mapping_mismatch(mapping) -> Optional[str]:
    """Why the target has no layout for ``mapping``, or None."""
    if mapping.pp_size != 1:
        return f"the Kimi K3.1 target has no pipeline parallelism (pp={mapping.pp_size})"
    if mapping.cp_size > 1 and not mapping.has_cp_helix():
        return (
            f"context parallelism {mapping.cp_config.get('cp_type', 'ULYSSES')}: the Kimi K3.1 "
            "target runs Helix CP (Ulysses CP is the sm_107 cep target's, on the ModelOpt NVFP4 "
            "export)"
        )
    if mapping.cp_size > 1 and (mapping.tp_size != 1 or mapping.enable_attention_dp):
        return "Kimi K3.1 Helix requires TP1 without attention DP"
    # Under Helix the CP ranks are TP ranks to everything but the MLA attention.
    tp = mapping.tp_size * mapping.cp_size
    moe_tp, moe_ep = _moe_split(mapping)
    if mapping.enable_attention_dp:
        if tp not in _ATTENTION_DP_SIZES:
            return f"attention DP over {tp} ranks: the target runs {_ATTENTION_DP_SIZES}"
        if moe_tp != 1:
            return f"attention DP runs the experts EP-only, got moe_tp={moe_tp}"
    elif tp not in _ATTENTION_TP_SIZES:
        return (
            f"attention TP {tp} without attention DP: the target runs {_ATTENTION_TP_SIZES} "
            f"(divisors of the {_NUM_HEADS} heads with an even drafter MLP)"
        )
    # Defensive: every Mapping the LLM arguments build, at every admitted size, passes the next three.
    if moe_tp * moe_ep != tp:
        return f"moe_tp {moe_tp} x moe_ep {moe_ep} must equal tp_size {tp}"
    if _NUM_HEADS % moe_tp:
        return f"moe_tp {moe_tp} must divide {_NUM_HEADS} (MXFP4 intermediate shard)"
    if _NUM_EXPERTS % moe_ep:
        return f"moe_ep {moe_ep} must divide the {_NUM_EXPERTS} experts"
    return None


def _layout(mapping) -> Tuple[bool, int, int, int]:
    """(attention DP, tp_size, moe_tp, moe_ep) of ``mapping``, as the target builds it."""
    return (bool(mapping.enable_attention_dp), mapping.tp_size, *_moe_split(mapping))


def per_layer_dynamic_eplb(config) -> bool:
    """Whether the load balancer configuration ``config`` is per-layer dynamic EPLB (``mode: per_layer_dynamic``)."""
    return getattr(config, "mode", "standard") == "per_layer_dynamic"


def per_layer_dynamic_mismatch(config, num_experts: int, moe_ep: int) -> Optional[str]:
    """Why per-layer dynamic EPLB with ``config`` cannot serve ``num_experts`` routed experts over ``moe_ep`` EP ranks,
    or ``None``: every rank keeps its ``num_experts / moe_ep`` resident experts and needs at least one helper slot,
    the same count on every rank."""
    num_slots = config.num_slots
    if num_slots <= num_experts or num_slots % moe_ep:
        return (
            f"moe_config.load_balancer per_layer_dynamic has num_slots={num_slots}: the Kimi K3.1 routed MoE needs "
            f"more slots than its {num_experts} experts, split evenly over the {moe_ep} EP ranks (each rank keeps "
            f"{num_experts // moe_ep} resident experts and gets (num_slots - {num_experts}) / {moe_ep} helper slots)"
        )
    return None


def static_placement_mismatch(config, num_experts: int) -> Optional[str]:
    """Why the load balancer configuration ``config`` is not a static placement of ``num_experts`` experts the fused
    routed MoE serves (one slot per expert, no online updates), or ``None``."""
    if per_layer_dynamic_eplb(config):
        return "moe_config.load_balancer is per-layer dynamic EPLB, not a static expert placement"
    if config.layer_updates_per_iter:
        return (
            f"moe_config.load_balancer has layer_updates_per_iter={config.layer_updates_per_iter}: the Kimi K3.1 "
            "routed MoE takes a static expert placement only (layer_updates_per_iter 0, no online EP load balancer)"
        )
    if config.num_slots != num_experts:
        return (
            f"moe_config.load_balancer has num_slots={config.num_slots}: the Kimi K3.1 routed MoE holds one slot "
            f"per expert (num_slots {num_experts}), no redundant expert slots"
        )
    return None


def _moe_mismatch(text, mapping) -> Optional[str]:
    """Why the routed experts cannot serve this checkpoint routing or exchange, or None."""
    if getattr(text, "moe_router_activation_func", "sigmoid") != "sigmoid":
        return f"Kimi K3.1 routes with sigmoid scores, got {text.moe_router_activation_func!r}"
    if not getattr(text, "moe_renormalize", True):
        return "Kimi K3.1 routing renormalizes the top-k weights (moe_renormalize)"
    groups = (getattr(text, "num_expert_group", 1) or 1, getattr(text, "topk_group", 1) or 1)
    if groups != (1, 1):
        return (
            f"Kimi K3.1 routes with one expert group, got num_expert_group, topk_group = {groups}"
        )
    inter = getattr(text, "moe_intermediate_size", None)
    if inter is not None and inter != _ROUTED_EXPERT_WIDTH:
        # Every admitted MoE TP split (1..32) then gives each rank whole 32-wide scale groups.
        return f"Kimi K3.1's routed experts are {_ROUTED_EXPERT_WIDTH} wide (moe_intermediate_size), got {inter}"
    if mapping.enable_attention_dp:
        forced = os.environ.get("TRTLLM_FORCE_COMM_METHOD")
        if forced is not None and forced.upper() not in _EXCHANGES:
            return (
                f"TRTLLM_FORCE_COMM_METHOD={forced}: the Kimi K3.1 routed MoE exchanges tokens with "
                f"{' or '.join(_EXCHANGES)} only"
            )
        if mpi_disabled():
            return "under attention DP the Kimi K3.1 routed MoE exchange runs in an MPI session (no Ray)"
    return None


def _engine_mismatch(llm_args: TorchLlmArgs, dep_perf_layout: bool, moe_ep: int) -> Optional[str]:
    """Why the engine settings rule the target out, or None. ``dep_perf_layout``: the deployment has the
    layout, GPU and FP8 latent KV cache of the attention-DP perf target; ``moe_ep``: its EP ranks."""
    moe = llm_args.moe_config
    if str(moe.backend).upper() not in ("AUTO", "TRTLLM"):
        return f"the target's routed experts run on the TRTLLM MoE backend, got {moe.backend}"
    if moe.load_balancer is not None:
        if not dep_perf_layout:
            return (
                "the target has a static contiguous expert partition: no MoE load balancer (EPLB); a static "
                "expert placement or per-layer dynamic EPLB (moe_config.load_balancer) runs on the sm_103 DEP16 "
                "perf target only (FP8 KV cache); drop moe_config.load_balancer or serve DEP16 on sm_103 with "
                "kv_cache_config.dtype='fp8'"
            )
        if default_target_requested():
            return (
                f"{DEFAULT_TARGET_ENV} selects the general target, which has a static contiguous "
                "expert partition: a static expert placement or per-layer dynamic EPLB (moe_config.load_balancer) "
                f"runs on the sm_103 DEP16 perf target; unset {DEFAULT_TARGET_ENV} or drop moe_config.load_balancer"
            )
        if per_layer_dynamic_eplb(moe.load_balancer):
            if (
                mismatch := per_layer_dynamic_mismatch(moe.load_balancer, _NUM_EXPERTS, moe_ep)
            ) is not None:
                return mismatch
        elif (mismatch := static_placement_mismatch(moe.load_balancer, _NUM_EXPERTS)) is not None:
            return mismatch
    if getattr(llm_args, "use_cute_dsl_bf16_gemm", False):
        return "the target's BF16 projections run cuBLAS GEMMs: no use_cute_dsl_bf16_gemm"
    low_m = os.environ.get("TRTLLM_LOW_M_GEMM_BACKEND", "off")
    if low_m.strip().lower().replace("_", "-") not in (
        "",
        "0",
        "none",
        "off",
        "cublas",
        "cublaslt",
    ):
        return (
            f"TRTLLM_LOW_M_GEMM_BACKEND={low_m!r}: the target's BF16 projections do not take the engine's low-M "
            "GEMM dispatcher; unset it"
        )
    kv = llm_args.kv_cache_config
    if kv.use_kv_cache_manager_v2 is False:
        return "the target's DSA indexer K-cache lives in KV cache manager V2"
    if (llm_args.max_beam_width or 1) > 1:
        return f"the target runs beam width 1, got {llm_args.max_beam_width}"
    if kv.enable_kv_pool_rebalance:
        return "the target runs without the KV pool rebalance (enable_kv_pool_rebalance)"
    if kv.tokens_per_block not in _TOKENS_PER_BLOCK:
        return f"tokens_per_block {kv.tokens_per_block} is not one of {_TOKENS_PER_BLOCK}"
    transceiver = llm_args.cache_transceiver_config
    if transceiver is not None and transceiver.backend is not None:
        backend, _ = transceiver._resolve_default_backend()
        if backend != "NIXL" or transceiver.transceiver_runtime not in ("PYTHON", "auto"):
            return (
                "disaggregated serving moves the KDA state and the indexer K-cache with the "
                "Python NIXL transceiver only (cache_transceiver_config backend NIXL, "
                f"transceiver_runtime PYTHON), got {backend} / {transceiver.transceiver_runtime}"
            )
    return None


def _kv_cache_quant_algo_name(model_config: ModelConfig, llm_args: TorchLlmArgs) -> Optional[str]:
    """The engine's KV-cache algorithm by name (None: BF16; a dtype the engine does not map stays
    the string given)."""
    algo = engine_kv_cache_quant_algo(model_config, llm_args)
    return algo.name if isinstance(algo, QuantAlgo) else algo


def disagg_role(llm_args: Any = None) -> str:
    """``context`` on a disaggregated context server, else ``generation``.

    ``trtllm-serve --server_role`` sets ``TRTLLM_DISAGG_ROLE`` in the environment and in
    ``llm_args.env_overrides``; the overrides are read first, since they also reach MPI ranks
    spawned before the variable was set. An aggregated server cannot be told from a generation
    server, and takes that role.
    """
    overrides = (
        llm_args.get("env_overrides")
        if isinstance(llm_args, _Mapping)
        else getattr(llm_args, "env_overrides", None)
    )
    role = (overrides or {}).get(DISAGG_ROLE_ENV)
    if role is None:
        role = os.environ.get(DISAGG_ROLE_ENV)
    return "context" if role == "context" else "generation"


def _lever_env(llm_args: Any) -> dict:
    """The variables of ``LEVER_ENV`` and ``LEVER_ENV_PREFIXES`` as the workers see them (the
    process environment, then the overrides); a variable that is not set is left out."""
    overrides = getattr(llm_args, "env_overrides", None) or {}
    environ = {**os.environ, **{k: str(v) for k, v in overrides.items()}}
    return {k: v for k, v in environ.items() if k in LEVER_ENV or k.startswith(LEVER_ENV_PREFIXES)}


def _names_indexer_k_dtype(llm_args: TorchLlmArgs) -> bool:
    """Whether the arguments' sparse attention config names the index-key dtype."""
    requested = llm_args.sparse_attention_config
    return requested is not None and "indexer_k_dtype" in requested.model_fields_set


def _indexer_k_dtype(model_config: ModelConfig, llm_args: TorchLlmArgs) -> str:
    """The index-key dtype the model is built with: the one the arguments set, else the loaded one.

    Routing also runs on the config read before the model defaults are applied, which is loaded
    without the arguments' sparse attention config: there the index-key dtype is the loader's
    default ('fp8'), as the checkpoint's dsa_config names none.
    """
    if _names_indexer_k_dtype(llm_args):
        return llm_args.sparse_attention_config.indexer_k_dtype
    return getattr(model_config.sparse_attention_config, "indexer_k_dtype", "fp8")


def _dsa_on_every_mla_layer(text) -> bool:
    """Whether the text config's DSA covers every MLA layer (both lists number layers from 1)."""
    mla_layers = set((getattr(text, "linear_attn_config", None) or {}).get("full_attn_layers", ()))
    dsa_layers = set((getattr(text, "dsa_config", None) or {}).get("dsa_layers") or ())
    return bool(mla_layers) and mla_layers <= dsa_layers


def _dsa_mismatch(model_config: ModelConfig, llm_args: TorchLlmArgs, text) -> Optional[str]:
    """Why a deployment of the ModelOpt export does not run DSA on every MLA layer with the TRTLLM
    attention backend, as every sm_107 target does, or None."""
    sparse = model_config.sparse_attention_config or llm_args.sparse_attention_config
    if getattr(sparse, "algorithm", None) != "dsa":
        return f"sparse attention {getattr(sparse, 'algorithm', None)!r}, not DSA"
    if not _dsa_on_every_mla_layer(text):
        return "the text config's dsa_config does not cover every MLA layer"
    if str(model_config.attn_backend).upper() != "TRTLLM":
        return f"attention backend {model_config.attn_backend}, not TRTLLM"
    return None


def _history_mismatch(spec_config) -> Optional[str]:
    """Why the drafter history is not the sm_107 targets' FP8 one, or None.

    The FP8 history is a constant of the targets: the config may leave it to them ("auto"; None
    where it has no such field) or name it ("fp8"), nothing else.
    """
    if (history := getattr(spec_config, "context_kv_dtype", None)) not in (None, "auto", "fp8"):
        return f"speculative_config.context_kv_dtype {history!r}"
    return None


def _history_only_mismatch(spec_config, layout: str, role: str) -> Optional[str]:
    """Why a server may not keep just the drafter history (``context_kv_only``), or None."""
    if getattr(spec_config, "context_kv_only", False) and role != "context":
        return (
            f"speculative_config.context_kv_only on a {layout} {role} server: it drafts, so it "
            "loads the whole drafter; only a context server keeps just the drafter history"
        )
    return None


def _specialized_recipe_mismatch(
    model_config: ModelConfig, llm_args: TorchLlmArgs, text
) -> Optional[str]:
    """Why a deployment of the ModelOpt export is not the specialized targets' recipe, or None."""
    if (mismatch := _dsa_mismatch(model_config, llm_args, text)) is not None:
        return mismatch
    kv_dtype = llm_args.kv_cache_config.dtype
    kv_algo = _kv_cache_quant_algo_name(model_config, llm_args)
    if kv_dtype != "fp8" or kv_algo != QuantAlgo.FP8.name:
        return f"kv_cache_config.dtype {kv_dtype!r} (KV cache {kv_algo})"
    if (index_k := _indexer_k_dtype(model_config, llm_args)) != "fp8":
        return f"indexer_k_dtype {index_k!r}"
    state = llm_args.kv_cache_config.mamba_ssm_cache_dtype
    # A Kimi model resolves 'auto' to float32.
    if state not in ("float32", "auto"):
        return f"mamba_ssm_cache_dtype {state!r}"
    spec_config = model_config.spec_config
    if spec_config is None:
        return "no speculative decoding"
    mode = spec_config.spec_dec_mode
    if mode.is_dspark():
        return "DSpark speculative decoding (the specialized targets draft with DFlash2)"
    if not (mode.is_dflash() and mode.use_one_engine()):
        return f"speculative decoding {mode.name}"
    if (mismatch := _dflash2_mla_mismatch(spec_config)) is not None:
        return mismatch
    return _history_mismatch(spec_config)


def _draft_schedule_mismatch(
    spec_config, targets: str = "the specialized targets"
) -> Optional[str]:
    """Why the speculative config does not draft a fixed block on every step, or None.

    ``targets`` names the targets the reason is about.
    """
    # A fixed block on every step and request (ragged verification comes only with DSpark).
    if (
        getattr(spec_config, "draft_len_schedule", None) is not None
        or getattr(spec_config, "max_concurrency", None) is not None
    ):
        return (
            "DFlash2 verifies a fixed block (the KDA replay width): no draft_len_schedule or "
            "max_concurrency"
        )
    if (
        getattr(spec_config, "acceptance_rate_threshold", None) is not None
        or getattr(spec_config, "acceptance_rate_window_size", None) is not None
    ):
        return (
            f"{targets} draft on every step: no acceptance_rate_threshold or "
            "acceptance_rate_window_size"
        )
    return None


def _engine_settings_mismatch(
    llm_args: TorchLlmArgs, targets: str = "the specialized targets", eplb_ep: Optional[int] = None
) -> Optional[str]:
    """Why the engine settings rule the sm_107 targets out, or None. ``targets`` names the targets
    the reason is about; ``eplb_ep``: the EP ranks of a layout whose target takes per-layer dynamic
    EPLB (``_DYNAMIC_EPLB``), None where it takes no load balancer."""
    if str(llm_args.prefill_cuda_graph_backend.value) != "disabled":
        return (
            f"prefill_cuda_graph_backend {str(llm_args.prefill_cuda_graph_backend.value)!r}: "
            f"{targets} run prefill eagerly ('piecewise' torch.compiles the model, and no graph "
            "masks the MoE padding rows)"
        )
    if llm_args.torch_compile_config is not None:
        return f"{targets}' kernels are not torch ops: no torch_compile_config"
    balancer = llm_args.moe_config.load_balancer
    if balancer is not None:
        if eplb_ep is None or not per_layer_dynamic_eplb(balancer):
            return (
                f"{targets} have a static expert partition: no MoE load balancer (per-layer dynamic "
                "EPLB runs on the cep and dep16 context servers)"
            )
        if (mismatch := per_layer_dynamic_mismatch(balancer, _NUM_EXPERTS, eplb_ep)) is not None:
            return mismatch
    kv = llm_args.kv_cache_config
    if kv.use_kv_cache_manager_v2 is False:
        return f"{targets} keep the index keys and the drafter history in KV-V2"
    if (llm_args.max_beam_width or 1) > 1:
        return f"{targets} run beam width 1, got {llm_args.max_beam_width}"
    if kv.enable_kv_pool_rebalance:
        return f"{targets} run without the KV pool rebalance"
    # Read only where the arguments set it: the target's model defaults apply after the first
    # routing pass.
    if (
        "tokens_per_block" in kv.model_fields_set
        and kv.tokens_per_block != _SPECIALIZED_TOKENS_PER_BLOCK
    ):
        return (
            f"tokens_per_block {kv.tokens_per_block}: {targets} run "
            f"{_SPECIALIZED_TOKENS_PER_BLOCK}-token pages"
        )
    transceiver = llm_args.cache_transceiver_config
    if transceiver is not None and transceiver.backend is not None:
        backend, _ = transceiver._resolve_default_backend()
        if backend != "NIXL" or transceiver.transceiver_runtime not in ("PYTHON", "auto"):
            return (
                "disaggregated serving moves the KDA state, the index keys and the drafter "
                "history with the Python NIXL transceiver only, got "
                f"{backend} / {transceiver.transceiver_runtime}"
            )
    return None


def _specialized_engine_mismatch(
    llm_args: TorchLlmArgs, spec_config, eplb_ep: Optional[int] = None
) -> Optional[str]:
    """Why the engine settings rule the specialized targets out, or None."""
    return _draft_schedule_mismatch(spec_config) or _engine_settings_mismatch(
        llm_args, eplb_ep=eplb_ep
    )


def _helix8_mismatch(llm_args: TorchLlmArgs) -> Optional[str]:
    """Why a helix8 generation server's settings rule it out, or None."""
    transceiver = llm_args.cache_transceiver_config
    if transceiver is None or transceiver.backend is None:
        return "helix8 serves a disaggregated generation server: set cache_transceiver_config"
    kv = llm_args.kv_cache_config
    # Read only where the arguments set it: the target's model defaults turn it off.
    if "enable_block_reuse" in kv.model_fields_set and kv.enable_block_reuse:
        return "a helix8 generation server runs without block reuse"
    return None


def _cp_kind(mapping) -> Optional[str]:
    """``helix`` or ``ulysses`` for a context-parallel ``mapping`` of that CP type, else None."""
    if mapping.has_cp_helix():
        return "helix"
    if mapping.has_cp_ulysses():
        return "ulysses"
    return None


def _specialized_layout(mapping) -> Optional[str]:
    """``cep8``, ``cep16``, ``dep16``, ``dep12``, ``dep8``, ``helix8`` or ``tep8``, or None."""
    if mapping.pp_size != 1:
        return None
    if mapping.has_cp_helix() and not getattr(mapping, "moe_tp_ep_user_specified", False):
        return None
    facts = (
        mapping.world_size,
        mapping.tp_size,
        mapping.cp_size,
        _cp_kind(mapping),
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    found = [name for name, want in _SPECIALIZED_LAYOUTS.items() if facts == want]
    return found[0] if found else None


def _cep_refusal(mapping) -> Optional[str]:
    """Why no cep target takes a cep-shaped layout (TP 1, Ulysses CP N without attention DP, MoE EP
    N) whose N is not in ``_CEP_SIZES``; None for another layout."""
    n = mapping.world_size
    facts = (
        mapping.tp_size,
        mapping.pp_size,
        mapping.cp_size,
        _cp_kind(mapping),
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    if facts != (1, 1, n, "ulysses", False, 1, n) or n in _CEP_SIZES:
        return None
    if _NUM_HEADS % n or _NUM_EXPERTS % n:
        return f"cep{n}: {n} ranks do not split the {_NUM_HEADS} KDA heads and the {_NUM_EXPERTS} experts"
    if n < min(_CEP_SIZES):
        return (
            f"cep{n}: a rank would hold {_NUM_EXPERTS // n} routed experts beside the whole "
            "attention and MLP weights, more than one Rubin GPU's memory"
        )
    return f"cep{n}: no sm_107 run of a context-parallel server over {n} ranks yet"


def _roles_of(layout: str) -> Tuple[str, ...]:
    """The server roles the specialized target of ``layout`` serves."""
    return tuple(role for name, role in _SPECIALIZED if name == layout)


def _role_refusal(layout: str, role: str) -> str:
    """Why the specialized target of ``layout`` does not serve a ``role`` server."""
    return f"{layout} serves {' and '.join(_roles_of(layout))} servers only, not a {role} server"


def _layout_facts(mapping) -> str:
    """The parallel layout as a refusal names it."""
    return (
        f"layout world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} helix={bool(mapping.has_cp_helix())} "
        f"ulysses={bool(mapping.has_cp_ulysses())} "
        f"attention_dp={bool(mapping.enable_attention_dp)} moe_tp={mapping.moe_tp_size} "
        f"moe_ep={mapping.moe_ep_size}"
    )


def _specialized_decision(
    model_config: ModelConfig, llm_args: TorchLlmArgs, text
) -> Tuple[Optional[str], Optional[str]]:
    """(layout, None) when a specialized target takes a deployment of the export on sm_107, else
    (None, why none does): the reason of the first check that fails, in this order."""
    from .config import modelopt_nvfp4_mismatch

    if (mismatch := modelopt_nvfp4_mismatch(model_config)) is not None:
        return None, f"{mismatch} ({_SPECIALIZED_RECIPE})"
    if (mismatch := _specialized_recipe_mismatch(model_config, llm_args, text)) is not None:
        return None, f"{mismatch} ({_SPECIALIZED_RECIPE})"
    mapping = model_config.mapping
    layout = _specialized_layout(mapping)
    if layout is None:
        cep = _cep_refusal(mapping)
        return None, (
            f"{_layout_facts(mapping)}: the specialized targets run cep8 and cep16 (TP 1, Ulysses "
            "CP N without attention DP, MoE EP N; context servers), dep16 (attention DP 16, MoE "
            "EP 16), dep12 (attention DP 12, MoE EP 12, context servers), dep8 (attention DP 8, "
            "MoE EP 8, context servers), helix8 (Helix CP 8, MoE TP 2 x EP 4, set by the "
            "arguments) and tep8 (TP 8, MoE EP 8)" + ("" if cep is None else f"; {cep}")
        )
    role = disagg_role(llm_args)
    if (layout, role) not in _SPECIALIZED:
        return None, _role_refusal(layout, role)
    backends, draft_lens = _SPECIALIZED[(layout, role)]
    spec_config = model_config.spec_config
    if (draft_len := spec_config.max_draft_len) not in draft_lens:
        why = f"; {_LONGER_DRAFT}" if draft_len > max(_DRAFT_LENS) else ""
        return None, (
            f"max_draft_len {draft_len}: a {layout} {role} server runs one of {draft_lens}{why}"
        )
    if (mismatch := _history_only_mismatch(spec_config, layout, role)) is not None:
        return None, mismatch
    backend = str(llm_args.moe_config.backend).upper()
    if backend not in backends:
        return None, (
            f"moe_config.backend {llm_args.moe_config.backend}: a {layout} {role} server runs "
            f"{' or '.join(backends)}"
        )
    # Per-layer dynamic EPLB serves the CuteDSL experts behind the all-to-all only.
    eplb = (layout, role) in _DYNAMIC_EPLB and backend == "CUTEDSL"
    eplb_ep = _SPECIALIZED_LAYOUTS[layout][6] if eplb else None
    if (mismatch := _specialized_engine_mismatch(llm_args, spec_config, eplb_ep)) is not None:
        return None, mismatch
    if mapping.enable_attention_dp and llm_args.enable_lm_head_tp_in_adp:
        return None, _LM_HEAD_TP_IN_ADP
    if layout == "helix8" and (mismatch := _helix8_mismatch(llm_args)) is not None:
        return None, mismatch
    return layout, None


def _route_nvfp4_default(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Resolve the configurable NVFP4 target on its supported architectures."""
    from .kimi_k31_pro_nvfp4__sm_10x__default.routing import validate_target

    validate_target(model_config, llm_args, compute_capability)
    # fmt: off
    # isort: off
    from .kimi_k31_pro_nvfp4__sm_10x__default.modeling import ModelingV2KimiK31ProNvfp4Sm10xDefault
    # isort: on
    # fmt: on
    return ModelingV2KimiK31ProNvfp4Sm10xDefault


def _route_modelopt(
    model_config: ModelConfig,
    llm_args: TorchLlmArgs,
    compute_capability: Tuple[int, int],
    text,
) -> type:
    """Prefer eligible SM107 specialized targets, otherwise use the configurable default."""
    if compute_capability in ((10, 0), (10, 3)):
        return _route_nvfp4_default(model_config, llm_args, compute_capability)
    if compute_capability != _RUBIN:
        raise NoTarget(
            "the ModelOpt NVFP4 export requires sm_100, sm_103, or sm_107; got "
            f"sm_{compute_capability[0]}{compute_capability[1]}"
        )
    layout, refusal = _specialized_decision(model_config, llm_args, text)
    if (
        refusal is None
        and not getattr(llm_args, "disable_mm_encoder", False)
        and (layout, disagg_role(llm_args)) not in _VISION_CELLS
    ):
        refusal = (
            "the specialized targets serve text-only deployments outside the dep16, dep12 and "
            "dep8 context servers"
        )
    if refusal is None and getattr(model_config.spec_config, "context_kv_dtype", None) != "fp8":
        refusal = "the specialized targets require explicit FP8 draft history"
    if refusal is not None:
        try:
            return _route_nvfp4_default(model_config, llm_args, compute_capability)
        except NoTarget as mismatch:
            raise NoTarget(f"{refusal}; configurable default: {mismatch}") from mismatch
    model_config.extra_attrs[DISAGG_ROLE_ATTR] = disagg_role(llm_args)
    # One line each on purpose: the claims suite reads the routed targets off
    # `from .<target>.modeling import <Class>` lines.
    # fmt: off
    # isort: off
    if layout.startswith("cep"):
        from .kimi_k31_pro_nvfp4__sm_107__cep.modeling import ModelingV2KimiK31ProNvfp4Sm107Cep
        target = ModelingV2KimiK31ProNvfp4Sm107Cep
    elif layout in ("dep16", "dep12", "dep8"):
        from .kimi_k31_pro_nvfp4__sm_107__dep16.modeling import ModelingV2KimiK31ProNvfp4Sm107Dep16
        target = ModelingV2KimiK31ProNvfp4Sm107Dep16
    elif layout == "helix8":
        from .kimi_k31_pro_nvfp4__sm_107__helix8.modeling import ModelingV2KimiK31ProNvfp4Sm107Helix8
        target = ModelingV2KimiK31ProNvfp4Sm107Helix8
    else:
        from .kimi_k31_pro_nvfp4__sm_107__tep8.modeling import ModelingV2KimiK31ProNvfp4Sm107Tep8
        target = ModelingV2KimiK31ProNvfp4Sm107Tep8
    # isort: on
    # fmt: on
    model_config.extra_attrs[LEVER_ENV_ATTR] = _lever_env(llm_args)
    return target


def _route_ull_generation(
    model_config: ModelConfig, llm_args: TorchLlmArgs, indexer_k_dtype: str
) -> type:
    """The ULL generation server (Helix CP16, MoE TP16 / EP1):
    ``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1`` with its ``ull16`` profile, or under
    ``TLLM_K31_HELIX16_WIDE`` its wide-batch fork ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` on
    the same profile. Validates the recipe (``decide``) and selects the profile before the
    target's model modules are imported."""
    from .ops.common import rubin_ull_features as ull

    mapping = model_config.mapping
    spec = model_config.spec_config
    mode = spec.spec_dec_mode.name if spec is not None else None
    role = disagg_role(llm_args)
    if llm_args.moe_config.load_balancer is not None:
        raise NoTarget("K3.1 ULL requires static experts; EPLB is unsupported")
    if llm_args.kv_cache_config.use_kv_cache_manager_v2 is False:
        raise NoTarget("K3.1 ULL requires KV cache manager V2")
    if (llm_args.max_beam_width or 1) != 1:
        raise NoTarget("K3.1 ULL requires beam width 1")
    if not mapping.moe_tp_ep_user_specified:
        raise NoTarget("K3.1 ULL requires explicit MoE TP16 / EP1")
    try:
        ull.validate_native_config(model_config, check_kv=False)
    except ValueError as error:
        raise NoTarget(str(error)) from error
    if llm_args.kv_cache_config.mamba_ssm_cache_dtype != "float32":
        raise NoTarget("K3.1 ULL requires FP32 KDA recurrent state")
    if llm_args.moe_config.use_low_precision_moe_combine:
        raise NoTarget("K3.1 ULL forbids low precision MoE combine")
    backend = str(llm_args.moe_config.backend).upper()
    kv_algo = _kv_cache_quant_algo_name(model_config, llm_args)
    # Read once: the recipe check (the wide-batch fork also verifies k3 / k4 / k5, ``decide_wide``; the
    # ULL target k7 only), the profile below and the import after it name the same target.
    wide = helix16_wide_requested()
    decision = (ull.decide_wide if wide else ull.decide)(
        indexer_k_dtype=indexer_k_dtype,
        kv_cache_quant_algo=kv_algo,
        spec_mode=mode,
        max_draft_len=spec.max_draft_len if spec is not None else None,
        moe_backend=backend,
        layout=ull.Layout(
            world_size=mapping.world_size,
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            cp_size=mapping.cp_size,
            helix=bool(mapping.has_cp_helix()),
            attention_dp=bool(mapping.enable_attention_dp),
            moe_tp_size=mapping.moe_tp_size,
            moe_ep_size=mapping.moe_ep_size,
        ),
        role=role,
    )
    if decision.reason is not None:
        raise NoTarget(decision.reason)
    if decision.topology != "ull16":
        raise NoTarget(f"the ULL generation layout decided {decision.topology}, not ull16")
    # The drafter: the DFlash2 MLA checkpoint, the FP8 history, a whole drafter, a fixed block.
    for mismatch in (
        _dflash2_mla_mismatch(spec),
        _history_mismatch(spec),
        _history_only_mismatch(spec, decision.topology, role),
        _draft_schedule_mismatch(spec, "the ULL generation target"),
    ):
        if mismatch is not None:
            raise NoTarget(f"{mismatch} ({ull.RECIPE})")
    key = ull.profile_key(decision.topology, role, llm_args.max_batch_size, spec.max_draft_len)
    name = (ull.WIDE_TARGETS if wide else ull.TARGETS)[decision.topology]
    ull.select(key)
    logger.info(
        f"modeling_v2 kimi_k31: {name} profile {key} "
        f"rank={mapping.rank} world_size={mapping.world_size} tp={mapping.tp_size} "
        f"cp={mapping.cp_size} attention_dp={mapping.enable_attention_dp} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} moe_backend={backend} "
        f"kv={kv_algo} spec={mode}; ULL flat generation"
    )
    if wide:
        model_config.extra_attrs[HELIX16_WIDE_ATTR] = helix16_wide_options()
    # One line each on purpose: the claims suite reads the routed targets off
    # `from .<target>.modeling import <Class>` lines.
    # fmt: off
    # isort: off
    if wide:
        from .kimi_k31_pro_nvfp4__sm_107__helix16_wide.modeling import ModelingV2KimiK31ProNvfp4Sm107Helix16Wide
        return ModelingV2KimiK31ProNvfp4Sm107Helix16Wide
    from .kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1.modeling import ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1
    # isort: on
    # fmt: on
    return ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1


@routes("KimiK31ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    A refusal is also logged as a warning that carries the reason and its fix: Kimi K3.1 has no
    Modeling V1 implementation to fall back to, so under ``modeling_v2: auto`` the engine would
    otherwise report only an unknown architecture.
    """
    try:
        target = _route(model_config, llm_args, compute_capability)
    except NoTarget as why:
        logger.warning(
            f"modeling_v2: no Kimi K3.1 target fits this deployment, and Kimi K3.1 has no "
            f"Modeling V1 implementation to fall back to: {why}"
        )
        raise
    logger.info(
        f"modeling_v2: 'KimiK31ForConditionalGeneration' routes to {target.__module__}.{target.__qualname__}"
    )
    return target


def _route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """The target class for this deployment, or ``NoTarget`` with the reason none fits."""
    pretrained_config = model_config.pretrained_config
    text = getattr(pretrained_config, "text_config", None) or pretrained_config
    shape = tuple(getattr(text, name, None) for name in _SHAPE_FIELDS)
    if shape != _SHAPE:
        raise NoTarget(f"text config shape {shape} is not Kimi K3.1's {_SHAPE}")
    if getattr(text, "dsa_config", None) is None:
        raise NoTarget("the Kimi K3.1 target needs the text config's dsa_config")
    mapping = model_config.mapping
    # The ULL generation layout (Helix CP16, attention TP1, MoE TP16 / EP1): no other target
    # serves it.
    if compute_capability in _ULL_COMPUTE_CAPABILITIES and (
        mapping.world_size == 16
        and mapping.tp_size == 1
        and mapping.pp_size == 1
        and mapping.cp_size == 16
        and mapping.has_cp_helix()
        and not mapping.enable_attention_dp
        and (mapping.moe_tp_size, mapping.moe_ep_size) == (16, 1)
    ):
        return _route_ull_generation(
            model_config, llm_args, _indexer_k_dtype(model_config, llm_args)
        )
    mxfp4 = _mxfp4_experts(pretrained_config)
    if (
        not mxfp4
        and getattr(model_config.quant_config, "quant_algo", None) == QuantAlgo.MIXED_PRECISION
    ):
        return _route_modelopt(model_config, llm_args, compute_capability, text)
    if (mismatch := _layer_layout_mismatch(text)) is not None:
        raise NoTarget(mismatch)
    if not mxfp4:
        raise NoTarget(
            "the Kimi K3.1 target needs the MXFP4 checkpoint (mxfp4-pack-quantized), or on "
            "sm_107 the ModelOpt NVFP4 export"
        )
    if (mismatch := _quantized_attention(model_config, text)) is not None:
        raise NoTarget(mismatch)
    if (mismatch := _unquantized_routed_experts(model_config, text)) is not None:
        raise NoTarget(mismatch)
    if compute_capability in _NOT_VALIDATED:
        raise NoTarget(_NOT_VALIDATED[compute_capability])
    if compute_capability not in _COMPUTE_CAPABILITIES:
        raise NoTarget(
            f"the Kimi K3.1 target runs on sm_103 (GB300, B300) and sm_107 (Rubin), "
            f"got {compute_capability}"
        )
    # The loader fills the DSA config from the checkpoint's dsa_config when the arguments name none.
    sparse = model_config.sparse_attention_config or llm_args.sparse_attention_config
    if getattr(sparse, "algorithm", None) != "dsa":
        raise NoTarget(
            "the Kimi K3.1 target runs DSA sparse attention: set sparse_attention_config "
            f"algorithm 'dsa' (got {getattr(sparse, 'algorithm', None)!r})"
        )
    # The arguments' index-key dtype wins: the loader has not applied it to the config yet.
    if (index_k := _indexer_k_dtype(model_config, llm_args)) != "fp8":
        raise NoTarget(
            f"indexer_k_dtype {index_k!r}: the Kimi K3.1 targets run FP8 index keys only; "
            "set sparse_attention_config.indexer_k_dtype='fp8'"
        )
    if getattr(sparse, "use_cute_dsl_topk", False):
        raise NoTarget(
            "the Kimi K3.1 DSA indexer runs the GVR / CUDA radix Top-K: no "
            "sparse_attention_config.use_cute_dsl_topk"
        )
    attn_backend = getattr(llm_args, "attn_backend", "TRTLLM")
    if str(attn_backend).upper() != "TRTLLM":
        raise NoTarget(
            f"attn_backend {attn_backend!r}: the Kimi K3.1 targets run the TRTLLM attention backend"
        )
    index_rope_dim = text.dsa_config.get("index_rope_dim")
    if index_rope_dim != 0:
        raise NoTarget(
            f"dsa_config index_rope_dim {index_rope_dim!r}: the Kimi K3.1 DSA indexer is position-free "
            "(index_rope_dim 0)"
        )
    kv_dtype = llm_args.kv_cache_config.dtype
    kv_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    # 'auto' over a checkpoint that declares no KV-cache scheme keeps the checkpoint's BF16 latents, which only the
    # general target serves.
    bf16_latents = kv_dtype == "auto" and kv_algo is None
    if kv_algo != QuantAlgo.FP8 and not bf16_latents:
        raise NoTarget(
            f"kv_cache_config.dtype={kv_dtype!r} stores {kv_algo} latents: the Kimi K3.1 targets run an FP8 "
            "latent KV cache (kv_cache_config.dtype='fp8') or the checkpoint's BF16 latents ('auto')"
        )
    state_dtype = llm_args.kv_cache_config.mamba_ssm_cache_dtype
    if state_dtype not in _KDA_STATE_DTYPES:
        raise NoTarget(
            f"kv_cache_config.mamba_ssm_cache_dtype={state_dtype!r}: the Kimi K3.1 targets keep the KDA "
            "recurrent state in FP32 ('auto' or 'float32')"
        )
    spec_config = model_config.spec_config
    if spec_config is not None:
        if (mismatch := _dflash2_mla_mismatch(spec_config)) is not None:
            raise NoTarget(mismatch)
        context_kv_dtype = getattr(spec_config, "context_kv_dtype", None)
        if context_kv_dtype not in _DRAFT_CONTEXT_KV_DTYPES:
            raise NoTarget(
                f"speculative_config.context_kv_dtype={context_kv_dtype!r}: the Kimi K3.1 DFlash2 drafter "
                "keeps its context latents in its own dtype; set context_kv_dtype='auto' or leave it unset"
            )
        if getattr(spec_config, "draft_len_schedule", None) is not None:
            raise NoTarget(
                "DFlash2 verifies a fixed block (the KDA replay width): no draft_len_schedule"
            )
        if (
            getattr(spec_config, "acceptance_rate_window_size", None) is not None
            or getattr(spec_config, "acceptance_rate_threshold", None) is not None
        ):
            raise NoTarget(
                "DFlash2 runs without the speculation gate (acceptance_rate_window_size / "
                "acceptance_rate_threshold): a trip turns drafting off mid-flight, which moves "
                "helix requests to a one-token-per-step position formula (wrong positions and "
                "KV slots, no error) and changes the KDA replay width"
            )
        if model_config.mapping.enable_attention_dp and llm_args.enable_lm_head_tp_in_adp:
            raise NoTarget(
                "the DFlash2 candidate selector reads full-vocabulary draft logits: no "
                "enable_lm_head_tp_in_adp with speculative decoding"
            )
    if (mismatch := _mapping_mismatch(model_config.mapping)) is not None:
        raise NoTarget(mismatch)
    if model_config.mapping.has_cp_helix():
        if llm_args.kv_cache_config.enable_block_reuse:
            raise NoTarget("Kimi K3.1 Helix requires block reuse disabled")
    if (mismatch := _moe_mismatch(text, model_config.mapping)) is not None:
        raise NoTarget(mismatch)
    dep_perf_layout = (
        compute_capability == _PERF_COMPUTE_CAPABILITY
        and _layout(model_config.mapping) in _SM103_DEP_LAYOUTS
        and not bf16_latents
    )
    if (
        mismatch := _engine_mismatch(llm_args, dep_perf_layout, _moe_split(model_config.mapping)[1])
    ) is not None:
        raise NoTarget(mismatch)
    # One line per import on purpose: the claims suite reads the routed targets off
    # `from .<target>.modeling import <Class>` lines.
    # fmt: off
    # isort: off
    if bf16_latents:
        logger.info(
            "modeling_v2: Kimi K3.1 with kv_cache_config.dtype='auto' keeps the checkpoint's BF16 latent KV cache, "
            "which the general target serves; kv_cache_config.dtype='fp8' selects the sm_103 perf targets"
        )
    elif compute_capability == _PERF_COMPUTE_CAPABILITY and not default_target_requested():
        layout = _layout(model_config.mapping)
        if layout in _SM103_TEP_LAYOUTS:
            from .kimi_k31_pro_mxfp4__sm_103__tep.modeling import ModelingV2KimiK31ProMxfp4Sm103Tep
            return ModelingV2KimiK31ProMxfp4Sm103Tep
        if layout in _SM103_DEP_LAYOUTS:
            from .kimi_k31_pro_mxfp4__sm_103__dep.modeling import ModelingV2KimiK31ProMxfp4Sm103Dep
            return ModelingV2KimiK31ProMxfp4Sm103Dep
    from .kimi_k31_pro_mxfp4__sm_10x__default.modeling import ModelingV2KimiK31ProMxfp4Sm10xDefault
    # isort: on
    # fmt: on
    return ModelingV2KimiK31ProMxfp4Sm10xDefault


def _modelopt_export(model_dir: str) -> bool:
    """Whether the checkpoint in ``model_dir`` is the ModelOpt export (``MIXED_PRECISION`` in its
    ``hf_quant_config.json``, else in its config.json's quantization_config)."""
    path = os.path.join(model_dir, "hf_quant_config.json")
    if os.path.isfile(path):
        with open(path) as f:
            return (json.load(f).get("quantization") or {}).get("quant_algo") == "MIXED_PRECISION"
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        config = json.load(f)
    quant = config.get("quantization_config") or (config.get("text_config") or {}).get(
        "quantization_config"
    )
    return isinstance(quant, dict) and quant.get("quant_algo") == "MIXED_PRECISION"


@dflash_context_for("KimiK31ForConditionalGeneration")
def dflash_context_bytes_per_slot(
    llm_args: TorchLlmArgs, draft_config: Optional[dict]
) -> Optional[int]:
    """Per request slot, the DFlash2 MLA drafter's context the target's worker allocates.

    The MXFP4 targets' worker keeps one 576-wide latent per layer in a ring bounded by the
    drafter's window (``ops/spec/dflash2_ring``), not the engine's dense ``max_seq_len`` x heads x
    head_dim arena. The specialized targets (the ModelOpt export) keep the history in the unified
    draft cache, pages of the KV cache manager V2 pool, and allocate nothing per slot.
    """
    if draft_config is None:
        return None
    if _modelopt_export(str(llm_args.model)):
        return 0
    from .ops.spec.dflash2_ring import dflash2_ctx_bytes_per_slot

    return dflash2_ctx_bytes_per_slot(
        draft_config, llm_args.max_seq_len, llm_args.speculative_config.max_draft_len
    )
