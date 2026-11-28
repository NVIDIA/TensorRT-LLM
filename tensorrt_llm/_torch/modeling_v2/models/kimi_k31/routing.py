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
it also admits a static expert placement from the engine's EP load balancer configuration
(``moe_config.load_balancer`` with one slot per expert, ``layer_updates_per_iter`` 0 and a
permutation of the experts for every routed-MoE layer, as
``examples/wide_ep/ep_load_balancer/generate_eplb_config.py`` writes it; a placement other than the
identity takes ``max_num_tokens`` up to 256). Every other admitted
deployment, every sm_107 one and every one with BF16 latents runs the general target, which takes no
static expert placement.

The ModelOpt NVFP4 export (routed experts NVFP4, shared experts and KDA projections MXFP8, the rest
BF16; ``config.modelopt_nvfp4_mismatch``) runs on sm_103 and sm_107; the general target and the sm_103
perf targets never take it. On sm_107 three specialized targets come first,
``kimi_k31_pro_nvfp4__sm_107__{dep16,helix8,tep8}``. Each is one layout of one recipe: an FP8
latent KV cache, FP8 index keys, an FP32 KDA state and DFlash2 with an FP8 drafter history. They
take the layouts, server roles, MoE backends and draft lengths of ``_SPECIALIZED`` and nothing
else; a context server keeps just the drafter history (``context_kv_only``). Their drafter
history lives in the unified draft cache of KV cache manager V2. The server role is the
one the runtime publishes for the server's ranks (:func:`server_role`); the targets read the same
role at construction.

``kimi_k31_pro_nvfp4__sm_10x__default`` takes the export on sm_103, and on sm_107 the deployments
the specialized targets refuse, on its validated cells only (``_NVFP4_DEFAULT_CELLS``): TP8 with MoE
EP8 (``tep8``) or attention DP16 with MoE EP16 (``dep16``) as aggregated servers on both GPUs, and on
sm_103 also as disaggregated servers (TP8 context servers without the vision tower; TP8 or DP16
generation servers, and Helix generation servers over CP 8, ``helix8``, without block reuse); the
CUTEDSL MoE backend (``AUTO`` when the arguments leave it unset), an FP8 latent KV cache, FP8 index
keys, an FP32 KDA state and DFlash2 speculative decoding, whose drafter history stays in the
drafter's BF16. Everything else is refused as not validated.

``modeling_v2="off"`` has no built-in K3.1 implementation to fall back to.
"""

from __future__ import annotations

import copy
import json
import os
from typing import TYPE_CHECKING, Optional, Tuple

from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, input_processor_for, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

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
# Rows (EP ranks x the step's largest per-rank token count) the DEP perf target's fused routed MoE
# dispatches in one call (``MAX_ROWS`` of its ``ops/moe/w4a8_ep_moe_fused``): a placed expert runs
# on that path only.
_PLACEMENT_MAX_ROWS = 4096
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
# The specialized targets' layouts, as (world_size, tp, cp, Helix, attention DP, moe_tp, moe_ep).
_SPECIALIZED_LAYOUTS = {
    "dep16": (16, 16, 1, False, True, 1, 16),
    "helix8": (8, 1, 8, True, False, 2, 4),
    "tep8": (8, 8, 1, False, False, 1, 8),
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
# The longest draft (max_draft_len) any target verifies.
_MAX_DRAFT_LEN = 7
# (layout, server role) -> (MoE backend, draft lengths) of the specialized targets. The target
# builds that backend whatever moe_config asks, so the arguments must name it. An aggregated
# server takes the generation role.
_SPECIALIZED = {
    ("dep16", "context"): ("CUTEDSL", _DRAFT_LENS),
    ("dep16", "generation"): ("MEGAMOE_CUTEDSL", _DRAFT_LENS),
    ("helix8", "generation"): ("CUTEDSL_FC12", _DRAFT_LENS),
    ("tep8", "context"): ("CUTEDSL", _DRAFT_LENS),
    ("tep8", "generation"): ("CUTEDSL_FC12", _DRAFT_LENS),
}
# The specialized targets' KV pages (their get_model_defaults).
_SPECIALIZED_TOKENS_PER_BLOCK = 64
_SPECIALIZED_RECIPE = (
    "the sm_107 specialized targets run the ModelOpt NVFP4 export with an FP8 latent KV cache "
    "(kv_cache_config.dtype 'fp8'), FP8 index keys, an FP32 KDA state "
    "(mamba_ssm_cache_dtype 'float32' or 'auto') and DFlash2 speculative decoding with the "
    "targets' FP8 drafter history"
)
# The NVFP4 default target's validated layouts, as _SPECIALIZED_LAYOUTS states theirs (the MoE split
# the target builds: EP-only), and the (layout, server) cells it is validated on per GPU. A server is
# "aggregated", "context" (a disaggregated context server) or "generation" (a disaggregated
# generation server).
_NVFP4_DEFAULT_LAYOUTS = {
    "dep16": (16, 16, 1, False, True, 1, 16),
    "helix8": (8, 1, 8, True, False, 1, 8),  # Helix CP 8 over TP 1
    "tep8": (8, 8, 1, False, False, 1, 8),
}
_NVFP4_DEFAULT_CELLS = {
    (10, 3): frozenset(
        {
            ("dep16", "aggregated"),
            ("dep16", "generation"),
            ("helix8", "generation"),
            ("tep8", "aggregated"),
            ("tep8", "context"),
            ("tep8", "generation"),
        }
    ),
    (10, 7): frozenset({("dep16", "aggregated"), ("tep8", "aggregated")}),
}
# The draft lengths the KDA verify kernels take (a block of k + 1 rows per request).
_NVFP4_DEFAULT_DRAFT_LENS = (1, 2, 3, 4, 5, 6, 7)
_NVFP4_DEFAULT = "the NVFP4 default target"
# The subject of the shared engine-setting refusals when the NVFP4 default target gives them.
_NVFP4_DEFAULT_SERVERS = "the NVFP4 default target's servers"
_NOT_VALIDATED_ON_DEFAULT = f"not validated on {_NVFP4_DEFAULT}"
_NVFP4_DEFAULT_RECIPE = (
    "it is validated with the CUTEDSL MoE backend, an FP8 latent KV cache (kv_cache_config.dtype "
    "'fp8'), FP8 index keys, an FP32 KDA state (mamba_ssm_cache_dtype 'float32' or 'auto') and DFlash2 "
    "speculative decoding"
)
_LM_HEAD_TP_IN_ADP = (
    "the DFlash2 candidate selector reads full-vocabulary draft logits: no enable_lm_head_tp_in_adp"
)


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
    """Why ``spec_config`` is not one-engine DFlash with a DFlash2 MLA drafter run as the targets run
    it (:func:`_dflash2_settings_mismatch`), or None.

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
    return _dflash2_settings_mismatch(spec_config)


def _dflash2_settings_mismatch(spec_config) -> Optional[str]:
    """Why the DFlash2 settings are not the ones every Kimi K3.1 target runs, or None.

    The drafter decodes its block with its own attention: ``attention_backend`` VANILLA (``AUTO``
    stands for it), which also keeps the engine from building a separate draft KV cache manager.
    With rejection sampling on, a sampling step verifies the selector's block by block
    verification (``ops/spec/dflash2_verify``), so a ``rejection_sample_method`` the arguments set
    names that rule; left unset, it is not read."""
    backend = getattr(spec_config, "attention_backend", "VANILLA")
    if str(backend).upper() not in ("VANILLA", "AUTO"):
        return (
            f"speculative_config.attention_backend {backend!r}: the Kimi K3.1 DFlash2 drafter decodes "
            "its block with its own attention (VANILLA); leave attention_backend unset or set it to "
            "VANILLA"
        )
    method = getattr(spec_config, "rejection_sample_method", None)
    if (
        "rejection_sample_method" in getattr(spec_config, "model_fields_set", ())
        and getattr(spec_config, "use_rejection_sampling", False)
        and method != "block"
    ):
        return (
            f"speculative_config.rejection_sample_method {method!r}: the Kimi K3.1 targets verify a "
            "sampled DFlash2 block by block verification; set rejection_sample_method 'block' or "
            "leave it unset"
        )
    return None


def _draft_heads_mismatch(spec_config, mapping) -> Optional[str]:
    """Why the DFlash2 drafter's attention heads do not split over the Helix ranks, or None: no
    Helix, or a drafter config.json that is not a local file yet (a hub drafter before download)."""
    if not mapping.has_cp_helix():
        return None
    heads = (_draft_checkpoint_config(spec_config) or {}).get("num_attention_heads")
    ranks = mapping.tp_size * mapping.cp_size
    if isinstance(heads, int) and heads % ranks:
        return (
            f"{_layout_facts(mapping)}: the DFlash2 drafter shards its {heads} attention heads over "
            f"the {ranks} Helix ranks (tp x cp), which must divide them"
        )
    return None


def _kda_heads_mismatch(text, mapping) -> Optional[str]:
    """Why the KDA heads do not split over the layout's attention ranks, or None.

    Attention TP and Helix split the KDA heads (``linear_attn_config`` ``num_heads``) over the
    tp x cp ranks; attention DP keeps them whole on every rank."""
    if mapping.enable_attention_dp:
        return None
    kinds = getattr(text, "linear_attn_config", None) or {}
    heads = kinds.get("num_heads")
    layers = getattr(text, "num_hidden_layers", 0) or 0
    if not isinstance(heads, int) or not any(
        layer <= layers for layer in kinds.get("kda_layers") or ()
    ):
        return None
    ranks = mapping.tp_size * mapping.cp_size
    if heads % ranks:
        return (
            f"{_layout_facts(mapping)}: the targets split the {heads} KDA heads (linear_attn_config "
            f"num_heads) over the {ranks} attention ranks (tp x cp), which must divide them"
        )
    return None


def _draft_cache_mismatch(
    llm_args: TorchLlmArgs, mapping, unified_always: bool = False
) -> Optional[str]:
    """Why the DFlash2 drafter's context store rules the deployment out, or None.

    A disaggregated context server's drafter (:func:`server_role` ``context``) only builds the
    prompt's context, which the cache transceiver carries to the generation server. The drafter's
    context lives in the unified draft cache of KV cache manager V2 on a disaggregated server,
    under Helix and, with ``unified_always``, on every server (the sm_107 specialized targets);
    its draft layers then form a layer group of their own, which takes neither SWA scratch reuse
    nor a ``pool_ratio`` (one weight per target layer group)."""
    disaggregated = _disaggregated(llm_args)
    if server_role() == "context" and not disaggregated:
        return (
            "a disaggregated context server (TRTLLM_DISAGG_ROLE=context) without "
            "cache_transceiver_config: its DFlash2 drafter only builds the prompt's context for the "
            "generation server, which the cache transceiver carries; set cache_transceiver_config"
        )
    helix = mapping.has_cp_helix() or getattr(mapping, "helix_repurposed_cp_size", 1) > 1
    if not (disaggregated or helix or unified_always):
        return None
    kv = llm_args.kv_cache_config
    if getattr(kv, "enable_swa_scratch_reuse", False):
        return (
            "kv_cache_config.enable_swa_scratch_reuse: the DFlash2 drafter's context lives in the "
            "unified draft cache here, whose draft layers take no SWA scratch reuse; turn it off"
        )
    if getattr(kv, "pool_ratio", None) is not None:
        return (
            "kv_cache_config.pool_ratio: the DFlash2 drafter's context lives in the unified draft "
            "cache here, and pool_ratio has no weight for its draft layers' group; drop pool_ratio"
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
        return (
            f"the Kimi K3.1 target has no pipeline parallelism "
            f"(pp={mapping.pp_size}, cp={mapping.cp_size})"
        )
    if mapping.cp_size > 1 and (
        not mapping.has_cp_helix() or mapping.tp_size != 1 or mapping.enable_attention_dp
    ):
        return "Kimi K3.1 Helix requires TP1 without attention DP"
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


def static_placement_mismatch(config, num_experts: int) -> Optional[str]:
    """Why the load balancer configuration ``config`` is not a static placement of ``num_experts`` experts the fused
    routed MoE serves (one slot per expert, no online updates), or ``None``."""
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


def _dep_perf_mismatch(llm_args: TorchLlmArgs, text, mapping) -> Optional[str]:
    """Why the sm_103 DEP16 perf target cannot build this deployment, or None.

    The target shards each KDA projection over pairs of attention-DP ranks of one node. A static
    expert placement (``moe_config.load_balancer``) names a permutation of the experts for every
    routed-MoE layer; a placement other than the identity runs every routed call on the fused
    routed MoE, which dispatches at most ``_PLACEMENT_MAX_ROWS`` rows in one call."""
    if mapping.gpus_per_node % 2:
        return (
            f"gpus_per_node {mapping.gpus_per_node}: the DEP16 perf target shards each KDA projection "
            "over pairs of attention-DP ranks of one node, so gpus_per_node must be even"
        )
    placement = llm_args.moe_config.load_balancer
    assignments = getattr(placement, "initial_global_assignments", None)
    if assignments is None:
        return None
    what = "moe_config.load_balancer initial_global_assignments"
    identity = list(range(_NUM_EXPERTS))
    placed = False
    for layer in _moe_layers(text):
        if layer not in assignments:
            return f"{what} has no entry for routed-MoE layer {layer}"
        slots = [int(expert) for expert in assignments[layer]]
        if len(slots) != placement.num_slots:
            return (
                f"{what} of layer {layer} has {len(slots)} entries, not one per slot "
                f"(num_slots {placement.num_slots})"
            )
        if sorted(slots) != identity:
            return f"{what} of layer {layer} is not a permutation of the {_NUM_EXPERTS} experts"
        placed = placed or slots != identity
    max_rows = _PLACEMENT_MAX_ROWS // _moe_split(mapping)[1]
    max_num_tokens = llm_args.max_num_tokens
    if placed and max_num_tokens is not None and max_num_tokens > max_rows:
        return (
            f"max_num_tokens {max_num_tokens} with a non-identity expert placement "
            "(moe_config.load_balancer): placed experts run on the fused routed MoE only, which "
            f"serves up to {max_rows} tokens per rank in one call; lower max_num_tokens to "
            f"{max_rows} or use identity placement"
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


def _bf16_projection_mismatch(llm_args: TorchLlmArgs) -> Optional[str]:
    """Why the engine's GEMM settings rule out the targets' BF16 projections (cuBLAS), or None."""
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
    return None


def _engine_mismatch(llm_args: TorchLlmArgs, dep_perf_layout: bool) -> Optional[str]:
    """Why the engine settings rule the target out, or None. ``dep_perf_layout``: the deployment has the
    layout, GPU and FP8 latent KV cache of the attention-DP perf target."""
    moe = llm_args.moe_config
    backend = str(moe.backend).upper()
    # An unset backend is AUTO until the target's model defaults name TRTLLM; an AUTO the arguments
    # set stays AUTO, which the engine resolves to another backend for Kimi K3.1.
    if backend != "TRTLLM" and not (backend == "AUTO" and "backend" not in moe.model_fields_set):
        return (
            f"the target's routed experts run on the TRTLLM MoE backend, got {moe.backend}; leave "
            "moe_config.backend unset (the target's default) or set it to TRTLLM"
        )
    if moe.load_balancer is not None:
        if not dep_perf_layout:
            return (
                "the target has a static contiguous expert partition: no MoE load balancer (EPLB); a static "
                "expert placement (moe_config.load_balancer) runs on the sm_103 DEP16 perf target only (FP8 KV "
                "cache); drop moe_config.load_balancer or serve DEP16 on sm_103 with kv_cache_config.dtype='fp8'"
            )
        if (mismatch := static_placement_mismatch(moe.load_balancer, _NUM_EXPERTS)) is not None:
            return mismatch
    if (mismatch := _bf16_projection_mismatch(llm_args)) is not None:
        return mismatch
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


def _names_indexer_k_dtype(llm_args: TorchLlmArgs) -> bool:
    """Whether the arguments' sparse attention config names the index-key dtype."""
    requested = llm_args.sparse_attention_config
    return requested is not None and "indexer_k_dtype" in requested.model_fields_set


def _indexer_k_dtype(model_config: ModelConfig, llm_args: TorchLlmArgs) -> str:
    """The index-key dtype the model is built with: the one the arguments set, else the loaded one.

    Routing also runs on the config read before the model defaults are applied, which is loaded
    without the arguments' sparse attention config: there the index-key dtype is the DSA default
    ('fp8'), as the checkpoint's dsa_config names none.
    """
    if _names_indexer_k_dtype(llm_args):
        return llm_args.sparse_attention_config.indexer_k_dtype
    return getattr(model_config.sparse_attention_config, "indexer_k_dtype", "fp8")


def _kv_cache_quant_algo_name(model_config: ModelConfig, llm_args: TorchLlmArgs) -> Optional[str]:
    """The engine's KV-cache algorithm by name (None: BF16; a dtype the engine does not map stays
    the string given)."""
    algo = engine_kv_cache_quant_algo(model_config, llm_args)
    return algo.name if isinstance(algo, QuantAlgo) else algo


def server_role() -> str:
    """``context`` on a disaggregated context server, else ``generation``: the role the runtime
    published for this server's ranks (``ops/attention/dsa_prefill_share.disagg_role``). An
    aggregated server cannot be told from a generation server, and takes that role."""
    from .ops.attention.dsa_prefill_share import disagg_role

    return "context" if disagg_role() == "context" else "generation"


def _dsa_on_every_mla_layer(text) -> bool:
    """Whether the text config's DSA covers every MLA layer (both lists number layers from 1)."""
    mla_layers = set((getattr(text, "linear_attn_config", None) or {}).get("full_attn_layers", ()))
    dsa_layers = set((getattr(text, "dsa_config", None) or {}).get("dsa_layers") or ())
    return bool(mla_layers) and mla_layers <= dsa_layers


def _dsa_indexer_mismatch(text) -> Optional[str]:
    """Why the text config's ``dsa_config`` states another indexer than the one the targets' model
    defaults configure (``config.DSA_INDEXER``), or None."""
    from .config import DSA_INDEXER

    stated = {key: text.dsa_config.get(key) for key in DSA_INDEXER}
    if stated != dict(DSA_INDEXER):
        return f"dsa_config indexer {stated}: the Kimi K3.1 targets run {dict(DSA_INDEXER)}"
    # The reference defaults to an RMSNorm key norm; the targets implement the LayerNorm one.
    k_norm_type = text.dsa_config.get("index_k_norm_type", "rmsnorm")
    if k_norm_type != "layernorm":
        return f"dsa_config index_k_norm_type {k_norm_type!r}: the Kimi K3.1 indexer key norm is a LayerNorm"
    return None


def _dsa_mismatch(model_config: ModelConfig, llm_args: TorchLlmArgs, text) -> Optional[str]:
    """Why a deployment of the ModelOpt export does not run DSA on every MLA layer with the TRTLLM
    attention backend, as every sm_107 target does, or None."""
    # None before the model defaults are applied, when the arguments name none: those defaults
    # supply DSA (config.dsa_model_defaults).
    sparse = model_config.sparse_attention_config or llm_args.sparse_attention_config
    if sparse is not None and getattr(sparse, "algorithm", None) != "dsa":
        return f"sparse attention {getattr(sparse, 'algorithm', None)!r}, not DSA"
    if not _dsa_on_every_mla_layer(text):
        return "the text config's dsa_config does not cover every MLA layer"
    if str(model_config.attn_backend).upper() != "TRTLLM":
        return f"attention backend {model_config.attn_backend}, not TRTLLM"
    return None


def _nvfp4_model_mismatch(model_config: ModelConfig, llm_args: TorchLlmArgs, text) -> Optional[str]:
    """Settings shared by the default and specialized NVFP4 implementations."""
    if (mismatch := _layer_layout_mismatch(text)) is not None:
        return mismatch
    index_rope_dim = text.dsa_config.get("index_rope_dim")
    if index_rope_dim != 0:
        return (
            f"dsa_config index_rope_dim {index_rope_dim!r}: the Kimi K3.1 DSA indexer is "
            "position-free (index_rope_dim 0)"
        )
    sparse = model_config.sparse_attention_config or llm_args.sparse_attention_config
    if getattr(sparse, "use_cute_dsl_topk", False):
        return "the Kimi K3.1 DSA indexer runs the GVR / CUDA radix Top-K: no use_cute_dsl_topk"
    return _bf16_projection_mismatch(llm_args)


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
    return _dflash2_mla_mismatch(spec_config)


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
    llm_args: TorchLlmArgs, targets: str = "the specialized targets"
) -> Optional[str]:
    """Why the engine settings rule the sm_107 targets out, or None. ``targets`` names the targets
    the reason is about."""
    if str(llm_args.prefill_cuda_graph_backend.value) != "disabled":
        return (
            f"prefill_cuda_graph_backend {str(llm_args.prefill_cuda_graph_backend.value)!r}: "
            f"{targets} run prefill eagerly ('piecewise' torch.compiles the model, and no graph "
            "masks the MoE padding rows)"
        )
    if llm_args.torch_compile_config is not None:
        return f"{targets}' kernels are not torch ops: no torch_compile_config"
    if llm_args.moe_config.load_balancer is not None:
        return f"{targets} have a static expert partition: no MoE load balancer"
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


def _specialized_engine_mismatch(llm_args: TorchLlmArgs, spec_config) -> Optional[str]:
    """Why the engine settings rule the specialized targets out, or None."""
    return _draft_schedule_mismatch(spec_config) or _engine_settings_mismatch(llm_args)


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


def _specialized_layout(mapping) -> Optional[str]:
    """``dep16``, ``helix8`` or ``tep8``, or None."""
    if mapping.pp_size != 1:
        return None
    if mapping.has_cp_helix() and not getattr(mapping, "moe_tp_ep_user_specified", False):
        return None
    facts = (
        mapping.world_size,
        mapping.tp_size,
        mapping.cp_size,
        bool(mapping.has_cp_helix()),
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    return next((name for name, want in _SPECIALIZED_LAYOUTS.items() if facts == want), None)


def _layout_facts(mapping) -> str:
    """The parallel layout as a refusal names it."""
    return (
        f"layout world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} helix={bool(mapping.has_cp_helix())} "
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
        return None, (
            f"{_layout_facts(mapping)}: the specialized targets run dep16 (attention DP 16, "
            "MoE EP 16), helix8 (Helix CP 8, MoE TP 2 x EP 4, set by the arguments) and tep8 "
            "(TP 8, MoE EP 8)"
        )
    role = server_role()
    if (layout, role) not in _SPECIALIZED:
        return None, f"{layout} serves generation servers only, not a {role} server"
    backend, draft_lens = _SPECIALIZED[(layout, role)]
    spec_config = model_config.spec_config
    if (draft_len := spec_config.max_draft_len) not in draft_lens:
        why = f"; {_LONGER_DRAFT}" if draft_len > max(_DRAFT_LENS) else ""
        return None, (
            f"max_draft_len {draft_len}: a {layout} {role} server runs one of {draft_lens}{why}"
        )
    if str(llm_args.moe_config.backend).upper() != backend:
        return None, (
            f"moe_config.backend {llm_args.moe_config.backend}: a {layout} {role} server runs "
            f"{backend}"
        )
    if layout in ("tep8", "helix8") and llm_args.allreduce_strategy not in (None, "AUTO", "MNNVL"):
        return (
            None,
            f"{layout} requires MNNVL MoE collectives, got allreduce_strategy={llm_args.allreduce_strategy}",
        )
    if (mismatch := _specialized_engine_mismatch(llm_args, spec_config)) is not None:
        return None, mismatch
    if mapping.enable_attention_dp and llm_args.enable_lm_head_tp_in_adp:
        return None, _LM_HEAD_TP_IN_ADP
    if layout == "helix8" and (mismatch := _helix8_mismatch(llm_args)) is not None:
        return None, mismatch
    if (mismatch := _draft_cache_mismatch(llm_args, mapping, unified_always=True)) is not None:
        return None, mismatch
    # Read as the engine's use_py_mamba_cache_manager reads it.
    if os.environ.get("TRTLLM_USE_PY_MAMBA", "0") == "1":
        return None, (
            "TRTLLM_USE_PY_MAMBA=1: the specialized targets keep the KDA state, the index keys and "
            "the drafter history in their own KV cache manager V2; unset TRTLLM_USE_PY_MAMBA"
        )
    return layout, None


def _disaggregated(llm_args: TorchLlmArgs) -> bool:
    """Whether the server is one side of a disaggregated deployment (it has a cache transceiver)."""
    transceiver = llm_args.cache_transceiver_config
    return transceiver is not None and transceiver.backend is not None


def _server_kind(llm_args: TorchLlmArgs) -> str:
    """``context`` or ``generation`` on a disaggregated server (its published role), else ``aggregated``."""
    if server_role() == "context":
        return "context"
    return "generation" if _disaggregated(llm_args) else "aggregated"


def _nvfp4_default_layout(mapping) -> Optional[str]:
    """``dep16``, ``helix8`` or ``tep8``, the NVFP4 default target's layout of ``mapping``, or None."""
    if mapping.pp_size != 1:
        return None
    facts = (
        mapping.world_size,
        mapping.tp_size,
        mapping.cp_size,
        bool(mapping.has_cp_helix()),
        bool(mapping.enable_attention_dp),
        *_moe_split(mapping),
    )
    return next((name for name, want in _NVFP4_DEFAULT_LAYOUTS.items() if facts == want), None)


def _nvfp4_default_mismatch(
    model_config: ModelConfig,
    llm_args: TorchLlmArgs,
    compute_capability: Tuple[int, int],
    text,
) -> Optional[str]:
    """Why the NVFP4 default target does not take a deployment of the export on ``compute_capability``,
    or None: the first check that fails, in this order. A setting outside its validated cells
    (``_NVFP4_DEFAULT_CELLS``) is refused as not validated."""
    from .config import modelopt_nvfp4_mismatch

    kv_dtype = llm_args.kv_cache_config.dtype
    kv_algo = _kv_cache_quant_algo_name(model_config, llm_args)
    if kv_dtype != "fp8" or kv_algo != QuantAlgo.FP8.name:
        return (
            f"kv_cache_config.dtype {kv_dtype!r} (KV cache {kv_algo}): {_NOT_VALIDATED_ON_DEFAULT}; "
            f"{_NVFP4_DEFAULT_RECIPE}"
        )
    if (mismatch := modelopt_nvfp4_mismatch(model_config)) is not None:
        return mismatch
    if (mismatch := _dsa_mismatch(model_config, llm_args, text)) is not None:
        return mismatch
    if (index_k := _indexer_k_dtype(model_config, llm_args)) != "fp8":
        return f"indexer_k_dtype {index_k!r}: {_NOT_VALIDATED_ON_DEFAULT}; {_NVFP4_DEFAULT_RECIPE}"
    state = llm_args.kv_cache_config.mamba_ssm_cache_dtype
    if state not in _KDA_STATE_DTYPES:
        return (
            f"mamba_ssm_cache_dtype {state!r}: {_NOT_VALIDATED_ON_DEFAULT}; {_NVFP4_DEFAULT_RECIPE}"
        )
    spec_config = model_config.spec_config
    if spec_config is None:
        return f"no speculative decoding: {_NOT_VALIDATED_ON_DEFAULT}; {_NVFP4_DEFAULT_RECIPE}"
    if (mismatch := _dflash2_mla_mismatch(spec_config)) is not None:
        return mismatch
    if (draft_len := spec_config.max_draft_len) not in _NVFP4_DEFAULT_DRAFT_LENS:
        why = f"; {_LONGER_DRAFT}" if draft_len > max(_NVFP4_DEFAULT_DRAFT_LENS) else ""
        return f"max_draft_len {draft_len}: {_NVFP4_DEFAULT} runs one of {_NVFP4_DEFAULT_DRAFT_LENS}{why}"
    if (mismatch := _draft_schedule_mismatch(spec_config, _NVFP4_DEFAULT_SERVERS)) is not None:
        return mismatch
    mapping = model_config.mapping
    if mapping.enable_attention_dp and llm_args.enable_lm_head_tp_in_adp:
        return _LM_HEAD_TP_IN_ADP
    layout = _nvfp4_default_layout(mapping)
    if layout is None:
        return (
            f"{_layout_facts(mapping)}: {_NOT_VALIDATED_ON_DEFAULT}, which runs tep8 (TP 8, MoE EP 8), "
            "dep16 (attention DP 16, MoE EP 16) and helix8 (Helix CP 8 over TP 1, MoE EP 8)"
        )
    kind = _server_kind(llm_args)
    cells = _NVFP4_DEFAULT_CELLS[compute_capability]
    if (layout, kind) not in cells:
        validated = ", ".join(f"{name} {server}" for name, server in sorted(cells))
        gpu = f"sm_{compute_capability[0]}{compute_capability[1]}"
        return (
            f"a {layout} {kind} server on {gpu}: {_NOT_VALIDATED_ON_DEFAULT} there (validated: "
            f"{validated})"
        )
    if layout == "helix8" and (mismatch := _helix8_mismatch(llm_args)) is not None:
        return mismatch
    if (mismatch := _kda_heads_mismatch(text, mapping)) is not None:
        return mismatch
    if (mismatch := _draft_heads_mismatch(spec_config, mapping)) is not None:
        return mismatch
    if (mismatch := _draft_cache_mismatch(llm_args, mapping)) is not None:
        return mismatch
    if kind == "context" and not getattr(llm_args, "disable_mm_encoder", False):
        return (
            "a context server with the vision tower: image requests through disaggregated serving "
            f"are {_NOT_VALIDATED_ON_DEFAULT}; set disable_mm_encoder: true on its context servers"
        )
    moe = llm_args.moe_config
    backend = str(moe.backend).upper()
    if backend != "CUTEDSL" and not (backend == "AUTO" and "backend" not in moe.model_fields_set):
        return (
            f"moe_config.backend {moe.backend}: {_NOT_VALIDATED_ON_DEFAULT}, which runs CUTEDSL "
            "(its model default: leave moe_config.backend unset or set it to CUTEDSL)"
        )
    if (mismatch := _moe_mismatch(text, mapping)) is not None:
        return mismatch
    return _engine_settings_mismatch(llm_args, _NVFP4_DEFAULT_SERVERS)


def _modelopt_target(
    model_config: ModelConfig,
    llm_args: TorchLlmArgs,
    compute_capability: Tuple[int, int],
    text,
) -> str:
    """The target of a deployment of the ModelOpt NVFP4 export by layout: on sm_107 a specialized
    target first (``dep16``, ``helix8`` or ``tep8``), else ``default``, or ``NoTarget``;
    :func:`route` returns its class."""
    if compute_capability in _NOT_VALIDATED:
        raise NoTarget(_NOT_VALIDATED[compute_capability])
    if compute_capability not in _COMPUTE_CAPABILITIES:
        raise NoTarget(
            "the Kimi K3.1 NVFP4 export runs on sm_103 (GB300, B300) and sm_107 (Rubin), got "
            f"{compute_capability}"
        )
    if (mismatch := _nvfp4_model_mismatch(model_config, llm_args, text)) is not None:
        raise NoTarget(mismatch)
    if compute_capability != _RUBIN:
        if (
            mismatch := _nvfp4_default_mismatch(model_config, llm_args, compute_capability, text)
        ) is not None:
            raise NoTarget(f"{_NVFP4_DEFAULT}: {mismatch}")
        return "default"
    layout, refusal = _specialized_decision(model_config, llm_args, text)
    if refusal is None:
        return layout
    if (
        mismatch := _nvfp4_default_mismatch(model_config, llm_args, compute_capability, text)
    ) is not None:
        if refusal.startswith(mismatch):
            raise NoTarget(refusal)
        raise NoTarget(f"{refusal}; {_NVFP4_DEFAULT}: {mismatch}")
    return "default"


@routes("KimiK31ForConditionalGeneration", has_builtin=False)
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits."""
    pretrained_config = model_config.pretrained_config
    text = getattr(pretrained_config, "text_config", None) or pretrained_config
    shape = tuple(getattr(text, name, None) for name in _SHAPE_FIELDS)
    if shape != _SHAPE:
        raise NoTarget(f"text config shape {shape} is not Kimi K3.1's {_SHAPE}")
    if getattr(text, "dsa_config", None) is None:
        raise NoTarget("the Kimi K3.1 target needs the text config's dsa_config")
    if (mismatch := _dsa_indexer_mismatch(text)) is not None:
        raise NoTarget(mismatch)
    from .config import mxfp4_experts, routes_to_nvfp4_targets

    if llm_args.enable_return_routed_experts:
        raise NoTarget("Kimi K3.1 does not support enable_return_routed_experts")
    overrides = llm_args.env_overrides or {}
    for name, required in (("TRTLLM_ENABLE_PDL", "1"), ("TRTLLM_DISABLE_FUSED_DSA_METADATA", "0")):
        value = str(overrides.get(name, os.environ.get(name, required)))
        if value != required:
            raise NoTarget(f"Kimi K3.1 requires {name}={required}, got {value!r}")

    spec = model_config.spec_config or llm_args.speculative_config
    if (
        llm_args.watermark_config is not None
        and spec is not None
        and spec.spec_dec_mode.is_dflash()
    ):
        raise NoTarget("Kimi K3.1 DFlash2 verification does not support watermark_config")

    mxfp4 = mxfp4_experts(pretrained_config)
    nvfp4 = None
    if routes_to_nvfp4_targets(
        pretrained_config, getattr(model_config.quant_config, "quant_algo", None)
    ):
        nvfp4 = _modelopt_target(model_config, llm_args, compute_capability, text)
    # One line per import, each returned here, on purpose: the claims suite reads the routed
    # targets off this function's returns of `from .<target>.modeling import <Class>` names.
    # fmt: off
    # isort: off
    if nvfp4 == "dep16":
        from .kimi_k31_pro_nvfp4__sm_107__dep16.modeling import ModelingV2KimiK31ProNvfp4Sm107Dep16
        return ModelingV2KimiK31ProNvfp4Sm107Dep16
    if nvfp4 == "helix8":
        from .kimi_k31_pro_nvfp4__sm_107__helix8.modeling import ModelingV2KimiK31ProNvfp4Sm107Helix8
        return ModelingV2KimiK31ProNvfp4Sm107Helix8
    if nvfp4 == "tep8":
        from .kimi_k31_pro_nvfp4__sm_107__tep8.modeling import ModelingV2KimiK31ProNvfp4Sm107Tep8
        return ModelingV2KimiK31ProNvfp4Sm107Tep8
    if nvfp4 == "default":
        from .kimi_k31_pro_nvfp4__sm_10x__default.modeling import ModelingV2KimiK31ProNvfp4Sm10xDefault
        return ModelingV2KimiK31ProNvfp4Sm10xDefault
    # isort: on
    # fmt: on
    if (mismatch := _layer_layout_mismatch(text)) is not None:
        raise NoTarget(mismatch)
    if not mxfp4:
        raise NoTarget(
            "the Kimi K3.1 target needs the MXFP4 checkpoint (mxfp4-pack-quantized) or the ModelOpt "
            "NVFP4 export"
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
    # The loaded DSA config, else the arguments'. None before the model defaults are applied, when
    # the arguments name none: those defaults supply DSA (config.dsa_model_defaults).
    sparse = model_config.sparse_attention_config or llm_args.sparse_attention_config
    if sparse is not None and getattr(sparse, "algorithm", None) != "dsa":
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
        if (draft_len := spec_config.max_draft_len) > _MAX_DRAFT_LEN:
            raise NoTarget(
                f"max_draft_len {draft_len}: the Kimi K3.1 targets verify at most {_MAX_DRAFT_LEN} "
                f"drafts; {_LONGER_DRAFT}"
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
    if (mismatch := _kda_heads_mismatch(text, model_config.mapping)) is not None:
        raise NoTarget(mismatch)
    if spec_config is not None:
        if (mismatch := _draft_heads_mismatch(spec_config, model_config.mapping)) is not None:
            raise NoTarget(mismatch)
        if (mismatch := _draft_cache_mismatch(llm_args, model_config.mapping)) is not None:
            raise NoTarget(mismatch)
    if model_config.mapping.has_cp_helix():
        kv = llm_args.kv_cache_config
        if "enable_block_reuse" in kv.model_fields_set and kv.enable_block_reuse:
            raise NoTarget("Kimi K3.1 Helix requires block reuse disabled")
    if (mismatch := _moe_mismatch(text, model_config.mapping)) is not None:
        raise NoTarget(mismatch)
    dep_perf_layout = (
        compute_capability == _PERF_COMPUTE_CAPABILITY
        and _layout(model_config.mapping) in _SM103_DEP_LAYOUTS
        and not bf16_latents
    )
    if (mismatch := _engine_mismatch(llm_args, dep_perf_layout)) is not None:
        raise NoTarget(mismatch)
    if (
        dep_perf_layout
        and (mismatch := _dep_perf_mismatch(llm_args, text, model_config.mapping)) is not None
    ):
        raise NoTarget(mismatch)
    # One line per import on purpose: the claims suite reads the routed targets off
    # `from .<target>.modeling import <Class>` lines.
    # fmt: off
    # isort: off
    if compute_capability == _PERF_COMPUTE_CAPABILITY and not bf16_latents:
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


@input_processor_for("KimiK31ForConditionalGeneration")
def input_processor_class(config) -> type:
    """K3's image input processor (K3.1's vision input is K3's), with the prompt placeholders of model_type
    ``kimi_k31``. The driver process builds it without building the model."""
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    from .input_processor import KIMI_K31_PLACEHOLDER_METADATA, K31InputProcessor

    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        "kimi_k31", KIMI_K31_PLACEHOLDER_METADATA, registrant_module=__name__
    )
    K31InputProcessor._registered_model_type = "kimi_k31"
    return K31InputProcessor
