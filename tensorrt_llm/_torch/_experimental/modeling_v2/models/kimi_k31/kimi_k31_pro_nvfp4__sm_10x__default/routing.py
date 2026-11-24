# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission checks for the SM100/SM103/SM107 NVFP4 target."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional, Tuple

from tensorrt_llm.quantization.mode import QuantAlgo

from ...._router_index import NoTarget, engine_kv_cache_quant_algo
from .ops.attention.cache_policy import get_kimi_k31_fp8_kv_layer_ids

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
# Attention TP sizes (no attention DP): divisors of the 96 MLA / KDA / drafter heads that also
# divide the drafter MLP and keep the vision tower's 4096-wide MLP even.
_ATTENTION_TP_SIZES = (1, 2, 4, 8, 16, 32)
# Attention DP sizes (EP-only MoE; 896 % size == 0 and a power of two).
_ATTENTION_DP_SIZES = (2, 4, 8, 16, 32)
_NUM_HEADS = 96
_NUM_EXPERTS = 896
# Page sizes of the paged indexer logits (DeepGEMM block_kv).
_TOKENS_PER_BLOCK = (32, 64, 128)
# Attention projections the target builds on an MLA / a KDA layer, all BF16.
_MLA_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "g_proj", "o_proj")
_KDA_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
# Module-name spellings of a text layer: per-layer quant entries use the first three; exclusions
# may also be written under the wrapper's ``language_model.layers.`` prefix.
_LAYER_PREFIXES = ("language_model.model.", "model.", "")
_EXCLUDED_LAYER_PREFIXES = _LAYER_PREFIXES + ("language_model.",)


def _nvfp4_experts(model_config: ModelConfig) -> bool:
    """Whether the checkpoint declares native NVFP4 routed experts."""
    declarations = model_config.quant_config_dict or {}
    experts = [
        recipe.quant_algo
        for name, recipe in declarations.items()
        if name.endswith((".block_sparse_moe.experts", ".mlp.experts"))
    ]
    return bool(experts) and all(algo == QuantAlgo.NVFP4 for algo in experts)


def _quantized_attention(model_config: ModelConfig, text) -> Optional[str]:
    """Why an attention projection resolves to a quantized recipe, or None.

    Resolved as the target resolves each projection: an exclusion from quantization keeps it
    BF16, else a per-layer entry names its recipe, else the global one (none under
    MIXED_PRECISION). MLA projections and absorption weights remain unquantized. The NVIDIA
    checkpoint additionally declares MXFP8 KDA projections, admitted by its target.
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
            if layer_idx + 1 in kda_layers and algo == QuantAlgo.MXFP8:
                continue
            if algo is not None:
                return (
                    f"attention projection model.{module} is quantized as {algo}: the Kimi K3.1 "
                    "target loads BF16 attention projections, as the checkpoint ships them (no "
                    "FP8 block-scaled MLA absorption)"
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
    """Why ``spec_config`` is not one-engine DFlash with a DFlash2 MLA drafter, or None."""
    if not 1 <= spec_config.max_draft_len <= 7:
        return "Kimi K3.1 KDA verify kernels support max_draft_len 1 through 7"
    if getattr(spec_config, "context_kv_dtype", None) not in (None, "auto", "fp8"):
        return "DFlash history requires context_kv_dtype auto (BF16) or fp8"
    mode = spec_config.spec_dec_mode
    if not (mode.is_dflash() and mode.use_one_engine()):
        return (
            "the Kimi K3.1 target runs DFlash2 speculative decoding (one-engine DFlash with the "
            f"DFlash2 MLA drafter) or none, got {mode.name}"
        )
    if getattr(spec_config, "spec_dec_variant", None) != 2:
        return "the Kimi K3.1 target's DFlash drafter is a DFlash2 checkpoint"
    draft = _draft_checkpoint_config(spec_config)
    if draft is not None:
        attention_mode = (draft.get("dflash_config") or {}).get("attention_mode")
        if attention_mode != "mla":
            return (
                "the Kimi K3.1 target's DFlash2 drafter is the MLA one "
                f"(dflash_config.attention_mode 'mla'), got {attention_mode!r}"
            )
    return None


def _moe_split(mapping) -> Tuple[int, int]:
    """(moe_tp, moe_ep) the target builds: the user's split, else EP-only."""
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
            f"(divisors of the {_NUM_HEADS} heads with an even drafter MLP and vision tower)"
        )
    if moe_tp * moe_ep != tp:
        return f"moe_tp {moe_tp} x moe_ep {moe_ep} must equal tp_size {tp}"
    if _NUM_HEADS % moe_tp:
        return f"moe_tp {moe_tp} must divide {_NUM_HEADS} (attention head shard)"
    if _NUM_EXPERTS % moe_ep:
        return f"moe_ep {moe_ep} must divide the {_NUM_EXPERTS} experts"
    return None


def _engine_mismatch(llm_args: TorchLlmArgs) -> Optional[str]:
    """Why the engine settings rule the target out, or None."""
    if llm_args.prefill_cuda_graph_backend.value != "disabled":
        return "the target runs prefill eagerly; prefill_cuda_graph_backend must be disabled"
    if llm_args.torch_compile_config is not None:
        return "the target's direct kernel calls require torch_compile_config unset"
    moe = llm_args.moe_config
    supported = ("AUTO", "TRTLLM", "CUTEDSL", "MEGAMOE_CUTEDSL")
    if str(moe.backend).upper() not in supported:
        return f"the target's routed experts require MoE backend {supported}, got {moe.backend}"
    if moe.load_balancer is not None:
        return "the target has a static contiguous expert partition: no MoE load balancer (EPLB)"
    if str(llm_args.attn_backend).upper() != "TRTLLM":
        return "the target requires the TRTLLM attention backend"
    kv = llm_args.kv_cache_config
    if kv.mamba_ssm_cache_dtype not in ("auto", "float32", "bfloat16"):
        return "the target stores KDA state in FP32 or BF16"
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


def validate_target(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> None:
    """Validate this deployment, or raise ``NoTarget`` with the first mismatch."""
    pretrained_config = model_config.pretrained_config
    text = getattr(pretrained_config, "text_config", None) or pretrained_config
    shape = tuple(getattr(text, name, None) for name in _SHAPE_FIELDS)
    if shape != _SHAPE:
        raise NoTarget(f"text config shape {shape} is not Kimi K3.1's {_SHAPE}")
    if getattr(text, "dsa_config", None) is None:
        raise NoTarget("the Kimi K3.1 target needs the text config's dsa_config")
    if not _nvfp4_experts(model_config):
        raise NoTarget("the SM100/SM103/SM107 target requires native NVFP4 routed experts")
    if (mismatch := _quantized_attention(model_config, text)) is not None:
        raise NoTarget(mismatch)
    if compute_capability not in ((10, 0), (10, 3), (10, 7)):
        raise NoTarget(
            f"the NVFP4 Kimi K3.1 target requires SM100, SM103, or SM107, got {compute_capability}"
        )
    # The loader fills the DSA config from the checkpoint's dsa_config when the arguments name none.
    sparse = model_config.sparse_attention_config or llm_args.sparse_attention_config
    if getattr(sparse, "algorithm", None) != "dsa":
        raise NoTarget(
            "the Kimi K3.1 target runs DSA sparse attention: set sparse_attention_config "
            f"algorithm 'dsa' (got {getattr(sparse, 'algorithm', None)!r})"
        )
    if getattr(sparse, "indexer_k_dtype", "fp8") not in ("fp8", "fp4"):
        raise NoTarget(
            f"indexer_k_dtype {sparse.indexer_k_dtype!r} is not in the default "
            "target: it runs FP8 or FP4 index keys"
        )
    kv_dtype = llm_args.kv_cache_config.dtype
    kv_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    unsupported_layouts = ("fp8_ds_mla",)
    supported_algorithms = (QuantAlgo.FP8, QuantAlgo.NVFP4, None)
    if kv_dtype in unsupported_layouts or kv_algo not in supported_algorithms:
        supported = "an NVFP4, FP8, or BF16 latent KV cache"
        raise NoTarget(
            f"kv_cache_config.dtype={kv_dtype!r} ({kv_algo}) is not in the default target: it runs "
            f"{supported}"
        )
    if kv_algo == QuantAlgo.NVFP4:
        mla_layers = set(
            (getattr(text, "linear_attn_config", None) or {}).get("full_attn_layers", ())
        )
        dsa_layers = set(text.dsa_config.get("dsa_layers", ()))
        if not mla_layers or not mla_layers.issubset(dsa_layers):
            raise NoTarget("NVFP4 latent KV cache requires DSA on every MLA layer")
    try:
        fp8_layers = get_kimi_k31_fp8_kv_layer_ids(
            pretrained_config, nvfp4_enabled=kv_algo == QuantAlgo.NVFP4
        )
    except ValueError as error:
        raise NoTarget(str(error)) from error
    if fp8_layers and kv_algo != QuantAlgo.NVFP4:
        raise NoTarget("TLLM_KIMI_K3_FP8_KV_LAYERS requires NVFP4 cache storage")
    spec_config = model_config.spec_config
    if spec_config is not None:
        if (mismatch := _dflash2_mla_mismatch(spec_config)) is not None:
            raise NoTarget(mismatch)
        if getattr(spec_config, "context_kv_only", False):
            from ..routing import disagg_role

            if disagg_role(llm_args) != "context":
                raise NoTarget("context_kv_only is supported only on a context server")
        if getattr(spec_config, "draft_len_schedule", None) is not None:
            raise NoTarget(
                "DFlash2 verifies a fixed block (the KDA replay width): no draft_len_schedule"
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
    if (mismatch := _engine_mismatch(llm_args)) is not None:
        raise NoTarget(mismatch)
