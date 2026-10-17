# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MuseSparkOpenOmniForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployment the target below was written for. When it does
not fit, ``route`` raises ``NoTarget`` stating those facts. In ``auto`` the
engine then uses the built-in MuseSparkOpen implementation; in ``require`` it
raises that error.

The checkpoint publishes ``MuseSparkForConditionalGeneration`` with
``model_type: muse_spark``; the config loader translates that layout into the
flat ``MuseSparkOpenConfig`` and writes ``MuseSparkOpenOmniForCausalLM`` as its
architecture, so ``_router_index`` keys on that name and every criterion below is
read off the flat config.

The quantization is not a criterion. The open-format translation drops the
inline ``quantization_config`` (``hf_quant_config.json`` is the one copy the
engine reads, into ``ModelConfig.quant_config``), and ``route`` sees only the
pretrained config. The target runs the built-in loader and forward, which serve
whatever that file declares, so a checkpoint of the same shape in another
precision is served the built-in way; like a fine-tune of the same shape, it is
outside what the accuracy gate pins.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig

from ..._router_index import NoTarget

# The one GPU architecture the target is written for.
_SM = (10, 3)

# The checkpoint's config shape, read off the flat config; see the gpt_oss routing
# module for what shape-sniffing does and does not pin. The attention fields fix
# the per-rank attention operands at eight ranks, the MoE fields the expert
# operands, and the last two the layer pattern: every fourth layer, counted from
# the last, is a full-attention NoPE layer, and every other layer is a 2048-token
# sliding-window RoPE layer.
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_experts",
    "moe_topk",
    "expert_hidden_size",
    "intermediate_size",
    "vocab_size",
    "sliding_window",
    "every_n_layers_nope",
)
_SHAPE = (62, 8192, 128, 16, 64, 256, 8, 4096, 4096, 202048, 2048, 4)

# Ranks of tensor-parallel attention and of expert-parallel MoE.
_RANKS = 8

# The most draft tokens per request: the decode target takes the target token plus
# at most seven draft tokens, the at-most-eight rows per request that the built-in
# small-M decode paths it starts from (low-M GEMM, sharded pool/unpool projections)
# compute exactly.
_MAX_DRAFT_LEN = 7


def _tep8(m: Any) -> bool:
    """Attention heads tensor-parallel and experts expert-parallel over eight ranks,
    with one expert slice per rank and no attention data parallelism.

    The split decides which weights each rank loads and into which shapes, so it
    selects a target rather than a runtime branch.
    """
    return (
        m.world_size == _RANKS
        and m.tp_size == _RANKS
        and m.pp_size == 1
        and m.cp_size == 1
        and m.moe_ep_size == _RANKS
        and m.moe_tp_size == 1
        and not m.enable_attention_dp
    )


def _dflash_or_none(spec_config: Optional[Any]) -> bool:
    """No speculation, or DFlash as a linear chain of at most ``_MAX_DRAFT_LEN`` drafts.

    DFlash's drafter is a separate checkpoint driven by the built-in one-engine
    DFlash worker, which this target inherits unchanged from the built-in causal
    LM. The other modes the built-in model admits (external drafters, the suffix
    automaton) do not route here.
    """
    if spec_config is None:
        return True
    return (
        isinstance(spec_config, DFlashDecodingConfig)
        and spec_config.spec_dec_mode.is_dflash()
        and 1 <= spec_config.max_draft_len <= _MAX_DRAFT_LEN
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
    )


def _speculation(spec_config: Optional[Any]) -> str:
    if spec_config is None:
        return "off"
    mode = getattr(spec_config, "spec_dec_mode", None)
    return (
        f"{getattr(mode, 'name', type(spec_config).__name__)} "
        f"max_draft_len={getattr(spec_config, 'max_draft_len', None)} "
        f"max_total_draft_tokens={getattr(spec_config, 'max_total_draft_tokens', None)}"
    )


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    # A config without one of these fields is another assembly: it reads as None
    # and the shape does not match.
    shape = tuple(getattr(pretrained_config, name, None) for name in _SHAPE_FIELDS)

    # Muse-Spark-Open-816B-A42B on sm103, tensor-parallel attention and
    # expert-parallel MoE over eight ranks, DFlash or no speculation.
    served = (
        compute_capability == _SM
        and shape == _SHAPE
        and _tep8(mapping)
        and _dflash_or_none(spec_config)
    )
    if not served:
        raise NoTarget(
            f"compute capability {compute_capability}, config shape {shape}, parallel "
            f"ws={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
            f"cp={mapping.cp_size} moe_ep={mapping.moe_ep_size} "
            f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp}, "
            f"speculative decoding {_speculation(spec_config)}; the one musespark_open "
            "target is Muse-Spark-Open-816B-A42B on sm103, tep8 (8 ranks of "
            "tensor-parallel attention and expert-parallel MoE, no attention DP), with "
            "DFlash at max_draft_len 1..7 or no speculative decoding"
        )

    # One line, and at this depth, on purpose: the claims suite reads the routed
    # targets off `from .<target>.modeling import <Class>` lines, so the import
    # cannot be wrapped to fit the line limit.
    # fmt: off
    # isort: off
    from .muse_spark_open_816b_a42b_nvfp4__sm_103__tep8.modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep8
    # isort: on
    # fmt: on

    return ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep8
