# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MuseSparkOpenOmniForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision on three facts: the GPU, the model
and the parallel layout (TEP). When they do not fit, ``route`` raises ``NoTarget``
stating every fact it read. In ``auto`` the engine then uses the built-in
MuseSparkOpen implementation; in ``require`` it raises that error.

The checkpoint publishes ``MuseSparkForConditionalGeneration`` with
``model_type: muse_spark``; the config loader translates that layout into the
flat ``MuseSparkOpenConfig`` and writes ``MuseSparkOpenOmniForCausalLM`` as its
architecture, so ``_router_index`` keys on that name and every criterion below is
read off the flat config.

The target claims every tensor-parallel size of its layout. Which fast paths a
deployment runs is the target's own decision at construction
(``dispatch.compose``): the composition of its tensor-parallel size when the
GPU topology and the speculative decoding mode are the ones the compositions
are written for, and the built-in forward through the target otherwise. So
``route`` reads neither the speculative-decoding config nor the GPUs per node
(launcher state: the visible devices, or the tensor-parallel size under Ray),
and a deployment the target serves without fast paths still lands here.

The quantization is not a criterion either: the open-format translation drops
the inline ``quantization_config`` (``hf_quant_config.json`` is the one copy the
engine reads, into ``ModelConfig.quant_config``), and ``route`` sees only the
pretrained config. An NVFP4 checkpoint is one of the preconditions the target
checks when it is built.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..._router_index import NoTarget

# The one GPU architecture the target is written for.
_SM = (10, 3)

# The checkpoint's config shape, read off the flat config; see the gpt_oss routing
# module for what shape-sniffing does and does not pin. The attention fields fix
# the attention operands, the MoE fields the expert operands, and the last two the
# layer pattern: every fourth layer, counted from the last, is a full-attention
# NoPE layer, and every other layer is a 2048-token sliding-window RoPE layer.
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


def _tep(m: Any) -> bool:
    """Attention heads tensor-parallel and experts expert-parallel over the same ranks,
    with one expert slice per rank and no attention data parallelism.

    The split decides which weights each rank loads and into which shapes, so it
    selects a target rather than a runtime branch. Any tensor-parallel size.
    """
    return (
        m.world_size == m.tp_size
        and m.moe_ep_size == m.tp_size
        and m.moe_tp_size == 1
        and m.pp_size == 1
        and m.cp_size == 1
        and not m.enable_attention_dp
    )


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    ``spec_config`` is part of the routing signature and is not read here (see the
    module docstring). The target is imported where it is returned, so declining
    costs no import of modeling code.
    """
    # A config without one of these fields is another assembly: it reads as None
    # and the shape does not match.
    shape = tuple(getattr(pretrained_config, name, None) for name in _SHAPE_FIELDS)

    # Muse-Spark-Open-816B-A42B on sm103, tensor-parallel attention and
    # expert-parallel MoE over the same ranks.
    served = compute_capability == _SM and shape == _SHAPE and _tep(mapping)
    if not served:
        raise NoTarget(
            f"compute capability {compute_capability}, "
            f"config shape {shape}, parallel ws={mapping.world_size} tp={mapping.tp_size} "
            f"pp={mapping.pp_size} cp={mapping.cp_size} moe_ep={mapping.moe_ep_size} "
            f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp}; the one "
            "musespark_open target is Muse-Spark-Open-816B-A42B on sm103, attention "
            "tensor-parallel and the experts expert-parallel over the same ranks at any size "
            "(moe_tp 1, pp 1, cp 1, no attention DP)"
        )

    # One line, and at this depth, on purpose: the claims suite reads the routed
    # targets off `from .<target>.modeling import <Class>` lines, so the import
    # cannot be wrapped to fit the line limit.
    # fmt: off
    # isort: off
    from .muse_spark_open_816b_a42b_nvfp4__sm_103__tep.modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
    # isort: on
    # fmt: on

    return ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
