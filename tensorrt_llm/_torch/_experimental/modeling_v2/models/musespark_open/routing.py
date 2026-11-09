# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
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

There is one target per GPU architecture: ``muse_spark_open_816b_a42b_nvfp4__sm_103__tep``
on sm103 and ``muse_spark_open_816b_a42b_nvfp4__sm_107__tep`` on sm107. They share no
code. Each claims every tensor-parallel size of its layout. Which fast paths a
deployment runs is the target's own decision at construction
(``dispatch.compose``): the composition of its tensor-parallel size when the GPU
topology and the speculative decoding mode are the ones the compositions are
written for, and the built-in forward through the target otherwise. So
``route`` decides on neither the speculative-decoding config nor the GPUs per node
(launcher state: the visible devices, or the tensor-parallel size under Ray),
and a deployment a target serves without fast paths still lands there.

When it returns the sm107 target, ``route`` applies that target's forced-acceptance
policy to the speculative config (``rs_common.apply_forced_policy_to_spec_config``:
under ``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` rejection sampling is off at
TP8, where the ``rejection_sampler`` fast path is built, unless that lever is
turned off; ``rs_common.policy_applies``). It is the engine's own config object,
and this is the one point the target sees it before the model loader deep-copies
the model config for the model. Elsewhere, and on sm103, the config is left as it is.

The quantization is not a criterion either: the open-format translation drops
the inline ``quantization_config`` (``hf_quant_config.json`` is the one copy the
engine reads, into ``ModelConfig.quant_config``), and ``route`` sees only the
pretrained config. An NVFP4 checkpoint is one of the preconditions the target
checks when it is built.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..._router_index import NoTarget

# The GPU architectures routed here, one target each: GB300 (sm103) and Vera Rubin
# (sm107). The sm107 target's compositions need the built-in fused kernels opted in
# (its ``dispatch.SM107_FUSED_KERNELS``).
_SM103 = (10, 3)
_SM107 = (10, 7)

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


#: GB300 TEP8 F11 (``muse_spark_open_816b_a42b_nvfp4__sm_103__tep8``): opt-in, the same switch the served GB300
#: TEP8 images read (``sm107_defaults.SM103_TEP8_F11_ENV`` there).
_SM103_TEP8_F11_ENV = "MUSESPARK_OPEN_SM103_TEP8_F11"


def _sm103_tep8_f11(m: Any) -> bool:
    """sm103 TEP8 under ``MUSESPARK_OPEN_SM103_TEP8_F11=1``: the GB300 F11 target (the TEP8 levers,
    their ``MUSESPARK_OPEN_SM103_TEP8_*`` / ``MUSESPARK_OPEN_TEP8_*`` switches read as they were served).
    Unset: the sm103 TEP target as upstream."""
    import os

    return int(m.tp_size) == 8 and os.environ.get(_SM103_TEP8_F11_ENV, "") == "1"


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    ``spec_config`` does not take part in the decision (see the module docstring);
    when the sm107 target is returned, its forced-acceptance policy is applied to it
    (at TP8 only).
    Each target is imported where it is returned, so declining costs no import of
    modeling code.

    Raises:
        NoTarget: the deployment is neither target's.
    """
    # A config without one of these fields is another assembly: it reads as None
    # and the shape does not match.
    shape = tuple(getattr(pretrained_config, name, None) for name in _SHAPE_FIELDS)

    # Muse-Spark-Open-816B-A42B on sm103 or sm107, tensor-parallel attention and
    # expert-parallel MoE over the same ranks.
    capability = tuple(compute_capability)
    served = capability in (_SM103, _SM107) and shape == _SHAPE and _tep(mapping)
    if not served:
        raise NoTarget(
            f"compute capability {compute_capability}, "
            f"config shape {shape}, parallel ws={mapping.world_size} tp={mapping.tp_size} "
            f"pp={mapping.pp_size} cp={mapping.cp_size} moe_ep={mapping.moe_ep_size} "
            f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp}; the "
            "musespark_open targets are Muse-Spark-Open-816B-A42B on sm103 and on sm107, "
            "attention tensor-parallel and the experts expert-parallel over the same ranks at "
            "any size (moe_tp 1, pp 1, cp 1, no attention DP)"
        )

    # One line per target, and at this depth, on purpose: the claims suite reads the
    # routed targets off `from .<target>.modeling import <Class>` lines, so the imports
    # cannot be wrapped to fit the line limit.
    if capability == _SM103:
        if _sm103_tep8_f11(mapping):
            # fmt: off
            # isort: off
            from .muse_spark_open_816b_a42b_nvfp4__sm_103__tep8.modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep8
            # isort: on
            # fmt: on
            return ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep8

        # fmt: off
        # isort: off
        from .muse_spark_open_816b_a42b_nvfp4__sm_103__tep.modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
        # isort: on
        # fmt: on
        return ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep

    # fmt: off
    # isort: off
    from .muse_spark_open_816b_a42b_nvfp4__sm_107__tep.modeling import ModelingV2MuseSparkOpen816bA42bNvfp4Sm107Tep
    # isort: on
    # fmt: on
    from .muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.rs_sampler import rs_common

    major, minor = compute_capability
    rs_common.apply_forced_policy_to_spec_config(
        spec_config, 10 * int(major) + int(minor), mapping.tp_size
    )
    return ModelingV2MuseSparkOpen816bA42bNvfp4Sm107Tep
