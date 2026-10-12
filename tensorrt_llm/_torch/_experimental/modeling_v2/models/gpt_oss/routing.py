# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GptOssForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in GptOss implementation; in ``require`` it raises that error.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..._router_index import NoTarget

# Target class name -> the module whose import registers it.
TARGET_MODULES = {
    "ModelingV2GptOss120bSm103Tp1": "models.gpt_oss.gpt_oss_120b__sm_103__tp1.modeling",
}


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> str:
    """Name the target for this deployment, or raise ``NoTarget`` saying why none fits."""
    # The checkpoint is recognized by its config shape, the upstream idiom
    # (``is_mla``, ``is_nemotron_hybrid`` do the same). A fine-tune with the
    # same shape routes here too; the accuracy gate is what pins the identity.
    # The expert count is in the shape because it fixes the MoE operand
    # geometry the target declares; layer count alone separates 120b from 20b.
    shape = (
        pretrained_config.num_hidden_layers,
        pretrained_config.hidden_size,
        pretrained_config.num_local_experts,
    )

    # gpt-oss-120b on one GB300 (sm103), no speculative decoding: the target
    # implements no draft path, so claiming such a deployment would crash in
    # the model instead of falling back.
    if (
        compute_capability == (10, 3)
        and shape == (36, 2880, 128)
        and mapping.world_size == 1
        and spec_config is None
    ):
        return "ModelingV2GptOss120bSm103Tp1"

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"{mapping.world_size} ranks, speculative decoding "
        f"{type(spec_config).__name__ if spec_config is not None else 'off'}; "
        "the one gpt_oss target is gpt-oss-120b on sm103, tp1, without speculative decoding"
    )
