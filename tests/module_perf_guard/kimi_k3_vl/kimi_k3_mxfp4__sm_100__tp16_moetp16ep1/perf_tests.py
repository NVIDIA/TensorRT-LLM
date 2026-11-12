# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf tests of Kimi K3 MXFP4 with TP16 attention and MoE TP16 x EP1, without speculation.

The tests use one rank's shapes in decode. Each request adds one token per step, so a module's
batch is the concurrency. The concurrencies in ``golden.json`` are the cases of the
``kimi_k3_tp16_moetp16ep1_nospec_1k1k`` config in
``tests/scripts/perf-sanity/aggregated/kimi_k3_mxfp4_mv2_4_nodes_grace_blackwell.yaml``.
"""

from collections.abc import Callable

import torch

from tensorrt_llm._torch.modules.rms_norm import RMSNorm

# Kimi K3 hidden size and RMSNorm epsilon
HIDDEN_SIZE = 7168
RMS_NORM_EPS = 1e-6


def build_input_layernorm(concurrency: int) -> Callable[[], torch.Tensor]:
    """Build one decode step of the decoder layer's input RMSNorm."""
    norm = RMSNorm(
        hidden_size=HIDDEN_SIZE,
        eps=RMS_NORM_EPS,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    # [concurrency, HIDDEN_SIZE], bf16
    hidden_states = torch.randn(concurrency, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    return lambda: norm(hidden_states)


# Module or kernel name in golden.json to the function that builds it for a concurrency.
MODULE_NAME_TO_BUILDER: dict[str, Callable[[int], Callable[[], object]]] = {
    "input_layernorm": build_input_layernorm,
}
