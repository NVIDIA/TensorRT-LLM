# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's latent all-reduce at decode size as the consumer of pushed partials: the sum over the TP group of the
routed partials every rank's producer stored into a caller-owned :class:`K3LatentExchange`, in the MNNVL one-shot's
order."""

import torch

# Importing the op module registers trtllm::k3_latent_reduce; the state type is the one its buffers come from.
from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.latent_op import K3LatentExchange

__all__ = ["K3LatentExchange", "k3_latent_reduce"]


def k3_latent_reduce(num_tokens: int, exchange: K3LatentExchange) -> torch.Tensor:
    """Return the latent rows ``[num_tokens, 3584]`` bf16: the sum over ``exchange``'s TP group of the partial rows
    every rank pushed into it since the previous reduce. Empties the words it read and advances ``exchange`` by one
    call: every rank pushes, then reduces, the same token count on it in the same order."""
    return torch.ops.trtllm.k3_latent_reduce(exchange.uc, exchange.flags, num_tokens)
