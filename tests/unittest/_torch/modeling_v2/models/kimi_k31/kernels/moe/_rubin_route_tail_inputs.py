# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router logits of the routing tail tests: ``router_logits`` for the kernel's tests,
``request_logits`` for the rubin_route_tail and rubin_route_topk_out entry tests."""

import torch

NUM_EXPERTS = 896


def router_logits(m, seed, mode="rand"):
    """``(logits, bias)`` of ``m`` tokens on the GPU, FP32. ``mode``: ``rand``, ``realistic``, or
    ``ties`` (all-equal rows and rows with few distinct values)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "realistic":
        # Router-like logits: hidden states (unit norm) against unit expert directions, plus
        # per-expert popularity (some hot experts); the tokens of one request (an MTP verify
        # window of 7) share most of their hidden state; bias spread ~0.19 as in Kimi K3.1.
        dim = 512
        directions = torch.randn((NUM_EXPERTS, dim), device="cuda", generator=gen)
        directions = directions / directions.norm(dim=1, keepdim=True)
        popularity = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.7
        requests = torch.randn(((m + 6) // 7, dim), device="cuda", generator=gen)
        hidden = requests.repeat_interleave(7, dim=0)[:m] * 0.85
        hidden = hidden + torch.randn((m, dim), device="cuda", generator=gen) * 0.5
        hidden = hidden / hidden.norm(dim=1, keepdim=True)
        logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
        bias = (torch.rand((NUM_EXPERTS,), device="cuda", generator=gen) - 0.5) * 0.19
        return logits.contiguous(), bias
    if mode == "rand":
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen) * 2
        bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    else:  # ties
        logits = torch.zeros((m, NUM_EXPERTS), device="cuda")
        bias = torch.zeros((NUM_EXPERTS,), device="cuda")
        half = m // 2
        logits[:half] = torch.randint(-2, 3, (half, NUM_EXPERTS), device="cuda", generator=gen)
    return logits, bias


def request_logits(rows, seed, mode="realistic", device="cpu"):
    """``(logits, bias)`` of ``rows`` tokens drawn on the CPU and moved to ``device``, FP32:
    router-like logits (requests of 8 rows share most of their hidden state, hot experts) and a
    bias spread of ~0.19, as K3.1's; or tied scores with ``mode="ties"``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "ties":
        logits = torch.zeros(rows, NUM_EXPERTS)
        logits[: rows // 2] = torch.randint(-2, 3, (rows // 2, NUM_EXPERTS), generator=g).float()
        return logits.to(device), torch.zeros(NUM_EXPERTS, device=device)
    dim = 512
    directions = torch.randn(NUM_EXPERTS, dim, generator=g)
    directions /= directions.norm(dim=1, keepdim=True)
    popularity = torch.randn(NUM_EXPERTS, generator=g) * 0.7
    hidden = torch.randn((rows + 7) // 8, dim, generator=g).repeat_interleave(8, 0)[:rows] * 0.85
    hidden = hidden + torch.randn(rows, dim, generator=g) * 0.5
    hidden /= hidden.norm(dim=1, keepdim=True)
    logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
    bias = (torch.rand(NUM_EXPERTS, generator=g) - 0.5) * 0.19
    return logits.contiguous().to(device), bias.to(device)
