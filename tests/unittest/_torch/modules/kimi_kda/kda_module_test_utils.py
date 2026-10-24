# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module, packed-batch, and numerical parity fixtures for KDA GPU tests."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention


def make_attention(
    heads: int,
    hidden_size: int = 7168,
    *,
    weight_seed: int = 17,
    bf16_weights: bool = False,
    flashinfer: bool = True,
) -> KimiKDALinearAttention:
    """Build an attention module with reproducible synthetic checkpoint values.

    ``flashinfer`` selects the opt-in FlashInfer prefill (``TLLM_KDA_FLASHINFER_PREFILL=1``)
    or the default in-tree prefill; the weights depend on ``weight_seed`` only.
    """
    import torch

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            num_heads=heads,
            head_dim=128,
            short_conv_kernel_size=4,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
    )
    config = ModelConfig(
        quant_config=QuantConfig()
        if bf16_weights
        else QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(weight_seed)
        flag = {"TLLM_KDA_FLASHINFER_PREFILL": "1" if flashinfer else "0"}
        with torch.device("cuda"), mock.patch.dict(os.environ, flag):
            attention = KimiKDALinearAttention(cfg, layer_idx=0, model_config=config)
        generator = torch.Generator(device="cuda").manual_seed(weight_seed)
        for module in attention.children():
            if isinstance(module, Linear):
                codes = (
                    (torch.randn(module.weight.shape, device="cuda", generator=generator) * 16)
                    .clamp(-64, 64)
                    .to(torch.float8_e4m3fn)
                )
                scales = torch.full(
                    module.weight_scale.shape, 1 / (16 * module.in_features**0.5), device="cuda"
                )
                module.load_weights([{"weight": codes, "weight_scale": scales}])
        attention.dt_bias.data.zero_()
        attention.finalize_decode_weights()
        for module in attention.modules():
            if isinstance(module, Linear):
                module.post_load_weights()
    attention.eval()
    if attention._dispatch.prefill_kernel_path != "optimized":
        raise RuntimeError("The test requires optimized production prefill")
    if attention._kda_flashinfer_prefill != flashinfer:
        raise RuntimeError(
            f"The test requires the {'FlashInfer' if flashinfer else 'in-tree'} prefill"
        )
    return attention


def use_fla_reference(attention: KimiKDALinearAttention) -> KimiKDALinearAttention:
    """Route ``attention``'s prefill to the in-tree FLA ``chunk_kda`` reference path."""
    attention._kda_flashinfer_prefill = False
    attention._dispatch.prefill_kernel_path = "fla"
    return attention


def make_batch(
    heads: int,
    lengths: list[int],
    hidden_size: int = 7168,
    *,
    initial: str = "fresh",
    input_seed: int = 23,
    padded: bool = False,
) -> SimpleNamespace:
    """Prepare all index/offset conversions and synthetic inputs outside execution."""
    import torch

    n, total = len(lengths), sum(lengths)
    generator = torch.Generator(device="cuda").manual_seed(input_seed)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    indices = torch.arange(n, 0, -1, dtype=torch.int32, device="cuda")
    has_init = [initial == "continuation" or (initial == "mixed" and i % 2 == 0) for i in range(n)]
    metadata = SimpleNamespace(
        has_initial_states=torch.tensor(has_init, dtype=torch.bool, device="cuda"),
        use_initial_states=any(has_init),
        state_indices=indices,
        query_start_loc=cu.to(torch.int32),
        query_start_loc_long=cu,
        kda_max_seqlen=max(lengths, default=0),
        _arange_buffer=torch.arange(n + 1, dtype=torch.int32, device="cuda"),
    )
    hidden = (
        torch.randn(total, hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator)
        * 0.1
    )
    conv = (
        torch.randn(
            n + 2, 3 * heads * 128, 3, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.05
    )
    state = torch.empty_strided(
        (n + 2, heads, 128, 128),
        (heads * 16384 + (8 if padded else 0), 16384, 128, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state.copy_(torch.randn(state.shape, device="cuda", generator=generator) * 0.05)
    return SimpleNamespace(
        hidden=hidden,
        cu=cu,
        indices=indices,
        metadata=metadata,
        conv=conv,
        state=state,
        n=n,
        lengths=lengths,
    )


def clone_pool(pool: torch.Tensor) -> torch.Tensor:
    """Copy pool values while preserving padding between rows."""
    import torch

    return torch.empty_strided(
        pool.shape, pool.stride(), device=pool.device, dtype=pool.dtype
    ).copy_(pool)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Report the existing KDA parity criteria and elementwise diagnostics."""
    import torch

    contract = actual.shape == expected.shape and actual.dtype == expected.dtype
    a, b = actual.float().flatten(), expected.float().flatten()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    error = a - b
    norm = b.norm().item()
    relative = error.norm().item() / max(norm, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0).item() if norm else 1.0
    return dict(
        passed=contract and finite and relative < 0.03 and cosine > 0.999,
        shape_dtype_match=contract,
        finite=finite,
        max_abs=error.abs().max().item() if finite else None,
        relative_l2=relative if finite else None,
        cosine=cosine if finite else None,
        mismatches=int((~torch.isclose(a, b, rtol=0.01, atol=0.01)).sum()),
        rtol=0.01,
        atol=0.01,
        relative_l2_limit=0.03,
        cosine_min=0.999,
    )
