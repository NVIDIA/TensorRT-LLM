# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM router_gemm entry.

The entry owns the contract. This file turns a cell into tensors and holds the claims no cell
carries: bitwise agreement with the shared router GEMV, determinism, every token count up to the
ceiling, CUDA graph replay, the op's own rejections, and the guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm.router_gemm import (  # noqa: E501
    HIDDEN_SIZE,
    MAX_TOKENS,
    NUM_EXPERTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm.router_gemm import (  # noqa: E501
    router_gemm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "router_gemm requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM router GEMV is certified on sm_107 only"
)

DEV = "cuda"
BF16 = torch.bfloat16


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """FP32 accumulation against an FP64 reference, scored in FP32 ulps.

    Both sides sum the same products (a BF16 x BF16 product is exact in FP32); the reference in
    FP64, the kernel in FP32: per thread a chain of 48 FMAs, then a 5-level warp butterfly and the
    four warp sums. With randn operands the logits sit near sqrt(6144) ~ 80 (row maxima ~230),
    and the FP32 roundings of those additions add up to about 2.5e-5 per logit: about 3 ulp
    (2^-23) of relative RMS and about 3 ulp of the row maximum at worst.

    So: 64 ulp element-wise against the row scale and 32 ulp of relative RMS, tenfold over the
    estimate. A BF16 rounding of the logits alone would be 2^-9, four orders above, and a missing
    k chunk, expert or token misses by more.
    """
    assert_within_ulp(out, ref, element_ulp=64.0, rms_ulp=32.0, ulp=ULP[torch.float32])


def _build(tokens, seed):
    """randn activations and gate weight, both BF16."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    hidden_states = torch.randn(tokens, HIDDEN_SIZE, generator=g, device=DEV).to(BF16)
    weight = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, generator=g, device=DEV).to(BF16)
    return hidden_states, weight


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    hidden_states, weight = _build(cell.spec["tokens"], zlib.crc32(cell.why.encode()) % 2**31)
    logits = op(hidden_states, weight)
    assert logits.shape == (hidden_states.shape[0], NUM_EXPERTS)
    assert logits.dtype == torch.float32 and logits.is_contiguous()
    _compare(logits, op.reference(hidden_states, weight))


# ── the claims no cell carries ────────────────────────────────────────────


@pytest.mark.parametrize("tokens", range(1, 17))
def test_bitwise_the_shared_router_gemv(tokens) -> None:
    """dsv3_router_gemm_op runs the same loads and FMA chain; only the trigger and the L2
    prefetch differ, and neither touches the arithmetic. Every token count is an instance."""
    hidden_states, weight = _build(tokens, seed=tokens)
    logits = op(hidden_states, weight)
    shared = torch.ops.trtllm.dsv3_router_gemm_op(hidden_states, weight.t(), None, torch.float32)
    assert torch.equal(logits, shared)
    _compare(logits, op.reference(hidden_states, weight))


@pytest.mark.parametrize("tokens", [8, 17, 18, 24, 32])
def test_deterministic(tokens) -> None:
    hidden_states, weight = _build(tokens, seed=21)
    first = op(hidden_states, weight)
    second = op(hidden_states, weight)
    assert torch.equal(first, second) and first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("tokens", [1, 18, 32])
def test_cuda_graph_replay(tokens) -> None:
    """The decode step runs as one CUDA graph: the captured activations change between replays."""
    hidden_states, weight = _build(tokens, seed=4)
    op(hidden_states, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = op(hidden_states, weight)
    for seed in (5, 6):
        hidden_states.copy_(_build(tokens, seed)[0])
        graph.replay()
        torch.cuda.synchronize()
        _compare(logits, op.reference(hidden_states, weight))


# ── the op's own rejections ───────────────────────────────────────────────


def test_op_rejects_what_the_kernel_cannot_run() -> None:
    """Loud on the op's own, with no guard interposed."""
    hidden_states, weight = _build(8, seed=9)
    g = torch.Generator(device=DEV).manual_seed(10)
    hidden_7168 = torch.randn(8, 7168, generator=g, device=DEV).to(BF16)
    weight_7168 = torch.randn(NUM_EXPERTS, 7168, generator=g, device=DEV).to(BF16)
    weight_128 = weight[:128]
    hidden_33 = torch.randn(33, HIDDEN_SIZE, generator=g, device=DEV).to(BF16)
    strided_hidden_states = torch.randn(HIDDEN_SIZE, 8, generator=g, device=DEV).to(BF16).t()
    flat = torch.zeros(8 * HIDDEN_SIZE + 8, dtype=BF16, device=DEV)
    misaligned = flat[1 : 1 + 8 * HIDDEN_SIZE].view(8, HIDDEN_SIZE)
    cases = [
        ("fp16 hidden_states", lambda: op(hidden_states.half(), weight)),
        ("fp32 weight", lambda: op(hidden_states, weight.float())),
        ("3D hidden_states", lambda: op(hidden_states.unsqueeze(0), weight)),
        ("hidden 7168", lambda: op(hidden_7168, weight_7168)),
        ("128 experts", lambda: op(hidden_states, weight_128)),
        ("33 tokens", lambda: op(hidden_33, weight)),
        ("0 tokens", lambda: op(hidden_states[:0], weight)),
        ("strided hidden_states", lambda: op(strided_hidden_states, weight)),
        ("weight.t()", lambda: op(hidden_states, weight.t())),
        ("misaligned hidden_states", lambda: op(misaligned, weight)),
        ("cpu weight", lambda: op(hidden_states, weight.cpu())),
    ]
    for tag, call in cases:
        with pytest.raises((RuntimeError, NotImplementedError)):
            call()
            pytest.fail(f"{tag} was accepted")


# ── the guards ────────────────────────────────────────────────────────────


def test_guard_refuses_another_hidden_size() -> None:
    g = torch.Generator(device=DEV).manual_seed(1)
    hidden_states = torch.randn(1, 7168, generator=g, device=DEV).to(BF16)
    weight = torch.randn(NUM_EXPERTS, 7168, generator=g, device=DEV).to(BF16)
    with pytest.raises(AssertionError, match=r"\[tokens, 6144\]"):
        with validating(op):
            op(hidden_states, weight)


def test_guard_refuses_another_expert_count() -> None:
    hidden_states, weight = _build(1, seed=2)
    with pytest.raises(AssertionError, match=r"weight must be \[256, 6144\]"):
        with validating(op):
            op(hidden_states, weight[:128])


@pytest.mark.parametrize("tokens", [0, 33, 64])
def test_guard_refuses_a_token_count_without_an_instance(tokens) -> None:
    hidden_states, weight = _build(max(tokens, 1), seed=3)
    with pytest.raises(AssertionError, match="1 to 32 tokens"):
        with validating(op):
            op(hidden_states[:tokens], weight)


@pytest.mark.parametrize("tokens", range(17, MAX_TOKENS + 1))
def test_tensorcore_router_all_token_counts(tokens) -> None:
    hidden_states, weight = _build(tokens, seed=100 + tokens)
    _compare(op(hidden_states, weight), op.reference(hidden_states, weight))
