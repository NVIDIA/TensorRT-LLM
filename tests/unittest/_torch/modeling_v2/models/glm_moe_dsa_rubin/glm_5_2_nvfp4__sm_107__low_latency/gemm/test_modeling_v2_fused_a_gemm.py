# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM fused_a_gemm entry.

The entry owns the contract. This file turns a cell into tensors and holds the claims no cell
carries: the 2624 columns of the 2784-row call, determinism and the list as a pure hint,
agreement with the shared fused-A kernel, CUDA graph replay, the op's own rejections, and the
guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm.fused_a_gemm import (  # noqa: E501
    HIDDEN_SIZE,
    INDEXER_OUT_FEATURES,
    OUT_FEATURES,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm.fused_a_gemm import (  # noqa: E501
    fused_a_gemm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — needs the path declared above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "fused_a_gemm requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM fused-A GEMM is certified on sm_107 only"
)

DEV = "cuda"
BF16 = torch.bfloat16


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One GEMM's worth of accumulation-order difference, then one BF16 rounding on each side.

    Kernel and reference sum the same products (a BF16 x BF16 product is exact in FP32) in FP32
    and round once to BF16; only the summation order differs. With randn operands the outputs sit
    near sqrt(6144) ~ 80, the two FP32 sums differ by about 1e-5 there (FP32 rounding over a
    few dozen additions of partials up to that size), and the BF16 spacing there is 0.5. So an
    output either rounds to the same BF16 value or lands one rounding step away, and a step is
    at most 2 ulp (2^-8) of the row's largest magnitude.

    So: 4 ulp element-wise against the row scale (twice the worst rounding step), 2 ulp of
    relative RMS (a step on 1 % of the outputs would read as about 0.15). A wrong k stage, row
    or token is off by a fraction of the row scale and misses both by orders.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _tensor(shape, dtype, generator):
    if dtype.is_floating_point:
        return torch.randn(shape, generator=generator, device=DEV).to(dtype)
    return torch.randint(0, 1 << 20, shape, generator=generator, device=DEV, dtype=dtype)


def _build(spec, seed):
    """Materialize one cell: randn activations and weight, and the list tensors at their shapes."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    hidden_states = _tensor((spec["tokens"], HIDDEN_SIZE), BF16, g)
    weight = _tensor((spec["out_features"], HIDDEN_SIZE), BF16, g)
    l2_prefetch = [_tensor(shape, dtype, g) for shape, dtype in spec["l2_prefetch"]]
    return hidden_states, weight, l2_prefetch


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    hidden_states, weight, l2_prefetch = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    snapshots = [t.clone() for t in l2_prefetch]
    out = op(hidden_states, weight, l2_prefetch)
    assert out.shape == (hidden_states.shape[0], weight.shape[0])
    assert out.dtype == BF16 and out.is_contiguous()
    _compare(out, op.reference(hidden_states, weight, l2_prefetch))
    # The list is a cache hint: read, never written.
    for tensor, snapshot in zip(l2_prefetch, snapshots):
        assert torch.equal(tensor, snapshot)


# ── the claims no cell carries ────────────────────────────────────────────


@pytest.mark.parametrize("tokens", [1, 8])
def test_indexer_rows_leave_the_kv_a_columns_bitwise_unchanged(tokens) -> None:
    """A target keeps one 2784-row tensor and reads kv_a_proj_with_mqa as its first 2624 rows;
    every 16-row tile is computed alike in both grids."""
    g = torch.Generator(device=DEV).manual_seed(tokens)
    hidden_states = _tensor((tokens, HIDDEN_SIZE), BF16, g)
    extended = _tensor((INDEXER_OUT_FEATURES, HIDDEN_SIZE), BF16, g)
    kv_a = extended[:OUT_FEATURES]
    assert kv_a.is_contiguous() and kv_a.data_ptr() == extended.data_ptr()
    out_extended = op(hidden_states, extended, [])
    out_kv_a = op(hidden_states, kv_a, [])
    assert torch.equal(out_extended[:, :OUT_FEATURES], out_kv_a)


def test_deterministic_and_the_list_does_not_enter_the_result() -> None:
    g = torch.Generator(device=DEV).manual_seed(11)
    hidden_states = _tensor((8, HIDDEN_SIZE), BF16, g)
    weight = _tensor((INDEXER_OUT_FEATURES, HIDDEN_SIZE), BF16, g)
    l2_prefetch = [_tensor((4096, 2048), BF16, g), _tensor((6144, 1024), BF16, g)]
    first = op(hidden_states, weight, l2_prefetch)
    second = op(hidden_states, weight, l2_prefetch)
    without_list = op(hidden_states, weight, [])
    assert torch.equal(first, second)
    assert torch.equal(first, without_list)
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("tokens", [1, 8])
def test_agrees_with_the_shared_fused_a_kernel(tokens) -> None:
    """dsv3_fused_a_gemm_op (32-row tiles, 4 MMA warps) sums the same products in another order:
    nearly every output is bitwise equal, the rest one BF16 rounding step apart."""
    g = torch.Generator(device=DEV).manual_seed(100 + tokens)
    hidden_states = _tensor((tokens, HIDDEN_SIZE), BF16, g)
    weight = _tensor((OUT_FEATURES, HIDDEN_SIZE), BF16, g)
    out = op(hidden_states, weight, [])
    shared = torch.ops.trtllm.dsv3_fused_a_gemm_op(hidden_states, weight.t(), None, None)
    equal = (out == shared).float().mean().item()
    print(f"  {equal:.4%} of the outputs bitwise equal to dsv3_fused_a_gemm_op")
    assert equal >= 0.99
    assert_within_ulp(out, shared, element_ulp=4.0, rms_ulp=2.0)


def test_cuda_graph_replay() -> None:
    """The decode step runs as one CUDA graph: the tensor maps and the list are fixed at capture,
    the contents of the captured activations are not."""
    g = torch.Generator(device=DEV).manual_seed(3)
    hidden_states = _tensor((8, HIDDEN_SIZE), BF16, g)
    weight = _tensor((INDEXER_OUT_FEATURES, HIDDEN_SIZE), BF16, g)
    l2_prefetch = [_tensor((4096, 2048), BF16, g), _tensor((4, 256, 512), BF16, g)]
    op(hidden_states, weight, l2_prefetch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(hidden_states, weight, l2_prefetch)
    for _ in range(2):
        hidden_states.copy_(_tensor((8, HIDDEN_SIZE), BF16, g))
        graph.replay()
        torch.cuda.synchronize()
        _compare(out, op.reference(hidden_states, weight, l2_prefetch))


# ── the op's own rejections ───────────────────────────────────────────────


def test_op_rejects_what_the_kernel_cannot_run() -> None:
    """Loud on the op's own, with no guard interposed."""
    g = torch.Generator(device=DEV).manual_seed(5)
    hidden_states = _tensor((8, HIDDEN_SIZE), BF16, g)
    weight = _tensor((OUT_FEATURES, HIDDEN_SIZE), BF16, g)
    hidden_7168, weight_7168 = _tensor((8, 7168), BF16, g), _tensor((2112, 7168), BF16, g)
    strided_hidden_states = _tensor((HIDDEN_SIZE, 8), BF16, g).t()
    strided_weight = _tensor((HIDDEN_SIZE, OUT_FEATURES), BF16, g).t()
    flat = torch.zeros(8 * HIDDEN_SIZE + 8, dtype=BF16, device=DEV)
    misaligned = flat[1 : 1 + 8 * HIDDEN_SIZE].view(8, HIDDEN_SIZE)
    cases = [
        ("fp16 hidden_states", lambda: op(hidden_states.half(), weight, [])),
        ("fp32 weight", lambda: op(hidden_states, weight.float(), [])),
        ("3D hidden_states", lambda: op(hidden_states.unsqueeze(0), weight, [])),
        ("hidden 7168", lambda: op(hidden_7168, weight_7168, [])),
        ("2640 rows", lambda: op(hidden_states, _tensor((2640, HIDDEN_SIZE), BF16, g), [])),
        ("33 tokens", lambda: op(_tensor((33, HIDDEN_SIZE), BF16, g), weight, [])),
        ("0 tokens", lambda: op(hidden_states[:0], weight, [])),
        ("strided hidden_states", lambda: op(strided_hidden_states, weight, [])),
        ("strided weight", lambda: op(hidden_states, strided_weight, [])),
        ("misaligned hidden_states", lambda: op(misaligned, weight, [])),
        ("cpu weight", lambda: op(hidden_states, weight.cpu(), [])),
        ("five list tensors", lambda: op(hidden_states, weight, [weight] * 5)),
        ("cpu list tensor", lambda: op(hidden_states, weight, [weight.cpu()])),
        ("strided list tensor", lambda: op(hidden_states, weight, [weight.t()])),
        ("misaligned list tensor", lambda: op(hidden_states, weight, [misaligned])),
    ]
    for tag, call in cases:
        with pytest.raises((RuntimeError, NotImplementedError, ValueError)):
            call()
            pytest.fail(f"{tag} was accepted")


# ── the guards ────────────────────────────────────────────────────────────


def _ok(out_features=INDEXER_OUT_FEATURES, tokens=8):
    g = torch.Generator(device=DEV).manual_seed(out_features + tokens)
    return _tensor((tokens, HIDDEN_SIZE), BF16, g), _tensor((out_features, HIDDEN_SIZE), BF16, g)


def test_guard_refuses_another_hidden_size() -> None:
    g = torch.Generator(device=DEV).manual_seed(1)
    hidden_states, weight = _tensor((8, 7168), BF16, g), _tensor((2112, 7168), BF16, g)
    with pytest.raises(AssertionError, match=r"\[tokens, 6144\]"):
        with validating(op):
            op(hidden_states, weight, [])


def test_guard_refuses_a_weight_without_an_instance() -> None:
    hidden_states, _ = _ok()
    g = torch.Generator(device=DEV).manual_seed(2)
    with pytest.raises(AssertionError, match="2624 or 2784 rows"):
        with validating(op):
            op(hidden_states, _tensor((2640, HIDDEN_SIZE), BF16, g), [])


@pytest.mark.parametrize("tokens", [0, 33])
def test_guard_refuses_a_token_count_outside_supported_range(tokens) -> None:
    hidden_states, weight = _ok(tokens=max(tokens, 1))
    hidden_states = hidden_states[:tokens]
    with pytest.raises(AssertionError, match="1 to 32 tokens"):
        with validating(op):
            op(hidden_states, weight, [])


@pytest.mark.parametrize("out_features", [OUT_FEATURES, INDEXER_OUT_FEATURES])
def test_guard_refuses_fewer_sms_than_ctas(out_features, monkeypatch) -> None:
    """The op runs anyway, in two waves; the caller takes the shared op there."""
    hidden_states, weight = _ok(out_features)
    ctas = out_features // 16

    class _Props:
        multi_processor_count = ctas - 1

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: _Props())
    with pytest.raises(AssertionError, match=f"{ctas} CTAs need as many SMs"):
        with validating(op):
            op(hidden_states, weight, [])
