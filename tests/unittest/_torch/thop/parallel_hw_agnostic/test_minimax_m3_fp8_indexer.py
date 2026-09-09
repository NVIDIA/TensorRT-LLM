# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
FP8_AVAILABLE = CUDA_AVAILABLE and torch.cuda.get_device_capability() >= (8, 9)
pytestmark = pytest.mark.skipif(
    not FP8_AVAILABLE, reason="FP8 requires CUDA compute capability >= 8.9"
)


# The specialized and generic paths are separate CUDA kernels. A one-bin-wide
# numerical tolerance would hide precisely the regressions this test targets,
# so small comparisons deliberately require exact bytes while larger tensors
# permit strictly fewer than 0.1% mismatches.
def _assert_fp8_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Require essentially byte-identical E4M3 results across CUDA kernels."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.float8_e4m3fn
    byte_matches = actual.view(torch.uint8) == expected.view(torch.uint8)
    total = byte_matches.numel()
    mismatches = total - int(byte_matches.sum().item())
    mismatch_budget = (total - 1) // 1000
    assert mismatches <= mismatch_budget, (
        f"FP8 byte mismatches {mismatches} exceed the strict >99.9% budget "
        f"of {mismatch_budget} out of {total} values"
    )


def _reference(
    qk: torch.Tensor,
    num_heads_q: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the established BF16 fused-kernel-then-E4M3-cast result."""
    reference = qk.clone()
    torch.ops.trtllm.fused_qk_norm_rope(
        reference,
        num_heads_q,
        1,
        0,
        128,
        64,
        1e-5,
        q_weight,
        k_weight,
        10000.0,
        True,  # is_neox
        position_ids,
        1.0,
        0.0,
        0.0,
        1.0,
        True,  # is_qk_norm
        True,  # use_gemma
        False,  # use_mrope
        0,
        0,
    )
    q, k = reference.split([num_heads_q * 128, 128], dim=-1)
    return q.view(q.shape[0], num_heads_q, 128).to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)


def _strided_cache(num_pages: int, page_size: int = 128, stride_scale: int = 7) -> torch.Tensor:
    """Allocate an HND cache with the production-style noncontiguous page stride."""
    backing = torch.zeros(
        num_pages * stride_scale,
        1,
        page_size,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return backing[::stride_scale]


def _run(
    qk: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads_q: int = 4,
    head_dim: int = 128,
    rotary_dim: int = 64,
    eps: float = 1e-5,
    base: float = 10000.0,
) -> torch.Tensor:
    """Invoke the specialized indexer operator with overridable geometry."""
    return torch.ops.trtllm.minimax_m3_fp8_indexer_qk_norm_rope(
        qk,
        cache,
        slots,
        num_heads_q,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        position_ids,
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 129])
def test_minimax_m3_fp8_indexer_matches_bf16_then_cast(num_tokens: int) -> None:
    torch.manual_seed(1234)
    num_heads_q = 4
    page_size = 128
    qk = torch.randn(
        num_tokens,
        (num_heads_q + 1) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 8192
    within = torch.arange(num_tokens, dtype=torch.int32, device="cuda") % page_size
    pages = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    slots = pages * page_size + within
    cache = _strided_cache(num_tokens)

    q_out = _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)
    k_out = cache[pages.long(), 0, within.long()]

    _assert_fp8_close(q_out, q_ref)
    _assert_fp8_close(k_out, k_ref)


def test_minimax_m3_fp8_indexer_defensively_skips_invalid_direct_op_slots() -> None:
    # Production slot mapping supplies valid slots for all live tokens. This
    # direct-op regression verifies that malformed sentinel/stale slots still
    # cannot write into adjacent cache pages.
    torch.manual_seed(2345)
    num_tokens = 3
    num_heads_q = 4
    qk = torch.randn(
        num_tokens,
        (num_heads_q + 1) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    slots = torch.tensor([0, -1, 128], dtype=torch.int32, device="cuda")

    backing = torch.zeros(3, 1, 128, 128, dtype=torch.float8_e4m3fn, device="cuda")
    cache = backing[1:2]
    q_out = _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)

    _assert_fp8_close(q_out, q_ref)
    _assert_fp8_close(cache[0, 0, 0], k_ref[0])
    assert torch.count_nonzero(backing[0].view(torch.uint8)).item() == 0
    assert torch.count_nonzero(backing[2].view(torch.uint8)).item() == 0


def test_minimax_m3_fp8_indexer_cuda_graph_replay_updates_outputs() -> None:
    torch.manual_seed(5678)
    num_tokens = 16
    num_heads_q = 4
    qk = torch.randn(
        num_tokens,
        (num_heads_q + 1) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 4096
    pages = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    within = (pages * 11) % 128
    slots = pages * 128 + within
    cache = _strided_cache(num_tokens)

    for _ in range(3):
        _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q_out = _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)

    first_q = q_out.clone()
    qk.copy_(torch.randn_like(qk))
    graph.replay()
    torch.cuda.synchronize()
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)
    k_out = cache[pages.long(), 0, within.long()]

    assert not torch.equal(q_out.view(torch.uint8), first_q.view(torch.uint8))
    _assert_fp8_close(q_out, q_ref)
    _assert_fp8_close(k_out, k_ref)


def test_minimax_m3_fp8_indexer_accepts_zero_tokens() -> None:
    """Zero-token batches return an empty Q tensor without launching CUDA."""
    qk = torch.empty(0, 5 * 128, dtype=torch.bfloat16, device="cuda")
    cache = _strided_cache(1)
    slots = torch.empty(0, dtype=torch.int32, device="cuda")
    weights = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    positions = torch.empty(0, dtype=torch.int32, device="cuda")

    q_out = _run(qk, cache, slots, weights, weights, positions)

    assert q_out.shape == (0, 4, 128)
    assert q_out.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    ("head_dim", "rotary_dim", "message"),
    [
        (64, 64, "head_dim=128"),
        (128, 32, "rotary_dim=64"),
    ],
)
def test_minimax_m3_fp8_indexer_rejects_unsupported_geometry(
    head_dim: int, rotary_dim: int, message: str
) -> None:
    """The Python-visible operator rejects geometry the CUDA kernel hardcodes."""
    qk = torch.empty(1, 5 * head_dim, dtype=torch.bfloat16, device="cuda")
    cache = torch.empty(1, 1, 128, head_dim, dtype=torch.float8_e4m3fn, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    weights = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match=message):
        _run(
            qk,
            cache,
            slots,
            weights,
            weights,
            positions,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
        )


def test_minimax_m3_fp8_indexer_rejects_bad_cache_contracts() -> None:
    """Cache dtype, rank, and slot-vector length are validated before launch."""
    qk = torch.empty(2, 5 * 128, dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(2, dtype=torch.int32, device="cuda")
    weights = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros(2, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match=r"must use torch\.float8_e4m3fn"):
        _run(
            qk,
            torch.empty(1, 1, 128, 128, dtype=torch.bfloat16, device="cuda"),
            slots,
            weights,
            weights,
            positions,
        )
    with pytest.raises(RuntimeError, match="must be HND"):
        _run(
            qk,
            torch.empty(128, 128, dtype=torch.float8_e4m3fn, device="cuda"),
            slots,
            weights,
            weights,
            positions,
        )
    with pytest.raises(RuntimeError, match="shorter than num_tokens"):
        _run(
            qk,
            _strided_cache(1),
            slots[:1],
            weights,
            weights,
            positions,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eps": 0.0}, "eps must be finite and greater than zero"),
        ({"eps": float("nan")}, "eps must be finite and greater than zero"),
        ({"eps": 1e300}, "eps must remain finite and positive in float32"),
        ({"base": 0.0}, "RoPE base must be finite and greater than zero"),
        ({"base": float("inf")}, "RoPE base must be finite and greater than zero"),
        ({"base": 1e300}, "RoPE base must remain finite and positive in float32"),
    ],
)
def test_minimax_m3_fp8_indexer_rejects_invalid_scalars(
    kwargs: dict[str, float], message: str
) -> None:
    """RMS epsilon and RoPE base must define finite, positive operations."""
    qk = torch.empty(1, 5 * 128, dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    weights = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match=message):
        _run(
            qk,
            _strided_cache(1),
            slots,
            weights,
            weights,
            positions,
            **kwargs,
        )


def test_minimax_m3_fp8_indexer_rejects_misaligned_vector_accesses() -> None:
    """Vectorized loads/stores reject misaligned bases and cache page strides."""
    num_qk_elements = 5 * 128
    qk_storage = torch.empty(num_qk_elements + 1, dtype=torch.bfloat16, device="cuda")
    misaligned_qk = qk_storage[1:].view(1, num_qk_elements)
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    weights = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros(1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="8-byte-aligned"):
        _run(
            misaligned_qk,
            _strided_cache(1),
            slots,
            weights,
            weights,
            positions,
        )

    qk = torch.empty(1, num_qk_elements, dtype=torch.bfloat16, device="cuda")
    cache_elements = 128 * 128
    cache_storage = torch.empty(cache_elements + 1, dtype=torch.float8_e4m3fn, device="cuda")
    misaligned_cache = cache_storage[1:].view(1, 1, 128, 128)
    with pytest.raises(RuntimeError, match="4-byte-aligned"):
        _run(qk, misaligned_cache, slots, weights, weights, positions)

    page_stride = cache_elements + 1
    strided_storage = torch.empty(
        page_stride + cache_elements, dtype=torch.float8_e4m3fn, device="cuda"
    )
    bad_stride_cache = strided_storage.as_strided(
        (2, 1, 128, 128), (page_stride, cache_elements, 128, 1)
    )
    with pytest.raises(RuntimeError, match="page stride must be a multiple of 4"):
        _run(qk, bad_stride_cache, slots, weights, weights, positions)
