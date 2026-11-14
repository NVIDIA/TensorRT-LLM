# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""kernels/megamoe: the fused MegaMoE top-k reduce tail and the add + RMSNorm (MXFP8) kernels, bit for bit against
the stock MegaMoE reduce and FlashInfer's fused add + RMSNorm."""

import functools

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="needs an SM100+ GPU",
)

EPS = 1e-6


@functools.cache
def _stock_reduce(num_rows: int, num_topk: int, hidden_size: int):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.topk_reduce import TopkReduce
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def import CombineFormat

    reducer = TopkReduce(hidden_size, num_topk, CombineFormat.parse("bf16"))

    def to_cute(tensor):
        return from_dlpack(tensor, assumed_align=16 if tensor.dtype == torch.bfloat16 else 4)

    def stream():
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    plane = torch.empty(num_rows, num_topk, hidden_size, dtype=torch.bfloat16, device="cuda")
    scores = torch.empty(num_rows, num_topk, dtype=torch.float32, device="cuda")
    out = torch.empty(num_rows, hidden_size, dtype=torch.bfloat16, device="cuda")
    compiled = cute.compile(reducer, to_cute(plane), None, to_cute(out), to_cute(scores), stream())

    def run(plane, scores):
        out = torch.empty(plane.shape[0], hidden_size, dtype=torch.bfloat16, device="cuda")
        compiled(to_cute(plane), None, to_cute(out), to_cute(scores), stream())
        return out

    return run


def _inputs(num_rows, num_topk, hidden_size, case, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    plane = randn(num_rows, num_topk, hidden_size)
    if case == "wide_range":
        exponent = torch.randint(-30, 30, plane.shape, generator=generator, device="cuda").float()
        plane = plane * torch.exp2(exponent)
    elif case == "cancellation":
        plane[:, 1::2] = -plane[:, 0::2][:, : plane[:, 1::2].shape[1]] * (1 + 2**-7)
    elif case == "subnormal":
        plane = plane * 2.0**-130
    plane = plane.to(torch.bfloat16)
    raw = torch.rand(num_rows, num_topk, generator=generator, device="cuda")
    scores = (raw / raw.sum(dim=-1, keepdim=True) * 2.5).float()
    shared = randn(num_rows, hidden_size).to(torch.bfloat16)
    residual = (randn(num_rows, hidden_size) * 4).to(torch.bfloat16)
    weight = (1 + 0.1 * randn(hidden_size)).to(torch.bfloat16)
    return plane, scores, shared, residual, weight


def _stock_moe(plane, scores, shared):
    routed = _stock_reduce(*plane.shape)(plane, scores)
    return shared.clone().add_(routed)


def _bits(tensor):
    return tensor.view(torch.int16)


CASES = ("normal", "wide_range", "cancellation", "subnormal")


@pytest.mark.parametrize("hidden_size,num_topk", [(6144, 4), (3072, 8)])
@pytest.mark.parametrize("num_rows", [1, 3, 48, 64, 192])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_shared_add_form_matches_stock(hidden_size, num_topk, num_rows, case, enable_pdl):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_shared_add,
    )

    plane, scores, shared, _, _ = _inputs(num_rows, num_topk, hidden_size, case)
    expected = _stock_moe(plane, scores, shared)
    plane_before = plane.clone()
    megamoe_topk_reduce_shared_add(plane, scores, shared, enable_pdl=enable_pdl)
    torch.cuda.synchronize()
    assert torch.equal(_bits(shared), _bits(expected))
    assert torch.equal(_bits(plane), _bits(plane_before))


def test_oracle_separates_fma_from_mul_add():
    plane, scores, shared, _, _ = _inputs(64, 4, 6144, "cancellation")
    routed = _stock_reduce(64, 4, 6144)(plane, scores)
    terms = plane.float() * scores.unsqueeze(-1)
    unfused = terms[:, 0].clone()
    for k in range(1, 4):
        unfused = unfused + terms[:, k]
    assert not torch.equal(_bits(routed), _bits(unfused.to(torch.bfloat16)))


def _norm_matches():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm_matches_flashinfer,
    )

    return megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(torch.device("cuda"), 6144)


@pytest.mark.parametrize("num_rows", [1, 3, 48, 64, 192])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_norm_form_matches_stock(num_rows, case, enable_pdl):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm,
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    plane, scores, shared, residual, weight = _inputs(num_rows, 4, 6144, case)
    expected_out = _stock_moe(plane, scores, shared)
    expected_residual = residual.clone()
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
    megamoe_topk_reduce_add_rmsnorm(
        plane, scores, shared, residual, weight, EPS, enable_pdl=enable_pdl
    )
    torch.cuda.synchronize()
    assert torch.equal(_bits(residual), _bits(expected_residual))
    assert torch.equal(_bits(shared), _bits(expected_out))


@pytest.mark.parametrize("form", ["shared_add", "norm"])
def test_graph_replay_matches_stock(form):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe import (
        megamoe_topk_reduce_tail as tail,  # noqa: E501
    )

    if form == "norm" and not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    plane, scores, shared, residual, weight = _inputs(48, 4, 6144, "normal")
    shared_in, residual_in = shared.clone(), residual.clone()

    def run():
        shared.copy_(shared_in)
        residual.copy_(residual_in)
        if form == "shared_add":
            tail.megamoe_topk_reduce_shared_add(plane, scores, shared, enable_pdl=True)
        else:
            tail.megamoe_topk_reduce_add_rmsnorm(
                plane, scores, shared, residual, weight, EPS, enable_pdl=True
            )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in (1, 2):
        new_plane, new_scores, new_shared, new_residual, _ = _inputs(48, 4, 6144, "normal", seed)
        plane.copy_(new_plane)
        scores.copy_(new_scores)
        shared_in.copy_(new_shared)
        residual_in.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        expected = _stock_moe(new_plane, new_scores, new_shared)
        if form == "norm":
            expected_residual = new_residual.clone()
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected, expected_residual, weight, EPS)
            assert torch.equal(_bits(residual), _bits(expected_residual))
        assert torch.equal(_bits(shared), _bits(expected))


def test_zero_rows_launch_nothing():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_shared_add,
    )

    plane, scores, shared, _, _ = _inputs(0, 4, 6144, "normal")
    megamoe_topk_reduce_shared_add(plane, scores, shared, enable_pdl=False)


# Widest first, so each narrower step's last scale tile still holds the previous step's rows.
MXFP8_ROWS = (448, 192, 129, 88, 64, 8, 3, 1)


def _mxfp8_out(num_rows, hidden_size=6144):
    """Garbage-filled MXFP8 outputs, so a byte the kernel leaves unwritten fails the comparison."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.mxfp8_rows import (
        scale_bytes,
    )

    q = torch.full((num_rows, hidden_size), 0xAB, dtype=torch.uint8, device="cuda")
    sf = torch.full((scale_bytes(num_rows, hidden_size),), 0xAB, dtype=torch.uint8, device="cuda")
    return q.view(torch.float8_e4m3fn), sf


def _assert_stock_quantize(rows, q, sf):
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(rows, True)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert ref_sf.numel() == sf.numel()
    assert torch.equal(sf, ref_sf.view(torch.uint8).view(-1))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_norm_form_with_mxfp8_rows_is_the_stock_tail_then_the_quantize(case, enable_pdl):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm,
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    for num_rows in MXFP8_ROWS:
        plane, scores, shared, residual, weight = _inputs(num_rows, 4, 6144, case, num_rows)
        expected_out = _stock_moe(plane, scores, shared)
        expected_residual = residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
        q, sf = _mxfp8_out(num_rows)
        megamoe_topk_reduce_add_rmsnorm(
            plane, scores, shared, residual, weight, EPS, enable_pdl=enable_pdl, q=q, sf=sf
        )
        torch.cuda.synchronize()
        assert torch.equal(_bits(residual), _bits(expected_residual)), num_rows
        assert torch.equal(_bits(shared), _bits(expected_out)), num_rows
        _assert_stock_quantize(expected_out, q, sf)


def _add_rmsnorm_inputs(num_rows, case, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    rows = randn(num_rows, 6144)
    residual = randn(num_rows, 6144) * 4
    if case == "wide_range":
        exponent = torch.randint(-30, 30, rows.shape, generator=generator, device="cuda").float()
        rows = rows * torch.exp2(exponent)
    rows, residual = rows.to(torch.bfloat16), residual.to(torch.bfloat16)
    if case == "zero_rows":
        rows[::3] = 0
        residual[::3] = 0
    weight = (1 + 0.1 * randn(6144)).to(torch.bfloat16)
    return rows, residual, weight


@pytest.mark.parametrize("case", ["normal", "wide_range", "zero_rows"])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_add_rmsnorm_mxfp8_is_flashinfer_then_the_quantize(case, enable_pdl):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_mxfp8,  # noqa: E501
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    for num_rows in MXFP8_ROWS:
        rows, residual, weight = _add_rmsnorm_inputs(num_rows, case, num_rows)
        expected_out, expected_residual = rows.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
        q, sf = _mxfp8_out(num_rows)
        add_rmsnorm_mxfp8(rows, residual, weight, EPS, q, sf, enable_pdl=enable_pdl)
        torch.cuda.synchronize()
        assert torch.equal(_bits(residual), _bits(expected_residual)), num_rows
        assert torch.equal(_bits(rows), _bits(expected_out)), num_rows
        _assert_stock_quantize(expected_out, q, sf)


def test_add_rmsnorm_mxfp8_graph_replay_matches_stock():
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_mxfp8,  # noqa: E501
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    rows, residual, weight = _add_rmsnorm_inputs(88, "normal", 0)
    rows_in, residual_in = rows.clone(), residual.clone()
    q, sf = _mxfp8_out(88)

    def run():
        rows.copy_(rows_in)
        residual.copy_(residual_in)
        add_rmsnorm_mxfp8(rows, residual, weight, EPS, q, sf, enable_pdl=True)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in (1, 2):
        new_rows, new_residual, _ = _add_rmsnorm_inputs(88, "normal", seed)
        rows_in.copy_(new_rows)
        residual_in.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        expected_residual = new_residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(new_rows, expected_residual, weight, EPS)
        assert torch.equal(_bits(residual), _bits(expected_residual))
        assert torch.equal(_bits(rows), _bits(new_rows))
        _assert_stock_quantize(new_rows, q, sf)


TINY_BLOCK = 176


def _scale_matrix(sf, num_rows, hidden_size):
    """The swizzled 128x4 scale bytes as a [num_rows, hidden_size / 32] matrix."""
    r = torch.arange(num_rows, device=sf.device)[:, None]
    b = torch.arange(hidden_size // 32, device=sf.device)[None, :]
    tiles_k = (hidden_size // 32 + 3) // 4
    offset = (
        (r // 128) * (tiles_k * 512)
        + (b // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + b % 4
    )
    return sf.view(torch.uint8).view(-1)[offset]


def _assert_sf0_fixed_quantize(rows, q, sf):
    """q / sf are trtllm::mxfp8_quantize(rows, True) with each block of scale byte 0 quantized to zero, signs kept, as
    the fixed quantizer writes them; an unfixed quantizer's NaN / +-448 codes there are replaced before comparing."""
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(rows, True)
    num_rows, hidden_size = rows.shape
    zero_blocks = (_scale_matrix(ref_sf, num_rows, hidden_size) == 0).repeat_interleave(32, dim=1)
    signed_zero = torch.where(torch.signbit(rows), 0x80, 0x00).to(torch.uint8)
    ref_q = torch.where(zero_blocks, signed_zero, ref_q.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), ref_q)
    assert torch.equal(sf, ref_sf.view(torch.uint8).view(-1))


def _with_tiny_block(*tensors, seed):
    """Block TINY_BLOCK of the first tensor's rows gets values of magnitude <= 3.49e-39 and exact zeros, and of the
    others zeros, so the summed and normalized rows' block takes the MXFP8 scale byte 0 (amax <= 448 * 2^-127)."""
    first = tensors[0]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    block = (torch.rand((first.shape[0], 32), generator=gen, device="cuda") * 2 - 1) * 3.49e-39
    block[:, 0::5] = 0.0
    span = slice(TINY_BLOCK * 32, (TINY_BLOCK + 1) * 32)
    first[:, span] = block.to(first.dtype)
    for t in tensors[1:]:
        t[..., span] = 0


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_the_mxfp8_epilogues_quantize_a_scale_byte_0_block_to_zero(enable_pdl):
    """A normalized block with 0 < amax <= 448 * 2^-127 quantizes to zero, signs kept, in the standalone norm and in
    form (b); every other block is the stock quantize's, and no code is NaN."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_mxfp8,  # noqa: E501
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm,
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    for num_rows in (128, 88, 8):
        rows, residual, weight = _add_rmsnorm_inputs(num_rows, "normal", num_rows)
        _with_tiny_block(rows, residual, seed=num_rows)
        expected_out, expected_residual = rows.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
        q, sf = _mxfp8_out(num_rows)
        add_rmsnorm_mxfp8(rows, residual, weight, EPS, q, sf, enable_pdl=enable_pdl)
        torch.cuda.synchronize()
        assert torch.equal(_bits(rows), _bits(expected_out)), num_rows
        assert bool((_scale_matrix(sf, num_rows, 6144)[:, TINY_BLOCK] == 0).all()), num_rows
        assert not bool(((q.view(torch.uint8) & 0x7F) == 0x7F).any()), num_rows
        _assert_sf0_fixed_quantize(expected_out, q, sf)

        plane, scores, shared, residual, weight = _inputs(
            num_rows, 4, 6144, "normal", seed=num_rows
        )
        _with_tiny_block(shared, plane, residual, seed=num_rows + 1)
        expected_out = _stock_moe(plane, scores, shared)
        expected_residual = residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
        q, sf = _mxfp8_out(num_rows)
        megamoe_topk_reduce_add_rmsnorm(
            plane, scores, shared, residual, weight, EPS, enable_pdl=enable_pdl, q=q, sf=sf
        )
        torch.cuda.synchronize()
        assert torch.equal(_bits(shared), _bits(expected_out)), num_rows
        assert not bool(((q.view(torch.uint8) & 0x7F) == 0x7F).any()), num_rows
        _assert_sf0_fixed_quantize(expected_out, q, sf)


def _replicas_out(num_rows, copies, hidden_size=6144):
    """A garbage-filled replicas buffer, so an element the kernel leaves unwritten fails the comparison."""
    full = torch.full((num_rows, copies * hidden_size), -1, dtype=torch.int16, device="cuda")
    return full.view(torch.bfloat16)


@pytest.mark.parametrize("case", ["normal", "wide_range", "zero_rows"])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("copies", [3, 1])
def test_add_rmsnorm_replicated_is_flashinfer_then_the_cat(case, enable_pdl, copies):
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_replicated,  # noqa: E501
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    for num_rows in MXFP8_ROWS:
        rows, residual, weight = _add_rmsnorm_inputs(num_rows, case, num_rows)
        expected_out, expected_residual = rows.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(expected_out, expected_residual, weight, EPS)
        replicas = _replicas_out(num_rows, copies)
        add_rmsnorm_replicated(rows, residual, weight, EPS, replicas, enable_pdl=enable_pdl)
        torch.cuda.synchronize()
        assert torch.equal(_bits(residual), _bits(expected_residual)), num_rows
        assert torch.equal(_bits(rows), _bits(expected_out)), num_rows
        expected = torch.cat([expected_out] * copies, dim=-1)
        assert torch.equal(_bits(replicas), _bits(expected)), num_rows


def test_add_rmsnorm_replicated_graph_replay_matches_stock():
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_replicated,  # noqa: E501
    )

    if not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    rows, residual, weight = _add_rmsnorm_inputs(88, "normal", 0)
    rows_in, residual_in = rows.clone(), residual.clone()
    replicas = _replicas_out(88, 3)

    def run():
        rows.copy_(rows_in)
        residual.copy_(residual_in)
        add_rmsnorm_replicated(rows, residual, weight, EPS, replicas, enable_pdl=True)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seed in (1, 2):
        new_rows, new_residual, _ = _add_rmsnorm_inputs(88, "normal", seed)
        rows_in.copy_(new_rows)
        residual_in.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        expected_residual = new_residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(new_rows, expected_residual, weight, EPS)
        assert torch.equal(_bits(residual), _bits(expected_residual))
        assert torch.equal(_bits(rows), _bits(new_rows))
        assert torch.equal(_bits(replicas), _bits(torch.cat([new_rows] * 3, dim=-1)))


def test_add_rmsnorm_replicated_bad_arguments_raise():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_replicated,  # noqa: E501
    )

    rows, residual, weight = _add_rmsnorm_inputs(8, "normal", 0)
    replicas = _replicas_out(8, 3)
    for bad in (
        replicas[:4],
        replicas[:, :100],
        replicas[:, :0],
        replicas.float(),
        _replicas_out(8, 6)[:, ::2],
    ):
        with pytest.raises(ValueError, match="replicas"):
            add_rmsnorm_replicated(rows, residual, weight, EPS, bad, enable_pdl=False)
    with pytest.raises(ValueError, match="do not match"):
        add_rmsnorm_replicated(rows, residual[:4], weight, EPS, replicas, enable_pdl=False)


def test_mxfp8_rows_bad_arguments_raise():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.add_rmsnorm_mxfp8 import (
        add_rmsnorm_mxfp8,  # noqa: E501
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm,
    )

    plane, scores, shared, residual, weight = _inputs(8, 4, 6144, "normal")
    q, sf = _mxfp8_out(8)
    with pytest.raises(ValueError, match="go together"):
        megamoe_topk_reduce_add_rmsnorm(
            plane, scores, shared, residual, weight, EPS, enable_pdl=False, q=q
        )
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        add_rmsnorm_mxfp8(shared, residual, weight, EPS, q[:4], sf, enable_pdl=False)
    with pytest.raises(ValueError, match="at least"):
        add_rmsnorm_mxfp8(shared, residual, weight, EPS, q, sf[:16], enable_pdl=False)
    with pytest.raises(ValueError, match="BF16"):
        add_rmsnorm_mxfp8(shared.float(), residual, weight, EPS, q, sf, enable_pdl=False)
    with pytest.raises(ValueError, match="do not match"):
        add_rmsnorm_mxfp8(shared, residual[:4], weight, EPS, q, sf, enable_pdl=False)


def test_bad_arguments_raise():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_shared_add,
    )

    plane, scores, shared, _, _ = _inputs(8, 4, 6144, "normal")
    with pytest.raises(ValueError, match="shapes disagree"):
        megamoe_topk_reduce_shared_add(plane, scores[:, :3], shared, enable_pdl=False)
    with pytest.raises(ValueError, match="FP32"):
        megamoe_topk_reduce_shared_add(plane, scores.half(), shared, enable_pdl=False)
    with pytest.raises(ValueError, match="contiguous"):
        megamoe_topk_reduce_shared_add(
            plane.transpose(0, 1).contiguous().transpose(0, 1), scores, shared, enable_pdl=False
        )


@pytest.mark.parametrize("form", ["stock", "shared_add", "norm"])
def test_masked_rows_ignore_their_stale_plane_slots(form):
    """Rows the pad mask sends to no expert carry weights 0 over plane slots no expert wrote this step."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm,
        megamoe_topk_reduce_shared_add,
    )

    if form == "norm" and not _norm_matches():
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    num_rows, num_topk, hidden_size = 48, 4, 6144
    plane, scores, shared, residual, weight = _inputs(num_rows, num_topk, hidden_size, "normal")
    masked = torch.arange(num_rows, device="cuda") % 5 == 2
    scores[masked] = 0.0
    clean = plane.clone()
    clean[masked] = 0
    alternating = torch.where(torch.arange(hidden_size, device="cuda") % 2 == 0, 3e38, -3e38)
    stale = plane.clone()
    stale[masked] = alternating.to(torch.bfloat16)

    def reduce(plane_in):
        if form == "stock":
            return (_stock_reduce(*plane_in.shape)(plane_in, scores),)
        out, res = shared.clone(), residual.clone()
        if form == "shared_add":
            megamoe_topk_reduce_shared_add(plane_in, scores, out, enable_pdl=False)
            return (out,)
        megamoe_topk_reduce_add_rmsnorm(plane_in, scores, out, res, weight, EPS, enable_pdl=False)
        return out, res

    expected, got = reduce(clean), reduce(stale)
    torch.cuda.synchronize()
    for want, have in zip(expected, got, strict=True):
        assert torch.equal(_bits(have[~masked]), _bits(want[~masked]))
        # A zero weight over a negative stale slot may give -0 where the clean plane gives +0.
        assert torch.equal(have[masked], want[masked])
    if form == "stock":
        assert bool((got[0][masked] == 0).all())
    elif form == "shared_add":
        assert torch.equal(_bits(got[0][masked]), _bits(shared[masked]))
