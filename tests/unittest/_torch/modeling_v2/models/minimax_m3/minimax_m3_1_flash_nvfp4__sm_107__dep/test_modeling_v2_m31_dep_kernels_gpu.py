# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 DEP target's kernels on an SM107 GPU.

TOPP_DET with keep_all_ge1 off is bitwise FlashInfer's deterministic top_p_renorm_probs on every row, including rows
with top_p = 1 (greedy and CUDA-graph padding rows), at the DEP K7 draft and verify widths; LC2's split workspace
reserve takes the attention-DP geometry (4 KV heads); LC2's MXFP8 o_proj hand-off gives the o_proj GEMM the stock
quantize's operands bit for bit; a boundary norm's MXFP8 qkv_proj hand-off is the stock norm, and gives the qkv_proj
GEMM the stock quantize's operands, bit for bit; a post-attention norm's gate operand is the stock norm, and gives the
gate the stock logits bit for bit, eagerly and under graph replay.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    lc2_decode,
    merge_mxfp8,
    norm_mxfp8,
    router_operand,
    topp_det,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

_VOCAB = 200064


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="needs an SM107 GPU")


def _top_p(rows: int, mode: str) -> torch.Tensor:
    top_p = torch.full((rows,), 0.95, dtype=torch.float32, device="cuda")
    if mode == "mixed":
        top_p[1::2] = 1.0
    elif mode == "ones":
        top_p.fill_(1.0)
    elif mode == "low":
        top_p.fill_(0.5)
    return top_p


@pytest.mark.parametrize("mode", ["p095", "mixed", "ones", "low"])
@pytest.mark.parametrize("rows", [19, 48, 96, 107, 128, 152, 159, 160, 280, 384, 448])
@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_topp_det_is_flashinfer_on_every_row(rows, mode, scale):
    flashinfer = pytest.importorskip("flashinfer")
    assert topp_det.topp_det_op() is not None, (
        "the family library lacks minimax_m31_topp_renorm_det_v3"
    )
    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + int(scale))
    logits = torch.randn(rows, _VOCAB, device="cuda", generator=gen) * scale
    probs = torch.softmax(logits, dim=-1).contiguous()
    top_p = _top_p(rows, mode)
    expected = flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)
    out = topp_det.top_p_renorm_det(probs, top_p)
    assert out is not None
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("rows", [107, 128, 159])
def test_topp_split_is_flashinfer_under_graph_replay(rows):
    """TOPP_SPLIT's two launches share the persistent workspace by row slices: replays stay bitwise FlashInfer."""
    flashinfer = pytest.importorskip("flashinfer")
    assert topp_det.topp_det_op() is not None, (
        "the family library lacks minimax_m31_topp_renorm_det_v3"
    )
    if not topp_det.TOPP_SPLIT:
        pytest.skip("TRTLLM_M31_DISABLE_TOPP_SPLIT=1")
    if torch.cuda.get_device_properties(0).multi_processor_count != 212:
        pytest.skip("the split window is priced at 212 SMs")
    assert len(topp_det.launch_plan(rows, 212)) == 2
    topp_det.prebuild()
    gen = torch.Generator(device="cuda").manual_seed(rows)
    probs = torch.softmax(
        torch.randn(rows, _VOCAB, device="cuda", generator=gen) * 4.0, dim=-1
    ).contiguous()
    top_p = _top_p(rows, "mixed")
    topp_det.top_p_renorm_det(probs, top_p)  # warm-up outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = topp_det.top_p_renorm_det(probs, top_p)
    for scale in (4.0, 1.0, 8.0):
        probs.copy_(
            torch.softmax(torch.randn(rows, _VOCAB, device="cuda", generator=gen) * scale, dim=-1)
        )
        graph.replay()
        expected = flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int32), expected.view(torch.int32))


def test_lc2_reserve_takes_4_kv_heads():
    assert hasattr(family_ops(), "minimax_m3_sm107_tp2_sparse_decode_lc_prepare"), (
        "the family library lacks minimax_m3_sm107_tp2_sparse_decode_lc_prepare"
    )
    reserved = lc2_decode.reserve_workspace(torch.device("cuda"))
    # 0 when TRTLLM_M31_DISABLE_TP2WIDE_LC2_SPLIT=1; otherwise 448 rows x 4 heads x 16 x 16 partials of 129 floats.
    assert reserved == 0 or reserved >= 448 * 4 * 16 * 16 * (128 + 1) * 4


def _lc2_step(rows: int, seed: int):
    """The sparse decode hook's arguments for one K7 verify step of rows // 8 requests at 4 KV heads: random
    packed-NVFP4 K / V over 3..9 pages per request, E4M3 block scales in [0.25, 2], and top-16 tables that select
    every causal block of a row (none has more than 9)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    reqs = rows // 8
    lens = [128 * (3 + (r * 5) % 7) - 40 for r in range(reqs)]
    pages = [(n + 127) // 128 for n in lens]
    table = torch.zeros(reqs, max(pages), dtype=torch.int32)
    topk = torch.full((rows, 4, 16), -1, dtype=torch.int32)
    start = 0
    for r, n in enumerate(pages):
        table[r, :n] = torch.arange(start, start + n, dtype=torch.int32)
        start += n
        for t in range(8):
            blocks = (lens[r] - 7 + t + 127) // 128
            topk[r * 8 + t, :, :blocks] = torch.arange(blocks, dtype=torch.int32)
    cache = torch.randint(
        0, 256, (start, 2, 4, 128, 64), dtype=torch.uint8, device="cuda", generator=gen
    )
    scales = torch.rand((start, 2, 4, 128, 8), device="cuda", generator=gen) * 1.75 + 0.25
    scales = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    q = (torch.randn((rows, 64, 128), device="cuda", generator=gen) * 2.0).to(torch.float8_e4m3fn)
    args = (
        q,
        cache[:, 0],
        cache[:, 1],
        topk.cuda().permute(1, 0, 2),
        table.cuda(),
        torch.tensor(lens).int().cuda(),
    )
    kwargs = dict(
        sm_scale=128**-0.5,
        output=torch.empty((rows, 64, 128), dtype=torch.bfloat16, device="cuda"),
        decode_query_len=8,
        k_block_scale=scales[:, 0],
        v_block_scale=scales[:, 1],
        k_global_scale=torch.full((4,), 0.05, device="cuda")[:1],
        v_global_scale=torch.full((4,), 0.03, device="cuda")[:1],
    )
    return args, kwargs


def test_merge_mxfp8_hand_off_is_the_o_proj_quantize_and_gemm(monkeypatch):
    """From the widest step down, so each narrower step's scale tiles hold the previous step's rows past its own:
    the hand-off LC2 publishes is bitwise trtllm::mxfp8_quantize of its BF16 output, and the o_proj GEMM on it is
    bitwise the stock quantize + GEMM, natively and as the decode graphs' tuned pair (each at its untuned tactics)."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        mxfp8_gemm_autotuned,
        mxfp8_quantize_gemm_autotuned,
    )

    assert lc2_decode.lc2_op() is not None, (
        "the family library lacks minimax_m3_sm107_tp2_sparse_decode_lc"
    )
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", False)
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", True)
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    device = torch.device("cuda", torch.cuda.current_device())
    lc2_decode.reserve_workspace(device)
    merge_mxfp8.reserve(device)
    gen = torch.Generator(device="cuda").manual_seed(7)
    weight, weight_scale = torch.ops.trtllm.mxfp8_quantize(
        torch.randn((6144, 8192), device="cuda", generator=gen).to(torch.bfloat16), True
    )
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    for rows in (448, 256, 192, 136, 88, 64, 40, 32, 16, 8):
        args, kwargs = _lc2_step(rows, seed=rows)
        lc2_decode.dep_lc2_decode(*args, **kwargs)
        out = kwargs["output"].view(rows, 8192)
        q8, sf = merge_mxfp8.SLOT.take(out)
        ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(out, True)
        assert torch.equal(q8.view(torch.uint8), ref_q.view(torch.uint8)), rows
        stock = torch.ops.trtllm.mxfp8_mxfp8_gemm(
            ref_q, ref_sf, weight, weight_scale, alpha, torch.bfloat16
        )
        ours = torch.ops.trtllm.mxfp8_mxfp8_gemm(
            q8, sf, weight, weight_scale, alpha, torch.bfloat16
        )
        assert torch.equal(ours.view(torch.int16), stock.view(torch.int16)), rows
        stock = mxfp8_quantize_gemm_autotuned(out, weight, weight_scale, torch.bfloat16)
        ours = mxfp8_gemm_autotuned(q8, sf, weight, weight_scale, torch.bfloat16)
        assert torch.equal(ours.view(torch.int16), stock.view(torch.int16)), rows


@pytest.mark.parametrize("rows", [88, 16], ids=["split", "cluster"])
def test_merge_mxfp8_hand_off_quantizes_a_byte_0_block_like_mxfp8_quantize(monkeypatch, rows):
    """A V scale small enough that the decode's 32-dim groups take UE8M0 byte 0 (0 < amax <= 448 * 2^-127) or byte
    1: the hand-off is still bitwise trtllm::mxfp8_quantize of the BF16 output, so its byte-0 groups are zero codes,
    and no code is a NaN."""
    assert lc2_decode.lc2_op() is not None, (
        "the family library lacks minimax_m3_sm107_tp2_sparse_decode_lc"
    )
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", False)
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", True)
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    device = torch.device("cuda", torch.cuda.current_device())
    lc2_decode.reserve_workspace(device)
    merge_mxfp8.reserve(device)
    args, kwargs = _lc2_step(rows, seed=rows)
    kwargs["v_global_scale"] = torch.full((1,), 3e-37, device="cuda")
    lc2_decode.dep_lc2_decode(*args, **kwargs)
    out = kwargs["output"].view(rows, 8192)
    q8, sf = merge_mxfp8.SLOT.take(out)
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(out, True)
    codes = q8.view(torch.uint8)
    assert torch.equal(codes, ref_q.view(torch.uint8))
    # This step's own scale bytes in the 128 x 4 swizzled tiles; the tiles' rows past it hold earlier steps'.
    r = torch.arange(rows, device="cuda").view(-1, 1)
    k = torch.arange(256, device="cuda").view(1, -1)
    at = (r // 128) * 64 * 512 + (k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4
    assert torch.equal(sf.view(torch.uint8).flatten()[at], ref_sf.view(torch.uint8).flatten()[at])
    amax = out.float().abs().view(rows, 256, 32).amax(-1)
    byte_0 = (amax > 0) & (amax <= 448 * 2.0**-127)
    assert byte_0.any(), "no group reached UE8M0 byte 0"
    assert not (codes.view(rows, 256, 32)[byte_0] & 0x7F).any()
    assert not ((codes & 0x7F) == 0x7F).any()


def test_norm_mxfp8_hand_off_is_the_qkv_proj_norm_quantize_and_gemm(monkeypatch):
    """From the widest step down: a boundary norm at a site writes FlashInfer's fused add + RMSNorm bit for bit and
    publishes trtllm::mxfp8_quantize of its output, and the qkv_proj GEMM on that is bitwise the stock quantize + GEMM,
    natively and as the decode graphs' tuned pair (each at its untuned tactics)."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm_matches_flashinfer,
    )
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        mxfp8_gemm_autotuned,
        mxfp8_quantize_gemm_autotuned,
    )
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    device = torch.device("cuda", torch.cuda.current_device())
    if not megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(device, 6144):
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    monkeypatch.setattr(norm_mxfp8, "DEP_NORM_MXFP8", True)
    monkeypatch.setattr(norm_mxfp8, "_BUFFERS", {})
    # DEP_GEMM_PROVIDER would run the qkv_proj GEMM on cuBLASLt for 65..128 rows (not bitwise; its own test covers it).
    monkeypatch.setattr(merge_mxfp8._provider, "DEP_GEMM_PROVIDER", False)
    norm_mxfp8.reserve(device, 6144)
    gen = torch.Generator(device="cuda").manual_seed(11)
    norm = RMSNorm(hidden_size=6144, eps=1e-6, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(6144, device="cuda", generator=gen))
    norm.forward = types.MethodType(norm_mxfp8._norm_forward, norm)
    weight, weight_scale = torch.ops.trtllm.mxfp8_quantize(
        torch.randn((2048, 6144), device="cuda", generator=gen).to(torch.bfloat16), True
    )
    alpha = torch.ones(1, dtype=torch.float32, device="cuda")
    for rows in (448, 256, 129, 88, 40, 8, 1):
        x = torch.randn((rows, 6144), device="cuda", generator=gen).to(torch.bfloat16)
        residual = (torch.randn((rows, 6144), device="cuda", generator=gen) * 4).to(torch.bfloat16)
        ref_x, ref_residual = x.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(ref_x, ref_residual, norm.weight, 1e-6)
        out, res = norm(x, residual)
        q8, sf = norm_mxfp8.SLOT.take(out)
        assert torch.equal(out.view(torch.int16), ref_x.view(torch.int16)), rows
        assert torch.equal(res.view(torch.int16), ref_residual.view(torch.int16)), rows
        ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(ref_x, True)
        assert torch.equal(q8.view(torch.uint8), ref_q.view(torch.uint8)), rows
        stock = torch.ops.trtllm.mxfp8_mxfp8_gemm(
            ref_q, ref_sf, weight, weight_scale, alpha, torch.bfloat16
        )
        ours = torch.ops.trtllm.mxfp8_mxfp8_gemm(
            q8, sf, weight, weight_scale, alpha, torch.bfloat16
        )
        assert torch.equal(ours.view(torch.int16), stock.view(torch.int16)), rows
        stock = mxfp8_quantize_gemm_autotuned(ref_x, weight, weight_scale, torch.bfloat16)
        ours = mxfp8_gemm_autotuned(q8, sf, weight, weight_scale, torch.bfloat16)
        assert torch.equal(ours.view(torch.int16), stock.view(torch.int16)), rows


def _router_site(monkeypatch, device):
    """A wrapped post-attention norm and MoE gate at the DEP width (hidden 6144, 128 experts), buffers reserved."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.megamoe.megamoe_topk_reduce_tail import (  # noqa: E501
        megamoe_topk_reduce_add_rmsnorm_matches_flashinfer,
    )
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if not megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(device, 6144):
        pytest.skip("FlashInfer does not run FusedAddRMSNormKernel with one CTA per row here")
    monkeypatch.setattr(router_operand, "DEP_ROUTER_OPERAND", True)
    monkeypatch.setattr(router_operand, "_BUFFERS", {})
    router_operand.SLOT.clear()
    router_operand.reserve(device, 6144)
    gen = torch.Generator(device="cuda").manual_seed(13)
    norm = RMSNorm(hidden_size=6144, eps=1e-6, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(6144, device="cuda", generator=gen))
    gate = MiniMaxM3Gate(6144, 128, 8, 2.5).to(device)
    gate.load_weights([{"weight": torch.randn((128, 6144), device="cuda", generator=gen) * 0.02}])
    norm.forward = types.MethodType(router_operand._norm_forward, norm)
    gate.forward = types.MethodType(router_operand._gate_forward, gate)
    return norm, gate, gen


def _router_rows(rows, gen):
    x = torch.randn((rows, 6144), device="cuda", generator=gen).to(torch.bfloat16)
    residual = (torch.randn((rows, 6144), device="cuda", generator=gen) * 4).to(torch.bfloat16)
    return x, residual


def test_router_operand_is_the_stock_norm_cat_and_gate(monkeypatch):
    """From the widest step down: a post-attention norm at a site is FlashInfer's fused add + RMSNorm bit for bit, and
    its gate's logits on the published copies are bitwise the stock gate's (``torch.cat`` + ``trtllm::cublas_mm``)."""
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate

    device = torch.device("cuda", torch.cuda.current_device())
    norm, gate, gen = _router_site(monkeypatch, device)
    for rows in (448, 256, 129, 88, 40, 8, 1):
        x, residual = _router_rows(rows, gen)
        ref_x, ref_residual = x.clone(), residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(ref_x, ref_residual, norm.weight, 1e-6)
        stock = MiniMaxM3Gate.forward(gate, ref_x)
        out, res = norm(x, residual)
        assert router_operand.SLOT.pending is not None, rows
        logits = gate(out)
        assert router_operand.SLOT.pending is None, rows
        assert torch.equal(out.view(torch.int16), ref_x.view(torch.int16)), rows
        assert torch.equal(res.view(torch.int16), ref_residual.view(torch.int16)), rows
        assert logits.dtype == torch.float32 and logits.shape == (rows, 128), rows
        assert torch.equal(logits.view(torch.int32), stock.view(torch.int32)), rows


def test_router_operand_graph_replay_is_the_stock_gate(monkeypatch):
    """The norm and the gate captured together replay the stock norm and logits on new rows."""
    from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate

    device = torch.device("cuda", torch.cuda.current_device())
    norm, gate, gen = _router_site(monkeypatch, device)
    x, residual = _router_rows(88, gen)
    x_in, residual_in = x.clone(), residual.clone()
    result = {}

    def run():
        x.copy_(x_in)
        residual.copy_(residual_in)
        out, _ = norm(x, residual)
        result["logits"] = gate(out)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        new_x, new_residual = _router_rows(88, gen)
        x_in.copy_(new_x)
        residual_in.copy_(new_residual)
        graph.replay()
        torch.cuda.synchronize()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(new_x, new_residual, norm.weight, 1e-6)
        stock = MiniMaxM3Gate.forward(gate, new_x)
        assert torch.equal(x.view(torch.int16), new_x.view(torch.int16))
        assert torch.equal(residual.view(torch.int16), new_residual.view(torch.int16))
        assert torch.equal(result["logits"].view(torch.int32), stock.view(torch.int32))
