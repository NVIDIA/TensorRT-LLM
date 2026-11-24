# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_kda_mtp_verify entry (sm_107).

Every certified cell runs the op (through its guard) and the entry's plain-torch reference on
identical inputs in the target's runtime layouts and compares the output (BF16, or the MXFP8 codes
dequantized with their scales) and every tensor the call writes in place.
"""

import zlib

import pytest
import torch
from _rubin_kda_verify_inputs import MUTATED, K, clone_mutated, dequantize, verify_args

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (
    dequantize_mxfp8,
    mxfp8_scales_from_r128c4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_mtp_verify import (
    rubin_kda_mtp_verify as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)


#: The band of the BF16 output, elementwise (check_output).
BF16_ATOL, BF16_RTOL = 3.2e-2, 8e-3


def check_output(out, ref, spec) -> None:
    if spec["out"] == "mxfp8_exact":
        # The catalog 1x32 form: the same band as the 1x128 form below, per 32-value group.
        rows, k = out[0].shape
        got_exp = mxfp8_scales_from_r128c4(out[1], rows, k).to(torch.int32)
        want_exp = mxfp8_scales_from_r128c4(ref[1], rows, k).to(torch.int32)
        assert (got_exp - want_exp).abs().max().item() <= 1, "scale exponents"
        got, want = dequantize_mxfp8(*out), dequantize_mxfp8(*ref)
        exp = torch.maximum(got_exp, want_exp).repeat_interleave(32, dim=1)
        scale = torch.pow(2.0, exp - 127.0)
        top = torch.maximum(got.abs(), want.abs())
        step = torch.maximum(torch.pow(2.0, torch.frexp(top).exponent - 4.0), scale * 2.0**-9)
        band = step + BF16_ATOL + BF16_RTOL * (top + step)
        diff = (got - want).abs()
        assert bool((diff <= band).all()), f"{(diff / band).max().item():.2f} of the band"
        return
    if spec["out"] == "mxfp8":
        heads = spec["H"]
        got, got_exp = dequantize(*out, heads)
        want, want_exp = dequantize(*ref, heads)
        # The bf16 values agree far within a factor of two, so a scale may land one binade apart
        # where a row's amax sits at 448 * 2^k.
        assert (got_exp - want_exp).abs().max().item() <= 1, "scale exponents"
        # The kernel's codes are the e4m3 rounding of its bf16 output (test_rubin_kda_mxfp8_input
        # pins that bitwise), the reference's the same rounding of its bf16 values. Those agree
        # within the BF16 band (|x| <= top + step), and each rounding moves a value by at most half
        # an e4m3 step at the larger magnitude top: 2^(floor(log2 top) - 3), or 2^-9 of the scale
        # below the normal range. So a rounding boundary between the two values puts the codes one
        # step apart: up to 2^-3 of the row amax in its top binade (2^-4 bounds one rounding).
        scale = torch.pow(2.0, torch.maximum(got_exp, want_exp) - 127.0).unsqueeze(-1)
        top = torch.maximum(got.abs(), want.abs())
        step = torch.maximum(torch.pow(2.0, torch.frexp(top).exponent - 4.0), scale * 2.0**-9)
        band = step + BF16_ATOL + BF16_RTOL * (top + step)
        diff = (got - want).abs()
        assert bool((diff <= band).all()), f"{(diff / band).max().item():.2f} of the band"
        return
    assert out.dtype is torch.bfloat16 and out.shape == ref.shape
    assert torch.isfinite(out.float()).all()
    # 3xTF32 recurrence against fp32 steps: one-ulp flips of single bf16 elements (the family's
    # kda_mtp_verify bands, GB300: rel RMS <= 1.1e-3, max abs 1.6e-2).
    diff = (out.float() - ref.float()).pow(2).mean().sqrt()
    rel = (diff / ref.float().pow(2).mean().sqrt().clamp_min(1e-12)).item()
    assert rel < 2.5e-3, f"normed core relative RMS {rel:.3e}"
    torch.testing.assert_close(out.float(), ref.float(), atol=BF16_ATOL, rtol=BF16_RTOL)


def check_writes(op_args, ref_args, untouched, before) -> None:
    slots = op_args["ssm_state_indices"].long()
    got, want = (
        op_args["recurrent_state"][slots].float(),
        ref_args["recurrent_state"][slots].float(),
    )
    if op_args["recurrent_state"].dtype == torch.float32:
        rel = ((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).item()
        assert rel < 3e-5, f"committed state relative RMS {rel:.3e}"
        torch.testing.assert_close(got, want, atol=2e-3, rtol=1e-3)
    else:  # one bf16 rounding of a state that agrees to fp32 rounding
        torch.testing.assert_close(got, want, atol=1e-2, rtol=8e-3)
    assert torch.equal(op_args["recurrent_state"][untouched], before)
    for name in ("cs_q", "cs_k", "cs_v"):  # raw inputs are copied: exact
        assert torch.equal(op_args[name], ref_args[name]), name
    for name in ("k_cache", "v_cache", "beta_cache"):
        torch.testing.assert_close(
            op_args[name].float(), ref_args[name].float(), atol=1e-2, rtol=8e-3
        )
    check_gate_cache(op_args, ref_args)


def check_gate_cache(op_args, ref_args) -> None:
    got, want = op_args["g_cache"], ref_args["g_cache"]
    if op_args["fa"] is None:  # both sides gate the same bf16 g
        torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)
        return
    # With fa / w_fb both sides gate x = bf16(fa @ w_fb^T), but the kernel adds the 128 exact
    # products in mma.sync k16 steps and the reference in one fp32 GEMM. Where the two fp32 sums
    # straddle a bf16 rounding boundary, the two x are one bf16 step apart, 2^(floor(log2|x|) - 7),
    # and lower_bound * sigmoid((x + dt_bias) * exp(A_log)) moves by at most
    # |lower_bound| * exp(A_log) / 4 per unit of x. The sums differ by a few fp32 ulps, so that is
    # rare (a few gates in 10^5); a kernel that skipped the bf16 rounding would move most of them.
    slots = op_args["ssm_state_indices"].long()
    num_spec = op_args["num_spec"]
    with true_fp32_matmul():
        x = (op_args["fa"][0].float() @ op_args["w_fb"].float().t()).to(torch.bfloat16).float()
    x = x.view(slots.numel(), num_spec + 1, -1)[:, 1:]  # the draft rows the cache keeps
    slope = abs(op_args["lower_bound"]) * op_args["A_log"].float().exp().repeat_interleave(K) / 4
    band = torch.zeros_like(want)
    band[slots, :num_spec] = slope * torch.pow(2.0, torch.frexp(x).exponent - 8.0)
    diff = (got - want).abs()
    base = 2e-5 + 1e-5 * want.abs()
    assert bool((diff <= base + band).all()), "a forget gate beyond one bf16 step of its input"
    moved = int((diff > base).sum())
    assert moved <= diff.numel() * 2.0**-10, f"{moved} of {diff.numel()} forget gates moved"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    op_args = verify_args(cell.spec, seed)
    ref_args = clone_mutated(op_args)
    used = set(op_args["ssm_state_indices"].tolist())
    untouched = [s for s in range(op_args["recurrent_state"].shape[0]) if s not in used]
    before = op_args["recurrent_state"][untouched].clone()
    out = op(**op_args)
    ref = op.reference(**ref_args)
    torch.cuda.synchronize()
    check_output(out, ref, cell.spec)
    check_writes(op_args, ref_args, untouched, before)


@torch.inference_mode()
def test_chained_rounds_follow_the_reference() -> None:
    """Three rounds on the same pools, the accepted counts rewritten between rounds: the replay of
    the previous round's drafts keeps the op on the reference."""
    spec = dict(N=4, H=12, num_spec=5, state_dtype="float32", gate="g", out="bf16")
    op_args = verify_args(spec, seed=11)
    ref_args = clone_mutated(op_args)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(3):
        out = op(**op_args)
        ref = op.reference(**ref_args)
        check_output(out, ref, spec)
        slots = op_args["ssm_state_indices"].long()
        accepted = torch.randint(0, spec["num_spec"] + 1, (spec["N"],), generator=gen)
        for args in (op_args, ref_args):
            args["num_accepted_tokens"][slots] = accepted.to(torch.int32).cuda()
    torch.cuda.synchronize()
    check_writes(op_args, ref_args, [], op_args["recurrent_state"][[]])


@torch.inference_mode()
def test_cuda_graph_replay_matches_eager() -> None:
    spec = dict(N=1, H=12, num_spec=7, state_dtype="float32", gate="g", out="bf16", use_pdl=True)
    eager = verify_args(spec, seed=5)
    graphed = clone_mutated(eager)
    warm = clone_mutated(eager)
    op(**warm)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = op(**graphed)
    graphed_fresh = clone_mutated(eager)
    for name in MUTATED:  # the capture did not run: restore the inputs the graph reads
        graphed[name].copy_(graphed_fresh[name])
    graph.replay()
    out_eager = op(**eager)
    torch.cuda.synchronize()
    assert torch.equal(out_graph, out_eager)
    for name in MUTATED:
        assert torch.equal(graphed[name], eager[name]), name


def test_guard_refuses_an_accepted_count_past_num_spec() -> None:
    args = verify_args(dict(N=2, H=12, num_spec=4, state_dtype="float32", gate="g", out="bf16"), 0)
    args["num_accepted_tokens"][args["ssm_state_indices"][0].long()] = 5
    with pytest.raises(AssertionError, match="num_accepted_tokens"):
        with validating(op):
            op(**args)


def test_guard_refuses_a_repeated_slot() -> None:
    spec = dict(N=2, H=12, num_spec=4, pool=4, state_dtype="float32", gate="g", out="bf16")
    args = verify_args(spec, 0)
    args["ssm_state_indices"][1] = args["ssm_state_indices"][0]
    with pytest.raises(AssertionError, match="distinct"):
        with validating(op):
            op(**args)
