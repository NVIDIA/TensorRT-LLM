# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode FP8 projection with its weights loaded before the PDL wait (kernels/blackwell/resident_fp8_gemm.py): bitwise
equal to cuBLAS' nvjet (``torch.ops.trtllm.cublas_scaled_mm``) at the decode target-verify shapes, and in conv mode
to nvjet followed by Triton's causal_conv1d_update. Control: an FP32 GEMM of the same operands (another accumulation
order) must differ from nvjet on the test data, so equality is not a property of the data alone."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._utils import is_blackwell_datacenter

pytestmark = pytest.mark.skipif(
    not is_blackwell_datacenter(),
    reason="the kernel runs on SM 10.0 / 10.3 (bitwise against nvjet there)",
)

N, K = 2560, 4096


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _scales():
    return torch.tensor([4.0 / 448.0], device="cuda"), torch.tensor([0.08 / 448.0], device="cuda")


def _weight(seed, sb, scale):
    g = _gen(seed)
    w = torch.randn(N, K, generator=g, device="cuda") * scale
    w[:, torch.randint(0, K, (64,), generator=g, device="cuda")] *= 20.0  # outlier columns
    return (w / sb).clamp(-448, 448).to(torch.float8_e4m3fn)


def _act(m, seed, sa):
    g = _gen(seed)
    x = torch.randn(m, K, generator=g, device="cuda")
    x = (
        x * x * x * torch.exp(torch.randn(m, 1, generator=g, device="cuda"))
    )  # heavy tails, per-row scales
    x[:, torch.randint(0, K, (8,), generator=g, device="cuda")] *= 30.0  # hot columns
    return (x / sa).clamp(-448, 448).to(torch.float8_e4m3fn)


def _bits(y):
    return y.view(torch.int16)


def _fp32_gemm(x, w, sa, sb):
    """The GEMM in FP32 (another K order than nvjet's), rounded like the kernel's epilogue."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return ((x.float() @ w.float().t()) * (sa * sb)).to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_plan():
    from tensorrt_llm._torch.kernels.blackwell import resident_fp8_gemm as R

    assert R.plan(56, K, N) == (
        2,
        28,
        32,
        15,
    )  # 15 resident slots, 17 streamed tiles: the slot ring wraps
    assert R.plan(77, K, N) == (3, 26, 32, 15)
    assert R.plan(96, K, N) == (3, 32, 32, 15)
    assert R.plan(112, K, N) == (3, 38, 48, 11)
    assert R.plan(128, K, N) == (3, 43, 48, 11)
    assert R.plan(129, K, N) is None
    # K in whole x boxes of 8 K tiles, N in whole 128-feature pairs
    assert R.plan(56, 2048, N) == (2, 28, 32, 16) and R.plan(56, 1536, N) is None
    assert R.plan(56, K, 2500) is None
    assert R.plan(56, K, N, 216 * 1024) == (2, 28, 32, 18)
    # small budgets shrink the slot ring down to 1 slot; below that the plan falls back
    assert R.plan(56, K, N, 100 * 1024) == (2, 28, 32, 4) and R.plan(56, K, N, 90 * 1024) == (
        2,
        28,
        32,
        3,
    )
    assert (
        R.plan(56, K, N, 65536 + 2048 + 8192) == (2, 28, 32, 1)
        and R.plan(56, K, N, 65536 + 2048 + 4096) is None
    )
    assert R.plan_conv(56, 7, K, N) == (2, 28, 32, 15, 14)
    assert R.plan_conv(77, 7, K, N) == (3, 28, 32, 15, 14)
    assert R.plan_conv(13 * 7, 7, K, N) == (
        3,
        42,
        48,
        11,
        21,
    )  # 13-18 requests: 3 per CTA half at TN 48
    assert R.plan_conv(19 * 7, 7, K, N) is None
    # x + weight slots + barriers within the budget, and the budget within the 196 KB split beside three 512 B CTAs
    # (200,704 B - 3 x (512 B + 1 KB reserved) - this CTA's 1 KB reserved = 195,072 B)
    for m in (56, 112):
        _, _, tn, r = R.plan(m, K, N)
        assert r * R.W_TILE + (K // R.K_BOX) * (tn // 2) * R.K_BOX + 2048 <= R.SMEM_BUDGET <= 195072


def test_bitwise_smallest_ring():
    """1 resident slot (the minimum): all 31 other K tiles stream through it, one after the other."""
    from tensorrt_llm._torch.kernels.blackwell import resident_fp8_gemm as R

    sa, sb = _scales()
    bud = 65536 + 2048 + 8192
    assert R.plan(56, K, N, bud)[3] == 1
    for c in range(3):
        w = _weight(600 + c, sb, 0.02)
        x = _act(56, 700 + c, sa)
        ref = torch.ops.trtllm.cublas_scaled_mm(x, w.t(), sa, sb, None, torch.bfloat16)
        y = R.resident_fp8_gemm(x, w, sa, sb, budget=bud)
        torch.cuda.synchronize()
        assert torch.equal(_bits(y), _bits(ref)), c
        assert not torch.equal(_bits(_fp32_gemm(x, w, sa, sb)), _bits(ref)), c


@pytest.mark.parametrize("budget", [None, 216 * 1024], ids=["default", "b216"])
@pytest.mark.parametrize("m", [33, 56, 77, 96, 112, 128])
def test_bitwise_vs_nvjet(m, budget):
    from tensorrt_llm._torch.kernels.blackwell import resident_fp8_gemm as R

    bud = R.SMEM_BUDGET if budget is None else budget
    sa, sb = _scales()
    calls = 6
    diff = other_order = 0
    for c in range(calls):
        w = _weight(100 + c, sb, 0.02 if c % 2 == 0 else 0.05)
        x = _act(m, 200 + c, sa)
        ref = torch.ops.trtllm.cublas_scaled_mm(x, w.t(), sa, sb, None, torch.bfloat16)
        y = R.resident_fp8_gemm(x, w, sa, sb, budget=bud)
        torch.cuda.synchronize()
        diff += int(not torch.equal(_bits(y), _bits(ref)))
        other_order += int(not torch.equal(_bits(_fp32_gemm(x, w, sa, sb)), _bits(ref)))
    assert diff == 0, f"{diff} of {calls} calls differ from nvjet"
    # Without this the check could not tell a changed accumulation order from nvjet's.
    assert other_order == calls, f"the FP32 GEMM differs in only {other_order} of {calls}"


@pytest.mark.parametrize(
    "case",
    [
        (8, "dim_last"),
        (8, "dim_first"),
        (11, "dim_last"),
        (11, "dim_first"),
        (16, "dim_last"),
        (16, "dim_first"),
    ],
)
def test_conv_bitwise_vs_nvjet_and_triton_conv(case):
    """Conv mode (GDN MTP target-verify, 7 tokens per request): the convolved q|k|v columns, the raw z columns, the
    conv state of every slot and the per-token windows equal cuBLAS followed by Triton's causal_conv1d_update with
    the model's arguments (gdn_mixer._fused_verify), with a padded request in odd calls."""
    from tensorrt_llm._torch.kernels.blackwell import resident_fp8_gemm as R
    from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update

    nreq, layout = case
    qt, c_dim, slots_n = 7, 1536, 32
    m = nreq * qt
    sa, sb = _scales()
    calls = 4
    diff = {"qkv": 0, "z": 0, "state": 0, "windows": 0}
    other_order = 0
    for c in range(calls):
        gen = _gen(300 + c)
        w = _weight(400 + c, sb, 0.02)
        x = _act(m, 500 + c, sa)
        cw = (torch.randn(c_dim, 4, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
        if layout == "dim_last":
            st0 = torch.randn(slots_n, c_dim, 3, generator=gen, device="cuda").to(torch.bfloat16)
        else:
            st0 = (
                torch.randn(slots_n, 3, c_dim, generator=gen, device="cuda")
                .to(torch.bfloat16)
                .permute(0, 2, 1)
            )
        slots = torch.randperm(slots_n, generator=gen, device="cuda")[:nreq].to(torch.int32)
        if c % 2:
            slots[nreq // 2] = -1  # padding: no outputs compared, no state / window writes
        win0 = torch.zeros(16, qt, c_dim, 3, dtype=torch.bfloat16, device="cuda")
        keep = slots >= 0
        rows = keep.repeat_interleave(qt)
        raw = torch.ops.trtllm.cublas_scaled_mm(x, w.t(), sa, sb, None, torch.bfloat16)
        st_r, win_r = st0.clone(), win0.clone()
        ref = causal_conv1d_update(
            raw[:, :c_dim].reshape(nreq, qt, c_dim).transpose(1, 2),
            st_r,
            cw,
            None,
            "silu",
            conv_state_indices=slots,
            intermediate_conv_window=win_r,
            intermediate_state_indices=torch.arange(nreq, dtype=torch.int32, device="cuda"),
            hoist_x=True,
        )
        ref = ref.transpose(1, 2).reshape(m, c_dim)
        st, win = st0.clone(), win0.clone()
        y = R.resident_fp8_gemm(
            x,
            w,
            sa,
            sb,
            conv=R.ConvUpdate(weight=cw, state=st, slots=slots, windows=win, tokens=qt),
        )
        ctl = causal_conv1d_update(
            _fp32_gemm(x, w, sa, sb)[:, :c_dim].reshape(nreq, qt, c_dim).transpose(1, 2),
            st0.clone(),
            cw,
            None,
            "silu",
            conv_state_indices=slots,
            intermediate_conv_window=win0.clone(),
            intermediate_state_indices=torch.arange(nreq, dtype=torch.int32, device="cuda"),
            hoist_x=True,
        )
        ctl = ctl.transpose(1, 2).reshape(m, c_dim)
        torch.cuda.synchronize()
        diff["qkv"] += int(
            not torch.equal(_bits(y[rows, :c_dim].contiguous()), _bits(ref[rows].contiguous()))
        )
        diff["z"] += int(
            not torch.equal(_bits(y[:, c_dim:].contiguous()), _bits(raw[:, c_dim:].contiguous()))
        )
        diff["state"] += int(not torch.equal(_bits(st.contiguous()), _bits(st_r.contiguous())))
        diff["windows"] += int(
            not torch.equal(
                _bits(win[:nreq][keep].contiguous()), _bits(win_r[:nreq][keep].contiguous())
            )
        )
        other_order += int(
            not torch.equal(_bits(ctl[rows].contiguous()), _bits(ref[rows].contiguous()))
        )
    assert all(v == 0 for v in diff.values()), diff
    # A K-order change can leave every convolved output of a call unchanged after the BF16 roundings, so the
    # control must fire in most calls, not all.
    assert other_order >= calls - 1, other_order


def test_supports_only_operands_the_kernel_reads():
    """The FP8 linear's gate (``supports``): a planned shape of contiguous E4M3 operands at the compiled 128-byte
    alignment; a misaligned or strided view or a BF16 input falls back to cuBLAS instead of asserting."""
    from tensorrt_llm._torch.kernels.blackwell.resident_fp8_gemm import supports

    buf = torch.zeros((17 * K,), dtype=torch.float8_e4m3fn, device="cuda")
    w = torch.zeros((N, K), dtype=torch.float8_e4m3fn, device="cuda")
    x = buf[: 16 * K].view(16, K)
    assert supports(x, w)
    assert supports(buf[K : 17 * K].view(16, K), w)  # a whole-row offset keeps the alignment
    assert not supports(buf[16 : 16 + 16 * K].view(16, K), w)  # 16-byte aligned only
    assert not supports(x.t().contiguous().t(), w)
    assert not supports(x.to(torch.bfloat16), w)


def test_supports_conv_rejects_strided_slots():
    """Conv mode reads the slots as one contiguous int32 row (``torch.as_strided(slots, (1,), (1,))``): a strided
    slot view falls back to the unfused path instead of reading the wrong slots."""
    from tensorrt_llm._torch.kernels.blackwell import resident_fp8_gemm as R

    qt, c_dim, nreq = 7, 1536, 8
    x = torch.zeros((nreq * qt, K), dtype=torch.float8_e4m3fn, device="cuda")
    w = torch.zeros((N, K), dtype=torch.float8_e4m3fn, device="cuda")
    cw = torch.zeros((c_dim, 4), dtype=torch.bfloat16, device="cuda")
    st = torch.zeros((32, c_dim, 3), dtype=torch.bfloat16, device="cuda")
    win = torch.zeros((16, qt, c_dim, 3), dtype=torch.bfloat16, device="cuda")
    slots = torch.arange(2 * nreq, dtype=torch.int32, device="cuda")

    def conv(s: torch.Tensor) -> "R.ConvUpdate":
        return R.ConvUpdate(weight=cw, state=st, slots=s, windows=win, tokens=qt)

    assert R.supports_conv(x, w, conv(slots[:nreq]))
    assert not R.supports_conv(x, w, conv(slots[::2]))
