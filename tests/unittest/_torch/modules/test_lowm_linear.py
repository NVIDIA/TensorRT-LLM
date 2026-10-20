# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M warp-MMA GEMM for Qwen3.5 decode (modules/lowm_linear.py, cute_dsl_kernels/blackwell/lowm_mma_gemm.py)."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import lowm_linear as L

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _fp8_weight(n, k, seed):
    w = (torch.randn(n, k, device="cuda", generator=_gen(seed)) * 48).clamp(-448, 448)
    return w.to(torch.float8_e4m3fn), torch.tensor(1.7e-3, device="cuda")


def _bf16_weight(n, k, seed):
    return (torch.randn(n, k, device="cuda", generator=_gen(seed)) * 0.02).to(torch.bfloat16)


def _act(m, k, seed, outliers=True):
    x = torch.randn(m, k, device="cuda", generator=_gen(seed))
    if outliers:
        x[:, ::97] *= 12.0  # saturate at the static scale below
        x[:, 5::131] *= 1e-4  # E4M3 subnormals / zero
    return x.to(torch.bfloat16)


def _scale(x):
    return (x.float().abs().amax() / 448.0 * 0.5).reshape(())


def _squant(x, s):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x.contiguous(), s.reshape(1))[0]


def _assert_close(y, ref, tol=1.6e-2):
    err = (y.float() - ref.float()).abs().max().item()
    assert err <= tol * ref.float().abs().max().item(), err


@pytest.mark.parametrize("m", [1, 7, 9, 14, 28])
@pytest.mark.parametrize("n,k,tactic", [(2560, 4096, (4, 1)), (2560, 4096, (16, 4)),
                                        (4096, 1024, (8, 2)), (2560, 4096, (8, 2, 1, 1))])
def test_fp8_and_in_register_quant(m, n, k, tactic):
    """FP8 activations match the dequantized reference; BF16 activations quantized in registers give the
    same FP8 values as the static quantize kernel (bitwise equal outputs); a different scale changes them."""
    w, s_w = _fp8_weight(n, k, 1)
    x = _act(m, k, 2)
    s_in = _scale(x)
    q = _squant(x, s_in)
    y8 = L.lowm_gemm(x8=q, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="fp8", tactic=tactic)
    _assert_close(y8, (q.float() @ w.float().t()) * (s_in * s_w))
    yq = L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="quant", tactic=tactic)
    assert torch.equal(y8, yq)
    yb = L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in * 1.01, fp8_act="quant", tactic=tactic)
    assert not torch.equal(yb, yq)


@pytest.mark.parametrize("m", [1, 7, 16])
def test_bf16_segments_and_mixed_launch(m):
    """BF16 weights (also strided activation rows), and an FP8 + BF16 launch equal to two launches."""
    x = _act(m, 4096, 3, outliers=False)
    wb = _bf16_weight(512, 4096, 4)
    _assert_close(L.lowm_gemm(x16=x, w_bf16=wb), x.float() @ wb.float().t())
    wq = _bf16_weight(2560, 4096, 9)  # MTP qkv: swapped (16, 2) at M <= 8, (8, 2, min_blocks 2) up to 16
    _assert_close(L.lowm_gemm(x16=x, w_bf16=wq), x.float() @ wq.float().t())
    big = _act(m, 8 * 1024, 5, outliers=False)
    xs = big[:, 3 * 1024:4 * 1024]
    wf = _bf16_weight(4096, 1024, 6)
    _assert_close(L.lowm_gemm(x16=xs, w_bf16=wf), xs.float() @ wf.float().t())
    w, s_w = _fp8_weight(2560, 4096, 7)
    wba = _bf16_weight(16, 4096, 8)
    s_in = _scale(x)
    a, b = L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, w_bf16=wba, split_out=2560,
                       tactic=(16, 4), tactic_bf16=(16, 1))
    assert torch.equal(a, L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, tactic=(16, 4)))
    assert torch.equal(b, L.lowm_gemm(x16=x, w_bf16=wba, tactic_bf16=(16, 1)))


@pytest.mark.parametrize("m", [1, 7, 28])
def test_shared_expert_launch(m):
    """gate_up + SwiGLU -> E4M3 (down's input scale) + the BF16 gate as an n_valid segment, then down,
    against the stock kernels (quantize, cuBLAS FP8 GEMM, silu_and_mul, cuBLAS FP8 GEMM, F.linear)."""
    inter, k = 128, 4096
    w_gu, s_gu = _fp8_weight(2 * inter, k, 9)
    w_dn, _ = _fp8_weight(k, inter, 10)
    s_dn_w = torch.tensor(2.3e-3, device="cuda")
    s_dn_in = torch.tensor(0.05, device="cuda")
    x = _act(m, k, 11)
    s_in = _scale(x)
    gate_w = torch.zeros(8, k, dtype=torch.bfloat16, device="cuda")
    gate_w[:1] = _bf16_weight(1, k, 12)
    gu = torch.ops.trtllm.cublas_scaled_mm(_squant(x, s_in), w_gu.t(), scale_a=s_in.reshape(1),
                                           scale_b=s_gu.reshape(1), bias=None, out_dtype=torch.bfloat16)
    ref_h8 = torch.ops.trtllm.silu_and_mul(gu, scale=s_dn_in.reshape(1), dtype=torch.float8_e4m3fn)
    h8, gate = L.lowm_gemm(x16=x, w_fp8=L.swiglu_pairs(w_gu), weight_scale=s_gu, input_scale=s_in, swiglu_fp8=True,
                           w_bf16=gate_w, split_out=inter, n_valid=inter + 1, out_scale=s_dn_in)
    assert h8.dtype == torch.float8_e4m3fn and gate.shape == (m, 1) and gate.is_contiguous()
    codes = (h8.view(torch.uint8).int() - ref_h8.view(torch.uint8).int()).abs()
    assert (codes == 0).float().mean().item() > 0.97 and codes.max().item() <= 1
    _assert_close(gate, x.float() @ gate_w[:1].float().t())
    out = L.lowm_gemm(x8=h8, w_fp8=w_dn, weight_scale=s_dn_w, input_scale=s_dn_in, fp8_act="fp8")
    _assert_close(out, (h8.float() @ w_dn.float().t()) * (s_dn_in * s_dn_w))


@pytest.mark.parametrize("m", [1, 7, 14, 28])
def test_tp16_shapes_with_table_tactics(m):
    """Per-rank shapes at attention TP16 with the measured tactics: FP8 (E4M3 input and in-register quant), the
    fused GDN in_proj launch (1280 FP8 + 8 BF16 rows) as the model issues it, and BF16 projections."""
    k = 4096
    for n, kk in ((1280, 4096), (1536, 4096), (4096, 512)):
        x = _act(m, kk, 40)
        s_in = _scale(x)
        q = _squant(x, s_in)
        w, s_w = _fp8_weight(n, kk, 41)
        ref = (q.float() @ w.float().t()) * (s_in * s_w)
        assert L.lowm_rows_ok(m, n, kk, "fp8") == (m <= {1280: 32, 1536: 32, 4096: 16}[n])
        assert L.lowm_rows_ok(m, n, kk, "quant") == (m <= 32)
        _assert_close(L.lowm_gemm(x8=q, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="fp8"), ref)
        _assert_close(L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="quant"), ref)
    x = _act(m, k, 42)
    s_in = _scale(x)
    q = _squant(x, s_in)
    w, s_w = _fp8_weight(1280, k, 43)
    wba = _bf16_weight(8, k, 44)
    for act, mode in (("quant", "mixed"), ("fp8", "mixed8")):
        if not L.lowm_rows_ok(m, 1280, k, mode):
            continue
        tac = L.default_tactic(m, 1280, k, mode=mode)
        a, b = L.lowm_gemm(x16=x, x8=q if act == "fp8" else None, w_fp8=w, weight_scale=s_w, input_scale=s_in,
                           fp8_act=act, w_bf16=wba, split_out=1280, tactic=tac,
                           tactic_bf16=(tac[0], 1, tac[2] if len(tac) > 2 else 1))
        _assert_close(a, (q.float() @ w.float().t()) * (s_in * s_w))
        _assert_close(b, x.float() @ wba.float().t())
    for n, kk in ((8, 4096), (1536, 4096), (4096, 512)):
        xb = _act(m, kk, 45, outliers=False)
        wb = _bf16_weight(n, kk, 46)
        assert L.lowm_rows_ok(m, n, kk, "bf16") == (m <= 32)
        _assert_close(L.lowm_gemm(x16=xb, w_bf16=wb), xb.float() @ wb.float().t())


@pytest.mark.parametrize("inter", [128, 64])
@pytest.mark.parametrize("m", [7, 14, 28])
def test_shared_expert_module_path(monkeypatch, m, inter):
    """The MoE block's shared expert, built as in the model (a GatedMLP whose Linears carry LoRA layers),
    takes the two-launch path with TRTLLM_Q35_LOWM_SE=1 and matches the stock GatedMLP and gate; a row
    limit (TRTLLM_Q35_LOWM_CAPS) sends it back to the stock path."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextSparseMoeBlock
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    k = 4096  # inter 128 / 64: the per-rank shared expert at TP8 / TP16
    mlp = GatedMLP(hidden_size=k, intermediate_size=inter, bias=False, dtype=torch.bfloat16,
                   config=ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)),
                   reduce_output=False, is_shared_expert=True).cuda()
    assert mlp.down_proj.lora is not None
    gate = Linear(k, 1, bias=False, dtype=torch.bfloat16, quant_config=None).cuda()
    x = _act(m, k, 30)
    s_in = _scale(x)
    s_w = torch.tensor(1.7e-3, device="cuda").reshape(1)
    mlp.gate_up_proj.load_weights([{"weight": _fp8_weight(inter, k, seed)[0], "weight_scale": s_w,
                                    "input_scale": s_in.reshape(1)} for seed in (31, 32)])
    gu = mlp.gate_up_proj(x).float()
    s_dn_in = _scale(F.silu(gu[:, :inter]) * gu[:, inter:])
    mlp.down_proj.load_weights([{"weight": _fp8_weight(k, inter, 33)[0], "weight_scale": s_w,
                                 "input_scale": s_dn_in.reshape(1)}])
    gate.load_weights([{"weight": _bf16_weight(1, k, 34)}])
    blk = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(blk)
    blk.shared_expert, blk.shared_expert_gate, blk._se_lowm = mlp, gate, None
    ref_out, ref_gate = mlp(x), gate(x)
    monkeypatch.delenv("TRTLLM_Q35_LOWM_SE", raising=False)
    monkeypatch.delenv("TRTLLM_Q35_LOWM_CAPS", raising=False)
    assert blk._shared_expert_lowm(x) is None
    monkeypatch.setenv("TRTLLM_Q35_LOWM_SE", "1")
    got = blk._shared_expert_lowm(x)
    assert got is not None
    out, logits = got
    _assert_close(out, ref_out)
    _assert_close(logits, ref_gate)
    monkeypatch.setenv("TRTLLM_Q35_LOWM_CAPS", f"se:{m - 1}")
    assert blk._shared_expert_lowm(x) is None
    monkeypatch.setenv("TRTLLM_Q35_LOWM_CAPS", f"se:{m}")
    assert torch.equal(blk._shared_expert_lowm(x)[0], out)


@pytest.mark.parametrize("tactic", [(4, 4, 4), (8, 4, 2), (8, 2, 1, 1), (4, 2, 2, 1)])
def test_ksplit_and_swap_deterministic(tactic):
    """Cluster split-K and swapped operands: close to the reference and deterministic."""
    m = 7
    w, s_w = _fp8_weight(2560, 4096, 13)
    x = _act(m, 4096, 14)
    s_in = _scale(x)
    q = _squant(x, s_in)
    y = L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, tactic=tactic)
    _assert_close(y, (q.float() @ w.float().t()) * (s_in * s_w))
    for _ in range(3):
        assert torch.equal(y, L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, tactic=tactic))


@pytest.mark.parametrize("early", ["0", "1"])
def test_cuda_graph_pdl_chain(monkeypatch, early):
    """Dependent launches (early dependent release on/off) captured in one graph equal eager launches."""
    monkeypatch.setenv("TRTLLM_Q35_LOWM_EARLY_TRIGGER", early)
    w1, s1 = _fp8_weight(1024, 4096, 15)
    w2, s2 = _fp8_weight(4096, 1024, 16)
    x = _act(7, 4096, 17)
    si = _scale(x)
    si2 = (si * 0.02).clone()

    def chain():
        h = L.lowm_gemm(x16=x, w_fp8=w1, weight_scale=s1, input_scale=si)
        h2 = L.lowm_gemm(x16=h, w_fp8=w2, weight_scale=s2, input_scale=si2)
        return L.lowm_gemm(x16=h2, w_fp8=w1, weight_scale=s1, input_scale=si)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = chain()
    for i in range(5):
        x.copy_(_act(7, 4096, 100 + i))
        ref = chain()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def test_delayed_producer_pdl():
    """A producer that releases its dependents at its start and writes the activations ~200 us later: the
    consumer (programmatic dependent launch) reads fresh data; without its dependency wait it reads stale
    data (positive control)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _spin_then_copy(dst, src, n, spin_ns, BLOCK: tl.constexpr):
        tl.extra.cuda.gdc_launch_dependents()
        t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
        t = t0
        while t - t0 < spin_ns:
            t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False,
                                          pack=1)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

    w, s_w = _fp8_weight(2560, 4096, 18)
    src = _act(7, 4096, 19)
    s_in = _scale(src)
    x = torch.zeros_like(src)
    expected = L.lowm_gemm(x16=src, w_fp8=w, weight_scale=s_w, input_scale=s_in)
    results = {}
    for skip in (False, True):
        L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, _unsafe_skip_wait=skip)
        fresh = 0
        for _ in range(5):
            x.zero_()
            torch.cuda.synchronize()
            _spin_then_copy[(triton.cdiv(src.numel(), 1024),)](x, src, src.numel(), 200_000, BLOCK=1024)
            y = L.lowm_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, _unsafe_skip_wait=skip)
            torch.cuda.synchronize()
            fresh += int(torch.equal(y, expected))
        results[skip] = fresh
    assert results[False] == 5
    assert results[True] < 5


@pytest.mark.parametrize("tp16,n,pad", [(False, 1, False), (True, 4, False), (False, 2, True)])
def test_gdn_conv_epilogue_bitwise(tp16, n, pad):
    """The GDN conv1d update in the fused GDN in_proj launch (TRTLLM_Q35_GDN_FUSE=4) equals the plain launch
    followed by the Triton causal_conv1d_update, bitwise: conv outputs, raw z / ba columns, intermediate windows
    and conv states. A padded request (slot -1) leaves its windows and every state untouched; a perturbed conv
    weight is detected (negative control)."""
    from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update

    t, k = 7, 4096
    h, hv = (1, 4) if tp16 else (2, 8)  # per-rank key / value heads of Qwen3.5-397B
    cd = 2 * h * 128 + hv * 128
    proj = cd + hv * 128
    m = n * t
    x = _act(m, k, 50)
    s_in = _scale(x)
    q = _squant(x, s_in)
    w, s_w = _fp8_weight(proj, k, 51)
    wba = _bf16_weight(2 * hv, k, 52)
    conv_w = (torch.randn(cd, 4, device="cuda", generator=_gen(53)) * 0.4).to(torch.bfloat16)
    state0 = torch.randn(16, cd, 3, device="cuda", generator=_gen(54)).to(torch.bfloat16)
    slots = torch.tensor([3, 11, 6, 9][:n], dtype=torch.int32, device="cuda")
    if pad:
        slots[-1] = -1
    tac = L.default_tactic(m, proj, k, mode="mixed8")
    assert L.lowm_conv_supported(m, tac, cd, t)
    kw = dict(x16=x, x8=q, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="fp8", w_bf16=wba,
              split_out=proj, tactic=tac, tactic_bf16=(tac[0], 1, tac[2] if len(tac) > 2 else 1))
    raw, ba = L.lowm_gemm(**kw)
    st_ref = state0.clone()
    win_ref = torch.zeros(n, t, cd, 3, dtype=torch.bfloat16, device="cuda")
    ref = causal_conv1d_update(raw[:, :cd].reshape(n, t, cd).transpose(1, 2), st_ref, conv_w, None, "silu",
                               conv_state_indices=slots, intermediate_conv_window=win_ref,
                               intermediate_state_indices=torch.arange(n, dtype=torch.int32, device="cuda"))
    ref = ref.transpose(1, 2).reshape(m, cd)
    valid = (slots >= 0).repeat_interleave(t)

    def epilogue(weight):
        st, win = state0.clone(), torch.zeros_like(win_ref)
        out, out_ba = L.lowm_gemm(**kw, conv=dict(weight=weight, state=st, slots=slots, windows=win, tokens=t))
        return out, out_ba, st, win

    got, got_ba, st, win = epilogue(conv_w)
    assert torch.equal(got[valid, :cd], ref[valid])
    assert torch.equal(got[:, cd:], raw[:, cd:]) and torch.equal(got_ba, ba)
    assert torch.equal(win, win_ref) and torch.equal(st, st_ref)
    if pad:
        used = set(slots[slots >= 0].tolist())
        untouched = [i for i in range(state0.shape[0]) if i not in used]
        assert torch.equal(st[untouched], state0[untouched]) and not win[-1].any()
    bad_w = conv_w.clone()
    bad_w[5, 2] += 0.25
    assert not torch.equal(epilogue(bad_w)[0][valid, :cd], ref[valid])


def test_unmeasured_shapes_keep_stock(monkeypatch):
    """Shapes without measurements (e.g. another TP size before tuning) keep the stock path unless opted in
    with a tactic override or TRTLLM_Q35_LOWM_MAX_M; measured shapes stop at their crossover."""
    monkeypatch.delenv("TRTLLM_Q35_LOWM_MAX_M", raising=False)
    monkeypatch.delenv("TRTLLM_Q35_LOWM_TACTIC", raising=False)
    assert L.lowm_rows_ok(7, 2560, 4096, "quant") and not L.lowm_rows_ok(17, 2560, 4096, "quant")
    assert not L.lowm_rows_ok(7, 2568, 4096, "quant")
    monkeypatch.setenv("TRTLLM_Q35_LOWM_TACTIC", "2568x4096:8x1")
    assert L.lowm_rows_ok(7, 2568, 4096, "quant")
    monkeypatch.delenv("TRTLLM_Q35_LOWM_TACTIC")
    monkeypatch.setenv("TRTLLM_Q35_LOWM_MAX_M", "32")
    assert L.lowm_rows_ok(7, 2568, 4096, "quant") and L.lowm_rows_ok(30, 2560, 4096, "quant")
    assert not L.lowm_rows_ok(33, 2560, 4096, "quant")


def test_linear_hook(monkeypatch):
    """An opted-in FP8 Linear takes the low-M path only with the flag, within the row cap, and not when skipped."""
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    lin = Linear(4096, 2560, bias=False, dtype=torch.bfloat16, quant_config=QuantConfig(quant_algo=QuantAlgo.FP8))
    lin = lin.cuda()
    w, s_w = _fp8_weight(2560, 4096, 20)
    x = _act(7, 4096, 21)
    s_in = _scale(x)
    lin.load_weights([{"weight": w, "weight_scale": s_w.reshape(1), "input_scale": s_in.reshape(1)}])
    ref = lin(x)
    lin._q35_lowm = "test"
    monkeypatch.setenv("TRTLLM_Q35_LOWM_FP8", "1")
    monkeypatch.setenv("TRTLLM_Q35_LOWM_TACTIC", "2560x4096:16x4")
    y = lin(x)
    _assert_close(y, ref)
    assert torch.equal(y, lin(_squant(x, s_in)))
    monkeypatch.setenv("TRTLLM_Q35_LOWM_SKIP", "test")
    assert torch.equal(lin(x), ref)
