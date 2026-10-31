# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M warp-MMA GEMM (modules/low_m_mma_gemm.py, kernels/blackwell/low_m_mma_gemm.py) and its runners in the
low-M GEMM race (modules/low_m_gemm.py)."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules import low_m_mma_gemm as L

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


@pytest.mark.parametrize("m", [1, 9, 28])
@pytest.mark.parametrize(
    "n,k,tactic",
    [
        (2560, 4096, (16, 4)),
        (4096, 1024, (8, 2)),
        (2560, 4096, (8, 2, 1, 1)),  # swapped operands
        (2560, 4096, (4, 4, 4)),  # cluster split-K
        (2560, 4096, (4, 2, 2, 1)),
    ],
)
def test_fp8_and_in_register_quant(m, n, k, tactic):
    """FP8 activations match the dequantized reference, deterministically; in-register quantization is bitwise
    equal to the static quantize kernel (a different scale changes the output)."""
    w, s_w = _fp8_weight(n, k, 1)
    x = _act(m, k, 2)
    s_in = _scale(x)
    q = _squant(x, s_in)
    y8 = L.low_m_mma_gemm(
        x8=q, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="fp8", tactic=tactic
    )
    _assert_close(y8, (q.float() @ w.float().t()) * (s_in * s_w))
    yq = L.low_m_mma_gemm(
        x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="quant", tactic=tactic
    )
    assert torch.equal(y8, yq)
    kw = dict(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="quant", tactic=tactic)
    assert torch.equal(yq, L.low_m_mma_gemm(**kw))
    yb = L.low_m_mma_gemm(
        x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in * 1.01, fp8_act="quant", tactic=tactic
    )
    assert not torch.equal(yb, yq)


@pytest.mark.parametrize("m", [1, 7, 16])
def test_bf16_segments_and_mixed_launch(m):
    """BF16 weights (also strided activation rows), and an FP8 + BF16 launch equal to two launches."""
    x = _act(m, 4096, 3, outliers=False)
    wb = _bf16_weight(512, 4096, 4)
    _assert_close(L.low_m_mma_gemm(x16=x, w_bf16=wb), x.float() @ wb.float().t())
    wq = _bf16_weight(2560, 4096, 9)
    _assert_close(L.low_m_mma_gemm(x16=x, w_bf16=wq), x.float() @ wq.float().t())
    big = _act(m, 8 * 1024, 5, outliers=False)
    xs = big[:, 3 * 1024 : 4 * 1024]
    wf = _bf16_weight(4096, 1024, 6)
    _assert_close(L.low_m_mma_gemm(x16=xs, w_bf16=wf), xs.float() @ wf.float().t())
    w, s_w = _fp8_weight(2560, 4096, 7)
    wba = _bf16_weight(16, 4096, 8)
    s_in = _scale(x)
    a, b = L.low_m_mma_gemm(
        x16=x,
        w_fp8=w,
        weight_scale=s_w,
        input_scale=s_in,
        w_bf16=wba,
        split_out=2560,
        tactic=(16, 4),
        tactic_bf16=(16, 1),
    )
    assert torch.equal(
        a, L.low_m_mma_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, tactic=(16, 4))
    )
    assert torch.equal(b, L.low_m_mma_gemm(x16=x, w_bf16=wba, tactic_bf16=(16, 1)))


@pytest.mark.parametrize("m", [1, 7, 28])
def test_shared_expert_launch(m):
    """Shared expert as two launches (gate_up + SwiGLU -> E4M3 with the BF16 gate as an n_valid segment, then
    down) against the stock quantize / cuBLAS FP8 / silu_and_mul / F.linear kernels."""
    inter, k = 128, 4096
    w_gu, s_gu = _fp8_weight(2 * inter, k, 9)
    w_dn, _ = _fp8_weight(k, inter, 10)
    s_dn_w = torch.tensor(2.3e-3, device="cuda")
    s_dn_in = torch.tensor(0.05, device="cuda")
    x = _act(m, k, 11)
    s_in = _scale(x)
    gate_w = torch.zeros(8, k, dtype=torch.bfloat16, device="cuda")
    gate_w[:1] = _bf16_weight(1, k, 12)
    gu = torch.ops.trtllm.cublas_scaled_mm(
        _squant(x, s_in),
        w_gu.t(),
        scale_a=s_in.reshape(1),
        scale_b=s_gu.reshape(1),
        bias=None,
        out_dtype=torch.bfloat16,
    )
    ref_h8 = torch.ops.trtllm.silu_and_mul(gu, scale=s_dn_in.reshape(1), dtype=torch.float8_e4m3fn)
    h8, gate = L.low_m_mma_gemm(
        x16=x,
        w_fp8=L.swiglu_pairs(w_gu),
        weight_scale=s_gu,
        input_scale=s_in,
        swiglu_fp8=True,
        w_bf16=gate_w,
        split_out=inter,
        n_valid=inter + 1,
        out_scale=s_dn_in,
    )
    assert h8.dtype == torch.float8_e4m3fn and gate.shape == (m, 1) and gate.is_contiguous()
    codes = (h8.view(torch.uint8).int() - ref_h8.view(torch.uint8).int()).abs()
    assert (codes == 0).float().mean().item() > 0.97 and codes.max().item() <= 1
    _assert_close(gate, x.float() @ gate_w[:1].float().t())
    out = L.low_m_mma_gemm(
        x8=h8, w_fp8=w_dn, weight_scale=s_dn_w, input_scale=s_dn_in, fp8_act="fp8"
    )
    _assert_close(out, (h8.float() @ w_dn.float().t()) * (s_dn_in * s_dn_w))


@pytest.mark.parametrize("m", [1, 28])
@pytest.mark.parametrize("n,k", [(2560, 4096), (4096, 512), (256, 4096)])
def test_candidate_tactics(m, n, k):
    """Every tactic the race times runs (FP8 activations) and matches the dequantized reference."""
    w, s_w = _fp8_weight(n, k, 41)
    x = _act(m, k, 40)
    s_in = _scale(x)
    q = _squant(x, s_in)
    ref = (q.float() @ w.float().t()) * (s_in * s_w)
    tactics = L.candidate_tactics(m, n, k, fp8=True)
    assert tactics
    for tactic in tactics:
        _assert_close(
            L.low_m_mma_gemm(
                x8=q, w_fp8=w, weight_scale=s_w, input_scale=s_in, fp8_act="fp8", tactic=tactic
            ),
            ref,
        )


@pytest.mark.parametrize("inter", [128, 64])
def test_shared_expert_module_path(inter, monkeypatch, m=7):
    """The MoE block's shared expert (GatedMLP with LoRA layers, as in the model) takes the two-launch low-M path
    where the race picked the warp-MMA GEMM for its projections, matching the stock GatedMLP and gate; otherwise
    it declines."""
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextSparseMoeBlock
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.modules.low_m_gemm import LowMGemmDispatcher, prepare_low_m_gemm
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    k = 4096  # inter 128 / 64: TP8 / TP16
    mlp = GatedMLP(
        hidden_size=k,
        intermediate_size=inter,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(quant_config=QuantConfig(quant_algo=QuantAlgo.FP8)),
        reduce_output=False,
        is_shared_expert=True,
    ).cuda()
    gate = Linear(k, 1, bias=False, dtype=torch.bfloat16, quant_config=None).cuda()
    x = _act(m, k, 30)
    s_in = _scale(x)
    s_w = torch.tensor(1.7e-3, device="cuda").reshape(1)
    mlp.gate_up_proj.load_weights(
        [
            {
                "weight": _fp8_weight(inter, k, seed)[0],
                "weight_scale": s_w,
                "input_scale": s_in.reshape(1),
            }
            for seed in (31, 32)
        ]
    )
    gu = mlp.gate_up_proj(x).float()
    s_dn_in = _scale(F.silu(gu[:, :inter]) * gu[:, inter:])
    mlp.down_proj.load_weights(
        [
            {
                "weight": _fp8_weight(k, inter, 33)[0],
                "weight_scale": s_w,
                "input_scale": s_dn_in.reshape(1),
            }
        ]
    )
    gate.load_weights([{"weight": _bf16_weight(1, k, 34)}])
    blk = Qwen3NextSparseMoeBlock.__new__(Qwen3NextSparseMoeBlock)
    torch.nn.Module.__init__(blk)
    blk.shared_expert, blk.shared_expert_gate, blk._shared_expert_low_m = mlp, gate, None
    ref_out, ref_gate = mlp(x), gate(x)
    prepare_low_m_gemm(blk, force=True)
    blk._shared_expert_low_m = blk._shared_expert_low_m_weights()
    assert blk._shared_expert_low_m_forward(x) is None  # the race has not picked the warp-MMA GEMM

    # Each lookup names the race the plain projection ran: a BF16 or an E4M3 input.
    plain_fp8_input = {}
    for proj in (mlp.gate_up_proj, mlp.down_proj):
        proj.register_forward_pre_hook(
            lambda mod, args: plain_fp8_input.__setitem__(
                mod.weight.data_ptr(), args[0].dtype == torch.float8_e4m3fn
            )
        )
    mlp(x)
    looked_up = {}

    def warp_mma_tactic(self, rows, weight, *args, quantize=True, **kwargs):
        looked_up[weight.data_ptr()] = not quantize
        return L.default_tactic(rows, weight.shape[0], weight.shape[1])

    monkeypatch.setattr(LowMGemmDispatcher, "warp_mma_tactic", warp_mma_tactic)
    out, logits = blk._shared_expert_low_m_forward(x)
    _assert_close(out, ref_out)
    _assert_close(logits, ref_gate)
    assert looked_up == plain_fp8_input


@pytest.mark.parametrize("tp16,n,pad", [(False, 1, False), (True, 4, False), (False, 2, True)])
def test_gdn_conv_epilogue_bitwise(tp16, n, pad):
    """The conv epilogue equals the plain launch + Triton causal_conv1d_update bitwise (conv outputs, raw z / ba
    columns, windows, states). A padded request (slot -1) touches no window or state; a perturbed conv weight is
    detected."""
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
    tac = L.default_tactic(m, proj, k)
    assert L.conv_supported(m, tac, cd, t)
    kw = dict(
        x16=x,
        x8=q,
        w_fp8=w,
        weight_scale=s_w,
        input_scale=s_in,
        fp8_act="fp8",
        w_bf16=wba,
        split_out=proj,
        tactic=tac,
        tactic_bf16=(tac[0], 1, tac[2] if len(tac) > 2 else 1),
    )
    raw, ba = L.low_m_mma_gemm(**kw)
    st_ref = state0.clone()
    win_ref = torch.zeros(n, t, cd, 3, dtype=torch.bfloat16, device="cuda")
    ref = causal_conv1d_update(
        raw[:, :cd].reshape(n, t, cd).transpose(1, 2),
        st_ref,
        conv_w,
        None,
        "silu",
        conv_state_indices=slots,
        intermediate_conv_window=win_ref,
        intermediate_state_indices=torch.arange(n, dtype=torch.int32, device="cuda"),
    )
    ref = ref.transpose(1, 2).reshape(m, cd)
    valid = (slots >= 0).repeat_interleave(t)

    def epilogue(weight):
        st, win = state0.clone(), torch.zeros_like(win_ref)
        out, out_ba = L.low_m_mma_gemm(
            **kw, conv=dict(weight=weight, state=st, slots=slots, windows=win, tokens=t)
        )
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


def test_fp8_race():
    """An FP8 Linear with the low-M dispatcher attached races cuBLAS against the warp-MMA GEMM: after tuning
    the output is the winner's, and the fused users' tactic lookup names the warp-MMA tactic exactly when it
    won (never while tuning)."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm._torch.modules.low_m_gemm import low_m_gemm_dispatcher, prepare_low_m_gemm
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    lin = Linear(
        4096,
        2560,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
    ).cuda()
    w, s_w = _fp8_weight(2560, 4096, 20)
    x = _act(7, 4096, 21)
    s_in = _scale(x)
    lin.load_weights(
        [{"weight": w, "weight_scale": s_w.reshape(1), "input_scale": s_in.reshape(1)}]
    )
    ref = lin(x)
    prepare_low_m_gemm(lin, force=True)
    dispatcher = low_m_gemm_dispatcher(lin)
    AutoTuner.get().clear_cache()
    with autotune():
        assert dispatcher.warp_mma_tactic(7, w, s_w, s_in) is None
        lin(x)
    y = lin(x)
    _assert_close(y, ref)
    tactic = dispatcher.warp_mma_tactic(7, w, s_w, s_in)
    if tactic is None:
        assert torch.equal(y, ref)
    else:
        assert torch.equal(
            y, L.low_m_mma_gemm(x16=x, w_fp8=w, weight_scale=s_w, input_scale=s_in, tactic=tactic)
        )


def test_router_gate_under_compile_matches_eager():
    """The BF16 router gate with the low-M dispatcher gives the eager logits inside a torch.compile'd graph."""
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate
    from tensorrt_llm._torch.modules.low_m_gemm import prepare_low_m_gemm

    gate = Qwen3NextGate(hidden_size=4096, num_experts=512, top_k=10, dtype=torch.bfloat16).cuda()
    prepare_low_m_gemm(gate, force=True)
    gate.weight.data.normal_(0, 0.02)
    x = _act(8, 4096, 60, outliers=False)
    ref = (x.float() @ gate.weight.float().t()).to(torch.bfloat16)
    eager = gate(x)
    torch._dynamo.reset()
    compiled = torch.compile(gate.forward, backend="aot_eager", fullgraph=True)(x)
    torch.testing.assert_close(eager, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(compiled, ref, rtol=2e-2, atol=2e-2)


def test_dispatcher_declines_inside_torch_compile():
    """A forced dispatcher whose runners do not exist yet (first use inside a compiled forward) declines
    while torch.compile traces, so the graph takes the plain projection."""
    from tensorrt_llm._torch.modules.low_m_gemm import LowMGemmDispatcher

    dispatcher = LowMGemmDispatcher()
    dispatcher._force = True
    w = torch.ones(64, 128, dtype=torch.bfloat16, device="cuda")

    def projection(x):
        tactic = dispatcher.warp_mma_tactic(x.shape[0], w)
        out = dispatcher.apply(None, x, w, None, force_active=True)
        return x @ w.t() if tactic is None and out is None else x

    x = torch.ones(4, 128, dtype=torch.bfloat16, device="cuda")
    torch._dynamo.reset()
    out = torch.compile(projection, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(out, x @ w.t())
    assert not dispatcher._prepared
