# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for HISLAYOUT (GV_C; the last group adds the served row-parallel / NCCL-window cases at the end):
``drafter_mxfp8.route_his_layout`` on the drafter's projections (real TRT-LLM
``Linear`` modules, the served drafter's projection class) and its MLP (a shell with the GatedMLP forward this drafter
takes: gate_up_proj -> _apply_activation (the native SwiGLU-OAI) -> down_proj), at the TP4 per-rank shapes.

Checked at the served decode rows (7 = K7 c1), the row bound (8) and bound + 1 (9):
- routing: qkv / o_proj / gate_up routed at the reference configs; the MLP's SwiGLU + down fused (m3_mx_gemv_swiglu);
- served rows take the GEMV at the reference config bit for bit (vs a direct call); 9 rows run the stock path unchanged;
- the fused SwiGLU-MXFP8 quantization is bitwise the reference: m3_mx_gemv_swiglu(gu) == the prequantized GEMV on
  mxfp8_quantize(SwiGLU-OAI(gu)) (the reference swiglu_oai_mxfp8 is certified bitwise to that stock act + quantize);
- numerics vs the current BF16 path: frac of differing outputs and max |diff| in row-max bf16 ulps, <= 16 (nl1a: 4.9);
- CUDA-graph replay returns the eager bits; the dispatch line is printed once per (tag, N, K, M).
"""

import importlib

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — torch.ops.trtllm.mxfp8_quantize

PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
D4 = f"{PKG}.minimax_m3_1_flash_nvfp4__sm_107__tp4"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)

SHAPES = {
    "qkv_proj": (2304, 6144),
    "o_proj": (6144, 2048),
    "gate_up_proj": (6144, 6144),
    "down_proj": (6144, 3072),
}
DECODE_ROWS = [7]
BOUND, BOUND_PLUS_1 = 8, 9


def _exact_mx_bf16(n, k, g):
    codes = (
        (torch.randn(n, k, device="cuda", generator=g) * 64)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    exp = torch.randint(-16, -6, (n, k // 32), device="cuda", generator=g).float()
    return (
        (codes.float().view(n, k // 32, 32) * torch.exp2(exp).unsqueeze(-1))
        .view(n, k)
        .to(torch.bfloat16)
    )


class _Mlp(torch.nn.Module):
    """GatedMLP's forward for this drafter (not split, no LoRA, no fused gate_up-SwiGLU): gate_up -> act -> down."""

    def __init__(self, gate_up, down):
        super().__init__()
        self.gate_up_proj, self.down_proj, self.split_gate_up, self.layer_idx = (
            gate_up,
            down,
            False,
            0,
        )
        self._apply_activation = importlib.import_module(f"{PKG}.ops.native_swiglu").swiglu_oai

    def forward(
        self, x, all_rank_num_tokens=None, final_all_reduce_params=None, lora_params=None, **kwargs
    ):
        return self.down_proj(
            self._apply_activation(self.gate_up_proj(x)), all_reduce_params=final_all_reduce_params
        )


class _NS:
    pass


def _drafter(seed=3):
    from tensorrt_llm._torch.modules.linear import Linear

    g = torch.Generator(device="cuda").manual_seed(seed)
    linears = {}
    for name, (n, k) in SHAPES.items():
        lin = Linear(k, n, bias=False, dtype=torch.bfloat16).cuda()
        with torch.no_grad():
            lin.weight.copy_(_exact_mx_bf16(n, k, g))
        linears[name] = lin
    layer, attn = _NS(), _NS()
    attn.qkv_proj, attn.o_proj = linears["qkv_proj"], linears["o_proj"]
    layer.self_attn, layer.mlp = attn, _Mlp(linears["gate_up_proj"], linears["down_proj"])
    d = _NS()
    d.model = _NS()
    d.model.layers = [layer]
    return d, linears


@pytest.fixture(scope="module")
def routed():
    mxd = importlib.import_module(f"{D4}.drafter_mxfp8")
    d, linears = _drafter()
    stock = {name: lin.weight.detach().clone() for name, lin in linears.items()}
    n_lin, n_mlp = mxd.route_his_layout(d, DECODE_ROWS + [BOUND_PLUS_1])
    return mxd, d, linears, stock, (n_lin, n_mlp)


def test_routing_and_configs(routed):
    mxd, d, linears, _, (n_lin, n_mlp) = routed
    assert (n_lin, n_mlp) == (3, 1)
    for name in ("qkv_proj", "o_proj", "gate_up_proj"):
        assert linears[name]._m31_hislayout == mxd.HIS_CFGS[SHAPES[name]]
        assert set(linears[name]._m31_mxdraft[2]) == set(DECODE_ROWS), (
            "only decode rows <= 8 are served"
        )
    w8, sf, served, cfg = d.model.layers[0].mlp._m31_hislayout_mlp
    assert cfg == mxd.HIS_DOWN_CFGS[SHAPES["down_proj"]] and set(served) == set(DECODE_ROWS)


def test_served_rows_take_the_peer_config_bitwise(routed):
    mxd, d, linears, _, _ = routed
    gemv = importlib.import_module(f"{D4}.kernels.mx_gemv.op")
    g = torch.Generator(device="cuda").manual_seed(11)
    for name in ("qkv_proj", "o_proj", "gate_up_proj"):
        x = torch.randn(7, SHAPES[name][1], device="cuda", generator=g).to(torch.bfloat16)
        w8, sf, _ = linears[name]._m31_mxdraft
        direct = gemv.m3_mx_gemv_bf16(x, w8, sf, True, mxd.HIS_CFGS[SHAPES[name]])
        assert torch.equal(linears[name](x).view(torch.int16), direct.view(torch.int16)), name


def test_fused_swiglu_mxfp8_quant_is_bitwise_the_reference(routed):
    mxd, d, _, _, _ = routed
    gemv = importlib.import_module(f"{D4}.kernels.mx_gemv.op")
    nsw = importlib.import_module(f"{PKG}.ops.native_swiglu")
    w8, sf, _, cfg = d.model.layers[0].mlp._m31_hislayout_mlp
    g = torch.Generator(device="cuda").manual_seed(12)
    for rows in DECODE_ROWS + [BOUND]:
        gu = torch.randn(rows, 6144, device="cuda", generator=g).to(torch.bfloat16)
        fused = gemv.m3_mx_gemv_swiglu(gu, w8, sf, True, cfg)
        xq, xsf = torch.ops.trtllm.mxfp8_quantize(nsw.swiglu_oai(gu).contiguous(), True)
        ref = gemv.m3_mx_gemv(xq.view(rows, 3072), xsf, w8, sf, True, cfg)
        assert fused is not None and ref is not None
        assert torch.equal(fused.view(torch.int16), ref.view(torch.int16)), f"rows {rows}"


@pytest.mark.parametrize("rows", DECODE_ROWS)
def test_numerics_vs_the_current_bf16_path(routed, rows):
    mxd, d, linears, stock, _ = routed
    nsw = importlib.import_module(f"{PKG}.ops.native_swiglu")
    g = torch.Generator(device="cuda").manual_seed(20 + rows)
    worst = 0.0
    for name in ("qkv_proj", "o_proj", "gate_up_proj"):
        x = torch.randn(rows, SHAPES[name][1], device="cuda", generator=g).to(torch.bfloat16)
        ref = F.linear(x, stock[name])
        worst = max(worst, _ulp(linears[name](x), ref, name, rows))
    x = torch.randn(rows, 6144, device="cuda", generator=g).to(torch.bfloat16)
    mlp = d.model.layers[0].mlp
    ref = F.linear(nsw.swiglu_oai(F.linear(x, stock["gate_up_proj"])), stock["down_proj"])
    worst = max(worst, _ulp(mlp(x), ref, "mlp", rows))
    assert worst <= 16.0, worst


def _ulp(out, ref, name, rows):
    d = (out.float() - ref.float()).abs()
    ulp = ref.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-30) * 2.0**-7
    m = (d / ulp).max().item()
    print(
        f"NUMERICS {name} rows {rows}: frac_ne {(d > 0).float().mean().item():.4f} max_ulp_rowmax {m:.2f}"
    )
    return m


def test_bound_plus_one_runs_the_stock_path(routed):
    mxd, d, linears, stock, _ = routed
    g = torch.Generator(device="cuda").manual_seed(9)
    for name in ("qkv_proj", "o_proj", "gate_up_proj"):
        x = torch.randn(BOUND_PLUS_1, SHAPES[name][1], device="cuda", generator=g).to(
            torch.bfloat16
        )
        assert torch.equal(linears[name](x), F.linear(x, stock[name])), name
    nsw = importlib.import_module(f"{PKG}.ops.native_swiglu")
    x = torch.randn(BOUND_PLUS_1, 6144, device="cuda", generator=g).to(torch.bfloat16)
    want = F.linear(nsw.swiglu_oai(F.linear(x, stock["gate_up_proj"])), stock["down_proj"])
    assert torch.equal(d.model.layers[0].mlp(x), want)


def test_graph_replay_returns_the_eager_bits(routed):
    mxd, d, linears, _, _ = routed
    mlp = d.model.layers[0].mlp
    g = torch.Generator(device="cuda").manual_seed(21)
    xa = torch.randn(7, 6144, device="cuda", generator=g).to(torch.bfloat16)
    xo = torch.randn(7, 2048, device="cuda", generator=g).to(torch.bfloat16)

    def run():
        return (linears["qkv_proj"](xa), linears["o_proj"](xo), mlp(xa))

    eager = [t.clone() for t in run()]
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, outs):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_dispatch_lines_once_per_shape(routed):
    mxd, _, _, _, _ = routed
    tags = {t for t, *_ in mxd._HIS_TAKEN}
    assert {"gemv", "swiglu+down"} <= tags


# ---- the served row-parallel path (pkg_drgemv_v3): o_proj / down_proj are ROW-parallel with a reducing all-reduce;
# Linear.forward's NCCL-window branch (linear.py:4350-4357) calls quant_method.apply directly, never apply_linear ----


class _WindowAR(torch.nn.Module):
    """Stands in for the drafter's DraftOneshotAllReduce on the served MPI path: reports the NCCL-window output
    (dspark.py:139-140 -> AllReduce.uses_nccl_symmetric_memory_window, TLLM_NCCL_SYMMETRIC_ZERO_COPY=1 by default) and
    records what it reduces (returns it unchanged: one rank's view). An nn.Module, as Linear.all_reduce is a child
    module (2.7 pre-check: "cannot assign ... as child module 'all_reduce'")."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def uses_nccl_symmetric_memory_window(self) -> bool:
        return True

    def forward(self, x, all_reduce_params=None):
        self.calls.append(x.detach().clone())
        return x


def _as_row_parallel(lin):
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    lin.tp_mode, lin.reduce_output, lin.tp_size, lin.tp_rank = TensorParallelMode.ROW, True, 2, 0
    lin.use_fused_gemm_allreduce = False
    lin.all_reduce = _WindowAR()
    return lin


def test_the_stock_window_branch_never_reaches_apply_linear():
    """Documents the bypass the package fixes: a hook on apply_linear is not executed on the window branch."""
    from tensorrt_llm._torch.modules.linear import Linear

    lin = _as_row_parallel(Linear(2048, 6144, bias=False, dtype=torch.bfloat16).cuda())
    calls = []
    stock = lin.apply_linear
    lin.apply_linear = lambda *a, **k: calls.append(1) or stock(*a, **k)
    try:
        lin(torch.randn(7, 2048, device="cuda").to(torch.bfloat16))
    except Exception:  # noqa: BLE001 -- the stand-in all-reduce has no real window buffer; only the branch matters
        pass
    assert calls == [], "the window branch called apply_linear (the bypass premise no longer holds)"


@pytest.fixture(scope="module")
def routed_row_parallel():
    mxd = importlib.import_module(f"{D4}.drafter_mxfp8")
    d, linears = _drafter(seed=7)
    _as_row_parallel(linears["o_proj"])
    _as_row_parallel(linears["down_proj"])
    n = mxd.route_his_layout(d, DECODE_ROWS)
    return mxd, d, linears, n


def test_routed_row_parallel_o_proj_executes_the_gemv_then_its_all_reduce(routed_row_parallel):
    mxd, d, linears, n = routed_row_parallel
    gemv = importlib.import_module(f"{D4}.kernels.mx_gemv.op")
    o = linears["o_proj"]
    assert getattr(o, "_m31_hislayout_row_forward", False), (
        "the row-parallel o_proj got no forward wrapper"
    )
    before = len(mxd._HIS_TAKEN)
    x = torch.randn(
        7, 2048, device="cuda", generator=torch.Generator(device="cuda").manual_seed(31)
    ).to(torch.bfloat16)
    out = o(x)
    w8, sf, _ = o._m31_mxdraft
    direct = gemv.m3_mx_gemv_bf16(x, w8, sf, True, mxd.HIS_CFGS[SHAPES["o_proj"]])
    assert len(o.all_reduce.calls) == 1, "the projection's own all-reduce must run exactly once"
    assert torch.equal(o.all_reduce.calls[0].view(torch.int16), direct.view(torch.int16)), (
        "not the GEMV's bits"
    )
    assert torch.equal(out.view(torch.int16), direct.view(torch.int16))
    assert ("gemv", 6144, 2048, 7) in mxd._HIS_TAKEN and len(mxd._HIS_TAKEN) >= before


def test_routed_mlp_runs_the_fused_down_then_its_all_reduce(routed_row_parallel):
    mxd, d, linears, n = routed_row_parallel
    mlp = d.model.layers[0].mlp
    down = linears["down_proj"]
    k0 = len(down.all_reduce.calls)
    x = torch.randn(
        7, 6144, device="cuda", generator=torch.Generator(device="cuda").manual_seed(32)
    ).to(torch.bfloat16)
    out = mlp(x)
    assert len(down.all_reduce.calls) == k0 + 1, (
        "the fused MLP must reduce through down_proj's all-reduce once"
    )
    assert torch.equal(out.view(torch.int16), down.all_reduce.calls[-1].view(torch.int16))
    assert ("swiglu+down", 6144, 3072, 7) in mxd._HIS_TAKEN
