# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 ``fixed_low_m_gemm`` entry and ``drafter_fixed_gemms``.

The entry is the built-in low-M split-K kernel at a fixed tactic, so its claim
is accuracy, not bitwise equality with cuBLAS: every cell is gated against the
fp32 single-rounding ``reference`` within one bf16 ulp of the row scale, and
its error against an fp64 product must not exceed cuBLAS's by more than one
ulp. The install tests drive the drafter's own module classes before and after
``install_fixed_tactic_gemms``, and check that every candidate call site it
leaves built-in is reported with the reason.
"""

import types
import zlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.fixed_low_m_gemm import (  # noqa: E501
    SM107_SPLITK_TACTICS,
    SPLITK_TACTICS,
    SPLITK_TACTICS_BY_SM_TP,
    SPLITK_TACTICS_TP4,
    splitk_tactics,
    tactic_for,
    takes,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.fixed_low_m_gemm import (  # noqa: E501
    fixed_low_m_gemm as op,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlash2BlockConv, DFlashForCausalLM
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod

__extra_import_path__ = ["../../../../.."]
from _validating import ULP, assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "fixed_low_m_gemm requires a CUDA device"

#: Element band against the single-rounding fp32 reference, in units of
#: 2^-8 x the row's largest magnitude (``assert_within_ulp``): both sides
#: accumulate in fp32 (error ~K * 2^-24 relative, far below a bf16 ulp) and
#: round once, so they differ by at most one bf16 rounding step of an element
#: no larger than the row scale -- 2^(e-7) for a row scale in [2^e, 2^(e+1)),
#: i.e. up to 2 of these units. RMS: only the elements whose fp32 sums
#: straddle a rounding boundary differ (<1 % of them here).
ELEMENT_ULP, RMS_ULP = 2.0, 0.25
#: The compute capability the certified cells and the TP4 table were measured on.
_SM103 = (10, 3)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _operands(m, k, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    w = (0.02 * torch.randn(n, k, generator=g, device="cuda")).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    m, k, n = (cell.spec[d] for d in ("m", "k", "n"))
    tactic = splitk_tactics(cell.spec["tp"], _SM103)[(m, k, n)]
    x, w = _operands(m, k, n, _seed(cell.why))
    out = op(x, w, tactic, True)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16 and out.is_contiguous()
    ref = op.reference(x, w, tactic, True)
    assert_within_ulp(out, ref, ELEMENT_ULP, RMS_ULP)
    exact = x.double() @ w.double().t()
    err = (out.double() - exact).abs().max().item()
    cublas_err = (F.linear(x, w).double() - exact).abs().max().item()
    one_ulp = exact.abs().max().item() * ULP[torch.bfloat16]
    assert err <= cublas_err + one_ulp, (err, cublas_err, one_ulp)


def test_the_tactic_tables_by_sm_and_tp_size():
    """On sm103, TP4 and TP8 each have a table covering every drafter call site of that width;
    sm107 has the TP8 shapes; other pairs none."""
    assert splitk_tactics(8, _SM103) is SPLITK_TACTICS
    assert splitk_tactics(4, _SM103) is SPLITK_TACTICS_TP4
    assert splitk_tactics(8, (10, 7)) is SM107_SPLITK_TACTICS
    assert set(SM107_SPLITK_TACTICS) == set(SPLITK_TACTICS)
    for tp in (1, 2, 16):
        with pytest.raises(ValueError, match=rf"no split-K tactics for SM \(10, 3\) at TP {tp}"):
            splitk_tactics(tp, _SM103)
    with pytest.raises(ValueError, match=r"no split-K tactics for SM \(10, 7\) at TP 4"):
        splitk_tactics(4, (10, 7))
    # Every TP-sharded drafter projection is twice as wide (or deep) per rank at TP4.
    sharded = {(8192, 1280): (8192, 2560), (1024, 8192): (2048, 8192)}
    sharded.update({(3584, 8192): (7168, 8192), (8192, 7168): (8192, 14336)})
    for tp8, tp4 in sharded.items():
        assert (8, *tp8) in SPLITK_TACTICS and (8, *tp4) in SPLITK_TACTICS_TP4
    # The replicated convolution coefficient projection is the same at both widths.
    assert (8, 8192, 2048) in SPLITK_TACTICS and (8, 8192, 2048) in SPLITK_TACTICS_TP4
    # drafter_fc_shard's TP4 shard runs the built-in GEMM: it has no TP4 entry.
    assert (8, 40960, 1024) not in SPLITK_TACTICS_TP4
    assert {c.spec["tp"] for c in op.CELLS} == {4, 8}
    for cell in op.CELLS:
        spec = cell.spec
        assert (spec["m"], spec["k"], spec["n"]) in splitk_tactics(spec["tp"], _SM103), cell.why


@pytest.mark.parametrize("shape", sorted(SPLITK_TACTICS))
@torch.inference_mode()
def test_the_devices_tactic_meets_the_cells_band(shape):
    """``tactic_for`` gives this device's TP8 tactic (``SPLITK_TACTICS_BY_SM_TP``) for every
    listed shape, within the certified cells' band of the fp32 reference and no farther from
    fp64 than cuBLAS."""
    m, k, n = shape
    capability = torch.cuda.get_device_capability()
    tactic = tactic_for(m, k, n)
    assert tactic == SPLITK_TACTICS_BY_SM_TP[capability, 8][shape]
    x, w = _operands(m, k, n, _seed(f"device {shape}"))
    out = op(x, w, tactic, True)
    assert_within_ulp(out, op.reference(x, w, tactic, True), ELEMENT_ULP, RMS_ULP)
    exact = x.double() @ w.double().t()
    err = (out.double() - exact).abs().max().item()
    cublas_err = (F.linear(x, w).double() - exact).abs().max().item()
    one_ulp = exact.abs().max().item() * ULP[torch.bfloat16]
    assert err <= cublas_err + one_ulp, (err, cublas_err, one_ulp)


@pytest.mark.parametrize(
    "tp, shape", [(tp, shape) for tp in (8, 4) for shape in sorted(splitk_tactics(tp, _SM103))]
)
@torch.inference_mode()
def test_cuda_graph_replay_follows_new_inputs(tp, shape):
    m, k, n = shape
    tactic = splitk_tactics(tp, _SM103)[shape]
    x, w = _operands(m, k, n, 3)
    op(x, w, tactic, True)  # JIT outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, w, tactic, True)
    for scale in (1.0, 7.0, 0.01):
        x.copy_(torch.randn_like(x, dtype=torch.float32).mul_(scale).to(x.dtype))
        graph.replay()
        assert torch.equal(out, op(x, w, tactic, True))


@torch.inference_mode()
def test_takes_refuses_what_the_kernel_does_not_take():
    x, w = _operands(8, 1024, 256, 1)
    assert takes(x, w)
    assert not takes(x.float(), w.float())
    assert not takes(torch.cat([x, x], 1)[:, ::2], w)  # non-contiguous
    assert not takes(torch.zeros(33, 1024, dtype=torch.bfloat16, device="cuda"), w)
    assert not takes(x[:, :1000].contiguous(), w[:, :1000].contiguous())  # K % 128
    assert not takes(x, w[:, :512].contiguous())  # K mismatch
    with torch.inference_mode(False), torch.enable_grad():
        assert not takes(x.clone(), w.clone())  # autograd on: the built-in path


# --------------------------------------------------------------------------
# drafter_fixed_gemms install on the drafter's module classes
# --------------------------------------------------------------------------
#: The drafter's per-rank TP-sharded projections ``name: (K, N)`` at TP8 and TP4.
TP8_LAYER = dict(qkv_proj=(8192, 1280), o_proj=(1024, 8192), down_proj=(3584, 8192))
TP4_LAYER = dict(
    qkv_proj=(8192, 2560), o_proj=(2048, 8192), down_proj=(7168, 8192), gate_up_proj=(8192, 14336)
)


def _stand_in_drafter(layer_shapes=None):
    """A DFlashForCausalLM carrying the module classes drafter_fixed_gemms touches, at the real per-rank shapes."""
    drafter = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(drafter)
    hidden, taps, group = 8192, 2, 16
    groups = hidden // group
    g = torch.Generator(device="cuda").manual_seed(7)

    def conv():
        return DFlash2BlockConv(
            base_kernel=torch.randn(2, taps, hidden, generator=g, device="cuda").to(torch.bfloat16),
            kernel_projection_weight=(
                0.02 * torch.randn(2 * taps * groups, hidden, generator=g, device="cuda")
            ).to(torch.bfloat16),
            taps=taps,
            group_size=group,
            hidden_size=hidden,
        )

    drafter.attention_convs = nn.ModuleList([conv()])
    drafter.mlp_convs = nn.ModuleList([conv()])
    layer = nn.Module()
    for name, (k, n) in (TP8_LAYER if layer_shapes is None else layer_shapes).items():
        lin = Linear(k, n, bias=False, dtype=torch.bfloat16)
        lin.weight.data = (0.02 * torch.randn(n, k, generator=g, device="cuda")).to(torch.bfloat16)
        setattr(layer, name, lin)
    drafter.model = nn.Module()
    drafter.model.layers = nn.ModuleList([layer])
    return drafter


@torch.inference_mode()
def test_install_reclasses_only_and_runs_the_fixed_tactic():
    drafter = _stand_in_drafter()
    params = {n: p.data_ptr() for n, p in drafter.named_parameters()}
    layer = drafter.model.layers[0]
    x8 = {k: torch.randn(8, k, device="cuda").to(torch.bfloat16) for k in (8192, 1024, 3584)}
    before = {
        "conv": drafter.attention_convs[0].kernel_projection(x8[8192]),
        "qkv": layer.qkv_proj(x8[8192]),
        "o": layer.o_proj(x8[1024]),
        "down": layer.down_proj(x8[3584]),
    }
    sites = drafter_worker.install_fixed_tactic_gemms(drafter, enable_pdl=True)
    # 2 conv projections + 3 Linear modules + the context K/V GEMM.
    assert sites == drafter_worker.FixedTacticSites(6, ())
    assert drafter_worker.install_fixed_tactic_gemms(drafter, enable_pdl=True) == sites
    assert (
        type(drafter.attention_convs[0].kernel_projection) is drafter_worker.FixedTacticProjection
    )
    assert type(layer.qkv_proj.quant_method) is drafter_worker.FixedTacticLinearMethod
    assert {n: p.data_ptr() for n, p in drafter.named_parameters()} == params
    after = {
        "conv": drafter.attention_convs[0].kernel_projection(x8[8192]),
        "qkv": layer.qkv_proj(x8[8192]),
        "o": layer.o_proj(x8[1024]),
        "down": layer.down_proj(x8[3584]),
    }
    weights = {
        "conv": drafter.attention_convs[0].kernel_projection.weight,
        "qkv": layer.qkv_proj.weight,
        "o": layer.o_proj.weight,
        "down": layer.down_proj.weight,
    }
    inputs = {"conv": x8[8192], "qkv": x8[8192], "o": x8[1024], "down": x8[3584]}
    for name in after:
        w, x = weights[name], inputs[name]
        tactic = splitk_tactics(8)[(8, x.shape[1], w.shape[0])]
        assert torch.equal(after[name], op(x, w, tactic, True)), name
        assert_within_ulp(after[name], before[name], ELEMENT_ULP, 2 * RMS_ULP)
    # Shapes the table does not list run the built-in path, unchanged.
    x3 = torch.randn(3, 8192, device="cuda").to(torch.bfloat16)
    assert (3, 8192, 1280) not in SPLITK_TACTICS
    assert torch.equal(
        layer.qkv_proj(x3), UnquantizedLinearMethod().apply(layer.qkv_proj, x3, None)
    )
    assert torch.equal(
        drafter.attention_convs[0].kernel_projection(x3),
        F.linear(x3, drafter.attention_convs[0].kernel_projection.weight),
    )


@torch.inference_mode()
def test_tp4_install_runs_the_tp4_tactics_and_the_tp8_table_declines_tp4_shapes():
    """At TP4 every drafter call site takes the TP4 table's tactic, within the band of the built-in."""
    from types import SimpleNamespace

    if torch.cuda.get_device_capability() != _SM103:
        pytest.skip("the TP4 tactic table is measured on sm103 only")

    drafter = _stand_in_drafter(TP4_LAYER)
    layer = drafter.model.layers[0]
    # The TP8 table lists none of the TP4 per-rank shapes: the old refusal at TP4.
    declined = drafter_worker.install_fixed_tactic_gemms(_stand_in_drafter(TP4_LAYER), True)
    assert len(declined.declined) == 4 and all("no fixed tactic" in d for d in declined.declined)
    with pytest.raises(ValueError, match="no split-K tactics for SM"):
        drafter_worker.install(drafter, None, SimpleNamespace(tp_size=2), False, False, True)
    inputs = {
        name: torch.randn(8, k, device="cuda").to(torch.bfloat16)
        for name, (k, _) in TP4_LAYER.items()
    }
    before = {name: getattr(layer, name)(x) for name, x in inputs.items()}
    drafter_worker.install(drafter, None, SimpleNamespace(tp_size=4), False, False, True)
    for name, x in inputs.items():
        linear = getattr(layer, name)
        assert type(linear.quant_method) is drafter_worker.FixedTacticLinearMethod
        assert linear.quant_method.tactics is SPLITK_TACTICS_TP4
        tactic = SPLITK_TACTICS_TP4[(8, x.shape[1], linear.weight.shape[0])]
        after = linear(x)
        assert torch.equal(after, op(x, linear.weight, tactic, True)), name
        assert_within_ulp(after, before[name], ELEMENT_ULP, 2 * RMS_ULP)
    proj = drafter.attention_convs[0].kernel_projection
    assert proj._fixed_gemm_tactics is SPLITK_TACTICS_TP4
    x = inputs["qkv_proj"]
    assert torch.equal(proj(x), op(x, proj.weight, SPLITK_TACTICS_TP4[(8, 8192, 2048)], True))


@pytest.mark.parametrize("mode", ["inference_mode", "no_grad"])
def test_projection_with_trainable_weights_runs_under_inference(mode):
    """As served: weights built outside inference mode (the conv's ``nn.Linear``
    weight requires grad), the forward run under inference mode or no_grad."""
    drafter = _stand_in_drafter()
    proj = drafter.attention_convs[0].kernel_projection
    assert proj.weight.requires_grad and not proj.weight.is_inference()
    drafter_worker.install_fixed_tactic_gemms(drafter, enable_pdl=True)
    ctx = torch.inference_mode() if mode == "inference_mode" else torch.no_grad()
    with ctx:
        x = torch.randn(8, 8192, device="cuda").to(torch.bfloat16)
        out = proj(x)
        tactic = splitk_tactics(8)[(8, 8192, proj.weight.shape[0])]
        assert torch.equal(out, op(x, proj.weight.detach(), tactic, True))
        assert torch.equal(op(x, proj.weight, tactic, True), out)


@torch.inference_mode()
def test_context_kv_is_the_builtin_body_with_the_gemm_swapped(monkeypatch):
    drafter = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(drafter)
    g = torch.Generator(device="cuda").manual_seed(9)
    L, nkv, hd = 5, 1, 128
    drafter._fused_kv_weight = (
        0.02 * torch.randn(L * 2 * nkv * hd, 8192, generator=g, device="cuda")
    ).to(torch.bfloat16)
    drafter._fused_kv_bias = None
    drafter._num_attn_layers, drafter._num_kv_heads, drafter._head_dim = L, nkv, hd
    drafter._k_norm_stacked = torch.rand(L, hd, generator=g, device="cuda").to(torch.bfloat16)
    drafter._k_norm_eps = 1e-5
    drafter._input_ln_eps = None
    # RoPE is the built-in's, unchanged; a deterministic stand-in keeps the test local.
    drafter._fused_rope_inplace = lambda k_flat, positions, N, L: k_flat.mul_(0.5)
    drafter.attention_convs = drafter.mlp_convs = None
    drafter.model = nn.Module()
    drafter.model.layers = nn.ModuleList()
    ph = torch.randn(8, 8192, generator=g, device="cuda").to(torch.bfloat16)
    pos = torch.arange(100, 108, device="cuda")
    k0, v0 = DFlashForCausalLM.precompute_context_kv(drafter, ph, pos)
    assert drafter_worker.install_fixed_tactic_gemms(
        drafter, enable_pdl=True
    ) == drafter_worker.FixedTacticSites(1, ())
    k1, v1 = drafter.precompute_context_kv(ph, pos)
    assert_within_ulp(v1.reshape(8, -1), v0.reshape(8, -1), ELEMENT_ULP, 2 * RMS_ULP)
    # k: RMS-normalized per 128-wide head after the GEMM, scaled by k_norm, then
    # rounded again: two more roundings on each side.
    assert_within_ulp(k1.reshape(8, -1), k0.reshape(8, -1), 2 * ELEMENT_ULP, 4 * RMS_ULP)
    # With the GEMM put back, the written-out body is bitwise the built-in.
    monkeypatch.setattr(drafter_worker, "_gemm", lambda x, w, pdl, tactics: F.linear(x, w))
    k2, v2 = drafter.precompute_context_kv(ph, pos)
    assert torch.equal(k2, k0) and torch.equal(v2, v0)


@torch.inference_mode()
def test_a_linear_on_the_custom_cublas_gemm_stays_builtin(monkeypatch):
    """Its output may have to live in an NCCL symmetric-memory window, which only the built-in fills."""
    drafter = _stand_in_drafter()
    layer = drafter.model.layers[0]
    layer.o_proj.use_custom_cublas_mm = True
    sites = drafter_worker.install_fixed_tactic_gemms(drafter, enable_pdl=True)
    assert sites.installed == 5
    assert sites.declined == (
        "model.layers.0.o_proj runs the custom cuBLAS GEMM (output may live in an NCCL window)",
    )
    assert type(layer.o_proj.quant_method) is UnquantizedLinearMethod
    # A module switched to it after the install still gets the built-in path.
    layer.qkv_proj.use_custom_cublas_mm = True
    builtin_result = torch.zeros(1)
    monkeypatch.setattr(
        UnquantizedLinearMethod, "apply", lambda self, module, input, bias: builtin_result
    )
    monkeypatch.setattr(drafter_worker, "fixed_low_m_gemm", None)  # must not be reached
    x = torch.randn(8, 8192, device="cuda").to(torch.bfloat16)
    assert layer.qkv_proj.quant_method.apply(layer.qkv_proj, x, None) is builtin_result


@torch.inference_mode()
def test_install_reports_every_site_it_leaves_builtin():
    drafter = _stand_in_drafter()
    layer = drafter.model.layers[0]

    class OtherMethod(UnquantizedLinearMethod):
        """Another feature's linear method."""

    class OtherProjection(nn.Linear):
        """Another feature's projection class."""

    layer.o_proj.quant_method = OtherMethod()
    odd = Linear(8192, 1024 + 128, bias=False, dtype=torch.bfloat16)  # no listed tactic
    odd.weight.data = torch.zeros(1024 + 128, 8192, dtype=torch.bfloat16, device="cuda")
    layer.odd_proj = odd
    drafter.mlp_convs[0].kernel_projection.__class__ = OtherProjection
    foreign = types.MethodType(lambda self, projected_hidden, positions: None, drafter)
    drafter.precompute_context_kv = foreign

    sites = drafter_worker.install_fixed_tactic_gemms(drafter, enable_pdl=True)
    assert sites.installed == 3  # the attention conv projection, qkv_proj, down_proj
    assert sites.declined == (
        "mlp_convs.0.kernel_projection is OtherProjection, not a plain nn.Linear",
        "model.layers.0.o_proj quant method is OtherMethod",
        "model.layers.0.odd_proj no fixed tactic for a [1152, 8192] weight",
        "context K/V GEMM: precompute_context_kv is already replaced on this drafter by "
        f"{__name__}.test_install_reports_every_site_it_leaves_builtin.<locals>.<lambda>",
    )
    assert type(layer.o_proj.quant_method) is OtherMethod
    assert type(odd.quant_method) is UnquantizedLinearMethod
    assert type(drafter.mlp_convs[0].kernel_projection) is OtherProjection
    assert drafter.__dict__["precompute_context_kv"] is foreign
    assert type(layer.qkv_proj.quant_method) is drafter_worker.FixedTacticLinearMethod
