# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gdn_verify_tcgen05 (the tcgen05 GDN MTP target-verify kernel) against trtllm::gdn_verify with FP8 output.

The two differ only in the MMA accumulation order and the gated norm's sum order: y agrees up to FP8 rounding flips,
the committed state and the histories to bf16 rounding. A replay from the same inputs is bitwise identical.
"""

import pytest
import torch

from tensorrt_llm._torch.kernels.blackwell import gdn_verify_tcgen05 as gdn_verify_module
from tensorrt_llm._torch.kernels.blackwell.gdn_verify_tcgen05 import (
    compile_gdn_verify_tcgen05,
    gdn_verify_tcgen05,
    gdn_verify_tcgen05_supported,
)

skip_no_sm100f = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)
_DIM, _HIST, _SLOTS = 128, 16, 256
_T = 7


def _verify_case(heads, n, t, pnat, seed=1000):
    """Per-rank shapes of a 16 k-head / 64 v-head GDN layer: heads = (k heads, v heads)."""
    K = V = _DIM
    H, HV = heads
    g = torch.Generator(device="cuda").manual_seed(seed)
    cd = 2 * H * K + HV * V
    proj = (torch.randn(n * t, cd + HV * V, generator=g, device="cuda") * 0.7).to(torch.bfloat16)
    ba = torch.randn(n * t, 2 * HV, generator=g, device="cuda").to(torch.bfloat16)
    st = dict(
        a_log=torch.log(torch.rand(HV, generator=g, device="cuda") * 7 + 1).float(),
        dt_bias=(torch.rand(HV, generator=g, device="cuda") * 2 - 1).float(),
        norm_w=(torch.rand(V, generator=g, device="cuda") + 0.5).to(torch.bfloat16),
        fp8_scale=torch.tensor(0.02, dtype=torch.float32, device="cuda"),
        state_indices=torch.randperm(_SLOTS, generator=g, device="cuda")[:n].to(torch.int32),
        pnat=torch.full((_SLOTS,), pnat, dtype=torch.int32, device="cuda"),
        buf=torch.zeros(_SLOTS, dtype=torch.int32, device="cuda"),
    )
    st["buf"][2::3] = 1
    kh = torch.randn(_SLOTS, 2, _HIST, H, K, generator=g, device="cuda")
    dec = torch.rand(_SLOTS, 2, HV, _HIST, generator=g, device="cuda") * 0.49 + 0.01
    mut = dict(
        ssm=(torch.randn(_SLOTS, HV, V, K, generator=g, device="cuda") * 0.05).to(torch.bfloat16),
        old_k=(kh / kh.norm(dim=-1, keepdim=True)).to(torch.bfloat16),
        old_u=(torch.randn(_SLOTS, 2, _HIST, HV, V, generator=g, device="cuda") * 0.5).to(
            torch.bfloat16
        ),
        old_g=(-torch.cumsum(dec, dim=-1)).float().contiguous(),
    )
    return HV, cd, proj, ba, st, mut


def _verify_op(op, HV, cd, n, t, proj, ba, st, mut):
    """Run op (trtllm::gdn_verify's argument list, FP8 output) on clones of the mutable state."""
    m = {k: v.clone() for k, v in mut.items()}
    y = op(
        proj[:, :cd],
        proj[:, cd:].view(n * t, HV, _DIM),
        ba[:, HV:],
        ba[:, :HV],
        st["a_log"],
        st["dt_bias"],
        st["norm_w"],
        st["fp8_scale"],
        1e-6,
        m["ssm"],
        st["state_indices"],
        m["old_u"],
        m["old_k"],
        m["old_g"],
        st["buf"],
        st["pnat"],
    )
    return y, m


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _check_matches_reference(y, m, y_ref, m_ref, mut, state_indices):
    """y up to FP8 rounding flips; the state and histories of the requests' slots up to bf16 rounding (errors
    relative to those slots only), and every other slot left untouched by both ops."""
    assert y.dtype == torch.float8_e4m3fn and y.shape == y_ref.shape
    assert _rel(y, y_ref) < 5e-2
    assert (y.view(torch.uint8) != y_ref.view(torch.uint8)).float().mean().item() < 0.2
    slots = state_indices[state_indices >= 0].long()
    other = torch.ones(_SLOTS, dtype=torch.bool, device=slots.device)
    other[slots] = False
    for k in mut:
        assert torch.equal(m[k][other], mut[k][other]), k
        assert torch.equal(m_ref[k][other], mut[k][other]), k
    assert _rel(m["old_u"][slots], m_ref["old_u"][slots]) < 1e-3
    assert _rel(m["old_k"][slots], m_ref["old_k"][slots]) < 1e-2
    assert torch.allclose(m["old_g"][slots], m_ref["old_g"][slots], rtol=1e-5, atol=1e-6)
    assert _rel(m["ssm"][slots], m_ref["ssm"][slots]) < 1e-3


@pytest.mark.parametrize(
    "T,HV,HK,fp8_out,expected",
    [
        (7, 8, 2, True, True),
        (7, 64, 16, True, True),
        (2, 4, 1, True, True),
        (8, 8, 2, True, True),
        (7, 8, 2, False, False),
        (1, 8, 2, True, False),
        (9, 8, 2, True, False),
        (7, 8, 3, True, False),
        (7, 0, 2, True, False),
    ],
)
def test_gdn_verify_tcgen05_supported(T, HV, HK, fp8_out, expected):
    assert gdn_verify_tcgen05_supported(T, HV, HK, fp8_out) is expected


@skip_no_sm100f
@pytest.mark.parametrize(
    "heads,n",
    [
        ((2, 8), 1),
        ((2, 8), 16),
        ((2, 8), 64),
        ((2, 8), 128),
        ((16, 64), 8),
        ((16, 64), 52),
        # GQA groups HV / HK of 1, 2 and 3 (the cases above are group 4).
        ((4, 4), 1),
        ((4, 4), 52),
        ((4, 8), 1),
        ((4, 8), 52),
        ((4, 12), 1),
        ((4, 12), 52),
    ],
)
@pytest.mark.parametrize("pnat", [0, 12])
def test_gdn_verify_tcgen05_matches_gdn_verify(heads, n, pnat):
    HV, cd, proj, ba, st, mut = _verify_case(heads, n, _T, pnat)
    y_ref, m_ref = _verify_op(torch.ops.trtllm.gdn_verify, HV, cd, n, _T, proj, ba, st, mut)
    y, m = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
    torch.cuda.synchronize()
    _check_matches_reference(y, m, y_ref, m_ref, mut, st["state_indices"])
    if pnat + _T > _HIST:
        assert not torch.equal(m["ssm"], mut["ssm"]), "commit expected (history full)"


@skip_no_sm100f
@pytest.mark.parametrize("heads,n", [((2, 8), 128), ((16, 64), 52)])
def test_gdn_verify_tcgen05_replay_is_bitwise(heads, n):
    """Two runs from the same inputs (one committing slot in three requests) give bitwise-identical results."""
    H, HV = heads
    compile_gdn_verify_tcgen05(_T, HV, H, torch.device("cuda"))
    HV, cd, proj, ba, st, mut = _verify_case(heads, n, _T, 0)
    st["pnat"][st["state_indices"][::3].long()] = 12
    y0, m0 = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
    y1, m1 = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
    torch.cuda.synchronize()
    assert torch.equal(y0.view(torch.uint8), y1.view(torch.uint8))
    for k in m0:
        assert torch.equal(m0[k], m1[k]), k
    assert not torch.equal(m0["ssm"], mut["ssm"]), "commit expected (history full)"


@skip_no_sm100f
@pytest.mark.parametrize("t", [2, 4, 7, 8])
@pytest.mark.parametrize("pnat", ["zero", "offset", "fills", "mixed"])
@pytest.mark.parametrize("n", [16, 200])
def test_gdn_verify_tcgen05_matches_gdn_verify_across_draft_lengths(t, pnat, n):
    """Every admitted draft length, appends at an offset into the history, a history the step fills exactly
    (16 - T), batches mixing appends and commits, and more requests than one CTA wave of 128."""
    heads = (2, 8)
    start = {"zero": 0, "offset": 3, "fills": _HIST - t, "mixed": 0}[pnat]
    HV, cd, proj, ba, st, mut = _verify_case(heads, n, t, start, seed=2000 + t)
    if pnat == "mixed":
        # Every third request commits (its history overflows), the others append at offset 3.
        st["pnat"].fill_(3)
        st["pnat"][st["state_indices"][::3].long()] = 12
    y_ref, m_ref = _verify_op(torch.ops.trtllm.gdn_verify, HV, cd, n, t, proj, ba, st, mut)
    y, m = _verify_op(gdn_verify_tcgen05, HV, cd, n, t, proj, ba, st, mut)
    torch.cuda.synchronize()
    _check_matches_reference(y, m, y_ref, m_ref, mut, st["state_indices"])


@skip_no_sm100f
def test_gdn_verify_tcgen05_honours_enable_pdl(monkeypatch):
    """TRTLLM_ENABLE_PDL=0 compiles and launches the kernel without PDL; the results are bitwise those with PDL."""
    H, HV = 2, 8
    n = 52
    compile_pdl = []
    real_compile = gdn_verify_module.cute.compile

    def recording_compile(*args, **kwargs):
        compile_pdl.append(kwargs.get("PDL"))
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(gdn_verify_module.cute, "compile", recording_compile)
    gdn_verify_module._compiled.cache_clear()
    HV, cd, proj, ba, st, mut = _verify_case((H, HV), n, _T, 0)
    st["pnat"][st["state_indices"][::3].long()] = 12
    results = {}
    try:
        for env in ("1", "0"):
            monkeypatch.setenv("TRTLLM_ENABLE_PDL", env)
            compile_gdn_verify_tcgen05(_T, HV, H)
            results[env] = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
        torch.cuda.synchronize()
    finally:
        gdn_verify_module._compiled.cache_clear()
    assert compile_pdl == [True, False]
    (y_on, m_on), (y_off, m_off) = results["1"], results["0"]
    assert torch.equal(y_on.view(torch.uint8), y_off.view(torch.uint8))
    for k in m_on:
        assert torch.equal(m_on[k], m_off[k]), k


def _page_views(pages, HV, H):
    """Views of replay pages laid out as the V2 hybrid manager binds them: per replay slot a [prev_num_accepted,
    cache_buf_idx, ., .] int32 row, then the history planes (slot strides of a page)."""
    planes = {
        "old_u": (torch.bfloat16, (2, _HIST, HV, _DIM)),
        "old_k": (torch.bfloat16, (2, _HIST, H, _DIM)),
        "old_g": (torch.float32, (2, HV, _HIST)),
    }
    views, off = {}, 16
    for name, (dtype, shape) in planes.items():
        nbytes = dtype.itemsize * torch.Size(shape).numel()
        views[name] = pages[:, off : off + nbytes].view(dtype).unflatten(1, shape)
        off += nbytes
    meta = pages[:, :16].view(torch.int32)
    return views, meta[:, 0], meta[:, 1]


def _page_bytes(HV, H):
    nbytes = 16 + 2 * _HIST * (HV + H) * _DIM * 2 + 2 * HV * _HIST * 4
    return (nbytes + 48 + 15) // 16 * 16


@skip_no_sm100f
@pytest.mark.parametrize("heads,n", [((2, 8), 52), ((16, 64), 8)])
def test_gdn_verify_tcgen05_addresses_the_history_by_replay_slot(heads, n):
    """History and bookkeeping in replay pages of their own (the KV-cache replay region: slot strides of a page,
    replay slots other than the state slots): bitwise the result of the same step on per-state-slot contiguous
    history, trtllm::gdn_verify's up to the kernels' rounding, and no write outside the requests' history rows."""
    H, HV = heads
    HV, cd, proj, ba, st, mut = _verify_case(heads, n, _T, 0)
    st["pnat"][st["state_indices"][::3].long()] = 12  # every third request commits
    st["state_indices"][1] = -1  # a padded request
    num_replay_slots = _SLOTS + 40
    g = torch.Generator(device="cuda").manual_seed(7)
    replay = torch.randperm(num_replay_slots, generator=g, device="cuda")[:n].to(torch.int32)
    valid = st["state_indices"] >= 0
    rows, slots = replay[valid].long(), st["state_indices"][valid].long()

    pages = torch.randint(
        0,
        256,
        (num_replay_slots, _page_bytes(HV, H)),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    views, pnat, buf = _page_views(pages, HV, H)
    # every bookkeeping row valid (a padded request reads replay slot 0's)
    pnat.zero_()
    buf.zero_()
    for name, view in views.items():
        view[rows] = mut[name][slots]
    pnat[rows] = st["pnat"][slots]
    buf[rows] = st["buf"][slots]

    def run(op):
        p = pages.clone()
        v, pn, cb = _page_views(p, HV, H)
        ssm = mut["ssm"].clone()
        y = op(
            proj[:, :cd],
            proj[:, cd:].view(n * _T, HV, _DIM),
            ba[:, HV:],
            ba[:, :HV],
            st["a_log"],
            st["dt_bias"],
            st["norm_w"],
            st["fp8_scale"],
            1e-6,
            ssm,
            st["state_indices"],
            v["old_u"],
            v["old_k"],
            v["old_g"],
            cb,
            pn,
            replay,
        )
        return y, ssm, p, v

    y_c, m_c = _verify_op(gdn_verify_tcgen05, HV, cd, n, _T, proj, ba, st, mut)
    y_p, ssm_p, pages_p, v_p = run(gdn_verify_tcgen05)
    y_r, ssm_r, _, v_r = run(torch.ops.trtllm.gdn_verify)
    torch.cuda.synchronize()
    out = torch.arange(n * _T, device="cuda").view(n, _T)[valid].flatten()
    assert torch.equal(y_p.view(torch.uint8)[out], y_c.view(torch.uint8)[out])
    assert torch.equal(ssm_p, m_c["ssm"])
    for name in views:
        assert torch.equal(v_p[name][rows], m_c[name][slots]), name
    untouched = torch.ones(num_replay_slots, dtype=torch.bool, device="cuda")
    untouched[rows] = False
    assert torch.equal(pages_p[untouched], pages[untouched])
    assert torch.equal(pages_p[:, :16], pages[:, :16])  # the bookkeeping is the caller's to advance
    assert _rel(y_p[out], y_r[out]) < 5e-2
    assert _rel(v_p["old_u"][rows], v_r["old_u"][rows]) < 1e-3
    assert torch.allclose(v_p["old_g"][rows], v_r["old_g"][rows], rtol=1e-5, atol=1e-6)
    assert _rel(ssm_p[slots], ssm_r[slots]) < 1e-3
    other = torch.ones(_SLOTS, dtype=torch.bool, device="cuda")
    other[slots] = False
    assert torch.equal(ssm_p[other], mut["ssm"][other])
    assert torch.equal(ssm_r[other], mut["ssm"][other])


@pytest.mark.parametrize(
    "bad", ["z_rows", "z_heads", "a_heads", "b_rows", "old_u_buffers", "old_k_hist"]
)
def test_gdn_verify_tcgen05_checks_the_op_shapes(bad):
    """The wrapper refuses, before any launch, the z / a / b and history shapes trtllm::gdn_verify refuses."""
    n = 4
    HV, cd, proj, ba, st, mut = _verify_case((2, 8), n, _T, 0)
    z = proj[:, cd:].view(n * _T, HV, _DIM)
    a, b = ba[:, HV:], ba[:, :HV]
    old_u, old_k = mut["old_u"], mut["old_k"]
    if bad == "z_rows":
        z = z[:-1]
    elif bad == "z_heads":
        z = z[:, : HV // 2]
    elif bad == "a_heads":
        a = ba[:, HV + 1 :]
    elif bad == "b_rows":
        b = b[:-1]
    elif bad == "old_u_buffers":
        old_u = old_u[:, :1]
    else:
        old_k = old_k[:, :, : _HIST // 2]
    with pytest.raises(AssertionError):
        gdn_verify_tcgen05(
            proj[:, :cd], z, a, b, st["a_log"], st["dt_bias"], st["norm_w"], st["fp8_scale"], 1e-6,
            mut["ssm"].clone(), st["state_indices"], old_u, old_k, mut["old_g"], st["buf"], st["pnat"],
        )  # fmt: skip


@pytest.mark.parametrize(
    "bad",
    [
        "qkv_dtype",
        "z_dtype",
        "a_dtype",
        "b_dtype",
        "old_u_dtype",
        "old_k_dtype",
        "ssm_dtype",
        "scale_dtype",
        "scale_numel",
        "qkv_base",
        "ssm_base",
    ],
)
def test_gdn_verify_tcgen05_checks_the_op_dtypes_and_alignment(bad):
    """The wrapper refuses, before any launch, the dtypes, the FP8 scale and the 16-byte bases trtllm::gdn_verify
    refuses (gdnVerifyOp.cpp): the kernel would otherwise read them as BF16 / a FP32 scalar / aligned rows."""
    n = 4
    HV, cd, proj, ba, st, mut = _verify_case((2, 8), n, _T, 0)
    x = proj[:, :cd]
    z = proj[:, cd:].view(n * _T, HV, _DIM)
    a, b = ba[:, HV:], ba[:, :HV]
    old_u, old_k, ssm = mut["old_u"], mut["old_k"], mut["ssm"].clone()
    scale = st["fp8_scale"]
    if bad == "qkv_dtype":
        x = x.to(torch.float16)
    elif bad == "z_dtype":
        z = z.to(torch.float16)
    elif bad == "a_dtype":
        a = a.float()
    elif bad == "b_dtype":
        b = b.float()
    elif bad == "old_u_dtype":
        old_u = old_u.to(torch.float16)
    elif bad == "old_k_dtype":
        old_k = old_k.to(torch.float16)
    elif bad == "ssm_dtype":
        ssm = ssm.to(torch.float16)
    elif bad == "scale_dtype":
        scale = scale.to(torch.bfloat16)
    elif bad == "scale_numel":
        scale = scale.reshape(1).repeat(2)
    elif bad == "qkv_base":
        # Rows keep their 16-byte stride; the base moves by 8 elements (16 bytes) minus one.
        wide = torch.zeros(x.shape[0], x.shape[1] + 8, dtype=torch.bfloat16, device="cuda")
        x = wide.view(-1)[7 : 7 + x.numel()].view(x.shape[0], x.shape[1])
        assert x.stride(0) % 8 == 0 and x.data_ptr() % 16 != 0
    else:
        flat = torch.zeros(ssm.numel() + 8, dtype=torch.bfloat16, device="cuda")
        ssm = flat[1 : 1 + ssm.numel()].view(ssm.shape)
        assert ssm.data_ptr() % 16 != 0
    with pytest.raises(AssertionError):
        gdn_verify_tcgen05(
            x, z, a, b, st["a_log"], st["dt_bias"], st["norm_w"], scale, 1e-6,
            ssm, st["state_indices"], old_u, old_k, mut["old_g"], st["buf"], st["pnat"],
        )  # fmt: skip


def test_gdn_compiles_the_tcgen05_verify_only_for_an_fp8_out_proj(monkeypatch):
    """The tcgen05 verify runs only when the gated norm writes E4M3 (norm.fp8_scale, set for an FP8QDQ out_proj):
    cache_derived_state compiles it then, and a BF16 out_proj never compiles it."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod, UnquantizedLinearMethod
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm

    compiled = []
    monkeypatch.setattr(
        gdn_verify_module, "compile_gdn_verify_tcgen05", lambda *args: compiled.append(args)
    )

    def layer(quant_method):
        m = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
        torch.nn.Module.__init__(m)
        m.fused_verify = False
        m.norm = RMSNorm(8)
        m.num_v_heads_per_tp, m.num_k_heads_per_tp = 16, 4
        # What __init__ sets for a supported MTP verify (rows gate, draft rows per request).
        m.verify_tcgen05_min_rows, m.verify_tcgen05_tokens = 64, 7
        m.out_proj = SimpleNamespace(
            quant_method=quant_method,
            input_scale=torch.nn.Parameter(torch.tensor(0.5), requires_grad=False),
            force_dynamic_quantization=False,
        )
        return m

    layer(UnquantizedLinearMethod()).cache_derived_state()
    assert compiled == []
    layer(FP8QDQLinearMethod()).cache_derived_state()
    assert compiled == [(7, 16, 4)]
