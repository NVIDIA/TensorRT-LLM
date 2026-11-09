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
"""Kimi-K3 KDA verify with the forget gate computed in the kernel.

The fused path runs ``trtllm::k3_kda_fab_fa`` (the fab kernel's GEMM 1 and reduction, BF16
``[f_a | b]``) and ``kda_mtp_decode`` with ``fa`` / ``w_fb`` (the f_b MMA inside the verify
kernel), optionally as a programmatic dependent launch. It must reproduce the current
``k3_kda_fab_proj`` + ``kda_mtp_decode`` pair bit for bit: outputs, committed recurrent state,
conv windows and replay caches, over several verify rounds with mixed acceptance, eagerly and
under CUDA graph replay.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)

H = 6
K = 128
W = 4
HIDDEN = 7168
MUTATED = ("state", "cs_q", "cs_k", "cs_v", "k_cache", "g_cache", "v_cache", "beta_cache")


def _n1(heads):
    """[f_a | b] rows padded to a multiple of 8 (the decode weight's layout)."""
    return (K + heads + 7) // 8 * 8


# the default (H heads) layer's [f_a | b] rows and projection width
N1 = _n1(H)
DIM = H * K


@pytest.fixture(autouse=True)
def _register_ops():
    """Registers trtllm::k3_kda_fab_proj / k3_kda_fab_fa (on import of their module)."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda import k3_kda_fab  # noqa: F401


def _ops():
    from tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_kda_mtp_ops import kda_mtp_decode_impl

    return kda_mtp_decode_impl


def _layer(batch, num_spec, seed, heads=H):
    """Weights, projections and a decode state that already went through several rounds."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"
    H, N1, DIM = heads, _n1(heads), heads * K

    def rn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=g) * scale).to(dtype).to(dev)

    pool = max(2 * batch, 8)
    m = batch * (num_spec + 1)
    s = W - 1 + num_spec
    x = torch.randn(m, HIDDEN, generator=g)
    chan = torch.ones(HIDDEN)
    chan[torch.randperm(HIDDEN, generator=g)[:16]] = 30.0  # activation outlier channels
    w_bfa = torch.randn(N1, HIDDEN, generator=g) * 0.02
    w_bfa[K + H :] = 0
    # Per-row state scales spread over two orders of magnitude, like a long-running request.
    state = torch.randn(pool, H, K, K, generator=g) * 0.05
    state *= torch.logspace(-1, 1, K).view(1, 1, K, 1)
    perm = torch.randperm(pool, generator=g)[:batch].to(torch.int32)
    accepted = torch.zeros(pool, dtype=torch.int32)
    accepted[perm.long()] = torch.randint(0, num_spec + 1, (batch,), generator=g, dtype=torch.int32)
    return {
        "heads": heads,
        "x": (x * chan).bfloat16().to(dev),
        "w_bfa": w_bfa.bfloat16().to(dev),
        "w_fb": rn(DIM, K, scale=0.05, dtype=torch.bfloat16),
        "qkvg": rn(m, 4 * DIM, scale=0.5, dtype=torch.bfloat16),
        "w_q": rn(DIM, W, scale=0.5),
        "w_k": rn(DIM, W, scale=0.5),
        "w_v": rn(DIM, W, scale=0.5),
        "A_log": rn(H, scale=0.5),
        "dt_bias": rn(DIM, scale=0.5),
        "cs_q": rn(pool, s, DIM, dtype=torch.bfloat16).transpose(1, 2),
        "cs_k": rn(pool, s, DIM, dtype=torch.bfloat16).transpose(1, 2),
        "cs_v": rn(pool, s, DIM, dtype=torch.bfloat16).transpose(1, 2),
        "state": state.to(dev),
        "k_cache": rn(pool, num_spec, DIM, scale=0.08, dtype=torch.bfloat16),
        "g_cache": -torch.rand(pool, num_spec, DIM, generator=g).to(dev) * 0.5,
        "v_cache": rn(pool, num_spec, DIM, dtype=torch.bfloat16),
        "beta_cache": torch.rand(pool, num_spec, H, generator=g).to(torch.bfloat16).to(dev),
        "onorm_w": (0.75 + 0.5 * torch.rand(K, generator=g)).to(dev),
        "slots": perm.to(dev),
        "accepted": accepted.to(dev),
    }


def _clone(lay):
    out = dict(lay)
    for key in MUTATED:
        t = lay[key]
        c = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        c.copy_(t)
        out[key] = c
    return out


def _decode_kwargs(lay, batch, num_spec, *, g=None, beta=None, bfa=None, use_pdl=False):
    m = batch * (num_spec + 1)
    H = lay["heads"]
    DIM = H * K
    qkvg = lay["qkvg"]
    kw = dict(
        x_q=qkvg[:, :DIM].view(1, m, H, K),
        x_k=qkvg[:, DIM : 2 * DIM].view(1, m, H, K),
        x_v=qkvg[:, 2 * DIM : 3 * DIM].view(1, m, H, K),
        w_q=lay["w_q"],
        w_k=lay["w_k"],
        w_v=lay["w_v"],
        cs_q=lay["cs_q"],
        cs_k=lay["cs_k"],
        cs_v=lay["cs_v"],
        g=None if g is None else g.view(1, m, H, K),
        beta=(beta if bfa is None else bfa[:, K : K + H]).view(1, m, H),
        A_log=lay["A_log"],
        dt_bias=lay["dt_bias"],
        recurrent_state=lay["state"],
        k_cache=lay["k_cache"],
        g_cache=lay["g_cache"],
        v_cache=lay["v_cache"],
        beta_cache=lay["beta_cache"],
        ssm_state_indices=lay["slots"],
        cu_seqlens=None,
        num_spec=num_spec,
        num_accepted_tokens=lay["accepted"],
        lower_bound=-5.0,
        scale=K**-0.5,
        packed_token_layout=True,
        onorm_g=qkvg[:, 3 * DIM :].view(1, m, H, K),
        onorm_weight=lay["onorm_w"],
        onorm_eps=1e-5,
        fuse_output_norm=True,
        use_pdl=use_pdl,
    )
    if bfa is not None:
        kw["fa"] = bfa[:, :K].view(1, m, K)
        kw["w_fb"] = lay["w_fb"]
    return kw


def _bits(t):
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _assert_same(name, actual, expected):
    assert actual.shape == expected.shape, name
    assert torch.equal(_bits(actual), _bits(expected)), (
        f"{name}: max |diff| {(actual.float() - expected.float()).abs().max().item():.3e}"
    )


def _next_acceptance(lay, batch, num_spec, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    acc = torch.randint(0, num_spec + 1, (batch,), generator=g, dtype=torch.int32)
    lay["accepted"][lay["slots"].long()] = acc.to("cuda")


def _reference(lay, batch, num_spec):
    """The current pair: k3_kda_fab_proj (beta, forget gate) + kda_mtp_decode."""
    beta, gate = torch.ops.trtllm.k3_kda_fab_proj(lay["x"], lay["w_bfa"], lay["w_fb"], lay["heads"])
    return _ops()(**_decode_kwargs(lay, batch, num_spec, g=gate, beta=beta))


def _fused(lay, batch, num_spec, use_pdl):
    bfa = torch.ops.trtllm.k3_kda_fab_fa(lay["x"], lay["w_bfa"], lay["heads"], lay["w_fb"])
    return _ops()(**_decode_kwargs(lay, batch, num_spec, bfa=bfa, use_pdl=use_pdl))


@pytest.mark.parametrize("m", [1, 7, 8, 32, 56, 64, 100, 128, 168, 224, 256])
def test_fab_fa_matches_fab_beta(m):
    lay = _layer((m + 7) // 8, 7, seed=m)
    x = lay["x"][:m].contiguous()
    beta, _ = torch.ops.trtllm.k3_kda_fab_proj(x, lay["w_bfa"], lay["w_fb"], H)
    bfa = torch.ops.trtllm.k3_kda_fab_fa(x, lay["w_bfa"], H, lay["w_fb"])
    _assert_same("beta", bfa[:, K : K + H], beta)
    # f_a is the fab tail's BF16 rounding of the same FP32 sums: close to the FP32 reference
    ref = x.float() @ lay["w_bfa"].float().t()
    torch.testing.assert_close(bfa[:, :K].float(), ref[:, :K], rtol=2**-7, atol=1e-3)
    # deterministic run to run (CUDA-graph-safe workspace hand-off)
    _assert_same("bfa rerun", torch.ops.trtllm.k3_kda_fab_fa(x, lay["w_bfa"], H, lay["w_fb"]), bfa)


@pytest.mark.parametrize("m", [8, 16, 32, 64])
def test_fab_fa_matches_fab_beta_helix8(m):
    """12 heads per rank (helix8), the verify rows of 1 / 2 / 4 / 8 requests with seven drafts."""
    lay = _layer(m // 8, 7, seed=m, heads=12)
    beta, _ = torch.ops.trtllm.k3_kda_fab_proj(lay["x"], lay["w_bfa"], lay["w_fb"], 12)
    bfa = torch.ops.trtllm.k3_kda_fab_fa(lay["x"], lay["w_bfa"], 12, lay["w_fb"])
    _assert_same("beta", bfa[:, K : K + 12], beta)
    ref = lay["x"].float() @ lay["w_bfa"].float().t()
    torch.testing.assert_close(bfa[:, :K].float(), ref[:, :K], rtol=2**-7, atol=1e-3)


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize(
    "heads,batch,num_spec",
    [(H, 1, 7), (H, 4, 7), (H, 8, 7), (H, 16, 7), (H, 32, 6), (H, 32, 7), (H, 4, 5), (H, 16, 6)]
    + [(12, 1, 7), (12, 2, 7), (12, 4, 7), (12, 8, 7)],
)
@torch.no_grad()
def test_fused_gate_verify_bitwise(heads, batch, num_spec, use_pdl):
    lay = _layer(batch, num_spec, seed=100 * batch + num_spec, heads=heads)
    ref, fused = _clone(lay), _clone(lay)
    for rnd in range(3):
        out_ref = _reference(ref, batch, num_spec)
        out_fused = _fused(fused, batch, num_spec, use_pdl)
        torch.cuda.synchronize()
        assert torch.isfinite(out_ref.float()).all()
        _assert_same(f"round {rnd} out", out_fused, out_ref)
        for key in MUTATED:
            _assert_same(f"round {rnd} {key}", fused[key], ref[key])
        for state in (ref, fused):
            _next_acceptance(state, batch, num_spec, seed=rnd)


@pytest.mark.parametrize("batch,num_spec", [(4, 7), (16, 7), (32, 6)])
@torch.no_grad()
def test_pdl_verify_bitwise(batch, num_spec):
    """The PDL launch of the unchanged-gate path is bitwise the plain launch."""
    lay = _layer(batch, num_spec, seed=7 + batch)
    beta, gate = torch.ops.trtllm.k3_kda_fab_proj(lay["x"], lay["w_bfa"], lay["w_fb"], H)
    plain, pdl = _clone(lay), _clone(lay)
    for rnd in range(2):
        out_plain = _ops()(**_decode_kwargs(plain, batch, num_spec, g=gate, beta=beta))
        out_pdl = _ops()(**_decode_kwargs(pdl, batch, num_spec, g=gate, beta=beta, use_pdl=True))
        _assert_same(f"round {rnd} out", out_pdl, out_plain)
        for key in MUTATED:
            _assert_same(f"round {rnd} {key}", pdl[key], plain[key])
        for state in (plain, pdl):
            _next_acceptance(state, batch, num_spec, seed=50 + rnd)


@torch.no_grad()
def test_fused_gate_uses_w_fb():
    """Guard against a vacuous comparison: a different W_fb must change the result."""
    batch, num_spec = 4, 7
    lay = _layer(batch, num_spec, seed=3)
    a, b = _clone(lay), _clone(lay)
    b["w_fb"] = (lay["w_fb"].float() * 1.5).bfloat16()
    out_a = _fused(a, batch, num_spec, use_pdl=False)
    out_b = _fused(b, batch, num_spec, use_pdl=False)
    assert not torch.equal(out_a, out_b)


@pytest.mark.parametrize("batch,num_spec", [(8, 7), (16, 7), (32, 6)])
@torch.no_grad()
def test_fused_gate_cuda_graph(batch, num_spec):
    """fab (f_a mode) on a side stream + the PDL verify kernel, captured and replayed."""
    lay = _layer(batch, num_spec, seed=11 + batch)
    ref, graphed = _clone(lay), _clone(lay)
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    fork, join = torch.cuda.Event(), torch.cuda.Event()
    x_static = lay["x"].clone()

    def step(state):
        fork.record(main)
        with torch.cuda.stream(side):
            side.wait_event(fork)
            bfa = torch.ops.trtllm.k3_kda_fab_fa(x_static, state["w_bfa"], H, state["w_fb"])
            join.record(side)
        main.wait_event(join)
        return _ops()(**_decode_kwargs(state, batch, num_spec, bfa=bfa, use_pdl=True))

    # Eager warm-up on a scratch copy (compiles, creates the fab workspace outside capture).
    scratch = _clone(lay)
    with torch.cuda.stream(main):
        step(scratch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        out_graph = step(graphed)
    torch.cuda.synchronize()
    for rnd in range(3):
        out_ref = _reference(ref, batch, num_spec)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(f"replay {rnd} out", out_graph, out_ref)
        for key in MUTATED:
            _assert_same(f"replay {rnd} {key}", graphed[key], ref[key])
        # the next round: new tokens and acceptance, written into the captured buffers
        x_new = (lay["x"].float() * (0.5 + 0.25 * rnd)).bfloat16()
        x_static.copy_(x_new)
        ref["x"] = x_new
        for state in (ref, graphed):
            _next_acceptance(state, batch, num_spec, seed=90 + rnd)


def _low_m_o_proj(n, k, seed):
    """FP8 ``[n, k]`` weight with power-of-two 128x128 block scales, and its packed scale."""
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((n, k), device="cuda", generator=gen) * 0.02
    blocks = w.view(n // 128, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    w_fp8 = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    packed = transform_sf_into_required_layout(
        scale.float(), mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
    )
    return w_fp8, packed


@pytest.mark.parametrize("batch,num_spec", [(1, 7), (4, 7)])
@torch.no_grad()
def test_pdl_verify_releases_o_proj_cuda_graph(batch, num_spec):
    """The PDL verify releases its dependent before it completes: the low-M o_proj GEMM behind
    it (which streams its weights before its griddepcontrol.wait) still projects the completed
    output, captured and replayed."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    if torch.cuda.get_device_capability() == (10, 7) and not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("the SM107 low-M GEMM needs a Rubin-capable CuTe DSL")
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm  # noqa: F401

    m, dim = batch * (num_spec + 1), H * K
    w_fp8, packed = _low_m_o_proj(HIDDEN, dim, seed=5)

    def o_proj(core):
        return torch.ops.trtllm.fp8_block_scale_low_m_gemm(core.reshape(m, dim), w_fp8, packed)

    lay = _layer(batch, num_spec, seed=31 + batch)
    ref, graphed = _clone(lay), _clone(lay)
    x_static = lay["x"].clone()

    def step(state):
        bfa = torch.ops.trtllm.k3_kda_fab_fa(x_static, state["w_bfa"], H, state["w_fb"])
        out = _ops()(**_decode_kwargs(state, batch, num_spec, bfa=bfa, use_pdl=True))
        return out, o_proj(out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(_clone(lay))  # compiles and creates the fab workspace outside capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph, proj_graph = step(graphed)
    torch.cuda.synchronize()
    for rnd in range(3):
        out_ref = _reference(ref, batch, num_spec)
        proj_ref = o_proj(out_ref)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(f"replay {rnd} out", out_graph, out_ref)
        _assert_same(f"replay {rnd} o_proj", proj_graph, proj_ref)
        for key in MUTATED:
            _assert_same(f"replay {rnd} {key}", graphed[key], ref[key])
        x_new = (lay["x"].float() * (0.5 + 0.25 * rnd)).bfloat16()
        x_static.copy_(x_new)
        ref["x"] = x_new
        for state in (ref, graphed):
            _next_acceptance(state, batch, num_spec, seed=70 + rnd)
