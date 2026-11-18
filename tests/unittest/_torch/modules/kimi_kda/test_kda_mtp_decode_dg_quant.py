# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA MTP verify epilogue quantization with DeepGEMM-packed scales (SM100/103).

``kda_mtp_decode_impl(quantize_output=True, sf_deepgemm=True)`` returns the gated-RMSNorm output as
E4M3 plus UE8M0 1x128 scales in the packed MN-major int32 layout of
``fp8_quantize_1x128_packed_ue8m0(x, False)`` (the SM100/103 DeepGEMM o_proj input). It must
produce the same bytes as running the BF16-output verify and quantizing that output with the
standalone quantizer, and the DeepGEMM projection of the pair must equal the projection that
quantizes by itself. Device tests: SM100 family with the CuTe DSL.
"""

import pytest
import torch

try:
    import cutlass  # noqa: F401

    from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops as kda_ops
except ImportError as e:
    kda_ops = None
    _ERR = str(e)
else:
    _ERR = None

try:
    from test_kda_mtp_decode_pdl import _assert_bitwise, _clone, _make_inputs
except ImportError:  # pragma: no cover
    _make_inputs = None


def _skip_reason():
    if kda_ops is None or _make_inputs is None:
        return f"deps: {_ERR}"
    if not torch.cuda.is_available():
        return "needs a GPU"
    if torch.cuda.get_device_capability(0)[0] != 10 or torch.cuda.get_device_capability(0) == (
        10,
        7,
    ):
        return "DeepGEMM scale layout: SM100/103 only"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")

H = 12


def _run(inp, **kw):
    x = _clone(inp)
    x.update(kw)
    out = kda_ops.kda_mtp_decode_impl(**x)
    torch.cuda.synchronize()
    return out, x


@pytest.mark.parametrize(
    "N,num_spec",
    [(n, ns) for ns in (0, 1, 3, 5) for n in (1, 2, 3, 6, 11)]
    + [(n, 7) for n in range(1, 9)],  # production: K7 = 8 rows per request, N = 1..8 requests
)
def test_epilogue_matches_standalone_quantizer(N, num_spec, monkeypatch):
    monkeypatch.setenv("TRTLLM_KDA_MTP_BF16_MATH", "0")
    inp = _make_inputs(N, H, num_spec=num_spec, seed=99 + N)
    T = N * (1 + num_spec)
    ref_bf16, ref_x = _run(inp, quantize_output=False)
    ref_act, ref_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
        ref_bf16.reshape(T, H * 128).contiguous(), False
    )
    (act, words), new_x = _run(inp, quantize_output=True, sf_deepgemm=True)
    sf = kda_ops.deepgemm_scale_view(words, T, H)
    assert sf.shape == ref_sf.shape and sf.stride() == ref_sf.stride()
    _assert_bitwise("act", act, ref_act[:T])
    _assert_bitwise("scale", sf.contiguous(), ref_sf.contiguous())
    for name in ("cs_q", "cs_k", "cs_v", "recurrent_state", "k_cache", "g_cache", "v_cache"):
        _assert_bitwise(name, new_x[name], ref_x[name])


@pytest.mark.parametrize("N", [1, 2, 6])
def test_projection_of_the_pair_matches_the_quantizing_projection(N):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    num_spec = 5
    inp = _make_inputs(N, H, num_spec=num_spec, seed=7 + N)
    T = N * (1 + num_spec)
    torch.manual_seed(5)
    lin = torch.nn.Linear(H * 128, 7168, bias=False, dtype=torch.bfloat16, device="cuda")
    proj = _Fp8BlockScaleWeightReadLinear.from_linear(lin)
    proj.transform_weights()
    assert proj.supports_deepgemm_prequantized_input
    ref_bf16, _ = _run(inp, quantize_output=False)
    ref = proj(ref_bf16.reshape(T, H * 128).contiguous())
    (act, words), _ = _run(inp, quantize_output=True, sf_deepgemm=True)
    got = proj.forward_prequantized_deepgemm(act, kda_ops.deepgemm_scale_view(words, T, H))
    _assert_bitwise("o_proj output", got, ref)


def test_pdl_and_dg_scales_compose(monkeypatch):
    monkeypatch.setenv("TRTLLM_K3_KDA_VERIFY_PDL", "1")
    inp = _make_inputs(2, H, num_spec=5, seed=3)
    T = 12
    (act, words), _ = _run(inp, quantize_output=True, sf_deepgemm=True)
    monkeypatch.delenv("TRTLLM_K3_KDA_VERIFY_PDL")
    (act0, words0), _ = _run(inp, quantize_output=True, sf_deepgemm=True)
    _assert_bitwise("act", act, act0)
    _assert_bitwise(
        "scale",
        kda_ops.deepgemm_scale_view(words, T, H).contiguous(),
        kda_ops.deepgemm_scale_view(words0, T, H).contiguous(),
    )


# ---------------------------------------------------------------------------
# Scale rounding at amax = 448 * 2^k: the epilogue's rule against both standalone quantizers.
# ---------------------------------------------------------------------------
_BOUNDARY_AMAX = [448.0 * 2.0**k for k in range(-4, 5)]
_NEAR = [a * f for a in (448.0, 896.0, 224.0) for f in (0.99, 1.0, 1.01)]


def _epilogue_scale_bytes(amax_values):
    """The epilogue's two scale lines, run as a one-thread-per-value CuTe kernel:
    ``E8M0(bf16-exact amax * Float32(1/448))`` (kda_mtp_decode.py)."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def kern(amax: cute.Tensor, sf: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        scale = (amax[i] * cutlass.Float32(1.0 / 448.0)).to(cutlass.Float8E8M0FNU)
        sf_bytes = cute.make_tensor(
            cute.recast_ptr(sf.iterator, dtype=cutlass.Float8E8M0FNU),
            cute.make_layout(sf.shape),
        )
        sf_bytes[i] = scale

    @cute.jit
    def run(amax: cute.Tensor, sf: cute.Tensor, n: cutlass.Int32):
        kern(amax, sf).launch(grid=(n, 1, 1), block=(128, 1, 1))

    n = (len(amax_values) + 127) // 128
    a = torch.zeros(n * 128, dtype=torch.float32, device="cuda")
    a[: len(amax_values)] = torch.tensor(amax_values, dtype=torch.float32)
    sf = torch.zeros(n * 128, dtype=torch.uint8, device="cuda")
    compiled = cute.compile(run, from_dlpack(a), from_dlpack(sf), n)
    compiled(from_dlpack(a), from_dlpack(sf), n)
    torch.cuda.synchronize()
    return sf[: len(amax_values)].cpu()


def _standalone_scale_bytes(amax_values):
    """Scale byte of each 1x128 block with that amax, from (cuda, triton) quantizers."""
    n = len(amax_values)
    x = torch.full((n, 128), 0.25, dtype=torch.bfloat16, device="cuda")
    x[:, 0] = torch.tensor(amax_values, dtype=torch.float32).to(torch.bfloat16).cuda()
    # (n, 128): K = 128 is one block per row; packed layout holds 4 blocks per word, so use K=512.
    x4 = torch.full((n, 512), 0.25, dtype=torch.bfloat16, device="cuda")
    x4[:, 0] = x[:, 0]
    _, sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x4, False)
    cuda_bytes = (sf.contiguous()[:n, 0].view(torch.uint8).reshape(n, 4)[:, 0]).cpu()
    try:
        from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco

        _, tsf = tco._fp8_quantize_1x128_ue8m0(x4, 1)  # Fp8QuantKernelRunner.TACTIC_TRITON
        tri_bytes = (tsf.contiguous()[:n, 0].view(torch.uint8).reshape(n, 4)[:, 0]).cpu()
    except Exception as e:  # report, do not hide
        tri_bytes = None
        print("triton quantizer unavailable:", e)
    return cuda_bytes, tri_bytes


def test_scale_rounding_at_448_pow2_boundaries():
    amax = _BOUNDARY_AMAX + _NEAR
    ep = _epilogue_scale_bytes(amax)
    cu, tr = _standalone_scale_bytes(amax)
    rows = []
    for i, a in enumerate(amax):
        rows.append((a, int(ep[i]), int(cu[i]), None if tr is None else int(tr[i])))
    print("amax, epilogue, cuda_packed, triton (E8M0 bytes)")
    for r in rows:
        print(r)
    bad = [r for r in rows if r[1] != r[2] or (r[3] is not None and r[1] != r[3])]
    assert not bad, f"scale mismatches (amax, epilogue, cuda, triton): {bad}"
