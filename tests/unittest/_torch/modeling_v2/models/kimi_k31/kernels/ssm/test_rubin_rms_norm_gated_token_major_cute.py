# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuTe DSL token-major gated RMSNorm (``kernels/ssm/rubin_rms_norm_gated_token_major_cute.py``) against
the four-row token-major Triton kernel it stands in for, bit for bit (sm_107).

The reference is ``rms_norm_gated_token_major`` without ``prefetch`` (the Triton kernel at any row count; its
tiles per program do not change a bit). Covered: the KDA sigmoid and the GDN silu gate; bf16 and fp32 weights;
a dense gate and column slices of wider projections; odd row counts (a half-filled last pair), head counts
that split a pair across tokens; every one of the 65536 bf16 gate values; rows of zeros, tiny, huge, infinite
and NaN values; every launch shape the kernel takes (they do not change a bit); and the launcher's hand-off
(``prefetch=True`` runs the CuTe DSL kernel only where it applies).
"""

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_rms_norm_gated as triton_norm
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm import (
    rubin_rms_norm_gated_token_major_cute as cute_norm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the family's Rubin kernels: needs an SM107 GPU",
)

WIDTH = cute_norm.WIDTH
CONFIGS = [
    cute_norm.DEFAULT_CONFIG,
    (4, 8, 4, False),
    (8, 4, 8, False),
    (8, 4, 2, True),
    (4, 6, 4, True),
]


def _inputs(tokens, heads, *, pitch=None, seed=0, w_dtype=torch.bfloat16, scale=3.0):
    """x [tokens * heads, 128]; z [tokens, heads, 128] (a column slice of a [tokens, pitch] projection when
    ``pitch`` is given, else dense); the weight."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(tokens * heads, WIDTH, generator=gen, device="cuda") * scale).to(
        torch.bfloat16
    )
    if pitch is None:
        z = (torch.randn(tokens, heads, WIDTH, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    else:
        proj = (torch.randn(tokens, pitch, generator=gen, device="cuda") * 2).to(torch.bfloat16)
        g0 = pitch - heads * WIDTH - 8 * (pitch // 64)  # a 16-byte aligned interior column block
        z = proj[:, g0 : g0 + heads * WIDTH].view(tokens, heads, WIDTH)
    w = (1 + 0.1 * torch.randn(WIDTH, generator=gen, device="cuda")).to(w_dtype)
    return x, z, w


def _triton(x, z, w, gate, eps=1e-5):
    return triton_norm.rms_norm_gated_token_major(x, z, w, eps, gate_activation=gate)


def _cute(x, z, w, gate, eps=1e-5, config=cute_norm.DEFAULT_CONFIG):
    assert cute_norm.eligible(x, z, w)
    return cute_norm.rms_norm_gated_token_major_cute(
        x, z, w, eps, silu=gate == "silu", config=config
    )


def _assert_bits(a, b):
    torch.cuda.synchronize()
    assert a.shape == b.shape and a.dtype == b.dtype
    diff = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    assert diff == 0, f"{diff} of {a.numel()} values differ"


@pytest.mark.parametrize("gate", ["sigmoid", "silu"])
@pytest.mark.parametrize(
    "tokens, heads, pitch",
    [
        (16384, 96, 49376),  # a DEP16 context rank's KDA core, the stacked projection's g columns
        (2953, 96, 49376),
        (129, 96, None),
        (8191, 12, 6288),
        (1, 12, None),
        (4097, 3, None),  # odd rows: the last pair half filled; pairs split across tokens
        (777, 5, 1024),
    ],
)
@torch.inference_mode()
def test_rows_are_the_triton_kernels(tokens, heads, pitch, gate):
    x, z, w = _inputs(tokens, heads, pitch=pitch, seed=tokens + heads)
    _assert_bits(_cute(x, z, w, gate), _triton(x, z, w, gate))


@pytest.mark.parametrize("gate", ["sigmoid", "silu"])
@torch.inference_mode()
def test_fp32_weight_and_eps(gate):
    x, z, w = _inputs(3001, 12, seed=7, w_dtype=torch.float32)
    w = w + 1e-3 * torch.randn_like(w)  # not bf16 values
    _assert_bits(_cute(x, z, w, gate, eps=1e-6), _triton(x, z, w, gate, eps=1e-6))


@pytest.mark.parametrize("config", CONFIGS, ids=[str(c) for c in CONFIGS])
@torch.inference_mode()
def test_launch_shapes_do_not_change_a_bit(config):
    x, z, w = _inputs(5003, 96, pitch=49376, seed=11)
    ref = _triton(x, z, w, "sigmoid")
    _assert_bits(_cute(x, z, w, "sigmoid", config=config), ref)


@pytest.mark.parametrize("gate", ["sigmoid", "silu"])
@torch.inference_mode()
def test_every_bf16_gate_value(gate):
    """All 65536 bf16 bit patterns as gate values (each in 8 rows, with different x rows)."""
    heads = 4
    values = (
        (torch.arange(65536, dtype=torch.int32, device="cuda") - 32768)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    z = values.reshape(-1, heads, WIDTH).repeat(8, 1, 1).contiguous()
    tokens = z.shape[0]
    gen = torch.Generator(device="cuda").manual_seed(5)
    x = (torch.randn(tokens * heads, WIDTH, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(WIDTH, generator=gen, device="cuda")).to(torch.bfloat16)
    _assert_bits(_cute(x, z, w, gate), _triton(x, z, w, gate))


@pytest.mark.parametrize("tokens, heads", [(2953, 96), (8191, 12)])
@torch.inference_mode()
def test_edge_rows(tokens, heads):
    gen = torch.Generator(device="cuda").manual_seed(tokens)
    x = torch.randn(tokens * heads, WIDTH, generator=gen, device="cuda") * 3
    z = torch.randn(tokens, heads, WIDTH, generator=gen, device="cuda") * 2
    x[0] = 0
    z[1] = -1.0e4
    for h in range(heads):
        z[2, h] = -88.0 + 0.25 * (h % 9)
    x[4 * heads] = 3.0e38
    x[4 * heads + 1, :32] = 1.0e-38
    x[4 * heads + 2] = 1.0e-30
    x[4 * heads + 3, ::2] = 1.0e-20
    x[5 * heads, 7] = float("inf")
    x[5 * heads + 1, 9] = float("nan")
    z[6, 0, :32] = float("nan")
    z[7, 0, 5] = float("inf")
    z[7, 1, 6] = -float("inf")
    x, z = x.to(torch.bfloat16), z.to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(WIDTH, generator=gen, device="cuda")).to(torch.bfloat16)
    for gate in ("sigmoid", "silu"):
        ref = _triton(x, z, w, gate)
        assert bool(ref.isnan().any())
        _assert_bits(_cute(x, z, w, gate), ref)
        _assert_bits(_cute(x, z, w, gate, eps=0.0), _triton(x, z, w, gate, eps=0.0))


@torch.inference_mode()
def test_inputs_untouched_and_graph_replay():
    x, z, w = _inputs(16384, 96, pitch=49376, seed=3)
    copies = [t.clone() for t in (x, z, w)]
    eager = _cute(x, z, w, "sigmoid")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out = _cute(x, z, w, "sigmoid")
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    _assert_bits(out, eager)
    for a, b in zip((x, z, w), copies):
        assert torch.equal(a, b)


@torch.inference_mode()
def test_launcher_hands_off_only_where_it_applies(monkeypatch):
    """``prefetch=True`` runs the CuTe DSL kernel at prefill row counts with an eligible layout, and the Triton
    kernel for small calls, the FP8 output and layouts it does not take."""
    calls = []
    real = triton_norm.rms_norm_gated_token_major_cute

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(triton_norm, "rms_norm_gated_token_major_cute", spy)
    big = _inputs(16384, 96, pitch=49376, seed=1)
    small = _inputs(64, 96, seed=2)
    out = triton_norm.rms_norm_gated_token_major(
        *big, 1e-5, gate_activation="sigmoid", prefetch=True
    )
    assert calls == [16384 * 96]
    _assert_bits(out, _triton(*big, "sigmoid"))
    triton_norm.rms_norm_gated_token_major(*small, 1e-5, gate_activation="sigmoid", prefetch=True)
    triton_norm.rms_norm_gated_token_major(*big, 1e-5, gate_activation="sigmoid")
    scale = torch.tensor([0.05], device="cuda")
    fp8 = triton_norm.rms_norm_gated_token_major(
        *big, 1e-5, fp8_scale=scale, gate_activation="sigmoid", prefetch=True
    )
    assert fp8.dtype == torch.float8_e4m3fn
    x, z, w = big
    misaligned = torch.empty(x.shape[0] * WIDTH + 1, dtype=x.dtype, device="cuda")[1:].view_as(x)
    misaligned.copy_(x)
    out = triton_norm.rms_norm_gated_token_major(
        misaligned, z, w, 1e-5, gate_activation="sigmoid", prefetch=True
    )
    assert calls == [16384 * 96]
    _assert_bits(out, _triton(x, z, w, "sigmoid"))
