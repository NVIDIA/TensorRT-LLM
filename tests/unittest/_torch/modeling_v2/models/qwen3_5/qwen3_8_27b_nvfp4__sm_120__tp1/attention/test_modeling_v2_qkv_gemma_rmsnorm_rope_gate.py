# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 qkv_gemma_rmsnorm_rope_gate entry.

Per cell:

1. Bitwise against the in-tree fused q/k-norm RoPE gate op
   (`torch.ops.trtllm.fused_qkv_gemma_rmsnorm_rope_gate`, the same arithmetic reading a
   contiguous `[3, T]` copy of the positions and copying the gate out): `qkv_out` and
   the gate.
2. The gate is a view of the projection -- same storage, strides
   `(row stride, 2 * head_dim, 1)` -- and the projection is left unmodified.
3. Against the entry's fp32 reference: the gate and V bitwise; Q and K in
   `assert_within_ulp(element_ulp=6, rms_ulp=0.5)`. Derivation: the kernel rounds the
   normed head to bf16 like the reference, but its `rsqrt.approx` and tree-ordered sum
   can flip that rounding by one bf16 ulp (<= 2^-7 |x|) where the fp32 value sits by a
   midpoint. A rotated pair carries both flips into `x1 cos - x2 sin`, at most
   2^-7 |(x1, x2)| <= 2^-7 sqrt(2) max|out| (a rotation keeps the pair's norm, so some
   output of the row is at least |(x1, x2)| / sqrt(2)); with one more ulp for the output's
   own rounding that is 2 sqrt(2) + 2 < 6 units of 2^-8 of the row maximum. Flips are
   rare (~2^-11 an element), so the relative RMS stays far under 0.5 units; a wrong
   section, position or head is off by the values themselves.

Then CUDA-graph replay with the position buffer rewritten in place.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.attention.qkv_gemma_rmsnorm_rope_gate import (  # noqa: E501
    qkv_gemma_rmsnorm_rope_gate as op,
)
from tensorrt_llm._torch.attention.kernels import fused_qk_norm_rope_gate as _in_tree  # noqa: F401
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)

#: The runner's MRoPE position buffer `[3, 1, max_num_tokens]`, and the RoPE table length.
_MAX_TOKENS, _MAX_POSITIONS = 8192, 262144


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _cos_sin(rotary_dim: int, theta: float) -> torch.Tensor:
    """`[max_positions, 2, rotary_dim / 2]` fp32 NeoX table."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    freqs = torch.outer(torch.arange(_MAX_POSITIONS, dtype=torch.float32, device="cuda"), inv_freq)
    return torch.stack((freqs.cos(), freqs.sin()), dim=1).contiguous()


def _build(spec, seed):
    """The op's arguments for one spec, plus the position buffer the slice views."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, hq, hkv, d = spec["tokens"], spec["num_q_heads"], spec["num_kv_heads"], spec["head_dim"]
    width = 2 * hq * d + 2 * hkv * d
    qkv = (torch.randn(t, width, generator=g, device="cuda") * 2).to(torch.bfloat16)
    q_weight = (torch.randn(d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    k_weight = (torch.randn(d, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    # Three different sections, so a wrong section stride or section index shows.
    base = torch.randint(0, _MAX_POSITIONS - 64, (t,), generator=g, device="cuda")
    sections = torch.stack((base, base + 17, base + 41))
    if spec["positions"] == "slice":
        buffer = torch.randint(
            0, _MAX_POSITIONS, (3, 1, _MAX_TOKENS), generator=g, device="cuda"
        ).to(torch.int32)
        buffer[:, 0, :t] = sections.to(torch.int32)
        positions = buffer[:, :, :t]
    else:
        assert spec["positions"] == "contiguous_int64", spec["positions"]
        buffer = positions = sections.contiguous()
    args = [
        qkv,
        q_weight,
        k_weight,
        _cos_sin(spec["rotary_dim"], spec["rope_theta"]),
        positions,
        spec["eps"],
        hq,
        hkv,
        d,
        spec["rotary_dim"],
        tuple(spec["mrope_section"]),
    ]
    return args, buffer


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _in_tree_op(args):
    """The in-tree op on the same inputs: positions copied to a contiguous `[3, T]`."""
    qkv, qw, kw, cos_sin, positions, eps, hq, hkv, d, rot, section = args
    return torch.ops.trtllm.fused_qkv_gemma_rmsnorm_rope_gate(
        qkv,
        qw,
        kw,
        cos_sin,
        positions.reshape(3, -1).contiguous(),
        eps,
        hq,
        hkv,
        d,
        rot,
        True,
        section[1],
        section[2],
        False,  # enable_tiling: the per-token kernel, the arithmetic the target's op matches
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, _ = _build(cell.spec, seed=_seed(cell.why))
    qkv = args[0]
    hq, hkv, d = args[6], args[7], args[8]
    q_size, kv_size = hq * d, hkv * d
    qkv_before = qkv.clone()
    want_qkv, want_gate = _in_tree_op(args)

    qkv_out, gate = op(*args)

    assert torch.equal(_bits(qkv), _bits(qkv_before)), "the projection is read only"
    assert qkv_out.shape == (qkv.shape[0], q_size + 2 * kv_size) and qkv_out.is_contiguous()
    # 1. Bitwise the in-tree op.
    assert torch.equal(_bits(qkv_out), _bits(want_qkv)), "qkv_out differs from the in-tree op"
    assert torch.equal(_bits(gate.contiguous()), _bits(want_gate)), "gate differs"
    # 2. The gate is a view: no copy, the g_i half of each interleaved head.
    assert gate.untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    assert gate.data_ptr() == qkv.data_ptr() + d * qkv.element_size()
    # A size-1 dimension's stride is never used, and torch reports any value for it.
    assert gate.shape == (qkv.shape[0], hq, d)
    assert gate.stride()[1:] == (2 * d, 1)
    assert qkv.shape[0] == 1 or gate.stride(0) == qkv.stride(0)
    # 3. The fp32 reference.
    ref_qkv, ref_gate = op.reference(*args)
    assert torch.equal(_bits(gate), _bits(ref_gate))
    assert torch.equal(_bits(qkv_out[:, q_size + kv_size :]), _bits(ref_qkv[:, q_size + kv_size :]))
    assert_within_ulp(
        qkv_out[:, : q_size + kv_size], ref_qkv[:, : q_size + kv_size], element_ulp=6, rms_ulp=0.5
    )


def test_cuda_graph_replay_reads_the_position_buffer_in_place() -> None:
    """The decode graph captures the op on a slice of the runner's position buffer; each
    replay must read the positions (and the projection) written into it since, with the
    output poisoned in between."""
    spec = next(c.spec for c in op.CELLS if c.spec["tokens"] == 32)
    args, buffer = _build(spec, seed=_seed("graph"))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qkv_out, gate = op(*args)

    for replay in range(3):
        fresh, fresh_buffer = _build(spec, seed=_seed(f"replay {replay}"))
        for static, new in zip(args[:4], fresh[:4]):  # qkv, q/k weights, cos_sin
            static.copy_(new)
        buffer.copy_(fresh_buffer)
        want_qkv, want_gate = _in_tree_op(args)
        qkv_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(qkv_out), _bits(want_qkv)), f"replay {replay}: qkv_out"
        assert torch.equal(_bits(gate.contiguous()), _bits(want_gate)), f"replay {replay}: gate"
