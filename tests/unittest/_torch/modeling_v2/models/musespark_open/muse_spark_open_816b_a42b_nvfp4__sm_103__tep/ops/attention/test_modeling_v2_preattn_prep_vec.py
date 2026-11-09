# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``_preattn_prep_rows_vec_kernel`` is bit for bit ``_preattn_prep_rows_kernel``.

CPU: the switch (TRTLLM_MSO_ENABLE_CTX4_PREP_VEC) parses and is off by default.
Device (sm_103): both kernels on the same projection, every output compared as raw bits, at
row counts across the rows-kernel range (513 .. 32768, ragged tails included), RoPE and NoPE,
both weight layouts, the served per-rank heads (32 q + 4 kv) and the checkpoint TEP8 heads
(16 + 2), and heads of special values.
"""

import importlib

import pytest
import torch

MOD = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep"
)


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("0", False), ("", False), ("1", True), ("true", True), ("ON", True), ("no", False)],
)
def test_switch_parse(monkeypatch, value, expected):
    pp = importlib.import_module(MOD)
    if value is None:
        monkeypatch.delenv("TRTLLM_MSO_ENABLE_CTX4_PREP_VEC", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_MSO_ENABLE_CTX4_PREP_VEC", value)
    assert pp._rows_vec_enabled() is expected


def _sm103():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)


def _bits(t):
    return t.view(torch.int16)


def _projection(rows, width, special, gen):
    x = torch.randn(rows, width, generator=gen) * torch.exp(3 * torch.randn(rows, 1, generator=gen))
    if special:
        heads = x.view(rows, width // 64, 64)
        heads[0::7, 0] = 0.0  # whole-zero heads
        heads[1::7, 1] = 1e-39  # subnormal heads
        heads[2::7, 2] = 1.17549435e-38  # smallest normal
        heads[3::7, 3, 5] = float("inf")
        heads[4::7, 4, 9] = float("nan")
        heads[5::7, 5] = 3.3e38  # near the BF16 maximum
        heads[6::7, 6] = 1e-20  # norm output fine, RoPE products subnormal
    return x.to(torch.bfloat16).cuda()


def _launch(pp, kernel, qkv, cos_sin, positions, heads, kv_heads, permuted):
    rows = qkv.shape[0]
    qkv_out = torch.empty(rows, (heads + 2 * kv_heads) * 64, dtype=qkv.dtype, device=qkv.device)
    gate_out = torch.empty(rows, heads * 64, dtype=qkv.dtype, device=qkv.device)
    kernel[(-(-rows // pp.ROW_BLOCK), heads + kv_heads)](
        qkv, qkv_out, gate_out, cos_sin, positions, rows, qkv.stride(0), qkv_out.stride(0),
        gate_out.stride(0), 1e-5, pp.CKPT_Q_SCALE, NUM_Q_HEADS=heads, NUM_KV_HEADS=kv_heads,
        QKV_PERMUTED=permuted, APPLY_ROPE=cos_sin is not None, LAUNCH_WITH_PDL=False,
        BLOCK=pp.ROW_BLOCK, num_warps=4,
    )
    return qkv_out, gate_out


@pytest.mark.skipif(not _sm103(), reason="sm_103 device test")
@pytest.mark.parametrize("rows", [513, 777, 1024, 2047, 4096, 8183, 8192, 32768])
@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("heads, kv_heads", [(32, 4), (16, 2)])
@pytest.mark.parametrize("special", [False, True])
def test_vec_bitwise(rows, rope, permuted, heads, kv_heads, special):
    pp = importlib.import_module(MOD)
    gen = torch.Generator().manual_seed(rows * 31 + heads + 7 * rope + 3 * permuted + special)
    qkv = _projection(rows, 2 * (heads + kv_heads) * 64, special, gen)
    cos_sin = positions = None
    if rope:
        inv = 1.0 / (pp.CKPT_ROPE_THETA ** (torch.arange(0, 32, dtype=torch.float64) / 32))
        ang = torch.arange(1 << 20, dtype=torch.float64)[:, None] * inv[None]
        cos_sin = torch.stack([ang.cos(), ang.sin()], 1).float().contiguous().cuda()
        positions = torch.randint(0, 1 << 20, (rows,), generator=gen, dtype=torch.int32).cuda()
    ref = _launch(pp, pp._preattn_prep_rows_kernel, qkv, cos_sin, positions, heads, kv_heads, permuted)
    got = _launch(pp, pp._preattn_prep_rows_vec_kernel, qkv, cos_sin, positions, heads, kv_heads,
                  permuted)
    for r, g, name in zip(ref, got, ("qkv_out", "gate_out")):
        diff = (_bits(r) != _bits(g)).sum().item()
        assert diff == 0, f"{name}: {diff} elements differ"


@pytest.mark.skipif(not _sm103(), reason="sm_103 device test")
def test_op_dispatches_vec(monkeypatch):
    """With the switch read as on, the op above ROW_KERNEL_MIN_ROWS launches the vec kernel."""
    pp = importlib.import_module(MOD)
    monkeypatch.setattr(pp, "ROWS_VEC", True)
    launched = []
    real = pp._preattn_prep_rows_vec_kernel

    class Spy:
        def __getitem__(self, grid):
            launched.append(grid)
            return real[grid]

    monkeypatch.setattr(pp, "_preattn_prep_rows_vec_kernel", Spy())
    qkv = _projection(1024, 2 * 36 * 64, False, torch.Generator().manual_seed(1))
    pp.preattn_prep(qkv, None, None, 32, 4, 1e-5, pp.CKPT_Q_SCALE, False)
    assert launched
