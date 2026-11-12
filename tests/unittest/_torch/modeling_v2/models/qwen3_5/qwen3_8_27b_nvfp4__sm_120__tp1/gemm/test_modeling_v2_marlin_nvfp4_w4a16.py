# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the marlin_nvfp4_w4a16 entry.

Cells against the reference (Marlin layout unpacked, exact dequantization with
subnormal-scale blocks zeroed, fp32 product) within the band derived in `_compare`.
The FP8-copy cells use the decode recipe on FP8 weights with tiny rows (subnormal
block scales) and zero blocks; the lm_head cells a random NVFP4 checkpoint repacked
as stored. Beyond the cells: a PDL producer that writes the activation late under
eager and graph replay (the in-proj launch tables prefetch weights before their
wait), graph replay against eager with poisoned outputs, and the layout guard.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.marlin_nvfp4_w4a16 import (  # noqa: E501
    marlin_nvfp4_w4a16 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.quantization import (  # noqa: E501
    _nvfp4_recipes as recipes,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import capture, late_producer, seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
LM_HEAD = (248320, 5120)


def _compare(out: torch.Tensor, ref: torch.Tensor, workspace_mode: int) -> None:
    """Same dequantized operands, BF16 x BF16 products exact in fp32; what differs is
    the summation order and, in mode 0, the split-K partials each rounded to BF16 as
    they chain through the output. A partial is at most the row's scale, so each chain
    link adds at most half a BF16 ulp of the row maximum (one unit of the band's
    2**-8); these launches split K at most four ways, plus the final rounding: 8
    units element-wise in mode 0, 4 in mode 2, where the partials reduce in fp32 and only the last rounding
    is BF16. A wrong tile, scale or permutation is off by the values themselves."""
    element, rms = (8.0, 4.0) if workspace_mode == 0 else (4.0, 2.0)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert_within_ulp(out, ref, element_ulp=element, rms_ulp=rms)


def _fp8_copy(n: int, k: int, seed: int) -> recipes.MarlinNvfp4Weight:
    """The decode copy of an FP8 projection: rows scaled U(0.25, 1.75), 64 rows 1e-4
    smaller (subnormal block scales, which Marlin flushes), all-zero blocks."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=DEV) * 1.5 + 0.25)
    w[:64] *= 1e-4
    w[64:72, :1024] = 0
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    del w
    return recipes.marlin_pack(recipes.fp8_to_nvfp4_values(w8, s.reshape(())))


def _lm_head_copy(seed: int) -> recipes.MarlinNvfp4Weight:
    v, k = LM_HEAD
    g = torch.Generator(device=DEV).manual_seed(seed)
    packed = torch.randint(0, 256, (v, k // 2), generator=g, device=DEV, dtype=torch.int32).to(
        torch.uint8
    )
    sf = torch.randint(0x20, 0x48, (v, k // 16), generator=g, device=DEV, dtype=torch.int32)
    return recipes.lm_head_marlin(
        packed, sf.to(torch.uint8).view(torch.float8_e4m3fn), torch.tensor(2.9e-4, device=DEV)
    )


_COPIES: dict = {}


def _copy(n: int, k: int) -> recipes.MarlinNvfp4Weight:
    """One copy per shape, shared by its cells (building the lm_head copy is the slow part)."""
    if (n, k) not in _COPIES:
        _COPIES[(n, k)] = _lm_head_copy(11) if (n, k) == LM_HEAD else _fp8_copy(n, k, n + k)
    return _COPIES[(n, k)]


def _x(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(m, k, generator=g, device=DEV) * 2.0).to(torch.bfloat16)


def _args(c: recipes.MarlinNvfp4Weight, x: torch.Tensor, mode: int, hint: int):
    return (x, c.weight, c.weight_scale, c.weight_global_scale, c.size_n, c.size_k, mode, hint)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    c = _copy(s["n"], s["k"])
    x = _x(s["m"], s["k"], seed_of(cell.why))
    args = _args(c, x, s["workspace_mode"], s["launch_hint"])
    out = op(*args)
    assert out.shape == (s["m"], s["n"])
    _compare(out, op.reference(*args), s["workspace_mode"])


@pytest.mark.parametrize(
    "n,k,m,mode",
    [(16384, 5120, 8, 0), (14336, 5120, 16, 0), (14336, 5120, 32, 2), (5120, 6144, 8, 0)],
    ids=["qkvz-fullk", "qkv-m16", "qkv-m32", "out_proj-fullk"],
)
def test_input_written_late_by_a_pdl_producer(n, k, m, mode) -> None:
    """The in-proj launches of `launch_hint=1` stream weights into L2 before
    `griddepcontrol.wait`. A producer that releases its dependents at once and writes
    `x` 30 us later must still be waited for: a read before the wait sees the NaN fill."""
    c = _copy(n, k)
    src = _x(m, k, seed=n + m)
    want = op(*_args(c, src, mode, 1)).clone()
    buf = torch.empty_like(src)

    def step():
        buf.fill_(float("nan"))
        late_producer(src, buf)
        return op(*_args(c, buf, mode, 1))

    for _ in range(5):
        assert torch.equal(step(), want)
    graph, out = capture(step)
    for _ in range(10):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, want)


def test_graph_replay_matches_eager() -> None:
    """Two decode copies back to back in one graph (the persistent lock workspace is
    shared per stream): every replay, with a new activation and poisoned outputs,
    equals eager."""
    a, b = _copy(16384, 5120), _copy(14336, 5120)
    x = _x(9, 5120, seed=3)

    def step():
        return op(*_args(a, x, 0, 1)), op(*_args(b, x, 0, 1))

    graph, outs = capture(step)
    for trial in range(3):
        x.copy_(_x(9, 5120, seed=100 + trial))
        for o in outs:
            o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        want = step()
        assert all(torch.equal(o, w) for o, w in zip(outs, want))


def test_guard_refuses_a_strided_activation() -> None:
    c = _copy(5120, 6144)
    x = _x(4, 2 * 6144, seed=5)[:, ::2]
    with validating(op), pytest.raises(AssertionError, match="dense row-major"):
        op(*_args(c, x, 0, 1))


def test_guard_refuses_sizes_that_do_not_match_the_weight() -> None:
    c = _copy(5120, 6144)
    x = _x(4, 6144, seed=6)
    with validating(op), pytest.raises(AssertionError, match="Marlin"):
        op(x, c.weight, c.weight_scale, c.weight_global_scale, 2560, 6144, 0, 1)
