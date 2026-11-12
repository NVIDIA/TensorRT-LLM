# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 gdn_prefill_onepass entry.

Two gates per cell:

1. Bitwise against the in-tree chain it replaces, on a copy of the conv pool:
   `extract_transpose_prefill_slice` -> `causal_conv1d_fn` (the C++ channel-major conv)
   -> `fused_gdn_post_conv` -- q, k, v, g, beta and the whole conv pool. This holds for
   the Triton build whose layout choice `REDUCE` replays.
2. Against the entry's float64 reference: q, k, v in torch's default bf16 band (the conv
   is an fp32 FMA chain and a SiLU through `ex2.approx`, ~2^-21 relative before the one
   bf16 rounding, which flips now and then); g at rtol 2^-16, atol
   2^-19 * max(exp(A_log)): the kernel's `log(1 + exp(x))` and `exp` go through MUFU
   approximations (~2^-22 each, plus `|x| * 2^-24` from scaling the argument) and
   `1 + exp(x)` rounds at 2^-24 absolute, all under 2^-19 on softplus, times
   `exp(A_log)` <= 16; beta = sigmoid(b) at rtol 2^-16, atol 2^-22 (one approximate
   exp and a full-range divide); the conv pool bitwise (a copy of inputs).

Slots the call did not name stay bitwise; the coalesced cell's other layers too. Then
the guard on shared slots.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.gdn_prefill_onepass import (  # noqa: E501
    CONV_DIM,
    HEAD,
    HV,
    H,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.ssm.gdn_prefill_onepass import (  # noqa: E501
    gdn_prefill_onepass as op,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the qwen3.8-27b sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
QKVZ_DIM = CONV_DIM + HV * HEAD
_SLOTS, _LAYERS = 24, 48


def _build(spec, seed):
    """`(args, extras)`: `__call__`'s positional arguments; x the `[P, 10240]` column view
    of a `[P, 16384]` in_proj_qkvz output, a/b the halves of a `[P, 96]` in_proj_ba
    output, the conv pool compact or as the cache manager v2 coalesces it."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    seqlens = spec["seqlens"]
    n, total = len(seqlens), sum(seqlens)
    qkvz = (torch.randn(total, QKVZ_DIM, generator=gen, device=DEV) * 1.5).to(torch.bfloat16)
    ba = torch.randn(total, 2 * HV, generator=gen, device=DEV).to(torch.bfloat16)
    weight = (torch.randn(CONV_DIM, 4, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    if spec.get("pool") == "coalesced":
        backing = torch.randn(_SLOTS, _LAYERS, CONV_DIM, 3, generator=gen, device=DEV)
        backing = backing.to(torch.bfloat16)
        conv = backing[:, 0]
    else:
        backing = torch.randn(_SLOTS, CONV_DIM, 3, generator=gen, device=DEV).to(torch.bfloat16)
        conv = backing
    A_log = torch.log(torch.rand(HV, generator=gen, device=DEV) * 16)
    dt_bias = torch.rand(HV, generator=gen, device=DEV) * 2 - 1
    qsl = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32, device=DEV
    )
    slots = torch.randperm(_SLOTS, generator=gen, device=DEV)[:n].to(torch.int32)
    init = torch.tensor(spec.get("init", (True,) * n), dtype=torch.bool, device=DEV)
    args = [qkvz[:, :CONV_DIM], ba[:, HV:], ba[:, :HV], weight, conv, slots, init, qsl, A_log,
            dt_bias]  # fmt: skip
    return args, dict(backing=backing)


def _chain(args):
    """The in-tree chain, on a copy of the conv pool: `(q, k, v, g, beta, conv_after)`."""
    x, a, b, weight, conv, slots, init, qsl, A_log, dt_bias = args
    conv_copy = conv.clone()
    xt = extract_transpose_prefill_slice(x, x.shape[0], 0, x.shape[1])
    xt = causal_conv1d_fn(xt, weight, None, activation="silu", conv_states=conv_copy,
                          has_initial_state=init, cache_indices=slots, query_start_loc=qsl)  # fmt: skip
    return (*fused_gdn_post_conv(xt, None, a, b, A_log, dt_bias, H, HEAD, HV, HEAD), conv_copy)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, extras = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    conv = args[4]
    conv_before = conv.clone()
    backing_before = extras["backing"].clone()
    chain = _chain(args)
    ref = op.reference(*args)

    out = op(*args)

    total = sum(cell.spec["seqlens"])
    q, k, v, g, beta = out
    assert q.shape == k.shape == (1, total, H, HEAD) and v.shape == (1, total, HV, HEAD)
    assert g.shape == beta.shape == (1, total, HV) and g.dtype == beta.dtype == torch.float32
    assert all(t.is_contiguous() for t in out)
    # Gate 1: bitwise the in-tree chain (q, k, v, g, beta, then the conv pool).
    for name, got, want in zip(("q", "k", "v", "g", "beta"), out, chain[:5]):
        assert torch.equal(_bits(got), _bits(want)), f"{name} differs from the in-tree chain"
    assert torch.equal(conv, chain[5]), "the conv pool differs from causal_conv1d_fn's"
    # Gate 2: the reference.
    for got, want in zip((q, k, v), ref[:3]):
        torch.testing.assert_close(got, want)
    a_scale = torch.exp(args[8]).max().item()
    torch.testing.assert_close(g, ref[3], rtol=2.0**-16, atol=2.0**-19 * a_scale)
    torch.testing.assert_close(beta, ref[4], rtol=2.0**-16, atol=2.0**-22)
    assert torch.equal(conv, ref[5]), "final conv states"
    named = set(args[5].tolist())
    for s in range(conv.shape[0]):
        if s not in named:
            assert torch.equal(conv[s], conv_before[s]), f"slot {s} written"
    if cell.spec.get("pool") == "coalesced":
        assert torch.equal(extras["backing"][:, 1:], backing_before[:, 1:]), "another layer"


def test_guard_refuses_shared_slots() -> None:
    """Two sequences naming one conv slot race on its final state; refused."""
    args, _ = _build(dict(seqlens=(40, 40)), seed=13)
    args[5] = torch.tensor([3, 3], dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="sharing a conv slot"):
        with validating(op):
            op(*args)
