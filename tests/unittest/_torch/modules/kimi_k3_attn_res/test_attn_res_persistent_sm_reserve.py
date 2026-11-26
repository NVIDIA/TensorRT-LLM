# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The persistent attention-residual op sized to the SMs a step-resident kernel leaves free.

``sm_reserve`` only shrinks the persistent grid: every token is still mixed by one CTA with the
same arithmetic, so the outputs are bit-identical to the full grid.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the C++ trtllm ops)
from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

HIDDEN = 7168
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the persistent attention-residual kernel needs an SM100/SM103 GPU",
)


def _rows(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)


@pytest.mark.parametrize("with_add", [False, True])
@pytest.mark.parametrize("num_tokens, num_snapshots", [(300, 4), (4096, 8)])
def test_sm_reserve_keeps_results(num_tokens: int, num_snapshots: int, with_add: bool):
    layer = _rows(num_tokens, 1, HIDDEN, seed=1)
    addend = _rows(num_tokens, 1, HIDDEN, seed=2) if with_add else None
    block = _rows(num_snapshots, num_tokens, 1, HIDDEN, seed=3)
    weights = [_rows(HIDDEN, seed=s) for s in (4, 5, 6)]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
    ref = op(layer, addend, block, *weights, EPS, EPS)
    for reserve in (16, sms - 1):
        out = op(layer, addend, block, *weights, EPS, EPS, reserve)
        assert torch.equal(out[1], ref[1]), reserve
        if with_add:
            assert torch.equal(out[0], ref[0]), reserve


@pytest.mark.parametrize("reserve_offset", [None, 0, 7])
def test_sm_reserve_out_of_range_is_rejected(reserve_offset):
    """Like ``cute_dsl_utils.set_sm_reserve``: negative or >= the SM count fails loudly."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    reserve = -1 if reserve_offset is None else sms + reserve_offset
    layer = _rows(300, 1, HIDDEN, seed=1)
    block = _rows(4, 300, 1, HIDDEN, seed=3)
    weights = [_rows(HIDDEN, seed=s) for s in (4, 5, 6)]
    with pytest.raises(RuntimeError, match="sm_reserve"):
        torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd(
            layer, None, block, *weights, EPS, EPS, reserve
        )


def test_model_passes_the_sm_reserve(monkeypatch):
    """The model's persistent mixes leave the SMs reserved for a step-resident kernel
    (``cute_dsl_utils.set_sm_reserve``, which ``DwdpManager`` calls) out of the grid."""
    from torch import nn

    from tensorrt_llm._torch.cute_dsl_utils import set_sm_reserve
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    op = torch.ops.trtllm.attn_res_add_rmsnorm_persistent_fwd
    reserves = []

    def spy(*args):
        reserves.append(args[-1])
        return op(*args)

    monkeypatch.setattr(torch.ops.trtllm, "attn_res_add_rmsnorm_persistent_fwd", spy)
    proj = nn.Linear(HIDDEN, 1, bias=False, dtype=torch.bfloat16, device="cuda")
    norm = mkl.KimiK3RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    out_norm = RMSNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    prefix = _rows(300, HIDDEN, seed=1)
    block = _rows(4, 300, HIDDEN, seed=2)
    set_sm_reserve(16)
    try:
        mkl._attn_res_persistent_kernel(prefix, None, block, proj, norm, out_norm)
    finally:
        set_sm_reserve(0)
    mkl._attn_res_persistent_kernel(prefix, None, block, proj, norm, out_norm)
    assert reserves == [16, 0]
