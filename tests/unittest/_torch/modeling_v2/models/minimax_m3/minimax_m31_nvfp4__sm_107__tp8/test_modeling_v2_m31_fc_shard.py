# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC_SHARD on the sm_107 TP8 target: the drafter's fc as a column shard per rank + an all-gather.

Each 768-column shard computed by ``trtllm::minimax_m31_fc_shard_gemm`` with the pinned algo equals the matching
columns of ``torch.nn.functional.linear(x, w)``, for every shard, every row count up to MAX_ROWS, real-scale and
2^+-12 block-scaled inputs, eager and in a CUDA graph; the shards gathered in rank order are the full fc. The wrapper
takes a shard only for the fc it was built from and the checked row counts, and the DSpark drafter's
project_target_hidden takes it only within those rows. Default on; TRTLLM_M31_DISABLE_FC_SHARD=1 turns it off."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.gemm import (  # noqa: E501
    m31_fc_shard,
)

K, N, TP = 30720, 6144, 8
COLS = N // TP


def _op_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    op = m31_fc_shard.fc_shard_op()
    if op is None:
        pytest.skip("the minimax_m31_fc_shard_gemm op is not built")
    return op


@pytest.fixture(scope="module")
def weight():
    g = torch.Generator(device="cpu").manual_seed(2026)
    return (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16).cuda()


def _inputs(rows, stress):
    g = torch.Generator(device="cpu").manual_seed(100 + rows)
    x = torch.randn(rows, K, generator=g)
    if stress:
        e = torch.randint(-12, 13, (rows, K // 128), generator=g).float()
        x = x * torch.pow(2.0, e).repeat_interleave(128, dim=1)
    return x.to(torch.bfloat16).cuda()


@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("rows", list(range(1, m31_fc_shard.MAX_ROWS + 1)))
def test_every_shard_is_the_full_fc_columns(weight, rows, stress):
    op = _op_or_skip()
    x = _inputs(rows, stress)
    full = F.linear(x, weight)
    cfg = list(m31_fc_shard.PINNED_CFG)
    parts = []
    for r in range(TP):
        got = op(x, weight[r * COLS : (r + 1) * COLS].contiguous(), cfg)
        assert torch.equal(got, full[:, r * COLS : (r + 1) * COLS]), r
        parts.append(got)
    assert torch.equal(torch.cat(parts, dim=-1), full)


def test_the_shard_in_a_cuda_graph(weight):
    op = _op_or_skip()
    x = _inputs(8, True)
    w3 = weight[3 * COLS : 4 * COLS].contiguous()
    cfg = list(m31_fc_shard.PINNED_CFG)
    out = torch.empty(8, COLS, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out.copy_(op(x, w3, cfg))  # set the algo up outside the capture
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(op(x, w3, cfg))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, F.linear(x, weight)[:, 3 * COLS : 4 * COLS])


def test_the_wrapper_shards_checks_and_gathers(weight, monkeypatch):
    _op_or_skip()
    fc = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
    fc.weight.data.copy_(weight)
    x = _inputs(7, False)
    shards = []
    for r in range(TP):
        shard, why = m31_fc_shard.FcShard._check(fc, TP, r, SimpleNamespace(tp_size=TP, tp_rank=r))
        assert why is None and shard is not None, why
        shards.append(shard)
    s0 = shards[0]
    assert s0.applies(fc, 7) and s0.applies(fc, m31_fc_shard.MAX_ROWS)
    assert not s0.applies(fc, m31_fc_shard.MAX_ROWS + 1) and not s0.applies(fc, 0)
    other = torch.nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
    assert not s0.applies(other, 7)
    parts = [s.partial(x) for s in shards]
    assert all(p.shape == (7, COLS) for p in parts)
    # The all-gather concatenates the ranks' tensors in rank order along dim.
    monkeypatch.setattr(
        m31_fc_shard, "allgather", lambda y, mapping, dim: torch.cat(parts, dim=dim)
    )
    assert torch.equal(s0.gather(parts[0]), fc(x))


def test_a_non_applicable_fc_keeps_the_stock_path():
    _op_or_skip()
    fc = torch.nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16)
    shard, why = m31_fc_shard.FcShard._check(fc, TP, 0, SimpleNamespace(tp_size=TP, tp_rank=0))
    assert shard is None and "do not apply" in why


class _HalfShard:
    """A stand-in shard: the fc as two column halves, concatenated (the all-gather of a 2-rank group)."""

    def __init__(self, fc, rows):
        self.fc, self.rows, self.calls = fc, rows, 0

    def applies(self, fc, rows):
        return fc is self.fc and 0 < rows <= self.rows

    def __call__(self, x):
        self.calls += 1
        h = self.fc.weight.shape[0] // 2
        return torch.cat([F.linear(x, self.fc.weight[:h]), F.linear(x, self.fc.weight[h:])], dim=-1)


@pytest.mark.parametrize("rows", [1, 8, 9, 24])
def test_project_target_hidden_takes_the_shard_within_its_rows(rows):
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    g = torch.Generator().manual_seed(11)
    fc = torch.nn.Linear(96, 32, bias=False, dtype=torch.bfloat16)
    norm = torch.nn.RMSNorm(32, eps=1e-6, dtype=torch.bfloat16)
    with torch.no_grad():
        fc.weight.copy_(torch.randn(32, 96, generator=g).bfloat16())
        norm.weight.copy_((1 + 0.1 * torch.randn(32, generator=g)).bfloat16())
    shard = _HalfShard(fc, 8)
    model = SimpleNamespace(fc=fc, hidden_norm=norm, _dflash_fc_split=None, _dflash_fc_shard=shard)
    x = torch.randn(rows, 96, generator=g)
    out = DFlashForCausalLM.project_target_hidden(model, x)
    assert torch.equal(out, norm(fc(x.to(torch.bfloat16))))
    assert shard.calls == (1 if rows <= 8 else 0)


@pytest.mark.parametrize(
    "env, expected", [({}, True), ({"TRTLLM_M31_DISABLE_FC_SHARD": "1"}, False)]
)
def test_fc_shard_default_on_and_kill_switch(monkeypatch, env, expected):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark,
        m31_switches,
    )

    monkeypatch.delenv("TRTLLM_M31_DISABLE_FC_SHARD", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    importlib.reload(m31_switches)
    try:
        assert importlib.reload(dspark).FC_SHARD is expected
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_FC_SHARD", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(dspark)
    assert dspark.FC_SHARD is True
