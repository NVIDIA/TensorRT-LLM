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
"""CPU tests for the copy-engine all-gather of token-sharded TP (no GPU): the region layout
against a pure-Python reference, the padding drop against the helper's, the consumer check, the
fused op's fake under FakeTensorMode and torch.export, and the per-group state's probe fallback
and registry lifecycle on a gloo group (spawned ranks rendezvous through a file, no port).
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import itertools
import math
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from token_sharded_tp_test_utils import simulated_helper

from tensorrt_llm._torch.distributed import ce_protocol
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.parallel import token_sharded_ce_gather as tsc
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_ce_gather import (
    DEFAULT_ALIGN,
    CeGatherState,
    RegionLayout,
    drop_padding,
    get_state,
    mn_major_scales,
    packed_weight_scale_cols,
    register_state,
    release_state,
    scale_span_elems,
    validate_fp8_block_consumer,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    TokenShardPlan,
    fp8_scale_cols,
    fp8_scales_mn_major,
)
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_OP = torch.ops.trtllm.token_sharded_fp8_ce_gather_gemm


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Region layout vs a pure-Python reference
# =============================================================================


def _layout_ref(m: int, k: int, tp: int, align: int = 256) -> dict[str, int]:
    """Plain-int reference of the slot carving: payload, scale storage, region, slot."""
    p = (math.ceil(k / 128) + 3) // 4
    lead = math.ceil(m / 4) * 4
    payload = m * k
    scale_off = math.ceil(payload / align) * align
    scale_bytes = p * lead * 4
    region = math.ceil((scale_off + scale_bytes) / align) * align
    return {
        "scale_cols": p,
        "scale_lead": lead,
        "payload_nbytes": payload,
        "scale_offset": scale_off,
        "scale_nbytes": scale_bytes,
        "span_elems": (p - 1) * lead + m,
        "region_nbytes": region,
        "slot_nbytes": tp * region,
    }


_LAYOUT_CASES = list(
    itertools.product([1, 3, 4, 53, 105, 209, 256, 18900], [128, 256, 640, 5120], [2, 3, 4, 8])
)


@pytest.mark.parametrize("m,k,tp", _LAYOUT_CASES)
def test_region_layout_arithmetic(m, k, tp):
    layout = RegionLayout(m, k)
    ref = _layout_ref(m, k, tp)
    assert layout.align == DEFAULT_ALIGN
    for name, want in ref.items():
        got = layout.slot_nbytes(tp) if name == "slot_nbytes" else getattr(layout, name)
        assert got == want, (name, got, want)
    assert layout.scale_cols == fp8_scale_cols(k)
    assert layout.region_nbytes % DEFAULT_ALIGN == 0 and layout.scale_offset % 16 == 0
    assert layout.scale_offset >= layout.payload_nbytes
    assert layout.region_nbytes >= layout.scale_offset + layout.scale_nbytes
    assert layout.span_elems * 4 <= layout.scale_nbytes


def test_region_layout_rejects_invalid():
    with pytest.raises(ValueError, match="rows >= 1"):
        RegionLayout(0, 128)
    with pytest.raises(ValueError, match="multiple of 128"):
        RegionLayout(8, 192)
    with pytest.raises(ValueError, match="power of two"):
        RegionLayout(8, 128, align=24)
    with pytest.raises(ValueError, match="power of two"):
        RegionLayout(8, 128, align=8)


@pytest.mark.parametrize("m,k,tp", [(1, 128, 2), (5, 640, 3), (105, 5120, 4), (209, 256, 8)])
def test_region_views_round_trip(m, k, tp):
    """A push through the producer views of every region is read back through the consumer
    views as the quantizer's (fp8, MN-major scales) pair, with no overlap between regions."""
    layout = RegionLayout(m, k)
    slot = torch.zeros(layout.slot_nbytes(tp), dtype=torch.uint8)
    gen = torch.Generator().manual_seed(m + k + tp)
    cols = layout.scale_cols
    want = []
    for src in range(tp):
        fp8 = torch.randint(0, 256, (m, k), dtype=torch.uint8, generator=gen).view(
            torch.float8_e4m3fn
        )
        rows = torch.randint(0, 2**31 - 1, (m, cols), generator=gen).to(torch.int32)
        scale = fp8_scales_mn_major(rows)  # (m, P) strides (1, pad4(m)), as the quantizer
        assert scale.stride() == (1, layout.scale_lead)
        region = slot[src * layout.region_nbytes : (src + 1) * layout.region_nbytes]
        payload_dst, span_dst = layout.producer_views(region)
        assert (
            payload_dst.numel() == layout.payload_nbytes and span_dst.numel() == layout.span_elems
        )
        payload_dst.copy_(fp8.view(torch.uint8).view(-1))
        span_dst.copy_(scale.as_strided((layout.span_elems,), (1,), scale.storage_offset()))
        want.append((fp8, rows))
    for src in range(tp):
        region = slot[src * layout.region_nbytes : (src + 1) * layout.region_nbytes]
        fp8, scale = layout.consumer_views(region)
        assert fp8.dtype == torch.float8_e4m3fn and fp8.shape == (m, k) and fp8.is_contiguous()
        assert scale.dtype == torch.int32 and scale.shape == (m, cols)
        assert scale.stride() == (1, layout.scale_lead)
        assert torch.equal(fp8.view(torch.uint8), want[src][0].view(torch.uint8))
        assert torch.equal(scale, want[src][1])
    with pytest.raises(ValueError, match="1-D uint8 region"):
        layout.consumer_views(slot[: layout.region_nbytes - 1])
    with pytest.raises(ValueError, match="1-D uint8 region"):
        layout.producer_views(slot.view(torch.int32))


# =============================================================================
# Scale helpers
# =============================================================================


@pytest.mark.parametrize("cols", [1, 2, 10, 27])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 53, 105, 209, 256])
def test_scale_span_and_mn_major(m, cols):
    assert scale_span_elems(m, cols) == (cols - 1) * pad_up(m, 4) + m
    rows = torch.randint(0, 2**31 - 1, (m, cols)).to(torch.int32)
    quantizer_like = fp8_scales_mn_major(rows)
    assert mn_major_scales(quantizer_like) is quantizer_like  # already MN-major: no copy
    for other in (rows, rows.t().contiguous().t()):  # row-major, and strides (1, m)
        fixed = mn_major_scales(other)
        assert fixed.stride() == (1, pad_up(m, 4)) and torch.equal(fixed, rows)
    # The span covers exactly the elements the MN-major view can address.
    span = quantizer_like.as_strided((scale_span_elems(m, cols),), (1,), 0)
    assert span.numel() <= quantizer_like.untyped_storage().nbytes() // 4
    assert span[-1] == quantizer_like[m - 1, cols - 1]


def test_packed_weight_scale_cols():
    assert packed_weight_scale_cols(128) == 1
    assert packed_weight_scale_cols(512) == 1
    assert packed_weight_scale_cols(640) == 2
    assert packed_weight_scale_cols(5120) == 10
    assert packed_weight_scale_cols(5248) == 11


# =============================================================================
# drop_padding == TokenShardedTP._drop_padding
# =============================================================================


@pytest.mark.parametrize("tp", [2, 3, 4, 8])
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_drop_padding_matches_helper(batch, tp):
    for seq in (1, 5, 7, 8, 12, 105, 418):
        plan = TokenShardPlan.build(batch, seq, tp, 0)
        ts = simulated_helper(plan)
        t = torch.randn(plan.padded_rows, 3)
        got = drop_padding(t, plan.batch_size, plan.seq_len, plan.padded_seq_len)
        want = ts._drop_padding(t)
        assert got.shape == (plan.num_tokens, 3) and torch.equal(got, want)
        if not plan.is_padded or plan.batch_size == 1:
            assert got.data_ptr() == t.data_ptr()  # a view, no copy
        # The pad rows are the ones dropped: the kept rows are the per-sample prefixes.
        ref = t.view(batch, plan.padded_seq_len, 3)[:, :seq].reshape(-1, 3)
        assert torch.equal(got, ref)


# =============================================================================
# The consumer check
# =============================================================================


def _fp8_block_linear(k=256, n=128, **kwargs):
    return Linear(
        k,
        n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        **kwargs,
    )


def _packed_weight_scale(n: int, k: int) -> torch.Tensor:
    """The int32 (N, ceil(ceil(K/128)/4)) MN-major layout ``transform_weights`` produces."""
    cols = packed_weight_scale_cols(k)
    return torch.zeros((cols, pad_up(n, 4)), dtype=torch.int32).t()[:n]


def test_validate_fp8_block_consumer():
    lin = _fp8_block_linear(k=5120, n=256)
    assert lin.weight.dtype == torch.float8_e4m3fn
    # Before loading the scale is the checkpoint's float32 grid, which the kernel cannot read:
    # the check is for after post_load_weights (the packed int32 scale).
    assert lin.weight_scale.dtype == torch.float32
    with pytest.raises(ValueError, match="packed int32 .* after loading; got torch.float32"):
        validate_fp8_block_consumer(lin)
    lin.weight_scale = torch.nn.Parameter(_packed_weight_scale(256, 5120), requires_grad=False)
    validate_fp8_block_consumer(lin)
    with pytest.raises(ValueError, match="must be a Linear with the FP8BlockScalesLinearMethod"):
        validate_fp8_block_consumer(Linear(256, 128, bias=False, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="must be a Linear"):
        validate_fp8_block_consumer(None)
    with pytest.raises(ValueError, match="K=192 is not a multiple of 128"):
        validate_fp8_block_consumer(_fp8_block_linear(k=192))
    bad = _fp8_block_linear(k=5120, n=256)
    bad.weight_scale = torch.nn.Parameter(
        _packed_weight_scale(256, 5120).contiguous(), requires_grad=False
    )
    with pytest.raises(ValueError, match="weight_scale must be the packed int32"):
        validate_fp8_block_consumer(bad)  # right shape, row-major strides
    bad.weight_scale = torch.nn.Parameter(
        _packed_weight_scale(256, 5248), requires_grad=False
    )  # right dtype and strides, one packed column too many
    with pytest.raises(ValueError, match=r"packed int32 \(256, 10\)"):
        validate_fp8_block_consumer(bad)
    bad.weight_scale = torch.nn.Parameter(torch.zeros(2, 40, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="got torch.bfloat16"):
        validate_fp8_block_consumer(bad)
    bad.weight = torch.nn.Parameter(torch.zeros(256, 5120, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="weight must be a contiguous float8_e4m3fn"):
        validate_fp8_block_consumer(bad)


# =============================================================================
# The fused op: fake shape and tracing
# =============================================================================


def _op_inputs(m, k, n, device="cpu"):
    x = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=device)
    sf = torch.empty_strided((m, fp8_scale_cols(k)), (1, pad_up(m, 4)), dtype=torch.int32)
    w = torch.zeros(n, k, dtype=torch.float8_e4m3fn, device=device)
    ws = _packed_weight_scale(n, k)
    return x, sf, w, ws


@pytest.mark.parametrize("batch,seq,tp", [(1, 8, 2), (2, 5, 2), (3, 7, 4), (1, 420, 4)])
def test_op_fake_shape(batch, seq, tp):
    from torch._subclasses.fake_tensor import FakeTensorMode

    plan = TokenShardPlan.build(batch, seq, tp, 0)
    k, n = 640, 256
    with FakeTensorMode():
        x, sf, w, ws = _op_inputs(plan.local_rows, k, n)
        out = _OP(x, sf, w, ws, "tp", 0, tp, batch, seq, plan.padded_seq_len)
    assert out.shape == (batch * seq, n) and out.dtype == torch.bfloat16


def test_op_traces_with_torch_export():
    """A graph through the op: the group name and the plan ints are constants of the call."""
    plan = TokenShardPlan.build(2, 5, 2, 0)  # padded: S_pad = 6, m = 6
    k, n = 256, 128

    class Boundary(torch.nn.Module):
        def forward(self, x, sf, w, ws):
            return _OP(x, sf, w, ws, "tp", 0, 2, 2, 5, 6) + 1.0

    args = _op_inputs(plan.local_rows, k, n)
    ep = torch.export.export(Boundary(), args)
    calls = [node for node in ep.graph.nodes if node.target is _OP.default]
    assert len(calls) == 1
    assert list(calls[0].args[4:]) == ["tp", 0, 2, 2, 5, 6]
    assert tuple(calls[0].meta["val"].shape) == (10, n)


def test_op_without_a_registered_state_raises():
    with pytest.raises(KeyError):
        get_state("no-such-group")
    x, sf, w, ws = _op_inputs(4, 256, 128)
    with pytest.raises(RuntimeError, match="no CeGatherState is registered for TP group"):
        _OP(x, sf, w, ws, "no-such-group", 0, 2, 1, 8, 8)


# =============================================================================
# CeGatherState on a gloo group: the probe falls back, the registry lifecycle
# =============================================================================


class _Recorder:
    """Stands in for ``tensorrt_llm.logger.logger`` inside the worker."""

    def __init__(self):
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def info_once(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))


def _state_logic(rank, world_size, device):
    group = dist.group.WORLD
    name = group.group_name
    recorder = _Recorder()
    tsc.logger = ce_protocol.logger = recorder  # the probe's one warning is the base's
    with pytest.raises(ValueError, match="process group; got None"):
        CeGatherState(None, name)
    with pytest.raises(ValueError, match="timeout_ms >= 0"):
        CeGatherState(group, name, timeout_ms=-1)
    state = CeGatherState(group, name)
    assert state.tp_size == world_size and state.rank == rank
    assert not state.effective and state.reason == "not probed yet"
    assert register_state(state) is state
    assert register_state(CeGatherState(group, name)) is state  # two TPs over one group share
    assert get_state(name) is state
    state.note_consumer(5120)
    state.note_consumer(640)
    with pytest.raises(ValueError, match="positive multiple of 128"):
        state.note_consumer(100)
    assert state.consumer_ks == {640, 5120}
    plan = TokenShardPlan.build(1, 8, world_size, rank)
    state.prepare(plan)  # probes: a gloo group is not a CUDA NCCL group
    assert not state.effective and "NCCL" in state.reason
    assert state.transport is None and state.pool is None
    assert len(recorder.warnings) == 1 and state.reason in recorder.warnings[0]
    state.prepare(plan)  # probed once: no second warning, still a no-op
    assert len(recorder.warnings) == 1 and not recorder.infos
    assert [state.next_slot() for _ in range(3)] == [0, 1, 0]
    state.begin_forward()
    assert state.next_slot() == 0
    # The op refuses (rank-symmetrically) when the state is registered but not effective.
    x, sf, w, ws = _op_inputs(plan.local_rows, 256, 128)
    with pytest.raises(RuntimeError, match="not effective"):
        _OP(x, sf, w, ws, name, rank, world_size, 1, 8, plan.padded_seq_len)

    # A resync disagreement raised by the transport (CeResyncError: the ranks' boundary counts
    # differ after a failed boundary) takes the group off the protocol, CeState.disable:
    # effective False, the reason, the error propagated (the state's one warning was already
    # given by the probe); the adapters' next call reads effective and gathers with NCCL.
    # Nothing was opened, so reset() must not run.
    def _raise_resync(*args):
        raise ce_protocol.CeResyncError("boundary counts differ across the ranks: [1, 2]")

    def _no_reset():
        raise AssertionError("reset() must not run for a resync disagreement")

    state.effective = True
    state.transport = SimpleNamespace(
        tp_size=world_size, rank=rank, consume_order=(), issue=_raise_resync, reset=_no_reset
    )
    with pytest.raises(ce_protocol.CeResyncError, match="boundary counts differ"):
        _OP(x, sf, w, ws, name, rank, world_size, 1, 8, plan.padded_seq_len)
    assert not state.effective and "boundary counts differ" in state.reason
    assert len(recorder.warnings) == 1  # one warning per state: the probe's
    state.transport = None
    release_state(name)
    with pytest.raises(KeyError):
        get_state(name)
    state.close()  # idempotent
    release_state(name)  # no-op without a state
    # A closed state refuses to start a forward (a sibling helper still holding it is told).
    assert not state.effective and state.reason == "closed"
    with pytest.raises(RuntimeError, match="is closed"):
        state.prepare(plan)
    # A fresh state for the same name probes again (and warns again, once).
    fresh = register_state(CeGatherState(group, name))
    assert fresh is not state and get_state(name) is fresh
    fresh.prepare(plan)
    assert len(recorder.warnings) == 2
    fresh.close()


def _worker(rank, world_size, test_fn, init_file):
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        test_fn(rank, world_size, torch.device("cpu"))
        dist.barrier()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_state_on_gloo(world_size, tmp_path):
    init_file = str(tmp_path / "rendezvous")
    mp.spawn(_worker, args=(world_size, _state_logic, init_file), nprocs=world_size, join=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.parametrize("batch,seq,tp", [(1, 8, 2), (2, 5, 3), (1, 5, 4), (3, 7, 4)])
def test_shared_op_one_exchange_all_readers_before_release(batch, seq, tp, monkeypatch):
    from types import SimpleNamespace

    plan = TokenShardPlan.build(batch, seq, tp, 0)
    m, k = plan.local_rows, 128
    full = (
        (torch.arange(plan.padded_rows) % 7)
        .to(torch.float8_e4m3fn)[:, None]
        .expand(-1, k)
        .contiguous()
    )
    weights = [torch.full((n, k), i + 1).to(torch.float8_e4m3fn) for i, n in enumerate((8, 16, 24))]
    scales = [_packed_weight_scale(w.shape[0], k) for w in weights]
    events = []
    current = []

    class Transport:
        rank, tp_size = 0, tp
        consume_order = list(range(1, tp))

        def issue(self, x, sf, slot):
            events.append("issue")
            assert torch.equal(x.float(), full[:m].float())

        def chunk(self, src, slot):
            assert not current or current[-1] == len(weights)
            current.append(0)
            events.append(("chunk", src))
            return full[src * m : (src + 1) * m], None

        def done(self, slot):
            assert current == [len(weights)] * tp
            events.append("done")

        def reset(self):
            events.append("reset")

    def gemm(x, sf, w, ws, out):
        current[-1] += 1
        out.copy_(x.to(torch.bfloat16) @ w.to(torch.bfloat16).t())

    monkeypatch.setattr(
        tsc, "get_state", lambda group: SimpleNamespace(transport=Transport(), next_slot=lambda: 0)
    )
    monkeypatch.setattr(tsc, "fp8_block_gemm_out", gemm)
    sf = torch.zeros(m, fp8_scale_cols(k), dtype=torch.int32)
    result = tsc.ce_gather_gemms_impl(
        full[:m], sf, weights, scales, "cpu", 0, tp, batch, seq, plan.padded_seq_len
    )
    for out, w in zip(result, weights):
        expected = drop_padding(
            full.to(torch.bfloat16) @ w.to(torch.bfloat16).t(), batch, seq, plan.padded_seq_len
        )
        assert torch.equal(out, expected) and out.is_contiguous()
    assert events == ["issue"] + [("chunk", src) for src in range(tp)] + ["done"]
    # Reject a later incompatible consumer before any rank starts its exchange.
    events.clear()
    bad = torch.zeros(8, k * 2, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="expected this rank"):
        tsc.ce_gather_gemms_impl(
            full[:m],
            sf,
            [weights[0], bad],
            scales[:2],
            "cpu",
            0,
            tp,
            batch,
            seq,
            plan.padded_seq_len,
        )
    assert events == []

    # A failure after one reader has consumed a chunk must release through reset(), not done().
    current.clear()

    def failing_gemm(x, sf, w, ws, out):
        if len(current) == 2 and current[-1] == 1:
            raise RuntimeError("injected second-consumer failure")
        gemm(x, sf, w, ws, out)

    monkeypatch.setattr(tsc, "fp8_block_gemm_out", failing_gemm)
    with pytest.raises(RuntimeError, match="injected"):
        tsc.ce_gather_gemms_impl(
            full[:m], sf, weights, scales, "cpu", 0, tp, batch, seq, plan.padded_seq_len
        )
    assert events[-1] == "reset" and "done" not in events


def test_shared_op_fake_and_export():
    from torch._subclasses.fake_tensor import FakeTensorMode

    op = torch.ops.trtllm.token_sharded_fp8_ce_gather_gemms

    class Boundary(torch.nn.Module):
        def forward(self, x, sf, w, ws, w2, ws2):
            return op(x, sf, [w, w2], [ws, ws2], "tp", 0, 2, 2, 5, 6)

    with FakeTensorMode():
        x, sf, w, ws = _op_inputs(6, 256, 128)
        _, _, w2, ws2 = _op_inputs(6, 256, 256)
        outputs = Boundary()(x, sf, w, ws, w2, ws2)
    assert [tuple(t.shape) for t in outputs] == [(10, 128), (10, 256)]
    args = (*_op_inputs(6, 256, 128), *_op_inputs(6, 256, 256)[2:])
    exported = torch.export.export(Boundary(), args)
    assert sum(node.target is op.default for node in exported.graph.nodes) == 1
