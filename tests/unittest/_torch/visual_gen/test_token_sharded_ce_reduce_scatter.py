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
"""CPU tests for the copy-engine reduce-scatter of token-sharded TP (no GPU, no Triton): the
bf16 region layout against a pure-Python reference and its ``[tp, m, N]`` partials view, the
fixed-order reduce contract through the torch-chain reference (and the wrapper's CPU fallback),
the row-consumer check, the adapters' route around main's FP8 block-scale low-M GEMM, the
fused op's fake under FakeTensorMode and torch.export, the
transport's signal protocol and its recovery bookkeeping on a fake pool and fake streams
(every rank in one process; a scheduler that fails where the pool would trap), and the
per-group state's probe fallback and registry lifecycle on a gloo group (spawned ranks
rendezvous through a file, no port).
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import contextlib
import itertools
import math
import sys
import traceback
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from token_sharded_tp_test_utils import simulated_helper

from tensorrt_llm._torch.distributed import ce_protocol
from tensorrt_llm._torch.distributed.symm_mem_pool import PoolLayout
from tensorrt_llm._torch.modules.linear import (
    Fp8BlockScaleLowMGemmRange,
    Linear,
    TensorParallelMode,
)
from tensorrt_llm._torch.visual_gen.parallel import token_sharded_ce_reduce_scatter as tsr
from tensorrt_llm._torch.visual_gen.parallel import token_sharded_modules, token_sharded_tp
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_ce_gather import packed_weight_scale_cols
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_ce_reduce_scatter import (
    DEFAULT_ALIGN,
    SLOTS,
    Bf16RowsRegionLayout,
    CeReduceScatter,
    CeReduceScatterState,
    fixed_order_reduce_bf16,
    get_rs_state,
    reference_fixed_order_reduce,
    register_rs_state,
    release_rs_state,
    validate_fp8_block_row_consumer,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import TokenShardPlan, fp8_scale_cols
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_OP = torch.ops.trtllm.token_sharded_fp8_ce_gemm_reduce_scatter


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Region layout vs a pure-Python reference
# =============================================================================


def _layout_ref(m: int, n: int, tp: int, align: int = 256) -> dict[str, int]:
    """Plain-int reference of the slot carving: payload, region, source stride, slot."""
    payload = 2 * m * n
    region = math.ceil(payload / align) * align
    return {
        "payload_nbytes": payload,
        "region_nbytes": region,
        "src_stride": region // 2,
        "slot_nbytes": tp * region,
    }


_LAYOUT_CASES = list(
    itertools.product(
        [1, 3, 4, 53, 105, 209, 256, 18900, 37800], [8, 128, 5120, 5121], [2, 3, 4, 8]
    )
)


@pytest.mark.parametrize("m,n,tp", _LAYOUT_CASES)
def test_bf16_rows_region_layout_arithmetic(m, n, tp):
    layout = Bf16RowsRegionLayout(m, n)
    ref = _layout_ref(m, n, tp)
    assert layout.align == DEFAULT_ALIGN
    for name, want in ref.items():
        got = layout.slot_nbytes(tp) if name == "slot_nbytes" else getattr(layout, name)
        assert got == want, (name, got, want)
    assert layout.region_nbytes % DEFAULT_ALIGN == 0
    assert layout.region_nbytes >= layout.payload_nbytes
    assert layout.region_nbytes - layout.payload_nbytes < DEFAULT_ALIGN
    assert layout.src_stride * 2 == layout.region_nbytes and layout.src_stride % 8 == 0
    # Both measured Wan layouts carve exactly: the source stride is m * N elements.
    if n == 5120 and m in (18900, 37800):
        assert layout.src_stride == m * n


def test_bf16_rows_region_layout_rejects_invalid():
    with pytest.raises(ValueError, match="rows >= 1"):
        Bf16RowsRegionLayout(0, 128)
    with pytest.raises(ValueError, match="cols >= 1"):
        Bf16RowsRegionLayout(8, 0)
    with pytest.raises(ValueError, match="power of two"):
        Bf16RowsRegionLayout(8, 128, align=24)
    with pytest.raises(ValueError, match="power of two"):
        Bf16RowsRegionLayout(8, 128, align=8)


@pytest.mark.parametrize("m,n,tp", [(1, 8, 2), (5, 24, 3), (105, 5120, 4), (209, 136, 8)])
def test_region_views_round_trip(m, n, tp):
    """A push through the producer view of every region is read back through the consumer
    view and through the ``[tp, m, N]`` partials view, with no overlap between regions."""
    layout = Bf16RowsRegionLayout(m, n)
    slot = torch.zeros(layout.slot_nbytes(tp), dtype=torch.uint8)
    gen = torch.Generator().manual_seed(m + n + tp)
    want = []
    for src in range(tp):
        block = torch.randn((m, n), generator=gen).to(torch.bfloat16)
        region = slot[src * layout.region_nbytes : (src + 1) * layout.region_nbytes]
        dst = layout.producer_view(region)
        assert dst.dtype == torch.uint8 and dst.numel() == layout.payload_nbytes
        dst.copy_(block.view(torch.uint8).view(-1))
        want.append(block)
    view = layout.partials_view(slot, tp)
    assert view.dtype == torch.bfloat16 and tuple(view.shape) == (tp, m, n)
    assert view.stride() == (layout.src_stride, n, 1)
    for src in range(tp):
        region = slot[src * layout.region_nbytes : (src + 1) * layout.region_nbytes]
        got = layout.consumer_view(region)
        assert got.dtype == torch.bfloat16 and got.shape == (m, n) and got.is_contiguous()
        assert torch.equal(got, want[src]) and torch.equal(view[src], want[src])
    # The view also works from the first region alone (what the transport hands it).
    view0 = layout.partials_view(slot[: layout.region_nbytes], tp)
    assert view0.stride() == view.stride() and torch.equal(view0, view)
    # Writing through the partials view lands in the regions (the own GEMM's target).
    view[tp - 1].fill_(2.0)
    assert bool((layout.consumer_view(slot[(tp - 1) * layout.region_nbytes :]) == 2.0).all())
    with pytest.raises(ValueError, match="1-D uint8 region"):
        layout.consumer_view(slot[: layout.region_nbytes - 1])
    with pytest.raises(ValueError, match="1-D uint8 region"):
        layout.producer_view(slot.view(torch.int32))
    with pytest.raises(ValueError, match="apart from byte"):
        layout.partials_view(slot, tp + 1)  # the storage holds tp regions only


@pytest.mark.parametrize("m,n,tp", [(8, 640, 4), (5, 24, 3), (105, 5120, 2)])
def test_partials_view_follows_the_pools_region_stride(m, n, tp):
    """Inside a grow-only pool the regions sit the POOL's region stride apart, not the
    layout's packed ``src_stride``: after a larger reservation (a bigger token shape in the
    warm-up, a consumer with a larger N) a boundary's partials view must take that stride to
    read sources 1 .. tp - 1 where the pushes and the own GEMM landed (the transport passes
    ``pool.region_nbytes``)."""
    big = Bf16RowsRegionLayout(4 * m + 3, n + 128)  # the earlier, larger reservation
    pool = PoolLayout.for_request(big.slot_nbytes(tp), 1, tp)
    layout = Bf16RowsRegionLayout(m, n)  # this boundary
    assert pool.region_nbytes > layout.region_nbytes
    buf = torch.zeros(pool.nbytes, dtype=torch.uint8)
    gen = torch.Generator().manual_seed(m + n + tp)
    want = []
    for src in range(tp):  # what push (peer_region) and the own GEMM (region) write
        start, stop = pool.region_span(0, src, 0, layout.region_nbytes)
        block = torch.randn((m, n), generator=gen).to(torch.bfloat16)
        layout.producer_view(buf[start:stop]).copy_(block.view(torch.uint8).view(-1))
        want.append(block)
    start, stop = pool.region_span(0, 0, 0, layout.region_nbytes)
    region0 = buf[start:stop]
    view = layout.partials_view(region0, tp, region_stride_nbytes=pool.region_nbytes)
    assert view.stride() == (pool.region_nbytes // 2, n, 1)
    assert all(torch.equal(view[src], want[src]) for src in range(tp))
    out = torch.empty((m, n), dtype=torch.bfloat16)
    fixed_order_reduce_bf16(view, None, out)
    assert torch.equal(out, reference_fixed_order_reduce(want))
    # The packed stride reads source 0 only from such a pool (the trap this test pins).
    packed = layout.partials_view(region0, tp)
    assert torch.equal(packed[0], want[0]) and not all(
        torch.equal(packed[src], want[src]) for src in range(1, tp)
    )
    with pytest.raises(ValueError, match="region_stride_nbytes must be a multiple of"):
        layout.partials_view(region0, tp, region_stride_nbytes=layout.region_nbytes - 256)
    with pytest.raises(ValueError, match="region_stride_nbytes must be a multiple of"):
        layout.partials_view(region0, tp, region_stride_nbytes=pool.region_nbytes + 2)
    with pytest.raises(ValueError, match="apart from byte"):
        layout.partials_view(region0, tp, region_stride_nbytes=2 * pool.region_nbytes)


# =============================================================================
# The reduce contract: the torch chain reference and the wrapper's CPU fallback
# =============================================================================


def _hand_chain(parts, bias=None):
    acc = parts[0].float()
    if bias is not None:
        acc = (acc + bias.float()).to(torch.bfloat16).float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("tp", [2, 3, 4, 8])
def test_reference_fixed_order_reduce_semantics(tp):
    gen = torch.Generator().manual_seed(tp)
    m, n = 6, 16
    parts = [
        (torch.randn((m, n), generator=gen) * (1.0 + 0.3 * (2 * r / (tp - 1) - 1))).to(
            torch.bfloat16
        )
        for r in range(tp)
    ]
    bias = torch.randn(n, generator=gen).to(torch.bfloat16)
    ref = reference_fixed_order_reduce(parts)
    assert ref.dtype == torch.bfloat16 and ref.shape == (m, n)
    assert torch.equal(ref, _hand_chain(parts))
    assert torch.equal(reference_fixed_order_reduce(parts, bias), _hand_chain(parts, bias))
    # A [tp, m, N] tensor (the slot view) iterates over sources exactly as the list does.
    stacked = torch.stack(parts)
    assert torch.equal(reference_fixed_order_reduce(stacked), ref)
    assert torch.equal(reference_fixed_order_reduce(stacked, bias), _hand_chain(parts, bias))
    # The bias is added to source 0 only, in bf16 (PR #9's rank-0 `output + bias`), before
    # the chain: not to the fp32 accumulator and not to any other source.
    biased0 = [(parts[0] + bias)] + parts[1:]
    assert torch.equal(reference_fixed_order_reduce(parts, bias), _hand_chain(biased0))
    with pytest.raises(ValueError, match="at least one partial"):
        reference_fixed_order_reduce([])


def test_reference_fixed_order_reduce_rounding():
    # Ties round to even: 1 + 2^-8 lies halfway between 1 and 1 + 2^-7 (bf16 ulp at 1 is 2^-7).
    tie = torch.zeros((2, 1, 8), dtype=torch.bfloat16)
    tie[0, 0, :4] = 1.0
    tie[1, 0, :4] = 2.0**-8
    tie[0, 0, 4:] = 1.0 + 2.0**-7
    tie[1, 0, 4:] = 2.0**-8
    r = reference_fixed_order_reduce(tie)
    assert bool((r[0, :4] == 1.0).all()) and bool((r[0, 4:] == 1.0 + 2.0**-6).all())
    # Sequential != tree in general (a 24-binade spread with ties): the fixed order is the
    # contract, which is why the kernel never accumulates in arrival order.
    s = torch.tensor([[[2.0**24]], [[1.0]], [[1.0]], [[-(2.0**24)]]], dtype=torch.bfloat16)
    seq = reference_fixed_order_reduce(s)
    tree = ((s[0].float() + s[1].float()) + (s[2].float() + s[3].float())).to(torch.bfloat16)
    assert float(seq) == 0.0 and float(tree) == 1.0
    # One rounding: the fp32 chain is exact for small-spread inputs where a bf16 ring is not.
    ring_in = torch.tensor([[[1.0, 1.0]], [[2.0**-8, 2.0**-8]], [[2.0**-8, 2.0**-8]]]).to(
        torch.bfloat16
    )
    one_rounding = reference_fixed_order_reduce(ring_in)
    ring = ((ring_in[0] + ring_in[1]) + ring_in[2]).to(torch.bfloat16)  # two bf16 roundings
    assert bool((one_rounding == 1.0 + 2.0**-7).all()) and bool((ring == 1.0).all())


@pytest.mark.parametrize("tp,m,n", [(2, 3, 8), (4, 6, 16), (8, 5, 24)])
def test_fixed_order_reduce_bf16_cpu_fallback(tp, m, n):
    """On CPU the wrapper runs the chain (no Triton): bitwise the reference, strided input."""
    layout = Bf16RowsRegionLayout(m, n)
    slot = torch.zeros(layout.slot_nbytes(tp), dtype=torch.uint8)
    view = layout.partials_view(slot, tp)
    gen = torch.Generator().manual_seed(100 + tp)
    for src in range(tp):
        view[src].copy_(torch.randn((m, n), generator=gen) * (0.5 + src))
    bias = torch.randn(n, generator=gen).to(torch.bfloat16)
    out = torch.empty((m, n), dtype=torch.bfloat16)
    assert fixed_order_reduce_bf16(view, None, out) is out
    assert torch.equal(out, reference_fixed_order_reduce(view))
    fixed_order_reduce_bf16(view, bias, out)
    assert torch.equal(out, reference_fixed_order_reduce(view, bias))
    with pytest.raises(ValueError, match=r"partials must be bf16 \[tp, m, N\]"):
        fixed_order_reduce_bf16(view[0], None, out)
    with pytest.raises(ValueError, match=r"strides \(S, N, 1\)"):
        fixed_order_reduce_bf16(view.transpose(1, 2), None, out)
    with pytest.raises(ValueError, match="out must be a contiguous bf16"):
        fixed_order_reduce_bf16(view, None, out.float())
    with pytest.raises(ValueError, match="out must be a contiguous bf16"):
        fixed_order_reduce_bf16(view, None, out.t())
    with pytest.raises(ValueError, match=r"bias must be bf16 \["):
        fixed_order_reduce_bf16(view, bias.float(), out)
    with pytest.raises(ValueError, match=r"bias must be bf16 \["):
        fixed_order_reduce_bf16(view, bias[:-1], out)


def test_reduce_kernel_is_built_lazily():
    """The module imports without Triton (the import sits inside ``_reduce_kernel``); where
    Triton is present the kernel is built once and cached."""
    pytest.importorskip("triton")
    tsr._reduce_kernel.cache_clear()
    kernel = tsr._reduce_kernel()
    assert kernel is tsr._reduce_kernel()
    info = tsr._reduce_kernel.cache_info()
    assert info.misses == 1 and info.hits == 1


# =============================================================================
# The row-consumer check
# =============================================================================


def _fp8_block_linear(k=256, n=128, bias=False, **kwargs):
    return Linear(
        k,
        n,
        bias=bias,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        **kwargs,
    )


def _packed_weight_scale(n: int, k: int) -> torch.Tensor:
    """The int32 (N, ceil(ceil(K/128)/4)) MN-major layout ``transform_weights`` produces."""
    cols = packed_weight_scale_cols(k)
    return torch.zeros((cols, pad_up(n, 4)), dtype=torch.int32).t()[:n]


def test_validate_fp8_block_row_consumer():
    lin = _fp8_block_linear(k=1280, n=5120, bias=True)
    assert lin.weight.dtype == torch.float8_e4m3fn and lin.bias.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="copy-engine reduce-scatter: .* packed int32"):
        validate_fp8_block_row_consumer(lin)  # the checkpoint's float32 scale, before loading
    lin.weight_scale = torch.nn.Parameter(_packed_weight_scale(5120, 1280), requires_grad=False)
    validate_fp8_block_row_consumer(lin)
    no_bias = _fp8_block_linear(k=3456, n=5120)
    no_bias.weight_scale = torch.nn.Parameter(_packed_weight_scale(5120, 3456), requires_grad=False)
    validate_fp8_block_row_consumer(no_bias)
    with pytest.raises(ValueError, match="must be a Linear with the FP8BlockScalesLinearMethod"):
        validate_fp8_block_row_consumer(Linear(256, 128, bias=False, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="copy-engine reduce-scatter"):
        validate_fp8_block_row_consumer(None)
    with pytest.raises(ValueError, match="K=192 is not a multiple of 128"):
        validate_fp8_block_row_consumer(_fp8_block_linear(k=192))
    lin.bias = torch.nn.Parameter(torch.zeros(5120, dtype=torch.float32), requires_grad=False)
    with pytest.raises(ValueError, match=r"bias must be bf16 \[5120\]; got torch.float32"):
        validate_fp8_block_row_consumer(lin)
    lin.bias = torch.nn.Parameter(torch.zeros(5121, dtype=torch.bfloat16), requires_grad=False)
    with pytest.raises(ValueError, match=r"bias must be bf16 \[5120\]"):
        validate_fp8_block_row_consumer(lin)


# =============================================================================
# The route around the FP8 block-scale low-M GEMM
# =============================================================================


def _ce_rs_helper(monkeypatch):
    """A copy-engine reduce-scatter helper whose probe accepted the mode (no group), on a
    simulated SM100 so that the FP8 block-scale rule holds for a CPU-built Linear."""
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    ts = simulated_helper(TokenShardPlan.build(2, 8, 2, 0), reduce_scatter_mode="copy_engine")
    ts._rs_state = SimpleNamespace(effective=True, reason="")
    return ts


def test_low_m_gemm_rows_keep_the_module_gemm(monkeypatch):
    """The stock Linear runs main's FP8 block-scale low-M GEMM, not DeepGEMM, on the rows its
    ``fp8_low_m_gemm`` range serves (``1 <= M <= max_m`` contiguous bf16 rows of the 2-D input
    its prologue forms); the fused op's per-block GEMM is DeepGEMM, so those calls keep the
    module's GEMM and the NCCL reduce-scatter. The default range (max_m 0: an unserved shape or
    device) and a Linear without the attribute never block."""
    monkeypatch.setattr(token_sharded_tp, "MIN_CE_REDUCE_SCATTER_ROWS", 0)
    ts = _ce_rs_helper(monkeypatch)
    lin = _fp8_block_linear(bias=True)
    assert lin.fp8_low_m_gemm == Fp8BlockScaleLowMGemmRange()  # create_weights: max_m 0

    def rows(m, dtype=torch.bfloat16):
        return torch.zeros(m, 256, dtype=dtype)

    assert ts.uses_ce_reduce_scatter(lin, rows(16)) and ts.uses_ce_reduce_scatter(lin, rows(1))
    lin.fp8_low_m_gemm = Fp8BlockScaleLowMGemmRange(max_m=256, kernel_max_m=64)
    for m in range(1, 257):  # the whole range, both kernels of the GEMM
        assert not ts.uses_ce_reduce_scatter(lin, rows(m)), m
    for m in (257, 258, 512, 4096):
        assert ts.uses_ce_reduce_scatter(lin, rows(m)), m
    # The rows the stock prologue flattens a [B, S, K] input to, not its leading dim.
    assert not ts.uses_ce_reduce_scatter(lin, rows(256).view(2, 128, 256))
    assert ts.uses_ce_reduce_scatter(lin, rows(258).view(2, 129, 256))
    # serves() wants a contiguous input (the stock Linear runs DeepGEMM otherwise).
    assert ts.uses_ce_reduce_scatter(lin, torch.zeros(16, 512, dtype=torch.bfloat16)[:, :256])
    # A non-bf16 input is out by the fused op's own rule (it quantizes bf16) before the range
    # is read, in and above the range alike.
    assert not ts.uses_ce_reduce_scatter(lin, rows(16, torch.float32))
    assert not ts.uses_ce_reduce_scatter(lin, rows(512, torch.float32))
    # The other rules still decide first.
    assert not ts.uses_ce_reduce_scatter(lin, rows(512), {"adapter": 1})  # active LoRA
    ts._rs_state = SimpleNamespace(effective=False, reason="probe failed")
    assert not ts.uses_ce_reduce_scatter(lin, rows(512))
    ts._rs_state = SimpleNamespace(effective=True, reason="")
    # No attribute (a quant method without the GEMM): never blocked by it.
    del lin.fp8_low_m_gemm
    assert ts.uses_ce_reduce_scatter(lin, rows(16)) and ts.uses_ce_reduce_scatter(lin, rows(512))


def test_row_adapter_decides_on_its_padded_input(monkeypatch):
    """``TokenShardedRow`` hands the rule the tensor its GEMM takes -- the input zero-padded
    per sample (``B * S_pad`` rows, contiguous when the plan pads) -- not the caller's tensor,
    so the route is the stock Linear's own: a padded plan whose padding alone crosses the range
    takes the fused op (the module's GEMM would be DeepGEMM on ``B * S_pad`` rows), and a
    viewable non-contiguous input the range serves once padded keeps the module's GEMM."""
    monkeypatch.setattr(token_sharded_tp, "is_sm_100f", lambda: True)
    monkeypatch.setattr(token_sharded_tp, "MIN_CE_REDUCE_SCATTER_ROWS", 0)
    plan = TokenShardPlan.build(1, 5, 2, 0)  # 5 real rows, S_pad 6
    assert plan.is_padded and plan.num_tokens == 5 and plan.padded_rows == 6
    ts = simulated_helper(plan, reduce_scatter_mode="copy_engine")
    ts._rs_state = SimpleNamespace(
        effective=True, reason="", note_row_consumer=lambda k, out_features=None: None
    )
    calls = []
    ts.ce_gemm_reduce_scatter = lambda consumer, act: (
        calls.append(("ce_gemm_reduce_scatter", act))
        or torch.zeros(plan.local_rows, 128, dtype=torch.bfloat16)
    )
    ts.reduce_scatter = lambda partial: (
        calls.append(("reduce_scatter", partial))
        or torch.zeros(plan.local_rows, 128, dtype=torch.bfloat16)
    )
    # The module's GEMM cannot run on CPU: a stand-in keyed on the input it gets.
    monkeypatch.setattr(
        Linear,
        "forward",
        lambda self, x, *a, **kw: (
            calls.append(("gemm", x)) or torch.zeros(x.shape[0], 128, dtype=torch.bfloat16)
        ),
    )
    lin = _fp8_block_linear(bias=True)  # [128, 256]: K_local 256 at tp 2
    lin.tp_size, lin.tp_mode = 2, TensorParallelMode.ROW
    token_sharded_modules._convert(lin, "row", ts, "blocks.0.attn1.to_out.0")
    x = torch.randn(1, 5, 256, dtype=torch.bfloat16)

    # The range serves the caller's 5 rows but not the GEMM's 6: the fused op, on the
    # caller's tensor (it pads itself); the rule on the caller's tensor alone would say no.
    lin.fp8_low_m_gemm = Fp8BlockScaleLowMGemmRange(max_m=5, kernel_max_m=5)
    assert not ts.uses_ce_reduce_scatter(lin, x)
    out = lin(x)
    assert [c[0] for c in calls] == ["ce_gemm_reduce_scatter"] and calls[0][1] is x
    assert out.shape == (len(plan.entry_batch), plan.rows_per_entry, 128)
    # The range serves the 6 padded rows: the module's GEMM on the padded (contiguous) input
    # and the NCCL reduce-scatter, for a contiguous input and for a viewable non-contiguous one
    # alike (the rule on the latter's own strides would say the fused op).
    lin.fp8_low_m_gemm = Fp8BlockScaleLowMGemmRange(max_m=6, kernel_max_m=6)
    wide = torch.randn(1, 5, 512, dtype=torch.bfloat16)[..., :256]
    assert not wide.is_contiguous() and ts.uses_ce_reduce_scatter(lin, wide)
    for inp in (x, wide):
        calls.clear()
        out = lin(inp)
        assert [c[0] for c in calls] == ["gemm", "reduce_scatter"]
        assert calls[0][1].shape == (6, 256) and calls[0][1].is_contiguous()
        assert torch.equal(calls[0][1][:5], inp[0]) and not calls[0][1][5].any()
        assert out.shape == (len(plan.entry_batch), plan.rows_per_entry, 128)
    # Above the range on the padded rows: the fused op again.
    lin.fp8_low_m_gemm = Fp8BlockScaleLowMGemmRange(max_m=5, kernel_max_m=5)
    calls.clear()
    lin(wide)
    assert [c[0] for c in calls] == ["ce_gemm_reduce_scatter"]


# =============================================================================
# The fused op: fake shape and tracing
# =============================================================================


def _op_inputs(padded_rows, k, n, with_bias):
    """CPU operands of the fused op in the shapes and dtypes it checks (never run here)."""
    x = torch.zeros(padded_rows, k, dtype=torch.float8_e4m3fn)
    sf = torch.empty_strided(
        (padded_rows, fp8_scale_cols(k)), (1, pad_up(padded_rows, 4)), dtype=torch.int32
    )
    w = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    ws = _packed_weight_scale(n, k)
    bias = torch.zeros(n, dtype=torch.bfloat16) if with_bias else None
    return x, sf, w, ws, bias


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize(
    "batch,seq,tp", [(1, 8, 2), (2, 5, 2), (3, 7, 4), (1, 420, 4), (2, 105, 3)]
)
def test_rs_op_fake_shape(batch, seq, tp, with_bias):
    from torch._subclasses.fake_tensor import FakeTensorMode

    plan = TokenShardPlan.build(batch, seq, tp, 0)
    k, n = 640, 256
    with FakeTensorMode():
        x, sf, w, ws, bias = _op_inputs(plan.padded_rows, k, n, with_bias)
        out = _OP(x, sf, w, ws, bias, "tp", 0, tp, batch, seq, plan.padded_seq_len)
    assert out.shape == (plan.local_rows, n) and out.dtype == torch.bfloat16


@pytest.mark.parametrize("with_bias", [False, True])
def test_rs_op_traces_with_torch_export(with_bias):
    """A graph through the op: the group name and the plan ints are constants of the call; the
    bias is a graph input (or the None constant)."""
    plan = TokenShardPlan.build(1, 5, 2, 0)  # padded: S_pad = 6, m = 3
    assert plan.padded_seq_len == 6 and plan.local_rows == 3
    k, n = 256, 128

    class Boundary(torch.nn.Module):
        def forward(self, x, sf, w, ws, bias):
            return _OP(x, sf, w, ws, bias, "tp", 0, 2, 1, 5, 6) + 1.0

    args = _op_inputs(plan.padded_rows, k, n, with_bias)
    ep = torch.export.export(Boundary(), args)
    calls = [node for node in ep.graph.nodes if node.target is _OP.default]
    assert len(calls) == 1
    assert list(calls[0].args[5:]) == ["tp", 0, 2, 1, 5, 6]
    bias_arg = calls[0].args[4]
    assert (bias_arg is None) == (not with_bias)
    assert tuple(calls[0].meta["val"].shape) == (plan.local_rows, n)


def test_rs_op_without_a_registered_state_raises():
    with pytest.raises(KeyError):
        get_rs_state("no-such-group")
    x, sf, w, ws, bias = _op_inputs(8, 256, 128, True)
    with pytest.raises(RuntimeError, match="no CeReduceScatterState is registered for TP group"):
        _OP(x, sf, w, ws, bias, "no-such-group", 0, 2, 1, 8, 8)


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_rs_op_chunks_destination_rows(tp, m, monkeypatch):
    """Run the op body with a CPU transport: tails, row offsets, staging reuse and bias.

    Synthetic partials encode both source and destination row. The transport checks each
    outgoing block and supplies the remote sources' bytes without a process group.
    """
    chunk_rows, n, k = 2, 8, 128
    monkeypatch.setattr(tsr, "CHUNK_ROWS", chunk_rows)
    monkeypatch.setattr(tsr, "mn_major_scales", lambda sf: sf)
    all_rows = torch.arange(tp * m).to(torch.float8_e4m3fn).view(-1, 1).expand(-1, k).contiguous()
    row_values = all_rows[:, 0].float().to(torch.bfloat16)
    bias = torch.full((n,), 0.25, dtype=torch.bfloat16)

    for rank in range(tp):

        class Transport:
            tp_size = tp

            def __init__(self):
                self.rank = rank
                self.dest_order = [(rank - i) % tp for i in range(1, tp)]
                self.start = 0
                self.active = False
                self.capacity = 0
                self.pointers = set()

            def covers(self, layout):
                return self.capacity >= layout.slot_nbytes(tp)

            def reserve(self, layout):
                assert layout.rows <= chunk_rows
                self.capacity = layout.slot_nbytes(tp)

            def begin(self, slot, layout):
                assert not self.active and self.covers(layout)
                self.active = True
                self.count = layout.rows
                rows = row_values[rank * m + self.start : rank * m + self.start + self.count]
                self.parts = torch.stack(
                    [(rows[:, None].expand(-1, n) + src).to(torch.bfloat16) for src in range(tp)]
                )
                self.pushed = []

            def push(self, dst, slot, block):
                rows = row_values[dst * m + self.start : dst * m + self.start + self.count]
                assert torch.equal(block, (rows[:, None].expand(-1, n) + rank).to(torch.bfloat16))
                self.pointers.add(block.untyped_storage().data_ptr())
                self.pushed.append(dst)

            def own_view(self, slot):
                return self.parts[rank]

            def wait_all(self, slot):
                assert self.pushed == self.dest_order

            def partials(self, slot):
                return self.parts

            def done(self, slot):
                self.active = False
                self.start += self.count

            def reset(self):
                pytest.fail("the chunked op unexpectedly reset the transport")

        transport = Transport()
        state = SimpleNamespace(transport=transport, next_slot=lambda: 0)
        monkeypatch.setattr(tsr, "get_rs_state", lambda name: state)

        def gemm(x, sf, w, ws, out):
            # Verify scales were sliced with the same destination/chunk offsets as activations.
            assert torch.equal(x[:, 0].float(), sf[:, 0].float())
            out.copy_((x[:, :1].float() + rank).expand_as(out))

        monkeypatch.setattr(tsr, "fp8_block_gemm_out", gemm)
        _, sf, w, ws, _ = _op_inputs(tp * m, k, n, False)
        sf[:, 0] = row_values.to(torch.int32)
        out = tsr.ce_gemm_reduce_scatter_impl(
            all_rows, sf, w, ws, bias, "cpu", rank, tp, 1, tp * m, tp * m
        )
        local = row_values[rank * m : (rank + 1) * m, None].expand(-1, n)
        expected = reference_fixed_order_reduce(
            [(local + src).to(torch.bfloat16) for src in range(tp)], bias
        )
        assert torch.equal(out, expected)
        assert transport.start == m and not transport.active
        assert len(transport.pointers) == 1  # one bounded staging allocation, reused each round
        assert transport.capacity == Bf16RowsRegionLayout(min(m, chunk_rows), n).slot_nbytes(tp)


def test_rs_prepare_caps_workspace_at_large_token_counts(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reserved = []

    def covers(layout):
        return bool(reserved) and reserved[-1].region_nbytes >= layout.region_nbytes

    state = SimpleNamespace(
        probe_once=lambda **kwargs: True,
        group_name="cpu-cap-test",
        world_size=4,
        consumer_ns={5120},
        begin_forward=lambda: None,
        transport=SimpleNamespace(covers=covers, reserve=reserved.append),
    )
    for tokens in (768, 32768, 65536, 768):
        plan = TokenShardPlan.build(1, tokens, 4, 0)
        CeReduceScatterState.prepare(state, plan)
    assert [layout.rows for layout in reserved] == [192, tsr.CHUNK_ROWS]
    assert reserved[-1].slot_nbytes(4) == 320 * 1024**2


# =============================================================================
# Recovery bookkeeping: every rank's signal protocol on a fake pool, in one process
# =============================================================================


class _FakeStream:
    """One CUDA stream of one rank: a FIFO of ops the fabric executes in order."""

    def __init__(self, fabric: "_Fabric", rank: int, name: str):
        self.fabric, self.rank, self.name = fabric, rank, name
        self.ops: list[tuple] = []
        self.done = 0

    def wait_event(self, event: "_FakeEvent") -> None:
        self.ops.append(("event", event))

    def wait_stream(self, other: "_FakeStream") -> None:
        event = _FakeEvent(self.fabric)
        event.record(other)
        self.wait_event(event)

    def __repr__(self) -> str:
        return self.name


class _FakeEvent:
    """A CUDA event: 'everything enqueued on ``stream`` before the record has completed'."""

    def __init__(self, fabric: "_Fabric"):
        self.fabric = fabric
        self.stream: _FakeStream | None = None
        self.index = 0

    def record(self, stream: _FakeStream | None = None) -> None:
        self.stream = stream if stream is not None else self.fabric.current_stream()
        self.index = len(self.stream.ops)

    def ready(self) -> bool:
        return self.stream is not None and self.stream.done >= self.index

    def __repr__(self) -> str:
        return f"event({self.stream}@{self.index})"


class _Fabric:
    """Single-process model of what the devices do with the work the transports of ALL ranks
    enqueue: the signal pads (one bit per ``(channel, src -> dst)``, kept the way
    ``put_signal`` / ``wait_signal`` keep them) and the CUDA streams of every rank (one op
    queue per stream; events order queues). :meth:`run` executes the queues as the devices
    would -- a put is runnable only while its flag is clear, a wait only once it is set -- and
    fails where the real pool traps the device after its deadline: when no queue can make
    progress. Puts and waits are counted per ``(channel, src -> dst)`` for the balance check.
    """

    def __init__(self, tp: int):
        self.tp = tp
        self.rank = 0  # the rank whose transport code is running
        self.active: _FakeStream | None = None  # set by the torch.cuda.stream context
        self.streams: list[_FakeStream] = []
        self.main = [self._new_stream(r, f"rank{r}/current") for r in range(tp)]
        self.flags: dict[tuple[int, int, int], int] = {}  # (channel, src, dst) -> 0 / 1
        self.puts: Counter = Counter()
        self.waits: Counter = Counter()
        self.pushes: Counter = Counter()  # (slot, src, dst) -> pushes the driver issued
        self.bufs: dict[int, torch.Tensor] = {}  # rank -> its symmetric buffer

    def _new_stream(self, rank: int, name: str) -> _FakeStream:
        stream = _FakeStream(self, rank, name)
        self.streams.append(stream)
        return stream

    # -- the torch.cuda surface the transport uses --

    def Stream(self, device=None, priority=0) -> _FakeStream:  # noqa: N802 (torch.cuda.Stream)
        n = sum(1 for s in self.streams if s.rank == self.rank) - 1
        return self._new_stream(self.rank, f"rank{self.rank}/side{n}")

    def Event(self) -> _FakeEvent:  # noqa: N802 (torch.cuda.Event)
        return _FakeEvent(self)

    def current_stream(self) -> _FakeStream:
        return self.active if self.active is not None else self.main[self.rank]

    @contextlib.contextmanager
    def stream(self, stream: _FakeStream):
        prev, self.active = self.active, stream
        try:
            yield
        finally:
            self.active = prev

    @staticmethod
    def is_current_stream_capturing() -> bool:
        return False

    def install(self, monkeypatch) -> None:
        for name in ("Stream", "Event", "current_stream", "stream", "is_current_stream_capturing"):
            monkeypatch.setattr(torch.cuda, name, getattr(self, name))
        # The base's resync after a reset: every rank reset in a rank-symmetric case, so the
        # counts agree (the base's own test covers the disagreement).
        monkeypatch.setattr(
            ce_protocol, "_gather_boundary_counts", lambda group, count: [count] * self.tp
        )

    # -- the pads --

    def put(self, channel: int, src: int, dst: int) -> None:
        self.current_stream().ops.append(("put", channel, src, dst))

    def wait(self, channel: int, src: int, dst: int) -> None:
        self.current_stream().ops.append(("wait", channel, src, dst))

    def _step(self, op: tuple) -> bool:
        if op[0] == "event":
            return op[1].ready()
        kind, channel, src, dst = op
        key = (channel, src, dst)
        if kind == "put":
            if self.flags.get(key, 0):
                return False  # put_signal spins while the previous flag is unconsumed
            self.flags[key] = 1
            self.puts[key] += 1
        else:
            if not self.flags.get(key, 0):
                return False  # wait_signal spins until the flag is set
            self.flags[key] = 0
            self.waits[key] += 1
        return True

    def run(self) -> None:
        """Execute every queue to its end, or fail where the devices would trap."""
        while True:
            progressed = pending = False
            for stream in self.streams:
                while stream.done < len(stream.ops):
                    if not self._step(stream.ops[stream.done]):
                        pending = True
                        break
                    stream.done += 1
                    progressed = True
            if not pending:
                return
            if not progressed:
                heads = "; ".join(
                    f"{s}: {s.ops[s.done]}" for s in self.streams if s.done < len(s.ops)
                )
                raise AssertionError(
                    f"the devices would trap at the signal deadline; blocked: {heads}"
                )


class _FakePool:
    """``SymmMemPool`` stand-in over a :class:`_Fabric`: real CPU buffers (one per rank, the
    peers' mapped through ``peer_region``), the pool's own region arithmetic, and signals that
    enqueue into the fabric instead of launching kernels."""

    def __init__(self, fabric: _Fabric, rank: int):
        self.fabric = fabric
        self.rank = rank
        self.world_size = fabric.tp
        self.align = DEFAULT_ALIGN
        self.channels = 4  # ready and consumed for the two slots the tests use
        self.group = None  # the resync gather is the fabric's
        self.group_name = f"fake{fabric.tp}"
        self._layout: PoolLayout | None = None

    @property
    def reserved(self) -> bool:
        return self._layout is not None

    @property
    def nbytes(self) -> int:
        return self._layout.nbytes if self._layout else 0

    @property
    def slot_nbytes(self) -> int:
        return self._layout.slot_nbytes if self._layout else 0

    @property
    def slots(self) -> int:
        return self._layout.slots if self._layout else 0

    @property
    def region_nbytes(self) -> int:
        return self._layout.region_nbytes if self._layout else 0

    def reserve(self, slot_nbytes: int, slots: int, min_channels: int) -> bool:
        want = PoolLayout.for_request(slot_nbytes, slots, self.world_size)
        if self._layout is not None:
            if self._layout.covers(want):
                return False
            want = self._layout.merged(want)
        self._layout = want
        self.fabric.bufs[self.rank] = torch.zeros(want.nbytes, dtype=torch.uint8)
        for key in self.fabric.flags:  # the real pool zeroes this rank's pad on every allocation
            if key[2] == self.rank:
                self.fabric.flags[key] = 0
        return True

    def region(self, slot: int, rank: int, offset: int, nbytes: int) -> torch.Tensor:
        start, stop = self._layout.region_span(slot, rank, offset, nbytes)
        return self.fabric.bufs[self.rank][start:stop]

    def peer_region(self, peer: int, slot: int, offset: int, nbytes: int) -> torch.Tensor:
        start, stop = self._layout.region_span(slot, self.rank, offset, nbytes)
        return self.fabric.bufs[peer][start:stop]

    def signal(self, peer: int, channel: int) -> None:
        self.fabric.put(channel, self.rank, peer)

    def wait(self, src: int, channel: int) -> None:
        self.fabric.wait(channel, src, self.rank)

    def release(self) -> None:
        self._layout = None


def _block(src: int, dst: int, m: int, n: int, boundary: int) -> torch.Tensor:
    """Source ``src``'s partial of destination ``dst``'s rows at boundary ``boundary``."""
    seed = 1 + src + 16 * dst + 256 * boundary
    return torch.randn((m, n), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _transports(fabric: _Fabric, slots: int, layout: Bf16RowsRegionLayout) -> list[CeReduceScatter]:
    transports = []
    for r in range(fabric.tp):
        fabric.rank = r
        transports.append(
            CeReduceScatter(_FakePool(fabric, r), fabric.tp, r, torch.device("cpu"), slots=slots)
        )
    for r, transport in enumerate(transports):
        fabric.rank = r
        assert transport.reserve(layout)
    return transports


def _boundary(fabric, transports, slot, layout, index, fail=None) -> dict[int, torch.Tensor]:
    """One boundary on every rank, enqueued in rank order as the op body does, then run on the
    fabric. ``fail`` injects the op's rank-symmetric raise and its except path (``reset``):
    ``"before_begin"``; an int ``k`` = the GEMM of push ``k`` failed after ``k`` pushes
    (``k = tp - 1`` is the own GEMM: every push issued); ``"before_wait_all"``;
    ``"in_reduce"`` (after ``wait_all``). Returns each rank's ``[tp, m, N]`` partials view
    (a normal boundary only), valid once the fabric ran."""
    tp = len(transports)
    m, n = layout.rows, layout.cols
    views: dict[int, torch.Tensor] = {}
    for r, transport in enumerate(transports):
        fabric.rank = r
        try:
            if fail == "before_begin":
                raise RuntimeError("injected before begin")
            transport.begin(slot, layout)
            for i, dst in enumerate(transport.dest_order):
                if fail == i:
                    raise RuntimeError(f"injected at GEMM {i}")
                transport.push(dst, slot, _block(r, dst, m, n, index))
                fabric.pushes[(slot, r, dst)] += 1
            if fail == tp - 1:
                raise RuntimeError("injected at the own GEMM")
            transport.own_view(slot).copy_(_block(r, r, m, n, index))
            if fail == "before_wait_all":
                raise RuntimeError("injected before wait_all")
            transport.wait_all(slot)
            if fail == "in_reduce":
                raise RuntimeError("injected inside the reduce")
            views[r] = transport.partials(slot)
            transport.done(slot)
        except RuntimeError:
            assert fail is not None, "the transport raised in a normal boundary"
            transport.reset()  # the op body's except path
    fabric.run()
    return views


def _check_partials(views: dict[int, torch.Tensor], tp: int, m: int, n: int, index: int) -> None:
    """Every destination holds all ``tp`` partials of THIS boundary in source order and its
    fixed-order reduce is the reference chain."""
    assert sorted(views) == list(range(tp))
    for d, view in views.items():
        parts = [_block(src, d, m, n, index) for src in range(tp)]
        for src in range(tp):
            assert torch.equal(view[src], parts[src]), (d, src)
        out = torch.empty((m, n), dtype=torch.bfloat16)
        fixed_order_reduce_bf16(view, None, out)
        assert torch.equal(out, reference_fixed_order_reduce(parts)), d


def _assert_balanced(fabric: _Fabric, transports: list[CeReduceScatter]) -> None:
    """The invariant the transport documents (module docstring, 'Slot state machine'): after
    every completed boundary (``done`` or ``reset``) and for every slot ``s`` and ordered pair
    ``p -> d``: the ``ready(s)`` flag is clear and ``puts = waits`` = the pushes ``p`` issued
    to ``d`` on ``s`` (``uses[s]`` unless a boundary was cut short before that push); the
    ``consumed(s)`` flag is set iff ``uses[s] > 0``, with ``puts = uses[s]`` and
    ``waits = uses[s] - 1`` -- one unconsumed flag per wait of the next boundary, on both
    sides."""
    tp, slots = fabric.tp, transports[0].slots
    for s in range(slots):
        uses = {t._uses[s] for t in transports}
        assert len(uses) == 1, f"slot {s}: uses differ across ranks: {uses}"
        (uses,) = uses
        for p in range(tp):
            for d in range(tp):
                if p == d:
                    continue
                ready, consumed = (s, p, d), (slots + s, p, d)
                assert fabric.flags.get(ready, 0) == 0, ("ready set", s, p, d)
                issued = fabric.pushes[(s, p, d)]
                assert issued <= uses, ("pushes", s, p, d)
                assert fabric.puts[ready] == fabric.waits[ready] == issued, ("ready", s, p, d)
                assert fabric.flags.get(consumed, 0) == (1 if uses else 0), ("consumed", s, p, d)
                assert fabric.puts[consumed] == uses, ("consumed puts", s, p, d)
                assert fabric.waits[consumed] == max(uses - 1, 0), ("consumed waits", s, p, d)


_RECOVERY_TPS = (2, 3, 4, 8)


def _recovery_paths(tp: int) -> list:
    """Every point of the op body a rank-symmetric raise can hit at ``tp`` ranks: before
    ``begin``; at GEMM ``k`` for every ``k`` (after ``k`` pushes; ``k = tp - 1`` is the own
    GEMM, every push issued); before ``wait_all``; inside the reduce."""
    return ["before_begin", *range(tp), "before_wait_all", "in_reduce"]


def _recovery_cases() -> list:
    return [
        pytest.param(tp, fail, id=f"tp{tp}-{fail if isinstance(fail, str) else f'gemm{fail}'}")
        for tp in _RECOVERY_TPS
        for fail in _recovery_paths(tp)
    ]


@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("tp", [2, 3, 4])
def test_boundaries_keep_the_signals_balanced(tp, slots, monkeypatch):
    """Normal boundaries: the first use of each slot skips the consumed wait, every later one
    waits it; after each boundary the flags are balanced and every destination reduced the
    boundary's own partials."""
    fabric = _Fabric(tp)
    fabric.install(monkeypatch)
    layout = Bf16RowsRegionLayout(5, 24)
    transports = _transports(fabric, slots, layout)
    for index in range(3 * slots):
        views = _boundary(fabric, transports, index % slots, layout, index)
        _check_partials(views, tp, 5, 24, index)
        _assert_balanced(fabric, transports)
    assert all(t._uses == [3] * slots for t in transports)
    # The op body's drain between forwards and a reserve that does not grow touch no flag.
    for r, t in enumerate(transports):
        fabric.rank = r
        t.drain()
        assert not t.reserve(layout)
    fabric.run()
    _assert_balanced(fabric, transports)


@pytest.mark.parametrize("first_use", [False, True])
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("tp,fail", _recovery_cases())
def test_reset_after_a_rank_symmetric_raise_keeps_the_signals_balanced(
    tp, fail, slots, first_use, monkeypatch
):
    """A rank-symmetric raise at every point of the op body -- before ``begin``, at GEMM
    ``k`` for every ``k`` (after ``k`` pushes; ``k = tp - 1`` is the own GEMM, every push
    issued), before ``wait_all``, inside the reduce -- on a warm slot and on a slot's first
    use, at both slot counts and ``tp`` 2, 3, 4, 8: ``reset`` completes the boundary's
    protocol like ``done`` (the balance invariant holds right after it), the recovered slot
    never re-enters first use, no device would trap (the job-4800143 symptom: ``reset``
    putting ``consumed`` to a destination that never pushed in the failed boundary, a second
    put onto a set flag), and the following boundaries are plain boundaries whose reduce is
    the reference chain -- first with a smaller layout the reservation covers (a warm slot's
    views are dropped and re-carved by ``_bind_layout``), then with the original."""
    fabric = _Fabric(tp)
    fabric.install(monkeypatch)
    m, n = 5, 24
    layout = Bf16RowsRegionLayout(m, n)
    transports = _transports(fabric, slots, layout)
    index = 0
    for _ in range(0 if first_use else 2 * slots):
        views = _boundary(fabric, transports, index % slots, layout, index)
        _check_partials(views, tp, m, n, index)
        _assert_balanced(fabric, transports)
        index += 1
    failed_slot = index % slots
    uses_before = transports[0]._uses[failed_slot]
    assert _boundary(fabric, transports, failed_slot, layout, index, fail=fail) == {}
    _assert_balanced(fabric, transports)
    completed = fail != "before_begin"  # nothing was in flight: reset() had nothing to release
    assert all(t._uses[failed_slot] == uses_before + int(completed) for t in transports)
    assert not any(any(t._in_flight) for t in transports)
    index += 1
    small = Bf16RowsRegionLayout(3, n)  # same aligned region, a different layout
    assert all(t.covers(small) for t in transports)
    for i in range(2 * slots):
        after = small if i < slots else layout
        views = _boundary(fabric, transports, index % slots, after, index)
        _check_partials(views, tp, after.rows, after.cols, index)
        _assert_balanced(fabric, transports)
        index += 1
    for r, t in enumerate(transports):
        fabric.rank = r
        t.drain()


@pytest.mark.parametrize("first_use", [False, True])
@pytest.mark.parametrize(
    "pair",
    [(0, "before_wait_all"), ("in_reduce", 0)],
    ids=["gemm0+before_wait_all", "in_reduce+gemm0"],
)
@pytest.mark.parametrize("slots", [1, 2])
@pytest.mark.parametrize("tp", _RECOVERY_TPS)
def test_back_to_back_recoveries_keep_the_signals_balanced(tp, slots, pair, first_use, monkeypatch):
    """Two rank-symmetric raises on every slot with no good boundary between them (from a
    warm slot and from a slot's first use): the IDLE state ``reset`` leaves -- every
    ``consumed`` flag set, no ``ready`` flag, ``uses`` counted -- is the one ``done`` leaves,
    so the second recovery starts from it as a plain boundary would, and the boundaries
    after both are plain ones whose reduce is the reference chain."""
    fabric = _Fabric(tp)
    fabric.install(monkeypatch)
    m, n = 5, 24
    layout = Bf16RowsRegionLayout(m, n)
    transports = _transports(fabric, slots, layout)
    index = 0
    for _ in range(0 if first_use else slots):
        views = _boundary(fabric, transports, index % slots, layout, index)
        _check_partials(views, tp, m, n, index)
        index += 1
    for i in range(2 * slots):  # slot s fails with pair[0], then on its next use with pair[1]
        failed = _boundary(fabric, transports, index % slots, layout, index, fail=pair[i // slots])
        assert failed == {}
        _assert_balanced(fabric, transports)
        index += 1
    assert all(t._uses == [2 + (0 if first_use else 1)] * slots for t in transports)
    assert not any(any(t._in_flight) for t in transports)
    for _ in range(2 * slots):
        views = _boundary(fabric, transports, index % slots, layout, index)
        _check_partials(views, tp, m, n, index)
        _assert_balanced(fabric, transports)
        index += 1
    for r, t in enumerate(transports):
        fabric.rank = r
        t.drain()


def test_fabric_detects_the_trap():
    """The fabric fails where the pool traps: a second put onto a set flag, a wait with no put."""
    fabric = _Fabric(2)
    fabric.put(0, 0, 1)
    fabric.put(0, 0, 1)
    with pytest.raises(AssertionError, match=r"would trap .* \('put', 0, 0, 1\)"):
        fabric.run()
    fabric = _Fabric(2)
    fabric.wait(1, 1, 0)
    with pytest.raises(AssertionError, match=r"would trap .* \('wait', 1, 1, 0\)"):
        fabric.run()


# =============================================================================
# CeReduceScatterState on a gloo group: the probe falls back, the registry lifecycle
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
    tsr.logger = ce_protocol.logger = recorder  # the probe's one warning is the base's
    with pytest.raises(ValueError, match="process group; got None"):
        CeReduceScatterState(None, name)
    with pytest.raises(ValueError, match="timeout_ms >= 0"):
        CeReduceScatterState(group, name, timeout_ms=-1)
    state = CeReduceScatterState(group, name)
    assert state.tp_size == world_size and state.rank == rank and state.per_peer_streams
    assert not CeReduceScatterState(group, name, per_peer_streams=False).per_peer_streams
    assert not state.effective and state.reason == "not probed yet"
    assert register_rs_state(state) is state
    assert register_rs_state(CeReduceScatterState(group, name)) is state  # two TPs share one
    assert get_rs_state(name) is state
    state.note_row_consumer(1280, 5120)
    state.note_row_consumer(3456, 5120)
    state.note_row_consumer(640)  # K only: the op sizes the pool at its first eager call
    with pytest.raises(ValueError, match="positive multiple of 128"):
        state.note_row_consumer(100, 5120)
    with pytest.raises(ValueError, match="N must be >= 1"):
        state.note_row_consumer(128, 0)
    assert state.consumer_ks == {640, 1280, 3456} and state.consumer_ns == {5120}
    plan = TokenShardPlan.build(1, 8, world_size, rank)
    state.prepare(plan)  # probes: a gloo group is not a CUDA NCCL group
    assert not state.effective and "NCCL" in state.reason
    assert state.transport is None and state.pool is None
    assert len(recorder.warnings) == 1 and state.reason in recorder.warnings[0]
    assert "reduce-scatter" in recorder.warnings[0]
    state.prepare(plan)  # probed once: no second warning, still a no-op
    assert len(recorder.warnings) == 1 and not recorder.infos
    assert SLOTS == 1 and [state.next_slot() for _ in range(3)] == [0, 0, 0]
    state.begin_forward()
    assert state.next_slot() == 0
    # The op refuses (rank-symmetrically) when the state is registered but not effective.
    x, sf, w, ws, bias = _op_inputs(plan.padded_rows, 256, 128, True)  # the bias on every rank
    with pytest.raises(RuntimeError, match="not effective"):
        _OP(x, sf, w, ws, bias, name, rank, world_size, 1, 8, plan.padded_seq_len)

    # A resync disagreement raised by the transport (CeResyncError: the ranks' boundary counts
    # differ after a failed boundary) takes the group off the protocol, CeState.disable:
    # effective False, the reason, the error propagated (the state's one warning was already
    # given by the probe); the adapters' next call reads effective and reduce-scatters with
    # NCCL. Nothing was opened, so reset() must not run.
    def _raise_resync(*args):
        raise ce_protocol.CeResyncError("boundary counts differ across the ranks: [1, 2]")

    def _no_reset():
        raise AssertionError("reset() must not run for a resync disagreement")

    state.effective = True
    state.transport = SimpleNamespace(
        tp_size=world_size,
        rank=rank,
        covers=lambda layout: True,
        begin=_raise_resync,
        reset=_no_reset,
    )
    with pytest.raises(ce_protocol.CeResyncError, match="boundary counts differ"):
        _OP(x, sf, w, ws, bias, name, rank, world_size, 1, 8, plan.padded_seq_len)
    assert not state.effective and "boundary counts differ" in state.reason
    assert len(recorder.warnings) == 1  # one warning per state: the probe's
    state.transport = None
    release_rs_state(name)
    with pytest.raises(KeyError):
        get_rs_state(name)
    state.close()  # idempotent
    release_rs_state(name)  # no-op without a state
    # A closed state refuses to start a forward (a sibling helper still holding it is told).
    assert not state.effective and state.reason == "closed"
    with pytest.raises(RuntimeError, match="is closed"):
        state.prepare(plan)
    # A fresh state for the same name probes again (and warns again, once).
    fresh = register_rs_state(CeReduceScatterState(group, name))
    assert fresh is not state and get_rs_state(name) is fresh
    fresh.prepare(plan)
    assert len(recorder.warnings) == 2
    fresh.close()
    # The RS registry is separate from the gather's.
    from tensorrt_llm._torch.visual_gen.parallel import token_sharded_ce_gather as tsg

    with pytest.raises(KeyError):
        tsg.get_state(name)


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
def test_rs_state_on_gloo(world_size, tmp_path):
    init_file = str(tmp_path / "rendezvous")
    mp.spawn(_worker, args=(world_size, _state_logic, init_file), nprocs=world_size, join=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
