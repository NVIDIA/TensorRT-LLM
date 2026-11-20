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
"""One op for every batch size: ``spec_dsa.op`` from 1 to 1,024 rows (SM100).

The op picks the megakernel up to the spec's ``megakernel_max_batch`` rows and the large-batch kernels above it, for
the FP8 and the MXFP4 indexer K cache (the ``idx`` fixture). The case builder
and the fp64 references of the large-batch test check every row's top-512 and attention here too, so both
implementations are covered through the one interface: batch sizes around the switch and between the powers of two,
the CUDA-graph and in-place refresh path, and the argument checks of the op.
"""

import dataclasses

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import large_batch_op as _lbo

from . import test_cute_dsl_spec_dsa_large_batch as H

pytestmark = H.pytestmark
idx = H.idx
# The batch sizes at which the FP4 spec switches: its megakernel's last size, and the first size that runs the second
# compile of the large-batch indexer kernel (``large_batch_op._VARIANTS``).
FP4_LAST = H.SPEC_FP4.megakernel_max_batch
FP4_VARIANT = min((first for first, _ in _lbo._VARIANTS.get(H.SPEC_FP4.kernel_key, ())), default=64)
SWITCHES = {FP4_LAST, FP4_LAST + 1, FP4_VARIANT - 1, FP4_VARIANT}


def _op():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import op

    return op


def _prepare(op, inp, state=None, idx="fp8"):
    # host lengths: prepare() reads them without a device sync
    return op.prepare(
        inp["ctx_lens"].cpu(),
        inp["kv_lens"].cpu(),
        inp["idx_block_table"],
        inp["sinks"],
        state=state,
        config=H._spec_of(idx),
    )


def _launch(op, state, inp, topk, out):
    op.run(
        state,
        inp["q_idx"],
        inp["w_idx"],
        inp["k_cache"],
        inp["idx_block_table"],
        inp["ctx_lens"],
        inp["kv_lens"],
        inp["q_att"],
        inp["swa_pool"],
        inp["bt_swa"],
        inp["c_pool"],
        inp["bt_c"],
        topk,
        out,
        softmax_scale=H.SM_SCALE,
        **({"q_scale": inp["q_scale"]} if "q_scale" in inp else {}),
    )


def test_implementation_for(idx):
    op = _op()
    spec = H._spec_of(idx)
    last = spec.megakernel_max_batch
    assert op.implementation_for(1, spec) == "megakernel"
    assert op.implementation_for(last, spec) == "megakernel"
    assert op.implementation_for(last + 1, spec) == "large_batch"
    assert op.implementation_for(spec.max_batch, spec) == "large_batch"
    for bad in (-1, 0, spec.max_batch + 1):
        with pytest.raises(ValueError, match="batch size"):
            op.implementation_for(bad, spec)


@pytest.mark.parametrize(
    "batch",
    sorted({1, 2, 3, 4, 5, 8, 9, 13, 16, 17, 24, 32, 33, 40, 47, 48, 49, 64, 149, 1024} | SWITCHES),
)
def test_matches_reference(batch, idx):
    op = _op()
    inp, bases = H._make_case(batch, H._plain(H.LENGTHS), seed=100 + batch, idx=idx)
    state = _prepare(op, inp, idx=idx)
    assert state.batch_size == batch
    assert state.implementation == op.implementation_for(batch, H._spec_of(idx))
    topk, out = H._new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    assert int(op.error_bits(state).max()) == 0
    H._check_rows(inp, bases, topk, out, "realistic")


@pytest.mark.parametrize("batch", sorted({4, 5, 8, 13, 24, 40, 47, 48, 60} | SWITCHES))
def test_refresh_in_place_and_cuda_graph(batch, idx):
    op = _op()
    ns = H.LENGTHS[:batch] if batch <= len(H.LENGTHS) else H.LENGTHS
    inp, bases = H._make_case(batch, H._plain(ns), seed=7 + batch, idx=idx)
    state = _prepare(op, inp, idx=idx)
    topk, out = H._new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp, bases, topk, out, "realistic")
    # the same state refreshed in place for another step, then one captured launch replayed
    inp2, bases2 = H._make_case(batch, H._plain(ns[::-1]), seed=70 + batch, idx=idx)
    assert _prepare(op, inp2, state=state, idx=idx) is state
    topk.fill_(-1)
    out.zero_()
    _launch(op, state, inp2, topk, out)
    torch.cuda.synchronize()
    H._check_rows(inp2, bases2, topk, out, "realistic")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _launch(op, state, inp2, topk, out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(op, state, inp2, topk, out)
    for _ in range(2):
        topk.fill_(-1)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        H._check_rows(inp2, bases2, topk, out, "realistic")
    assert int(op.error_bits(state).max()) == 0


@pytest.mark.parametrize("batch", sorted({1, 3, 9, 64} | SWITCHES))
def test_negative_logits(batch, idx):
    """Rows whose logits are all <= 0 through the op: the megakernel sizes and the large-batch sizes of both specs."""
    op = _op()
    ns = H.LENGTHS[:batch] if batch <= len(H.LENGTHS) else H.LENGTHS
    inp, bases = H._make_case(batch, [{"n": n, "neg": True} for n in ns], seed=170 + batch, idx=idx)
    state = _prepare(op, inp, idx=idx)
    assert state.implementation == op.implementation_for(batch, H._spec_of(idx))
    topk, out = H._new_outputs(batch)
    for launch in range(3):
        topk.fill_(-5)
        _launch(op, state, inp, topk, out)
        torch.cuda.synchronize()
        assert int(op.error_bits(state).max()) == 0
        if launch == 0:
            H._check_rows(inp, bases, topk, out, "realistic")
        else:
            H._check_selection(bases, topk)


def test_rejects_bad_arguments():
    op = _op()
    dev = torch.device("cuda")
    i32 = torch.int32
    sinks = torch.zeros(64, dtype=torch.float32, device=dev)
    bt = torch.zeros((2, 128), dtype=i32, device=dev)
    with pytest.raises(ValueError, match="sinks"):
        op.prepare(
            torch.tensor([4096], dtype=i32),
            torch.tensor([16384], dtype=i32),
            bt[:1],
            sinks[:8],
        )
    with pytest.raises(ValueError, match="compressed lengths"):
        op.prepare(torch.tensor([300], dtype=i32), torch.tensor([1200], dtype=i32), bt[:1], sinks)
    state = op.prepare(
        torch.tensor([4096], dtype=i32), torch.tensor([16384], dtype=i32), bt[:1], sinks
    )
    with pytest.raises(ValueError, match="state is for"):
        op.prepare(
            torch.full((2,), 4096, dtype=i32),
            torch.full((2,), 16384, dtype=i32),
            bt,
            sinks,
            state=state,
        )
    # a state keeps the spec it was built for
    other = dataclasses.replace(H.SPEC, name="other", topk=1024)
    with pytest.raises(ValueError, match="state is for"):
        op.prepare(
            torch.tensor([4096], dtype=i32),
            torch.tensor([16384], dtype=i32),
            bt[:1],
            sinks,
            state=state,
            config=other,
        )
    assert state.config is H.SPEC
    many = H.SPEC.megakernel_max_batch + 1
    bt_many = torch.zeros((many, 128), dtype=i32, device=dev)
    large_state = op.prepare(
        torch.full((many,), 4096, dtype=i32), torch.full((many,), 16384, dtype=i32), bt_many, sinks
    )
    with pytest.raises(ValueError, match="state is for"):
        op.prepare(
            torch.full((1,), 4096, dtype=i32),
            torch.full((1,), 16384, dtype=i32),
            bt[:1],
            sinks,
            state=large_state,
        )
