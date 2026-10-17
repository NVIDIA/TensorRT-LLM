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
"""The attention_output_tail contract on one GPU: no attention_output_tail entry point communicates.

The attention_output_tail contract: "attention_output_tail owns W_UV + local o_proj.
router_gemm/native all-reduce consumes the identical partial. No communication is hidden inside
attention_output_tail."

The real DSA ``MLA`` of TP8 rank 0 with real layer-3 weights gets an
``o_proj`` that reduces its output, as in the model at TP8
(``reduce_output=True``). A recording stand-in (``SpyAllReduce``) replaces
the all-reduce module. It reports an NCCL symmetric-memory window, as the
model's AUTO all-reduce does by default. The ``AttnTail`` entry points
(``arm``, ``before_attention``, ``defer_w_uv``, ``project``) are driven with
synthetic latents for every engine and named schedule the package ships:
attention_output_tail's default, the ``attention_output_tail_prefetch`` and
``attention_output_into_hop`` schedules (``INTO_HOP_SPARSE_DECODE_SCHEDULE`` is the served
one), the into-hop publication, the ``fused`` and ``triton`` engines, and
``attention_output_tail_prefetch``'s ``native`` engine with the cuBLAS and ring o_proj. The decoder
layer's two ``AllReduceParams`` are used: enabled (dense layers, where ``Linear.forward`` reduces)
and disabled (MoE layers, where the reduction is fused downstream).

Checks:

* no all-reduce call happens while any tail entry point runs;
* ``project`` returns the rank-local partial: for the attention_output_tail kernels it equals
  the functional ``attn_tail`` bit for bit; the ``native`` engine returns
  the native ``o_proj`` GEMM output (bit-identical for cuBLAS, within the
  frozen budget for the ring kernel) before any reduction;
* captured in a CUDA graph, the tail still enqueues no all-reduce.

``MLA.forward``'s reduction of this partial is tested at the real call site
in ``test_attn_tail_contract_o_forward.py`` and ``test_attn_tail_mla_forward_mpi.py``.
"""

import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import tail_reference  # noqa: E402
import test_attn_tail_mla_adapter as tail_adapter  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E402, E501
    DEFAULT_CONFIG,
    INTO_HOP_SCHEDULE,
    INTO_HOP_SPARSE_DECODE_SCHEDULE,
    TAIL_PREFETCH_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
    TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE,
    TAIL_PREFETCH_STAGED_NATIVE_SCHEDULE,
    TailConfig,
    attn_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E402, E501
    AttnTail,
)

pytestmark = tail_adapter.pytestmark

ROWS = 6
_R = dataclasses.replace

ATTN_TAIL_KERNEL_CONFIGS = {
    "attention_output_tail": DEFAULT_CONFIG,
    "attention_output_tail_prefetch": TAIL_PREFETCH_SCHEDULE,
    "tail_prefetch_sparse_mla": TAIL_PREFETCH_SPARSE_DECODE_SCHEDULE,
    "tail_prefetch_sparse_mla_attn_hop": TAIL_PREFETCH_SPARSE_DECODE_HOP_SCHEDULE,
    "attention_output_into_hop": INTO_HOP_SCHEDULE,
    "into_hop_sparse_mla": INTO_HOP_SPARSE_DECODE_SCHEDULE,
    "fused": TailConfig(oproj_engine="fused"),
    "triton": TailConfig(oproj_engine="triton"),
}


TAIL_ENTRY_POINTS = ("arm", "before_attention", "defer_w_uv", "project")


class SpyAllReduce:
    """Stand-in for ``o_proj.all_reduce`` that records every call.

    ``flag.in_tail`` is set while an attention_output_tail entry point runs. An enabled call
    returns twice its input (two identical ranks), so a reduction that
    happens is visible in the values; a disabled call returns its input, as
    ``AllReduce.forward`` does.
    """

    def __init__(self, flag=None):
        self.calls = []
        self.flag = flag if flag is not None else SimpleNamespace(in_tail=False)

    def uses_nccl_symmetric_memory_window(self) -> bool:
        return True

    def __call__(self, input, all_reduce_params=None):
        self.calls.append(
            SimpleNamespace(
                in_tail=self.flag.in_tail, params=all_reduce_params, data_ptr=input.data_ptr()
            )
        )
        if all_reduce_params is not None and not all_reduce_params.enable_allreduce:
            return input
        return input * 2


def guard_tail(tail: AttnTail, spy: SpyAllReduce) -> None:
    """Flag every all-reduce call made while a tail entry point runs."""
    for name in TAIL_ENTRY_POINTS:
        entry = getattr(tail, name)

        def guarded(*args, _entry=entry, **kwargs):
            spy.flag.in_tail = True
            try:
                return _entry(*args, **kwargs)
            finally:
                spy.flag.in_tail = False

        setattr(tail, name, guarded)


_MLA = {}


def _reducing_mla():
    """The real DSA MLA with an ``o_proj`` that reduces into a spy."""
    if "m" not in _MLA:
        mla = tail_adapter._build_mla()
        mla.o_proj.reduce_output = True
        _MLA["m"] = mla
    mla = _MLA["m"]
    spy = SpyAllReduce()
    mla.o_proj.all_reduce = spy
    return mla, spy


def _params(enable: bool):
    from tensorrt_llm._torch.distributed import AllReduceParams

    return AllReduceParams(enable_allreduce=enable)


def _transport(rank=3):
    """A post_attention_hop transport stand-in on one GPU (see attention_output_tail/test_into_hop_producer.py)."""
    words = 8 * 8 * 6144 * 2 // 4
    mc = torch.full((3 * words,), -(2**31), dtype=torch.int32, device="cuda")
    flags = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device="cuda")
    return SimpleNamespace(mc_words=mc, flags=flags, rank=rank, cur=1)


def _tail(cfg, mla, spy):
    tail = AttnTail("layer3", cfg)
    if cfg.successor_prefetch:
        # The router weight an attention_output_tail_prefetch tail stages for the op after it.
        tail.set_successor_weights([torch.zeros((256, 6144), dtype=torch.bfloat16, device="cuda")])
    guard_tail(tail, spy)
    return tail


def _attn_tail_call(tail, mla, latent, params, hop_transport=None):
    rows = latent.shape[0]
    attn_out = torch.empty((rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(rows)
    if hop_transport is not None:
        tail.request_into_hop(hop_transport, rows)
    assert tail.arm(mla, meta)
    tail.before_attention(mla, point=tail.config.prefetch_fork)
    assert tail.defer_w_uv(mla, meta, latent, attn_out)
    return tail.project(mla, attn_out, params)


@pytest.mark.parametrize("enable", [True, False], ids=["allreduce_on", "allreduce_fused"])
@pytest.mark.parametrize("name", list(ATTN_TAIL_KERNEL_CONFIGS))
def test_attn_tail_kernels_return_partial_without_communication(name, enable):
    cfg = ATTN_TAIL_KERNEL_CONFIGS[name]
    mla, spy = _reducing_mla()
    latent = tail_adapter._latent(ROWS, seed=11)
    want = attn_tail(latent, mla.v_b_proj, mla.o_proj.weight, ROWS, config=cfg)
    tail = _tail(cfg, mla, spy)
    got = _attn_tail_call(tail, mla, latent, _params(enable))
    torch.cuda.synchronize()
    assert spy.calls == [], "the attention_output_tail tail called the all-reduce"
    # Engaged with a window all-reduce attached, as at TP8 in the model.
    assert tail.static_reason is None
    assert tail.engaged_calls == 1 and tail._pending is None
    assert torch.equal(got, want), name
    if cfg.successor_prefetch:
        assert tail.prefetch_forks == 1 and not tail._prefetch.forked


@pytest.mark.parametrize("enable", [True, False], ids=["allreduce_on", "allreduce_fused"])
@pytest.mark.parametrize("name", ["attention_output_into_hop", "into_hop_sparse_mla"])
def test_into_hop_publication_has_no_communication(name, enable):
    cfg = ATTN_TAIL_KERNEL_CONFIGS[name]
    mla, spy = _reducing_mla()
    latent = tail_adapter._latent(ROWS, seed=12)
    want = attn_tail(latent, mla.v_b_proj, mla.o_proj.weight, ROWS, config=cfg)
    tail = _tail(cfg, mla, spy)
    got = _attn_tail_call(tail, mla, latent, _params(enable), hop_transport=_transport())
    torch.cuda.synchronize()
    assert tail.take_into_hop_published()
    assert spy.calls == []
    assert torch.equal(got, want)


def _native_engine_call(tail, mla, latent, params):
    """What the op and MLA.forward do with attention_output_tail_prefetch's ``native`` engine."""
    from tensorrt_llm._torch.attention.mla import MLA

    rows = latent.shape[0]
    attn_out = torch.zeros((rows, 2048), dtype=torch.bfloat16, device="cuda")
    meta = tail_adapter._meta(rows)
    assert tail.arm(mla, meta)
    tail.before_attention(mla, point="attn")
    assert not tail.defer_w_uv(mla, meta, latent, attn_out)
    v_view = attn_out.view(rows, mla.num_heads_tp, mla.v_head_dim)
    MLA._bmm_bf16_out(
        mla,
        latent.transpose(0, 1),
        mla.v_b_proj,
        mla.v_b_proj.transpose(1, 2),
        v_view.transpose(0, 1),
    )
    return tail.project(mla, attn_out, params), attn_out


@pytest.mark.parametrize("enable", [True, False], ids=["allreduce_on", "allreduce_fused"])
@pytest.mark.parametrize("join", ["before_oproj", "after_oproj"])
@pytest.mark.parametrize("oproj", ["cublas", "ring"])
def test_native_engine_returns_gemm_output_without_communication(oproj, join, enable):
    mla, spy = _reducing_mla()
    cfg = _R(TAIL_PREFETCH_STAGED_NATIVE_SCHEDULE, native_oproj=oproj, prefetch_join=join)
    tail = _tail(cfg, mla, spy)
    latent = tail_adapter._latent(ROWS, seed=13)
    got, attn_out = _native_engine_call(tail, mla, latent, _params(enable))
    gemm = mla.o_proj.apply_linear(attn_out, None)
    torch.cuda.synchronize()
    assert spy.calls == []
    assert tail.engaged_calls == 1 and tail.prefetch_forks == 1
    assert not tail._prefetch.forked
    if oproj == "cublas":
        assert torch.equal(got, gemm)
    else:
        ref = attn_out.double() @ mla.o_proj.weight.double().t()
        e_ring = tail_reference.error_stats(got, ref)
        e_nat = tail_reference.error_stats(gemm, ref)
        assert e_ring["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6
        assert e_ring["max_abs"] <= (2 * e_nat["max_abs"] + 2**-8 * e_ring["ref_max"])


@pytest.mark.parametrize("enable", [True, False], ids=["allreduce_on", "allreduce_fused"])
@pytest.mark.parametrize(
    "name", ["attention_output_tail", "into_hop_sparse_mla", "attention_output_tail_prefetch"]
)
def test_graph_capture_enqueues_no_communication(name, enable):
    cfg = ATTN_TAIL_KERNEL_CONFIGS[name]
    mla, spy = _reducing_mla()
    tail = _tail(cfg, mla, spy)
    lat = tail_adapter._latent(ROWS, seed=14).contiguous()
    params = _params(enable)
    _attn_tail_call(tail, mla, lat, params)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _attn_tail_call(tail, mla, lat, params)
    assert spy.calls == []
    for seed in range(4):
        lat.copy_(tail_adapter._latent(ROWS, seed=140 + seed))
        graph.replay()
        torch.cuda.synchronize()
        want = attn_tail(lat, mla.v_b_proj, mla.o_proj.weight, ROWS, config=cfg)
        assert torch.equal(captured, want)
    assert spy.calls == []
    del graph
