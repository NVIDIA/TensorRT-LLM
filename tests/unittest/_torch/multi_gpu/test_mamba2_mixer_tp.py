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
"""Multi-GPU equivalence test for Mamba2 tensor-parallel group replication.

Compares a TP>1 ``Mamba2Mixer`` prefill against a TP=1 reference built from
the same full-precision weights: the output, the per-rank SSM (temporal)
cache state, and the per-rank conv cache state must each match the
corresponding slice of the reference run.

The conv-state check slices the B/C rows by *group*, not by an equal
``tp_size``-way split of the ``[x | B | C]`` channels, because the two
Mamba2 TP layouts (see ``mamba2_tp.py``) disagree on what "this rank's B/C
rows" means:

* ``tp_size <= n_groups``: groups split evenly, each rank owns
  ``n_groups // tp_size`` whole groups, and a plain ``tp_size``-way row
  split of the reference's B/C rows lines up with the rank's shard.
* ``tp_size > n_groups``: one group is replicated across the
  ``tp_size // n_groups`` ranks whose heads belong to it, so every rank in
  that replica set narrows to the *same* ``d_state``-wide group slice
  (``shard.group_index(rank)``) instead of taking a disjoint split.

Runs on 2 and 4 GPUs via MPI; on single-GPU development boxes the module
still collects normally and every test is skipped.
"""

import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import PythonMambaCacheManager
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

D_MODEL, HEAD_DIM, D_STATE, D_CONV, CHUNK = 64, 16, 16, 4, 8
SEQ_LENS = [5, 7]


def _full_weights(nheads, n_groups, seed=0):
    g = torch.Generator().manual_seed(seed)
    d_inner = nheads * HEAD_DIM
    grouped = n_groups * D_STATE
    d_in_proj = 2 * d_inner + 2 * grouped + nheads

    def rn(*shape):
        return torch.randn(*shape, generator=g) * 0.1

    return dict(
        in_proj=rn(d_in_proj, D_MODEL),
        conv_w=rn(d_inner + 2 * grouped, D_CONV),
        conv_b=rn(d_inner + 2 * grouped),
        A_log=torch.rand(nheads, generator=g) + 0.5,
        D=torch.rand(nheads, generator=g) + 0.5,
        dt_bias=rn(nheads),
        norm_w=torch.rand(d_inner, generator=g) + 0.5,
        out_proj=rn(D_MODEL, d_inner),
    )


def _build_mixer(tp, rank, nheads, n_groups, w):
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    cfg = ModelConfig(mapping=mapping, allreduce_strategy=AllReduceStrategy.NCCL)
    mixer = Mamba2Mixer(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        nheads=nheads,
        n_groups=n_groups,
        head_dim=HEAD_DIM,
        chunk_size=CHUNK,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=cfg,
    )
    s = mixer.shard
    bf = torch.bfloat16
    mixer.in_proj.load_weights([dict(weight=s.rearrange_in_proj_rows(w["in_proj"].to(bf)))])
    mixer.conv1d.load_weights(
        [
            dict(
                weight=s.rearrange_conv1d_rows(w["conv_w"].to(bf)),
                bias=s.rearrange_conv1d_rows(w["conv_b"].to(bf)),
            )
        ]
    )
    mixer.out_proj.load_weights(
        [dict(weight=w["out_proj"].to(bf))]
    )  # ROW split along dim 1 = heads
    mixer.A.data.copy_(s.shard_heads(-torch.exp(w["A_log"]), rank))
    mixer.D.data.copy_(s.shard_heads(w["D"], rank))
    mixer.dt_bias.data.copy_(s.shard_heads(w["dt_bias"], rank))
    mixer.norm.weight.data.copy_(s.shard_heads(w["norm_w"], rank).to(bf))
    mixer.cuda()
    mixer.post_load_weights()
    return mixer, mapping


def _run_prefill(mixer, mapping, nheads, n_groups, hidden):
    n = len(SEQ_LENS)
    # The manager reserves padding/dummy slots out of its free pool, so size
    # it with slack above the number of real requests.
    slots = n + 4
    mgr = PythonMambaCacheManager(
        d_state=D_STATE,
        d_conv=D_CONV,
        num_heads=nheads,
        n_groups=n_groups,
        head_dim=HEAD_DIM,
        num_layers=1,
        max_batch_size=slots,
        spec_state_size=slots,
        mapping=mapping,
        dtype=torch.bfloat16,
        # Matches the mixer, whose ModelConfig() leaves mamba_ssm_cache_dtype
        # unset so SSM states stay in the activation dtype.
        ssm_cache_dtype=torch.bfloat16,
    )
    request_ids = list(range(n))
    mgr._prepare_mamba_cache_blocks(request_ids)
    seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int)
    attn = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=n,
        num_ctx_tokens=sum(SEQ_LENS),
        num_tokens=sum(SEQ_LENS),
        kv_cache_manager=mgr,
        request_ids=request_ids,
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=torch.zeros(n, dtype=torch.int)),
    )
    md = Mamba2Metadata(max_batch_size=slots, chunk_size=CHUNK)
    md.prepare(attn)
    with torch.inference_mode():
        out = mixer(hidden, attn, md)
    slots = mgr.get_state_indices(request_ids, [False] * n)
    cache = mgr.mamba_layer_cache(0)
    return out, cache.conv[slots].clone(), cache.temporal[slots].clone()


def _worker(tp, nheads, n_groups):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        w = _full_weights(nheads, n_groups)
        hidden = (
            (torch.randn(sum(SEQ_LENS), D_MODEL, generator=torch.Generator().manual_seed(1)) * 0.5)
            .to(torch.bfloat16)
            .cuda()
        )
        ref_mixer, ref_map = _build_mixer(1, 0, nheads, n_groups, w)
        ref_out, ref_conv, ref_ssm = _run_prefill(ref_mixer, ref_map, nheads, n_groups, hidden)
        assert ref_out.abs().mean() > 0.05, "reference output is degenerate"
        mixer, mapping = _build_mixer(tp, rank, nheads, n_groups, w)
        out, conv, ssm = _run_prefill(mixer, mapping, nheads, n_groups, hidden)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref_out, atol=5e-2, rtol=5e-2)
        s = mixer.shard
        heads = slice(rank * s.tp_nheads, (rank + 1) * s.tp_nheads)
        torch.testing.assert_close(ssm, ref_ssm[:, heads], atol=1e-2, rtol=1e-2)
        d_inner = nheads * HEAD_DIM
        x_rows = slice(rank * s.tp_d_inner, (rank + 1) * s.tp_d_inner)
        if s.replicated:
            g = s.group_index(rank)
            b_rows = slice(d_inner + g * D_STATE, d_inner + (g + 1) * D_STATE)
            c_rows = slice(
                d_inner + n_groups * D_STATE + g * D_STATE,
                d_inner + n_groups * D_STATE + (g + 1) * D_STATE,
            )
        else:
            gw = s.tp_grouped_state_dim
            b_rows = slice(d_inner + rank * gw, d_inner + (rank + 1) * gw)
            c_rows = slice(
                d_inner + n_groups * D_STATE + rank * gw,
                d_inner + n_groups * D_STATE + (rank + 1) * gw,
            )
        expected_conv = torch.cat(
            [ref_conv[:, x_rows], ref_conv[:, b_rows], ref_conv[:, c_rows]], dim=1
        )
        torch.testing.assert_close(conv, expected_conv, atol=1e-2, rtol=1e-2)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize("nheads, n_groups", [(8, 1), (8, 2)], ids=["replicated-r2", "even-groups"])
def test_mamba2_mixer_tp2(nheads, n_groups, mpi_pool_executor):
    results = mpi_pool_executor.map(_worker, *zip(*[(2, nheads, n_groups)] * 2))
    assert all(results)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
@pytest.mark.parametrize(
    "nheads, n_groups", [(8, 2), (8, 1)], ids=["replicated-r2", "replicated-r4"]
)
def test_mamba2_mixer_tp4(nheads, n_groups, mpi_pool_executor):
    results = mpi_pool_executor.map(_worker, *zip(*[(4, nheads, n_groups)] * 4))
    assert all(results)
