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
"""DSpark drafter all-reduce sites: the pushed all-reduce + residual +
RMSNorm site kernel and the split-K GEMM push epilogue are bitwise the C++ MNNVL fused
all-reduce of the same partials, with C++ calls interleaved on the workspace and under CUDA-graph
replay; the owner-multicast embedding site is bitwise the vocab-sharded lookup + all-reduce. The
Lamport slots come from the Kimi K3 package's pushed all-reduce (``ops/comm/sm107_allreduce.py``)."""

import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168
EPS = 1e-6


def _input(m, h, seed, rank, special=False):
    gen = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    x = (torch.randn(m, h, device="cuda", generator=gen) * 0.25).to(torch.bfloat16)
    if special:
        bits = x.view(torch.int16)
        r = torch.rand(m, h, device="cuda", generator=gen)
        bits[r < 0.05] = -32768  # -0
        bits[(r >= 0.05) & (r < 0.08)] = 0
        den = (r >= 0.08) & (r < 0.1)
        bits[den] = torch.randint(1, 127, (int(den.sum()),), device="cuda", generator=gen).to(
            torch.int16
        )
    return x


def _differ(ref, out):
    return int((ref.view(torch.int16) != out.view(torch.int16)).sum())


def _run_rank(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.primitives import (  # noqa: E501
        DraftArSites,
        push_tactic,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_gemm.low_m_bf16_splitk import (  # noqa: E501
        run_splitk_dense,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_spec.dspark_draft_ar import (  # noqa: E501
        ar_norm_config,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_spec.dspark_draft_gemm_push import (  # noqa: E501
        run_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.sm107_allreduce import (  # noqa: E501  # noqa: E501
        TpAllReduce,
        lamport_push_target,
        lamport_push_tensor,
        mnnvl_lamport_push_target,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec.k3_draft_ar_norm import (  # noqa: E501
        k3_draft_ar_norm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec.k3_draft_gemm_push import (  # noqa: E501
        k3_draft_gemm_push,
    )
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        allreduce = TpAllReduce(mapping, torch.bfloat16, AllReduceStrategy.MNNVL)
        assert allreduce.mnnvl_allreduce is not None
        gen = torch.Generator(device="cuda").manual_seed(5)
        gamma = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=gen)).to(torch.bfloat16)
        norm = SimpleNamespace(weight=gamma, variance_epsilon=EPS)
        bad = 0

        def reference(partial, residual):
            return allreduce(
                partial,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=gamma,
                    eps=EPS,
                ),
            )

        def site(target, residual):
            out = torch.empty_like(residual)
            res_out = torch.empty_like(residual)
            k3_draft_ar_norm(
                target.workspace,
                target.one_shot,
                target.cpp_one_shot,
                residual,
                gamma,
                EPS,
                out,
                res_out,
            )
            return out, res_out

        # Push kernel + site against the C++ fused all-reduce, both transports, C++ calls between.
        for m in (1, 7, 8, 14, 16, 21, 28, 32, 56, 64, 112, 128):
            if not ar_norm_config(m, HIDDEN, torch.device("cuda", rank))[0]:
                continue
            for special in (False, True):
                x = _input(m, HIDDEN, 7 * m + special, rank, special)
                res = _input(m, HIDDEN, 11 * m + special, 0, special)
                ref_out, ref_res = reference(x, res)
                for one_shot in (True, False):
                    target = mnnvl_lamport_push_target(
                        allreduce.mnnvl_allreduce, m, HIDDEN, one_shot=one_shot
                    )
                    lamport_push_tensor(allreduce, target, x)
                    out, res_out = site(target, res)
                    bad += _differ(ref_out, out) + _differ(ref_res, res_out)
                    again_out, again_res = reference(x, res)
                    bad += _differ(ref_out, again_out) + _differ(ref_res, again_res)

        # Split-K GEMM push epilogue + site against the same GEMM's store-only variant (== the
        # vendored split-K kernel up to 32 rows) + C++ fused all-reduce.
        for role, k, m in (
            ("o_proj", 1024, 7),
            ("o_proj", 1024, 8),
            ("o_proj", 1024, 14),
            ("down", 1792, 16),
            ("down", 1792, 21),
            ("o_proj", 1024, 32),
            ("o_proj", 1024, 56),
            ("down", 1792, 56),
            ("down", 1792, 64),
        ):
            w = (torch.randn(HIDDEN, k, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
            tactic = push_tactic(role, m, HIDDEN, k)
            assert tactic is not None, (role, m)
            x = _input(m, k, 13 * m + k, rank)
            res = _input(m, HIDDEN, 17 * m, 0)
            y = run_gemm(x, w, torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16), tactic)
            if m <= 32:
                y_vendored = torch.empty_like(y)
                run_splitk_dense(x, w.t(), None, y_vendored, True, tactic)
                bad += _differ(y_vendored, y)
            ref_out, ref_res = reference(y, res)
            target = lamport_push_target(allreduce, m, HIDDEN)
            k3_draft_gemm_push(x, w, target.output, target, tactic, store_out=True)
            bad += _differ(y, target.output)
            out, res_out = site(target, res)
            bad += _differ(ref_out, out) + _differ(ref_res, res_out)
            sites = DraftArSites(allreduce)
            out, res_out = sites.linear_ar_norm(x, w, role, res, norm)
            bad += _differ(ref_out, out) + _differ(ref_res, res_out)
            # The split-K GEMM is a reordered accumulation of the cuBLAS one.
            torch.testing.assert_close(y.float(), (x.float() @ w.float().t()), atol=2e-2, rtol=2e-2)

        # One CUDA graph: o_proj site -> C++ fused all-reduce -> down site, replayed on new inputs.
        m = 14
        w_o = (torch.randn(HIDDEN, 1024, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        w_d = (torch.randn(HIDDEN, 1792, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        x_o = _input(m, 1024, 1, rank)
        x_d = _input(m, 1792, 2, rank)
        p_c = _input(m, HIDDEN, 3, rank)
        res = _input(m, HIDDEN, 4, 0)
        sites = DraftArSites(allreduce)

        def chain():
            a, r = sites.linear_ar_norm(x_o, w_o, "o_proj", res, norm)
            b, r = reference(p_c, r)
            c, r = sites.linear_ar_norm(x_d, w_d, "down", r, norm)
            return a, b, c, r

        def expected():
            y_o = run_gemm(
                x_o,
                w_o,
                torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16),
                push_tactic("o_proj", m, HIDDEN, 1024),
            )
            a, r = reference(y_o, res)
            b, r = reference(p_c, r)
            y_d = run_gemm(
                x_d,
                w_d,
                torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16),
                push_tactic("down", m, HIDDEN, 1792),
            )
            c, r = reference(y_d, r)
            return a, b, c, r

        chain()
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = chain()
        for rep in range(3):
            x_o.copy_(_input(m, 1024, 100 + rep, rank))
            x_d.copy_(_input(m, 1792, 200 + rep, rank))
            p_c.copy_(_input(m, HIDDEN, 300 + rep, rank))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(outs, expected()):
                bad += _differ(want, got)
        del graph
        torch.cuda.synchronize()
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ from the C++ path"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _run_rank_embed(world_size):
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.primitives import (  # noqa: E501
        draft_embed_site,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.embedding.sm107_embedding import (  # noqa: E501
        EmbeddingTable,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.spec.k3_vocab_shard_embedding import (  # noqa: E501
        k3_vocab_shard_embedding,
    )

    def vocab_shard_embedding(embedding, ids):
        local = k3_vocab_shard_embedding(
            ids, embedding.weight, embedding.vocab_start, embedding.vocab_end
        )
        return embedding.all_reduce(local)

    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        vocab = 1000 * world_size - 3  # a shorter last shard
        embedding = EmbeddingTable(
            vocab,
            HIDDEN,
            dtype=torch.bfloat16,
            mapping=mapping,
            vocab_parallel=True,
            device="cuda",
        )
        gen = torch.Generator(device="cuda").manual_seed(9)
        full = torch.randn(vocab, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
        full.view(torch.int16)[:, ::7] = -32768  # -0 lanes: +0 after the all-reduce
        embedding.load(full)
        bad = 0
        for n in (1, 2, 9, 17, 129):
            for dtype in (torch.int64, torch.int32):
                ids = torch.randint(0, vocab, (n,), device="cuda", generator=gen).to(dtype)
                ids[0] = vocab - 1
                if n > 2:
                    ids[1] = -1  # outside the vocab: a zero row
                    ids[2] = vocab
                expected = vocab_shard_embedding(embedding, ids)
                out = draft_embed_site(embedding, ids)
                assert out is not None
                bad += _differ(expected, out)
        # CUDA graph replay.
        ids = torch.randint(0, vocab, (9,), device="cuda", generator=gen)
        draft_embed_site(embedding, ids)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = draft_embed_site(embedding, ids)
        for rep in range(3):
            ids.copy_(torch.randint(0, vocab, (9,), device="cuda", generator=gen))
            torch.cuda.synchronize()
            comm.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            bad += _differ(vocab_shard_embedding(embedding, ids), out)
        del graph
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _run_rank_mlp(world_size):
    """The drafter glue: a TP GatedMlp whose down projection is deferred to the pushed site
    (``_draft_mlp(defer_down=True)`` + ``_reduce_partial``) against the split-K down GEMM + C++
    fused all-reduce, and the fallback above the split-K rows against the module path."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.drafter import (  # noqa: E501
        K3DSparkDrafter,
        _draft_mlp,
        _PendingDown,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.primitives import (  # noqa: E501
        DraftArSites,
        push_tactic,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_spec.dspark_draft_gemm_push import (  # noqa: E501
        run_gemm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation.sm107_mlp import (  # noqa: E501  # noqa: E501  # noqa: E501
        GatedMlp,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.sm107_allreduce import (
        TpAllReduce,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.sm107_norm import (
        RMSNormWeight,
    )
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        fused_ar = TpAllReduce(mapping, torch.bfloat16, AllReduceStrategy.MNNVL)
        mlp = GatedMlp(
            HIDDEN,
            1792 * world_size,
            activation="silu",
            dtype=torch.bfloat16,
            mapping=mapping,
            allreduce_strategy=AllReduceStrategy.MNNVL,
            device="cuda",
        )
        gen = torch.Generator(device="cuda").manual_seed(40 + rank)
        with torch.no_grad():
            mlp.gate_up_proj.weight.copy_(
                torch.randn(mlp.gate_up_proj.weight.shape, device="cuda", generator=gen) * 0.02
            )
            mlp.down_proj.weight.copy_(
                torch.randn(mlp.down_proj.weight.shape, device="cuda", generator=gen) * 0.02
            )
        norm = RMSNormWeight(HIDDEN, EPS, torch.bfloat16, device="cuda")
        stub = SimpleNamespace(_fused_ar_residual_norm=K3DSparkDrafter._fused_ar_residual_norm)
        sites = DraftArSites(fused_ar)
        no_reduce = AllReduceParams(enable_allreduce=False)
        bad = 0
        with torch.inference_mode():
            for m in (7, 14, 28, 56, 112):
                x = _input(m, HIDDEN, 50 + m, 0)
                res = _input(m, HIDDEN, 60 + m, 0)
                module_partial = _draft_mlp(mlp, x, no_reduce)
                ref_module = K3DSparkDrafter._reduce_partial(
                    stub, fused_ar, None, module_partial, res, norm
                )
                pending = _draft_mlp(mlp, x, no_reduce, defer_down=True)
                assert isinstance(pending, _PendingDown)
                out = K3DSparkDrafter._reduce_partial(stub, fused_ar, sites, pending, res, norm)
                k = pending.h2.shape[1]
                tactic = push_tactic("down", m, HIDDEN, k)
                if tactic is None:
                    want = ref_module  # no pushed site above the split-K rows
                else:
                    y = run_gemm(
                        pending.h2,
                        mlp.down_proj.weight.detach(),
                        torch.empty(m, HIDDEN, device="cuda", dtype=torch.bfloat16),
                        tactic,
                    )
                    want = fused_ar(
                        y,
                        all_reduce_params=AllReduceParams(
                            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                            residual=res,
                            norm_weight=norm.weight,
                            eps=norm.variance_epsilon,
                        ),
                    )
                    torch.testing.assert_close(
                        out[0].float(), ref_module[0].float(), atol=0.1, rtol=0.05
                    )
                bad += _differ(want[0], out[0]) + _differ(want[1], out[1])
        assert comm.allreduce(bad) == 0, f"rank {rank}: {bad} values differ"
    except Exception:
        traceback.print_exc()
        raise
    return True


_needs = pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability()[0] != 10
    or os.environ.get("TLLM_TEST_SKIP_MNNVL") == "1",
    reason="needs two SM100-family GPUs with MNNVL",
)
_WORLD_SIZES = [2, 4] if torch.cuda.device_count() >= 4 else [2]


@_needs
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_dspark_draft_ar_sites(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    for r in mpi_pool_executor.map(_run_rank, *zip(*[(world,)] * world)):
        assert r is True


@_needs
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_dspark_draft_embed_site(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    for r in mpi_pool_executor.map(_run_rank_embed, *zip(*[(world,)] * world)):
        assert r is True


@_needs
@pytest.mark.parametrize("mpi_pool_executor", _WORLD_SIZES, indirect=True)
def test_dspark_draft_mlp_site(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    for r in mpi_pool_executor.map(_run_rank_mlp, *zip(*[(world,)] * world)):
        assert r is True
