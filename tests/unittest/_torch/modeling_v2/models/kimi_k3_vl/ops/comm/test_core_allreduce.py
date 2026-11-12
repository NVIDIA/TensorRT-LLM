# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``layers`` all-reduce, gated MLP (dense and MoE shared expert), row-parallel BF16
projection and vocab-parallel embedding over 2 / 4 ranks against main's ``AllReduce`` /
``GatedMLP`` / ``Linear`` / ``Embedding``: bitwise, on every rank.

``TLLM_TEST_MNNVL=1`` makes both sides take MNNVL on one node, as AUTO does across nodes; the
pushed all-reduce of ``layers.py`` runs on the primitive's handle against main's result."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import skip_pre_blackwell

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

_K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"


def _same(a, b) -> int:
    """Number of differing elements (bit patterns), 0 when equal."""
    if isinstance(a, (tuple, list)):
        return sum(_same(x, y) for x, y in zip(a, b)) + abs(len(a) - len(b))
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel())
    if a.dtype.is_floating_point and a.element_size() == 2:
        return int((a.view(torch.int16) != b.view(torch.int16)).sum())
    return int((a != b).sum())


def _run_rank(world_size, mnnvl):
    if mnnvl:
        os.environ["TLLM_TEST_MNNVL"] = "1"
    else:
        os.environ.pop("TLLM_TEST_MNNVL", None)
    import importlib

    from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.embedding import Embedding
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
    from tensorrt_llm._torch.modules.situ import SituAndMul
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    core_ar = importlib.import_module(f"{_K3}.ops.comm.sm107_allreduce")
    core_mlp = importlib.import_module(f"{_K3}.ops.activation.sm107_mlp")
    core_emb = importlib.import_module(f"{_K3}.ops.embedding.sm107_embedding")
    core_linear = importlib.import_module(f"{_K3}.ops.gemm.sm107_linear")
    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    bad = {}
    try:
        mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
        strategies = (
            [AllReduceStrategy.AUTO]
            if mnnvl
            else [AllReduceStrategy.NCCL, AllReduceStrategy.ONESHOT]
        )
        for strategy in strategies:
            ref = AllReduce(mapping=mapping, strategy=strategy, dtype=torch.bfloat16)
            prim = core_ar.TpAllReduce(mapping, torch.bfloat16, strategy)
            assert (ref.mnnvl_allreduce is not None) == (prim.transport == "mnnvl") == mnnvl
            gen = torch.Generator(device="cuda").manual_seed(rank)
            # One-shot and two-shot MNNVL sizes (threshold 1 MiB of tokens * hidden * ranks * 2);
            # 3584: the routed MoE latent (RMS_NORM fusion).
            shapes = ((1, 7168), (8, 7168), (32, 7168), (64, 7168), (300, 7168), (5, 512))
            for m, h in (*shapes, (16, 3584), (200, 3584)):
                x = (torch.randn(m, h, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
                key = f"{strategy.name}/none/{m}x{h}"
                bad[key] = _same(prim(x), ref(x))
                if not mnnvl:
                    continue
                w = (torch.randn(h, device="cuda", generator=gen)).to(torch.bfloat16)
                r = (torch.randn(m, h, device="cuda", generator=gen)).to(torch.bfloat16)
                for op in (AllReduceFusionOp.RESIDUAL_RMS_NORM, AllReduceFusionOp.RMS_NORM):
                    params = AllReduceParams(
                        fusion_op=op,
                        residual=r if op == AllReduceFusionOp.RESIDUAL_RMS_NORM else None,
                        norm_weight=w,
                        eps=1e-6,
                    )
                    bad[f"{key}/{op.name}"] = _same(
                        prim(x, all_reduce_params=params), ref(x, all_reduce_params=params)
                    )

        if mnnvl:
            # FP32, as the helix vocab-parallel pair exchange runs main's MNNVLAllReduce directly.
            from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

            ref = MNNVLAllReduce(mapping, torch.float32)
            prim = core_ar.TpAllReduce(mapping, torch.float32)
            assert prim.transport == "mnnvl"
            for m, h in ((4, 64), (8, 10240), (100, 7168)):
                x = torch.randn(m, h, device="cuda")
                want = ref(x, all_reduce_params=AllReduceParams())
                bad[f"fp32/none/{m}x{h}"] = _same(prim(x), want)

            # The pushed all-reduce (ops/comm/sm107_allreduce.py) on the primitive's handle.
            lamport = importlib.import_module(f"{_K3}.ops.comm.sm107_allreduce")
            ref = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
            prim = core_ar.TpAllReduce(mapping, torch.bfloat16)
            for m in (1, 8, 100):
                x = torch.randn(m, 7168, device="cuda").to(torch.bfloat16)
                want = ref(x)
                target = lamport.lamport_push_target(prim, m, 7168)
                assert target is not None
                lamport.lamport_push_tensor(prim, target, x)
                bad[f"lamport/{m}"] = _same(lamport.finish_lamport_push(prim, target), want)

        # The SiTU MLP: the MoE shared expert (the model's TP group, is_shared_expert) and the dense
        # MLP over the full group and, for 4 ranks, its TP-2 subgroups; reduced, and as the
        # partial (enable_allreduce=False). Without MNNVL both sides take NCCL (main autotunes
        # AUTO's strategy there, the primitive does not).
        strategy = AllReduceStrategy.AUTO if mnnvl else AllReduceStrategy.NCCL
        no_reduce = AllReduceParams(enable_allreduce=False)
        torch.manual_seed(0)
        hidden, inter = 2048, 3072
        gate = (torch.randn(inter, hidden, device="cuda") * 0.02).to(torch.bfloat16)
        up = (torch.randn(inter, hidden, device="cuda") * 0.02).to(torch.bfloat16)
        down = (torch.randn(hidden, inter, device="cuda") * 0.02).to(torch.bfloat16)
        for sub in (None, *sorted({world_size, 2})):
            with torch.device("cuda"):
                ref = GatedMLP(
                    hidden_size=hidden,
                    intermediate_size=inter,
                    bias=False,
                    activation=SituAndMul(beta=1.0, linear_beta=8.0, use_fused_activation=True),
                    dtype=torch.bfloat16,
                    config=ModelConfig(mapping=mapping, allreduce_strategy=strategy),
                    overridden_tp_size=sub,
                    reduce_output=True,
                    is_shared_expert=sub is None,
                )
            ref.gate_up_proj.load_weights([{"weight": gate}, {"weight": up}])
            ref.down_proj.load_weights([{"weight": down}])
            prim = core_mlp.GatedMlp(
                hidden,
                inter,
                activation="situ",
                beta=1.0,
                linear_beta=8.0,
                mapping=mapping if sub is None else core_mlp.subgroup_mapping(mapping, sub),
                allreduce_strategy=strategy,
                device="cuda",
            )
            prim.load(gate, up, down)
            tag = "shared" if sub is None else f"tp{sub}"
            down_ar, ref_ar = prim.down_proj.all_reduce, ref.down_proj.all_reduce
            assert prim.down_proj.reduce_output == ref.down_proj.reduce_output
            assert (down_ar.mnnvl_allreduce is None) == (ref_ar.mnnvl_allreduce is None)
            assert down_ar.mapping.tp_group == ref_ar.mapping.tp_group
            for m in (1, 16, 200):
                x = torch.randn(m, hidden, device="cuda").to(torch.bfloat16)
                bad[f"mlp/{tag}/{m}"] = _same(prim(x), ref(x))
                bad[f"mlp/{tag}/partial/{m}"] = _same(
                    prim(x, final_all_reduce_params=no_reduce),
                    ref(x, final_all_reduce_params=no_reduce),
                )

        # A BF16 row-parallel projection with its all-reduce (the drafter's o_proj): plain, the
        # partial, and the residual RMSNorm fusion.
        hidden, width = 7168, 1024 * world_size
        weight = (torch.randn(hidden, width, device="cuda") * 0.02).to(torch.bfloat16)
        with torch.device("cuda"):
            ref = Linear(
                width,
                hidden,
                bias=False,
                dtype=torch.bfloat16,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.ROW,
                reduce_output=True,
                allreduce_strategy=strategy,
            )
        ref.load_weights([{"weight": weight}])
        prim = core_linear.Bf16Linear(
            width,
            hidden,
            tp_size=world_size,
            tp_rank=rank,
            tp_mode=TensorParallelMode.ROW,
            all_reduce=core_ar.TpAllReduce(mapping, torch.bfloat16, strategy),
            device="cuda",
        )
        prim.load(weight)
        norm_w = torch.randn(hidden, device="cuda").to(torch.bfloat16)
        for m in (1, 8, 64):
            x = torch.randn(m, prim.in_features, device="cuda").to(torch.bfloat16)
            res = torch.randn(m, hidden, device="cuda").to(torch.bfloat16)
            fused = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=res,
                norm_weight=norm_w,
                eps=1e-6,
            )
            for name, params in (("plain", None), ("partial", no_reduce), ("res_norm", fused)):
                bad[f"row/{name}/{m}"] = _same(
                    prim(x, all_reduce_params=params), ref(x, all_reduce_params=params)
                )

        # Vocab-parallel embedding with its all-reduce.
        vocab = 1003
        table = torch.randn(vocab, 512, device="cuda").to(torch.bfloat16)
        with torch.device("cuda"):
            ref = Embedding(
                vocab,
                512,
                dtype=torch.bfloat16,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                allreduce_strategy=strategy,
            )
        ref.load_weights([{"weight": table}])
        prim = core_emb.EmbeddingTable(
            vocab,
            512,
            dtype=torch.bfloat16,
            mapping=mapping,
            vocab_parallel=True,
            allreduce_strategy=strategy,
            device="cuda",
        )
        prim.load(table)
        ids = torch.tensor([0, 5, vocab - 1, -1, 400, 251, 252, 753], device="cuda")
        bad["embedding"] = _same(prim(ids), ref(ids))
    except Exception:
        traceback.print_exc()
        raise
    return bad


@skip_pre_blackwell
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("mnnvl", [True, False], ids=["mnnvl", "op"])
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_core_collectives_match_main(mpi_pool_executor, mnnvl):
    world_size = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    results = list(mpi_pool_executor.map(_run_rank, *zip(*[(world_size, mnnvl)] * world_size)))
    for rank, bad in enumerate(results):
        failed = {key: n for key, n in bad.items() if n}
        assert not failed, f"rank {rank}: {failed}"
        assert bad, f"rank {rank} compared nothing"
