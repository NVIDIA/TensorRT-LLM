# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-parallel family projections sum over the TP group as the engine modules do, on one GB300 tray.

Each case runs on four MPI ranks: every rank builds the engine ``Linear`` / ``GatedMLP`` and the family ``K31Linear``
/ ``K31SituMLP`` with the same mapping and strategy, loads the same shard into both, and compares their outputs bit
for bit (and the bits across ranks). The cases are the row-parallel Kimi K3.1 projections at TP4 shapes: the MLA
``o_proj``, the shared experts and the dense MLP.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]

# (name, kind, full in / intermediate features): o_proj 96 heads x 128, shared experts 2 x 3072, dense MLP 33792.
CASES = (
    ("o_proj", "linear", 12288),
    ("shared experts", "mlp", 6144),
    ("dense MLP", "mlp", 33792),
)
ROWS = (1, 5, 8, 40, 300, 2048)


def _rank_case(kind: str, features: int, rows_list) -> dict:
    """One rank's half of a case; runs inside an MPI pool worker."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import (
        K31Linear,
        K31SituMLP,
    )
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.distributed import AllReduceParams, AllReduceStrategy
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
    from tensorrt_llm._torch.modules.situ import SituAndMul
    from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
    from tensorrt_llm.mapping import Mapping

    bf16 = torch.bfloat16
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
    AutoTuner.get().setup_distributed_state(mapping)
    strategy = AllReduceStrategy.AUTO
    gen = torch.Generator().manual_seed(features)
    if kind == "linear":
        engine = Linear(
            features,
            7168,
            bias=False,
            dtype=bf16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            allreduce_strategy=strategy,
        ).cuda()
        ours = K31Linear(
            features, 7168, dtype=bf16, mapping=mapping, tp_mode="row", allreduce_strategy=strategy
        ).cuda()
        full = (torch.randn(7168, features, generator=gen) * 0.02).to(bf16)
        engine.weight.data.copy_(engine.load_shard(full, device=torch.device("cuda")))
        ours.weight.data.copy_(ours.load_shard(full, device=torch.device("cuda")))
        pairs = [(engine, ours)]
    else:
        engine = GatedMLP(
            hidden_size=7168,
            intermediate_size=features,
            bias=False,
            activation=SituAndMul(beta=4.0, linear_beta=25.0, use_fused_activation=True),
            dtype=bf16,
            config=ModelConfig(mapping=mapping, allreduce_strategy=strategy),
            reduce_output=True,
            layer_idx=0,
        ).cuda()
        ours = K31SituMLP(
            7168,
            features,
            dtype=bf16,
            mapping=mapping,
            beta=4.0,
            linear_beta=25.0,
            allreduce_strategy=strategy,
        ).cuda()
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(engine, name).weight
            w.data.copy_((torch.randn(w.shape, generator=gen) * 0.02).to(bf16))
            getattr(ours, name).weight.data.copy_(w.data)
        pairs = [(engine, ours)]
    results = []
    for rows in rows_list:
        xg = torch.Generator(device="cuda").manual_seed(rows)
        local_k = features // WORLD if kind == "linear" else 7168
        x = torch.randn(rows, local_k, generator=xg, device="cuda").to(bf16)
        if kind == "linear":
            # Every rank's partial input differs, as each holds its own heads.
            x = x + rank
        for engine_mod, our_mod in pairs:
            with torch.inference_mode(), discard_nccl_window_tensor_outputs((x,)):
                expected = engine_mod(x)
                actual = our_mod(x)
                again = our_mod(x)
                equal = bool(torch.equal(actual, expected)) and bool(torch.equal(again, expected))
                no_sum = AllReduceParams(enable_allreduce=False)
                if kind == "linear":
                    engine_partial = engine_mod(x, all_reduce_params=no_sum)
                else:
                    engine_partial = engine_mod(x, final_all_reduce_params=no_sum)
                partial_equal = bool(torch.equal(our_mod(x, reduce=False), engine_partial))
                bits = expected.contiguous().view(torch.int16).to(torch.int64)
                weights = torch.arange(bits.numel(), device="cuda") % 65521 + 1
                digest = int((bits.flatten() * weights).sum())
                torch.cuda.synchronize()
                del expected, actual, again, engine_partial
            results.append(dict(rank=rank, rows=rows, equal=equal and partial_equal, hash=digest))
    return results


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("name, kind, features", CASES, ids=[c[0] for c in CASES])
def test_row_parallel_sum_matches_the_engine(name, kind, features, mpi_pool_executor) -> None:
    per_rank = list(mpi_pool_executor.map(_rank_case, *zip(*[(kind, features, ROWS)] * WORLD)))
    for results in per_rank:
        for r in results:
            assert r["equal"], (
                f"{name}: rank {r['rank']}, {r['rows']} rows: holder bits differ from the engine's"
            )
    for i, rows in enumerate(ROWS):
        assert len({results[i]["hash"] for results in per_rank}) == 1, (
            f"{name}, {rows} rows: ranks differ"
        )
