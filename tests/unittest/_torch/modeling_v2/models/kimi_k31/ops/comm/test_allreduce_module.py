# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 allreduce_module entry: the engine AllReduce module on one GB300 tray.

Each cell runs on four MPI ranks (one tray), the way a served engine runs: every rank builds the
same `AllReduce` module the target builds (bf16, the cell's strategy), regenerates every rank's
input from the cell's seed, calls the entry on its own input and gates the result against the
entry's reference over all four inputs. The ranks report their deviation and a bit hash of their
output, so the parent also checks that every rank got the same tensor.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import pickle
import sys
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
    allreduce_module,
    tp_allreduce,
    uses_mnnvl,
)
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceStrategy

__extra_import_path__ = ["../../../.."]
from _validating import ULP, validating  # noqa: E402 — needs the path declared above

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    # The MPI pool is reused across cells; its manager thread outlives the first one.
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs (sm_107 driven separately)",
    ),
]

# Gate in bf16 ulps: per element against its row's largest magnitude, and the relative RMS.
# Measured on GB300 over every cell: the one-shot, two-shot and MNNVL NONE sums are bit-exact
# against the fp32 sum rounded once; NCCL (bf16 reduction steps) reaches 2.3 ulp element / 0.76
# ulp RMS; every RMS_NORM path, which rounds the normalized value as well as the product, reaches
# 2.8 ulp / 0.91 ulp (NCCL plus the separate norm kernel). A reduction that drops one rank's
# partial misses both bands by orders of magnitude (test_gate_refuses_a_dropped_rank).
ELEMENT_ULP, RMS_ULP = 4.0, 1.5


def _inputs(spec: dict, seed: int):
    """Every rank's input, in rank order, plus the norm weight: identical on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, hidden = spec["rows"], spec["hidden"]
    inputs = [
        torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)
        for _ in range(spec["world"])
    ]
    norm_weight = (1 + 0.1 * torch.randn(hidden, generator=gen, device="cuda")).to(torch.bfloat16)
    return inputs, norm_weight


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(worst element deviation against its row's scale, relative RMS deviation), in bf16 ulp."""
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / row).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    return element, rms / ULP[torch.bfloat16]


def _rank_cell(spec: dict, seed: int) -> dict:
    """One rank's half of a cell; runs inside an MPI pool worker."""
    import tensorrt_llm
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    # This function is pickled by value, so its own globals are fresh on every call; keep the
    # modules (and with them their workspaces) in a worker-process registry instead.
    modules = vars(
        sys.modules.setdefault("_k31_allreduce_modules", types.ModuleType("_k31_allreduce_modules"))
    ).setdefault("by_key", {})
    key = (spec["world"], spec["strategy"])
    if key not in modules:
        AutoTuner.get().setup_distributed_state(mapping)
        # What the target builds: tp_allreduce with the configured strategy (its int value, as the
        # cell carries it) and the default bf16.
        modules[key] = tp_allreduce(mapping, spec["strategy"])
    module = modules[key]
    if spec["strategy"] == int(AllReduceStrategy.MNNVL):
        assert uses_mnnvl(module), "the MNNVL cell did not build the MNNVL path"

    inputs, norm_weight = _inputs(spec, seed)
    fusion_op, eps = spec["fusion_op"], spec["eps"]
    args = (inputs[rank], module, fusion_op, norm_weight, eps)
    before = inputs[rank].clone()
    # Under MPI the AUTO path can hand back an NCCL-window tensor, which must be released inside
    # a window-tensor scope -- the target's decoder layers are such scopes. Everything that reads
    # the output stays inside it.
    with torch.inference_mode(), discard_nccl_window_tensor_outputs((inputs, norm_weight)):
        allreduce_module.is_valid(*args)
        if spec["strategy"] == int(AllReduceStrategy.AUTO):
            # A served engine tunes AUTO's kernel choice at warm-up; certify the tuned choice.
            with autotune():
                allreduce_module(*args)
        out = allreduce_module(*args)
        again = allreduce_module(*args)
        ref = allreduce_module.reference(inputs, fusion_op, norm_weight, eps)
        torch.cuda.synchronize()
        element, rms = _deviation(out, ref)
        bits = out.contiguous().view(torch.int16).to(torch.int64)
        weights = torch.arange(1, bits.numel() + 1, device="cuda", dtype=torch.int64) % 65521
        digest = int((bits.flatten() * weights).sum().item())
        repeat_equal = bool(torch.equal(out, again))
        fresh = out.data_ptr() != inputs[rank].data_ptr()
        shape, dtype = tuple(out.shape), str(out.dtype)
        del out, again
    return dict(
        rank=rank,
        shape=shape,
        dtype=dtype,
        element_ulp=element,
        rms_ulp=rms,
        hash=digest,
        repeat_equal=repeat_equal,
        input_untouched=bool(torch.equal(inputs[rank], before)),
        fresh=fresh,
    )


def _run(executor, spec: dict, seed: int) -> list:
    results = list(executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD)))
    return sorted(results, key=lambda r: r["rank"])


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", allreduce_module.CELLS, ids=[c.why[4:64] for c in allreduce_module.CELLS]
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    assert spec["world"] == WORLD
    results = _run(mpi_pool_executor, spec, zlib.crc32(cell.why.encode()) % 2**31)
    worst = max(r["element_ulp"] for r in results), max(r["rms_ulp"] for r in results)
    print(f"{cell.why}: element {worst[0]:.3f} ulp, rms {worst[1]:.3f} ulp")
    for r in results:
        assert r["shape"] == (spec["rows"], spec["hidden"]) and r["dtype"] == "torch.bfloat16", r
        assert r["fresh"] and r["input_untouched"], r
        assert r["repeat_equal"], f"rank {r['rank']}: two identical calls disagree"
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, (
            f"rank {r['rank']}: element {r['element_ulp']:.2f} ulp, rms {r['rms_ulp']:.2f} ulp"
        )
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def test_gate_refuses_a_dropped_rank() -> None:
    """The band must refuse a sum that leaves one rank's partial out, for both fusions."""
    spec = dict(world=WORLD, rows=8, hidden=3584, eps=1e-5)
    inputs, norm_weight = _inputs(spec, seed=1)
    for fusion_op in (int(AllReduceFusionOp.NONE), int(AllReduceFusionOp.RMS_NORM)):
        right = allreduce_module.reference(inputs, fusion_op, norm_weight, 1e-5)
        wrong = allreduce_module.reference(inputs[:-1], fusion_op, norm_weight, 1e-5)
        element, rms = _deviation(wrong, right)
        assert element > ELEMENT_ULP and rms > RMS_ULP, (fusion_op, element, rms)


def test_one_rank_group_returns_the_input() -> None:
    """A one-rank group is the identity, without a kernel: the target relies on it at TP1."""
    from tensorrt_llm.mapping import Mapping

    module = tp_allreduce(Mapping(world_size=1, tp_size=1, rank=0))
    x = torch.randn(4, 7168, device="cuda").to(torch.bfloat16)
    assert allreduce_module(x, module, int(AllReduceFusionOp.NONE), None, None) is x


def test_guard_refuses_an_uncertified_fusion() -> None:
    from tensorrt_llm.mapping import Mapping

    module = tp_allreduce(Mapping(world_size=1, tp_size=1, rank=0))
    x = torch.randn(4, 3584, device="cuda").to(torch.bfloat16)
    weight = torch.ones(3584, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="certified"):
        with validating(allreduce_module):
            allreduce_module(x, module, int(AllReduceFusionOp.RESIDUAL_RMS_NORM), weight, 1e-5)


def test_guard_refuses_a_short_norm_weight() -> None:
    from tensorrt_llm.mapping import Mapping

    module = tp_allreduce(Mapping(world_size=1, tp_size=1, rank=0))
    x = torch.randn(4, 3584, device="cuda").to(torch.bfloat16)
    weight = torch.ones(3584 // 2, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="norm_weight elements"):
        with validating(allreduce_module):
            allreduce_module(x, module, int(AllReduceFusionOp.RMS_NORM), weight, 1e-5)


def test_tp_allreduce_builds_the_configured_strategy_in_bf16() -> None:
    """None means AUTO; an int strategy is the enum's value; the workspaces are sized for bf16."""
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    for strategy, expected in (
        (None, AllReduceStrategy.AUTO),
        (int(AllReduceStrategy.NCCL), AllReduceStrategy.NCCL),
        (AllReduceStrategy.ONESHOT, AllReduceStrategy.ONESHOT),
    ):
        module = tp_allreduce(mapping, strategy)
        assert module.strategy == expected
    # A one-rank group builds no MNNVL path.
    assert not uses_mnnvl(tp_allreduce(mapping))
