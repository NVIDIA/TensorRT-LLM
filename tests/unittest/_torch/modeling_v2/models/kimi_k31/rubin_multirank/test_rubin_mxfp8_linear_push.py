# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (four ranks, one VR200 tray) for ``ops/gemm/rubin_mxfp8_linear_push``.

Every rank builds the engine MNNVL AllReduce on the shared workspace, regenerates every rank's
MXFP8 operands from the cell's seed (the activation in the prequantized entry's scale layout),
pushes its product into the call through the entry and finishes it with ``rubin_lamport_finish``:

* the entry pushes from the GEMM's epilogue at every cell (helix8's o_proj rows);
* the finish equals ``rubin_lamport_finish.reference`` of every rank's plain product
  (``rubin_mxfp8_linear_prequantized``) and the C++ all-reduce of this rank's plain product on
  the same workspace, bit for bit, and is the same on every rank;
* this rank's payload is the entry's reference within the GEMM's rounding.

Then, on one workspace: 100 calls alternating with C++ all-reduces of other data, and 300 replays
of a CUDA graph of push GEMM -> finish over alternating operands, each finish bit for bit its
reference, at helix8's b9 (two-shot) and b1 (one-shot) rows (the low-M kernel with split-K) and
its b11 rows (the mid-M kernel).

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import os
import pickle
import sys
import traceback
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or torch.cuda.device_count() < WORLD,
        reason="sm_107 kernels on four GPUs (one tray)",
    ),
]

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_lamport_push import (  # noqa: E402,E501
    rubin_lamport_finish,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear_push import (  # noqa: E402,E501
    rubin_mxfp8_linear_prequantized_push,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

MIN_COSINE, MAX_RELATIVE_L2 = 0.99999, 1e-3


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _setup():
    """This rank's (rank, mapping, MNNVL AllReduce), built once per worker process."""
    os.environ["TLLM_TEST_MNNVL"] = "1"
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    import tensorrt_llm
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_kimi_k31_lin_push", types.ModuleType("_lin_push")))
    if "allreduce" not in reg:
        mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
        reg["mapping"] = mapping
        reg["allreduce"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        assert reg["allreduce"].mnnvl_allreduce is not None
    return rank, reg["mapping"], reg["allreduce"]


def _quantize(x: torch.Tensor, n: int):
    """The low-M GEMM's MXFP8 of BF16 rows in the scale layout the prequantized entry reads."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
        mxfp8_act_layout,
    )

    m, k = x.shape
    sf_rows, layout = mxfp8_act_layout(m, n, k)
    bits = x.view(torch.int16).to(torch.int32) & 0x7FFF
    amax = bits.view(m, k // 32, 32).amax(-1).clamp(min=0x38D2)
    exp = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).to(torch.int32)
    inv = ((254 - exp) << 7).to(torch.int16).view(torch.bfloat16)
    q = (x.view(m, k // 32, 32) * inv.unsqueeze(-1)).to(torch.float8_e4m3fn).view(m, k)
    rows = torch.full((sf_rows, k // 32), 127, dtype=torch.uint8, device=x.device)
    rows[:m] = exp.to(torch.uint8)
    tiles = rows.view(sf_rows // 128, 4, 32, k // 128, 4)
    order = (3, 0, 2, 1, 4) if layout == "mid" else (0, 3, 2, 1, 4)
    return q.contiguous(), tiles.permute(*order).reshape(-1).contiguous()


def _operands(spec: dict, seed: int, r: int):
    """Rank ``r``'s ``(act, act_scale, weight, weight_scale)``."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_gemm import (  # noqa: E501
        mxfp8_scales_to_r128c4,
        quantize_mxfp8_reference,
    )

    g = torch.Generator(device="cuda").manual_seed(seed * 1009 + r)
    x = torch.randn(spec["rows"], spec["k"], generator=g, device="cuda")
    x[:, ::7] *= 30.0
    w = torch.randn(spec["n"], spec["k"], generator=g, device="cuda") * spec["k"] ** -0.5
    b, b_scale = quantize_mxfp8_reference(w.bfloat16())
    return (*_quantize(x.bfloat16(), spec["n"]), b, mxfp8_scales_to_r128c4(b_scale))


def _plain(ops):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
        rubin_mxfp8_linear_prequantized,
    )

    return rubin_mxfp8_linear_prequantized(*ops)


def _mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    x, y = a.float().flatten(), b.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    rel = ((x - y).norm() / (y.norm() + 1e-12)).item()
    return cosine > MIN_COSINE and rel < MAX_RELATIVE_L2


def _digest(t: torch.Tensor) -> int:
    bits = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    weights = torch.arange(1, bits.numel() + 1, device=bits.device, dtype=torch.int64) % 65521
    return int((bits * weights).sum().item())


def _workspace(spec: dict, mapping, allreduce, rank: int):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm.rubin_lamport_ar import (  # noqa: E501
        LamportWorkspace,
        required_bytes,
        row_push_desc,
    )
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace

    rows, n = spec["rows"], spec["n"]
    need = max(
        required_bytes(rows, n, WORLD, True),
        required_bytes(rows, n, WORLD, False),
        allreduce.mnnvl_allreduce.get_required_workspace_size(rows, n, WORLD, torch.bfloat16),
    )
    workspace = get_or_scale_allreduce_mnnvl_workspace(
        mapping, torch.bfloat16, buffer_size_bytes=need
    )
    ws = LamportWorkspace.from_mnnvl(workspace, WORLD, rank)
    return ws, row_push_desc(ws)


def _cell_rank(spec: dict, seed: int) -> dict:
    try:
        rank, mapping, allreduce = _setup()
        ws, desc = _workspace(spec, mapping, allreduce, rank)
        one_shot = spec["one_shot"]
        operands = [_operands(spec, seed, r) for r in range(WORLD)]
        mine = operands[rank]
        payloads = [_plain(ops) for ops in operands]
        args = (*mine, ws, desc, one_shot, True)
        rubin_mxfp8_linear_prequantized_push.is_valid(*args)
        fused = rubin_mxfp8_linear_prequantized_push(*args)
        got = torch.empty_like(payloads[rank])
        rubin_lamport_finish(ws, got, one_shot, True, False, None, False)
        ref = rubin_lamport_finish.reference(payloads, one_shot, None)
        cpp = allreduce(payloads[rank])
        torch.cuda.synchronize()
        return dict(
            rank=rank,
            fused=bool(fused),
            ref=_mismatch(got, ref),
            cpp=_mismatch(got, cpp),
            payload_close=_close(
                payloads[rank], rubin_mxfp8_linear_prequantized_push.reference(*args)
            ),
            digest=_digest(got),
        )
    except Exception:
        traceback.print_exc()
        raise


def _run(executor, fn, *args):
    results = list(executor.map(fn, *zip(*[args] * WORLD)))
    return sorted(results, key=lambda r: r["rank"])


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell",
    rubin_mxfp8_linear_prequantized_push.CELLS,
    ids=[c.why[:56] for c in rubin_mxfp8_linear_prequantized_push.CELLS],
)
def test_mxfp8_linear_push_cells(mpi_pool_executor, cell) -> None:
    """Every cell of rubin_mxfp8_linear_prequantized_push (``.CELLS``): the epilogue pushes, and
    the finish of the pushed products is bit for bit the reference and the C++ all-reduce of the
    plain products, the same bits on every rank."""
    results = _run(mpi_pool_executor, _cell_rank, cell.spec, _seed(cell))
    for r in results:
        assert r["fused"], r
        assert r["ref"] == 0 and r["cpp"] == 0 and r["payload_close"], r
    assert len({r["digest"] for r in results}) == 1, results


def _rotation_rank(seed: int, rows: int, one_shot: bool) -> dict:
    try:
        spec = dict(rows=rows, n=7168, k=1536, one_shot=one_shot)
        rank, mapping, allreduce = _setup()
        ws, desc = _workspace(spec, mapping, allreduce, rank)
        sets = []
        for s in range(2):
            operands = [_operands(spec, seed + s, r) for r in range(WORLD)]
            payloads = [_plain(ops) for ops in operands]
            sets.append((operands[rank], rubin_lamport_finish.reference(payloads, one_shot, None)))
        # The same rows on every rank: their sum is four times each, exact in BF16.
        g = torch.Generator(device="cuda").manual_seed(seed)
        other = torch.randn(13, 7168, generator=g, device="cuda").bfloat16()
        other_sum = (other.float() * WORLD).bfloat16()
        failures = 0

        def push(ops):
            assert rubin_mxfp8_linear_prequantized_push(*ops, ws, desc, one_shot, True)

        # 100 calls alternating with C++ all-reduces of other rows on the same workspace.
        for i in range(100):
            ops, ref = sets[i % 2]
            push(ops)
            got = torch.empty(rows, 7168, dtype=torch.bfloat16, device="cuda")
            rubin_lamport_finish(ws, got, one_shot, True, False, None, False)
            cpp = allreduce(other)
            torch.cuda.synchronize()
            failures += int(_mismatch(got, ref) != 0)
            failures += int(_mismatch(cpp, other_sum) != 0)
        # 300 replays of one CUDA graph of push GEMM -> finish over alternating operands.
        live = [t.clone() for t in sets[0][0]]
        out = torch.empty(rows, 7168, dtype=torch.bfloat16, device="cuda")
        push(live)
        rubin_lamport_finish(ws, out, one_shot, True, False, None, False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            push(live)
            rubin_lamport_finish(ws, out, one_shot, True, False, None, False)
        for i in range(300):
            ops, ref = sets[i % 2]
            for dst, src in zip(live, ops):
                dst.copy_(src)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            failures += int(_mismatch(out, ref) != 0)
        return dict(rank=rank, failures=int(failures))
    except Exception:
        traceback.print_exc()
        raise


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "rows,one_shot",
    [(54, False), (6, True), (66, False)],
    ids=["b9-two-shot", "b1-one-shot", "b11-two-shot-mid-M"],
)
def test_mxfp8_linear_push_rotation_and_graph_replay(mpi_pool_executor, rows, one_shot) -> None:
    """One workspace: 100 pushed calls alternating with C++ all-reduces, then 300 replays of a
    captured push GEMM -> finish; every finish is its reference, bit for bit."""
    results = _run(mpi_pool_executor, _rotation_rank, 11, rows, one_shot)
    assert all(r["failures"] == 0 for r in results), results
