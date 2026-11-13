# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ops/comm/k3_allreduce.K3AllReduce`` against the built-in ``AllReduce`` where the TP group does not run over MNNVL
(B200 / B300 NVLink in one node; across nodes TP16 over two 8-GPU nodes), on the same inputs, at W ranks (default 4).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_allreduce_fallback_op_matrix.py [--world-size 4]

Not a pytest module: one fixed sequence of checks inside one W-rank job (the autotuner's tactic cache and the NCCL
window pool are per process and shared by both modules). The collected entry point is
``test_modeling_v2_k3_allreduce_fallback_op_matrix.py``.

Every call runs inside an NCCL window tensor scope, as the engine runs every forward (an NCCL_SYMMETRIC output
freed outside one aborts the process), and every capture binds its window buffers to its graph pool after two eager
warmups, as the engine's CUDA graph runner does.

Checks, each bit for bit between the two modules and across the ranks:
  * Both strategies the targets route (AUTO, MNNVL) leave MNNVL off here, in both modules.
  * Autotuner on, untuned (each shape's tactic is the cache-miss fallback, NCCL_SYMMETRIC): the plain sum at the K3
    widths (7168, the routed latent 3584) for 1 to 8192 rows, and the drafter's fusions (residual + RMSNorm,
    RMSNorm) at 7168.
  * Autotuner on, after an ``autotune()`` pass over those shapes through the built-in module: both modules look up
    the same tactic (one cache per process, keyed by group, dtype, fusion and the window flag).
  * Autotuner off (``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``): both run the raw op's AUTO, plain sum and residual +
    RMSNorm. RMS_NORM is left out: the built-in module itself crashes on it there (SIGSEGV on B200 at 4 ranks,
    1 row), and K3AllReduce makes the same call.
  * ``enable_allreduce=False`` returns the input.
  * CUDA graph: both modules' decode-size calls captured in one graph and replayed with new inputs, equal to each
    other and to the eager calls.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "comm"))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the all-reduce requires CUDA devices"

DEADLINE_S = 1500
HIDDENS = (7168, 3584)
PLAIN_ROWS = (1, 2, 8, 9, 64, 128, 2048, 8192)
FUSED_ROWS = (1, 8, 64)
GRAPH_ROWS = (1, 8)
EPS = 1e-6

R = None
STOCK = NEW = None
STATS = {"cases": 0}


def _payload(rows, hidden, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + R.rank + 1)
    return torch.randn(rows, hidden, generator=gen, device="cuda").bfloat16()


def _bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _tuple(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _autotune(on: bool) -> None:
    if on:
        os.environ.pop("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", None)
    else:
        os.environ["TLLM_DISABLE_ALLREDUCE_AUTOTUNE"] = "1"


def _params(fusion, rows, hidden, seed):
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams

    if fusion == "none":
        return None
    shared = torch.Generator(device="cuda").manual_seed(seed)
    weight = (1 + 0.1 * torch.randn(hidden, generator=shared, device="cuda")).bfloat16()
    if fusion == "rms_norm":
        return AllReduceParams(fusion_op=AllReduceFusionOp.RMS_NORM, norm_weight=weight, eps=EPS)
    residual = torch.randn(rows, hidden, generator=shared, device="cuda").bfloat16()
    return AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        residual=residual,
        norm_weight=weight,
        eps=EPS,
    )


def _cases(fusions=("none", "residual_rms_norm", "rms_norm")):
    if "none" in fusions:
        for hidden in HIDDENS:
            for rows in PLAIN_ROWS:
                yield "none", hidden, rows
    for fusion in ("residual_rms_norm", "rms_norm"):
        if fusion in fusions:
            for rows in FUSED_ROWS:
                yield fusion, 7168, rows


def _compare_all(tag, seed0, tune=False, fusions=("none", "residual_rms_norm", "rms_norm")):
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs

    for i, (fusion, hidden, rows) in enumerate(_cases(fusions)):
        x = _payload(rows, hidden, seed0 + i)
        params = _params(fusion, rows, hidden, seed0 + 500 + i)
        name = f"{tag}, {fusion}, hidden {hidden}, {rows} rows"
        if R.rank == 0:
            print(f"[rank 0] {name}", flush=True)
        with discard_nccl_window_tensor_outputs((x, params)):
            if tune:
                with autotune():
                    STOCK(x, all_reduce_params=params)
            raw_a = STOCK(x, all_reduce_params=params)
            if fusion == "rms_norm":
                torch.cuda.synchronize()
                if R.rank == 0:
                    print(f"[rank 0] {name}: the built-in module returned", flush=True)
            raw_b = NEW(x, all_reduce_params=params)
            # The same container too (a fusion's outputs come back as a list off MNNVL).
            assert type(raw_a) is type(raw_b), (name, type(raw_a), type(raw_b))
            a, b = _tuple(raw_a), _tuple(raw_b)
            assert len(a) == len(b), (name, len(a), len(b))
            assert type(STOCK(x, all_reduce_params=params)) is type(
                NEW(x, all_reduce_params=params)
            ), f"{name}: the two modules return different containers"
            same = all(_same(p, q) for p, q in zip(a, b))
            agree = R.same_on_ranks(*b)
            torch.cuda.synchronize()
        assert same, f"{name}: the two modules differ"
        assert agree, f"{name}: ranks disagree"
        STATS["cases"] += 1


def check_mnnvl_is_off():
    global STOCK, NEW
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
        K3AllReduce,
    )
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy

    AutoTuner.get().clear_cache()
    AutoTuner.get().setup_distributed_state(R.mapping)
    for strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL):
        stock = AllReduce(mapping=R.mapping, strategy=strategy, dtype=torch.bfloat16)
        new = K3AllReduce(R.mapping, strategy, torch.bfloat16)
        assert stock.mnnvl_allreduce is None and new.mnnvl_allreduce is None, (
            f"{strategy.name}: MNNVL taken on a topology without it"
        )
        if strategy == AllReduceStrategy.AUTO:
            STOCK, NEW = stock, new


def check_autotuner_on_untuned():
    _autotune(True)
    _compare_all("autotuner on, untuned", 100)


def check_autotuner_off():
    _autotune(False)
    try:
        _compare_all("autotuner off", 200, fusions=("none", "residual_rms_norm"))
    finally:
        _autotune(True)


def check_autotuner_on_tuned():
    _autotune(True)
    _compare_all("autotuner on, tuned", 300, tune=True)


def check_skip_returns_the_input():
    from tensorrt_llm.functional import AllReduceParams

    x = _payload(4, 7168, 7)
    assert NEW(x, all_reduce_params=AllReduceParams(enable_allreduce=False)) is x


def check_graph_replay():
    from tensorrt_llm._torch.nccl_window_graph import (
        nccl_window_graph_capture,
        release_nccl_window_graph_owner,
    )
    from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs

    _autotune(True)
    for rows in GRAPH_ROWS:
        static = _payload(rows, 7168, 600 + rows)

        def body():
            return STOCK(static), NEW(static)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                with discard_nccl_window_tensor_outputs(static):
                    body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        R.barrier()
        graph = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with nccl_window_graph_capture(graph, pool):
            out_stock, out_new = body()
        torch.cuda.synchronize()
        R.barrier()
        for step in range(3):
            static.copy_(_payload(rows, 7168, 610 + 10 * rows + step))
            graph.replay()
            torch.cuda.synchronize()
            R.barrier()
            with discard_nccl_window_tensor_outputs(static):
                eager_stock, eager_new = body()
                ok = (
                    _same(out_stock, out_new)
                    and _same(out_new, eager_new)
                    and _same(eager_stock, eager_new)
                )
                agree = R.same_on_ranks(out_new)
                torch.cuda.synchronize()
            assert ok, f"graph, {rows} rows, replay {step}: replay and eager differ"
            assert agree, f"graph, {rows} rows, replay {step}: ranks disagree"
            R.barrier()
        del out_stock, out_new
        graph.reset()
        release_nccl_window_graph_owner(pool)
        torch.cuda.synchronize()
        R.barrier()


CHECKS = [
    check_mnnvl_is_off,
    check_autotuner_on_untuned,
    check_autotuner_off,
    check_autotuner_on_tuned,
    check_skip_returns_the_input,
    check_graph_replay,
]


def _run_one_rank(args) -> int:
    global R
    R = ls.Rank(args)
    with torch.inference_mode():
        code = ls.run_checks(R, CHECKS)
    if R.rank == 0:
        print(f"[rank 0] world {R.world}; {STATS}", flush=True)
    return code


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
