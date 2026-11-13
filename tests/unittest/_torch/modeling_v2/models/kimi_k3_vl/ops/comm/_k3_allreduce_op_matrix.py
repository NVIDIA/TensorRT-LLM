# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ops/comm/k3_allreduce.K3AllReduce`` against the built-in ``AllReduce``, on the same inputs, at W
ranks of one GB200 tray (default 4).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_allreduce_op_matrix.py [--world-size 4]

Not a pytest module: one fixed sequence of checks inside one W-rank job (both modules share the mapping's MNNVL
workspace rotation). The collected entry point is ``test_modeling_v2_k3_allreduce_op_matrix.py``.

Checks:
  * MNNVL strategy (MNNVL on one tray, as the K3 targets' AUTO is across trays): both modules take MNNVL. The plain
    sum at the K3 widths (7168, the latent 3584) for 1 to 8192 rows, one-shot and two-shot, before
    ``use_decode_one_shot`` (the built-in kernel against the family copy at the built-in ceiling): bit for bit.
    After it on both modules (the targets' state after post_load_weights): the plain sum (decode rows, and
    ``wide_all_reduce``) and the drafter's fusions (residual + RMSNorm, RMSNorm) bit for bit. The fusions before
    it (not a path the targets run): ``residual_out`` bit for bit, ``norm_out`` within ``NORM_TOL``, recorded.
  * A CUDA graph of the new module's decode call replayed with new inputs: bit for bit the eager call.
  * AUTO strategy on one tray (no MNNVL): both modules fall back. With the built-in autotuner off
    (``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``) both run the raw op's AUTO: bit for bit. The autotuner-on fallback
    (NCCL_SYMMETRIC window outputs, which need an NCCL window tensor scope) is driven by
    ``_k3_allreduce_fallback_op_matrix.py``.
  * ``enable_allreduce=False`` returns the input; every output is the same on every rank.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "comm"))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the all-reduce requires CUDA devices"

DEADLINE_S = 1200
HIDDENS = (7168, 3584)  # K3's hidden size; the routed latent the MoE all-reduce sums
ROWS = (1, 2, 3, 4, 5, 8, 9, 16, 17, 18, 32, 36, 37, 64, 65, 73, 74, 128, 1024, 2048, 8192)
DECODE_ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 36)
WIDE_ROWS = (9, 16, 32, 64)
FUSED_ROWS = (1, 8, 36, 64, 2048)
AUTO_ROWS = (1, 8, 64, 2048, 8192)
EPS = 1e-6
NORM_TOL = 1e-2  # norm_out: max |err| / max |ref|; the kernels round each square to bf16

R = None
STATS = {
    "plain_bitwise": 0,
    "fused_norm_bitwise": 0,
    "fused_cases": 0,
    "fused_differs": [],
}


def _payload(rows, hidden, seed, exact=False):
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + R.rank + 1)
    if exact:
        return ls.exact_bf16(gen, (rows, hidden), -64, 64, 1.0 / 16)
    return torch.randn(rows, hidden, generator=gen, device="cuda").bfloat16()


def _bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _modules(strategy):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
        K3AllReduce,
    )
    from tensorrt_llm._torch.distributed import AllReduce

    stock = AllReduce(mapping=R.mapping, strategy=strategy, dtype=torch.bfloat16)
    new = K3AllReduce(R.mapping, strategy, torch.bfloat16)
    return stock, new


def _check_plain(stock, new, rows_list, seed0, tag, wide=None):
    for hidden in HIDDENS:
        for i, rows in enumerate(rows_list):
            x = _payload(rows, hidden, seed0 + 31 * i + hidden)
            if wide is None:
                a, b = stock(x), new(x)
            else:
                a, b = wide(stock, x), wide(new, x)
            assert _same(a, b), f"{tag}: hidden {hidden}, {rows} rows: the outputs differ"
            assert R.same_on_ranks(b), f"{tag}: hidden {hidden}, {rows} rows: ranks disagree"
            STATS["plain_bitwise"] += 1


def check_mnnvl_plain_default_ceiling():
    from tensorrt_llm.functional import AllReduceStrategy

    global MNNVL_STOCK, MNNVL_NEW
    MNNVL_STOCK, MNNVL_NEW = _modules(AllReduceStrategy.MNNVL)
    assert MNNVL_STOCK.mnnvl_allreduce is not None, "the built-in module did not take MNNVL"
    assert MNNVL_NEW.mnnvl_allreduce is not None, "K3AllReduce did not take MNNVL"
    _check_plain(MNNVL_STOCK, MNNVL_NEW, ROWS, 100, "MNNVL, built-in ceiling")


def _fused_params(rows, hidden, seed, fusion):
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams

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


def _check_fused(stock, new, rows_list, seed0, tag, bitwise):
    """The drafter's fusions (DSpark's residual + RMSNorm, DFlash's RMSNorm of the context projection) at hidden
    7168. ``bitwise``: every output bit for bit; else ``residual_out`` bit for bit, ``norm_out`` within NORM_TOL and
    recorded per row count."""
    hidden = 7168
    for fusion in ("residual_rms_norm", "rms_norm"):
        for i, rows in enumerate(rows_list):
            x = _payload(rows, hidden, seed0 + i)
            params = _fused_params(rows, hidden, seed0 + 400 + i, fusion)
            a, b = stock(x, all_reduce_params=params), new(x, all_reduce_params=params)
            a = a if isinstance(a, tuple) else (a,)
            b = b if isinstance(b, tuple) else (b,)
            assert len(a) == len(b), (tag, fusion, rows, len(a), len(b))
            name = f"{tag}, {fusion}, {rows} rows"
            assert R.same_on_ranks(*b), f"{name}: ranks disagree"
            same = [_same(x_, y_) for x_, y_ in zip(a, b)]
            if bitwise:
                assert all(same), f"{name}: outputs differ ({same})"
                continue
            if len(a) == 2:
                assert same[1], f"{name}: residual_out differs"
            err = ls.rel_err(b[0], a[0])
            assert err <= NORM_TOL, f"{name}: norm_out off by {err:.3e}"
            STATS["fused_cases"] += 1
            STATS["fused_norm_bitwise"] += int(same[0])
            if not same[0]:
                diff = (_bits(a[0]) != _bits(b[0])).float().mean().item()
                STATS["fused_differs"].append((fusion, rows, round(err, 6), round(diff, 6)))


def check_mnnvl_fused_default_ceiling():
    """Before ``use_decode_one_shot`` the built-in module runs the built-in MNNVL kernel and K3AllReduce the K3 copy:
    not a path the targets take (post_load_weights raises the ceiling before any forward), recorded only."""
    _check_fused(MNNVL_STOCK, MNNVL_NEW, FUSED_ROWS, 500, "fused, built-in ceiling", bitwise=False)


def check_skip_returns_the_input():
    from tensorrt_llm.functional import AllReduceParams

    x = _payload(4, 7168, 7)
    assert MNNVL_NEW(x, all_reduce_params=AllReduceParams(enable_allreduce=False)) is x


def check_mnnvl_plain_decode_ceiling():
    """Both modules after ``use_decode_one_shot``: the same K3 MNNVL kernel at the decode ceiling, and a wide step's
    call at the wide ceiling."""
    import importlib

    from torch import nn

    dc = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.decode_comm"
    )
    holder = nn.ModuleList([MNNVL_STOCK, MNNVL_NEW])
    dc.use_decode_one_shot(holder)
    for module in (MNNVL_STOCK, MNNVL_NEW):
        mnnvl = module.mnnvl_allreduce
        assert mnnvl.one_shot_max_bytes == dc.DECODE_AR_ONE_SHOT_MAX_BYTES and mnnvl.early_trigger
    _check_plain(MNNVL_STOCK, MNNVL_NEW, DECODE_ROWS, 300, "MNNVL, decode ceiling")
    _check_plain(MNNVL_STOCK, MNNVL_NEW, WIDE_ROWS, 400, "MNNVL, wide", wide=dc.wide_all_reduce)
    # The drafter's fused all-reduces, as the targets run them: both on the K3 kernel at the decode ceiling.
    _check_fused(MNNVL_STOCK, MNNVL_NEW, FUSED_ROWS, 450, "fused, decode ceiling", bitwise=True)


def check_graph_replay():
    hidden, rows = 7168, 8
    static = _payload(rows, hidden, 600)
    MNNVL_NEW(static)  # the workspace holds the call before capture
    torch.cuda.synchronize()
    R.barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out = MNNVL_NEW(static)
    torch.cuda.synchronize()
    R.barrier()
    for step in range(3):
        static.copy_(_payload(rows, hidden, 610 + step))
        graph.replay()
        torch.cuda.synchronize()
        R.barrier()
        eager = MNNVL_NEW(static)
        assert _same(out, eager), f"replay {step}: differs from the eager call"
        assert R.same_on_ranks(out), f"replay {step}: ranks disagree"
        R.barrier()


def _autotune(on: bool) -> None:
    if on:
        os.environ.pop("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", None)
    else:
        os.environ["TLLM_DISABLE_ALLREDUCE_AUTOTUNE"] = "1"


def _progress(msg):
    if R.rank == 0:
        print(f"[rank 0] {msg}", flush=True)


def check_auto_autotuner_off():
    """Both modules off MNNVL with the built-in autotuner off: both run the raw op's AUTO."""
    from tensorrt_llm.functional import AllReduceStrategy

    global AUTO_STOCK, AUTO_NEW
    AUTO_STOCK, AUTO_NEW = _modules(AllReduceStrategy.AUTO)
    assert AUTO_STOCK.mnnvl_allreduce is None and AUTO_NEW.mnnvl_allreduce is None, (
        "AUTO on one tray takes MNNVL in one of the modules"
    )
    _autotune(False)
    for hidden in HIDDENS:
        for i, rows in enumerate(AUTO_ROWS):
            _progress(f"autotuner off, hidden {hidden}, {rows} rows")
            x = _payload(rows, hidden, 700 + i)
            a, b = AUTO_STOCK(x), AUTO_NEW(x)
            assert _same(a, b), f"AUTO, autotuner off, hidden {hidden}, {rows} rows: differs"
            assert R.same_on_ranks(b)
            del a, b
            torch.cuda.synchronize()


CHECKS = [
    check_mnnvl_plain_default_ceiling,
    check_mnnvl_fused_default_ceiling,
    check_skip_returns_the_input,
    check_mnnvl_plain_decode_ceiling,
    check_graph_replay,
    check_auto_autotuner_off,
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
