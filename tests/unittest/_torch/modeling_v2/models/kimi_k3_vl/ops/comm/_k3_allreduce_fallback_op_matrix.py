# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ops/comm/k3_allreduce.K3AllReduce`` where the TP group does not run over MNNVL (B200 / B300 NVLink in one node;
across nodes TP16 over two 8-GPU nodes), at W ranks (default 4): every call reaches the runtime ``AllReduce``'s own
all-reduce with the arguments a runtime module of the same mapping and strategy passes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _k3_allreduce_fallback_op_matrix.py [--world-size 4]

Not a pytest module: one fixed sequence of checks inside one W-rank job (the autotuner's tactic cache and the NCCL
window pool are per process and shared by both modules). The collected entry point is
``test_modeling_v2_k3_allreduce_fallback_op_matrix.py``.

The three runtime all-reduce symbols (``autotuned_allreduce``, ``tunable_allreduce``, ``allreduce``) are wrapped
before the modules are built, so each module call records the symbol it reaches and its arguments. Every call runs
inside an NCCL window tensor scope, as the engine runs every forward (an NCCL_SYMMETRIC output freed outside one
aborts the process).

Checks:
  * Both strategies the targets route (AUTO, MNNVL): ``K3AllReduce`` is a runtime ``AllReduce`` whose ``forward`` is
    the runtime's, without an MNNVL all-reduce, on the IPC workspace a runtime module of the mapping uses.
  * Autotuner on (the plain sum at 7168 and 3584, the drafter's residual + RMSNorm and RMSNorm at 7168) and off
    (``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``; the plain sum and residual + RMSNorm): each call of ``K3AllReduce`` and of
    a runtime module reaches the same symbol once, with the same arguments, returns the same container, and gives
    the same bits on every rank. The autotuner-off RMSNorm is left out: the runtime module crashes on it there
    (SIGSEGV on B200 at 4 ranks), and ``K3AllReduce`` makes the same call.
  * ``enable_allreduce=False`` returns the input without a call.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "comm"))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "the all-reduce requires CUDA devices"

DEADLINE_S = 900
ROWS = (1, 64)
EPS = 1e-6
SYMBOLS = ("autotuned_allreduce", "tunable_allreduce", "allreduce")
# The arguments compared between the two modules' calls: the scalars by value, the tensors by identity (the same
# input and operands, the same IPC workspace).
SCALAR_ARGS = (
    "group",
    "strategy",
    "op",
    "eps",
    "trigger_completion_at_end",
    "exclude_nccl_symmetric",
)
TENSOR_ARGS = ("input", "residual", "norm_weight", "scale", "bias", "workspace")

R = None
STOCK = NEW = None
CALLS = []
DEPTH = [0]
STATS = {"cases": 0}


def _spy_on_symbols():
    """Wrap the runtime all-reduce symbols on ``torch.ops.trtllm`` so each outermost call appends ``(symbol,
    kwargs)`` to ``CALLS`` (``tunable_allreduce`` calls ``allreduce`` inside). Done before the modules are built: the
    raw-op path binds its symbol at construction."""
    for name in SYMBOLS:
        real = getattr(torch.ops.trtllm, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            if DEPTH[0] == 0:
                assert not args, f"{_name} called with positional arguments"
                CALLS.append((_name, kwargs))
            DEPTH[0] += 1
            try:
                return _real(*args, **kwargs)
            finally:
                DEPTH[0] -= 1

        setattr(torch.ops.trtllm, name, spy)


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


def _cases(fusions):
    if "none" in fusions:
        for hidden in (7168, 3584):
            for rows in ROWS:
                yield "none", hidden, rows
    for fusion in ("residual_rms_norm", "rms_norm"):
        if fusion in fusions:
            for rows in ROWS:
                yield fusion, 7168, rows


def _one_call(module, x, params, name):
    """``module(x, all_reduce_params=params)`` and the one runtime symbol call it made."""
    CALLS.clear()
    out = module(x, all_reduce_params=params)
    assert len(CALLS) == 1, (
        f"{name}: {type(module).__name__} made {len(CALLS)} runtime all-reduce calls"
    )
    return out, CALLS[0]


def _compare_all(tag, seed0, symbols, fusions):
    from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs

    for i, (fusion, hidden, rows) in enumerate(_cases(fusions)):
        x = _payload(rows, hidden, seed0 + i)
        params = _params(fusion, rows, hidden, seed0 + 500 + i)
        name = f"{tag}, {fusion}, hidden {hidden}, {rows} rows"
        if R.rank == 0:
            print(f"[rank 0] {name}", flush=True)
        with discard_nccl_window_tensor_outputs((x, params)):
            raw_a, (sym_a, args_a) = _one_call(STOCK, x, params, name)
            raw_b, (sym_b, args_b) = _one_call(NEW, x, params, name)
            assert sym_b in symbols, (
                f"{name}: K3AllReduce reached {sym_b}, expected one of {symbols}"
            )
            assert sym_a == sym_b, (
                f"{name}: the runtime module reached {sym_a}, K3AllReduce {sym_b}"
            )
            for key in SCALAR_ARGS:
                assert args_a.get(key) == args_b.get(key), (
                    name,
                    key,
                    args_a.get(key),
                    args_b.get(key),
                )
            for key in TENSOR_ARGS:
                assert args_a.get(key) is args_b.get(key), (name, key)
            assert args_b["input"] is x, f"{name}: the input was copied"
            assert args_b["workspace"] is not None, f"{name}: no IPC workspace passed"
            # The same container too (a fusion's outputs come back as a list off MNNVL).
            assert type(raw_a) is type(raw_b), (name, type(raw_a), type(raw_b))
            a, b = _tuple(raw_a), _tuple(raw_b)
            assert len(a) == len(b), (name, len(a), len(b))
            same = all(_same(p, q) for p, q in zip(a, b))
            agree = R.same_on_ranks(*b)
            torch.cuda.synchronize()
        assert same, f"{name}: the two modules differ"
        assert agree, f"{name}: ranks disagree"
        STATS["cases"] += 1


def check_structure():
    """K3AllReduce is the runtime module, without an MNNVL all-reduce here, on the mapping's IPC workspace."""
    global STOCK, NEW
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import K3AllReduce
    from tensorrt_llm.functional import AllReduceStrategy

    _spy_on_symbols()
    AutoTuner.get().clear_cache()
    AutoTuner.get().setup_distributed_state(R.mapping)
    assert K3AllReduce.forward is AllReduce.forward, (
        "K3AllReduce overrides the runtime module's forward"
    )
    for strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL):
        stock = AllReduce(mapping=R.mapping, strategy=strategy, dtype=torch.bfloat16)
        new = K3AllReduce(R.mapping, strategy, torch.bfloat16)
        assert isinstance(new, AllReduce)
        assert stock.mnnvl_allreduce is None and new.mnnvl_allreduce is None, (
            f"{strategy.name}: MNNVL taken on a topology without it"
        )
        assert new.workspace is not None and new.workspace is stock.workspace, (
            f"{strategy.name}: not the mapping's IPC workspace"
        )
        assert new.output_buffer_kind == stock.output_buffer_kind, strategy.name
        if strategy == AllReduceStrategy.AUTO:
            STOCK, NEW = stock, new


def check_autotuner_on():
    _autotune(True)
    _compare_all(
        "autotuner on",
        100,
        symbols=("autotuned_allreduce", "tunable_allreduce"),
        fusions=("none", "residual_rms_norm", "rms_norm"),
    )


def check_autotuner_off():
    _autotune(False)
    try:
        _compare_all(
            "autotuner off", 200, symbols=("allreduce",), fusions=("none", "residual_rms_norm")
        )
    finally:
        _autotune(True)


def check_skip_returns_the_input():
    from tensorrt_llm.functional import AllReduceParams

    x = _payload(4, 7168, 7)
    CALLS.clear()
    assert NEW(x, all_reduce_params=AllReduceParams(enable_allreduce=False)) is x
    assert not CALLS, "enable_allreduce=False made a runtime all-reduce call"


CHECKS = [
    check_structure,
    check_autotuner_on,
    check_autotuner_off,
    check_skip_returns_the_input,
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
