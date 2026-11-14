# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the Kimi K3 family's ``ops/comm/autotuned_allreduce`` entry, at W ranks of one node
(default 4).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _autotuned_allreduce_op_matrix.py [--world-size 4]

The entry needs MPI (the built-in module runs the autotuner only with MPI on), so the ranks start under mpirun
(``_lockstep``). Not a pytest module; the collected entry point is
``test_modeling_v2_autotuned_allreduce_op_matrix.py``.

Every cell of ``autotuned_allreduce.CELLS`` at this world size runs inside ``validating``, inside an NCCL window
tensor scope (as the engine runs every forward), twice: untuned (each shape's cache-miss tactic, NCCL_SYMMETRIC) and
after an ``autotune()`` pass over the shape. Each output is gated against the entry's reference within
``catalog/comm/allreduce``'s band (5 ulp per element, 1.25 ulp RMS) and compared bitwise across the ranks.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "comm"))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "autotuned_allreduce requires CUDA devices"

DEADLINE_S = 1200

R = None


def _payload(rank, rows, hidden, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return torch.randn(rows, hidden, generator=gen, device="cuda").bfloat16()


def _residual_and_weight(rows, hidden, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed * 31 + 7)
    residual = torch.randn(rows, hidden, generator=gen, device="cuda").bfloat16()
    weight = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device="cuda")).bfloat16()
    return residual, weight


def check_certified_cells():
    from _validating import assert_within_ulp, validating

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm import (
        autotuned_allreduce as entry,
    )
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs

    AutoTuner.get().clear_cache()
    AutoTuner.get().setup_distributed_state(R.mapping)
    args = entry.autotuned_allreduce_args(R.mapping)
    cells = [c for c in entry.autotuned_allreduce.CELLS if c.spec["world"] == R.world]
    assert cells, f"no cell is certified at world size {R.world}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        seed = 4000 + 17 * i
        rows, hidden, op, eps = spec["rows"], spec["hidden"], spec["op"], spec["eps"]
        x = _payload(R.rank, rows, hidden, seed)
        residual, weight = _residual_and_weight(rows, hidden, seed)
        if op == entry.NONE:
            residual = weight = None
        elif op == entry.RMS_NORM:
            residual = None
        ref = entry.autotuned_allreduce.reference(
            [_payload(r, rows, hidden, seed) for r in range(R.world)], residual, weight, eps, op
        )
        for tuned in (False, True):
            with discard_nccl_window_tensor_outputs((x, residual)):
                if tuned:
                    with autotune():
                        entry.autotuned_allreduce(
                            x, residual, weight, None, None, op=op, eps=eps,
                            trigger_completion_at_end=True, **args,
                        )  # fmt: skip
                with validating(entry.autotuned_allreduce):
                    out = entry.autotuned_allreduce(
                        x, residual, weight, None, None, op=op, eps=eps,
                        trigger_completion_at_end=True, **args,
                    )  # fmt: skip
                outs = out if isinstance(out, tuple) else (out,)
                refs = ref if isinstance(ref, tuple) else (ref,)
                assert len(outs) == len(refs), (cell.why, len(outs), len(refs))
                try:
                    for o, r in zip(outs, refs):
                        assert_within_ulp(o, r, element_ulp=5.0, rms_ulp=1.25)
                except AssertionError as exc:
                    raise AssertionError(f"{cell.why}, tuned={tuned}: {exc}") from exc
                assert R.same_on_ranks(*outs), f"{cell.why}, tuned={tuned}: ranks disagree"
                torch.cuda.synchronize()
        if R.rank == 0:
            print(f"[rank 0] passed {cell.why}", flush=True)


def _run_one_rank(args) -> int:
    global R
    R = ls.Rank(args)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    with torch.inference_mode():
        return ls.run_checks(R, [check_certified_cells])


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
