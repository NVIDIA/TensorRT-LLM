# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-path collectives (``decode_comm.py``) of the Kimi K3 target ``kimi_k3_mxfp4__sm_100__tp16_moetp16ep1``
against its built-in path, at W ranks of one GB200 tray (default 4), at the TP16 per-rank shapes.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _kimi_k3_decode_comm_tp16_moetp16ep1_op_matrix.py [--world-size 4]
    srun -n 4 --mpi=pmix python _kimi_k3_decode_comm_tp16_moetp16ep1_op_matrix.py --launcher srun --world-size 4

Not a pytest module: one fixed sequence of checks inside one W-rank job, sharing the target's collective state. The
collected entry point is ``test_modeling_v2_kimi_k3_decode_comm_tp16_moetp16ep1_op_matrix.py``.

The checks are ``_kimi_k3_decode_comm_checks.py`` of the family test directory, at the steps this target serves: it
decodes without speculation, so no step has more than one token per generation request (no DSpark verify, no wide
step), and no layer is tapped (the MoE tail tap check is left out). Its MoE decode path (route B, ``decode_moe.py``
with the k3_moe_m1 / k3_moe_m2 engines) has its own matrix, ``_kimi_k3_route_b_decode_moe_op_matrix.py``.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "comm"))
import _kimi_k3_decode_comm_checks as dc  # noqa: E402

if TYPE_CHECKING:
    from _torch.modeling_v2.comm import _lockstep as ls
else:
    import _lockstep as ls  # noqa: E402

DEADLINE_S = dc.DEADLINE_S

CHECKS = [
    dc.check_takes_oproj,
    dc.check_decode_one_shot,
    dc.check_fused_vs_builtin,
    dc.check_sandwich_vs_mnnvl_bitwise,
    dc.check_graph_capture_and_replay,
    dc.check_moe_tail_deferral,
    dc.check_moe_tail_graph,
]


def _run_one_rank(args) -> int:
    dc.start_rank(args)
    import importlib

    modeling = importlib.import_module(
        "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl."
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.modeling"
    )
    return dc.run_matrix(
        modeling,
        CHECKS,
        steps=[name for name, _, _ in dc.ALL_STEPS if name not in dc.SPECULATIVE_STEPS],
    )


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
