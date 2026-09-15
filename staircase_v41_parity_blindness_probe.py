# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce review's blindness experiment against the dense parity driver.

Review demonstrated that `parity_dense.py` could not see a wrong TP rank order:
monkeypatching `torch.cat` to reverse every four-tensor, 3-D, `dim=-1`
reconstruction mutated 12 full-width calls and the driver still printed
`DENSE MODULE PARITY: PASS` and exited 0. That is the exact experiment repeated
here, unchanged in shape, as a standing control.

A harness that returns 0 under this mutation cannot close module wiring, so the
verdict is inverted: this probe exits 0 only when the mutated driver FAILS, and
reports the mutation count so a run that silently patched nothing is
distinguishable from one the driver genuinely survived.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

TARGET_DIR = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM"
    "/tensorrt_llm/_torch/staircase/models/deepseek_v41/targets/v41_flash/sm_103/dep4"
)
DRIVER = TARGET_DIR / "parity_dense.py"

#: dep4.
WORLD = 4


def main() -> int:
    calls = 0
    real_cat = torch.cat

    def reversing_cat(tensors, dim=0, *args, **kwargs):
        """Reverse exactly the reconstructions review reversed, and nothing else."""
        nonlocal calls
        seq = list(tensors)
        if (
            len(seq) == WORLD
            and dim == -1
            and all(torch.is_tensor(t) and t.dim() == 3 for t in seq)
            and len({t.shape for t in seq}) == 1
        ):
            calls += 1
            seq = list(reversed(seq))
        return real_cat(seq, dim, *args, **kwargs)

    # The monkeypatch is the whole experiment, so the type checker's complaint
    # that a plain function is not `torch.cat`'s overload set is correct and
    # beside the point -- review performed exactly this substitution.
    torch.cat = reversing_cat  # ty: ignore[invalid-assignment]
    sys.argv = [str(DRIVER)]
    try:
        runpy.run_path(str(DRIVER), run_name="__main__")
        rc = 0
    except SystemExit as exc:
        rc = int(exc.code or 0)
    finally:
        torch.cat = real_cat

    print()
    print(f"RANK_ORDER_MUTATION_CALLS {calls}")
    print(f"RANK_ORDER_MUTATION_RC {rc}")
    if calls == 0:
        print(
            "BLINDNESS PROBE INCONCLUSIVE: the mutation never fired, so this run says nothing "
            "about whether the driver can see a wrong rank order"
        )
        return 2
    if rc == 0:
        print(
            f"BLINDNESS PROBE FAIL: {calls} rank reconstructions were reversed and the parity "
            f"driver still exited 0. The harness cannot see a wrong TP rank order"
        )
        return 1
    print(
        f"BLINDNESS PROBE PASS: {calls} rank reconstructions reversed, parity driver exited "
        f"{rc}. The harness sees the defect it is supposed to see"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
