# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MARKOV_GRID on 4 real ranks (``_m31_markov_grid_op_matrix.py``, 4 GPUs of one SM107 tray): the TP4 target's
batch-1 chain on its wider grid returns bitwise the catalog op's tokens and KV-length rewind (random / tie / equal /
extreme logits, forced ties at rank and CTA boundaries of both grids, LL and Lamport exchanges, back-to-back mixed
launches, CUDA-graph replays), and the worker's prebuild agreement on the grid holds over the real TP collective.

A separate file from test_modeling_v2_m31_r6_markov_grid.py: it runs 4 processes that compile the CuTe DSL variants
at once, so a gate budgets it on its own (the launcher's deadline is 600 s; this wrapper allows 720 s; a harness
running this file needs a per-file timeout of at least 900 s).
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

MATRIX_TIMEOUT_S = 720  # the launcher's DEADLINE_S (600) + 120 s for its imports and teardown


def _sm107_tray() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 4
        and torch.cuda.get_device_capability() == (10, 7)
        and torch.cuda.get_device_properties(0).multi_processor_count >= 198
    )


@pytest.mark.no_xdist
@pytest.mark.skipif(
    not _sm107_tray(), reason="needs 4 SM107 GPUs of one tray (multicast workspace)"
)
def test_markov_grid_matrix_on_4_ranks():
    launcher = Path(__file__).resolve().with_name("_m31_markov_grid_op_matrix.py")
    done = subprocess.run(
        [sys.executable, str(launcher)], capture_output=True, text=True, timeout=MATRIX_TIMEOUT_S
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("OK"), (
        f"{launcher.name} exited {done.returncode}\n--- stdout ---\n{done.stdout[-6000:]}\n"
        f"--- stderr ---\n{done.stderr[-6000:]}"
    )
