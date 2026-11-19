# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_kda_prefill entry (sm_107).

Each cell runs the op through its guard and the entry's plain-torch reference on identical inputs:
the output, the selected state rows and the snapshot rows within the chunked-prefill bands of an
fp32 recurrence, every other pool row bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (
    rubin_kda_prefill,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _kda_ops_util import (  # noqa: E402 — needs the path declared above
    assert_rows,
    assert_scaled_close,
    cell_seed,
    clone_written,
    step_schedule,
)
from _rubin_kda_prefill_inputs import prefill_args  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(107)

#: A boundary and a typical cell of ``rubin_kda_prefill.CELLS``.
_CELLS = sample_cells(rubin_kda_prefill.CELLS, size_key="heads")[:2]


# 16-token chunks with bf16 tensor-core products (the entry's note); a bf16 pool adds one rounding
# of the state per chunk.
TOL_OUT = 2e-2
TOL_STATE = {torch.float32: 1e-2, torch.bfloat16: 2e-2}


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(rubin_kda_prefill)
    args = prefill_args(cell.spec, cell_seed(cell))
    seq_order, cu_chunks = step_schedule(cell.spec["seq_lens"], "cuda")
    args.update(seq_order=seq_order, cu_chunks=cu_chunks)
    ref_args = clone_written(args, ("state_pool",))
    out = rubin_kda_prefill(**args)
    ref = rubin_kda_prefill.reference(**ref_args)
    torch.cuda.synchronize()
    assert_scaled_close(out, ref, TOL_OUT, "output")
    pool, ref_pool = args["state_pool"], ref_args["state_pool"]
    written = set(cell.spec["slots"]) | {r for r in cell.spec.get("snapshot_rows", ()) if r >= 0}
    assert_rows(pool, ref_pool, written, TOL_STATE[pool.dtype])
