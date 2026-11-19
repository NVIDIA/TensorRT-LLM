# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 CuTe DSL W4A8 routed-expert entry: a sample of its cells bitwise against the
TRTLLM-Gen runner (an unfinalized cell once finalized)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (  # noqa: E501
    run_moe_finalize,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_cute import (  # noqa: E501
    k31_w4a8_moe_cute,
    max_tiles_for,
    moe_workspace_bytes,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_runner import (  # noqa: E501
    HIDDEN,
    k31_w4a8_moe_runner,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above
from _w4a8_moe_inputs import (  # noqa: E402 — needs the path declared above
    runner_args,
    w4a8_operands,
)

op = k31_w4a8_moe_cute

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip(
        "the W4A8 CuTe DSL MoE is certified on SM 10.3, where the MXFP4 targets call it",
        allow_module_level=True,
    )

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _build(spec, seed):
    """``k31_w4a8_moe_cute``'s arguments for one rank of ``spec``: ``w4a8_operands`` with the rank
    owning local experts [0, local), and a zeroed workspace."""
    rows, local, inter, n_tile = spec["rows"], spec["local"], spec["inter"], spec["n_tile"]
    ws = torch.zeros(moe_workspace_bytes(max_tiles_for(rows, local, n_tile), inter, n_tile), dtype=torch.uint8,
                     device="cuda")  # fmt: skip
    return dict(
        w4a8_operands(rows, local, inter, seed),
        local_expert_offset=0,
        local_num_experts=local,
        n_tile=n_tile,
        workspace=ws,
        finalize=bool(spec.get("finalize", True)),
        pair=bool(spec.get("pair", False)),
    )


def _runner(a):
    """The TRTLLM-Gen runner on the same ids, weights and MXFP8 input."""
    return k31_w4a8_moe_runner(*runner_args(a, None))


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    a = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(**a)
    if not a["finalize"]:
        y, w, e2p = out
        out = torch.empty(cell.spec["rows"], HIDDEN, dtype=torch.bfloat16, device=y.device)
        run_moe_finalize(y, e2p, w, out)
    # the same arithmetic as the runner (K order, SiTU, requant, finalize order): bitwise
    assert out.shape == (cell.spec["rows"], HIDDEN) and out.dtype is torch.bfloat16
    assert torch.equal(out.view(torch.int16), _runner(a).view(torch.int16)), (
        "not bitwise vs the runner"
    )
