# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one GPU) for the K3.1 Rubin attn_res entries that read their addend from a Lamport
workspace: the cells without a finish (``world == 1``).

A one-rank stand-in workspace holds the rows of the all-reduce that completed last in its
broadcast stage. Every cell: the updated prefix bit for bit against the reference, the output
against the fp32 reference, and, with the fast-math add, every output bit for bit against
rubin_attn_res_add_rmsnorm on the tensor addend (same kernel). The cells that finish a pushed
call run on four ranks in ``rubin_multirank/test_rubin_comm_entries.py``. Needs the kernel at its
family path (``kernels/norm/rubin_attn_res.py``).
"""

import zlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="sm_107 kernel (VR200)",
)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_lamport_add_rmsnorm import (  # noqa: E402,E501
    rubin_attn_res_lamport_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_rmsnorm import (  # noqa: E402,E501
    rubin_attn_res_add_rmsnorm,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

H = 7168
EPS = 1e-5
NO_FINISH = (0, 0, False, 0, 0)  # finish_world, finish_chunk, finish_two_shot, mc_base, rank
MIN_COSINE, MAX_RELATIVE_L2 = 0.9999, 1e-3


def _single(entry) -> list:
    return [c for c in entry.CELLS if c.spec["world"] == 1]


def _inputs(rows: int, candidates: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    addend = rn(rows, H, sc=0.5)
    addend[addend == 0] = 0  # a Lamport row never holds -0
    return dict(
        prefix=rn(rows, H),
        addend=addend,
        block=rn(candidates - 1, rows, H),
        weights=(
            rn(H, sc=0.05),
            (1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),
            (1.0 + rn(H, sc=0.1).float()).to(torch.bfloat16),
        ),
    )


def _workspace(rows: torch.Tensor, dirty: int = 2):
    """A one-rank Lamport workspace (three buffers of two stages) whose last all-reduce left
    ``rows`` in the broadcast stage of buffer ``dirty``; every other word is the sentinel."""
    m = rows.shape[0]
    stage_bytes = -(-m * H * 2 // 1024) * 1024
    bytes_per_buffer = 2 * stage_bytes
    uc = torch.full((3 * bytes_per_buffer // 4,), -(2**31), dtype=torch.int32, device="cuda")
    flags = torch.zeros(9, dtype=torch.int32, device="cuda")
    flags[0], flags[1], flags[2] = (dirty + 1) % 3, dirty, bytes_per_buffer
    start = (dirty * 2 + 1) * stage_bytes // 4
    uc[start : start + m * H // 2].copy_(rows.view(torch.int32).flatten())
    return uc, flags


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


def _assert_reference_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    a, e = out.float().flatten(), ref.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    rel = ((a - e).norm() / (e.norm() + 1e-12)).item()
    assert cosine > MIN_COSINE and rel < MAX_RELATIVE_L2, f"cos {cosine:.7f} rel {rel:.2e}"


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _tensor_twin(t, early_trigger):
    """rubin_attn_res_add_rmsnorm on the tensor addend: the same kernel, the same bits."""
    return rubin_attn_res_add_rmsnorm(
        t["prefix"], t["addend"], t["block"], *t["weights"], EPS, EPS, early_trigger=early_trigger
    )


@pytest.mark.parametrize("entry", (rubin_attn_res_lamport_add_rmsnorm,))
def test_completed_call_cells(entry, certifying) -> None:  # noqa: F811 — fixture by name
    """``finish_world == 0``: the addend is the call finished last, read from the workspace."""
    certifying(entry)
    for cell in _single(entry):
        spec = cell.spec
        t = _inputs(spec["rows"], spec["candidates"], _seed(cell))
        add_rn = spec["add_rn"]
        uc, flags = _workspace(t["addend"])
        call_args = (t["prefix"], uc.data_ptr(), flags.data_ptr(), t["block"], *t["weights"])
        flags_args = (EPS, EPS, *NO_FINISH, add_rn, False, False, False)
        got = entry(*call_args, *flags_args, early_trigger=spec["early_trigger"])
        ref = entry.reference(
            t["prefix"], [t["addend"]], t["block"], *t["weights"], EPS, EPS, 1, add_rn
        )
        assert torch.equal(_bits(got[0]), _bits(ref[0])), (cell.why, "updated prefix")
        _assert_reference_close(got[1], ref[1])
        if not add_rn:
            for a, b in zip(got[:2], _tensor_twin(t, spec["early_trigger"])):
                assert torch.equal(_bits(a), _bits(b)), cell.why


def test_guard_refuses_a_finish_flag_without_a_finish() -> None:
    t = _inputs(8, 4, seed=1)
    uc, flags = _workspace(t["addend"])
    args = (t["prefix"], uc.data_ptr(), flags.data_ptr(), t["block"], *t["weights"], EPS, EPS)
    with pytest.raises(AssertionError, match="early_poll"):
        rubin_attn_res_lamport_add_rmsnorm.is_valid(
            *args, *NO_FINISH, False, True, False, False, early_trigger=True
        )
