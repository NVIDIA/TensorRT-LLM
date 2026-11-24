# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attention-TP target's attn_res_mailbox_add_rmsnorm entry: the pre-attention mix
that takes its MoE addend from the fused latent-MoE tail's mailbox and re-arms the rows it reads."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.norm.attn_res_mailbox_add_rmsnorm import (  # noqa: E501
    k31_attn_res_mailbox_add_rmsnorm as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm._attn_res_common import (  # noqa: E501
    HIDDEN,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../../..", "../../../ops/norm"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_attn_res_inputs import OUT_EPS, RMS_EPS, assert_mix_close, mix_inputs  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the attn_res entries run a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "the attn_res entries require a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    prefix, moe, block, weights = mix_inputs(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**28)
    capacity = 192
    mailbox = torch.full((1, capacity, HIDDEN), -0.0, dtype=torch.bfloat16, device="cuda")
    rows = moe.clone()
    rows.view(torch.int16)[rows.view(torch.int16) == -0x8000] = 0  # the tail stores +0 for -0
    mailbox[0, : moe.shape[0]] = rows
    ref_updated, ref_out = op.reference(prefix, mailbox.clone(), block, *weights, RMS_EPS, OUT_EPS)
    with validating(op):
        updated, out = op(prefix, mailbox, block, *weights, RMS_EPS, OUT_EPS)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16))
    assert_mix_close(out, ref_out, weights[2])
    assert bool((mailbox.view(torch.int16) == -0x8000).all()), "the rows read must be re-armed"
