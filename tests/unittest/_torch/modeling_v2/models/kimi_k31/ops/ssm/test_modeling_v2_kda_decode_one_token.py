# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_decode entry.

Each cell builds the KDA layer's one-token decode call: q/k/v/gate column views of the fused [q | k | v | g]
GEMM output, beta a column view of the merged [f_a | b | pad] projection, the three section views of a padded-slot
packed conv pool (updated in place) and the indexed fp32 state pool. Output and state rows within the decode bands
of an fp32 evaluation, conv windows bit for bit, every other slot untouched.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_decode import kda_decode

from ._kda_ops_util import (
    HEAD_DIM,
    assert_rows,
    assert_scaled_close,
    bits,
    clone_strided,
    decode_args,
    generator,
    requires_sm103,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

#: A boundary and a typical cell of ``kda_decode.CELLS``.
_CELLS = sample_cells(kda_decode.CELLS, size_key="batch")[:2]


# Approximate exp / rcp in the kernel (main's decode entry bands).
TOL_OUT, TOL_STATE = 2e-2, 1e-3


def _reference_args(args, conv_src):
    """Same-layout copies of everything the op writes."""
    ref_conv = clone_strided(conv_src)
    dim = args["x_q"].shape[-2] * HEAD_DIM
    return dict(
        args,
        conv_state_q=ref_conv[:, :dim],
        conv_state_k=ref_conv[:, dim : 2 * dim],
        conv_state_v=ref_conv[:, 2 * dim :],
        state=clone_strided(args["state"]),
        output=torch.empty_like(args["output"]),
    ), ref_conv


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(kda_decode)
    args, conv_src = decode_args(cell.spec, generator(cell))
    ref_args, ref_conv = _reference_args(args, conv_src)
    assert kda_decode(**args) is None
    kda_decode.reference(**ref_args)
    assert_scaled_close(args["output"], ref_args["output"], TOL_OUT, "output")
    assert torch.equal(bits(conv_src), bits(ref_conv)), (
        "conv windows differ (selected or other slots)"
    )
    selected = set(args["ssm_state_indices"].tolist())
    assert_rows(args["state"], ref_args["state"], selected, TOL_STATE)
