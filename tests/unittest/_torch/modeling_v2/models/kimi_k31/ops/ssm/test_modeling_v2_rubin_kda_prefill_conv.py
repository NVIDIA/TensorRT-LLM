# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_kda_prefill_conv entry on sm_107.

The cells run the fused convolution through its guard against the entry's reference (outputs
within its band, the conv pool bit for bit).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill_conv import (
    rubin_kda_prefill_conv,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _kda_ops_util import (  # noqa: E402 — needs the path declared above
    assert_scaled_close,
    bits,
    clone_written,
)
from _rubin_kda_prefill_inputs import conv_args  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(107)

#: A boundary and a typical cell of ``rubin_kda_prefill_conv.CELLS``.
_CELLS = sample_cells(rubin_kda_prefill_conv.CELLS, size_key="pool_slots")[:2]


# fp32 accumulation and a fast SiLU, one rounding per output (the entry's note).
TOL_CONV = 1.5e-2


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(rubin_kda_prefill_conv)
    args = conv_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    ref_args = clone_written(args, ("conv_state",))
    out = rubin_kda_prefill_conv(**args)
    ref = rubin_kda_prefill_conv.reference(**ref_args)
    torch.cuda.synchronize()
    for name, got, want in zip("qkv", out, ref):
        assert_scaled_close(got, want, TOL_CONV, name)
    # The window update copies raw inputs: every slot bit for bit, selected or not.
    assert torch.equal(bits(args["conv_state"]), bits(ref_args["conv_state"]))
