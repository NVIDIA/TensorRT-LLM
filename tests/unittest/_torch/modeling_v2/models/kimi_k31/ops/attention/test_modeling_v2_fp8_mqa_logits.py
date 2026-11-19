# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 fp8_mqa_logits entry (DeepGEMM prefill logits).

On the GPU, each cell is a prefill chunk: per request a cached prefix and new query rows, the
chunk's keys back to back, each row's window its request's causal keys. Only the windows are
compared (the columns outside them hold unspecified values). A scheduled cell passes DeepGEMM's
schedule of the call.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.fp8_mqa_logits import (  # noqa: E501
    fp8_mqa_logits as op,
)

__extra_import_path__ = ["../../../..", ".", "../.."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key=None)[:2]


def _schedule(args: dict) -> torch.Tensor:
    """DeepGEMM's schedule of the call ``args`` describe."""
    from tensorrt_llm.deep_gemm import get_mqa_logits_metadata

    return get_mqa_logits_metadata(
        args["cu_seqlen_ks"], args["cu_seqlen_ke"], args["k"].shape[0], args["q"].shape[1]
    )


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    ks, ke, num_keys = inputs.chunk_windows(spec["requests"], spec["q_rows"])
    rows = ks.shape[0]
    k, k_scale = inputs.keys(num_keys, spec["data"], gen)
    args = dict(
        q=inputs.queries(rows, spec["data"], gen),
        k=k.cuda(),
        k_scale=k_scale.cuda(),
        weights=inputs.weights(rows, spec["data"], gen),
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )
    if spec["scheduled"]:
        args["schedule_meta"] = _schedule(args)
    return args


@inputs.on_sm103
@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:48] for c in _CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    window = torch.isfinite(ref)
    if cell.spec["data"] == "int":
        assert torch.equal(out[window], ref[window])
    else:
        kf = args["k"].double() * args["k_scale"].double()[:, None]
        band = inputs.accumulation_band(args["q"], kf, args["weights"], window)
        err = (out.double() - ref.double()).masked_fill(~window, 0).abs()
        assert (err <= band).all(), f"max excess {float((err - band).max()):.3e}"
