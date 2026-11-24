# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 kda_prefill_segments entry.

Cells are built like the kda_prefill cells (packed raw inputs, strided beta, padded-slot V-first pool, planned
metadata). Output and the selected state rows are checked against the fp32 token-by-token recurrence with
kda_prefill's bands; every other pool row must be untouched. The segment-parallel cells must also stay within
those bands of the sequential kda_prefill; the fallback cells must equal it bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.ssm import (
    kda_prefill_segments as segments_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_prefill_segments import (  # noqa: E501
    MIN_TOKENS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.ssm.kda_prefill_segments import (  # noqa: E501
    k31_kda_prefill_segments as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill import (
    kda_prefill,
)

__extra_import_path__ = ["../../../../..", "../../../ops/ssm"]
from _kda_ops_util import (  # noqa: E402 — needs the path declared above
    assert_scaled_close,
    bits,
    clone_strided,
    generator,
    requires_sm103,
)
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name
from test_kda_prefill import POOL_SLOTS, TOL_OUT, TOL_STATE, _inputs  # noqa: E402

pytestmark = requires_sm103


def _segmented(spec) -> bool:
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return (
        len(spec["seq_lens"]) == 1
        and spec["seq_lens"][0] >= MIN_TOKENS
        and 10 * spec["heads"] <= sms
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _inputs(cell.spec, generator(cell))
    pool = args["state_pool"]
    seq_args = dict(args, state_pool=clone_strided(pool))
    ref_args = dict(args, state_pool=clone_strided(pool))
    out = op(**args)
    seq = kda_prefill(**seq_args)
    expected = op.reference(**ref_args)
    assert out.shape == args["v"].shape and out.dtype == torch.bfloat16
    assert_scaled_close(out, expected, TOL_OUT, "output")
    selected = set(cell.spec["slots"])
    for slot in range(POOL_SLOTS):
        got = pool[slot]
        if slot in selected:
            assert_scaled_close(got, ref_args["state_pool"][slot], TOL_STATE, f"state slot {slot}")
        else:
            assert torch.equal(bits(got), bits(ref_args["state_pool"][slot])), (
                f"slot {slot} changed"
            )
    if _segmented(cell.spec):
        assert_scaled_close(out, seq, TOL_OUT, "output vs kda_prefill")
        for slot in selected:
            assert_scaled_close(
                pool[slot], seq_args["state_pool"][slot], TOL_STATE, "state vs kda_prefill"
            )
    else:
        assert torch.equal(bits(out), bits(seq)), "fallback output differs from kda_prefill"
        for slot in selected:
            assert torch.equal(bits(pool[slot]), bits(seq_args["state_pool"][slot]))


@torch.inference_mode()
def test_repeat_calls_agree() -> None:
    """Deterministic: the same inputs give the same output and final state twice."""
    spec = op.CELLS[0].spec
    args = _inputs(spec, torch.Generator("cuda").manual_seed(9))
    first = dict(args, state_pool=clone_strided(args["state_pool"]))
    second = dict(args, state_pool=clone_strided(args["state_pool"]))
    a, b = op(**first), op(**second)
    assert torch.equal(bits(a), bits(b))
    slot = spec["slots"][0]
    assert torch.equal(bits(first["state_pool"][slot]), bits(second["state_pool"][slot]))


def test_zero_values_share_one_buffer_per_row_shape() -> None:
    """Prefill chunks of any length slice one zero buffer per row shape, so its memory stays bounded."""
    segments_kernel._ZEROS.clear()
    for tokens in (4096, 1000, 16384, 77, 16383):
        v = torch.empty(1, tokens, 4, 128, dtype=torch.bfloat16, device="cuda")
        zeros = segments_kernel._zeros_like(v)
        assert zeros.shape == v.shape and zeros.dtype == v.dtype and zeros.is_contiguous()
        assert not zeros.any()
    assert len(segments_kernel._ZEROS) == 1
    assert segments_kernel._ZEROS[((4, 128), torch.bfloat16, "cuda:0")].shape[1] == 16384
    segments_kernel._ZEROS.clear()
