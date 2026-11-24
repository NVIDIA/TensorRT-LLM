# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context KDA entries read the packed projections in place (sm_107).

A context step hands the convolution the first 3D columns of the fused [q | k | v | g] projection
and the prefill the beta columns of the [f_a | b | pad] projection, both strided views of GEMM
outputs (the FP8 and MXFP8 projections write the same packed layout). Reading them in place must
give the bits of dense copies: outputs, conv pool and state rows.
"""

import pytest
import torch
from _rubin_kda_prefill_inputs import HEAD_DIM, conv_args, prefill_args

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill import (
    rubin_kda_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_prefill_conv import (
    rubin_kda_prefill_conv,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


@pytest.mark.parametrize("heads", [96, 12])
@torch.inference_mode()
def test_conv_reads_the_packed_projection_in_place(heads) -> None:
    spec = dict(
        heads=heads,
        seq_lens=(1, 17, 64, 300),
        has_init=(True, False, True, False),
        pool_slots=8,
        src_width_of=4,
    )
    packed = conv_args(spec, seed=heads)
    assert packed["src"].stride(0) == 4 * heads * HEAD_DIM
    dense = dict(packed, src=packed["src"].contiguous(), conv_state=packed["conv_state"].clone())
    out_packed = rubin_kda_prefill_conv(**packed)
    out_dense = rubin_kda_prefill_conv(**dense)
    torch.cuda.synchronize()
    for name, a, b in zip("qkv", out_packed, out_dense):
        assert torch.equal(_bits(a), _bits(b)), name
    assert torch.equal(_bits(packed["conv_state"]), _bits(dense["conv_state"]))


@pytest.mark.parametrize(
    "heads,seq_lens",
    [(12, (5, 1, 16, 17, 33, 2, 64, 9)), (12, (100, 64, 300, 37)), (96, (100, 64, 300, 37))],
)
@torch.inference_mode()
def test_prefill_reads_beta_from_the_merged_projection(heads, seq_lens) -> None:
    """The decomposition path reads the strided beta directly, the engine path a packed copy (the
    occupancy router picks by sequences x heads). Either way the bits of a dense beta."""
    spec = dict(
        heads=heads,
        seq_lens=seq_lens,
        slots=tuple(range(len(seq_lens))),
        fresh=tuple(i % 2 == 0 for i in range(len(seq_lens))),
        state_dtype="float32",
        beta_row_width=(HEAD_DIM + heads + 7) // 8 * 8,
    )
    strided = prefill_args(spec, seed=len(seq_lens) + heads)
    assert not strided["beta"].is_contiguous()
    dense = dict(
        strided, beta=strided["beta"].contiguous(), state_pool=strided["state_pool"].clone()
    )
    out_strided = rubin_kda_prefill(**strided)
    out_dense = rubin_kda_prefill(**dense)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out_strided), _bits(out_dense))
    assert torch.equal(_bits(strided["state_pool"]), _bits(dense["state_pool"]))
