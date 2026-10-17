# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default attention_output_tail engine on the real tail weights of all 78 target layers.

For every target layer (3 dense destinations, 75 MoE destinations) the test
loads the real TP8 ``v_b_proj`` / ``o_proj`` shards of one rank (rank
``layer % 8``, so every rank is covered across the sweep) and of all eight
ranks for layers 1, 41 and 76, and runs the default ``cute_stream`` engine
with PDL at R ∈ {1, 6, 8} on real-weight, synthetic-activation latents (real
compressed-KV rows of the same layer, random softmax mixtures).

Checks per (layer, rank, rows):

* the frozen partial-output budget of ``test_attn_tail_attn_tail.py`` against the
  FP64 oracle with the native BF16 rounding of ``v``;
* the strict intermediate check for the default engine: attention_output_tail's BF16 ``v``
  equals native ``v`` (``MLA._bmm_bf16_out``) bit for bit, or differs by at
  most one BF16 ulp per element; the count of non-identical elements is
  printed. (The relaxed FP64-bound check of ``test_attn_tail_attn_tail.py`` applies
  to the non-default engines.)

Weights are re-read per layer from the checkpoint, so the device sees 78
distinct weight sets (no warm reuse across layers).
"""

import os
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E501
    DEFAULT_CONFIG,
    attn_tail,
    check_tail_shapes,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_fixtures  # noqa: E402
import tail_reference  # noqa: E402

TP_SIZE = 8
LOCAL_HEADS, V_DIM, C_DIM = 8, 256, 512
NUM_TARGET_LAYERS = 78
ALL_RANK_LAYERS = (1, 41, 76)
ROWS = (1, 6, 8)
_CKPT = tail_fixtures.checkpoint_dir()

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="attention_output_tail kernels target SM100/SM103",
    ),
    pytest.mark.skipif(
        not (_CKPT and os.path.isdir(_CKPT)), reason=f"{tail_fixtures.GLM52_CKPT_ENV} not available"
    ),
]


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-element distance in BF16 ulps (+0 and -0 are equal)."""

    def ordered(x):
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        mag = bits & 0x7FFF
        return torch.where(bits < 0, -mag, mag)

    return (ordered(a) - ordered(b)).abs()


@pytest.mark.parametrize("layer", range(NUM_TARGET_LAYERS))
def test_default_engine_layer(layer):
    ranks = list(range(TP_SIZE)) if layer in ALL_RANK_LAYERS else [layer % TP_SIZE]
    weights = tail_fixtures.load_tail_weights(_CKPT, layer, ranks, TP_SIZE)
    compressed = tail_fixtures.compressed_kv_rows(_CKPT, layer)
    destination = "dense" if layer < 3 else "moe"
    for rank in ranks:
        w_uv, w_o = (t.cuda() for t in weights[rank])
        for rows in ROWS:
            latent = tail_fixtures.mix_latent(
                compressed, rows, num_local_heads=LOCAL_HEADS, seed=layer * 100 + rank * 10 + rows
            )
            assert check_tail_shapes(latent, w_uv, w_o, rows, rows) is None
            v_attn_tail = torch.empty(
                (rows, LOCAL_HEADS * V_DIM), dtype=torch.bfloat16, device="cuda"
            )
            attention_output_tail = attn_tail(
                latent, w_uv, w_o, rows, v_scratch=v_attn_tail, config=DEFAULT_CONFIG
            )
            native, v_nat = tail_reference.native_attn_tail(latent, w_uv, w_o, rows, return_v=True)
            ref, _ = tail_reference.oracle_attn_tail(latent, w_uv, w_o, rows)
            e_attn_tail = tail_reference.error_stats(attention_output_tail, ref)
            e_nat = tail_reference.error_stats(native, ref)
            label = (layer, destination, rank, rows)
            assert e_attn_tail["rel_rms"] <= 1.25 * e_nat["rel_rms"] + 1e-6, (
                label,
                e_attn_tail,
                e_nat,
            )
            assert e_attn_tail["max_abs"] <= (2 * e_nat["max_abs"] + 2**-8 * e_nat["ref_max"]), (
                label,
                e_attn_tail,
                e_nat,
            )
            torch.testing.assert_close(
                attention_output_tail, native, rtol=2**-6, atol=2**-6 * e_nat["ref_rms"]
            )
            ulps = bf16_ulp_distance(v_attn_tail, v_nat)
            mismatches = int((ulps > 0).sum())
            assert int(ulps.max()) <= 1, (label, int(ulps.max()), mismatches)
            print(
                f"[attention_output_tail-pdl-stress-layer] layer={layer} dest={destination} "
                f"rank={rank} rows={rows} rel_rms attention_output_tail={e_attn_tail['rel_rms']:.3e} "
                f"native={e_nat['rel_rms']:.3e} "
                f"v_mismatch={mismatches}/{v_attn_tail.numel()} "
                f"v_max_ulp={int(ulps.max())}"
            )
        del w_uv, w_o
