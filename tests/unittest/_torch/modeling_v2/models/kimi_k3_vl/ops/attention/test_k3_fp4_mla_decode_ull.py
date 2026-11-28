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
"""Single-rank Rubin checks for the Kimi K3 ULL FP4 KV update and decode."""

import importlib.util
from pathlib import Path

import torch


def test_ull_kv_append_and_decode_match_backend() -> None:
    """Check the ULL KV-only update and packed decode on one rank of a Helix step."""
    from tensorrt_llm._torch.attention.backends import fp4_mla
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.attention import _fp4_mla_decode_ull
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.attention.fp4_mla_kv_append_ull import (  # noqa: E501
        fp4_mla_kv_append,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.attention.k3_fp4_mla_decode_ull import (  # noqa: E501
        k3_fp4_mla_decode,
    )

    # Reuse the family's FP4 fixtures and backend reference under either pytest import mode.
    path = Path(__file__).with_name("test_k3_fp4_mla_decode.py")
    spec = importlib.util.spec_from_file_location("_k3_fp4_mla_decode_test_helpers", path)
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    qf = helpers._sm107_q_fused_helpers()
    mgr, md = qf._fp4_case(1, [513], seed=29)
    try:
        rot = qf._rotary_table(md.max_seq_len, "identity")
        attn = qf._attention(rot)
        latent, q_input, q_pe = helpers._decode_inputs(qf, qf.Q_LEN, seed=29)
        pools = qf._pools(mgr)
        initial_pools = [pool.clone() for pool in pools]

        qf._scatter(md, latent, rot, q_pe, q_input[..., qf.LORA :], q_input)
        state = md.fp4_mla_state
        query_bytes = state.prequantized_q.clone()
        query_scales = state.prequantized_q_sf.clone()
        sm_scale = 1.0 / (attn.q_scaling * (attn.qk_nope_head_dim + attn.qk_rope_head_dim) ** 0.5)
        expected, expected_stats = helpers._run(
            md, qf, fp4_mla.run_fp4_mla_attention_decode, helix=True, sm_scale=sm_scale
        )
        expected_pools = [pool.clone() for pool in pools]
        for pool, initial in zip(pools, initial_pools):
            pool.copy_(initial)

        _fp4_mla_decode_ull.begin_step(md)
        fp4_mla_kv_append(attn, latent, md, qf.H)
        # Give both decode paths identical query rows so this test isolates KV append and decode.
        state.prequantized_q.copy_(query_bytes)
        state.prequantized_q_sf.copy_(query_scales)
        assert expected_stats is not None
        stats = torch.full_like(expected_stats, float("nan"))
        out = k3_fp4_mla_decode(attn, q_input.flatten(1), md, softmax_stats_tensor=stats)

        for pool, reference in zip(pools, expected_pools):
            assert torch.equal(pool, reference)
        helpers._assert_close_rows(out.view_as(expected), expected)
        torch.testing.assert_close(stats[..., 0], expected_stats[..., 0], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(stats[..., 1], expected_stats[..., 1], atol=5e-2, rtol=2e-2)
    finally:
        mgr.shutdown()
