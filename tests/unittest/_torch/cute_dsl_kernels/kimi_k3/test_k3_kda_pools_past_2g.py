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
"""The fused KDA ops (``trtllm::k3_kda_decode_attn``, ``trtllm::k3_kda_attn``, ``trtllm::k3_kda_verify``) on pools
whose last slot starts past element 2^31, at the per-rank TP16 shape (6 heads, K = V = 128, conv width 4).

The Mamba cache manager coalesces a rank's KDA layers inside each slot, so a layer's state pool is a view at slot
stride layers x 98,304 fp32 elements (its conv pool at layers x 6,912 bf16): with enough slots the pools reach past
2^31 elements. Here each pool is the last layer's view of such a pool ([slots, layers, ...] underneath), and the
per-token verify states are dense at 7 x 98,304 elements per slot; 3,122 slots put the last slot of each past 2^31.

Each op runs once on requests in the pools' first, middle and last slots (``k3_kda_attn``: one request, the last slot)
and once on small pools holding copies of those slots; the outputs and every slot the call updates must agree bit for
bit. About 19 GB of device memory.

  pytest test_k3_kda_pools_past_2g.py
"""

import pytest
import torch

H = 6
K = V = 128
HK = H * K
W = 4
PROJ = 4 * HK + K + H + 2  # the fused [q | k | v | onorm gate | f_a | b | pad] row (3208 columns)
K_IN = 7168
NUM_SPEC = 7
NT = NUM_SPEC + 1
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = K**-0.5
SLOTS = 3122
SSM_LAYERS = 8
CONV_LAYERS = 100
TWO_G = 2**31


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")


def _ops():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn import op
    from tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify import op as _verify  # noqa: F401

    return op


def make_weights(seed: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=g, device="cuda") * scale

    return {
        "w": rnd(PROJ, K_IN, scale=0.02).bfloat16(), "w_fb": rnd(HK, K, scale=0.05).bfloat16(),
        "w_q": rnd(HK, W, scale=0.3), "w_k": rnd(HK, W, scale=0.3), "w_v": rnd(HK, W, scale=0.3),
        "a_log": rnd(H, scale=0.5), "dt_bias": rnd(HK, scale=0.5), "onorm_w": (1 + 0.1 * rnd(V)).float(),
    }  # fmt: skip


def layer_view(layers: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """The last layer's view of a pool that keeps ``layers`` layers' states in each of its slots."""
    return torch.zeros(SLOTS, layers, *shape, dtype=dtype, device="cuda")[:, -1]


def last_slot_offset(t: torch.Tensor) -> int:
    return (t.shape[0] - 1) * t.stride(0)


def verify_pools() -> dict:
    """Conv caches [slots, HK, W - 1 + NUM_SPEC] (dim-contiguous), the SSM state, the drafts' records, the pending
    counts."""
    p = {
        name: torch.zeros(SLOTS, W - 1 + NUM_SPEC, HK, device="cuda").transpose(1, 2)
        for name in ("cs_q", "cs_k", "cs_v")
    }
    p["ssm"] = layer_view(SSM_LAYERS, (H, V, K), torch.float32)
    p["state_tok"] = torch.zeros(SLOTS, NUM_SPEC, H, V, K, device="cuda")
    p["pending"] = torch.zeros(SLOTS, dtype=torch.int32, device="cuda")
    assert last_slot_offset(p["ssm"]) >= TWO_G and last_slot_offset(p["state_tok"]) >= TWO_G
    return p


def fill_verify_slots(p: dict, used: list, pending: list, seed: int) -> dict:
    """Random contents in the used slots; returns small pools holding copies of them, in the same layouts."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    for i, s in enumerate(used):
        for name in ("cs_q", "cs_k", "cs_v"):
            p[name][s] = torch.randn(HK, W - 1 + NUM_SPEC, generator=g, device="cuda") * 0.5
        p["ssm"][s] = torch.randn(H, V, K, generator=g, device="cuda") * 0.05
        p["state_tok"][s] = torch.rand(NUM_SPEC, H, V, K, generator=g, device="cuda") * 0.5
        p["pending"][s] = pending[i]
    small = {
        name: torch.stack([p[name][s].t() for s in used]).transpose(1, 2)
        for name in ("cs_q", "cs_k", "cs_v")
    }
    for name in ("ssm", "state_tok", "pending"):
        small[name] = p[name][used]
    return small


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-equal (NaN-safe)."""
    bits = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(bits), b.contiguous().view(bits))


def check_slots(p: dict, small: dict, used: list, names: tuple) -> None:
    for i, s in enumerate(used):
        for name in names:
            assert same(p[name][s], small[name][i]), f"slot {s} {name}"


def test_decode_attn():
    op = _ops()
    with torch.inference_mode():
        wt = make_weights(1)
        p = {
            "ssm": layer_view(SSM_LAYERS, (H, V, K), torch.float32),
            "conv": layer_view(CONV_LAYERS, (3 * HK, W - 1), torch.bfloat16),
        }
        assert last_slot_offset(p["ssm"]) >= TWO_G and last_slot_offset(p["conv"]) >= TWO_G
        used = [0, SLOTS // 2, SLOTS - 1]
        g = torch.Generator(device="cuda").manual_seed(2)
        for s in used:
            p["ssm"][s] = torch.randn(H, V, K, generator=g, device="cuda") * 0.05
            p["conv"][s] = (torch.randn(3 * HK, W - 1, generator=g, device="cuda") * 0.5).bfloat16()
        small = {name: t[used] for name, t in p.items()}
        x = torch.randn(len(used), K_IN, generator=g, device="cuda").bfloat16()

        def call(q: dict, slots: list) -> torch.Tensor:
            bufs = op.make_buffers(torch.device("cuda"), op.FUSED_CTAS)
            return torch.ops.trtllm.k3_kda_decode_attn(
                x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
                q["conv"], q["ssm"], torch.tensor(slots, dtype=torch.int32, device="cuda"), *bufs, LOWER_BOUND,
                SCALE, EPS,
            )  # fmt: skip

        want = call(small, list(range(len(used))))
        got = call(p, used)
        torch.cuda.synchronize()
        assert same(got, want)
        check_slots(p, small, used, ("ssm", "conv"))


def test_attn():
    op = _ops()
    with torch.inference_mode():
        wt = make_weights(3)
        p = verify_pools()
        used = [SLOTS - 1]
        small = fill_verify_slots(p, used, [5], 4)
        x = torch.randn(
            NT, K_IN, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda"
        ).bfloat16()

        def call(q: dict, slot: int) -> torch.Tensor:
            bufs = op.make_buffers(torch.device("cuda"), op.FUSED_CTAS)
            return torch.ops.trtllm.k3_kda_attn(
                x, wt["w"], wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
                q["cs_q"], q["cs_k"], q["cs_v"], q["ssm"], q["state_tok"],
                torch.tensor([slot], dtype=torch.int32, device="cuda"), q["pending"], *bufs, NUM_SPEC, LOWER_BOUND,
                SCALE, EPS,
            )  # fmt: skip

        want = call(small, 0)
        got = call(p, used[0])
        torch.cuda.synchronize()
        assert same(got, want)
        check_slots(p, small, used, ("cs_q", "cs_k", "cs_v", "ssm", "state_tok"))


def test_verify():
    _ops()
    with torch.inference_mode():
        wt = make_weights(6)
        p = verify_pools()
        used = [0, SLOTS // 2, SLOTS - 1]
        small = fill_verify_slots(p, used, [0, 3, 7], 7)
        g = torch.Generator(device="cuda").manual_seed(8)
        proj = torch.randn(len(used) * NT, PROJ, generator=g, device="cuda").bfloat16()

        def call(q: dict, slots: list) -> torch.Tensor:
            return torch.ops.trtllm.k3_kda_verify(
                proj, wt["w_fb"], wt["w_q"], wt["w_k"], wt["w_v"], wt["a_log"], wt["dt_bias"], wt["onorm_w"],
                q["cs_q"], q["cs_k"], q["cs_v"], q["ssm"], q["state_tok"],
                torch.tensor(slots, dtype=torch.int32, device="cuda"), q["pending"], NUM_SPEC, LOWER_BOUND, SCALE,
                EPS,
            )  # fmt: skip

        want = call(small, list(range(len(used))))
        got = call(p, used)
        torch.cuda.synchronize()
        assert same(got, want)
        check_slots(p, small, used, ("cs_q", "cs_k", "cs_v", "ssm", "state_tok"))
