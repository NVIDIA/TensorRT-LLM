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
"""SM-count contract of the self-sampling GVR top-K host route.

``route()`` and its factored forms are pure functions of
``(b, n, npad, k, num_sms, sm_version)``. These checks pin three things
without a GPU:

1. at 148 SMs the plans are byte-identical to the ones the dispatch produced
   before the CTA-split envelopes were sized from ``num_sms`` (the values
   below were frozen from that implementation);
2. on a device whose SM count differs from 148 the per-row CTA split and the
   one-/two-wave batch bounds follow ``num_sms`` and never oversubscribe it;
3. the SM103 large-batch k=2048 register-plan exception stays exactly as
   written (architecture-, SM-count- and shape-exact).
"""

import pytest

pytest.importorskip("torch")
try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import (
        gvr_topk_decode_self_sampling_host as ss_host,
    )
except ImportError as exc:  # the package __init__ pulls in the CuTe DSL kernels
    pytest.skip(
        f"self-sampling GVR host route module not importable here: {exc}",
        allow_module_level=True,
    )

pytestmark = pytest.mark.cpu_only

B200_SMS = 148
OTHER_SMS = (132, 160, 212, 296)

# (b, n, npad, k, sm_version) -> route(..., num_sms=148, sm_version) before the
# SM-count envelopes were parameterised. One entry per kernel family / rung,
# plus the one-wave (148) and two-wave (296) batch edges of the streaming slab.
FROZEN_148 = {
    (64, 1024, 1024, 512, 100): {
        "kernel": "reg",
        "tpl": (256, 1, 8, 1, True, True, False, 1024),
        "rt": {"n": 1024, "npad": 1024, "k": 512, "CMP": 1024, "IMGOFF": 1024, "QC": 96},
        "grid": (64, 1),
        "cluster": 1,
        "block": 256,
        "smem": 12288,
        "ws": False,
    },
    (1024, 4096, 4096, 1024, 100): {
        "kernel": "reg",
        "tpl": (512, 2, 4, 1, True, True, False, 1024),
        "rt": {"n": 4096, "npad": 4096, "k": 1024, "CMP": 4096, "IMGOFF": 1024, "QC": 96},
        "grid": (1024, 1),
        "cluster": 1,
        "block": 512,
        "smem": 36864,
        "ws": False,
    },
    (64, 4096, 4096, 512, 100): {
        "kernel": "regimg",
        "tpl": (1024, 1, 2, 1, True, False, True, 2048),
        "rt": {"n": 4096, "npad": 4096, "k": 512, "CMP": 2560, "IMGOFF": 2048, "QC": 96},
        "grid": (64, 1),
        "cluster": 1,
        "block": 1024,
        "smem": 28672,
        "ws": False,
    },
    (8, 65536, 65536, 1024, 100): {
        "kernel": "reg_clus",
        "tpl": (1024, 2, 8),
        "rt": {"n": 65536, "npad": 65536, "k": 1024},
        "grid": (8, 8),
        "cluster": 8,
        "block": 1024,
        "smem": 45056,
        "ws": False,
    },
    (64, 8192, 8192, 512, 100): {
        "kernel": "reg",
        "tpl": (1024, 2, 1, 1, True, False, False, 2048),
        "rt": {"n": 8192, "npad": 8192, "k": 512, "CMP": 2560, "IMGOFF": 2048, "QC": 96},
        "grid": (64, 1),
        "cluster": 1,
        "block": 1024,
        "smem": 28672,
        "ws": False,
    },
    (256, 8192, 8192, 1024, 100): {
        "kernel": "reg",
        "tpl": (512, 4, 2, 2, True, False, False, 1024),
        "rt": {"n": 8192, "npad": 8192, "k": 1024, "CMP": 2560, "IMGOFF": 1024, "QC": 96},
        "grid": (256, 1),
        "cluster": 1,
        "block": 512,
        "smem": 24576,
        "ws": False,
    },
    (64, 262144, 262144, 1024, 100): {
        "kernel": "clus",
        "tpl": (1024, 8, 1, 256, 2),
        "rt": {
            "n": 262144,
            "npad": 262144,
            "k": 1024,
            "SCAP": 8192,
            "CMP": 2048,
            "SMP": 170,
            "TGT": 31,
            "Q": 32768,
            "SS2": 96,
            "TGT2": 10,
        },
        "grid": (2, 64),
        "cluster": 2,
        "block": 1024,
        "smem": 84000,
        "ws": False,
    },
    (1, 1048576, 1048576, 1024, 100): {
        "kernel": "main",
        "tpl": (1024, 1, 1, 256, 1, True, False),
        "rt": {
            "n": 1048576,
            "npad": 1048576,
            "k": 1024,
            "SCAP_": 8192,
            "CMP_": 2048,
            "R": 148,
            "SMP": 585,
            "TGT": 15,
            "Q": 1772,
            "SS2": 224,
            "TGT2": 4,
        },
        "grid": (148, 1),
        "cluster": 1,
        "block": 1024,
        "smem": 81960,
        "ws": True,
    },
    (20, 262144, 262144, 2048, 100): {
        "kernel": "main",
        "tpl": (1024, 8, 1, 256, 2, True, False),
        "rt": {
            "n": 262144,
            "npad": 262144,
            "k": 2048,
            "SCAP_": 8192,
            "CMP_": 4096,
            "R": 7,
            "SMP": 385,
            "TGT": 48,
            "Q": 9363,
            "SS2": 85,
            "TGT2": 24,
        },
        "grid": (7, 20),
        "cluster": 1,
        "block": 1024,
        "smem": 98344,
        "ws": True,
    },
    (148, 131072, 131072, 1024, 100): {
        "kernel": "main",
        "tpl": (1024, 8, 1, 256, 1, False, False),
        "rt": {
            "n": 131072,
            "npad": 131072,
            "k": 1024,
            "SCAP_": 16384,
            "CMP_": 2048,
            "R": 1,
            "SMP": 256,
            "TGT": 64,
            "Q": 32768,
            "SS2": 64,
            "TGT2": 16,
        },
        "grid": (1, 148),
        "cluster": 1,
        "block": 1024,
        "smem": 147496,
        "ws": True,
    },
    (149, 131072, 131072, 1024, 100): {
        "kernel": "main",
        "tpl": (512, 8, 2, 256, 2, False, False),
        "rt": {
            "n": 131072,
            "npad": 131072,
            "k": 1024,
            "SCAP_": 4096,
            "CMP_": 1024,
            "R": 1,
            "SMP": 744,
            "TGT": 63,
            "Q": 32768,
            "SS2": 22,
            "TGT2": 46,
        },
        "grid": (1, 149),
        "cluster": 1,
        "block": 512,
        "smem": 41000,
        "ws": True,
    },
    (296, 131072, 131072, 1024, 100): {
        "kernel": "main",
        "tpl": (512, 8, 2, 256, 2, False, False),
        "rt": {
            "n": 131072,
            "npad": 131072,
            "k": 1024,
            "SCAP_": 4096,
            "CMP_": 1024,
            "R": 1,
            "SMP": 744,
            "TGT": 63,
            "Q": 32768,
            "SS2": 22,
            "TGT2": 46,
        },
        "grid": (1, 296),
        "cluster": 1,
        "block": 512,
        "smem": 41000,
        "ws": True,
    },
    (297, 131072, 131072, 1024, 100): {
        "kernel": "main",
        "tpl": (256, 8, 4, 256, 4, False, False),
        "rt": {
            "n": 131072,
            "npad": 131072,
            "k": 1024,
            "SCAP_": 4096,
            "CMP_": 1024,
            "R": 1,
            "SMP": 744,
            "TGT": 63,
            "Q": 32768,
            "SS2": 22,
            "TGT2": 46,
        },
        "grid": (1, 297),
        "cluster": 1,
        "block": 256,
        "smem": 24600,
        "ws": True,
    },
    # SM103 large-batch k=2048 exception: register plan on sm_103, streaming
    # slab on sm_100 and just outside the exception's shape window.
    (512, 8192, 8192, 2048, 103): {
        "kernel": "reg",
        "tpl": (512, 4, 2, 1, True, True, False, 1024),
        "rt": {"n": 8192, "npad": 8192, "k": 2048, "CMP": 8192, "IMGOFF": 1024, "QC": 96},
        "grid": (512, 1),
        "cluster": 1,
        "block": 512,
        "smem": 69632,
        "ws": False,
    },
    (1024, 8192, 8192, 2048, 103): {
        "kernel": "reg",
        "tpl": (512, 4, 2, 1, True, True, False, 1024),
        "rt": {"n": 8192, "npad": 8192, "k": 2048, "CMP": 8192, "IMGOFF": 1024, "QC": 96},
        "grid": (1024, 1),
        "cluster": 1,
        "block": 512,
        "smem": 69632,
        "ws": False,
    },
    (512, 8192, 8192, 2048, 100): {
        "kernel": "main",
        "tpl": (256, 8, 4, 256, 8, False, False),
        "rt": {
            "n": 8192,
            "npad": 8192,
            "k": 2048,
            "SCAP_": 8192,
            "CMP_": 1024,
            "R": 1,
            "SMP": 32,
            "TGT": 88,
            "Q": 2048,
            "SS2": 32,
            "TGT2": 64,
        },
        "grid": (1, 512),
        "cluster": 1,
        "block": 256,
        "smem": 40984,
        "ws": True,
    },
    (1025, 8192, 8192, 2048, 103): {
        "kernel": "main",
        "tpl": (256, 8, 4, 256, 8, False, False),
        "rt": {
            "n": 8192,
            "npad": 8192,
            "k": 2048,
            "SCAP_": 8192,
            "CMP_": 1024,
            "R": 1,
            "SMP": 32,
            "TGT": 88,
            "Q": 2048,
            "SS2": 32,
            "TGT2": 64,
        },
        "grid": (1, 1025),
        "cluster": 1,
        "block": 256,
        "smem": 40984,
        "ws": True,
    },
    (512, 8192, 8192, 1024, 103): {
        "kernel": "main",
        "tpl": (256, 8, 4, 256, 4, False, False),
        "rt": {
            "n": 8192,
            "npad": 8192,
            "k": 1024,
            "SCAP_": 4096,
            "CMP_": 1024,
            "R": 1,
            "SMP": 46,
            "TGT": 63,
            "Q": 2048,
            "SS2": 22,
            "TGT2": 46,
        },
        "grid": (1, 512),
        "cluster": 1,
        "block": 256,
        "smem": 24600,
        "ws": True,
    },
}


@pytest.mark.parametrize("shape", sorted(FROZEN_148))
def test_route_unchanged_at_148_sms(shape):
    b, n, npad, k, sm_version = shape
    expected = FROZEN_148[shape]
    assert ss_host.route(b, n, npad, k, B200_SMS, sm_version) == expected
    assert ss_host.route_split(b, n, npad, k, B200_SMS, sm_version) == expected
    if expected["kernel"] in ("main", "clus"):
        assert ss_host.route_streaming(b, n, npad, k, num_sms=B200_SMS) == expected


def test_route_default_sm_count_is_148():
    for (b, n, npad, k, sm_version), expected in FROZEN_148.items():
        assert ss_host.route(b, n, npad, k, sm_version=sm_version) == expected


@pytest.mark.parametrize("b", (1, 2, 3, 4))
@pytest.mark.parametrize("num_sms", (B200_SMS,) + OTHER_SMS)
def test_small_batch_split_follows_sm_count(b, num_sms):
    """Deep-slab SPLIT: the per-row CTA count is min(num_sms // b, chunks) and
    the (R x b) grid never exceeds one CTA per SM."""
    n, k = 1 << 20, 1024
    chunks = ((n >> 2) + 1023) // 1024
    plan = ss_host.route(b, n, n, k, num_sms, 100)
    assert plan["kernel"] == "main"
    r_rows = plan["grid"][0]
    assert r_rows == plan["rt"]["R"] == min(num_sms // b, chunks)
    assert r_rows * b <= num_sms
    assert plan["tpl"][5] is True  # SPLIT
    assert plan["rt"]["Q"] == ((n >> 2) + r_rows - 1) // r_rows
    baseline = ss_host.route(b, n, n, k, B200_SMS, 100)["grid"][0]
    if num_sms >= B200_SMS:
        assert r_rows >= baseline
    else:
        assert r_rows <= baseline
    streaming = ss_host.route_streaming(b, n, n, k, force_main=True, num_sms=num_sms)
    assert streaming == plan
    assert ss_host.route_split(b, n, n, k, num_sms, 100) == plan


@pytest.mark.parametrize("num_sms", (B200_SMS,) + OTHER_SMS)
def test_streaming_wave_bounds_follow_sm_count(num_sms):
    """One wave of BLK=1024 CTAs up to num_sms rows, two BLK=512 CTAs per SM up
    to 2 * num_sms rows, BLK=256 beyond -- the (148, 296) edges generalised."""
    n, k = 131072, 1024
    for rows, block in (
        (num_sms, 1024),
        (num_sms + 1, 512),
        (2 * num_sms, 512),
        (2 * num_sms + 1, 256),
    ):
        plan = ss_host.route(rows, n, n, k, num_sms, 100)
        assert plan["kernel"] == "main" and plan["rt"]["R"] == 1, (rows, plan)
        assert plan["block"] == block, (rows, plan)
        assert plan["grid"] == (1, rows)
    # prefill tiers are the same three bands; one representative row each
    tiers = ss_host._prefill_tier_rows(num_sms)
    assert tiers[1] == num_sms + 1 and tiers[2] == 2 * num_sms + 1
    for tier, rows in enumerate(tiers):
        assert ss_host._prefill_tier(rows, 32768, 512, num_sms) == tier
        plan = ss_host.route_streaming(rows, 32768, 32768, 512, force_main=True, num_sms=num_sms)
        assert plan["kernel"] == "main" and plan["rt"]["R"] == 1
        assert plan["block"] == (1024, 512, 256)[tier]


def test_clustered_register_split_bounded_by_sm_count():
    """reg_clus admits cs CTAs per row only when they fit one wave
    (cs <= num_sms // b). Past the b > 15 GPC-packing veto cs is 4, so the
    band ends at b = num_sms // 4 and the shape falls back to the R=2 clus
    split -- the edge moves with the SM count."""
    n, k = 65536, 1024
    for num_sms in (B200_SMS,) + OTHER_SMS:
        for b in (16, 24, 32, 33, 37, 40, 48, 53, 64, 74):
            plan = ss_host.route(b, n, n, k, num_sms, 100)
            if num_sms // b >= 4:
                assert plan["kernel"] == "reg_clus" and plan["cluster"] == 4, (num_sms, b, plan)
                assert plan["cluster"] * b <= num_sms
            else:
                assert plan["kernel"] == "clus" and plan["cluster"] == 2, (num_sms, b, plan)


def test_sm103_large_batch_k2048_exception_unchanged():
    """The exception is sm_103 + 148 SMs + 512 <= b <= 1024 + k == 2048 only."""
    n = 8192
    for b in (512, 768, 1024):
        plan = ss_host.route(b, n, n, 2048, B200_SMS, 103)
        assert plan["kernel"] == "reg" and plan["tpl"][:3] == (512, 4, 2), (b, plan)
        assert ss_host.route(b, n, n, 2048, B200_SMS, 100)["kernel"] == "main"
        assert ss_host.route(b, n, n, 1024, B200_SMS, 103)["kernel"] == "main"
        for num_sms in OTHER_SMS:
            # not the B200 SM count: the exception does not fire, the
            # generic two-wave bound decides instead
            plan = ss_host.route(b, n, n, 2048, num_sms, 103)
            assert plan["kernel"] == ("reg" if b <= 2 * num_sms else "main"), (b, num_sms, plan)
    assert ss_host.route(511, n, n, 2048, B200_SMS, 103)["kernel"] == "main"
    assert ss_host.route(1025, n, n, 2048, B200_SMS, 103)["kernel"] == "main"


def test_device_profile_keys_distinct_per_sm_count():
    keys = {ss_host._pack_device_profile(s, 100) for s in (B200_SMS,) + OTHER_SMS}
    assert len(keys) == 1 + len(OTHER_SMS)
    for num_sms in (B200_SMS,) + OTHER_SMS:
        for sm_version in (100, 103):
            packed = ss_host._pack_device_profile(num_sms, sm_version)
            assert ss_host._unpack_device_profile(packed) == (num_sms, sm_version)


def test_route_factorization_lossless_at_other_sm_counts():
    """route_split (static + dynamic) must reproduce route() for every SM count,
    and route_streaming must agree with route() wherever route() is streaming."""
    checked = 0
    for num_sms in OTHER_SMS:
        for b in (1, 3, 16, 32, 64, 148, 149, 296, 297, 512):
            for k in (512, 1024, 2048):
                for n in (k + 1, 4096, 8192, 16384, 65536, 163840, 1 << 20):
                    npad = (n + 63) // 64 * 64
                    plan = ss_host.route(b, n, npad, k, num_sms, 100)
                    assert ss_host.route_split(b, n, npad, k, num_sms, 100) == plan
                    if plan["kernel"] in ("main", "clus"):
                        assert ss_host.route_streaming(b, n, npad, k, num_sms=num_sms) == plan
                        if plan["grid"][0] > 1:
                            assert plan["grid"][0] * b <= num_sms
                    checked += 1
    assert checked > 500
