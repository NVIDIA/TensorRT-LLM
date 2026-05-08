# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for ``_validate_partition_config``.

Verifies that DwdpConfig partition fields (num_experts_per_worker,
num_prefetch_experts) are validated against the loaded chunk shape
and total expert count, catching the four distinct failure modes:

  1. non-positive size or stride
  2. stride > size (gaps between consecutive ranks)
  3. coverage < num_experts (last rank misses experts)
  4. chunk shape mismatch (DwdpConfig disagrees with fused MoE loader)

The uniform integer-division case used by current dwdp=4 / dwdp=8
configurations must pass cleanly.
"""
import unittest

from tensorrt_llm._torch.modules.dwdp.setup import _validate_partition_config


def _kwargs(**overrides):
    """Default-good DSv3 dwdp=4 partition config; override fields per test."""
    base = dict(
        num_experts_per_worker=64,
        num_prefetch_experts=64,
        num_experts_total=256,
        dwdp_size=4,
        loaded_local_experts=64,
    )
    base.update(overrides)
    return base


class TestValidatePartitionConfig(unittest.TestCase):

    # ------------------------------------------------------------------
    # Happy paths — must NOT raise
    # ------------------------------------------------------------------

    def test_uniform_dwdp4(self):
        # 256 / 4 = 64. Stride == size. Standard case.
        _validate_partition_config(**_kwargs())

    def test_uniform_dwdp8(self):
        # 256 / 8 = 32. Cross-tray DWDP=8 config (verified 2026-04-28).
        _validate_partition_config(**_kwargs(
            num_experts_per_worker=32,
            num_prefetch_experts=32,
            dwdp_size=8,
            loaded_local_experts=32,
        ))

    def test_uniform_dwdp2(self):
        # DSv3-Lite 72 / 2 = 36. Used by accuracy integration test.
        _validate_partition_config(**_kwargs(
            num_experts_per_worker=36,
            num_prefetch_experts=36,
            num_experts_total=72,
            dwdp_size=2,
            loaded_local_experts=36,
        ))

    # ------------------------------------------------------------------
    # Failure mode 1: non-positive size or stride
    # ------------------------------------------------------------------

    def test_zero_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_experts_per_worker"):
            _validate_partition_config(**_kwargs(num_experts_per_worker=0))

    def test_negative_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_experts_per_worker"):
            _validate_partition_config(**_kwargs(num_experts_per_worker=-5))

    def test_zero_stride_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_prefetch_experts"):
            _validate_partition_config(**_kwargs(num_prefetch_experts=0))

    def test_negative_stride_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_prefetch_experts"):
            _validate_partition_config(**_kwargs(num_prefetch_experts=-1))

    # ------------------------------------------------------------------
    # Failure mode 2: stride > size (gaps)
    # ------------------------------------------------------------------

    def test_stride_exceeds_size_rejected(self):
        # size=64, stride=128 leaves [64, 128) un-owned between rank 0 and 1.
        with self.assertRaisesRegex(ValueError, "stride larger than the range"):
            _validate_partition_config(**_kwargs(num_prefetch_experts=128))

    # ------------------------------------------------------------------
    # Failure mode 3: insufficient coverage
    # ------------------------------------------------------------------

    def test_partial_coverage_rejected(self):
        # 4 ranks of size 32, stride 32 covers only [0, 128) of 256 experts.
        with self.assertRaisesRegex(ValueError, "does not cover all experts"):
            _validate_partition_config(**_kwargs(
                num_experts_per_worker=32,
                num_prefetch_experts=32,
                loaded_local_experts=32,
            ))

    def test_coverage_exact_match_passes(self):
        # 4 ranks of size 64, stride 64 covers exactly [0, 256). Boundary case.
        _validate_partition_config(**_kwargs())  # uniform default already covers exactly

    def test_coverage_with_overlap_passes(self):
        # 4 ranks of size 70, stride 62: 3*62+70 = 256 — exact equality.
        # 8-expert overlap between adjacent ranks is the redundancy case.
        _validate_partition_config(**_kwargs(
            num_experts_per_worker=70,
            num_prefetch_experts=62,
            loaded_local_experts=70,
        ))

    def test_dwdp3_mode_b_overlap_passes(self):
        # dwdp=3, 256 experts: Mode B with stride=85, size=86.
        # 2*85 + 86 = 256 — exact equality, 1-expert overlap between
        # adjacent ranks.  This is the canonical "dwdp does not divide
        # num_experts" recipe.
        _validate_partition_config(
            num_experts_per_worker=86,
            num_prefetch_experts=85,
            num_experts_total=256,
            dwdp_size=3,
            loaded_local_experts=86,
        )

    def test_dwdp5_mode_b_overlap_passes(self):
        # dwdp=5, 256 experts: stride=51, size=52. 4*51 + 52 = 256.
        _validate_partition_config(
            num_experts_per_worker=52,
            num_prefetch_experts=51,
            num_experts_total=256,
            dwdp_size=5,
            loaded_local_experts=52,
        )

    def test_dwdp7_mode_b_overlap_passes(self):
        # dwdp=7, 256 experts: 256 % 7 = 4, so Mode B requires more
        # overlap.  size=40, stride=36: 6*36 + 40 = 256.
        _validate_partition_config(
            num_experts_per_worker=40,
            num_prefetch_experts=36,
            num_experts_total=256,
            dwdp_size=7,
            loaded_local_experts=40,
        )

    # ------------------------------------------------------------------
    # Failure mode 3b: tail-padding (coverage > num_experts) rejected
    # ------------------------------------------------------------------

    def test_tail_padding_rejected_dwdp3(self):
        # dwdp=3, 256 experts with size=stride=86 → 2*86+86=258 > 256.
        # Last rank's storage [172, 258) extends past num_experts=256.
        # GB200 fabric-handle ABI cannot map this partially, so reject.
        with self.assertRaisesRegex(ValueError, "exceeds num_experts"):
            _validate_partition_config(
                num_experts_per_worker=86,
                num_prefetch_experts=86,
                num_experts_total=256,
                dwdp_size=3,
                loaded_local_experts=86,
            )

    def test_tail_padding_rejected_dwdp5(self):
        # dwdp=5, 256 experts with size=stride=52 → 4*52+52=260 > 256.
        with self.assertRaisesRegex(ValueError, "exceeds num_experts"):
            _validate_partition_config(
                num_experts_per_worker=52,
                num_prefetch_experts=52,
                num_experts_total=256,
                dwdp_size=5,
                loaded_local_experts=52,
            )

    # ------------------------------------------------------------------
    # Failure mode 4: chunk shape mismatch
    # ------------------------------------------------------------------

    def test_chunk_shape_smaller_than_size_rejected(self):
        # DwdpConfig says size=64 but loader produced 32-expert chunks.
        with self.assertRaisesRegex(ValueError, "fused MoE chunk shape"):
            _validate_partition_config(**_kwargs(loaded_local_experts=32))

    def test_chunk_shape_larger_than_size_rejected(self):
        # DwdpConfig says size=64 but loader produced 96-expert chunks (e.g.
        # if ep_size differs from dwdp_size, which would be a misconfiguration).
        with self.assertRaisesRegex(ValueError, "fused MoE chunk shape"):
            _validate_partition_config(**_kwargs(loaded_local_experts=96))

    # ------------------------------------------------------------------
    # Multi-failure: stricter check fires first
    # ------------------------------------------------------------------

    def test_multiple_failures_one_raised(self):
        # All checks are independent; first violated check should raise.
        # zero size triggers before any other condition is examined.
        with self.assertRaises(ValueError):
            _validate_partition_config(**_kwargs(
                num_experts_per_worker=0,
                num_prefetch_experts=128,
                loaded_local_experts=0,
            ))


if __name__ == "__main__":
    unittest.main()
