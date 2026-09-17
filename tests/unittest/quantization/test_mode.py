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
import unittest

import pytest

from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.quantization.mode import (
    NVFP4_MARLIN_SM_VERSIONS,
    QuantAlgo,
    get_fp4_support_error_message,
    get_fp4_supported_sm_versions,
    is_fp4_supported,
)

pytestmark = pytest.mark.cpu_only


class TestQuantMode(unittest.TestCase):
    def test_all(self):
        # Set activations and weights flags.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS

        # Make sure _all returns True when asked for both ACTIVATIONS and INT8_WEIGHTS.
        self.assertTrue(qm._all(QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS))
        # Make sure _all returns False when asked only for ACTIVATIONS.
        self.assertFalse(qm._all(QuantMode.ACTIVATIONS))
        # Make sure _all returns True when asked only for ACTIVATIONS if limited to ACTIVATIONS flag.
        self.assertTrue(qm._all(QuantMode.ACTIVATIONS, mask=QuantMode.ACTIVATIONS))

    def test_any(self):
        # Set activations and weights flags.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS

        # Make sure _any returns True when asked for both ACTIVATIONS and INT8_WEIGHTS.
        self.assertTrue(qm._any(QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS))
        # Make sure _any returns True when asked only for ACTIVATIONS.
        self.assertTrue(qm._any(QuantMode.ACTIVATIONS))
        # Make sure _any returns False when asked for PER_TOKEN.
        self.assertFalse(qm._any(QuantMode.PER_TOKEN))

    def test_count(self):
        # Make sure the COUNT value is as expected - change that test if you add a new flag.
        self.assertEqual(QuantMode.COUNT.value, 1 << 19)

    def test_from_description(self):
        # Test weight only.
        qm = QuantMode.from_description(True, False, False, False)
        # Make sure only the INT8_WEIGHTS flag is set.
        self.assertEqual(qm, QuantMode.INT8_WEIGHTS)

        # Test weight only.
        qm = QuantMode.use_weight_only()
        # Make sure only the INT8_WEIGHTS flag is set.
        self.assertEqual(qm, QuantMode.INT8_WEIGHTS)

        # Test weight only (int4).
        qm = QuantMode.from_description(True, False, False, False, False, True)
        # Make sure only the INT4_WEIGHTS flag is set.
        self.assertEqual(qm, QuantMode.INT4_WEIGHTS)

        # Test weight only.
        qm = QuantMode.use_weight_only(use_int4_weights=True)
        # Make sure only the INT4_WEIGHTS flag is set.
        self.assertEqual(qm, QuantMode.INT4_WEIGHTS)

        # Test activation/weight per-tensor.
        qm = QuantMode.from_description(True, True, False, False)
        # The reference.
        expected_qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS
        # Make sure ACTIVATIONS and INT8_WEIGHTS flags are set.
        self.assertEqual(qm, expected_qm)

        # Test activation/weight per-tensor.
        qm = QuantMode.use_smooth_quant()
        # Make sure ACTIVATIONS and INT8_WEIGHTS flags are set.
        self.assertEqual(qm, expected_qm)

        # Test activation/weight per-tensor & per-channel.
        qm = QuantMode.from_description(True, True, False, True)
        # The reference.
        expected_qm = expected_qm | QuantMode.PER_CHANNEL
        # Make sure ACTIVATIONS, INT8_WEIGHTS and PER_CHANNEL flags are set.
        self.assertEqual(qm, expected_qm)

        # Test activation/weight per-tensor & per-channel.
        qm = QuantMode.use_smooth_quant(per_channel=True)
        # Make sure ACTIVATIONS, INT8_WEIGHTS and PER_CHANNEL flags are set.
        self.assertEqual(qm, expected_qm)

        # Test activation/weight per-token & per-channel.
        qm = QuantMode.from_description(True, True, True, True)
        # The expected result.
        expected_qm = expected_qm | QuantMode.PER_TOKEN
        # Make sure all flags are set.
        self.assertEqual(qm, expected_qm)

        # Test activation/weight per-token & per-channel.
        qm = QuantMode.use_smooth_quant(True, True)
        # Make sure all flags are set.
        self.assertEqual(qm, expected_qm)

    def test_per_channel(self):
        # Set per-channel flag.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS | QuantMode.PER_CHANNEL
        # Make sure it returns True for per-channel.
        self.assertTrue(qm.has_per_channel_scaling())
        # Do not set per-channel flag.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS
        # Make sure it returns False for per-channel.
        self.assertFalse(qm.has_per_channel_scaling())

    def test_per_token(self):
        # Set per-token flag.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS | QuantMode.PER_TOKEN
        # Make sure it returns True for per-token.
        self.assertTrue(qm.has_per_token_dynamic_scaling())
        # Make sure it returns False for per-tensor.
        self.assertFalse(qm.has_act_static_scaling())

        # Do not set per-token flag.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS
        # Make sure it returns False for per-token.
        self.assertFalse(qm.has_per_token_dynamic_scaling())
        # Make sure it returns True for per-tensor.
        self.assertTrue(qm.has_act_static_scaling())

    def test_weights_only(self):
        # Set weights flags.
        qm = QuantMode.INT8_WEIGHTS
        # Make sure it returns True for weight-only.
        self.assertTrue(qm.is_weight_only())
        # Make sure it returns True for weight-only.
        self.assertTrue(qm.is_int8_weight_only())

        # Set weights flags.
        qm = QuantMode.INT4_WEIGHTS
        # Make sure it returns True for weight-only.
        self.assertTrue(qm.is_weight_only())
        # Make sure it returns True for weight-only.
        self.assertTrue(qm.is_int4_weight_only())

        # Set activations and weights flags.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS
        # Make sure it returns False for weight-only.
        self.assertFalse(qm.is_weight_only())

    def test_int8_kv_cache(self):
        # Set int8 kv cache flags.
        qm = QuantMode.INT8_KV_CACHE
        # Make sure it returns True for kv_cache.
        self.assertTrue(qm.has_int8_kv_cache())
        # Make sure it returns True for any quantization.
        self.assertTrue(qm.has_any_quant())

        # Set weights flags.
        qm = QuantMode.INT8_WEIGHTS
        # Make sure it returns True for any quantization.
        self.assertTrue(qm.has_any_quant())
        # Set int8 KV cache flag.
        qm = qm.set_int8_kv_cache()
        # Make sure it returns True for kv_cache.
        self.assertTrue(qm.has_int8_kv_cache())
        # Make sure it returns True for weight-only.
        self.assertTrue(qm.is_weight_only())
        # Make sure it returns True for weight-only.
        self.assertTrue(qm.is_int8_weight_only())

    def test_failure_quant(self):
        # Expect failure if weights are not quantized, but activations are.
        self.assertRaises(ValueError, lambda: QuantMode.from_description(False, True, False, False))

        # Expect failure if per token and per channel quantization, but weights and activations are not quantized.
        self.assertRaises(ValueError, lambda: QuantMode.from_description(False, False, True, True))


@pytest.mark.parametrize(
    ("sm", "quant_algo", "marlin_available", "supported"),
    [
        # W4A16_NVFP4 dequantizes to the activation dtype before the GEMM, so
        # it never needed FP4 tensor cores and must not be swept in with the
        # W4A4 modes that do.
        (80, QuantAlgo.W4A16_NVFP4, False, True),
        (90, QuantAlgo.W4A16_NVFP4, False, True),
        (100, QuantAlgo.W4A16_NVFP4, False, True),
        (120, QuantAlgo.W4A16_NVFP4, False, True),
        # NVFP4 needs FP4 tensor cores -- unless the caller allowed Marlin,
        # which serves the same checkpoint weight-only on Ada and Hopper.
        (100, QuantAlgo.NVFP4, False, True),
        (103, QuantAlgo.NVFP4, False, True),
        (120, QuantAlgo.NVFP4, False, True),
        (121, QuantAlgo.NVFP4, False, True),
        (89, QuantAlgo.NVFP4, False, False),
        (90, QuantAlgo.NVFP4, False, False),
        (89, QuantAlgo.NVFP4, True, True),
        (90, QuantAlgo.NVFP4, True, True),
        (99, QuantAlgo.NVFP4, True, True),
        (120, QuantAlgo.NVFP4, True, True),
        # Marlin reaches Ada and Hopper only; Ampere stays out of range.
        (80, QuantAlgo.NVFP4, True, False),
        (86, QuantAlgo.NVFP4, True, False),
        (100, QuantAlgo.NVFP4_AWQ, False, True),
        (90, QuantAlgo.NVFP4_AWQ, True, True),
        # NVFP4_ARC keeps its own linear method, which is never swapped for
        # the Marlin one, so the opt-in buys it nothing.
        (90, QuantAlgo.NVFP4_ARC, True, False),
        (100, QuantAlgo.NVFP4_ARC, False, True),
        # The W4A8 modes have no weight-only fallback at all.
        (90, QuantAlgo.W4A8_NVFP4_FP8, True, False),
        (100, QuantAlgo.W4A8_NVFP4_FP8, False, True),
        (121, QuantAlgo.W4A8_NVFP4_FP8, False, True),
        (100, QuantAlgo.W4A8_MXFP4_FP8, False, True),
        (103, QuantAlgo.W4A8_MXFP4_FP8, False, True),
        (90, QuantAlgo.W4A8_MXFP4_FP8, False, False),
        (120, QuantAlgo.W4A8_MXFP4_FP8, False, False),
        (100, QuantAlgo.W4A8_MXFP4_MXFP8, False, True),
        (121, QuantAlgo.W4A8_MXFP4_MXFP8, False, True),
        (90, QuantAlgo.W4A8_MXFP4_MXFP8, False, False),
        # W4A16_MXFP4 is the one FP4 mode that runs on Hopper and nothing newer.
        (90, QuantAlgo.W4A16_MXFP4, False, True),
        (100, QuantAlgo.W4A16_MXFP4, False, False),
        # Nothing to restrict: not an FP4 algorithm, or no algorithm at all.
        (90, QuantAlgo.FP8, False, True),
        (90, QuantAlgo.MXFP8, False, True),
        (90, None, False, True),
    ],
)
def test_is_fp4_supported(sm, quant_algo, marlin_available, supported):
    assert is_fp4_supported(sm, quant_algo, marlin_available) is supported


@pytest.mark.parametrize("sm", [None, -1])
def test_unknown_architecture_is_never_rejected(sm):
    """Failing fast on a guess would break every CPU-side construction.

    ``get_sm_version`` reports -1 with no visible device, which says nothing
    about what the eventual GPU can run.
    """
    assert is_fp4_supported(sm, QuantAlgo.W4A8_MXFP4_FP8)
    assert is_fp4_supported(sm, QuantAlgo.NVFP4)


def test_marlin_only_widens_the_algorithms_it_can_serve():
    marlin = set(NVFP4_MARLIN_SM_VERSIONS)
    assert marlin.issubset(
        set(get_fp4_supported_sm_versions(QuantAlgo.NVFP4, marlin_available=True))
    )
    assert not marlin & set(get_fp4_supported_sm_versions(QuantAlgo.NVFP4, marlin_available=False))
    assert not marlin & set(
        get_fp4_supported_sm_versions(QuantAlgo.W4A8_MXFP4_FP8, marlin_available=True)
    )
    assert get_fp4_supported_sm_versions(QuantAlgo.W4A16_NVFP4) is None
    assert get_fp4_supported_sm_versions(QuantAlgo.FP8) is None


@pytest.mark.parametrize(
    ("sm", "quant_algo", "expected_fragments"),
    [
        # "newer architectures only" was false here: SM120/SM121 are newer
        # than the SM100/SM103 this mode needs.
        (120, QuantAlgo.W4A8_MXFP4_FP8, ["SM120", "SM100, SM103"]),
        # ... and false the other way for the one Hopper-only mode.
        (100, QuantAlgo.W4A16_MXFP4, ["SM100", "SM90"]),
        (90, QuantAlgo.W4A8_NVFP4_FP8, ["SM90", "SM100, SM103, SM120, SM121"]),
    ],
)
def test_error_message_names_the_supported_architectures(sm, quant_algo, expected_fragments):
    message = get_fp4_support_error_message(sm, quant_algo)
    assert quant_algo.name in message
    for fragment in expected_fragments:
        assert fragment in message
    assert "newer architectures" not in message


def test_nvfp4_error_message_points_at_the_marlin_opt_in():
    message = get_fp4_support_error_message(90, QuantAlgo.NVFP4)
    assert "SM89-SM99" in message
    assert "marlin" in message
    # Nothing to suggest once the caller already allowed Marlin.
    assert "marlin" not in get_fp4_support_error_message(80, QuantAlgo.NVFP4, marlin_available=True)


def test_error_message_handles_an_unknown_architecture():
    assert "unknown GPU" in get_fp4_support_error_message(None, QuantAlgo.NVFP4)
    assert "unknown GPU" in get_fp4_support_error_message(-1, QuantAlgo.NVFP4)


if __name__ == "__main__":
    unittest.main()
