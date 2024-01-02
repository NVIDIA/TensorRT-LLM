# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm.quantization import QuantMode


class TestQuantMode(unittest.TestCase):

    def test_all(self):
        # Set activations and weights flags.
        qm = QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS

        # Make sure _all returns True when asked for both ACTIVATIONS and INT8_WEIGHTS.
        self.assertTrue(qm._all(QuantMode.ACTIVATIONS | QuantMode.INT8_WEIGHTS))
        # Make sure _all returns False when asked only for ACTIVATIONS.
        self.assertFalse(qm._all(QuantMode.ACTIVATIONS))
        # Make sure _all returns True when asked only for ACTIVATIONS if limited to ACTIVATIONS flag.
        self.assertTrue(
            qm._all(QuantMode.ACTIVATIONS, mask=QuantMode.ACTIVATIONS))

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
        self.assertEqual(QuantMode.COUNT.value, 1 << 9)

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
        qm = QuantMode.use_weight_only(True)
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
        self.assertRaises(
            ValueError,
            lambda: QuantMode.from_description(False, True, False, False))

        # Expect failure if per token and per channel quantization, but weights and activations are not quantized.
        self.assertRaises(
            ValueError,
            lambda: QuantMode.from_description(False, False, True, True))


if __name__ == '__main__':
    unittest.main()
