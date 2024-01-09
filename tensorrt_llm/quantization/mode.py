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
from enum import IntFlag, auto


class QuantMode(IntFlag):
    # [WARNING] KEEP BELOW DEFINITION IN SYNC WITH cpp/tensorrt_llm/common/quantization.h

    # The weights are quantized to 4 bits.
    INT4_WEIGHTS = auto()
    # The weights are quantized to 8 bits.
    INT8_WEIGHTS = auto()
    # The activations are quantized.
    ACTIVATIONS = auto()
    # The method uses one scaling factor per channel. It's pre-computed (static) from the weights.
    PER_CHANNEL = auto()
    # The method uses one scaling factor per token. It's computed on-the-fly.
    PER_TOKEN = auto()
    # The method uses one scaling factor per group. It's pre-computed (static) from the weights.
    PER_GROUP = auto()
    # The KV cache is quantized in INT8.
    INT8_KV_CACHE = auto()
    # The KV cache is quantized in FP8.
    FP8_KV_CACHE = auto()
    # FP8 QDQ
    FP8_QDQ = auto()

    # The smallest power-of-two that is not used by a flag. Do not call auto() after that line.
    COUNT = auto()

    # Bitmask to detect if weights, activations or both are quantized.
    WEIGHTS_AND_ACTIVATIONS = INT4_WEIGHTS | INT8_WEIGHTS | ACTIVATIONS
    # The mask of all valid flags.
    VALID_FLAGS = COUNT - 1

    # All the bits set? You can restrict the test to the bits indicated by "mask".
    def _all(self, bits, mask=VALID_FLAGS):
        return (self & mask) == bits

    # Is one of the bits of the mask set?
    def _any(self, bits):
        return (self & bits) != 0

    def is_int8_weight_only(self):
        return self._all(self.INT8_WEIGHTS, self.WEIGHTS_AND_ACTIVATIONS)

    def is_int4_weight_only(self):
        return self._all(self.INT4_WEIGHTS, self.WEIGHTS_AND_ACTIVATIONS)

    def is_weight_only(self):
        return self.is_int4_weight_only() or self.is_int8_weight_only()

    def is_int4_weight_only_per_group(self):
        return self.is_int4_weight_only() and self._any(self.PER_GROUP)

    def has_act_and_weight_quant(self):
        return self._all(self.INT8_WEIGHTS | self.ACTIVATIONS,
                         self.WEIGHTS_AND_ACTIVATIONS)

    def has_act_or_weight_quant(self):
        return self._any(self.INT4_WEIGHTS | self.INT8_WEIGHTS
                         | self.ACTIVATIONS)

    def has_per_token_dynamic_scaling(self):
        return self._any(self.PER_TOKEN)

    def has_act_static_scaling(self):
        return not self.has_per_token_dynamic_scaling()

    def has_per_channel_scaling(self):
        return self._any(self.PER_CHANNEL)

    def has_per_group_scaling(self):
        return self._any(self.PER_GROUP)

    def has_int8_kv_cache(self):
        return self._any(self.INT8_KV_CACHE)

    def has_fp8_kv_cache(self):
        return self._any(self.FP8_KV_CACHE)

    def has_kv_cache_quant(self):
        return self.has_int8_kv_cache() or self.has_fp8_kv_cache()

    def has_fp8_qdq(self):
        return self._any(self.FP8_QDQ)

    def has_any_quant(self):
        return self._any(self.INT4_WEIGHTS | self.INT8_WEIGHTS
                         | self.ACTIVATIONS
                         | self.INT8_KV_CACHE | self.FP8_KV_CACHE
                         | self.FP8_QDQ)

    def set_int8_kv_cache(self):
        return self | self.INT8_KV_CACHE

    def set_fp8_kv_cache(self):
        return self | self.FP8_KV_CACHE

    def set_fp8_qdq(self):
        return self | self.FP8_QDQ

    @staticmethod
    def from_description(quantize_weights=False,
                         quantize_activations=False,
                         per_token=False,
                         per_channel=False,
                         per_group=False,
                         use_int4_weights=False,
                         use_int8_kv_cache=False,
                         use_fp8_kv_cache=False,
                         use_fp8_qdq=False):

        def raise_error():
            raise ValueError(f"Unsupported combination of QuantMode args: "
                             f"{quantize_weights=}, "
                             f"{quantize_activations=}, "
                             f"{per_token=}, "
                             f"{per_channel=}, "
                             f"{per_group=}, "
                             f"{use_int4_weights=}"
                             f"{use_int8_kv_cache=}"
                             f"{use_fp8_kv_cache=}"
                             f"{use_fp8_qdq=}")

        # We must quantize weights when we quantize activations.
        if quantize_activations and not quantize_weights:
            raise_error()

        # If we set per_token or per_channel, we must quantize both weights and activations.
        if (per_token or per_channel) and not (quantize_weights
                                               and quantize_activations):
            raise_error()

        mode = QuantMode(0)

        # Do we quantize the weights - if so, do we use INT4 or INT8?
        if quantize_weights and use_int4_weights:
            mode = mode | QuantMode.INT4_WEIGHTS
        elif quantize_weights:
            mode = mode | QuantMode.INT8_WEIGHTS

        # Do we quantize the activations?
        if quantize_activations:
            mode = mode | QuantMode.ACTIVATIONS

        # Per-channel/per-token/per-group additional flags.
        if per_channel:
            mode = mode | QuantMode.PER_CHANNEL
        if per_token:
            mode = mode | QuantMode.PER_TOKEN
        if per_group:
            mode = mode | QuantMode.PER_GROUP

        # Int8 KV cache
        if use_int8_kv_cache:
            mode = mode | QuantMode.INT8_KV_CACHE

        # FP8 KV cache
        if use_fp8_kv_cache:
            mode = mode | QuantMode.FP8_KV_CACHE

        if use_fp8_qdq:
            mode = mode | QuantMode.FP8_QDQ

        return mode

    @staticmethod
    def use_smooth_quant(per_token=False, per_channel=False):
        return QuantMode.from_description(True, True, per_token, per_channel)

    @staticmethod
    def use_weight_only(use_int4_weights=False):
        return QuantMode.from_description(quantize_weights=True,
                                          quantize_activations=False,
                                          per_token=False,
                                          per_channel=False,
                                          per_group=False,
                                          use_int4_weights=use_int4_weights)
