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
from typing import Optional

from strenum import StrEnum

from .._utils import BaseEnumMeta


class QuantAlgo(StrEnum, metaclass=BaseEnumMeta):
    W8A16 = auto()
    W4A16 = auto()
    W4A16_AWQ = auto()
    W4A8_AWQ = auto()
    W8A16_GPTQ = auto()
    W4A16_GPTQ = auto()
    W8A8_SQ_PER_CHANNEL = auto()
    W8A8_SQ_PER_TENSOR_PLUGIN = auto()
    W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN = auto()
    W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN = auto()
    W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN = auto()
    W4A8_QSERVE_PER_GROUP = auto()
    W4A8_QSERVE_PER_CHANNEL = auto()
    FP8 = auto()
    FP8_PER_CHANNEL_PER_TOKEN = auto()
    FP8_BLOCK_SCALES = auto()
    INT8 = auto()
    MIXED_PRECISION = auto()
    NVFP4 = auto()
    W4A8_NVFP4_FP8 = auto()
    W4A8_MXFP4_FP8 = auto()
    W4A8_MXFP4_MXFP8 = auto()
    W4A16_MXFP4 = auto()
    NO_QUANT = auto()


QUANT_ALGO_LIST = list(set(QuantAlgo) - {QuantAlgo.INT8})
KV_CACHE_QUANT_ALGO_LIST = [QuantAlgo.FP8, QuantAlgo.INT8, QuantAlgo.NVFP4]
W8A8_SQ_PLUGIN_LIST = [
    QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN,
    QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
    QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN,
    QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN,
]
MODELOPT_FLOW_QUANTIZATIONS = {
    QuantAlgo.W4A16_AWQ, QuantAlgo.FP8, QuantAlgo.W8A8_SQ_PER_CHANNEL,
    QuantAlgo.W4A8_AWQ
}


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
    # FP8 rowwise
    FP8_ROWWISE = auto()
    # FP8 block scales for Deepseek
    FP8_1x128_128x128 = auto()
    # W4A8 qserve
    W4A8_QSERVE = auto()
    # FP4
    NVFP4 = auto()
    NVFP4_KV_CACHE = auto()
    # W4A8 NVFP4
    W4A8_NVFP4_FP8 = auto()
    # W4A8 MXFP4
    W4A8_MXFP4_FP8 = auto()
    W4A8_MXFP4_MXFP8 = auto()
    W4A16_MXFP4 = auto()

    # The smallest power-of-two that is not used by a flag. Do not call auto() after that line.
    COUNT = auto()

    # Bitmask to detect if weights, activations or both are quantized.
    WEIGHTS_AND_ACTIVATIONS = INT4_WEIGHTS | INT8_WEIGHTS | ACTIVATIONS
    # The mask of all valid flags.
    VALID_FLAGS = COUNT - 1

    def __deepcopy__(self, memo):
        return self

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

    def is_int8_weight_only_per_group(self):
        return self.is_int8_weight_only() and self._any(self.PER_GROUP)

    # TODO: Using the current flags cannot distinguish between w4aFP8 AWQ and w4a8 QServe.
    def is_qserve_w4a8(self):
        return self._any(self.W4A8_QSERVE)

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

    def has_fp8_block_scales(self):
        return self._any(self.FP8_1x128_128x128)

    def has_act_static_scaling(self):
        return not self.has_per_token_dynamic_scaling(
        ) and not self.has_fp8_rowwise()

    def has_per_channel_scaling(self):
        return self._any(self.PER_CHANNEL)

    def has_per_group_scaling(self):
        return self._any(self.PER_GROUP)

    def has_int8_kv_cache(self):
        return self._any(self.INT8_KV_CACHE)

    def has_fp8_kv_cache(self):
        return self._any(self.FP8_KV_CACHE)

    def has_fp4_kv_cache(self):
        return self._any(self.NVFP4_KV_CACHE)

    def has_kv_cache_quant(self):
        return (self.has_int8_kv_cache() or self.has_fp8_kv_cache()
                or self.has_fp4_kv_cache())

    def has_fp8_qdq(self):
        return self._any(self.FP8_QDQ)

    def has_fp8_rowwise(self):
        return self._any(self.FP8_ROWWISE)

    def has_nvfp4(self):
        return self._any(self.NVFP4)

    def has_w4a8_nvfp4_fp8(self):
        return self._any(self.W4A8_NVFP4_FP8)

    def has_w4a8_mxfp4_fp8(self):
        return self._any(self.W4A8_MXFP4_FP8)

    def has_w4a8_mxfp4_mxfp8(self):
        return self._any(self.W4A8_MXFP4_MXFP8)

    def has_w4a16_mxfp4(self):
        return self._any(self.W4A16_MXFP4)

    def has_mxfp4(self):
        return self._any(self.W4A8_MXFP4_FP8 | self.W4A8_MXFP4_MXFP8
                         | self.W4A16_MXFP4)

    def has_weight_quant(self):
        return self._any(self.INT4_WEIGHTS | self.INT8_WEIGHTS)

    def has_any_quant(self, exclude_kv_cache: bool = False):
        has_quant = self._any(self.INT4_WEIGHTS
                              | self.INT8_WEIGHTS
                              | self.ACTIVATIONS
                              | self.FP8_QDQ | self.FP8_ROWWISE
                              | self.W4A8_QSERVE
                              | self.FP8_1x128_128x128
                              | self.NVFP4
                              | self.W4A8_NVFP4_FP8
                              | self.W4A8_MXFP4_FP8
                              | self.W4A16_MXFP4
                              | self.W4A8_MXFP4_MXFP8)
        if exclude_kv_cache:
            return has_quant

        return has_quant | self._any(self.INT8_KV_CACHE | self.FP8_KV_CACHE
                                     | self.NVFP4_KV_CACHE)

    def set_int8_kv_cache(self):
        return self | self.INT8_KV_CACHE

    def set_fp8_kv_cache(self):
        return self | self.FP8_KV_CACHE

    def set_fp4_kv_cache(self):
        return self | self.NVFP4_KV_CACHE

    def set_fp8_qdq(self):
        return self | self.FP8_QDQ

    def set_fp8_rowwise(self):
        return self | self.FP8_ROWWISE | self.PER_TOKEN | self.PER_CHANNEL

    @staticmethod
    def from_description(quantize_weights=False,
                         quantize_activations=False,
                         per_token=False,
                         per_channel=False,
                         per_group=False,
                         use_int4_weights=False,
                         use_int8_kv_cache=False,
                         use_fp8_kv_cache=False,
                         use_fp8_qdq=False,
                         use_fp8_block_scales=False,
                         use_fp8_rowwise=False,
                         use_nvfp4=False,
                         use_w4a8_nvfp4_fp8=False,
                         use_w4a8_qserve=False,
                         use_w4a8_mxfp4_fp8=False,
                         use_w4a8_mxfp4_mxfp8=False,
                         use_w4a16_mxfp4=False):

        def raise_error():
            raise ValueError(f"Unsupported combination of QuantMode args: "
                             f"{quantize_weights=}, "
                             f"{quantize_activations=}, "
                             f"{per_token=}, "
                             f"{per_channel=}, "
                             f"{per_group=}, "
                             f"{use_int4_weights=}, "
                             f"{use_int8_kv_cache=}, "
                             f"{use_fp8_kv_cache=}, "
                             f"{use_fp8_qdq=}, "
                             f"{use_fp8_block_scales=}, "
                             f"{use_fp8_rowwise=}, "
                             f"{use_nvfp4=}, "
                             f"{use_w4a8_qserve=}, "
                             f"{use_w4a8_mxfp4_fp8=}, "
                             f"{use_w4a8_mxfp4_mxfp8=}, "
                             f"{use_w4a16_mxfp4=}")

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

        if use_fp8_rowwise:
            mode = mode | QuantMode.FP8_ROWWISE | QuantMode.PER_TOKEN | QuantMode.PER_CHANNEL

        if use_fp8_block_scales:
            mode = mode | QuantMode.FP8_1x128_128x128

        if use_nvfp4:
            mode = mode | QuantMode.NVFP4

        if use_w4a8_nvfp4_fp8:
            mode = mode | QuantMode.W4A8_NVFP4_FP8

        # W4A8 QServe
        if use_w4a8_qserve:
            mode = mode | QuantMode.W4A8_QSERVE

        if use_w4a8_mxfp4_fp8:
            mode = mode | QuantMode.W4A8_MXFP4_FP8

        if use_w4a8_mxfp4_mxfp8:
            mode = mode | QuantMode.W4A8_MXFP4_MXFP8

        if use_w4a16_mxfp4:
            mode = mode | QuantMode.W4A16_MXFP4

        return mode

    @staticmethod
    def use_smooth_quant(per_token=False, per_channel=False):
        return QuantMode.from_description(True, True, per_token, per_channel)

    @staticmethod
    def use_qserve(per_group):
        return QuantMode.from_description(quantize_weights=True,
                                          quantize_activations=True,
                                          per_group=per_group,
                                          use_int4_weights=True,
                                          use_w4a8_qserve=True)

    @staticmethod
    def use_weight_only(use_int4_weights=False, per_group=False):
        return QuantMode.from_description(quantize_weights=True,
                                          quantize_activations=False,
                                          per_token=False,
                                          per_channel=False,
                                          per_group=per_group,
                                          use_int4_weights=use_int4_weights)

    @staticmethod
    def from_quant_algo(
        quant_algo: Optional[QuantAlgo] = None,
        kv_cache_quant_algo: Optional[QuantAlgo] = None,
    ) -> "QuantMode":
        assert quant_algo is None or quant_algo in QUANT_ALGO_LIST
        assert kv_cache_quant_algo is None or kv_cache_quant_algo in KV_CACHE_QUANT_ALGO_LIST
        if quant_algo == QuantAlgo.W8A16:
            quant_mode = QuantMode.use_weight_only(use_int4_weights=False)
        elif quant_algo == QuantAlgo.W4A16:
            quant_mode = QuantMode.use_weight_only(use_int4_weights=True)
        elif quant_algo == QuantAlgo.W4A16_AWQ:
            quant_mode = QuantMode.use_weight_only(use_int4_weights=True,
                                                   per_group=True)
        elif quant_algo == QuantAlgo.W4A8_AWQ:
            quant_mode = QuantMode.use_weight_only(use_int4_weights=True,
                                                   per_group=True)
        elif quant_algo == QuantAlgo.W4A16_GPTQ:
            quant_mode = QuantMode.use_weight_only(use_int4_weights=True,
                                                   per_group=True)
        elif quant_algo == QuantAlgo.W8A16_GPTQ:
            quant_mode = QuantMode.use_weight_only(use_int4_weights=False,
                                                   per_group=True)
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_CHANNEL:
            quant_mode = QuantMode.use_smooth_quant(per_token=False,
                                                    per_channel=True)
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN:
            quant_mode = QuantMode.use_smooth_quant(per_token=False,
                                                    per_channel=False)
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN:
            quant_mode = QuantMode.use_smooth_quant(per_token=True,
                                                    per_channel=True)
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN:
            quant_mode = QuantMode.use_smooth_quant(per_token=False,
                                                    per_channel=True)
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN:
            quant_mode = QuantMode.use_smooth_quant(per_token=True,
                                                    per_channel=False)
        elif quant_algo == QuantAlgo.W4A8_QSERVE_PER_GROUP:
            quant_mode = QuantMode.use_qserve(per_group=True)
        elif quant_algo == QuantAlgo.W4A8_QSERVE_PER_CHANNEL:
            quant_mode = QuantMode.use_qserve(per_group=False)
        elif quant_algo == QuantAlgo.FP8:
            quant_mode = QuantMode.from_description(use_fp8_qdq=True)
        elif quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN:
            quant_mode = QuantMode.from_description(use_fp8_rowwise=True)
        elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            quant_mode = QuantMode.from_description(use_fp8_block_scales=True)
        elif quant_algo == QuantAlgo.NVFP4:
            quant_mode = QuantMode.from_description(use_nvfp4=True)
        elif quant_algo == QuantAlgo.W4A8_NVFP4_FP8:
            quant_mode = QuantMode.from_description(use_w4a8_nvfp4_fp8=True)
        elif quant_algo == QuantAlgo.W4A8_MXFP4_FP8:
            quant_mode = QuantMode.from_description(use_w4a8_mxfp4_fp8=True)
        elif quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
            quant_mode = QuantMode.from_description(use_w4a8_mxfp4_mxfp8=True)
        elif quant_algo == QuantAlgo.W4A16_MXFP4:
            quant_mode = QuantMode.from_description(use_w4a16_mxfp4=True)
        else:
            quant_mode = QuantMode(0)

        if kv_cache_quant_algo == QuantAlgo.INT8:
            quant_mode = quant_mode.set_int8_kv_cache()
        elif kv_cache_quant_algo == QuantAlgo.FP8:
            quant_mode = quant_mode.set_fp8_kv_cache()
        elif kv_cache_quant_algo == QuantAlgo.NVFP4:
            quant_mode = quant_mode.set_fp4_kv_cache()

        return quant_mode

    def to_dict(self):
        return {
            'use_smooth_quant':
            self.has_act_and_weight_quant(),
            'per_channel':
            self.has_per_channel_scaling(),
            'per_token':
            self.has_per_token_dynamic_scaling(),
            'per_group':
            self.has_per_group_scaling(),
            'int8_kv_cache':
            self.has_int8_kv_cache(),
            'enable_fp8':
            self.has_fp8_qdq(),
            'enable_fp8_rowwise':
            self.has_fp8_rowwise(),
            'enable_fp8_block_scales':
            self.has_fp8_block_scales(),
            'enable_nvfp4':
            self.has_nvfp4(),
            'enable_w4a8_nvfp4_fp8':
            self.has_w4a8_nvfp4_fp8(),
            'enable_w4a8_mxfp4_fp8':
            self.has_w4a8_mxfp4_fp8(),
            'enable_w4a8_mxfp4_mxfp8':
            self.has_w4a8_mxfp4_mxfp8(),
            'enable_w4a16_mxfp4':
            self.has_w4a16_mxfp4(),
            'fp8_kv_cache':
            self.has_fp8_kv_cache(),
            'use_weight_only':
            self.is_weight_only(),
            'weight_only_precision':
            'int8' if self.is_int8_weight_only() else 'int4',
        }


class GroupwiseQuantAlgo:
    BIAS = 1
    ZERO = 2
    PRE_QUANT_SCALE = 4
    W4A8_ALPHA = 8
    INT8_WEIGHT = 16
