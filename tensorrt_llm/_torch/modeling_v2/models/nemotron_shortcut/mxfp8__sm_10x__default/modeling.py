# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text generation with block-scaled E4M3 projections and expert banks."""

import torch

from ..decoder import ShortcutDecoder, ShortcutForCausalLM
from ..manifest import build_manifest
from ..moe import grouped_quantized_experts
from .ops import Mxfp8Projection
from .weights import load


class Mxfp8Decoder(ShortcutDecoder):
    def __init__(self, model_config):
        self.quant_config = model_config.quant_config
        self.weight_manifest = None
        self.mxfp8_shapes = {}
        super().__init__(model_config)

    def _declare_matrix(self, weights, name, shape, P):
        if self.weight_manifest is None:
            self.weight_manifest = build_manifest(self)
        sources = {key for key, *_ in self.weight_manifest[name]}
        quantized = {
            key
            for key in sources
            if not self.quant_config.is_module_excluded_from_quantization(
                key.removesuffix(".weight")
            )
        }
        if not quantized:
            if len(shape) == 3 and not name.startswith("mtp_"):
                raise ValueError(f"The MXFP8 decoder requires quantized routed experts: {name}")
            super()._declare_matrix(weights, name, shape, P)
            return
        if len(sources) != 1 or name.endswith("_router"):
            raise ValueError(f"Unsupported MXFP8 text projection: {sorted(quantized)}")
        *bank, rows, cols = shape
        if cols % 32:
            raise ValueError(f"{name}: the local K dimension must contain complete MXFP8 blocks")
        self.mxfp8_shapes[name] = (rows, cols)
        padded_n, padded_k = (rows + 127) // 128 * 128, (cols + 127) // 128 * 128
        weights[name] = P(*bank, padded_n, padded_k, dtype=torch.float8_e4m3fn)
        weights[name + "_scale"] = P(*bank, padded_n * (padded_k // 32), dtype=torch.uint8)

    def _matrix_view(self, name):
        if name in self.mxfp8_shapes:
            return Mxfp8Projection(
                self.w[name], self.w[name + "_scale"], self.mxfp8_shapes[name][0]
            )
        return super()._matrix_view(name)

    @staticmethod
    def _project(x, weight, *, out_dtype=None):
        if isinstance(weight, Mxfp8Projection):
            return weight.apply(x)
        return ShortcutDecoder._project(x, weight, out_dtype=out_dtype)

    _grouped_experts = staticmethod(grouped_quantized_experts)


class ShortcutMxfp8ForCausalLM(ShortcutForCausalLM):
    decoder_cls = Mxfp8Decoder
    weight_loader = staticmethod(load)
