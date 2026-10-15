# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MXFP4, SM103, attention TP16 and MoE TP4 x EP4."""

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..modeling import K3TargetSpec, ModelingV2KimiK3
from ..modeling import _check_construction as _check_target_construction

_TARGET = K3TargetSpec(sm=(10, 3), quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8, moe_tp=4, moe_ep=4)


def _check_construction(model_config: ModelConfig) -> None:
    _check_target_construction(model_config, _TARGET)


class ModelingV2KimiK3Mxfp4Sm103Tp16Moetp4ep4(ModelingV2KimiK3):
    """Concrete deployment of the shared Kimi K3 text model."""

    target_spec = _TARGET
