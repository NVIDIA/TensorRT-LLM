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
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import torch
from torch.utils.data import DataLoader

try:
    import ammo.torch.quantization as atq
    from ammo.torch.export import export_model_config
except ImportError:
    raise ImportError("AMMO toolkit is not installed. Please install it first.")

from ...logger import logger


def _register_falcon_linears(model):
    """Register Falcon linear modules as Quantiation.

    As falcon models could use remote code, which will be loaded dynamically,
    to build their model. Therefore, we need to register the linear on the fly
    before quantization.

    """
    if type(model).__name__ in ["RWForCausalLM", "FalconForCausalLM"]:
        from ammo.torch.quantization import tensor_quant
        from ammo.torch.quantization.nn.modules.quant_module import \
            QuantLinearConvBase

        linear_type = type(model.transformer.h[0].self_attention.dense)

        class QuantFalconLinearRW1B(linear_type,
                                    QuantLinearConvBase):  # type: ignore
            default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

        atq.module_mapping.QUANT_MODULE_MAPPING[
            linear_type] = QuantFalconLinearRW1B.convert


def _quantize_model(model: torch.nn.Module,
                    qformat: Literal['fp8', 'int8_sq', 'int4_awq'],
                    calib_dataloader: DataLoader,
                    quant_cfg_dict: Optional[Dict] = None) -> torch.nn.Module:
    assert qformat in ['fp8', 'int8_sq', 'int4_awq'], \
        f'Got unsupported AMMO quantization format, {qformat} '
    if qformat == "fp8":
        quant_cfg = atq.FP8_DEFAULT_CFG
    elif qformat == "int8_sq":
        quant_cfg = atq.INT8_SMOOTHQUANT_CFG
    elif qformat == "int4_awq":
        quant_cfg = atq.INT4_AWQ_CFG
    else:
        raise ValueError(f"Unsupported quantization format: {qformat}")

    if quant_cfg_dict:
        for name, cfg in quant_cfg_dict.items():
            quant_cfg['quant_cfg'][name] = cfg

    def calibrate_loop():
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            logger.debug(f"Calibrating batch {idx}")
            model(data)

    _register_falcon_linears(model)

    logger.debug("Starting quantization...")
    print(quant_cfg)
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    logger.debug("Quantization done")
    return model


def quantize_and_export(
        model: torch.nn.Module,
        qformat: Literal['fp8', 'int8_sq', 'int4_awq'],
        calib_dataloader: DataLoader,
        export_path: Optional[Union[str, Path]] = None,
        tensor_parallel_size: int = 1,
        quant_cfg_dict: Optional[Dict] = None) -> torch.nn.Module:

    model_cls_name = type(model).__name__
    model_lookup = {
        ("llama", "mistral"): "llama",
        ("gptj", ): "gptj",
        ("falcon", "rw"): "falcon",
        ("baichuan", ): "baichuan",
        ("mpt", ): "mpt",
        ("gpt2", ): "gpt2",
        ("chatglm", ): "chatglm",
        ("qwen", ): "qwen",
    }
    for templates, model_type_target in model_lookup.items():
        if any(t in model_cls_name.lower() for t in templates):
            model_type = model_type_target
            break
    else:
        raise NotImplementedError(
            f"Deploying quantized model {model_cls_name} is not supported")

    model = _quantize_model(model,
                            qformat=qformat,
                            calib_dataloader=calib_dataloader,
                            quant_cfg_dict=quant_cfg_dict)

    if export_path:
        with torch.inference_mode():
            if qformat == "int4_awq" and model_type == "qwen" or \
                model_type == "chatglm":
                torch.save(model.state_dict(), export_path)
            else:
                export_model_config(
                    model,
                    model_type,
                    torch.float16,
                    export_dir=export_path,
                    inference_tensor_parallel=tensor_parallel_size,
                )
        logger.info(f"Quantized model exported to :{export_path}")
    return model
