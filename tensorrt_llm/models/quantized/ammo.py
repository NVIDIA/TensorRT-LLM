# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def _quantize_model(model: torch.nn.Module,
                    qformat: Literal['fp8', 'int8_sq', 'int4_awq'],
                    calib_dataloader: DataLoader,
                    quant_cfg_dict: Optional[Dict] = None) -> torch.nn.Module:
    assert qformat in ['fp8', 'int8_sq', 'int4_awq'], \
        f'Got unsupported AMMO quantization format, {qformat} '
    if qformat == "fp8":
        quant_cfg = atq.FP8_DEFAULT_CFG
        if quant_cfg_dict:
            for name, cfg in quant_cfg_dict.items():
                quant_cfg['quant_cfg'][name] = cfg
    elif qformat == "int8_sq":
        quant_cfg = atq.INT8_SMOOTHQUANT_CFG
    elif qformat == "int4_awq":
        quant_cfg = atq.INT4_AWQ_CFG
    else:
        raise ValueError(f"Unsupported quantization format: {qformat}")

    def calibrate_loop():
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            logger.debug(f"Calibrating batch {idx}")
            model(data)

    logger.debug("Starting quantization...")
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    logger.debug("Quantization done")
    return model


def quantize_and_export(model: torch.nn.Module,
                        qformat: Literal['fp8', 'int8_sq', 'int4_awq'],
                        calib_dataloader: DataLoader,
                        export_path: Optional[Union[str, Path]] = None,
                        tensor_parallel_size: int = 1) -> torch.nn.Module:

    model_cls_name = type(model).__name__
    if "Llama" in model_cls_name:
        model_type = "llama"
    elif "GPTJ" in model_cls_name:
        model_type = "gptj"
    elif "GPT2" in model_cls_name:
        model_type = "gpt2"
    elif "Falcon" in model_cls_name or "RW" in model_cls_name:
        model_type = "falcon"
    else:
        raise NotImplementedError(
            f"Deploying quantized model {model_cls_name} is not supported")

    model = _quantize_model(model,
                            qformat=qformat,
                            calib_dataloader=calib_dataloader)

    if export_path:
        with torch.inference_mode():
            if qformat == "int4_awq":
                torch.save(model.state_dict(), export_path)
            else:
                export_model_config(
                    model,
                    model_type,
                    torch.float16,
                    quantization=qformat,
                    export_dir=export_path,
                    inference_tensor_parallel=tensor_parallel_size,
                )
        logger.info(f"Quantized model exported to :{export_path}")
    return model
