#! /usr/bin/env python3
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
"""
This script applies preprocessing to the DoRA magnitude vector to speed up inference.
DoRA applies columnwise normalization and scaling to the LoRA output.
By applying the normalization to the scaling vector, we can skip calculating the normalization vector at inference time.
"""
import abc
import enum
import json
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch

StateDict = dict[str, torch.Tensor]

PEFT_MODULE_PREFIX = "base_model.model."
PEFT_MODULE_SUFFIXES = [".lora_A.weight", ".lora_B.weight"]
DORA_VECTOR_SUFFIXES = [
    ".lora_magnitude_vector",  # HF peft
    ".weight_m_wdecomp.weight"  # NVLabs
]


class HFWeightsReader(abc.ABC):

    @abc.abstractmethod
    def __init__(self, model_dir: str) -> None:
        ...

    @abc.abstractmethod
    def get_weight(self, weight_name: str) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_all(self) -> StateDict:
        ...


class HFSafeTensorsReader(HFWeightsReader):

    def __init__(self, model_dir: str) -> None:
        self.model_dir = Path(model_dir)

        self._fds = [
            st.safe_open(f, framework="torch")
            for f in self.model_dir.glob("*.safetensors")
        ]

        self._weight_to_fd = {}
        for f in self._fds:
            for k in f.keys():
                self._weight_to_fd[k] = f

    def get_weight(self, weight_name: str) -> torch.Tensor:
        return self._weight_to_fd[weight_name].get_tensor(weight_name)

    def get_all(self) -> StateDict:
        return {k: self.get_weight(k) for k in self._weight_to_fd.keys()}


class HFBinWeightsReader(HFWeightsReader):

    def __init__(self, model_dir: str) -> None:
        self.model_dir = Path(model_dir)

        self._weights = {}

        for f in self.model_dir.glob("*.bin"):
            self._weights.update(
                torch.load(f, weights_only=True, mmap=True, map_location="cpu"))

    def get_weight(self, weight_name: str) -> torch.Tensor:
        weight_name = f"{weight_name}.weight"
        return self._weights[weight_name]

    def get_all(self) -> StateDict:
        return self._weights


class WeightsFormat(enum.Enum):
    BINARY = enum.auto()
    SAFETENSORS = enum.auto()
    UNKNOWN = enum.auto()

    def __str__(self) -> str:
        if self == self.BINARY:
            return "bin"
        elif self == self.SAFETENSORS:
            return "safetensors"
        return "unknown"


def deduce_weights_format(model_dir: str) -> WeightsFormat:
    model_dir_p = Path(model_dir)

    if any(model_dir_p.glob("*.safetensors")):
        return WeightsFormat.SAFETENSORS
    elif any(model_dir_p.glob("*.bin")):
        return WeightsFormat.BINARY
    return WeightsFormat.UNKNOWN


def get_weights_reader(model_dir: str) -> HFWeightsReader:
    model_dir_p = Path(model_dir)

    if not model_dir_p.is_dir():
        raise ValueError(
            f"{model_dir} is not a valid model directory: not found")

    weights_format = deduce_weights_format(model_dir)

    if weights_format == WeightsFormat.SAFETENSORS:
        return HFSafeTensorsReader(model_dir)
    elif weights_format == WeightsFormat.BINARY:
        return HFBinWeightsReader(model_dir)
    else:
        raise ValueError(
            f"{model_dir} does not contain .safetensors or .bin weights")


def normalize_hf_peft_module_name(module_name: str) -> str:
    """
    Remove parts of the module name in the peft adapter to derive the module name of the base model.
    """

    if module_name.startswith(PEFT_MODULE_PREFIX):
        module_name = module_name[len(PEFT_MODULE_PREFIX):]

    for suffix in PEFT_MODULE_SUFFIXES + DORA_VECTOR_SUFFIXES:
        if module_name.endswith(suffix):
            module_name = module_name[:-len(suffix)]

    return module_name


def get_peft_module_names(base_module_name: str) -> tuple[str, ...]:
    """
    Convert the name of a base module to the names of its LoRA A and LoRA B weights.
    """
    return tuple([
        f"{PEFT_MODULE_PREFIX}{base_module_name}{suffix}"
        for suffix in PEFT_MODULE_SUFFIXES
    ])


def get_dora_magnitude_names(base_module_name: str) -> tuple[str, ...]:
    """
    Convert the name of a base module to the potential names of its DoRA magnitude vectors.
    """
    return tuple([
        f"{PEFT_MODULE_PREFIX}{base_module_name}{suffix}"
        for suffix in DORA_VECTOR_SUFFIXES
    ])


def normalize_dora_vector(W: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                          mag: torch.Tensor, scale: float) -> torch.Tensor:
    return mag / torch.linalg.norm(W + scale * B @ A, dim=1).to(W.dtype)


def normalize_dora_scales(lora_sd: StateDict,
                          weights_reader: HFWeightsReader,
                          alpha: float,
                          use_rslora: bool,
                          strip: bool = False) -> StateDict:
    out_sd = {}

    while lora_sd:
        # take some lora weight name
        module_name = next(iter(lora_sd.keys()))
        base_module_name = normalize_hf_peft_module_name(module_name)
        A_name, B_name = get_peft_module_names(base_module_name)
        magnitude_names = get_dora_magnitude_names(base_module_name)

        if module_name not in [A_name, B_name] + list(magnitude_names):
            raise ValueError(f"Encountered unknown weight: {module_name}")

        # get lora weights
        A = lora_sd.pop(A_name)
        B = lora_sd.pop(B_name)
        for name in magnitude_names:
            if name in lora_sd:
                mag_name = name
                mag = lora_sd.pop(mag_name).view(-1)
                break
        else:
            mag_name = ""
            mag = None

        out_sd[A_name] = A.contiguous()
        out_sd[B_name] = B.contiguous()

        if mag is not None and not strip:
            # get base weight and normalize
            W = weights_reader.get_weight(base_module_name + ".weight")

            adapter_size = A.size(0)

            if use_rslora:
                scale = alpha / np.sqrt(adapter_size)
            else:
                scale = alpha / adapter_size

            mag = normalize_dora_vector(W, A, B, mag, scale)
            out_sd[mag_name] = mag.contiguous()

    return out_sd


def save_state_dict(out_file: str, sd: StateDict) -> None:
    out_path = Path(out_file)

    if out_path.suffix == ".safetensors":
        st.save_file(sd, out_path)
    elif out_path.suffix == ".bin":
        torch.save(sd, out_path)
    else:
        raise ValueError(f"Unregornized weights format: {out_path.suffix}")


def normalize_peft_ckpt(model_dir: str,
                        base_model_dir: str,
                        out_dir: str,
                        strip: bool = False) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    weights_reader = get_weights_reader(base_model_dir)
    lora_sd = get_weights_reader(model_dir).get_all()

    adapter_config_path = Path(f"{model_dir}/adapter_config.json")
    with adapter_config_path.open() as f:
        adapter_config = json.load(f)

    alpha = adapter_config["lora_alpha"]
    use_rslora = adapter_config.get("use_rslora", False)

    new_sd = normalize_dora_scales(lora_sd, weights_reader, alpha, use_rslora,
                                   strip)

    with (out_path / "adapter_config.json").open("w") as f:
        json.dump(adapter_config, f)

    weights_format = deduce_weights_format(model_dir)
    save_state_dict(
        (out_path / f"adapter_model.{str(weights_format)}").as_posix(), new_sd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-dir',
        '-o',
        type=Path,
        help='path to output adapter weights with normalized DoRA vectors',
        required=True)
    parser.add_argument('--in-dir',
                        '-i',
                        type=Path,
                        help='path to input lora checkpoint file',
                        required=True)
    parser.add_argument("--base-model",
                        "-b",
                        help="Path to base model",
                        required=True)
    parser.add_argument("--strip",
                        action="store_true",
                        help="remove DoRA vectors entirely")

    args = parser.parse_args()

    normalize_peft_ckpt(args.in_dir, args.base_model, args.out_dir, args.strip)
