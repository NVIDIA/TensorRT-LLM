# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
# rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""FLUX.3 image package configuration and variant resolution."""

import json
import re
from pathlib import Path

from safetensors import safe_open

from tensorrt_llm.llmapi.utils import download_hf_partial

PACKAGE_LAYOUT = (
    "FLUX.3 requires a BFL flux-3-image-klein format_version=1 package root with "
    "config.json and model.safetensors. Select quantized weights with "
    "pipeline_config.variant from variants/<variant>/, or omit it for root weights. "
    "Shared assets use package-relative paths or hf:// repository references."
)


class Flux3Checkpoint:
    def __init__(self, root: str | Path, variant: str | None = None):
        self.root = Path(root)
        if variant is not None and (
            not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", variant)
        ):
            raise ValueError(f"Invalid FLUX.3 variant: {variant!r}")
        self.directory = self.root / "variants" / variant if variant else self.root
        config_path = self.directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing FLUX.3 config: {config_path}")
        try:
            self.config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {config_path}") from exc
        if (
            not isinstance(self.config, dict)
            or self.config.get("format_version") != 1
            or self.config.get("model") != "flux-3-image-klein"
        ):
            raise ValueError(f"Unsupported FLUX.3 package at {self.directory}. {PACKAGE_LAYOUT}")
        try:
            self._validate_config()
        except KeyError as exc:
            raise ValueError(f"Missing FLUX.3 config field: {exc.args[0]}") from exc
        self.weights = self.directory / "model.safetensors"
        if not self.weights.is_file():
            raise FileNotFoundError(f"Missing FLUX.3 weights: {self.weights}")

    def _validate_config(self) -> None:
        """Validate the supported image package configuration."""
        for section in ("dit", "text_encoder", "vae", "sampler"):
            if not isinstance(self.config.get(section), dict):
                raise ValueError(f"FLUX.3 config.{section} must be an object")
        dit = self.config["dit"]
        flow_map = dit.get("flow_map", False)
        if not isinstance(flow_map, bool):
            raise ValueError("FLUX.3 dit.flow_map must be a boolean")
        guidance = self.config["sampler"]["guidance"]
        if (flow_map and guidance is not None) or (
            not flow_map and type(guidance) not in (int, float)
        ):
            raise ValueError("FLUX.3 sampler.guidance must be null exactly for a flow map")
        if self.config["quantization"] not in (None, "fp8r", "nvfp4"):
            raise ValueError("Unsupported FLUX.3 checkpoint quantization")
        if self.config["torch_dtype"] != "bfloat16" or dit["vec_in_dim"] is not None:
            raise ValueError("FLUX.3 Klein requires BF16 activations and no vector embedder")
        for key in ("hidden", "heads", "depth", "depth_single", "theta"):
            value = dit.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"FLUX.3 dit.{key} must be a positive integer")
        if dit["hidden"] % dit["heads"]:
            raise ValueError("FLUX.3 dit.hidden must be divisible by dit.heads")
        axes = dit.get("axes_dim")
        if not isinstance(axes, list) or not all(
            type(x) is int and x > 0 and x % 2 == 0 for x in axes
        ):
            raise ValueError("FLUX.3 dit.axes_dim must contain positive even integers")
        if sum(axes) != dit["hidden"] // dit["heads"]:
            raise ValueError("FLUX.3 dit.axes_dim must sum to the head dimension")
        for key in ("mlp_ratio", "image_rope_scale"):
            value = dit.get(key)
            if type(value) not in (int, float) or not value > 0:
                raise ValueError(f"FLUX.3 dit.{key} must be positive")
        for section in ("text_encoder", "vae"):
            if self.config[section]["dtype"] not in ("bfloat16", "float16", "float32"):
                raise ValueError(f"Unsupported FLUX.3 {section}.dtype")

    def asset(self, name: str) -> Path:
        if name.startswith("hf://"):
            parts = name.removeprefix("hf://").split("/")
            if len(parts) < 3 or any(
                not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]*", part) for part in parts
            ):
                raise ValueError(f"Invalid FLUX.3 HF asset reference: {name}")
            owner, repo, *asset_parts = parts
            asset = "/".join(asset_parts)
            root = download_hf_partial(f"{owner}/{repo}", allow_patterns=[asset, f"{asset}/**"])
            path = root / asset
            if not path.exists():
                raise FileNotFoundError(f"Missing FLUX.3 asset {name} in {root}")
            return path
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Asset path escapes the FLUX.3 package: {name}")
        path = self.root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def transformer_config(self) -> dict:
        keys = self._validate_weights()
        d = self.config["dit"]
        channels = {k: v for k, v in d["in_channels"].items() if k != "text"}
        if set(channels) != {"image", "image_cond"} or d["out_modes"] != ["image"]:
            raise ValueError(
                "FLUX.3 image checkpoints require image/image_cond inputs and image output"
            )
        config = dict(
            in_channels=channels,
            sequence={f"x_{k}": k for k in channels},
            context_in_dim=d["in_channels"]["text"],
            hidden_size=d["hidden"],
            num_heads=d["heads"],
            depth=d["depth"],
            depth_single_blocks=d["depth_single"],
            depth_late_blocks=0,
            vec_in_dim=d["vec_in_dim"],
            flow_map=d.get("flow_map", False),
            out_modes=d["out_modes"],
            axes_dim=d["axes_dim"],
            theta=d["theta"],
            mlp_ratio=d["mlp_ratio"],
            torch_dtype=self.config["torch_dtype"],
            _name_or_path=str(self.root),
        )
        quantization = self.config["quantization"]
        if quantization:
            algo = {"fp8r": "FP8_PER_CHANNEL_PER_TOKEN", "nvfp4": "NVFP4"}[quantization]
            suffix = ".weight_scale_2" if quantization == "nvfp4" else ".weight_scale"
            excluded = sorted(
                k.removesuffix(".weight")
                for k in keys
                if k.endswith(".weight") and k.removesuffix(".weight") + suffix not in keys
            )
            config["quantization_config"] = {
                "quant_algo": algo,
                "dynamic": False,
                "ignore": excluded,
            }
        return config

    def _validate_weights(self) -> set[str]:
        """Validate tensor metadata and return keys for quantization exclusions."""
        d = self.config["dit"]
        hidden, head_dim = d["hidden"], d["hidden"] // d["heads"]
        mlp = int(hidden * d["mlp_ratio"])
        channels = {k: v for k, v in d["in_channels"].items() if k != "text"}
        expected = {f"emb_in.{name}.weight": [hidden, width] for name, width in channels.items()}
        expected.update(
            {
                "txt_in.weight": [hidden, d["in_channels"]["text"]],
                "time_in.in_layer.weight": [hidden, 256],
                "time_in.out_layer.weight": [hidden, hidden],
                "final_layer.image.linear.weight": [channels["image"], hidden],
                "final_layer.image.adaLN_modulation.1.weight": [2 * hidden, hidden],
            }
        )
        if d.get("flow_map", False):
            expected["ref_time_in.in_layer.weight"] = [hidden, 256]
            expected["ref_time_in.out_layer.weight"] = [hidden, hidden]
        for stage in ("early", "single"):
            for name in (*channels, "txt"):
                expected[f"{stage}_stream_modulations.{name}.lin.weight"] = [3 * hidden, hidden]
        blocks = {
            **{f"content_mode_blocks.{name}": d["depth"] for name in channels},
            "txt_mode_blocks": d["depth"],
            "single_blocks": d["depth_single"],
        }
        for prefix, depth in blocks.items():
            for index in range(depth):
                for name, shape in {
                    "q_proj.weight": [hidden, hidden],
                    "k_proj.weight": [hidden, hidden],
                    "v_proj.weight": [hidden, hidden],
                    "attn_out.weight": [hidden, hidden],
                    "mlp_in.weight": [2 * mlp, hidden],
                    "mlp_out.weight": [hidden, mlp],
                    "norm.query_norm.scale": [head_dim],
                    "norm.key_norm.scale": [head_dim],
                }.items():
                    expected[f"{prefix}.{index}.{name}"] = shape
        with safe_open(self.weights, framework="pt", device="cpu") as weights:
            keys = set(weights.keys())
            quant = self.config["quantization"]
            if quant and not any(k.endswith(".weight_scale") for k in keys):
                raise ValueError("Quantized FLUX.3 config requires quantized weights and scales")
            dtypes = {name: "BF16" for name in expected}
            for name, shape in list(expected.items()):
                if not name.endswith(".weight"):
                    continue
                prefix = name.removesuffix(".weight")
                if prefix + ".weight_scale" not in keys:
                    continue
                rows, cols = shape
                if quant == "fp8r":
                    dtypes[name] = "F8_E4M3"
                    expected[prefix + ".weight_scale"] = [rows]
                    dtypes[prefix + ".weight_scale"] = "F32"
                elif quant == "nvfp4":
                    expected[name] = [rows, cols // 2]
                    dtypes[name] = "U8"
                    expected[prefix + ".weight_scale"] = [rows, cols // 16]
                    dtypes[prefix + ".weight_scale"] = "F8_E4M3"
                    for suffix in (".weight_scale_2", ".input_scale"):
                        expected[prefix + suffix] = []
                        dtypes[prefix + suffix] = "F32"
            if keys != expected.keys():
                raise ValueError(
                    f"FLUX.3 tensor mismatch: missing={sorted(expected.keys() - keys)}, "
                    f"unexpected={sorted(keys - expected.keys())}"
                )
            for name, shape in expected.items():
                value = weights.get_slice(name)
                if value.get_shape() != shape or value.get_dtype() != dtypes[name]:
                    raise ValueError(
                        f"Invalid FLUX.3 tensor {name}: expected {shape} {dtypes[name]}, "
                        f"got {value.get_shape()} {value.get_dtype()}"
                    )
        return keys
