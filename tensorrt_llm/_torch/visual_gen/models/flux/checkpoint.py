# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native FLUX checkpoint discovery and supporting-component resolution."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open

from tensorrt_llm._torch.visual_gen.checkpoint_adapter import ResolvedCheckpoint
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_bundle import prepare_comfy_bundle
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_quant import inspect_comfy_quantization
from tensorrt_llm.llmapi.utils import download_hf_partial
from tensorrt_llm.logger import logger


@dataclass(frozen=True)
class NativeFluxCheckpoint:
    path: Path
    version: Literal[1, 2]
    shapes: dict[str, tuple[int, ...]]

    def resolve(self) -> ResolvedCheckpoint:
        base_id = f"black-forest-labs/FLUX.{self.version}-dev"
        logger.info(
            f"Detected FLUX.{self.version}: Comfy NVFP4 at {self.path}. "
            "Packed weights will be converted internally during loading."
        )
        patterns = ["model_index.json", "transformer/config.json"]
        base = download_hf_partial(base_id, allow_patterns=patterns)
        config = json.loads((base / "transformer" / "config.json").read_text())
        self.validate_config(config, base_id)
        components = ["text_encoder", "tokenizer", "vae", "scheduler"]
        components += ["text_encoder_2", "tokenizer_2"] if self.version == 1 else ["processor"]
        logger.info(
            f"Resolving FLUX supporting components from {base_id} "
            "(Hugging Face cache/download; original transformer weights are excluded)."
        )
        base = download_hf_partial(
            base_id,
            allow_patterns=patterns + [f"{name}/*" for name in components],
            revision=base.name,
        )
        stat = self.path.stat()
        identity = f"{self.path}:{stat.st_size}:{stat.st_mtime_ns}:{base}:transformer"
        digest = hashlib.sha256(identity.encode()).hexdigest()
        cache = Path(HF_HUB_CACHE).parent / "tensorrt_llm" / "visual_gen"
        cache.mkdir(parents=True, exist_ok=True)
        output = cache / digest
        with FileLock(str(cache / (digest + ".lock"))):
            if not output.exists():
                prepare_comfy_bundle(base, self.path, output)
        logger.info(f"FLUX supporting components: {base}; selected transformer: {self.path}")
        return ResolvedCheckpoint(str(output))

    def validate_config(self, config: dict[str, Any], base_id: str) -> None:
        """Reject incompatible variants before downloading ancillary model weights."""
        hidden = config["num_attention_heads"] * config["attention_head_dim"]
        expected = {
            "img_in.weight": (hidden, config["in_channels"]),
            "txt_in.weight": (hidden, config["joint_attention_dim"]),
            "final_layer.linear.weight": (
                (config.get("out_channels") or config["in_channels"])
                * config.get("patch_size", 1) ** 2,
                hidden,
            ),
        }
        if self.version == 1:
            expected["vector_in.in_layer.weight"] = (hidden, config["pooled_projection_dim"])
        else:
            expected["double_stream_modulation_img.lin.weight"] = (6 * hidden, hidden)
            expected["double_stream_modulation_txt.lin.weight"] = (6 * hidden, hidden)
            expected["single_stream_modulation.lin.weight"] = (3 * hidden, hidden)
        mismatches = {
            key: (self.shapes.get(key), shape)
            for key, shape in expected.items()
            if self.shapes.get(key) != shape
        }
        for prefix, count_key in (
            ("double_blocks", "num_layers"),
            ("single_blocks", "num_single_layers"),
        ):
            indices = {key.split(".")[1] for key in self.shapes if key.startswith(prefix + ".")}
            if indices != {str(i) for i in range(config[count_key])}:
                raise ValueError(f"Native FLUX block indices do not match {base_id}: {prefix}")
        architecture = "FluxTransformer2DModel" if self.version == 1 else "Flux2Transformer2DModel"
        if (
            mismatches
            or config.get("_class_name") != architecture
            or not config.get("guidance_embeds")
        ):
            raise ValueError(
                f"Native FLUX checkpoint dimensions do not match {base_id}: {mismatches}"
            )


def discover_native_flux(path: Path, version: Literal[1, 2]) -> NativeFluxCheckpoint | None:
    """Recognize native FLUX by tensor structure, independently of filenames."""
    if path.is_dir() and any(
        (path / name).is_file() for name in ("model_index.json", "modular_model_index.json")
    ):
        return None
    files = (
        [path]
        if path.is_file() and path.suffix == ".safetensors"
        else sorted({*path.glob("*.safetensors"), *path.glob("diffusion_models/*.safetensors")})
    )
    candidates = []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as checkpoint:
            keys = {key.removeprefix("model.diffusion_model."): key for key in checkpoint.keys()}
            signature = ("img_in.weight", "txt_in.weight", "double_blocks.0.img_attn.qkv.weight")
            if not all(key in keys for key in signature):
                continue
            marker = (
                "vector_in.in_layer.weight"
                if version == 1
                else "double_stream_modulation_img.lin.weight"
            )
            if marker not in keys:
                continue
            shapes = {
                key: tuple(checkpoint.get_slice(raw).get_shape()) for key, raw in keys.items()
            }
        candidates.append((file, shapes))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            "Multiple FLUX checkpoints found; provide a directory containing one model variant: "
            + ", ".join(str(file) for file, _ in candidates)
        )
    file, shapes = candidates[0]
    if "guidance_in.in_layer.weight" not in shapes:
        raise ValueError(f"Native FLUX.{version} checkpoint is not a supported dev variant: {file}")
    quantization = inspect_comfy_quantization([file])
    if quantization is None:
        raise ValueError(
            f"Detected native FLUX at {file}, but expected explicit Comfy NVFP4 metadata."
        )
    shapes.update(
        {
            name.removeprefix("model.diffusion_model.") + ".weight": shape
            for name, shape in quantization.nvfp4_layers.items()
        }
    )
    return NativeFluxCheckpoint(file.resolve(), version, shapes)


def probe_native_flux1(path: Path, options: Mapping[str, Any]) -> NativeFluxCheckpoint | None:
    return discover_native_flux(path, 1)


def probe_native_flux2(path: Path, options: Mapping[str, Any]) -> NativeFluxCheckpoint | None:
    return discover_native_flux(path, 2)
