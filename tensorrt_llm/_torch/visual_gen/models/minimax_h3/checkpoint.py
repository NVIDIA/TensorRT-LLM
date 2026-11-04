# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 native checkpoint discovery and supporting-component resolution."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock
from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open

from tensorrt_llm._torch.visual_gen.checkpoint_adapter import ResolvedCheckpoint
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_bundle import prepare_comfy_bundle
from tensorrt_llm._torch.visual_gen.checkpoints.comfy_quant import inspect_comfy_quantization
from tensorrt_llm.llmapi.utils import download_hf_partial
from tensorrt_llm.logger import logger

_H3_BASE = "MiniMaxAI/MiniMax-H3"
_H3_COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


@dataclass(frozen=True)
class NativeH3Checkpoint:
    path: Path
    workflow: str
    shapes: dict[str, tuple[int, ...]]

    def resolve(self) -> ResolvedCheckpoint:
        return ResolvedCheckpoint(resolve_native_h3(self), {"workflow": self.workflow})


def discover_native_h3(path: Path, workflow: str | None = None) -> NativeH3Checkpoint | None:
    """Identify H3 from tensor keys, never from a directory or model filename.

    Workflow labels distinguish FL2VA and Ref2VA, whose tensor shapes are equal.
    Alternative checkpoints are never merged or silently selected.
    """
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
    detected_files = []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as checkpoint:
            keys = {key.removeprefix("model.diffusion_model."): key for key in checkpoint.keys()}
            signature = (
                "video_patch_proj.weight",
                "audio_patch_proj.weight",
                "condition_proj.weight",
            )
            if not all(key in keys for key in signature):
                continue
            if workflow not in (None, "fl2va", "ref2va"):
                raise ValueError("MiniMax-H3 workflow must be 'fl2va' or 'ref2va'.")
            detected_files.append(file)
            shapes = {key: tuple(checkpoint.get_slice(keys[key]).get_shape()) for key in signature}
            metadata = checkpoint.metadata() or {}
        label = metadata.get("workflow")
        if label is None:
            labels = [name for name in ("fl2va", "ref2va") if name in file.stem.lower()]
            label = labels[0] if len(labels) == 1 else None
        if label not in (None, "fl2va", "ref2va"):
            raise ValueError(f"{file}: unsupported H3 workflow {label!r}")
        if workflow is not None and label is not None and label != workflow:
            continue
        candidates.append((file, label or workflow, shapes))
    if not candidates:
        if detected_files:
            raise ValueError(
                f"Detected H3 checkpoints do not match requested workflow {workflow!r}: "
                + ", ".join(str(file) for file in detected_files)
            )
        return None
    if len(candidates) != 1:
        raise ValueError(
            "Multiple H3 checkpoints found; provide a directory containing one model variant "
            "(or select the H3 workflow with pipeline_config): "
            + ", ".join(str(candidate[0]) for candidate in candidates)
        )
    file, selected_workflow, shapes = candidates[0]
    if selected_workflow is None:
        raise ValueError(
            f"Detected native H3 at {file}, but FL2VA/Ref2VA cannot be distinguished by shapes. "
            "Set pipeline_config.workflow to 'fl2va' or 'ref2va', or retain the publisher's "
            "workflow-labelled checkpoint filename."
        )
    quantization = inspect_comfy_quantization([file])
    if quantization is None:
        raise ValueError(
            f"Detected native H3 at {file}, but its weight encoding is unsupported: "
            "expected explicit Comfy NVFP4 metadata."
        )
    logical_shapes = {
        name.removeprefix("model.diffusion_model.") + ".weight": shape
        for name, shape in quantization.nvfp4_layers.items()
    }
    shapes = {name: logical_shapes.get(name, shape) for name, shape in shapes.items()}
    return NativeH3Checkpoint(file.resolve(), selected_workflow, shapes)


def resolve_native_h3(checkpoint: NativeH3Checkpoint) -> str:
    """Resolve the known H3 base and cache a pipeline view without rewriting weights."""
    component = "transformer_ref" if checkpoint.workflow == "ref2va" else "transformer"
    logger.info(
        f"Detected MiniMax-H3 {checkpoint.workflow}: Comfy NVFP4 at {checkpoint.path}. "
        "Packed weights will be converted internally during loading."
    )
    config_patterns = ["modular_model_index.json", f"{component}/config.json"]
    logger.info(
        f"Resolving H3 configuration and supporting components from {_H3_BASE} "
        "(Hugging Face cache/download; original transformer weights are excluded)."
    )
    base = download_hf_partial(_H3_BASE, allow_patterns=config_patterns)
    config = json.loads((base / component / "config.json").read_text())
    patch = config["patch_size"]
    expected = {
        "video_patch_proj.weight": (
            config["hidden_size"],
            config["in_channels"] * patch[0] * patch[1] * patch[2],
        ),
        "audio_patch_proj.weight": (config["hidden_size"], config["audio_in_channels"]),
        "condition_proj.weight": (config["hidden_size"], config["text_dim"]),
    }
    if checkpoint.shapes != expected:
        raise ValueError(
            f"Detected H3 checkpoint dimensions do not match {_H3_BASE}: "
            f"got {checkpoint.shapes}, expected {expected}"
        )
    base = download_hf_partial(
        _H3_BASE,
        allow_patterns=config_patterns + [f"{name}/*" for name in _H3_COMPONENTS],
        revision=base.name,
    )
    # Immutable Hub snapshots plus the selected file's identity separate cached views.
    stat = checkpoint.path.stat()
    identity = f"{checkpoint.path}:{stat.st_size}:{stat.st_mtime_ns}:{base}:{component}"
    digest = hashlib.sha256(identity.encode()).hexdigest()
    cache = Path(HF_HUB_CACHE).parent / "tensorrt_llm" / "visual_gen"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / digest
    with FileLock(str(cache / (digest + ".lock"))):
        if not output.exists():
            prepare_comfy_bundle(base, checkpoint.path, output, component=component)
    logger.info(f"H3 supporting components: {base}; selected transformer: {checkpoint.path}")
    return str(output)


def probe_native_h3(path: Path, options: Mapping[str, Any]) -> NativeH3Checkpoint | None:
    """Recognize H3 using local tensors and the existing task selection option."""
    return discover_native_h3(path, options.get("workflow"))
