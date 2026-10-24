# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pair a native Comfy transformer with an explicitly selected Diffusers pipeline.

Run with ``python -m tensorrt_llm._torch.visual_gen.checkpoints.comfy_bundle``.
The resulting directory references the original files; keep both sources available.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .comfy_flux import comfy_flux_quantization_config
from .comfy_h3 import comfy_h3_quantization_config
from .comfy_quant import inspect_comfy_quantization


def prepare_comfy_bundle(
    base_pipeline: Path, checkpoint: Path, output: Path, *, component: str = "transformer"
) -> Path:
    """Create a local pipeline directory without copying or requantizing model weights.

    The base supplies architecture and ancillary components. The selected component
    receives only its base config and the chosen checkpoint, avoiding accidental
    merging of alternative Comfy variants or loading the original transformer.
    """
    base_pipeline, checkpoint, output = (
        path.expanduser().resolve() for path in (base_pipeline, checkpoint, output)
    )
    if component not in ("transformer", "transformer_ref"):
        raise ValueError("Supported components are transformer and transformer_ref")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    if output.is_relative_to(base_pipeline):
        raise ValueError("Output must be outside the base pipeline directory")
    manifest_path = base_pipeline / "model_index.json"
    if not manifest_path.is_file():
        manifest_path = base_pipeline / "modular_model_index.json"
    if not manifest_path.is_file():
        raise ValueError(f"Base pipeline has no Diffusers manifest: {base_pipeline}")
    manifest = json.loads(manifest_path.read_text())
    if component not in manifest or manifest[component] is None:
        raise ValueError(f"Base pipeline does not declare {component}")
    config_path = base_pipeline / component / "config.json"
    config = json.loads(config_path.read_text())
    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        raise ValueError("Select one existing Comfy .safetensors checkpoint")
    detected = inspect_comfy_quantization([checkpoint])
    if detected is None:
        raise ValueError("Selected checkpoint has no explicit Comfy NVFP4 metadata")
    architecture = config.get("_class_name")
    attention_dim = config["num_attention_heads"] * config["attention_head_dim"]
    versions = {"FluxTransformer2DModel": 1, "Flux2Transformer2DModel": 2}
    if architecture in versions:
        comfy_flux_quantization_config(
            detected,
            version=versions[architecture],
            hidden_size=attention_dim,
            mlp_ratio=config.get("mlp_ratio", 4.0),
        )
    elif architecture == "MiniMaxH3Transformer3DModel":
        comfy_h3_quantization_config(
            detected, attention_dim=attention_dim, ffn_dim=config["ffn_dim"]
        )
    else:
        raise ValueError(f"Unsupported base transformer architecture: {architecture!r}")
    # Quantization belongs to the selected checkpoint, not the replaced base weights.
    config.pop("quantization_config", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = list(base_pipeline.iterdir())
    with tempfile.TemporaryDirectory(prefix=".comfy-bundle-", dir=output.parent) as temporary:
        staged = Path(temporary) / "pipeline"
        staged.mkdir()
        for entry in entries:
            if entry.name == component:
                continue
            (staged / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        target = staged / component
        target.mkdir()
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (target / "diffusion_pytorch_model.safetensors").symlink_to(checkpoint)
        # Copy the manifest so the prepared pipeline retains its own routing identity.
        (staged / manifest_path.name).unlink()
        shutil.copyfile(manifest_path, staged / manifest_path.name)
        staged.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pipeline", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--component", choices=("transformer", "transformer_ref"), default="transformer"
    )
    args = parser.parse_args()
    print(
        prepare_comfy_bundle(
            args.base_pipeline, args.checkpoint, args.output, component=args.component
        )
    )


if __name__ == "__main__":
    main()
