# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Auto-generate the AutoDeploy transforms reference page.

This script introspects the TransformRegistry to produce a reStructuredText
file listing every registered transform grouped by pipeline stage. The output
is placed at ``docs/source/features/auto_deploy/transforms/reference.rst``
and is consumed by Sphinx during ``make html``.

It is called from ``conf.py:setup()`` alongside the other generators in
``helper.py``.
"""

import inspect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


def _get_custom_config_fields(config_cls: Type) -> List[Dict[str, Any]]:
    """Return non-base config fields with their descriptions and defaults."""
    from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig

    base_fields = set(TransformConfig.model_fields.keys())
    custom = []
    for name, field_info in config_cls.model_fields.items():
        if name in base_fields:
            continue
        desc = field_info.description or ""
        default = field_info.default
        if default is None and field_info.default_factory is not None:
            try:
                default = field_info.default_factory()
            except Exception:
                default = "..."
        custom.append({"name": name, "description": desc, "default": default})
    return custom


def _get_first_docstring_line(cls: Type) -> str:
    """Extract the first non-empty line from a class docstring."""
    doc = inspect.getdoc(cls)
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _get_full_docstring(cls: Type) -> str:
    """Return the full cleaned docstring."""
    doc = inspect.getdoc(cls)
    return doc.strip() if doc else ""


def _rst_escape(text: str) -> str:
    """Escape special RST characters in inline text."""
    return text.replace("*", "\\*").replace("|", "\\|")


def generate_transforms_reference() -> None:
    """Generate the transforms reference RST file from the TransformRegistry."""
    try:
        # This import triggers auto-registration of all transforms
        from tensorrt_llm._torch.auto_deploy.transform import TransformRegistry
        from tensorrt_llm._torch.auto_deploy.transform.interface import Stages
    except ImportError as e:
        logger.warning(
            "Could not import transform registry, skipping transforms doc generation: %s",
            e,
        )
        return

    doc_dir = Path(__file__).parent / "features" / "auto_deploy" / "transforms"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / "reference.rst"

    # Load default.yaml to get the stage assignment for each transform.
    # The stage field in TransformConfig is mandatory (no default), so the
    # actual stage comes from the YAML config rather than the Python class.
    from importlib.resources import files as importlib_files

    import yaml

    config_pkg = importlib_files("tensorrt_llm._torch.auto_deploy.config")
    yaml_text = (config_pkg / "default.yaml").read_text()
    yaml_transforms = yaml.safe_load(yaml_text).get("transforms", {})

    # Also load transformers.yaml for transforms only defined there
    try:
        transformers_yaml_text = (config_pkg / "transformers.yaml").read_text()
        transformers_transforms = yaml.safe_load(transformers_yaml_text).get("transforms", {})
    except Exception:
        transformers_transforms = {}

    # Merge: default.yaml is primary, transformers.yaml fills gaps
    all_yaml_transforms = {**transformers_transforms, **yaml_transforms}

    # Build a name→Stages lookup from YAML
    stage_str_to_enum = {s.value: s for s in Stages}

    def _resolve_stage(name: str) -> Stages:
        yaml_cfg = all_yaml_transforms.get(name, {})
        stage_str = yaml_cfg.get("stage") if isinstance(yaml_cfg, dict) else None
        if stage_str and stage_str in stage_str_to_enum:
            return stage_str_to_enum[stage_str]
        return None

    # Collect all registered transforms grouped by stage
    by_stage = defaultdict(list)
    for name in sorted(TransformRegistry._registry.keys()):
        transform_cls = TransformRegistry.get(name)
        config_cls = transform_cls.get_config_class()
        stage = _resolve_stage(name)
        stage_key = stage.value if stage else "unknown"
        by_stage[stage_key].append(
            {
                "name": name,
                "cls": transform_cls,
                "config_cls": config_cls,
                "stage": stage,
            }
        )

    # Stage display names and descriptions
    stage_info = {
        Stages.FACTORY: (
            "Factory",
            "Build the model architecture on a meta device.",
        ),
        Stages.EXPORT: (
            "Export",
            "Export the PyTorch model to an FX GraphModule via ``torch.export``.",
        ),
        Stages.POST_EXPORT: (
            "Post-Export",
            "Low-level cleanups of the exported graph (remove no-ops, fix constraints).",
        ),
        Stages.PATTERN_MATCHER: (
            "Pattern Matcher",
            "High-level pattern matching to standardize and fuse graph operations.",
        ),
        Stages.SHARDING: (
            "Sharding",
            "Auto-sharding of the graph for multi-GPU parallelism.",
        ),
        Stages.WEIGHT_LOAD: (
            "Weight Load",
            "Load model weights from checkpoints onto the device.",
        ),
        Stages.POST_LOAD_FUSION: (
            "Post-Load Fusion",
            "Post-loading fusion and performance optimizations (KV-cache, quantization, kernels).",
        ),
        Stages.CACHE_INIT: (
            "Cache Init",
            "Initialize cached attention and KV cache structures.",
        ),
        Stages.VISUALIZE: (
            "Visualize",
            "Graph visualization for debugging.",
        ),
        Stages.COMPILE: (
            "Compile",
            "Graph compilation using backends like ``torch.compile`` with CUDA graphs.",
        ),
    }

    # Build RST content
    lines = []
    lines.append("AutoDeploy Transforms Reference")
    lines.append("=" * len(lines[0]))
    lines.append("")
    lines.append(".. note::")
    lines.append("")
    lines.append("   This page is auto-generated from the transform registry.")
    lines.append("   Do not edit manually.")
    lines.append("")
    lines.append(
        "This reference lists all registered AutoDeploy transforms, "
        "grouped by their pipeline stage. Each transform is applied in "
        "stage order during the inference optimization pipeline."
    )
    lines.append("")
    lines.append(".. contents:: Pipeline Stages")
    lines.append("   :local:")
    lines.append("   :depth: 2")
    lines.append("")

    # Emit pipeline overview
    lines.append("Pipeline Overview")
    lines.append("-" * len("Pipeline Overview"))
    lines.append("")
    lines.append(
        "The AutoDeploy inference optimizer applies transforms in the following stage order:"
    )
    lines.append("")
    for stage in Stages:
        display_name, desc = stage_info.get(stage, (stage.value, ""))
        count = len(by_stage.get(stage.value, []))
        lines.append(f"#. **{display_name}** ({count} transforms) -- {desc}")
    lines.append("")

    # Emit per-stage sections
    for stage in Stages:
        entries = by_stage.get(stage.value, [])
        display_name, stage_desc = stage_info.get(stage, (stage.value, ""))

        lines.append(display_name)
        lines.append("-" * max(len(display_name), 4))
        lines.append("")
        lines.append(stage_desc)
        lines.append("")

        if not entries:
            lines.append("*No transforms registered for this stage.*")
            lines.append("")
            continue

        # Summary table
        lines.append(".. list-table::")
        lines.append("   :header-rows: 1")
        lines.append("   :widths: 25 55 20")
        lines.append("")
        lines.append("   * - Transform")
        lines.append("     - Description")
        lines.append("     - Default")

        for entry in sorted(entries, key=lambda e: e["name"]):
            name = entry["name"]
            cls = entry["cls"]
            config_cls = entry["config_cls"]
            summary = _rst_escape(_get_first_docstring_line(cls))
            if not summary:
                summary = "*No description available.*"

            enabled_field = config_cls.model_fields.get("enabled")
            default_enabled = enabled_field.default if enabled_field is not None else True
            enabled_str = "enabled" if default_enabled else "disabled"

            lines.append(f"   * - ``{name}``")
            lines.append(f"     - {summary}")
            lines.append(f"     - {enabled_str}")

        lines.append("")

        # Detailed entries
        for entry in sorted(entries, key=lambda e: e["name"]):
            name = entry["name"]
            cls = entry["cls"]
            config_cls = entry["config_cls"]

            # Sub-heading for each transform
            lines.append(f"``{name}``")
            lines.append("^" * (len(name) + 4))
            lines.append("")

            # Full docstring
            full_doc = _get_full_docstring(cls)
            if full_doc:
                lines.append(full_doc)
            else:
                lines.append("*No documentation available.*")
            lines.append("")

            # Source location
            try:
                source_file = inspect.getfile(cls)
                # Make path relative to repo root
                repo_root = Path(__file__).parent.parent.parent
                rel_path = Path(source_file).relative_to(repo_root)
                source_line = inspect.getsourcelines(cls)[1]
                lines.append(f"**Source:** ``{rel_path}:{source_line}``")
                lines.append("")
            except (TypeError, ValueError, OSError):
                pass

            # Custom config fields
            custom_fields = _get_custom_config_fields(config_cls)
            if custom_fields:
                lines.append("**Config fields:**")
                lines.append("")
                for field in custom_fields:
                    default_repr = repr(field["default"])
                    desc = field["description"]
                    field_line = f"- ``{field['name']}`` (default: ``{default_repr}``)"
                    if desc:
                        field_line += f" -- {_rst_escape(desc)}"
                    lines.append(field_line)
                lines.append("")

    content = "\n".join(lines) + "\n"
    with open(doc_path, "w") as f:
        f.write(content)

    num_transforms = sum(len(v) for v in by_stage.values())
    logger.info(
        "Generated transforms reference: %d transforms across %d stages -> %s",
        num_transforms,
        len(by_stage),
        doc_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_transforms_reference()
