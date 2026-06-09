# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import pkgutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

AUTO_DEPLOY_TRANSFORM_LIBRARY_PACKAGE = "tensorrt_llm._torch.auto_deploy.transform.library"
AUTO_DEPLOY_TRANSFORM_LIBRARY_PATH = Path("tensorrt_llm/_torch/auto_deploy/transform/library")
AUTO_DEPLOY_TRANSFORM_CONFIGS = (
    ("graph", Path("tensorrt_llm/_torch/auto_deploy/config/default.yaml")),
    (
        "transformers",
        Path("tensorrt_llm/_torch/auto_deploy/config/transformers.yaml"),
    ),
)
AUTOCLASS_OPTIONS = (
    "   :members:",
    "   :show-inheritance:",
)

STAGE_TITLES = {
    "factory": "Factory",
    "export": "Export",
    "post_export": "Post-Export",
    "pattern_matcher": "Pattern Matching",
    "sharding": "Sharding",
    "weight_load": "Weight Loading",
    "post_load_fusion": "Post-Load Fusion",
    "cache_init": "Cache Initialization",
    "visualize": "Visualization",
    "compile": "Compilation",
}

TITLE_REPLACEMENTS = {
    "fp8": "FP8",
    "gdn": "GDN",
    "kv": "KV",
    "kvcache": "KV Cache",
    "l2": "L2",
    "mlir": "MLIR",
    "mla": "MLA",
    "moe": "MoE",
    "mrope": "mRoPE",
    "mxfp4": "MXFP4",
    "noop": "No-op",
    "nvfp4": "NVFP4",
    "rmsnorm": "RMSNorm",
    "rope": "RoPE",
    "silu": "SiLU",
    "ssm": "SSM",
    "swiglu": "SwiGLU",
    "trtllm": "TRT-LLM",
}


@dataclass(frozen=True)
class RegisteredTransform:
    key: str
    module_name: str
    class_name: str
    config_class_name: str
    config_module_name: str | None

    @property
    def qualified_class_name(self) -> str:
        return f"{AUTO_DEPLOY_TRANSFORM_LIBRARY_PACKAGE}.{self.module_name}.{self.class_name}"

    @property
    def qualified_module_name(self) -> str:
        return f"{AUTO_DEPLOY_TRANSFORM_LIBRARY_PACKAGE}.{self.module_name}"

    @property
    def qualified_config_class_name(self) -> str | None:
        if self.config_module_name is None:
            return None
        return (
            f"{AUTO_DEPLOY_TRANSFORM_LIBRARY_PACKAGE}.{self.config_module_name}"
            f".{self.config_class_name}"
        )


@dataclass(frozen=True)
class ParsedClass:
    module_name: str
    class_name: str
    base_class_names: tuple[str, ...]
    config_class_name: str | None
    transform_keys: tuple[str, ...]


@dataclass
class ConfiguredTransform:
    key: str
    stage: str
    modes: list[str]


def _repo_root_from_source_dir(source_dir: str) -> Path:
    """Return the nearest ancestor that contains the AutoDeploy transform library."""
    source_path = Path(source_dir).resolve()
    for path in (source_path, *source_path.parents):
        if (path / AUTO_DEPLOY_TRANSFORM_LIBRARY_PATH).is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find repository root containing {AUTO_DEPLOY_TRANSFORM_LIBRARY_PATH}"
    )


def _discover_transform_modules(library_path: Path) -> list[str]:
    """Discover public AutoDeploy transform modules without importing them."""
    if not library_path.is_dir():
        raise FileNotFoundError(f"AutoDeploy transform library not found: {library_path}")

    return sorted(
        module_info.name
        for module_info in pkgutil.iter_modules([str(library_path)])
        if not module_info.name.startswith("_")
    )


def _module_title(module_name: str) -> str:
    """Convert a transform module name into a readable section title."""
    words = [TITLE_REPLACEMENTS.get(part, part.title()) for part in module_name.split("_")]
    return " ".join(words)


def _mode_list(modes: list[str]) -> str:
    return ", ".join(f"``{mode}``" for mode in sorted(modes))


def _register_key_from_decorator(decorator: ast.expr) -> str | None:
    if not isinstance(decorator, ast.Call):
        return None
    if not isinstance(decorator.func, ast.Attribute):
        return None
    if decorator.func.attr != "register":
        return None
    if not isinstance(decorator.func.value, ast.Name):
        return None
    if decorator.func.value.id != "TransformRegistry":
        return None
    if not decorator.args:
        return None
    key_arg = decorator.args[0]
    if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
        return key_arg.value
    return None


def _name_from_expr(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _get_config_class_name(node: ast.ClassDef) -> str | None:
    for child_node in node.body:
        if not isinstance(child_node, ast.FunctionDef):
            continue
        if child_node.name != "get_config_class":
            continue
        for statement in child_node.body:
            if isinstance(statement, ast.Return):
                return _name_from_expr(statement.value)
    return None


def _parse_transform_classes(
    library_path: Path,
) -> tuple[list[ParsedClass], dict[str, list[ParsedClass]]]:
    parsed_classes: list[ParsedClass] = []
    classes_by_name: dict[str, list[ParsedClass]] = {}

    for module_name in _discover_transform_modules(library_path):
        module_path = library_path / f"{module_name}.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            parsed_class = ParsedClass(
                module_name=module_name,
                class_name=node.name,
                base_class_names=tuple(
                    base_class_name
                    for base in node.bases
                    if (base_class_name := _name_from_expr(base)) is not None
                ),
                config_class_name=_get_config_class_name(node),
                transform_keys=tuple(
                    transform_key
                    for decorator in node.decorator_list
                    if (transform_key := _register_key_from_decorator(decorator)) is not None
                ),
            )
            parsed_classes.append(parsed_class)
            classes_by_name.setdefault(parsed_class.class_name, []).append(parsed_class)

    return parsed_classes, classes_by_name


def _get_library_class(
    class_name: str,
    module_name: str,
    classes_by_name: dict[str, list[ParsedClass]],
) -> ParsedClass | None:
    classes = classes_by_name.get(class_name, [])
    if len(classes) == 1:
        return classes[0]
    for parsed_class in classes:
        if parsed_class.module_name == module_name:
            return parsed_class
    return None


def _resolve_config_class(
    parsed_class: ParsedClass,
    classes_by_name: dict[str, list[ParsedClass]],
    seen: set[str] | None = None,
) -> ParsedClass | None:
    """Resolve a transform's config class, following simple inheritance."""
    if parsed_class.config_class_name:
        if parsed_class.config_class_name == "TransformConfig":
            return None
        return _get_library_class(
            parsed_class.config_class_name,
            parsed_class.module_name,
            classes_by_name,
        )

    seen = seen or set()
    seen.add(parsed_class.class_name)
    for base_class_name in parsed_class.base_class_names:
        if base_class_name in seen:
            continue
        base_classes = classes_by_name.get(base_class_name, [])
        if len(base_classes) != 1:
            continue
        return _resolve_config_class(base_classes[0], classes_by_name, seen)

    return None


def _discover_registered_transforms(library_path: Path) -> dict[str, RegisteredTransform]:
    """Discover registered transform classes without importing transform modules."""
    registered_transforms: dict[str, RegisteredTransform] = {}
    parsed_classes, classes_by_name = _parse_transform_classes(library_path)

    for parsed_class in parsed_classes:
        config_class = _resolve_config_class(parsed_class, classes_by_name)
        for transform_key in parsed_class.transform_keys:
            if transform_key in registered_transforms:
                previous = registered_transforms[transform_key]
                raise ValueError(
                    f"Transform {transform_key!r} is registered by both "
                    f"{previous.qualified_class_name} and "
                    f"{parsed_class.module_name}.{parsed_class.class_name}"
                )
            registered_transforms[transform_key] = RegisteredTransform(
                key=transform_key,
                module_name=parsed_class.module_name,
                class_name=parsed_class.class_name,
                config_class_name=config_class.class_name if config_class else "TransformConfig",
                config_module_name=config_class.module_name if config_class else None,
            )

    return registered_transforms


def _load_configured_transforms(repo_root: Path) -> list[ConfiguredTransform]:
    """Load transform stage metadata from the checked-in AutoDeploy configs."""
    configured_by_key: dict[str, ConfiguredTransform] = {}
    configured_transforms: list[ConfiguredTransform] = []

    for mode, config_path in AUTO_DEPLOY_TRANSFORM_CONFIGS:
        config = yaml.safe_load((repo_root / config_path).read_text(encoding="utf-8"))
        transforms = config.get("transforms", {})

        for transform_key, transform_config in transforms.items():
            stage = transform_config.get("stage")
            if not stage:
                raise ValueError(
                    f"Transform {transform_key!r} in {config_path} does not define a stage"
                )

            configured_transform = configured_by_key.get(transform_key)
            if configured_transform is not None:
                if configured_transform.stage != stage:
                    raise ValueError(
                        f"Transform {transform_key!r} has stages "
                        f"{configured_transform.stage!r} and {stage!r}"
                    )
                configured_transform.modes.append(mode)
                continue

            configured_transform = ConfiguredTransform(
                key=transform_key,
                stage=stage,
                modes=[mode],
            )
            configured_by_key[transform_key] = configured_transform
            configured_transforms.append(configured_transform)

    return configured_transforms


def _transform_section(
    transform_key: str,
    registered_transform: RegisteredTransform,
    modes: list[str] | None = None,
) -> list[str]:
    title = _module_title(transform_key)
    config_lines: list[str] = [
        ".. rubric:: YAML configuration",
        "",
    ]
    if registered_transform.qualified_config_class_name is None:
        config_lines.extend(
            [
                "Uses the common ``TransformConfig`` fields documented in :doc:`core`.",
                "",
            ]
        )
    else:
        config_lines.extend(
            [
                "The fields below can be set under this transform's entry in the "
                "AutoDeploy config YAML.",
                "",
                f".. autopydantic_model:: {registered_transform.qualified_config_class_name}",
                "   :members:",
                "   :show-inheritance:",
                "   :no-index:",
                "",
            ]
        )

    return [
        title,
        "~" * len(title),
        "",
        f"Transform key: ``{transform_key}``",
        "",
        f"Source module: ``{registered_transform.qualified_module_name}``",
        "",
        *(["Configured modes: " + _mode_list(modes), ""] if modes else []),
        f".. autoclass:: {registered_transform.qualified_class_name}",
        *AUTOCLASS_OPTIONS,
        "",
        *config_lines,
    ]


def _note_auto_deploy_dependencies(directive: SphinxDirective, repo_root: Path) -> None:
    library_path = repo_root / AUTO_DEPLOY_TRANSFORM_LIBRARY_PATH
    directive.env.note_dependency(str(library_path))
    for path in sorted(library_path.glob("*.py")):
        directive.env.note_dependency(str(path))
    for _, config_path in AUTO_DEPLOY_TRANSFORM_CONFIGS:
        directive.env.note_dependency(str(repo_root / config_path))


class AutoDeployTransformStageDirective(SphinxDirective):
    """Render autodoc sections for configured transforms in one pipeline stage."""

    has_content = False
    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        stage = self.arguments[0]
        repo_root = _repo_root_from_source_dir(self.env.app.srcdir)
        library_path = repo_root / AUTO_DEPLOY_TRANSFORM_LIBRARY_PATH
        _note_auto_deploy_dependencies(self, repo_root)

        registered_transforms = _discover_registered_transforms(library_path)
        configured_transforms = [
            transform
            for transform in _load_configured_transforms(repo_root)
            if transform.stage == stage
        ]

        if not configured_transforms:
            title = STAGE_TITLES.get(stage, stage)
            return [
                nodes.paragraph(
                    text=f"No AutoDeploy transforms are configured for the {title} stage."
                )
            ]

        generated_lines = StringList()
        for configured_transform in configured_transforms:
            registered_transform = registered_transforms.get(configured_transform.key)
            if registered_transform is None:
                raise ValueError(
                    f"Configured transform {configured_transform.key!r} is not registered"
                )
            for line in _transform_section(
                configured_transform.key,
                registered_transform,
                configured_transform.modes,
            ):
                generated_lines.append(line, source=str(library_path))

        container = nodes.container()
        nested_parse_with_titles(self.state, generated_lines, container)
        return container.children


class AutoDeployAdditionalTransformsDirective(SphinxDirective):
    """Render registered transforms that are not referenced by checked-in configs."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        repo_root = _repo_root_from_source_dir(self.env.app.srcdir)
        library_path = repo_root / AUTO_DEPLOY_TRANSFORM_LIBRARY_PATH
        _note_auto_deploy_dependencies(self, repo_root)

        registered_transforms = _discover_registered_transforms(library_path)
        configured_keys = {transform.key for transform in _load_configured_transforms(repo_root)}
        additional_transforms = [
            registered_transform
            for transform_key, registered_transform in sorted(registered_transforms.items())
            if transform_key not in configured_keys
        ]

        if not additional_transforms:
            return [
                nodes.paragraph(
                    text="Every registered AutoDeploy transform is referenced by a checked-in config."
                )
            ]

        generated_lines = StringList()
        for registered_transform in additional_transforms:
            for line in _transform_section(
                registered_transform.key,
                registered_transform,
            ):
                generated_lines.append(line, source=str(library_path))

        container = nodes.container()
        nested_parse_with_titles(self.state, generated_lines, container)
        return container.children


def setup(app: Sphinx) -> dict[str, bool | str]:
    app.add_directive(
        "trtllm_auto_deploy_transform_stage",
        AutoDeployTransformStageDirective,
    )
    app.add_directive(
        "trtllm_auto_deploy_additional_transforms",
        AutoDeployAdditionalTransformsDirective,
    )
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
