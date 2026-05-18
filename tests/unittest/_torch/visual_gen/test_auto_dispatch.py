# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-path dispatch logic tests (CPU-only, no checkpoint required).

Verifies `AutoPipeline.resolve_target_class` and `AutoPipeline.from_config`
correctly route to the handwritten registry or the `AutoTransformerPipeline`
based on `pipeline_mode` × registered/unregistered `class_name`.

These tests don't load weights or build a real pipeline — they only check
the resolver's class selection, so they're cheap and CI-safe.
"""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest


def _write_model_index(tmpdir: str, class_name: str) -> str:
    """Create a minimal Diffusers `model_index.json` so the resolver's
    checkpoint detection has something to read.
    """
    path = os.path.join(tmpdir, "model_index.json")
    with open(path, "w") as f:
        json.dump({"_class_name": class_name, "_diffusers_version": "0.39.0"}, f)
    return tmpdir


def _mk_config(pipeline_mode: str) -> SimpleNamespace:
    """Minimal stand-in for `DiffusionModelConfig`. `resolve_target_class`
    only reads `.pipeline_mode`; the rest is irrelevant to dispatch.
    """
    return SimpleNamespace(pipeline_mode=pipeline_mode)


# ---------------------------------------------------------------------------
# resolve_target_class
# ---------------------------------------------------------------------------


def test_resolve_auto_mode_always_returns_auto():
    """`pipeline_mode='auto'` should bypass the registry entirely."""
    from tensorrt_llm._torch.visual_gen.auto.pipeline import AutoTransformerPipeline
    from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline

    with tempfile.TemporaryDirectory() as tmp:
        # Even with a checkpoint that WOULD register as FluxPipeline...
        _write_model_index(tmp, "FluxPipeline")
        cls = AutoPipeline.resolve_target_class(_mk_config("auto"), tmp)
    assert cls is AutoTransformerPipeline


def test_resolve_fallback_registered_returns_handwritten():
    """`pipeline_mode='fallback'` + registered class → handwritten pipeline."""
    from tensorrt_llm._torch.visual_gen.models import flux  # noqa: F401 — registers
    from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline

    # FluxPipeline should be in the registry after the import side effect.
    if "FluxPipeline" not in PIPELINE_REGISTRY:
        pytest.skip("FluxPipeline not registered (registry import side-effect missing)")

    with tempfile.TemporaryDirectory() as tmp:
        _write_model_index(tmp, "FluxPipeline")
        cls = AutoPipeline.resolve_target_class(_mk_config("fallback"), tmp)
    # `resolve_variant` may upgrade to a specialised subclass; verify the
    # resolved class is at least a subclass of the registered one.
    assert issubclass(cls, PIPELINE_REGISTRY["FluxPipeline"])


def test_resolve_fallback_unregistered_returns_auto():
    """`pipeline_mode='fallback'` + unregistered class → auto path."""
    from tensorrt_llm._torch.visual_gen.auto.pipeline import AutoTransformerPipeline
    from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline

    with tempfile.TemporaryDirectory() as tmp:
        # `SomeUnregisteredCustomPipeline` won't match any substring rule in
        # `_detect_from_checkpoint` and won't be in the registry.
        _write_model_index(tmp, "SomeUnregisteredCustomPipeline")
        cls = AutoPipeline.resolve_target_class(_mk_config("fallback"), tmp)
    assert cls is AutoTransformerPipeline


def test_resolve_strict_unregistered_raises():
    """`pipeline_mode='strict'` + unregistered class → ValueError."""
    from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline

    with tempfile.TemporaryDirectory() as tmp:
        _write_model_index(tmp, "SomeUnregisteredCustomPipeline")
        with pytest.raises(ValueError, match="strict"):
            AutoPipeline.resolve_target_class(_mk_config("strict"), tmp)


def test_resolve_strict_registered_returns_handwritten():
    """`pipeline_mode='strict'` + registered class → handwritten pipeline."""
    from tensorrt_llm._torch.visual_gen.models import flux  # noqa: F401 — registers
    from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline

    if "FluxPipeline" not in PIPELINE_REGISTRY:
        pytest.skip("FluxPipeline not registered")

    with tempfile.TemporaryDirectory() as tmp:
        _write_model_index(tmp, "FluxPipeline")
        cls = AutoPipeline.resolve_target_class(_mk_config("strict"), tmp)
    assert issubclass(cls, PIPELINE_REGISTRY["FluxPipeline"])


# ---------------------------------------------------------------------------
# Inheritance contract
# ---------------------------------------------------------------------------


def test_auto_transformer_pipeline_is_not_base_pipeline():
    """`AutoTransformerPipeline` is intentionally NOT a `BasePipeline` subclass.

    The loader (`pipeline_loader.PipelineLoader.load`) branches on
    `isinstance(pipeline, BasePipeline)` to gate handwritten-only lifecycle
    hooks (`_materialize_meta_tensors`, `load_transformer_weights`,
    `load_weights`, `load_standard_components`, `torch_compile`). If a
    future refactor accidentally re-introduces inheritance, those hooks
    will silently no-op against an auto path that does its load work in
    `__init__` instead — a regression that's hard to spot at code review
    but easy to catch with this test.
    """
    from tensorrt_llm._torch.visual_gen.auto.pipeline import AutoTransformerPipeline
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline

    assert not issubclass(AutoTransformerPipeline, BasePipeline), (
        "AutoTransformerPipeline must NOT inherit from BasePipeline — see "
        "auto/pipeline.py module docstring for the rationale."
    )


def test_auto_transformer_pipeline_implements_executor_surface():
    """The auto path doesn't subclass BasePipeline but DOES need to expose the
    methods/attrs the executor reads. Catch silent breakage if a future edit
    removes one.
    """
    from tensorrt_llm._torch.visual_gen.auto.pipeline import AutoTransformerPipeline

    for name in (
        "infer",  # executor.py:327
        "warmup",  # pipeline_loader.py:244 / :246
        "warmup_cache_key",  # executor.py:317
        "default_generation_params",  # executor.py:189
        "extra_param_specs",  # executor.py:190
    ):
        assert hasattr(AutoTransformerPipeline, name), (
            f"AutoTransformerPipeline is missing executor-required method/property {name!r}"
        )
