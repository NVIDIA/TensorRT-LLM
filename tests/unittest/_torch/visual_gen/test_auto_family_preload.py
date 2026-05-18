# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`_family_preload_overrides` context-manager tests (CPU-only).

Verifies the auto path's Diffusers class-attribute mutations are scoped to
the load duration (save + restore in `finally`), rather than the permanent
mutation pattern earlier versions used.

Regression target: a future edit that drops the save/restore around the
WAN `_keep_in_fp32_modules = []` mutation would let it leak into
mixed-mode evaluation in the same Python process.
"""

import json
import os
import tempfile

import pytest


def _write_index(tmp: str, class_name: str) -> str:
    path = os.path.join(tmp, "model_index.json")
    with open(path, "w") as f:
        json.dump({"_class_name": class_name, "_diffusers_version": "0.39.0"}, f)
    return tmp


def test_no_override_for_non_wan_family():
    """Override list is empty when the checkpoint isn't WAN — context is a no-op."""
    from tensorrt_llm._torch.visual_gen.auto.auto_pipeline import _family_preload_overrides

    with tempfile.TemporaryDirectory() as tmp:
        _write_index(tmp, "FluxPipeline")
        # Body executes cleanly; nothing to assert beyond "no exception."
        with _family_preload_overrides(tmp):
            pass


def test_wan_keep_in_fp32_modules_restored_on_normal_exit():
    """WAN: list is set to [] inside the `with`, restored on normal exit."""
    try:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    except ImportError:
        pytest.skip("diffusers WanTransformer3DModel not available")

    from tensorrt_llm._torch.visual_gen.auto.auto_pipeline import _family_preload_overrides

    # Inject a non-empty sentinel so we can detect both the mutation AND the
    # restore (a default-empty list would mask either).
    original = WanTransformer3DModel._keep_in_fp32_modules
    sentinel = ["__test_sentinel_module__"]
    WanTransformer3DModel._keep_in_fp32_modules = sentinel
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _write_index(tmp, "WanPipeline")
            with _family_preload_overrides(tmp):
                # Inside the context: empty list (the override).
                assert WanTransformer3DModel._keep_in_fp32_modules == []
            # After exit: original (sentinel) value restored.
            assert WanTransformer3DModel._keep_in_fp32_modules == sentinel
    finally:
        WanTransformer3DModel._keep_in_fp32_modules = original


def test_wan_keep_in_fp32_modules_restored_on_exception():
    """The override must restore even if the body raises — that's the whole
    point of the `try/finally` in `_family_preload_overrides`.
    """
    try:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    except ImportError:
        pytest.skip("diffusers WanTransformer3DModel not available")

    from tensorrt_llm._torch.visual_gen.auto.auto_pipeline import _family_preload_overrides

    original = WanTransformer3DModel._keep_in_fp32_modules
    sentinel = ["__test_sentinel_module__"]
    WanTransformer3DModel._keep_in_fp32_modules = sentinel
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _write_index(tmp, "WanPipeline")
            with pytest.raises(RuntimeError, match="simulated"):
                with _family_preload_overrides(tmp):
                    raise RuntimeError("simulated from_pretrained failure")
            # Despite the exception, the override is restored.
            assert WanTransformer3DModel._keep_in_fp32_modules == sentinel
    finally:
        WanTransformer3DModel._keep_in_fp32_modules = original


def test_wan_snapshot_is_a_copy_not_a_reference():
    """If the original list were captured by reference (and Diffusers ever
    mutated it in place between save and restore), the restore would just
    re-install the mutated list. We snapshot a `list(...)` copy.
    """
    try:
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    except ImportError:
        pytest.skip("diffusers WanTransformer3DModel not available")

    from tensorrt_llm._torch.visual_gen.auto.auto_pipeline import _family_preload_overrides

    original = WanTransformer3DModel._keep_in_fp32_modules
    sentinel = ["alpha", "beta"]
    WanTransformer3DModel._keep_in_fp32_modules = sentinel
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _write_index(tmp, "WanPipeline")
            with _family_preload_overrides(tmp):
                # Simulate an external mutation of the *original* list during
                # the `with` body. If the snapshot were by reference, this
                # mutation would be visible after restore.
                sentinel.append("gamma")
            # Restore should have re-installed the snapshot (alpha, beta) —
            # without the simulated `gamma` mutation.
            assert WanTransformer3DModel._keep_in_fp32_modules == ["alpha", "beta"]
    finally:
        WanTransformer3DModel._keep_in_fp32_modules = original


def test_missing_model_index_is_silent_noop():
    """If the directory has no `model_index.json` (e.g. single-safetensor
    checkpoints like LTX-2), `_family_preload_overrides` is a clean no-op.
    """
    from tensorrt_llm._torch.visual_gen.auto.auto_pipeline import _family_preload_overrides

    with tempfile.TemporaryDirectory() as tmp:
        # No model_index.json written.
        with _family_preload_overrides(tmp):
            pass
