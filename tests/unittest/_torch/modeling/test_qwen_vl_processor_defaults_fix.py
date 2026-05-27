# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``install_qwen_vl_processor_defaults_fix``.

The upstream Qwen2/2.5/3-VL processors' ``_get_num_multimodal_tokens`` does
``<ProcessorKwargs>._defaults.get("<modality>_kwargs", {}).update(kwargs)`` on
the class-level default dict (instead of a copy). Once any caller passes
processor *output* keys (e.g. ``video_grid_thw``) to
``get_num_multimodal_tokens``, those keys get baked into the per-modality
default and leak into every subsequent processor call's per-modality
``output_kwargs``, tripping ``ProcessorMixin._merge_kwargs``'s strict
``TypedDict`` validation.

The fix re-classes the loaded processor instance to a TRT-LLM subclass that
overrides ``_get_num_multimodal_tokens`` to take a defensive ``dict(...)``
copy before merging caller kwargs. These tests pin down:

  - The fix actually installs (instance is re-classed).
  - The class-level ``_defaults`` are not polluted after a call that passes
    output keys.
  - Repeated installs are idempotent.
"""

import pytest

transformers = pytest.importorskip("transformers")

from tensorrt_llm._torch.models.modeling_multimodal_utils import (  # noqa: E402
    install_qwen_vl_processor_defaults_fix,
)


def _import_qwen_vl_kwargs_cls():
    """Return the first available Qwen-VL ``*ProcessorKwargs`` class, or skip."""
    for module_path, kwargs_name, proc_name in (
        (
            "transformers.models.qwen3_vl.processing_qwen3_vl",
            "Qwen3VLProcessorKwargs",
            "Qwen3VLProcessor",
        ),
        (
            "transformers.models.qwen2_5_vl.processing_qwen2_5_vl",
            "Qwen2_5_VLProcessorKwargs",
            "Qwen2_5_VLProcessor",
        ),
        (
            "transformers.models.qwen2_vl.processing_qwen2_vl",
            "Qwen2VLProcessorKwargs",
            "Qwen2VLProcessor",
        ),
    ):
        try:
            mod = __import__(module_path, fromlist=[kwargs_name, proc_name])
            return getattr(mod, kwargs_name), getattr(mod, proc_name)
        except (ImportError, AttributeError):
            continue
    pytest.skip("no Qwen-VL processor available in this transformers install")


class _StubImageProcessor:
    merge_size = 1

    @staticmethod
    def get_number_of_image_patches(h, w, kwargs):
        return h * w


class _StubVideoProcessor:
    @staticmethod
    def get_number_of_video_patches(t, h, w, kwargs):
        return t * h * w


def _make_stub_processor(processor_cls):
    """Construct just enough of a processor instance to call
    ``_get_num_multimodal_tokens`` — bypassing the real ``__init__`` which
    requires loading a HuggingFace checkpoint."""
    instance = processor_cls.__new__(processor_cls)
    instance.image_processor = _StubImageProcessor()
    instance.video_processor = _StubVideoProcessor()
    return instance


def _snapshot_defaults(kwargs_cls):
    """Deep-snapshot the class-level _defaults for later equality check."""
    import copy

    return copy.deepcopy(kwargs_cls._defaults)


def test_install_returns_true_for_known_processor():
    kwargs_cls, processor_cls = _import_qwen_vl_kwargs_cls()
    proc = _make_stub_processor(processor_cls)
    original_cls = type(proc)
    assert install_qwen_vl_processor_defaults_fix(proc) is True
    assert type(proc) is not original_cls
    assert getattr(type(proc), "_tllm_qwen_vl_defaults_fixed", False)
    # Subclass of the original.
    assert issubclass(type(proc), original_cls)


def test_install_returns_false_for_unknown_processor():
    class _NotAQwenVLProcessor:
        pass

    obj = _NotAQwenVLProcessor()
    assert install_qwen_vl_processor_defaults_fix(obj) is False
    assert type(obj) is _NotAQwenVLProcessor


def test_install_is_idempotent():
    kwargs_cls, processor_cls = _import_qwen_vl_kwargs_cls()
    proc = _make_stub_processor(processor_cls)
    assert install_qwen_vl_processor_defaults_fix(proc) is True
    safe_cls = type(proc)
    assert install_qwen_vl_processor_defaults_fix(proc) is True
    assert type(proc) is safe_cls  # not re-wrapped


def test_defaults_not_polluted_after_call_with_output_keys():
    """The crucial regression: passing processor output keys (which leak via
    saved init_kwargs in production) must not bake them into the class-level
    _defaults dicts."""
    kwargs_cls, processor_cls = _import_qwen_vl_kwargs_cls()
    proc = _make_stub_processor(processor_cls)
    install_qwen_vl_processor_defaults_fix(proc)

    before = _snapshot_defaults(kwargs_cls)

    # Mimic what HF preprocessing ends up passing under the bug — output keys
    # leak into kwargs via tokenizer init_kwargs. _get_num_multimodal_tokens
    # is called with image_sizes/video_sizes + arbitrary kwargs.
    proc._get_num_multimodal_tokens(
        image_sizes=[(2, 2)],
        video_sizes=[(2, 2, 2)],
        video_grid_thw="leak1",
        image_grid_thw="leak2",
        pixel_values="leak3",
    )

    after = _snapshot_defaults(kwargs_cls)
    assert after == before, f"class-level _defaults was polluted: before={before!r} after={after!r}"
    for modality_key in ("images_kwargs", "videos_kwargs"):
        modality = kwargs_cls._defaults.get(modality_key, {})
        for leaked in ("video_grid_thw", "image_grid_thw", "pixel_values"):
            assert leaked not in modality, (
                f"{leaked!r} leaked into _defaults[{modality_key!r}]={modality!r}"
            )


def test_install_scrubs_already_polluted_defaults():
    """If a prior unpatched processor in the same process already leaked
    output keys into the class-level ``_defaults``, ``install_...`` must
    scrub them — otherwise ``_merge_kwargs`` will still copy the stale keys
    and trip TypedDict validation on the very first call."""
    kwargs_cls, processor_cls = _import_qwen_vl_kwargs_cls()
    import copy

    saved = copy.deepcopy(kwargs_cls._defaults)
    try:
        # Simulate prior pollution from an unpatched instance.
        kwargs_cls._defaults.setdefault("images_kwargs", {})["video_grid_thw"] = "stale1"
        kwargs_cls._defaults.setdefault("videos_kwargs", {})["pixel_values"] = "stale2"
        kwargs_cls._defaults.setdefault("common_kwargs", {})["image_grid_thw"] = "stale3"

        proc = _make_stub_processor(processor_cls)
        assert install_qwen_vl_processor_defaults_fix(proc) is True

        for modality_key in ("images_kwargs", "videos_kwargs", "common_kwargs"):
            modality = kwargs_cls._defaults.get(modality_key, {})
            for leaked in (
                "video_grid_thw",
                "image_grid_thw",
                "pixel_values",
                "pixel_values_videos",
                "second_per_grid_ts",
                "mm_token_type_ids",
            ):
                assert leaked not in modality, (
                    f"{leaked!r} not scrubbed from _defaults[{modality_key!r}]={modality!r}"
                )
    finally:
        kwargs_cls._defaults.clear()
        kwargs_cls._defaults.update(saved)


def test_install_preserves_remote_code_subclass():
    """When ``AutoProcessor`` returns a ``trust_remote_code`` subclass of a
    known Qwen processor, the safe class must inherit from the *concrete*
    subclass (not its known upstream base) so the subclass's overrides
    aren't dropped by re-classing."""
    _, processor_cls = _import_qwen_vl_kwargs_cls()

    # Define a fake remote-code subclass with an extra method we can detect.
    class _RemoteSubclass(processor_cls):
        REMOTE_MARKER = "kept"

        def custom_remote_method(self):
            return type(self).REMOTE_MARKER

    proc = _make_stub_processor(_RemoteSubclass)
    assert install_qwen_vl_processor_defaults_fix(proc) is True

    safe_cls = type(proc)
    # Safe class must inherit from the concrete remote subclass, not the
    # upstream base — otherwise REMOTE_MARKER / custom_remote_method are lost.
    assert issubclass(safe_cls, _RemoteSubclass), (
        f"safe class {safe_cls.__mro__} should keep _RemoteSubclass in MRO"
    )
    assert proc.custom_remote_method() == "kept"
    # And the fix is still installed.
    assert getattr(safe_cls, "_tllm_qwen_vl_defaults_fixed", False)


def test_unpatched_processor_pollutes_defaults():
    """Documents the upstream bug: without our fix, the same call sequence
    *does* pollute _defaults. This guards against the regression test above
    becoming a vacuous tautology if upstream silently fixes the bug."""
    kwargs_cls, processor_cls = _import_qwen_vl_kwargs_cls()
    proc = _make_stub_processor(processor_cls)
    # No fix installed; restore defaults at end so we don't affect other tests.
    import copy

    saved = copy.deepcopy(kwargs_cls._defaults)
    try:
        try:
            proc._get_num_multimodal_tokens(
                image_sizes=[(2, 2)],
                video_sizes=[(2, 2, 2)],
                video_grid_thw="leak",
            )
        except Exception:
            # The upstream method may raise on stub input; that's fine — the
            # mutation we care about happens before the raise.
            pass
        videos_kwargs = kwargs_cls._defaults.get("videos_kwargs", {})
        if "video_grid_thw" not in videos_kwargs:
            pytest.skip(
                "upstream Qwen-VL processor no longer mutates class-level "
                "_defaults; this fix may be obsolete on this transformers "
                "version"
            )
    finally:
        kwargs_cls._defaults.clear()
        kwargs_cls._defaults.update(saved)
