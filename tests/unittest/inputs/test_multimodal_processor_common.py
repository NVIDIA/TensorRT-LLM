# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-model invariants for multimodal input processors.

Mirrors vLLM's ``tests/models/multimodal/processing/test_common.py``:
a single sweep parametrized over every registered MM processor,
pinning the invariants that *every* :class:`BaseMultimodalInputProcessor`
subclass which opts into deterministic dummy sizing must satisfy:

* ``supported_modalities`` is non-empty and matches the per-model
  expectation pinned in ``conftest.py``.
* For every supported modality, ``get_size_with_most_features`` →
  ``get_num_mm_tokens`` round-trips inside the budget.
* Modalities the processor does *not* claim raise
  ``NotImplementedError`` (no silent garbage for AUDIO on a vision-only
  model, for instance).
* ``get_dummy_media`` returns the modality-typed payload the dummy
  loader expects.
* :meth:`MultimodalEncoderBudget.iter_modality_dummies` yields one
  entry per supported modality.

The sweep is parametrized off ``MM_PROCESSOR_STUBS`` in
``conftest.py``. Per-model math correctness is exercised by integration
tests that load the real model; this file covers only what holds across
models.

Adding a new model to the sweep
-------------------------------

1. On the model's ``InputProcessorBase`` class (in
   ``tensorrt_llm/_torch/models/modeling_<model>.py``), implement:

   - ``supported_modalities`` — tuple of :class:`Modality` members the
     processor handles (e.g. ``(Modality.IMAGE, Modality.AUDIO)``).
   - ``spatial_merge_unit`` — encoder→LLM token ratio (default ``1``
     means no merger).
   - ``get_num_mm_tokens(modality, **kwargs)`` — encoder attention
     tokens (pre-merger) for one media item.
   - ``get_size_with_most_features(modality, *, max_tokens)`` — inverse
     of the above: returns the largest size whose token count fits
     ``max_tokens`` (modality-shaped dict).
   - Override ``get_dummy_media`` only if the default
     (PIL image / list of frames / zero-filled ``np.ndarray``) is not
     what the HF processor expects.

2. In ``conftest.py``, add a stub builder:

   - If the model shares a vision/audio config shape with an existing
     family, reuse that family's ``_build_<family>_stub`` helper.
   - Otherwise add a new ``_build_<family>_stub`` function alongside
     it (one ``SimpleNamespace`` per nested config the math reads).

3. Append one :class:`MMProcessorStub` row to ``MM_PROCESSOR_STUBS``
   with the new ``id``, ``processor_cls``, ``builder``, and
   ``expected_modalities``.

After step 3, every test in this file runs against the new model
automatically — no per-model test file required for the contract
checks below.
"""

import numpy as np
import pytest
from PIL import Image

from tensorrt_llm._torch.pyexecutor.multimodal_budget import MultimodalEncoderBudget
from tensorrt_llm.inputs.modality import Modality


def test_supported_modalities_nonempty(mm_processor_stub):
    proc = mm_processor_stub.builder()
    assert len(proc.supported_modalities) >= 1


def test_supported_modalities_matches_expected(mm_processor_stub):
    """Pin the modality set: catches drift introduced by refactors."""
    proc = mm_processor_stub.builder()
    assert tuple(proc.supported_modalities) == mm_processor_stub.expected_modalities


def test_spatial_merge_unit_positive(mm_processor_stub):
    proc = mm_processor_stub.builder()
    assert proc.spatial_merge_unit >= 1


@pytest.mark.parametrize("budget", [256, 1024, 4096, 16384])
def test_get_size_roundtrip_fits_budget(mm_processor_stub, budget):
    proc = mm_processor_stub.builder()
    for modality in proc.supported_modalities:
        size = proc.get_size_with_most_features(modality, max_tokens=budget)
        tokens = proc.get_num_mm_tokens(modality, **size)
        assert tokens <= budget, (
            f"{mm_processor_stub.id}/{modality.value}: "
            f"size={size} → tokens={tokens} > "
            f"budget={budget}"
        )


def test_get_size_rejects_non_positive_budget(mm_processor_stub):
    proc = mm_processor_stub.builder()
    for modality in proc.supported_modalities:
        with pytest.raises(ValueError):
            proc.get_size_with_most_features(modality, max_tokens=0)


def test_unsupported_modality_raises(mm_processor_stub):
    """A modality the processor does not claim must fail loudly.

    Without this, a future bug where a model silently accepted, say,
    AUDIO and returned a nonsense size would let the encoder profiler
    over-allocate (or worse — under-allocate) without warning.
    """
    proc = mm_processor_stub.builder()
    unsupported = set(Modality) - set(proc.supported_modalities)
    for modality in unsupported:
        with pytest.raises(NotImplementedError):
            proc.get_size_with_most_features(modality, max_tokens=1024)


def test_get_dummy_media_typed_payload(mm_processor_stub):
    """The default ``get_dummy_media`` returns modality-shaped data.

    Type contract per modality (matches what the dummy loader feeds the
    HF processor):

    * ``IMAGE`` → PIL Image
    * ``VIDEO`` → list of PIL Images (≥ 1 frame)
    * ``AUDIO`` → 1-D numpy float array
    """
    proc = mm_processor_stub.builder()
    for modality in proc.supported_modalities:
        size = proc.get_size_with_most_features(modality, max_tokens=1024)
        media = proc.get_dummy_media(modality, size)
        if modality == Modality.IMAGE:
            assert isinstance(media, Image.Image)
        elif modality == Modality.VIDEO:
            assert isinstance(media, list) and len(media) >= 1
            assert all(isinstance(f, Image.Image) for f in media)
        elif modality == Modality.AUDIO:
            assert isinstance(media, np.ndarray)
            assert media.ndim == 1


def test_iter_modality_dummies_covers_supported_modalities(mm_processor_stub):
    """Iter yields one ``(modality, size)`` per supported modality, inside budget.

    ``MultimodalEncoderBudget.iter_modality_dummies`` is the bridge
    between a processor and the per-encoder profiling loop. It must
    yield exactly one pair per modality the processor claims — and
    every yielded size must round-trip back through
    ``get_num_mm_tokens`` within the budget.
    """
    proc = mm_processor_stub.builder()
    budget = MultimodalEncoderBudget(max_tokens_per_step=4096, max_items_per_step=8)
    yielded = list(budget.iter_modality_dummies(proc))
    assert tuple(mod for mod, _ in yielded) == tuple(proc.supported_modalities)
    for modality, size in yielded:
        assert isinstance(size, dict)
        tokens = proc.get_num_mm_tokens(modality, **size)
        assert tokens <= budget.max_tokens_per_step
