# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Modality enum shared across multimodal input handling.

Used by:

* :class:`tensorrt_llm.inputs.registry.BaseMultimodalInputProcessor` —
  ``supported_modalities`` declaration and modality dispatch in
  ``get_num_mm_tokens`` / ``get_size_with_most_features``.
* :class:`tensorrt_llm._torch.pyexecutor.multimodal_budget.MultimodalEncoderBudget`
  — iterating per-modality dummies for KV-cache profiling.

The string value of each member matches the convention used by HF
processor outputs (``"image"``, ``"video"``, ``"audio"``) so the enum
can be substituted for raw modality strings at integration points
without an extra translation layer.
"""

from enum import Enum


class Modality(str, Enum):
    """Categories of non-text input handled by multimodal encoders.

    Extension contract for adding a new modality (e.g. point cloud,
    robotics actions):

    1. Add a member here whose string value matches the corresponding
       HF processor / data-dict key.
    2. Implement the modality in concrete
       :class:`BaseMultimodalInputProcessor` subclasses by handling the
       new member in ``get_num_mm_tokens`` / ``get_size_with_most_features``
       dispatch, and add it to ``supported_modalities``.
    3. Teach :meth:`BaseMultimodalDummyInputsBuilder.get_dummy_media`
       (or a model-specific override) how to fabricate a dummy instance
       for the new modality.

    Order below is by current production frequency.
    """

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
