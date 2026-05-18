# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the multimodal input-processor common sweep.

The sweep in ``test_multimodal_processor_common.py`` is parametrized
off :data:`MM_PROCESSOR_STUBS`. Adding a new multimodal model to the
sweep requires only:

1. Implement :class:`BaseMultimodalInputProcessor` (and
   :class:`BaseMultimodalDummyInputsBuilder` for dummy sizing) on the
   model's processor base class.
2. Provide a stub builder — for an existing family, reuse the
   internal helper (e.g. ``_build_qwen_vl_stub``); for a new family,
   add a sibling ``_build_<family>_stub`` helper below.
3. Append one :class:`MMProcessorStub` row to :data:`MM_PROCESSOR_STUBS`.

Per-model math correctness is exercised end-to-end by integration
tests that load the real model — this sweep covers only the
cross-model invariants every opt-in processor must satisfy.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Tuple, Type

import pytest

from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase
from tensorrt_llm.inputs.modality import Modality
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor


@dataclass(frozen=True)
class MMProcessorStub:
    """One row in the common-sweep registry.

    ``builder`` returns a fresh processor instance per test (the sweep
    mutates nothing, but a frozen factory keeps state isolation cheap
    as we add more invariants).

    ``expected_modalities`` is asserted against
    :attr:`BaseMultimodalInputProcessor.supported_modalities` so silent
    drift (e.g. a refactor that drops VIDEO from Qwen3-VL) gets caught.
    """

    id: str
    processor_cls: Type[BaseMultimodalInputProcessor]
    builder: Callable[[], BaseMultimodalInputProcessor]
    expected_modalities: Tuple[Modality, ...]


def _build_qwen_vl_stub(
    processor_cls: Type[BaseMultimodalInputProcessor],
) -> BaseMultimodalInputProcessor:
    """Construct a Qwen-VL family processor instance with stubbed vision_config.

    Bypasses ``__init__`` (which would load HF tokenizer + image
    processor) and pins just the ``vision_config`` attrs the
    deterministic math reads.
    """
    instance = processor_cls.__new__(processor_cls)
    instance._config = SimpleNamespace(
        vision_config=SimpleNamespace(
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
    )
    return instance


# To onboard a new multimodal model: append one MMProcessorStub here.
# The full common sweep then runs against it automatically.
MM_PROCESSOR_STUBS: List[MMProcessorStub] = [
    MMProcessorStub(
        id="qwen2vl",
        processor_cls=Qwen2VLInputProcessorBase,
        builder=lambda: _build_qwen_vl_stub(Qwen2VLInputProcessorBase),
        expected_modalities=(Modality.IMAGE, Modality.VIDEO),
    ),
    MMProcessorStub(
        id="qwen3vl",
        processor_cls=Qwen3VLInputProcessorBase,
        builder=lambda: _build_qwen_vl_stub(Qwen3VLInputProcessorBase),
        expected_modalities=(Modality.IMAGE, Modality.VIDEO),
    ),
]


@pytest.fixture(params=MM_PROCESSOR_STUBS, ids=lambda stub: stub.id)
def mm_processor_stub(request) -> MMProcessorStub:
    """Parametrized fixture yielding one stub per registered MM processor."""
    return request.param
