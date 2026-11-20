# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item scheduling needs processor geometry for admission and profiling."""

import pytest
from _torch.multimodal.conformance.models import ModelEntry, inventory, inventory_or_none

from tensorrt_llm.inputs.registry import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
)

pytestmark = pytest.mark.cpu_only


def _entries() -> list:
    entries = inventory_or_none()
    if entries is None:
        # The test builds the inventory again and fails with its error.
        return [pytest.param(None, id="inventory")]
    return [
        pytest.param(entry, id=f"{arch}/{target}" if target else arch)
        for (arch, target), entry in entries.items()
        if getattr(entry.model_cls, "supports_mm_encoder_item_scheduling", False)
    ]


@pytest.mark.parametrize("entry", _entries())
def test_item_scheduling_processor(entry: ModelEntry | None) -> None:
    if entry is None:
        inventory()
        pytest.fail("the inventory failed while the tests were collected")
    assert entry.processor_classes, "P0 item scheduling requires a registered processor"
    for processor in entry.processor_classes:
        for default in (
            BaseMultimodalInputProcessor.get_mm_item_metadata,
            BaseMultimodalDummyInputsBuilder.get_mm_max_tokens_per_item,
            BaseMultimodalDummyInputsBuilder.get_dummy_mm_data,
            BaseMultimodalDummyInputsBuilder.get_max_mm_encoder_output_embeddings,
            BaseMultimodalDummyInputsBuilder.get_mm_encoder_attention_metadata_capacity,
        ):
            hook = getattr(processor, default.__name__, None)
            assert callable(hook) and hook is not default, (
                f"P0 item scheduling requires {processor.__name__} to implement {default.__name__}"
            )
