# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_llava_next import \
    LlavaNextInputProcessor
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("with_grid", [False, True], ids=["tensor", "grid"])
def test_llava_next_embeddings_keep_tensor_list(with_grid):
    """Models without mRoPE take a plain list of tensors and ignore a grid."""
    processor = object.__new__(LlavaNextInputProcessor)
    processor._config = SimpleNamespace(
        image_token_index=9, text_config=SimpleNamespace(hidden_size=4))
    embeddings = [torch.ones(3, 4), torch.zeros(2, 4)]
    items = embeddings
    if with_grid:
        # trtllm-serve passes an image_embeds part with a grid in this form.
        items = [{
            "mm_embeddings": embedding,
            "image_grid_thw": [1, 2, 2]
        } for embedding in embeddings]

    ids, extra = processor.attach_multimodal_embeddings(
        {"prompt_token_ids": [1, 9, 2, 9, 3]}, {"image": items},
        SamplingParams())

    assert ids == [1, 9, 9, 9, 2, 9, 9, 3]
    torch.testing.assert_close(extra["multimodal_data"]["multimodal_embedding"],
                               torch.cat(embeddings))


if __name__ == "__main__":
    pytest.main([__file__])
