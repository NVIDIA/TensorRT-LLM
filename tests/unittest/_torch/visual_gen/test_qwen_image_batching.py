# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical-shape and graph-cache contracts for Qwen request batching."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.batching import BatchInfo
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenImagePipeline
from tensorrt_llm._torch.visual_gen.models.qwen_image.pipeline_qwen_image_edit import (
    QwenImageEditPlusPipeline,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image.request_batch import (
    QwenImageBatchLayout,
    collate_prompt_encodings,
)
from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
    QwenImageTransformer2DModel,
)

pytestmark = pytest.mark.cpu_only


def test_prompt_encoding_preserves_each_request_padding():
    pipe = QwenImagePipeline.__new__(QwenImagePipeline)
    calls = []

    def encode(prompts, device, max_sequence_length):
        calls.append(prompts)
        length = max(map(len, prompts))
        embeddings = torch.ones(len(prompts), length, 4)
        mask = torch.tensor([[i < len(p) for i in range(length)] for p in prompts])
        return embeddings, None if mask.all() else mask

    pipe._encode_prompt = encode
    embeddings, mask, layout = pipe._encode_prompt_for_requests(
        ["ab", "c", "def"], torch.device("cpu"), 64, BatchInfo((1, 2), (1, 2), (42, 17))
    )
    assert calls == [["ab"], ["c", "def"]]
    assert embeddings.shape == (3, 3, 4)
    assert layout == QwenImageBatchLayout((1, 2), (2, 3), (True, False))
    assert mask.tolist() == [[True, True, False], [True, False, False], [True, True, True]]


def test_collation_preserves_explicit_all_valid_mask():
    a = torch.ones(1, 2, 4)
    _, mask, layout = collate_prompt_encodings(
        [(a, None), (a, torch.ones(1, 2, dtype=torch.bool))], (1, 1)
    )
    assert mask.all()
    assert layout.mask_is_none == (True, False)
    _, mask, _ = collate_prompt_encodings([(a, None), (a, None)], (1, 1))
    assert mask is None


@pytest.mark.parametrize("packed", [False, True])
def test_attention_matches_original_shapes_masks_and_strides(packed):
    torch.manual_seed(0)
    layout = QwenImageBatchLayout((1, 2), (2, 3), (True, False))
    # Three padded text tokens followed by two image tokens.
    q, k, v = torch.randn(3, 5, 24).split(8, dim=-1)
    if not packed:
        q, k, v = [x.contiguous() for x in (q, k, v)]
    mask = torch.tensor([[1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 1, 1]]).bool()
    calls = []

    def attention(q, k, v, *, timestep, key_padding_mask=None):
        calls.append((q.shape, q.stride(1), key_padding_mask, timestep))
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if key_padding_mask is None else key_padding_mask[:, None]
        )

    timesteps = torch.tensor([7, 8, 9])
    actual = layout.attention(attention, q, k, v, attention_mask=mask, timestep=timesteps)
    first = [torch.cat([x[:1, :2], x[:1, 3:]], dim=1) for x in (q, k, v)]
    expected_first = F.scaled_dot_product_attention(*first)
    expected_second = F.scaled_dot_product_attention(q[1:], k[1:], v[1:], attn_mask=mask[1:, None])
    torch.testing.assert_close(actual[:1, :2], expected_first[:, :2])
    torch.testing.assert_close(actual[:1, 3:], expected_first[:, 2:])
    torch.testing.assert_close(actual[1:], expected_second)
    assert actual[:1, 2].count_nonzero() == 0
    assert [c[0] for c in calls] == [torch.Size((1, 4, 8)), torch.Size((2, 5, 8))]
    assert [c[1] for c in calls] == [24 if packed else 8] * 2
    assert calls[0][2] is None
    assert torch.equal(calls[1][2], mask[1:])
    assert calls[0][3].tolist() == [7]
    assert calls[1][3].tolist() == [8, 9]


def test_sensitive_projections_receive_original_shapes():
    layout = QwenImageBatchLayout((1, 2), (2, 3), (True, True))
    shapes = []

    def projection(value):
        shapes.append(tuple(value.shape))
        assert value.is_contiguous()
        return value + 1

    text = torch.zeros(3, 3, 4)
    result = layout.apply_to_text(projection, text)
    assert shapes == [(1, 2, 4), (2, 3, 4)]
    assert result[0, 2].count_nonzero() == 0
    shapes.clear()
    image = torch.zeros(3, 5, 4)
    assert torch.equal(layout.apply_to_images(projection, image), image + 1)
    assert shapes == [(1, 5, 4), (2, 5, 4)]


def test_graph_keys_distinguish_request_layout_at_same_tensor_shape():
    model = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    torch.nn.Module.__init__(model)
    model.model_config = SimpleNamespace(attention=SimpleNamespace(sparse_attention_config=None))
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    assert runner.get_graph_key() == ()
    assert runner._extra_key_capture_limits == {"request_batch": 16}
    layouts = [
        QwenImageBatchLayout((1, 2), (2, 3), (True, True)),
        QwenImageBatchLayout((2, 1), (2, 3), (True, True)),
        QwenImageBatchLayout((1, 2), (3, 2), (True, True)),
        QwenImageBatchLayout((1, 2), (2, 3), (False, True)),
    ]
    assert len({runner.get_graph_key(request_batch=layout) for layout in layouts}) == len(layouts)


@pytest.mark.parametrize(
    "world_size,backend,quant,quant_dict,dynamic,expected",
    [
        (1, "VANILLA", None, {}, False, True),
        (2, "VANILLA", None, {}, False, False),
        (1, "FLASH_ATTN", None, {}, False, False),
        (1, "VANILLA", "FP8", {}, False, False),
        (1, "VANILLA", None, {"transformer": "FP8"}, False, False),
        (1, "VANILLA", None, {}, True, False),
    ],
)
def test_batching_capability_scope(world_size, backend, quant, quant_dict, dynamic, expected):
    pipe = QwenImagePipeline.__new__(QwenImagePipeline)
    pipe.pipeline_config = SimpleNamespace(
        mapping=SimpleNamespace(world_size=world_size),
        attention=SimpleNamespace(backend=backend),
        quant_config=SimpleNamespace(quant_algo=quant),
        quant_config_dict=quant_dict,
        dynamic_weight_quant=dynamic,
    )
    assert pipe.supports_dynamic_batching is expected
    assert not QwenImageEditPlusPipeline.supports_dynamic_batching


def test_latents_match_original_request_rng_shapes():
    pipe = QwenImagePipeline.__new__(QwenImagePipeline)
    pipe.vae_scale_factor = 8
    kwargs = dict(
        num_channels_latents=16,
        height=64,
        width=64,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    expected = torch.cat(
        [
            pipe._prepare_latents(
                batch_size=count, generator=torch.Generator().manual_seed(seed), **kwargs
            )
            for count, seed in [(2, 42), (1, 17)]
        ]
    )
    actual = pipe._prepare_latents(
        batch_size=3,
        generator=torch.Generator().manual_seed(0),
        batch_info=BatchInfo((1, 2), (2, 1), (42, 17)),
        **kwargs,
    )
    assert torch.equal(actual, expected)
