# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pytest
import torch
from _torch.pyexecutor.multimodal_utils import bare_mm_item_scheduler, make_llm_request

from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration
from tensorrt_llm._torch.models.modeling_mistral import Mistral3VLM
from tensorrt_llm._torch.pyexecutor.llm_request import initialize_multimodal_encoder_request
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
)


def _gemma4_mixed_requests():
    class Encoder(Gemma4ForConditionalGeneration):
        embedding_dtype = torch.float32

        def _get_image_features(self, pixel_values, image_position_ids, image_seq_lens):
            assert pixel_values.is_cuda and image_position_ids.is_cuda
            return pixel_values[:, 0, :1]

    model = Encoder.__new__(Encoder)
    torch.nn.Module.__init__(model)
    requests = []
    for index, modality in enumerate(("image", "image", "video"), 1):
        data = {
            modality: {
                "pixel_values": torch.full((1, 4, 2), float(index)),
                "image_position_ids": torch.zeros(1, 4, 2, dtype=torch.int64),
                "image_seq_lens": [4],
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata([(modality, 0)], [4], [1]),
        }
        requests.append((make_llm_request(index, data), 0))
    # Each request transfers pixels and positions; sequence lengths are host metadata.
    return model, requests, [32, 64] * 3


def _mistral_selected_image():
    class Encoder(Mistral3VLM):
        embedding_dtype = torch.float32

        def _get_image_features(self, pixel_values, image_sizes):
            assert pixel_values.is_cuda and pixel_values.shape == (1, 3, 2, 3)
            return pixel_values[:, 0, 0, :1]

    model = Encoder.__new__(Encoder)
    torch.nn.Module.__init__(model)
    model._vision_tower = model._multi_modal_projector = torch.nn.Identity()
    data = {
        "image": {"pixel_values": torch.ones(2, 3, 5, 5), "image_sizes": [[5, 5], [2, 3]]},
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
            [("image", 0), ("image", 1)], [25, 6], [1, 1]
        ),
    }
    # Only the second image's 3 x 2 x 3 float32 values cross the bus.
    return model, [(make_llm_request(1, data), 1)], [72]


def _check_transfers(case, tmp_path):
    model, requests, expected_bytes = case()
    scheduler = bare_mm_item_scheduler(model)
    scheduler.bytes_per_embedding = torch.float32.itemsize
    selected = {}
    for req, index in requests:
        initialize_multimodal_encoder_request(req, max_num_tokens=32)
        assert scheduler.encoder_cache.acquire(req.request_id, scheduler.bytes_per_embedding)
        req.py_mm_encoder_state.set_item_cache_key(index, req.request_id, ready=False)
        selected[req.request_id] = [index]

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        scheduler.forward_items([req for req, _ in requests], selected)
        torch.cuda.synchronize()

    trace_path = tmp_path / "transfers.json"
    trace.export_chrome_trace(str(trace_path))
    sizes = [
        event["args"]["bytes"]
        for event in json.loads(trace_path.read_text())["traceEvents"]
        if event.get("cat") == "gpu_memcpy" and "htod" in event["name"].lower()
    ]
    assert sorted(sizes) == sorted(expected_bytes)
    for req, _ in requests:
        assert scheduler.encoder_cache.get(req.request_id).item() == req.request_id


@pytest.mark.parametrize("case", [_gemma4_mixed_requests, _mistral_selected_image])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_encoding_preserves_per_request_transfers(case, tmp_path):
    # Kineto's first session can disable CUDA activities for the lifetime of a pytest worker.
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as process:
        process.submit(_check_transfers, case, tmp_path).result()
