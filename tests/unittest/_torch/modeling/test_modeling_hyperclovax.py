# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for HyperCLOVA X Vision grouped encoder inputs.

The input processor turns one request's processor output into per-image
tensors, and the vision model encodes the crops of every image in the batch
through one EncoderGroup. The oracle is the previous payload and encode path,
copied verbatim below. The vision tower is a deterministic stub, and the
projector is the real C-Abstractor forward with 1x1 convolutions in place of
its timm RegStage blocks.
"""

import math
from itertools import chain
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_hyperclovax import (
    HCXVisionCAbstractor,
    HCXVisionForCausalLM,
    HCXVisionInputProcessor,
    HCXVisionModel,
    adaptive_anyres_postprocessing,
    anyres_postprocessing,
)
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
)

pytestmark = pytest.mark.cpu_only

# ---------------------------------------------------------------------------
# Oracle: the previous payload and encode path, copied verbatim.
# ---------------------------------------------------------------------------


def _main_compute_adaptive_params(
    pixel_values: Optional[List[List[torch.FloatTensor]]] = None,
    num_queries_vis_abstractors: Optional[List[List[int]]] = None,
    num_queries_vis_abstractors_slow: Optional[List[List[int]]] = None,
    image_sizes: Optional[List[List[List[int]]]] = None,
    is_videos: Optional[List[bool]] = None,
    first_last_frames_slows: Optional[List[bool]] = None,
) -> Tuple[List[int], List[int], List[List[int]], List[bool], List[List[int]]]:
    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors
    ), "All values in num_queries_vis_abstractors must be integers >= 0."

    assert all(
        all(isinstance(value, int) and value >= 0 for value in sublist)
        for sublist in num_queries_vis_abstractors_slow
    ), "All values in num_queries_vis_abstractors_slow must be integers >= 0."

    assert is_videos is not None

    is_first_images = []
    is_last_images = []
    for is_video in is_videos:
        for idx, is_video_item in enumerate(is_video):
            if idx == 0:
                is_first_images.append(True)
            else:
                is_first_images.append(False)
            if idx == len(is_video) - 1:
                is_last_images.append(True)
            else:
                is_last_images.append(False)

    num_queries_vis_abstractors = list(chain(*num_queries_vis_abstractors))
    num_queries_vis_abstractors_slow = list(chain(*num_queries_vis_abstractors_slow))
    image_sizes = list(chain(*image_sizes))
    is_videos = list(chain(*is_videos))
    first_last_frames_slows = list(chain(*first_last_frames_slows))

    use_slowfast = any([num_query > 0 for num_query in num_queries_vis_abstractors_slow])
    num_grids = [pixel_value.shape[0] for pixel_value in chain(*pixel_values)]
    num_grids = [0] + num_grids
    group_ids = []

    if use_slowfast:
        new_num_grids = [num_grids[0]]
        new_num_queries = []
        new_image_sizes = []
        new_is_videos = []

        for (
            num_query,
            num_query_slow,
            num_grid,
            image_size,
            is_video,
            first_last_frames_slow,
            is_first_image,
            is_last_image,
        ) in zip(
            num_queries_vis_abstractors,
            num_queries_vis_abstractors_slow,
            num_grids[1:],
            image_sizes,
            is_videos,
            first_last_frames_slows,
            is_first_images,
            is_last_images,
        ):
            if not first_last_frames_slow and num_query_slow > 0:
                assert is_video

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                new_num_grids.append(new_num_grids[-1] + 1)
                new_num_queries.append(num_query_slow)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                if num_grid >= 2:
                    new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                    new_num_queries.append(num_query)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)
                    this_group_ids.append(this_group_ids[-1] + 1)

                group_ids.append(this_group_ids)
            elif (
                first_last_frames_slow and num_query_slow > 0 and (is_first_image or is_last_image)
            ):
                assert is_video

                this_group_ids = [group_ids[-1][-1] + 1 if group_ids else 0]

                if num_grid == 1:
                    new_num_grids.append(new_num_grids[-1] + 1)
                    new_num_queries.append(num_query_slow)
                    new_image_sizes.append(image_size)
                    new_is_videos.append(is_video)

                if num_grid >= 2:
                    if is_first_image:
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    elif is_last_image:
                        new_num_grids.append(new_num_grids[-1] + num_grid - 1)
                        new_num_queries.append(num_query)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        new_num_grids.append(new_num_grids[-1] + 1)
                        new_num_queries.append(num_query_slow)
                        new_image_sizes.append(image_size)
                        new_is_videos.append(is_video)
                        this_group_ids.append(this_group_ids[-1] + 1)
                    else:
                        raise Exception("This case should not be reached.")
                group_ids.append(this_group_ids)
            else:
                new_num_grids.append(new_num_grids[-1] + num_grid)
                new_num_queries.append(num_query)
                new_image_sizes.append(image_size)
                new_is_videos.append(is_video)

                start_group_id = group_ids[-1][-1] + 1 if group_ids else 0
                group_ids.append([start_group_id])

        num_grids = new_num_grids
        num_queries_vis_abstractors = new_num_queries
        image_sizes = new_image_sizes
        is_videos = new_is_videos
    else:
        num_grids = [sum(num_grids[:i]) for i in range(1, len(num_grids) + 1)]
        group_ids = [[group_id] for group_id in range(len(is_videos))]

    return num_queries_vis_abstractors, num_grids, image_sizes, is_videos, group_ids


def _main_payload(preprocessed_image):
    multimodal_data = {}
    multimodal_data["image"] = {
        "pixel_values": torch.stack(preprocessed_image["pixel_values"][0], dim=0).to(
            torch.bfloat16
        ),
        "image_sizes": preprocessed_image.get("image_sizes", None),
        "is_videos": preprocessed_image.get("is_videos", None),
        "num_queries_vis_abstractors": preprocessed_image.get("num_queries_vis_abstractors", None),
        "num_queries_vis_abstractors_slow": preprocessed_image.get(
            "num_queries_vis_abstractors_slow", None
        ),
        "first_last_frames_slows": preprocessed_image.get("first_last_frames_slows", None),
    }
    return multimodal_data


def _main_parse_and_batch_multimodal_data(
    multimodal_params: List[MultimodalParams],
) -> Tuple[List[torch.Tensor], Dict[str, List[Any]]]:
    """Parse and batch multimodal data from MultimodalParams objects."""
    pixel_values = [
        list(torch.unbind(multimodal_param.multimodal_data["image"]["pixel_values"], dim=0))
        for multimodal_param in multimodal_params
    ]
    mm_extra_data = {
        key: [
            multimodal_param.multimodal_data["image"][key][0]
            for multimodal_param in multimodal_params
        ]
        for key in multimodal_params[0].multimodal_data["image"].keys()
    }
    return pixel_values, mm_extra_data


@torch.inference_mode()
def _main_forward(self, multimodal_params: List[MultimodalParams]):
    pixel_values, mm_extra_data = _main_parse_and_batch_multimodal_data(multimodal_params)
    image_sizes = mm_extra_data.get("image_sizes", None)
    is_videos = mm_extra_data.get("is_videos", None)
    num_queries_vis_abstractors = mm_extra_data.get("num_queries_vis_abstractors", None)
    num_queries_vis_abstractors_slow = mm_extra_data.get("num_queries_vis_abstractors_slow", None)
    first_last_frames_slows = mm_extra_data.get("first_last_frames_slows", None)

    len_pixel_values = [len(pixel_value) for pixel_value in pixel_values]
    concat_pixel_values = torch.cat(list(chain(*pixel_values)), dim=0)  # list of list of 4D Tensor

    n_chunks = 1
    total_len = concat_pixel_values.size(0)
    chunk_size = math.ceil(total_len / n_chunks) if total_len > 0 else 1
    image_forward_outs_chunks = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = concat_pixel_values[start:end]
        if chunk.size(0) < chunk_size:
            pad_size = chunk_size - chunk.size(0)
            dummy_shape = (pad_size,) + tuple(concat_pixel_values.shape[1:])
            dummy = torch.zeros(
                dummy_shape,
                dtype=concat_pixel_values.dtype,
                device=concat_pixel_values.device,
            )
            chunk = torch.cat([chunk, dummy], dim=0)
        attn_metadata = self.vision_model.prepare_attn_metadata(chunk.shape[0])
        if self.use_nth_layer == -1:
            self.vision_model.vision_model.post_layernorm = nn.Identity()
            outs = self.vision_model(chunk, attn_metadata=attn_metadata)
            outs = outs[:, self.visual_token_idx :]
        else:
            outs = self.vision_model(chunk, attn_metadata=attn_metadata)
            outs = outs[self.use_nth_layer][:, self.visual_token_idx :]
        image_forward_outs_chunks.append(outs)

    image_forward_outs = torch.cat(image_forward_outs_chunks, dim=0).to(
        image_forward_outs_chunks[0].dtype
    )

    if num_queries_vis_abstractors is None:
        assert num_queries_vis_abstractors_slow is None
        image_sizes = list(chain(*image_sizes))
        if is_videos is not None:
            is_videos = list(chain(*is_videos))
        group_ids = None
        image_forward_outs = self.mm_projector(image_forward_outs)
    else:
        (
            num_queries_vis_abstractors,
            num_grids,
            image_sizes,
            is_videos,
            group_ids,
        ) = _main_compute_adaptive_params(
            pixel_values,
            num_queries_vis_abstractors,
            num_queries_vis_abstractors_slow,
            image_sizes,
            is_videos,
            first_last_frames_slows,
        )
        image_forward_outs = self.mm_projector(
            image_forward_outs,
            num_queries_vis_abstractors=num_queries_vis_abstractors,
            num_grids=num_grids,
        )
    if self.anyres:
        split_sizes = [pixel_value.shape[0] for pixel_value in chain(*pixel_values)]
        if num_queries_vis_abstractors is None:
            image_features = anyres_postprocessing(
                image_forward_outs=image_forward_outs,
                split_sizes=split_sizes,
                image_sizes=image_sizes,
                num_queries_vis_abstractor=self.num_queries_vis_abstractor,
                unpad=self.unpad,
                is_videos=is_videos,
                patch_size=self.vision_model.config.patch_size,
                grid_size=self.vision_model.config.image_size,
                image_newline=self.image_newline,
                possible_resolutions=self.possible_resolutions,
            )
        else:
            image_features = adaptive_anyres_postprocessing(
                image_forward_outs=image_forward_outs,
                image_sizes=image_sizes,
                num_queries_vis_abstractors=num_queries_vis_abstractors,
                unpad=self.unpad,
                is_videos=is_videos,
                grid_size=self.vision_model.config.image_size,
                image_newline=self.image_newline,
                possible_resolutions=self.possible_resolutions,
                group_ids=group_ids,
            )
    else:
        if num_queries_vis_abstractors is None:
            image_features = [image_forward_out for image_forward_out in image_forward_outs]
        else:
            image_features = [
                image_forward_out.unsqueeze(0) for image_forward_out in image_forward_outs
            ]
    image_features = [
        image_features[sum(len_pixel_values[:i]) : sum(len_pixel_values[: i + 1])]
        for i in range(len(len_pixel_values))
    ]
    mm_embeds = [torch.cat(list(chain(image_feature)), dim=0) for image_feature in image_features]
    return mm_embeds


# ---------------------------------------------------------------------------
# A small CPU vision model
# ---------------------------------------------------------------------------

GRID_SIZE = 56  # crop side in pixels
PATCH_SIZE = 14  # 4 x 4 tower tokens per crop
VISION_HIDDEN = 8
PROJECTOR_HIDDEN = 8
LLM_HIDDEN = 12


class _StubVisionTower(nn.Module):
    """Per-crop patch embedding; returns per-layer hidden states like the tower."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch_size=PATCH_SIZE, image_size=GRID_SIZE)
        self.embed = nn.Conv2d(3, VISION_HIDDEN, PATCH_SIZE, stride=PATCH_SIZE)
        self.calls = []

    def prepare_attn_metadata(self, batch_size):
        return None

    def forward(self, pixel_values, attn_metadata=None):
        self.calls.append(pixel_values.shape[0])
        hidden = self.embed(pixel_values.float()).flatten(2).transpose(1, 2)
        return [hidden, torch.tanh(hidden), hidden]


def _projector() -> HCXVisionCAbstractor:
    """The real C-Abstractor forward, with 1x1 convolutions for the RegStage blocks."""
    projector = HCXVisionCAbstractor.__new__(HCXVisionCAbstractor)
    nn.Module.__init__(projector)
    projector.num_input_tokens = (GRID_SIZE // PATCH_SIZE) ** 2
    projector.output_hidden_size = LLM_HIDDEN
    projector.pos_emb = nn.Parameter(
        torch.randn(1, projector.num_input_tokens, VISION_HIDDEN) * 0.02
    )
    projector.prenorm = None
    projector.net = nn.Sequential(
        nn.Conv2d(VISION_HIDDEN, PROJECTOR_HIDDEN, 1),
        nn.Identity(),
        nn.Conv2d(PROJECTOR_HIDDEN, PROJECTOR_HIDDEN, 1),
    )
    projector.readout = nn.Sequential(
        nn.Linear(PROJECTOR_HIDDEN, LLM_HIDDEN), nn.SiLU(), nn.Linear(LLM_HIDDEN, LLM_HIDDEN)
    )
    projector.dtype = torch.float32
    return projector


@pytest.fixture
def vision_model() -> HCXVisionModel:
    torch.manual_seed(0)
    model = HCXVisionModel.__new__(HCXVisionModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        anyres=True,
        max_num_grids=4,
        use_1x1_grid=True,
        vision_config=SimpleNamespace(image_size=GRID_SIZE),
    )
    model.vision_model = _StubVisionTower()
    model.mm_projector = _projector()
    model.image_newline = nn.Parameter(torch.randn(LLM_HIDDEN))
    model.unpad = True
    model.use_nth_layer = -2
    model.anyres = True
    model.visual_token_idx = 0
    model.possible_resolutions = HCXVisionModel._init_possible_resolutions(model)
    return model


# ---------------------------------------------------------------------------
# Requests: one processor output each, as the input processor receives it
# ---------------------------------------------------------------------------


def _crop_count(model: HCXVisionModel, image_size: List[int]) -> int:
    """Crops of one still image: its anyres grid plus the global view."""
    from tensorrt_llm._torch.models.modeling_hyperclovax import (
        select_best_resolution,  # noqa: PLC0415
    )

    width, height = image_size
    best_height, best_width = select_best_resolution((height, width), model.possible_resolutions)
    return (best_height // GRID_SIZE) * (best_width // GRID_SIZE) + 1


def _processor_output(
    crops: List[torch.Tensor],
    image_sizes: List[List[int]],
    vision_query_lengths: List[int],
    *,
    is_video: bool = False,
    num_queries: int = 4,
    num_queries_slow: int = 0,
    first_last_frames_slow: bool = False,
) -> Dict[str, Any]:
    """One request's processor output: one list per batch sample, sample 0."""
    count = len(crops)
    return {
        "pixel_values": [crops],
        "image_sizes": [image_sizes],
        "vision_query_lengths": [vision_query_lengths],
        "is_videos": [[is_video] * count],
        "num_queries_vis_abstractors": [[num_queries] * count],
        "num_queries_vis_abstractors_slow": [[num_queries_slow] * count],
        "first_last_frames_slows": [[first_last_frames_slow] * count],
    }


def _still_images(model: HCXVisionModel, image_sizes: List[List[int]], seed: int) -> Dict[str, Any]:
    """Still images of one request; their row counts come from the oracle."""
    torch.manual_seed(seed)
    crops = [torch.randn(_crop_count(model, size), 3, GRID_SIZE, GRID_SIZE) for size in image_sizes]
    rows = [
        _main_forward(
            model,
            [
                MultimodalParams(
                    multimodal_data=_main_payload(_processor_output([crop], [size], [0]))
                )
            ],
        )[0].shape[0]
        for crop, size in zip(crops, image_sizes)
    ]
    return _processor_output(crops, image_sizes, rows)


def _video_frames(num_frames: int, seed: int) -> Dict[str, Any]:
    """Video frames of one request, slow queries on its first and last frame.

    Each frame has two crops and no global view. A slow frame gives its first
    crop 4 queries and its second crop 1; any other frame gives 1 per crop.
    """
    torch.manual_seed(seed)
    crops = [torch.randn(2, 3, GRID_SIZE, GRID_SIZE) for _ in range(num_frames)]
    rows = [4 + 1 if index in (0, num_frames - 1) else 2 for index in range(num_frames)]
    return _processor_output(
        crops,
        [[GRID_SIZE, GRID_SIZE]] * num_frames,
        rows,
        is_video=True,
        num_queries=1,
        num_queries_slow=4,
        first_last_frames_slow=True,
    )


def _processed(processor_output: Dict[str, Any]) -> MultimodalParams:
    """Run the real input processor on a stubbed processor output."""
    processor = HCXVisionInputProcessor.__new__(HCXVisionInputProcessor)
    processor._preprocess = lambda text, images, kwargs: (
        torch.zeros(1, 4, dtype=torch.long),
        processor_output,
    )
    processor._post_process = lambda input_ids, preprocessed: input_ids[0]
    count = len(processor_output["pixel_values"][0])
    _, extra = processor.call_with_text_prompt(
        {"prompt": "", "multi_modal_data": {"image": [object()] * count}}, None
    )
    return MultimodalParams(multimodal_data=extra["multimodal_data"])


def _main_params(processor_output: Dict[str, Any]) -> MultimodalParams:
    return MultimodalParams(multimodal_data=_main_payload(processor_output))


BATCHES = {
    "one_image": lambda model: [_still_images(model, [[56, 56]], seed=0)],
    "one_request_two_images": lambda model: [_still_images(model, [[112, 56], [100, 50]], seed=1)],
    "requests_with_different_crops": lambda model: [
        _still_images(model, [[56, 56]], seed=2),
        _still_images(model, [[60, 170], [50, 160]], seed=3),
        _still_images(model, [[112, 56]], seed=4),
    ],
    "video_frames_slow_first_and_last_per_request": lambda model: [
        _video_frames(3, seed=5),
        _video_frames(1, seed=6),
        _video_frames(2, seed=7),
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_processor_emits_per_image_host_tensors(vision_model):
    output = _still_images(vision_model, [[60, 170], [50, 160]], seed=0)
    data = _processed(output).multimodal_data
    image = data["image"]
    crops = output["pixel_values"][0]
    assert torch.equal(image["pixel_values"], torch.cat(crops).to(torch.bfloat16))
    assert image["num_grids"].tolist() == [crop.shape[0] for crop in crops]
    assert image["image_sizes"].tolist() == [[60, 170], [50, 160]]
    assert image["is_videos"].tolist() == [False, False]
    assert image["num_queries_vis_abstractors"].tolist() == [4, 4]
    assert image["num_queries_vis_abstractors_slow"].tolist() == [0, 0]
    assert image["first_last_frames_slows"].tolist() == [False, False]
    assert image["is_first_images"].tolist() == [True, False]
    assert image["is_last_images"].tolist() == [False, True]
    for name, value in image.items():
        assert value.device.type == "cpu", name
        assert value.shape[0] == (
            len(crops) if name != "pixel_values" else sum(crop.shape[0] for crop in crops)
        ), name
    metadata = data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert isinstance(metadata, MultimodalItemMetadata)
    assert metadata.item_refs == (("image", 0), ("image", 1))
    assert metadata.output_embedding_lengths == tuple(output["vision_query_lengths"][0])
    # Only the crops are staged to the device; the encoder reads the rest on
    # the host.
    assert HCXVisionForCausalLM.multimodal_data_device_paths.fget(None) == [
        "image.pixel_values",
        "multimodal_embedding",
    ]


@pytest.mark.parametrize("batch", list(BATCHES))
def test_groups_match_the_previous_encode_path(vision_model, batch):
    outputs = BATCHES[batch](vision_model)
    expected = torch.cat(_main_forward(vision_model, [_main_params(output) for output in outputs]))
    params = [_processed(output) for output in outputs]
    vision_model.vision_model.calls.clear()

    (encoded,) = vision_model.forward(params)

    # One tower call over every crop of the batch, as before.
    assert vision_model.vision_model.calls == [
        sum(crop.shape[0] for output in outputs for crop in output["pixel_values"][0])
    ]
    assert encoded.shape[0] == sum(sum(output["vision_query_lengths"][0]) for output in outputs)
    assert torch.equal(encoded, expected)


def test_batched_rows_equal_per_request_rows(vision_model):
    """Each request's rows match encoding it alone, up to conv batch blocking."""
    params = [
        _processed(output) for output in BATCHES["requests_with_different_crops"](vision_model)
    ]
    (batched,) = vision_model.forward(params)
    alone = torch.cat([vision_model.forward([param])[0] for param in params])
    torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-5)
