# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tensorrt_llm import SamplingParams
from tensorrt_llm.inputs.multimodal_data import VideoData

from ._engine import shortcut_text_engine


@pytest.mark.parametrize("precision", ["nvfp4"])
def test_decoded_video_frames(model_root, precision, tmp_path):
    import cv2
    from transformers import AutoTokenizer

    from tensorrt_llm.inputs.media_io import VideoMediaIO
    from tensorrt_llm.inputs.registry import (
        create_input_processor,
        create_input_processor_with_hash,
    )

    model = model_root / precision
    if not model.is_dir():
        pytest.skip(f"Checkpoint unavailable: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    processor = create_input_processor(str(model), tokenizer, video_pruning_rate=0.5)
    text = {"prompt": "A photo of"}
    assert processor({**text, "multi_modal_data": {"video": []}}, SamplingParams()) == processor(
        text, SamplingParams()
    )
    frames = np.zeros((8, 56, 56, 3), dtype=np.uint8)
    frames[:, :, :, :] = np.arange(8)[:, None, None, None]
    clip = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 4, (56, 56))
    try:
        assert writer.isOpened()
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    video_options = processor.get_preferred_media_io_kwargs()["video"]
    decoded = VideoMediaIO.create(video_options, {}).load_bytes(clip.read_bytes())
    indices = decoded.metadata["frames_indices"]
    # Eight frames at 4 FPS, sampled by the shared reader at 2 FPS.
    assert len(decoded.frames) == len(indices) == 4
    request = {
        "prompt": "<nemo:video> Describe.",
        "multi_modal_data": {"video": [decoded]},
    }
    options = SamplingParams(add_special_tokens=False)
    plain = create_input_processor(str(model), tokenizer)
    raw_ids, _ = create_input_processor_with_hash(plain)(request, options)
    prompt = tokenizer.decode(raw_ids, skip_special_tokens=False)
    for n, index in enumerate(indices, 1):
        assert f"Frame {n} sampled at {index / 4:.2f} seconds" in prompt

    ids, extra = create_input_processor_with_hash(processor)(request, options)
    data = extra["multimodal_data"]["video"]
    patch = processor.config.image_token_id
    template = data["templates"][0]
    # Four 2x2 frames contribute 16 projected patches. EVS keeps half.
    assert template.count(patch) == 16
    assert ids.count(patch) == len(template) - 8
    patches = data["patches"].reshape(4, 16, -1)
    for actual, frame in zip(patches, decoded.frames, strict=True):
        expected, _ = processor.preprocess_image(frame)
        assert actual.equal(expected)


@pytest.mark.skip_less_device(4)
@pytest.mark.parametrize("precision", ["nvfp4"])
@pytest.mark.parametrize("rate", [0.0, 0.5], ids=["unpruned", "evs"])
def test_evs_conditioned_video(model_root, precision, rate, record_property):
    model = model_root / precision
    with shortcut_text_engine(
        model,
        disable_mm_encoder=False,
        multimodal_config={"video_pruning_rate": rate},
        max_batch_size=1,
        max_seq_len=2048,
        max_num_tokens=2048,
        cuda_graph_config=None,
        enable_chunked_prefill=False,
        kv_cache_config={
            "use_kv_cache_manager_v2": True,
            "enable_block_reuse": False,
        },
    ) as llm:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {
                        "type": "text",
                        "text": "Is the foreground subject a person or a cat? Answer with one word.",
                    },
                ],
            }
        ]
        prompt = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assets = Path(__file__).resolve().parents[6] / "examples/visual_gen"
        for name, filename in (
            ("cat", "cat_piano.png"),
            ("person", "serve/media/woman_skyline_original_720p.jpeg"),
        ):
            with Image.open(assets / filename) as source:
                image = source.convert("RGB")
                image.thumbnail((224, 224), Image.Resampling.LANCZOS)
                frame = np.asarray(image)
            frames = np.stack([np.roll(frame, shift, axis=1) for shift in range(8)])
            video = VideoData(frames=frames, metadata={"fps": 2.0, "total_num_frames": 8})
            sampling = SamplingParams(temperature=0, top_k=1, max_tokens=16)
            output = llm.generate(
                {"prompt": prompt, "multi_modal_data": {"video": [video]}},
                sampling,
            )
            reply = output.outputs[0]
            record_property(f"{name}_video", (reply.text, list(reply.token_ids)))
            labels = {"person", "woman", "human"} if name == "person" else {"cat", "kitten"}
            assert reply.text.strip().casefold().rstrip(".!") in labels, reply.text
