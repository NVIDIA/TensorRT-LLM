from typing import List, Union

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoProcessor


def load_image(image: str,
               format: str = "pt",
               device: str = "cuda") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    if image.startswith("http://") or image.startswith("https://"):
        image = Image.open(requests.get(image, stream=True, timeout=10).raw)
    else:
        image = Image.open(image)
    image = image.convert("RGB")
    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


def load_video(
        video: str,
        num_frames: int = 10,
        format: str = "pt",
        device: str = "cuda") -> Union[List[Image.Image], List[torch.Tensor]]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video)

    if not vidcap.isOpened():
        raise ValueError(
            f"Video '{video}' could not be opened. Make sure opencv is installed with video support."
        )

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video}' has no frames.")

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = Image.fromarray(frame)

    return [
        ToTensor()(frames[index]).to(
            device=device) if format == "pt" else frames[index]
        for index in indices if index in frames
    ]


"""
VLM input preparation.
"""


def format_vila_input(model_dir, inputs):
    """
    This function formats the input for the VILA/NVILA VL model.

    Arguments:
        model_dir: The directory of the model to load any preprocessor.
        inputs: The list of inputs to format.

    Returns:
        A list of dictionaries where "prompt" data is modified to a TextPrompt that combines text prompt and multimodal data.
    """

    def add_media_token(prompt, multi_modal_data):
        mm_tokens = ""
        if "image" in multi_modal_data:
            for _ in multi_modal_data["image"]:
                mm_tokens += "<image>"
        elif "video" in multi_modal_data:
            for _ in multi_modal_data["video"]:
                mm_tokens += "<vila/video>"
        return mm_tokens + prompt

    for input in inputs:
        input["prompt"] = add_media_token(input["prompt"],
                                          input["multi_modal_data"])
    return inputs


def format_llava_next_input(model_dir, inputs):
    """
    This function formats the input for the Llava Next VL model.

    Arguments:
        model_dir: The directory of the model to load any preprocessor.
        inputs: The list of inputs to format.

    Returns:
        A list of dictionaries where "prompt" data is modified to a TextPrompt that combines text prompt and multimodal data.
    """
    processor = AutoProcessor.from_pretrained(model_dir)

    # Single-image inference chat template. For multi-image template,
    # see https://huggingface.co/docs/transformers/en/model_doc/llava_next#multi-image-inference.
    def apply_template(prompt, multimodal_data):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image"
                    },
                ],
            },
        ]
        return processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

    for input in inputs:
        input["prompt"] = apply_template(input["prompt"],
                                         input["multi_modal_data"])
    return inputs


def format_qwen2_vl_input(model_dir, inputs):
    """
    This function formats the input for the Qwen2/Qwen2.5 VL model.

    Arguments:
        model_dir: The directory of the model to load any preprocessor.
        inputs: The list of inputs to format.

    Returns:
        A list of dictionaries where "prompt" data is modified to a TextPrompt that combines text prompt and multimodal data.
    """
    processor = AutoProcessor.from_pretrained(model_dir)

    def apply_template(prompt, multimodal_data):
        content = [{
            "type": media_type
        } for media_type, items in multimodal_data.items()
                   for _ in items] + [{
                       "type": "text",
                       "text": prompt
                   }]

        conversation = [{"role": "user", "content": content}]
        # print(conversation)
        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    for input in inputs:
        input["prompt"] = apply_template(input["prompt"],
                                         input["multi_modal_data"])
    return inputs


def default_image_loader(prompts, images, image_data_format="pt"):
    if len(images) > len(prompts) and len(prompts) == 1:
        # 1 prompt + N media
        images = [images]
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "image": [
                load_image(i, format=image_data_format, device="cuda")
                for i in image
            ] if isinstance(image, list) else
            [load_image(image, format=image_data_format, device="cuda")]
        }
    } for prompt, image in zip(prompts, images)]
    return inputs


def default_video_loader(prompts, videos, image_data_format="pt", num_frames=8):
    if len(videos) > len(prompts) and len(prompts) == 1:
        # 1 prompt + N media
        videos = [videos]
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "video": [
                load_video(
                    i, num_frames, format=image_data_format, device="cuda")
                for i in video
            ] if isinstance(video, list) else [
                load_video(
                    video, num_frames, format=image_data_format, device="cuda")
            ]
        }
    } for prompt, video in zip(prompts, videos)]
    return inputs


INPUT_FORMATTER_MAP = {
    "llava_llama": format_vila_input,
    "llava_next": format_llava_next_input,
    "qwen2_vl": format_qwen2_vl_input,
    "qwen2_5_vl": format_qwen2_vl_input,
}
