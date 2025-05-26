import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

import aiohttp
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoProcessor


def _load_and_convert_image(image):
    image = Image.open(image)
    image.load()
    return image.convert("RGB")


def load_image(image: str,
               format: str = "pt",
               device: str = "cuda") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        image = requests.get(image, stream=True, timeout=10).raw
        image = _load_and_convert_image(image)
    elif parsed_url.scheme == "data":
        data_spec, data = parsed_url.path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        content = base64.b64decode(data)
        image = _load_and_convert_image(BytesIO(content))
    else:
        image = _load_and_convert_image(image)

    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


async def async_load_image(
        image: str,
        format: str = "pt",
        device: str = "cuda") -> Union[Image.Image, torch.Tensor]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    parsed_url = urlparse(image)

    if parsed_url.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            async with session.get(image) as response:
                content = await response.read()
                image = _load_and_convert_image(BytesIO(content))
    elif parsed_url.scheme == "data":
        data_spec, data = parsed_url.path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        content = base64.b64decode(data)
        image = _load_and_convert_image(BytesIO(content))
    else:
        image = _load_and_convert_image(Path(parsed_url.path))

    if format == "pt":
        return ToTensor()(image).to(device=device)
    else:
        return image


def load_video(
        video: str,
        num_frames: int = 10,
        format: str = "pt",
        device: str = "cuda") -> Union[List[Image.Image], List[torch.Tensor]]:

    # Keep this import local to avoid importing cv2 if not needed
    import cv2

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


async def async_load_video(
        video: str,
        num_frames: int = 10,
        format: str = "pt",
        device: str = "cuda") -> Union[List[Image.Image], List[torch.Tensor]]:
    assert format in ["pt", "pil"], "format must be either Pytorch or PIL"

    parsed_url = urlparse(video)

    if parsed_url.scheme in ["http", "https"]:
        async with aiohttp.ClientSession() as session:
            async with session.get(video) as response:
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix='.mp4') as tmp:
                    tmp.write(await response.content.read())
                    video_path = tmp.name
    # TODO: add case for video encoded in base64
    else:
        video_path = video

    return load_video(video_path, num_frames, format, device)


# Copied from https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client_for_multimodal.py#L38
def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url, timeout=10) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


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


def format_generic_input(model_dir, inputs):
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
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    *[{
                        "type": "image"
                    } for _ in multimodal_data["image"]],
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
        return processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    for input in inputs:
        input["prompt"] = apply_template(input["prompt"],
                                         input["multi_modal_data"])
    return inputs


def default_image_loader(prompts: List[str],
                         images: Union[List[List[str]], List[str]],
                         image_data_format: str = "pt",
                         device: str = "cuda"):
    if len(images) > len(prompts) and len(prompts) == 1:
        # 1 prompt + N media
        images = [images]
    assert len(images) == len(prompts)
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "image": [
                load_image(i, format=image_data_format, device=device)
                for i in image
            ] if isinstance(image, list) else
            [load_image(image, format=image_data_format, device=device)]
        }
    } for prompt, image in zip(prompts, images)]
    return inputs


def default_video_loader(prompts: List[str],
                         videos: Union[List[List[str]], List[str]],
                         image_data_format: str = "pt",
                         device: str = "cuda",
                         num_frames: int = 8):
    if len(videos) > len(prompts) and len(prompts) == 1:
        # 1 prompt + N media
        videos = [videos]
    assert len(videos) == len(prompts)
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": {
            "video": [
                load_video(
                    i, num_frames, format=image_data_format, device=device)
                for i in video
            ] if isinstance(video, list) else [
                load_video(
                    video, num_frames, format=image_data_format, device=device)
            ]
        }
    } for prompt, video in zip(prompts, videos)]
    return inputs


INPUT_FORMATTER_MAP = {
    "llava_llama": format_vila_input,
    "llava_next": format_generic_input,
    "qwen2_vl": format_qwen2_vl_input,
    "qwen2_5_vl": format_qwen2_vl_input,
    "llama4": format_generic_input,
}
