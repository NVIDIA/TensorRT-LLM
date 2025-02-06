# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from enum import Enum, auto
from typing import List, Union

import cv2
import numpy as np
import requests
from PIL import Image
from transformers import AutoConfig

from tensorrt_llm._torch.models import get_model_architecture
from tensorrt_llm._torch.models.modeling_vila import MEDIA_TOKENS
from tensorrt_llm._torch.models.modeling_vila import \
    init_tokenizer as init_vila_tokenizer
from tensorrt_llm.inputs import TextPrompt, TokensPrompt
from tensorrt_llm.logger import logger

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".webm"]
"""
[VILA] Data loading and prompt template utilities
Based on https://github.com/NVlabs/VILA/llava/utils/
"""


def load_image(image: str) -> Image:
    if image.startswith("http://") or image.startswith("https://"):
        logger.info(f"Downloading image from {image}")
        image = Image.open(requests.get(image, stream=True).raw)
    else:
        image = Image.open(image)
    return image.convert("RGB")


def load_video(video: str, num_frames: int = None) -> List[Image.Image]:
    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video)

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
            logger.warning(
                f"Failed to read frame {index} from video '{video}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]


class VilaFormatter:
    """
    VILA conversation utilities
    based on https://github.com/NVlabs/VILA/llava/conversation.py
    """

    class SeparatorStyle(Enum):
        """Different separator style."""

        AUTO = auto()
        TWO = auto()
        MPT = auto()
        PLAIN = auto()
        LLAMA_3 = auto()

    @dataclasses.dataclass
    class Conversation:
        """A class that keeps all conversation history."""
        system: str
        roles: List[str]
        messages: List[List[str]]
        sep_style: Enum
        sep: str = "###"
        sep2: str = None
        version: str = "Unknown"

        def get_prompt(self):
            messages = self.messages
            if len(messages) > 0 and type(messages[0][1]) is tuple:
                messages = self.messages.copy()
                init_role, init_msg = messages[0].copy()
                init_msg = init_msg[0].replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)

            if self.sep_style == VilaFormatter.SeparatorStyle.TWO:
                seps = [self.sep, self.sep2]
                ret = self.system + seps[0]
                for i, (role, message) in enumerate(messages):
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += role + ": " + message + seps[i % 2]
                    else:
                        ret += role + ":"
            elif self.sep_style == VilaFormatter.SeparatorStyle.LLAMA_3:
                ret = self.system + self.sep
                for rid, (role, message) in enumerate(messages):
                    if message:
                        if type(message) is tuple:
                            message = message[0]
                        sep = self.sep if rid < len(messages) - 1 else self.sep2
                        ret += role + message + sep
                    else:
                        ret += role
            elif self.sep_style == VilaFormatter.SeparatorStyle.MPT:
                ret = self.system + self.sep
                for role, message in messages:
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += role + message + self.sep
                    else:
                        ret += role
            elif self.sep_style == VilaFormatter.SeparatorStyle.PLAIN:
                seps = [self.sep, self.sep2]
                ret = self.system
                for i, (role, message) in enumerate(messages):
                    if message:
                        if type(message) is tuple:
                            message, _, _ = message
                        ret += message + seps[i % 2]
                    else:
                        ret += ""
            else:
                raise ValueError(f"Invalid style: {self.sep_style}")

            return ret

        def append_message(self, role, message):
            self.messages.append([role, message])

        def copy(self):
            return VilaFormatter.Conversation(
                system=self.system,
                roles=self.roles,
                messages=[[x, y] for x, y in self.messages],
                sep_style=self.sep_style,
                sep=self.sep,
                sep2=self.sep2,
                version=self.version,
            )

    conv_auto = Conversation(
        system="",
        roles=("", ""),
        messages=(),
        sep_style=SeparatorStyle.AUTO,
        sep="\n",
    )

    conv_vicuna_v1 = Conversation(
        system=
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        version="v1",
        messages=(),
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )

    conv_llava_plain = Conversation(
        system="",
        roles=("", ""),
        messages=(),
        sep_style=SeparatorStyle.PLAIN,
        sep="\n",
    )

    hermes_2 = Conversation(
        system="<|im_start|>system\nAnswer the questions.",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        messages=(),
        version="hermes-2",
    )

    # Template added by Yukang. Note (kentang-mit@): sep is <|eot_id|> for official template.
    llama_3_chat = Conversation(
        system=
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language.",
        roles=("<|start_header_id|>user<|end_header_id|>\n\n",
               "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        version="llama_v3",
        messages=(),
        sep_style=SeparatorStyle.LLAMA_3,
        sep="<|eot_id|>",
        sep2="<|end_of_text|>",
    )

    conv_templates = {
        "auto": conv_auto,
        "hermes-2": hermes_2,
        "llama_3": llama_3_chat,
        "v1": conv_vicuna_v1,
        "vicuna_v1": conv_vicuna_v1,
        "plain": conv_llava_plain,
    }

    CONVERSATION_MODE_MAPPING = {
        "vila1.5-3b": "vicuna_v1",
        "vila1.5-8b": "llama_3",
        "vila1.5-13b": "vicuna_v1",
        "vila1.5-40b": "hermes-2",
        "llama-3": "llama_3",
        "llama3": "llama_3",
    }

    @classmethod
    def format(cls, checkpoint_dir, inputs):

        def _auto_set_conversation_mode(
                model_name_or_path: str) -> VilaFormatter.Conversation:
            # CAVEAT: VILA uses pathname-based check which is error prone, consider register with model
            default_conversation = VilaFormatter.conv_auto
            for k, v in VilaFormatter.CONVERSATION_MODE_MAPPING.items():
                if k in model_name_or_path.lower():
                    logger.info(
                        f"Setting conversation mode to `{v}` based on model name/path `{model_name_or_path}`."
                    )
                    default_conversation = VilaFormatter.conv_templates[v]
                    break
            return default_conversation

        conv = _auto_set_conversation_mode(checkpoint_dir)
        tokenizer = init_vila_tokenizer(os.path.join(checkpoint_dir, "llm"))

        def _apply_template(text):
            text = text.strip()
            if conv.sep_style == VilaFormatter.SeparatorStyle.AUTO:
                # VILA 2.0
                message = {}
                message["role"] = "user"
                message["content"] = text
                text = tokenizer.apply_chat_template(
                    [message],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                # VILA 1.0
                messages = [{"from": "human", "value": text}]
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

                # Skip the first message if it is not from human
                if messages[0]["from"] != "human":
                    messages = messages[1:]

                # Add a generation prompt if needed
                messages.append({"from": "gpt", "value": None})

                conv.messages = []
                for turn, message in enumerate(messages):
                    role = roles[message["from"]]
                    assert role == conv.roles[turn % 2]
                    conv.append_message(role, message["value"])
                text = conv.get_prompt()
            return text

        # Prepare multimodal data & Reformat text prompt
        formatted = []
        for prompts in inputs:
            text, mm_text = "", ""
            data = []
            for p in prompts:
                if isinstance(p, str):
                    is_image_path = any(
                        p.endswith(ext) for ext in IMAGE_EXTENSIONS)
                    is_video_path = any(
                        p.endswith(ext) for ext in VIDEO_EXTENSIONS)

                    if not is_image_path and not is_video_path:
                        # plain text prompt
                        for token in MEDIA_TOKENS.values():
                            if token in p:
                                logger.warning(
                                    f"Multimodal token '{token}' found in text: '{p}'. Removed."
                                )
                                p = p.replace(token, "").strip()
                        text += p
                    elif is_image_path:
                        data.append(load_image(p))
                        mm_text += MEDIA_TOKENS["image"]
                    else:
                        data.append(load_video(p))
                        mm_text += MEDIA_TOKENS["video"]
                elif isinstance(p, Image.Image):
                    # image tensor input
                    data.append(p)
                    mm_text += MEDIA_TOKENS["image"]
                elif isinstance(p, list) and isinstance(p[0], int):
                    logger.warning(
                        "Skip token ID prompt case in multimodal for now.")
                else:
                    raise ValueError(f"Unsupported prompt type: {type(p)}")

                # VILA has an implicit "image/video tokens first" order for the prompt
                text = mm_text + text

            text = _apply_template(text)
            mm_processor_kwargs = {}

            formatted.append([text, data, mm_processor_kwargs])

        return formatted


"""End of VILA utils"""

MODEL_FORMATTER_MAPPING = {
    "VilaModel": VilaFormatter,
}


class AutoFormatter:

    @classmethod
    def format(
            cls, checkpoint_dir: str,
            inputs: List[List[str]]) -> List[Union[TextPrompt, TokensPrompt]]:
        """Format the input data for multimodal models.
        Args:
            checkpoint_dir (str): The directory of the model checkpoint.
            inputs (List[List[str]]): The input data to be formatted. A batch_size length list, in which each item is a length-2 list of [multimodal_prompt (in the form of filepaths or urls of image/video data), text_prompt]
        Returns:
            List[Union[TextPrompt, TokensPrompt]]: The formatted input data. A batch_size length list of Prompt objects.
        """
        config = AutoConfig.from_pretrained(checkpoint_dir,
                                            trust_remote_code=True)
        model_cls, _ = get_model_architecture(config)
        inputs = MODEL_FORMATTER_MAPPING[model_cls.__name__].format(
            checkpoint_dir, inputs)

        formatted = []
        for text, data, mm_process_kwargs in inputs:
            data = data if isinstance(data, list) else [data]
            if isinstance(text, str):
                input = TextPrompt(prompt=text,
                                   multi_modal_data=data,
                                   mm_processor_kwargs=mm_process_kwargs)
            elif isinstance(text, list):
                assert isinstance(text[0], int)
                input = TokensPrompt(prompt_token_ids=text,
                                     multi_modal_data=data,
                                     mm_processor_kwargs=mm_process_kwargs)
            else:
                raise ValueError(f"Unsupported prompt type: {type(text)}")
            formatted.append(input)
        return formatted
