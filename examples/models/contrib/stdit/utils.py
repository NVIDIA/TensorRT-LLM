# Copyright 2024 HPC-AI Technology Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# reference: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/utils/ckpt_utils.py

import html
import json
import os
import random
import re
import sys
import urllib.parse as ul

import ftfy
import numpy as np
import pandas as pd
import requests
import torch
import video_transforms
from aspect import get_image_size, get_num_frames
from bs4 import BeautifulSoup
from colossalai.checkpoint_io import GeneralCheckpointIO
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import read_video, write_video
from torchvision.utils import save_image

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

URL_REGEX = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)

HF_ENDPOINT = os.environ.get("HF_ENDPOINT")
if HF_ENDPOINT is None:
    HF_ENDPOINT = "https://huggingface.co"
PRETRAINED_MODELS = {
    "DiT-XL-2-512x512.pt":
    "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt":
    "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt":
    HF_ENDPOINT + "/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
    "OpenSora-v1-16x256x256.pth":
    HF_ENDPOINT +
    "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth",
    "OpenSora-v1-HQ-16x256x256.pth":
    HF_ENDPOINT +
    "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth",
    "OpenSora-v1-HQ-16x512x512.pth":
    HF_ENDPOINT +
    "/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth",
    "PixArt-Sigma-XL-2-256x256.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-256x256.pth",
    "PixArt-Sigma-XL-2-512-MS.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-512-MS.pth",
    "PixArt-Sigma-XL-2-1024-MS.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-1024-MS.pth",
    "PixArt-Sigma-XL-2-2K-MS.pth":
    HF_ENDPOINT +
    "/PixArt-alpha/PixArt-Sigma/resolve/main/PixArt-Sigma-XL-2-2K-MS.pth",
}


def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def is_url(url):
    return re.match(URL_REGEX, url) is not None


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def download_url(input_path):
    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    try:
        img_data = requests.get(input_path).content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading URL {input_path}: {e}")
        return None
    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def reparameter(ckpt, name=None, model=None):
    model_name = name
    name = os.path.basename(name)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        print("loading pretrained model: %s", model_name)
    if name in ["DiT-XL-2-512x512.pt", "DiT-XL-2-256x256.pt"]:
        ckpt["x_embedder.proj.weight"] = ckpt[
            "x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    if name in ["Latte-XL-2-256x256-ucf101.pt"]:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt[
            "x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    if name in [
            "PixArt-XL-2-256x256.pth",
            "PixArt-XL-2-SAM-256x256.pth",
            "PixArt-XL-2-512x512.pth",
            "PixArt-XL-2-1024-MS.pth",
            "PixArt-Sigma-XL-2-256x256.pth",
            "PixArt-Sigma-XL-2-512-MS.pth",
            "PixArt-Sigma-XL-2-1024-MS.pth",
            "PixArt-Sigma-XL-2-2K-MS.pth",
    ]:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt[
            "x_embedder.proj.weight"].unsqueeze(2)
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]
    if name in [
            "PixArt-1B-2.pth",
    ]:
        ckpt = ckpt["state_dict"]
        if "pos_embed" in ckpt:
            del ckpt["pos_embed"]
    # no need pos_embed
    if "pos_embed_temporal" in ckpt:
        del ckpt["pos_embed_temporal"]
    if "pos_embed" in ckpt:
        del ckpt["pos_embed"]
    # different text length
    if "y_embedder.y_embedding" in ckpt:
        if ckpt["y_embedder.y_embedding"].shape[
                0] < model.y_embedder.y_embedding.shape[0]:
            print(
                "Extend y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            additional_length = model.y_embedder.y_embedding.shape[0] - ckpt[
                "y_embedder.y_embedding"].shape[0]
            new_y_embedding = torch.zeros(additional_length,
                                          model.y_embedder.y_embedding.shape[1])
            new_y_embedding[:] = ckpt["y_embedder.y_embedding"][-1]
            ckpt["y_embedder.y_embedding"] = torch.cat(
                [ckpt["y_embedder.y_embedding"], new_y_embedding], dim=0)
        elif ckpt["y_embedder.y_embedding"].shape[
                0] > model.y_embedder.y_embedding.shape[0]:
            print(
                "Shrink y_embedding from %s to %s",
                ckpt["y_embedder.y_embedding"].shape[0],
                model.y_embedder.y_embedding.shape[0],
            )
            ckpt["y_embedder.y_embedding"] = ckpt[
                "y_embedder.y_embedding"][:model.y_embedder.y_embedding.
                                          shape[0]]
    # stdit3 special case
    if type(model).__name__ == "STDiT3" and "PixArt-Sigma" in name:
        ckpt_keys = list(ckpt.keys())
        for key in ckpt_keys:
            if "blocks." in key:
                ckpt[key.replace("blocks.", "spatial_blocks.")] = ckpt[key]
                del ckpt[key]
    return ckpt


def download_model(model_name=None, local_path=None, url=None):
    """
    Downloads a pre-trained DiT model from the web.
    """
    if model_name is not None:
        assert model_name in PRETRAINED_MODELS
        local_path = f"pretrained_models/{model_name}"
        web_path = PRETRAINED_MODELS[model_name]
    else:
        assert local_path is not None
        assert url is not None
        web_path = url
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        dir_name = os.path.dirname(local_path)
        file_name = os.path.basename(local_path)
        download_url(web_path, dir_name, file_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def find_model(model_name, model=None):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in PRETRAINED_MODELS:  # Find/download our pre-trained DiT checkpoints
        model_ckpt = download_model(model_name)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(
            model_name), f"Could not find DiT checkpoint at {model_name}"
        model_ckpt = torch.load(model_name,
                                map_location=lambda storage, loc: storage)
        model_ckpt = reparameter(model_ckpt, model_name, model=model)
    return model_ckpt


def load_from_sharded_state_dict(model,
                                 ckpt_path,
                                 model_name="model",
                                 strict=False):
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model,
                       os.path.join(ckpt_path, model_name),
                       strict=strict)


def load_checkpoint(model,
                    ckpt_path,
                    save_as_pt=False,
                    model_name="model",
                    strict=False):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path, model=model)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                              strict=strict)
        print("Missing keys: %s", missing_keys)
        print("Unexpected keys: %s", unexpected_keys)
    elif ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                              strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model,
                                     ckpt_path,
                                     model_name,
                                     strict=strict)
        print("Model checkpoint loaded from %s", ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, model_name + "_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            print("Model checkpoint saved to %s", save_path)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def print_progress_bar(iteration, total, length=40):
    iteration += 1
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\rDenoising steps: |{bar}| {iteration}/{total}')
    sys.stdout.flush()


class PromptProcessor():

    @staticmethod
    def load_prompts(prompt_path, start_idx=None, end_idx=None):
        with open(prompt_path, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
        prompts = prompts[start_idx:end_idx]
        return prompts

    @staticmethod
    def extract_json_from_prompts(prompts, reference, mask_strategy):
        ret_prompts = []
        for i, prompt in enumerate(prompts):
            parts = re.split(r"(?=[{])", prompt)
            assert len(parts) <= 2, f"Invalid prompt: {prompt}"
            ret_prompts.append(parts[0])
            if len(parts) > 1:
                additional_info = json.loads(parts[1])
                for key in additional_info:
                    assert key in ["reference_path",
                                   "mask_strategy"], f"Invalid key: {key}"
                    if key == "reference_path":
                        reference[i] = additional_info[key]
                    elif key == "mask_strategy":
                        mask_strategy[i] = additional_info[key]
        return ret_prompts, reference, mask_strategy

    @staticmethod
    def split_prompt(prompt_text):
        if prompt_text.startswith("|0|"):
            # this is for prompts which look like
            # |0| a beautiful day |1| a sunny day |2| a rainy day
            # we want to parse it into a list of prompts with the loop index
            prompt_list = prompt_text.split("|")[1:]
            text_list = []
            loop_idx = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1].strip()
                text_list.append(text)
                loop_idx.append(start_loop)
            return text_list, loop_idx
        else:
            return [prompt_text], None

    @staticmethod
    def merge_prompt(text_list, loop_idx_list=None):
        if loop_idx_list is None:
            return text_list[0]
        else:
            prompt = ""
            for i, text in enumerate(text_list):
                prompt += f"|{loop_idx_list[i]}|{text}"
            return prompt

    @staticmethod
    def extract_prompts_loop(prompts, num_loop):
        ret_prompts = []
        for prompt in prompts:
            if prompt.startswith("|0|"):
                prompt_list = prompt.split("|")[1:]
                text_list = []
                for i in range(0, len(prompt_list), 2):
                    start_loop = int(prompt_list[i])
                    text = prompt_list[i + 1]
                    end_loop = int(prompt_list[i + 2]) if i + 2 < len(
                        prompt_list) else num_loop + 1
                    text_list.extend([text] * (end_loop - start_loop))
                prompt = text_list[num_loop]
            ret_prompts.append(prompt)
        return ret_prompts

    @staticmethod
    def append_score_to_prompts(prompts,
                                aes=None,
                                flow=None,
                                camera_motion=None):
        new_prompts = []
        for prompt in prompts:
            new_prompt = prompt
            if aes is not None and "aesthetic score:" not in prompt:
                new_prompt = f"{new_prompt} aesthetic score: {aes:.1f}."
            if flow is not None and "motion score:" not in prompt:
                new_prompt = f"{new_prompt} motion score: {flow:.1f}."
            if camera_motion is not None and "camera motion:" not in prompt:
                new_prompt = f"{new_prompt} camera motion: {camera_motion}."
            new_prompts.append(new_prompt)
        return new_prompts

    @classmethod
    def basic_clean(cls, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    @classmethod
    def clean_caption(cls, caption):
        BAD_PUNCT_REGEX = re.compile(r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" +
                                     "\]" + "\[" + "\}" + "\{" + "\|" + "\\" +
                                     "\/" + "\*" + r"]{1,}")  # noqa
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text
        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)
        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )
        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)
        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)
        # ip addresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)
        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)
        # \n
        caption = re.sub(r"\\n", " ", caption)
        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)",
                         "", caption)
        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""
        caption = re.sub(BAD_PUNCT_REGEX, r" ",
                         caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "
        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)
        caption = cls.basic_clean(caption)
        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231
        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "",
            caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)
        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ",
                         caption)  # j2d1a2a...
        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)
        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)
        caption.strip()
        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)
        return caption.strip()

    @classmethod
    def text_preprocessing(cls, text, use_text_preprocessing: bool = True):
        if use_text_preprocessing:
            # The exact text cleaning as was in the training stage:
            text = cls.clean_caption(text)
            text = cls.clean_caption(text)
            return text
        else:
            return text.lower().strip()


class VideoProcessor():
    MASK_DEFAULT = ["0", "0", "0", "0", "1", "0"]

    @staticmethod
    def get_image_size(resolution, ar_ratio):
        return get_image_size(resolution=resolution, ar_ratio=ar_ratio)

    @staticmethod
    def get_num_frames(num_frames):
        return get_num_frames(num_frames=num_frames)

    @staticmethod
    def get_latent_size(vae, input_size):
        return vae.get_latent_size(input_size=input_size)

    @staticmethod
    def center_crop_arr(pil_image, image_size):
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                         resample=Image.BOX)

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(
            round(x * scale) for x in pil_image.size),
                                     resample=Image.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return Image.fromarray(arr[crop_y:crop_y + image_size,
                                   crop_x:crop_x + image_size])

    @staticmethod
    def resize_crop_to_fill(pil_image, image_size):
        w, h = pil_image.size  # PIL is (W, H)
        th, tw = image_size
        rh, rw = th / h, tw / w
        if rh > rw:
            sh, sw = th, round(w * rh)
            image = pil_image.resize((sw, sh), Image.BICUBIC)
            i = 0
            j = int(round((sw - tw) / 2.0))
        else:
            sh, sw = round(h * rw), tw
            image = pil_image.resize((sw, sh), Image.BICUBIC)
            i = int(round((sh - th) / 2.0))
            j = 0
        arr = np.array(image)
        assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
        return Image.fromarray(arr[i:i + th, j:j + tw])

    @classmethod
    def get_transforms_image(cls, name="center", image_size=(256, 256)):
        if name is None:
            return None
        elif name == "center":
            assert image_size[0] == image_size[
                1], "Image size must be square for center crop"
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: cls.center_crop_arr(
                    pil_image, image_size[0])),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5],
                                     inplace=True),
            ])
        elif name == "resize_crop":
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: cls.resize_crop_to_fill(
                    pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5],
                                     inplace=True),
            ])
        else:
            raise NotImplementedError(f"Transform {name} not implemented")
        return transform

    @classmethod
    def get_transforms_video(cls, name="center", image_size=(256, 256)):
        if name is None:
            return None
        elif name == "center":
            assert image_size[0] == image_size[
                1], "image_size must be square for center crop"
            transform_video = transforms.Compose([
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5],
                                     inplace=True),
            ])
        elif name == "resize_crop":
            transform_video = transforms.Compose([
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5],
                                     inplace=True),
            ])
        else:
            raise NotImplementedError(f"Transform {name} not implemented")
        return transform_video

    @classmethod
    def read_image_from_path(cls,
                             path,
                             transform=None,
                             transform_name="center",
                             num_frames=1,
                             image_size=(256, 256)):
        image = pil_loader(path)
        if transform is None:
            transform = cls.get_transforms_image(image_size=image_size,
                                                 name=transform_name)
        image = transform(image)
        video = image.unsqueeze(0).repeat(num_frames, 1, 1, 1)
        video = video.permute(1, 0, 2, 3)
        return video

    @classmethod
    def read_video_from_path(cls,
                             path,
                             transform=None,
                             transform_name="center",
                             image_size=(256, 256)):
        vframes, aframes, info = read_video(filename=path,
                                            pts_unit="sec",
                                            output_format="TCHW")
        if transform is None:
            transform = cls.get_transforms_video(image_size=image_size,
                                                 name=transform_name)
        video = transform(vframes)  # T C H W
        video = video.permute(1, 0, 2, 3)
        return video

    @classmethod
    def read_from_path(cls, path, image_size, transform_name="center"):
        if is_url(path):
            path = download_url(path)
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return cls.read_video_from_path(path,
                                            image_size=image_size,
                                            transform_name=transform_name)
        else:
            assert ext.lower(
            ) in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return cls.read_image_from_path(path,
                                            image_size=image_size,
                                            transform_name=transform_name)

    @classmethod
    def collect_references_batch(cls, vae, reference_paths, image_size):
        refs_x = []  # refs_x: [batch, ref_num, C, T, H, W]
        for reference_path in reference_paths:
            if reference_path == "":
                refs_x.append([])
                continue
            ref_path = reference_path.split(";")
            ref = []
            for r_path in ref_path:
                r = cls.read_from_path(r_path,
                                       image_size,
                                       transform_name="resize_crop")
                r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
                r_x = r_x.squeeze(0)
                ref.append(r_x)
            refs_x.append(ref)
        return refs_x

    @staticmethod
    def append_generated(vae, generated_video, refs_x, mask_strategy, loop_i,
                         condition_frame_length, condition_frame_edit):
        ref_x = vae.encode(generated_video)
        for j, refs in enumerate(refs_x):
            if refs is None:
                refs_x[j] = [ref_x[j]]
            else:
                refs.append(ref_x[j])
            if mask_strategy[j] is None or mask_strategy[j] == "":
                mask_strategy[j] = ""
            else:
                mask_strategy[j] += ";"
            mask_strategy[
                j] += f"{loop_i},{len(refs)-1},-{condition_frame_length},0,{condition_frame_length},{condition_frame_edit}"
        return refs_x, mask_strategy

    @classmethod
    def parse_mask_strategy(cls, mask_strategy):
        mask_batch = []
        if mask_strategy == "" or mask_strategy is None:
            return mask_batch
        mask_strategy = mask_strategy.split(";")
        for mask in mask_strategy:
            mask_group = mask.split(",")
            num_group = len(mask_group)
            assert num_group >= 1 and num_group <= 6, f"Invalid mask strategy: {mask}"
            mask_group.extend(cls.MASK_DEFAULT[num_group:])
            for i in range(5):
                mask_group[i] = int(mask_group[i])
            mask_group[5] = float(mask_group[5])
            mask_batch.append(mask_group)
        return mask_batch

    @classmethod
    def find_nearest_point(cls, value, point, max_value):
        t = value // point
        if value % point > point / 2 and t < max_value // point - 1:
            t += 1
        return t * point

    @classmethod
    def apply_mask_strategy(cls, z, refs_x, mask_strategys, loop_i, align=None):
        masks = []
        no_mask = True
        for i, mask_strategy in enumerate(mask_strategys):
            no_mask = False
            mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
            mask_strategy = cls.parse_mask_strategy(mask_strategy)
            for mst in mask_strategy:
                loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
                if loop_id != loop_i:
                    continue
                ref = refs_x[i][m_id]

                if m_ref_start < 0:
                    # ref: [C, T, H, W]
                    m_ref_start = ref.shape[1] + m_ref_start
                if m_target_start < 0:
                    # z: [B, C, T, H, W]
                    m_target_start = z.shape[2] + m_target_start
                if align is not None:
                    m_ref_start = cls.find_nearest_point(
                        m_ref_start, align, ref.shape[1])
                    m_target_start = cls.find_nearest_point(
                        m_target_start, align, z.shape[2])
                m_length = min(m_length, z.shape[2] - m_target_start,
                               ref.shape[1] - m_ref_start)
                z[i, :, m_target_start:m_target_start +
                  m_length] = ref[:, m_ref_start:m_ref_start + m_length]
                mask[m_target_start:m_target_start + m_length] = edit_ratio
            masks.append(mask)
        if no_mask:
            return None
        masks = torch.stack(masks)
        return masks

    @staticmethod
    def dframe_to_frame(num):
        assert num % 5 == 0, f"Invalid num: {num}"
        return num // 5 * 17

    @staticmethod
    def save_sample(x,
                    save_path=None,
                    fps=8,
                    normalize=True,
                    value_range=(-1, 1),
                    force_video=False,
                    verbose=True):
        """
        Args:
            x (Tensor): shape [C, T, H, W]
        """
        assert x.ndim == 4
        if not force_video and x.shape[1] == 1:  # T = 1: save as image
            save_path += ".png"
            x = x.squeeze(1)
            save_image([x],
                       save_path,
                       normalize=normalize,
                       value_range=value_range)
        else:
            save_path += ".mp4"
            if normalize:
                low, high = value_range
                x.clamp_(min=low, max=high)
                x.sub_(low).div_(max(high - low, 1e-5))
            x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(
                "cpu", torch.uint8)
            write_video(save_path, x, fps=fps, video_codec="h264")
        if verbose:
            print(f"Saved to {save_path}")
        return save_path

    @staticmethod
    def add_watermark(
            input_video_path,
            watermark_image_path="./assets/images/watermark/watermark.png",
            output_video_path=None):
        # execute this command in terminal with subprocess
        # return if the process is successful
        if output_video_path is None:
            output_video_path = input_video_path.replace(
                ".mp4", "_watermark.mp4")
        cmd = f'ffmpeg -y -i {input_video_path} -i {watermark_image_path}'
        cmd += f'-filter_complex "[1][0]scale2ref=oh*mdar:ih*0.1[logo][video];[video][logo]overlay" {output_video_path}'
        import subprocess
        exit_code = subprocess.call(cmd, shell=True)
        is_success = exit_code == 0
        return is_success


class DataProcessor():

    def __init__(self, text_encoder, vae):
        self._text_encoder = text_encoder
        self._vae = vae

    @staticmethod
    def set_random_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    @staticmethod
    def prepare_multi_resolution_info(model_type, batch_size, image_size,
                                      num_frames, fps, device, dtype):
        IMG_FPS = 120
        ret = {}
        if model_type in ["STDiT3", "OpenSora"]:
            fps = fps if num_frames > 1 else IMG_FPS
            fps = torch.tensor([fps], device=device,
                               dtype=dtype).repeat(batch_size)
            height = torch.tensor([image_size[0]], device=device,
                                  dtype=dtype).repeat(batch_size)
            width = torch.tensor([image_size[1]], device=device,
                                 dtype=dtype).repeat(batch_size)
            num_frames = torch.tensor([num_frames], device=device,
                                      dtype=dtype).repeat(batch_size)
            ar = torch.tensor([image_size[0] / image_size[1]],
                              device=device,
                              dtype=dtype).repeat(batch_size)
            ret = dict(height=height,
                       width=width,
                       num_frames=num_frames,
                       ar=ar,
                       fps=fps)
        else:
            raise NotImplementedError(f"Model type is {model_type}")
        return ret

    @staticmethod
    def get_save_path_name(
            save_dir,
            sample_name=None,  # prefix
            sample_idx=None,  # sample index
            prompt=None,  # used prompt
            prompt_as_path=False,  # use prompt as path
            num_sample=1,  # number of samples to generate for one prompt
            k=None,  # kth sample
    ):
        if sample_name is None:
            sample_name = "" if prompt_as_path else "sample"
        sample_name_suffix = prompt if prompt_as_path else f"_{sample_idx:04d}"
        save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
        if num_sample != 1:
            save_path = f"{save_path}-{k}"
        return save_path

    @staticmethod
    def load_prompts(prompt_path, start_idx=None, end_idx=None):
        return PromptProcessor.load_prompts(prompt_path=prompt_path,
                                            start_idx=start_idx,
                                            end_idx=end_idx)

    @staticmethod
    def extract_json_from_prompts(prompts, reference, mask_strategy):
        return PromptProcessor.extract_json_from_prompts(
            prompts=prompts, reference=reference, mask_strategy=mask_strategy)

    @staticmethod
    def split_prompt(prompt_text):
        return PromptProcessor.split_prompt(prompt_text=prompt_text)

    @staticmethod
    def merge_prompt(text_list, loop_idx_list=None):
        return PromptProcessor.merge_prompt(text_list=text_list,
                                            loop_idx_list=loop_idx_list)

    @staticmethod
    def extract_prompts_loop(prompts, num_loop):
        return PromptProcessor.extract_prompts_loop(prompts=prompts,
                                                    num_loop=num_loop)

    @staticmethod
    def append_score_to_prompts(prompts,
                                aes=None,
                                flow=None,
                                camera_motion=None):
        return PromptProcessor.append_score_to_prompts(
            prompts=prompts, aes=aes, flow=flow, camera_motion=camera_motion)

    @staticmethod
    def text_preprocessing(text, use_text_preprocessing: bool = True):
        return PromptProcessor.text_preprocessing(
            text=text, use_text_preprocessing=use_text_preprocessing)

    @staticmethod
    def get_image_size(resolution, ar_ratio):
        return VideoProcessor.get_image_size(resolution=resolution,
                                             ar_ratio=ar_ratio)

    @staticmethod
    def get_num_frames(num_frames):
        return VideoProcessor.get_num_frames(num_frames=num_frames)

    def get_latent_size(self, input_size):
        return VideoProcessor.get_latent_size(self._vae, input_size=input_size)

    def collect_references_batch(self, reference_paths, image_size):
        return VideoProcessor.collect_references_batch(
            self._vae, reference_paths=reference_paths, image_size=image_size)

    def append_generated(self, generated_video, refs_x, mask_strategy, loop_i,
                         condition_frame_length, condition_frame_edit):
        return VideoProcessor.append_generated(
            self._vae,
            generated_video=generated_video,
            refs_x=refs_x,
            mask_strategy=mask_strategy,
            loop_i=loop_i,
            condition_frame_length=condition_frame_length,
            condition_frame_edit=condition_frame_edit)

    @staticmethod
    def apply_mask_strategy(z, refs_x, mask_strategys, loop_i, align=None):
        return VideoProcessor.apply_mask_strategy(z=z,
                                                  refs_x=refs_x,
                                                  mask_strategys=mask_strategys,
                                                  loop_i=loop_i,
                                                  align=align)

    @staticmethod
    def dframe_to_frame(num):
        return VideoProcessor.dframe_to_frame(num=num)

    @staticmethod
    def save_sample(x,
                    save_path=None,
                    fps=8,
                    normalize=True,
                    value_range=(-1, 1),
                    force_video=False,
                    verbose=True):
        return VideoProcessor.save_sample(x=x,
                                          save_path=save_path,
                                          fps=fps,
                                          normalize=normalize,
                                          value_range=value_range,
                                          force_video=force_video,
                                          verbose=verbose)

    @staticmethod
    def add_watermark(
            input_video_path,
            watermark_image_path="./assets/images/watermark/watermark.png",
            output_video_path=None):
        return VideoProcessor.add_watermark(
            input_video_path=input_video_path,
            watermark_image_path=watermark_image_path,
            output_video_path=output_video_path)
