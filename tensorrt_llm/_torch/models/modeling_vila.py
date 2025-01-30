#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/

import copy
import math
import os
import os.path as osp
import re
import warnings
from typing import List, Optional, Tuple

import torch
from huggingface_hub import repo_exists, snapshot_download
from huggingface_hub.utils import HFValidationError
from PIL import Image
from torch import nn
from transformers import (AutoConfig, AutoModel, AutoTokenizer, LlamaConfig,
                          LlavaConfig, PretrainedConfig, PreTrainedModel)

from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       get_hf_image_processor, register_input_processor)
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from .modeling_auto import AutoModelForCausalLM
from .modeling_siglip import SiglipVisionTower, SiglipVisionTowerS2
from .modeling_utils import ModelConfig, register_auto_model
from .modeling_vit import CLIPVisionTower, CLIPVisionTowerS2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class DownSampleBlock(nn.Module):

    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)],
                dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat(
                [x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)],
                dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        return x


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.mm_projector_type = mm_projector_type


class MultimodalProjector(PreTrainedModel):
    config_class = MultimodalProjectorConfig

    def __init__(self, mm_projector_cfg: MultimodalProjectorConfig,
                 config: PretrainedConfig):
        super().__init__(mm_projector_cfg)
        mm_projector_type = mm_projector_cfg.mm_projector_type
        if mm_projector_type == "identity":
            self.layers = IdentityMap()
        elif mm_projector_type == "linear":
            self.layers = nn.Linear(config.mm_hidden_size, config.hidden_size)
        elif mm_projector_type == "mlp_downsample":
            self.layers = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(
                        nn.Linear(config.hidden_size, config.hidden_size))
                self.layers = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


def init_mm_projector(model_type_or_path: str,
                      config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path:
        assert os.path.exists(
            model_type_or_path
        ), f"Pretrained mm projector path {model_type_or_path} does not exist!"
        return MultimodalProjector.from_pretrained(model_type_or_path,
                                                   config,
                                                   torch_dtype=_convert_dtype(
                                                       config.model_dtype))
    ## build from scratch
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config).to(
            _convert_dtype(config.model_dtype))
        return mm_projector


def init_vision_tower(model_name_or_path: str,
                      config: PretrainedConfig) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(
            model_name_or_path
        ), f"Pretrained vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = vision_tower_arch if vision_tower_arch is not None else model_name_or_path

    use_s2 = getattr(config, "s2", False)

    if "intern" in vision_tower_name.lower():
        raise NotImplementedError("Intern vision tower is not Implemented yet")
    elif "radio" in vision_tower_name:
        raise NotImplementedError("Radio vision tower is not Implemented yet")
    elif "clip" in vision_tower_name:
        if use_s2:
            vision_tower = CLIPVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = CLIPVisionTower(model_name_or_path, config)
    elif "siglip" in vision_tower_name:
        if use_s2:
            vision_tower = SiglipVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = vision_tower.config.hidden_size if not use_s2 else vision_tower.hidden_size
    return vision_tower


def init_llm(
    model_name_or_path: str,
    model_config: ModelConfig[PretrainedConfig],
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> PreTrainedModel:
    llm_cfg = LlamaConfig.from_dict(model_config.pretrained_config.llm_cfg)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        orig_ctx_len = getattr(llm_cfg, "max_position_embeddings", None)
        model_max_length = getattr(llm_cfg, "model_max_length", None)
        if orig_ctx_len and model_max_length > orig_ctx_len:
            print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
            scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
            llm_cfg.rope_scaling = {"type": "linear", "factor": scaling_factor}
    llm_cfg.model_max_length = 2540
    llm_cfg.max_new_tokens = 2540

    llm_model_config = copy.deepcopy(model_config)
    llm_model_config.pretrained_config = llm_cfg
    llm = AutoModelForCausalLM.from_config(llm_model_config)

    # Locate the tokenizer.
    llm_path = model_name_or_path

    try:
        getattr(llm_cfg, "architectures")[0].lower()
    except BaseException:
        warnings.warn(
            f'Cannot find LLM architecture, please check the "config.json" under "{llm_path}".'
        )

    model_config.pretrained_config.hidden_size = llm.config.hidden_size
    return llm, llm_cfg.vocab_size


def parse_model_name_or_path(config: PretrainedConfig,
                             model_name="llm",
                             suffix="_cfg"):
    target_model = f"{model_name}{suffix}"
    target_cfg = getattr(config, target_model, None)

    if isinstance(target_cfg, str):
        return target_cfg
    elif isinstance(target_cfg, dict):
        return target_cfg["architectures"][0]
    else:
        raise ValueError(f"Invalid {target_model} configuration!")


def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(
            f"Invalid configuration! Cannot find vision_tower in config:\n{config}"
        )

    config.model_dtype = kwargs.pop("torch_dtype").__str__()
    # siglip does not support device_map = "auto"
    vision_tower_name = parse_model_name_or_path(config, "vision_tower")
    if "siglip" in vision_tower_name.lower():
        kwargs["device_map"] = "cuda"


class VilaLlamaConfig(LlavaConfig):
    # TODO(yuanjings): change the name to be more specific, e.g. vila_llama
    model_type = "llava_llama"


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_projector_cfg"]

    root_path = None
    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path

    # download from huggingface
    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)
        else:
            raise RuntimeError(f"Invalid config type: {type(cfg)}")
    return return_list


def _split_prompt_by_images(tensor):
    batch_splits = []
    for batch in tensor:
        # Find indices where value is zero (<image>)
        zero_indices = (batch == 0).nonzero(as_tuple=False).squeeze(0)
        # Add starting point for slicing
        start_idx = 0
        splits = []
        for idx in zero_indices:
            if start_idx != idx:  # Ensure not slicing zero-length tensors
                splits.append(batch[start_idx:idx].unsqueeze(0))
            start_idx = idx + 1  # Move start index past the zero
        if start_idx < len(
                batch):  # Handle last segment if it's not zero-ending
            splits.append(batch[start_idx:].unsqueeze(0))
        # Remove empty tensors resulting from consecutive zeros
        splits = [split for split in splits if split.numel() > 0]
        batch_splits.append(splits)

    return batch_splits


def _ptuning_setup(prompt_table, input_ids, hidden_size):
    if prompt_table is not None:
        task_vocab_size = torch.tensor(
            [prompt_table.shape[1]],
            dtype=torch.int32,
        ).cuda()
        prompt_table = prompt_table.view(
            (prompt_table.shape[0] * prompt_table.shape[1],
             prompt_table.shape[2]))

        assert prompt_table.shape[
            1] == hidden_size, "Prompt table dimensions do not match hidden size"
    else:
        prompt_table = torch.empty([1, hidden_size]).cuda()
        task_vocab_size = torch.zeros([1]).cuda()

    tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

    return [prompt_table, tasks, task_vocab_size]


class VilaLlamaInputProcessor(InputProcessor):

    def __init__(self, model_config, tokenizer):
        self.model_config = model_config
        cfgs = get_model_config(self.model_config)
        if len(cfgs) == 3:
            llm_path, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config."
            )
        device = 'cuda'
        llm_cfg = AutoConfig.from_pretrained(llm_path)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                model_max_length=llm_cfg.model_max_length,
                padding_side="right",
                use_fast=False,
                legacy=False,
            )
        else:
            self.tokenizer = tokenizer
        self.model_dtype = _convert_dtype(self.model_config.model_dtype)
        self.vision_tower = init_vision_tower(
            vision_tower_cfg, self.model_config).to(device=device,
                                                    dtype=self.model_dtype)
        self.mm_projector = init_mm_projector(
            mm_projector_cfg, self.model_config).to(device=device,
                                                    dtype=self.model_dtype)

    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        prompt = inputs.get("prompt")
        prompt_chunks = [
            self.tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
        ]

        def insert_separator(X, sep):
            return [
                ele for sublist in zip(X, [sep] * len(X)) for ele in sublist
            ][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(
                prompt_chunks[0]
        ) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks,
                                  [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])
        batch_size = 1
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids[input_ids == IMAGE_TOKEN_INDEX] = 0
        input_ids = input_ids.unsqueeze(0).expand(batch_size, -1)
        batch_split_prompts = _split_prompt_by_images(input_ids)
        first_batch_split_prompts = batch_split_prompts[0]
        image_data = inputs.get("multi_modal_data")['image']
        mm_processor_kwargs = inputs.get("mm_processor_kwargs")
        image_tensor = self._process_image_for_llava_llama(
            image_data, **mm_processor_kwargs)
        visual_features = self.vision_tower(image_tensor)
        visual_features = self.mm_projector(visual_features)
        token_ids, ptuning_config = self._pad_token_ids_vila(
            batch_size, visual_features, first_batch_split_prompts)
        # reset
        self.reset()
        return token_ids, {"prompt_tuning_config": ptuning_config}

    def _pad_token_ids_vila(self, batch_size, visual_features, split_input_ids):
        vocab_size = self.model_config.llm_cfg["vocab_size"]
        hidden_size = self.model_config.hidden_size
        fake_prompt_counter = vocab_size
        if batch_size == 1:
            # only check for multi-image inference (mode 1)
            assert len(visual_features) <= len(
                split_input_ids
            ), "Unexpected number of visual features. Please check #<image> in prompt and the #image files."

        input_ids = []
        if batch_size == 1:
            # mode 1: multiple image as a whole, concat all prompts together, <pre><image1><inter><image2>...<post>
            input_ids = [split_input_ids[0]]
            for idx, visual_feature in enumerate(visual_features):
                fake_prompt_id = torch.arange(
                    fake_prompt_counter,
                    fake_prompt_counter + visual_feature.shape[0])
                fake_prompt_counter += visual_feature.shape[0]
                fake_prompt_id = fake_prompt_id.unsqueeze(0)
                input_ids.append(fake_prompt_id)
                # in case no inter or post prompt
                if len(split_input_ids) > idx + 1:
                    input_ids.append(split_input_ids[idx + 1])
        elif batch_size > 1:
            # mode 2: each image have individual prompt, <pre><image><post>
            for idx, visual_feature in enumerate(visual_features):
                input_ids.append(split_input_ids[0])
                fake_prompt_id = torch.arange(
                    fake_prompt_counter,
                    fake_prompt_counter + visual_feature.shape[0])
                fake_prompt_id = fake_prompt_id.unsqueeze(0)
                input_ids.append(fake_prompt_id)
                if len(split_input_ids) > 1:
                    input_ids.append(split_input_ids[1])

        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)
        input_ids = input_ids.reshape(batch_size, -1)
        ptuning_args = _ptuning_setup(visual_features, input_ids, hidden_size)
        return input_ids.squeeze(0).tolist(), ptuning_args

    def _process_image_for_llava_llama(self, data: dict[str, any],
                                       **mm_processor_kwargs):
        """process multimodal inputs"""
        model_name = self.model_config._name_or_path
        image_processor = get_hf_image_processor(model_name + "/vision_tower/",
                                                 trust_remote_code=True)
        if isinstance(data, Image.Image):
            data = [data]
        elif not isinstance(data, list):
            raise TypeError(f"Invalid image type: {type(data)}")

        def process_image(image_file, image_processor, image_aspect_ratio):
            image = image_file.convert("RGB")
            if image_aspect_ratio == "resize":
                if hasattr(image_processor, "crop_size"):
                    # CLIP vision tower
                    crop_size = image_processor.crop_size
                else:
                    # SIGLIP vision tower
                    assert hasattr(image_processor, "size")
                    crop_size = image_processor.size
                image = image.resize((crop_size["height"], crop_size["width"]))

            if image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width),
                                           background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height),
                                           background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image,
                    tuple(int(x * 255) for x in image_processor.image_mean))
                image = image_processor.preprocess(
                    image, return_tensors="pt")["pixel_values"][0]
            else:
                image = image_processor.preprocess(
                    image, return_tensors="pt")["pixel_values"][0]
            return image

        images_tensor = [
            process_image(image, image_processor,
                          self.model_config.image_aspect_ratio)
            for image in data
        ]
        if all(x.shape == images_tensor[0].shape for x in images_tensor):
            images_tensor = torch.stack(images_tensor, dim=0)
        return images_tensor.to(self.model_dtype)

    def reset(self):
        cfgs = get_model_config(self.model_config)
        if len(cfgs) == 3:
            llm_path, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config."
            )
        device = 'cuda'
        self.vision_tower = init_vision_tower(
            vision_tower_cfg, self.model_config).to(device=device,
                                                    dtype=torch.float16)


def _convert_dtype(dtype_str):
    if dtype_str == "torch.float16":
        return torch.float16
    elif dtype_str == "torch.bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupportede dtype for VILA: {dtype_str}")


# TODO(yuanjings): change the name to be more specific, e.g. VilaLlamaModel
@register_auto_model("LlavaLlamaModel")
@register_input_processor(VilaLlamaInputProcessor)
class VilaLlamaModel(PreTrainedModel):
    config_class = VilaLlamaConfig

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs) -> None:
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return

        self.model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn(
                "model_dtype not found in config, defaulting to torch.float16.")
        config.model_dtype = self.model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, _, _ = cfgs
        else:
            raise ValueError(
                "`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config."
            )
        self.llm, self.vocab_size = init_llm(llm_cfg, model_config, *args,
                                             **kwargs)
        self.eval()
        self.context_len = 2048
        device = kwargs.get("device", "cuda")
        self.model_dtype = _convert_dtype(self.model_dtype)
        self.llm.to(device=device, dtype=self.dtype)

        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        self.llm.load_weights(weights)

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def post_config(self):
        self.training = self.get_llm().training
        # use llm.config as config for pytorch model engine
        self.config = self.llm.config
        self.model_config = self.llm.config

    def _prepare_inputs_embeds_for_multimodal(
        self,
        input_ids: Optional[torch.LongTensor],
        batch_image_embeds: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        if len(batch_image_embeds) == 0:
            return input_ids, None
        assert len(batch_image_embeds) == attn_metadata.num_contexts, \
            "number of image embeds should be the same as number of context request"
        embedding_dim = self.llm.model.embed_tokens.in_features
        inputs_embeds = torch.zeros(input_ids.shape[0],
                                    embedding_dim,
                                    device=input_ids.device,
                                    dtype=self.model_dtype)
        text_token_mask = input_ids < self.vocab_size
        text_token_indices = torch.where(text_token_mask)[0]
        text_token_ids = input_ids[text_token_indices]
        text_token_embeddings = self.llm.model.embed_tokens(text_token_ids)
        inputs_embeds[text_token_indices, :] = text_token_embeddings
        token_offset = 0
        image_embeds_ptr = 0
        for seq_len in attn_metadata.seq_lens:
            if seq_len == 1:  # skip decode tokens
                token_offset += seq_len
                continue
            image_embeds = batch_image_embeds[image_embeds_ptr].to(
                inputs_embeds.dtype)
            image_embeds_ptr += 1
            image_embeds_mask = input_ids[token_offset:token_offset +
                                          seq_len] >= self.vocab_size
            image_embeds_indices = torch.where(image_embeds_mask)[0]
            inputs_embeds[image_embeds_indices, :] = image_embeds
            token_offset += seq_len

        return None, inputs_embeds

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        batch_image_embeds = kwargs.get("multi_modal_data", [])
        input_ids, inputs_embeds = self._prepare_inputs_embeds_for_multimodal(
            input_ids, batch_image_embeds, attn_metadata)
        return self.llm.forward(attn_metadata, input_ids, position_ids,
                                inputs_embeds, return_context_logits)


AutoConfig.register("llava_llama", VilaLlamaConfig)
AutoModel.register(VilaLlamaConfig, VilaLlamaModel)
