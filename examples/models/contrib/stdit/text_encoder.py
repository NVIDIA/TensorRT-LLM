# Copyright (C) 2023  PixArt-alpha/PixArt-alpha
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

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

# reference: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/text_encoder/t5.py

import collections.abc
from functools import partial
from itertools import repeat

import torch
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.modeling.jit import get_jit_fused_dropout_add_func
from colossalai.shardformer.modeling.t5 import (
    get_jit_fused_T5_layer_ff_forward, get_T5_layer_self_attention_forward)
from colossalai.shardformer.policies.base_policy import (
    Policy, SubModuleReplacementDescription)
from transformers import AutoTokenizer, T5EncoderModel
from transformers.models.t5.modeling_t5 import (T5LayerFF, T5LayerSelfAttention,
                                                T5Stack)


def default(var, default_var):
    return default_var if var is None else var


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class T5LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states

    @staticmethod
    def from_native_module(module, *args, **kwargs):
        assert module.__class__.__name__ == "FusedRMSNorm", (
            "Recovering T5LayerNorm requires the original layer to be apex's Fused RMS Norm."
            "Apex's fused norm is automatically used by Hugging Face Transformers https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L265C5-L265C48"
        )
        layer_norm = T5LayerNorm(module.normalized_shape, eps=module.eps)
        layer_norm.weight.data.copy_(module.weight.data)
        layer_norm = layer_norm.to(module.weight.device)
        return layer_norm


class T5EncoderPolicy(Policy):

    def config_sanity_check(self):
        assert not self.shard_config.enable_tensor_parallelism
        assert not self.shard_config.enable_flash_attention

    def preprocess(self):
        return self.model

    def module_policy(self):
        policy = {}
        # check whether apex is installed
        try:
            # recover hf from fused rms norm to T5 norm which is faster
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=T5LayerNorm,
                ),
                policy=policy,
                target_key=T5LayerFF,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5LayerSelfAttention,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="final_layer_norm", target_module=T5LayerNorm),
                policy=policy,
                target_key=T5Stack,
            )
        except (ImportError, ModuleNotFoundError):
            pass
        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_T5_layer_ff_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerFF,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_T5_layer_self_attention_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=T5LayerSelfAttention,
            )
        return policy

    def postprocess(self):
        return self.model


class T5Embedder:

    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
        local_files_only=False,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.float16
        self.cache_dir = cache_dir

        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }

            if use_offload_folder is not None:
                t5_model_kwargs["offload_folder"] = use_offload_folder
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder.embed_tokens": self.device,
                    "encoder.block.0": self.device,
                    "encoder.block.1": self.device,
                    "encoder.block.2": self.device,
                    "encoder.block.3": self.device,
                    "encoder.block.4": self.device,
                    "encoder.block.5": self.device,
                    "encoder.block.6": self.device,
                    "encoder.block.7": self.device,
                    "encoder.block.8": self.device,
                    "encoder.block.9": self.device,
                    "encoder.block.10": self.device,
                    "encoder.block.11": self.device,
                    "encoder.block.12": "disk",
                    "encoder.block.13": "disk",
                    "encoder.block.14": "disk",
                    "encoder.block.15": "disk",
                    "encoder.block.16": "disk",
                    "encoder.block.17": "disk",
                    "encoder.block.18": "disk",
                    "encoder.block.19": "disk",
                    "encoder.block.20": "disk",
                    "encoder.block.21": "disk",
                    "encoder.block.22": "disk",
                    "encoder.block.23": "disk",
                    "encoder.final_layer_norm": "disk",
                    "encoder.dropout": "disk",
                }
            else:
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder": self.device,
                }

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **t5_model_kwargs,
        ).eval()
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask


class Mlp(torch.nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU(),
        norm_layer=None,
        bias=True,
        drop=0.,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(torch.nn.Conv2d,
                               kernel_size=1) if use_conv else torch.nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer
        self.drop1 = torch.nn.Dropout(drop_probs[0])
        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else torch.nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = torch.nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CaptionEmbedder(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            hidden_size,
            uncond_prob,
            act_layer=torch.nn.GELU(approximate="tanh"),
            token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding,
                              caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class T5Encoder:

    def __init__(
        self,
        from_pretrained=None,
        model_max_length=120,
        caption_channels=4096,
        hidden_size=1152,
        class_dropout_prob=0.1,
        y_embedding=None,
        device="cuda",
        dtype=torch.float,
        cache_dir=None,
        shardformer=False,
        local_files_only=False,
    ):
        assert from_pretrained is not None, "Please specify the path to the T5 model"
        self.t5 = T5Embedder(
            device=device,
            torch_dtype=dtype,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            local_files_only=local_files_only,
        )
        self.t5.model.to(dtype=dtype)
        # [NOTE] disable y_embedder
        if False:
            self.y_embedder = CaptionEmbedder(
                in_channels=caption_channels,
                hidden_size=hidden_size,
                uncond_prob=class_dropout_prob,
                act_layer=torch.nn.GELU(approximate="tanh"),
                token_num=model_max_length,
            )
        else:
            self.y_embedder = None
        self.y_embedding = default(
            y_embedding,
            torch.randn(model_max_length, caption_channels) /
            caption_channels**0.5).to(device)

        self.model_max_length = model_max_length
        self.output_dim = self.t5.model.config.d_model
        self.dtype = dtype

        if shardformer:
            self.shardformer_t5()

    def shardformer_t5(self):
        shard_config = ShardConfig(
            tensor_parallel_process_group=None,
            pipeline_stage_manager=None,
            enable_tensor_parallelism=False,
            enable_fused_normalization=False,
            enable_flash_attention=False,
            enable_jit_fused=True,
            enable_sequence_parallelism=False,
            enable_sequence_overlap=False,
        )
        shard_former = ShardFormer(shard_config=shard_config)
        optim_model, _ = shard_former.optimize(self.t5.model,
                                               policy=T5EncoderPolicy())
        self.t5.model = optim_model.to(self.dtype)

        # ensure the weights are frozen
        for p in self.t5.model.parameters():
            p.requires_grad = False

    def encode(self, text):
        caption_embs, emb_masks = self.t5.get_text_embeddings(text)
        caption_embs = caption_embs[:, None]
        return dict(y=caption_embs, mask=emb_masks)

    def null(self, n):
        if self.y_embedder is None:
            null_y = self.y_embedding[None].repeat(n, 1, 1)[:, None]
        else:
            null_y = self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        return null_y

    def encode_with_null(self, text):
        batch_size = len(text)
        encoded_outputs = self.encode(text)
        y_null = self.null(batch_size)
        encoded_outputs["y"] = torch.cat([encoded_outputs["y"], y_null], 0)
        return encoded_outputs
