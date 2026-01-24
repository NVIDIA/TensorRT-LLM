# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import contextmanager


class BaseOpManager:
    @classmethod
    def get_impl(cls, name=None):
        raise NotImplementedError("Subclass must implement this method")

    @classmethod
    def get_registered_types(cls):
        raise NotImplementedError("Subclass must implement this method")

    @classmethod
    def op_type(cls):
        raise NotImplementedError("Subclass must implement this method")


class AttentionOpManager(BaseOpManager):
    _attn_registry = {}
    attn_type = "default"
    attn_choices = ["default", "sage-attn"]  # choices for auto tuning
    high_precision_attn_type = (
        "default"  # high precision attention type, default is the same as attn_type
    )
    num_timesteps_high_precision = 0.0  # when timestep < num_inference_steps * num_timesteps_high_precision, use `high_precision_attn_type` attention operators
    num_layers_high_precision = 0.0  # when layer idx < num_layers * num_layers_high_precision, use `high_precision_attn_type` attention operators
    num_heads = None
    head_dim = None
    num_key_value_heads = None
    cosine_similarity_threshold = None
    mse_threshold = None
    record_io_tensors = False

    @classmethod
    def op_type(cls):
        return "attention"

    @classmethod
    def set_attn_config(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"'{cls.__name__}' has no attribute '{key}'")

    @classmethod
    def register_attn(cls, attn_type):
        def decorator(attn_class):
            # Register the attention class
            cls._attn_registry[attn_type] = attn_class
            return attn_class

        return decorator

    @classmethod
    def get_impl(cls, name=None):
        if name is None:
            name = cls.attn_type
        attn_class = cls._attn_registry.get(name)
        if attn_class is None:
            raise ValueError(f"Attention function {name} not found in registry")
        return attn_class()  # Create and return an instance

    @classmethod
    def get_registered_types(cls):
        return list(cls._attn_registry.keys())


class SparseVideogenConfig:
    _instance = None
    _initialized = False

    def __init__(self):
        if not self._initialized:
            self.num_sampled_rows = 0
            self.sample_mse_max_row = 0
            self.sparsity = 0
            self.attention_masks = []
            self.first_layers_fp = 0  # not used
            self.first_times_fp = 0  # not used
            self.context_length = 0
            self.num_frame = 0
            self.frame_size = 0
            self.block_mask = None
            SparseVideogenConfig._initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
        cls._initialized = False

    @classmethod
    def num_sampled_rows(cls):
        return cls.get_instance().num_sampled_rows

    @classmethod
    def sample_mse_max_row(cls):
        return cls.get_instance().sample_mse_max_row

    @classmethod
    def sparsity(cls):
        return cls.get_instance().sparsity

    @classmethod
    def attention_masks(cls):
        return cls.get_instance().attention_masks

    @classmethod
    def context_length(cls):
        return cls.get_instance().context_length

    @classmethod
    def num_frame(cls):
        return cls.get_instance().num_frame

    @classmethod
    def frame_size(cls):
        return cls.get_instance().frame_size

    @classmethod
    def block_mask(cls):
        return cls.get_instance().block_mask

    @classmethod
    def first_layers_fp(cls):
        return cls.get_instance().first_layers_fp

    @classmethod
    def first_times_fp(cls):
        return cls.get_instance().first_times_fp

    @classmethod
    def update(cls, **kwargs):
        """Update multiple attributes at once"""
        instance = cls.get_instance()
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                raise AttributeError(f"'{cls.__name__}' has no attribute '{key}'")

    @classmethod
    def is_valid(cls):
        return (
            cls.get_instance().num_sampled_rows > 0
            and cls.get_instance().sample_mse_max_row > 0
            and cls.get_instance().block_mask is not None
            and cls.get_instance().num_frame > 0
            and cls.get_instance().frame_size > 0
            and cls.get_instance().sparsity > 0
            and cls.get_instance().attention_masks is not None
        )


class SparseVideogenConfig2:
    _instance = None
    _initialized = False

    def __init__(self):
        if not self._initialized:
            self.num_layers = 0
            self.num_q_centroids = 0
            self.num_k_centroids = 0
            self.top_p_kmeans = 0
            self.min_kc_ratio = 0
            self.kmeans_iter_init = 0
            self.kmeans_iter_step = 0
            self.logging_file = None
            self.zero_step_kmeans_init = False
            self.first_layers_fp = 0  # not used
            self.first_times_fp = 0  # not used
            self.context_length = 0
            self.num_frame = 0
            self.frame_size = 0
            SparseVideogenConfig2._initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
        cls._initialized = False

    @classmethod
    def num_layers(cls):
        return cls.get_instance().num_layers

    @classmethod
    def num_q_centroids(cls):
        return cls.get_instance().num_q_centroids

    @classmethod
    def num_k_centroids(cls):
        return cls.get_instance().num_k_centroids

    @classmethod
    def top_p_kmeans(cls):
        return cls.get_instance().top_p_kmeans

    @classmethod
    def min_kc_ratio(cls):
        return cls.get_instance().min_kc_ratio

    @classmethod
    def kmeans_iter_init(cls):
        return cls.get_instance().kmeans_iter_init

    @classmethod
    def kmeans_iter_step(cls):
        return cls.get_instance().kmeans_iter_step

    @classmethod
    def logging_file(cls):
        return cls.get_instance().logging_file

    @classmethod
    def zero_step_kmeans_init(cls):
        return cls.get_instance().zero_step_kmeans_init

    @classmethod
    def context_length(cls):
        return cls.get_instance().context_length

    @classmethod
    def num_frame(cls):
        return cls.get_instance().num_frame

    @classmethod
    def frame_size(cls):
        return cls.get_instance().frame_size

    @classmethod
    def first_layers_fp(cls):
        return cls.get_instance().first_layers_fp

    @classmethod
    def first_times_fp(cls):
        return cls.get_instance().first_times_fp

    @classmethod
    def update(cls, **kwargs):
        """Update multiple attributes at once"""
        instance = cls.get_instance()
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                raise AttributeError(f"'{cls.__name__}' has no attribute '{key}'")

    @classmethod
    def is_valid(cls):
        return (
            cls.get_instance().num_q_centroids > 0
            and cls.get_instance().num_k_centroids > 0
            and cls.get_instance().top_p_kmeans > 0
            and cls.get_instance().min_kc_ratio > 0
            and cls.get_instance().kmeans_iter_init > 0
            and cls.get_instance().kmeans_iter_step > 0
            and cls.get_instance().num_frame > 0
            and cls.get_instance().frame_size > 0
            and cls.get_instance().context_length > 0
        )


class LinearOpManager(BaseOpManager):
    _linear_registry = {}
    linear_type = "default"
    linear_recipe = "dynamic"
    linear_choices = ["default"]  # choices for auto tuning
    cosine_similarity_threshold = None
    mse_threshold = None
    record_io_tensors = False

    @classmethod
    def op_type(cls):
        return "linear"

    @classmethod
    def set_linear_type(cls, linear_type):
        cls.linear_type = linear_type

    @classmethod
    def set_linear_config(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"'{cls.__name__}' has no attribute '{key}'")

    @classmethod
    def register_linear(cls, name):
        def decorator(linear_class):
            cls._linear_registry[name] = linear_class
            return linear_class

        return decorator

    @classmethod
    def get_impl(cls, name=None):
        if name is None:
            name = cls.linear_type
        linear_class = cls._linear_registry.get(name)
        if linear_class is None:
            raise ValueError(f"Linear function {name} not found in registry")
        return linear_class()  # Create and return an instance

    @classmethod
    def get_registered_types(cls):
        return list(cls._linear_registry.keys())


@contextmanager
def linear_op_context(linear_type: str):
    original_linear_type = LinearOpManager.linear_type
    LinearOpManager.linear_type = linear_type
    try:
        yield
    finally:
        LinearOpManager.linear_type = original_linear_type


@contextmanager
def attention_op_context(attention_type: str):
    original_attn_type = AttentionOpManager.attn_type
    AttentionOpManager.attn_type = attention_type
    try:
        yield
    finally:
        AttentionOpManager.attn_type = original_attn_type
