# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import copy
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from accelerate.utils import modeling
from transformers import AutoModelForCausalLM
from transformers.models.llama4.configuration_llama4 import Llama4Config
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_NAME

from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    hf_load_state_dict_with_device,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


@pytest.fixture(autouse=True)
def restore_custom_model_mapping():
    old_mapping = copy.copy(AutoModelForCausalLMFactory._custom_model_mapping)
    yield
    AutoModelForCausalLMFactory._custom_model_mapping = old_mapping


@pytest.fixture
def mock_factory():
    with (
        patch.object(AutoModelForCausalLMFactory, "prefetch_checkpoint"),
        patch.object(AutoModelForCausalLMFactory, "_load_quantization_config"),
    ):
        # Create factory instance with mocked methods to avoid HTTP requests
        factory = AutoModelForCausalLMFactory(model="dummy_model")
        # Set model path directly to avoid prefetch
        factory._prefetched_model_path = "/dummy/path"
        yield factory


def test_get_checkpoint_files_from_single_file_folder(mock_factory, tmp_path):
    checkpoint_file = tmp_path / SAFE_WEIGHTS_NAME
    checkpoint_file.touch()

    assert mock_factory._get_checkpoint_files(tmp_path) == [str(checkpoint_file)]


def test_get_checkpoint_files_from_pytorch_bin_folder(mock_factory, tmp_path):
    checkpoint_file = tmp_path / WEIGHTS_NAME
    checkpoint_file.touch()

    assert mock_factory._get_checkpoint_files(tmp_path) == [str(checkpoint_file)]


def test_get_checkpoint_files_from_sharded_index_are_sorted(mock_factory, tmp_path):
    shard_1 = tmp_path / "model-00001-of-00002.safetensors"
    shard_2 = tmp_path / "model-00002-of-00002.safetensors"
    index = {
        "weight_map": {
            "model.layers.1.weight": shard_2.name,
            "model.layers.0.weight": shard_1.name,
        }
    }
    (tmp_path / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")

    assert mock_factory._get_checkpoint_files(tmp_path) == [str(shard_1), str(shard_2)]


def test_hf_load_state_dict_with_device():
    original_load_state_dict = MagicMock()

    with patch.object(modeling, "load_state_dict", original_load_state_dict):
        with hf_load_state_dict_with_device(device="cpu"):
            modeling.load_state_dict("dummy_checkpoint")
            original_load_state_dict.assert_called_once_with(
                "dummy_checkpoint", device_map={"": "cpu"}
            )
            original_load_state_dict.reset_mock()

        modeling.load_state_dict("dummy_checkpoint", device_map="original_device_map")
        original_load_state_dict.assert_called_once_with(
            "dummy_checkpoint", device_map="original_device_map"
        )
        original_load_state_dict.reset_mock()

    if torch.cuda.is_available():
        with patch.object(modeling, "load_state_dict", original_load_state_dict):
            with hf_load_state_dict_with_device(device="cuda"):
                modeling.load_state_dict("dummy_checkpoint")
                original_load_state_dict.assert_called_once_with(
                    "dummy_checkpoint", device_map={"": "cuda"}
                )


def test_disable_preload_uses_accelerate_loader(mock_factory):
    model = SimpleModel()
    ckpt_file = "/dummy/path/model.safetensors.index.json"

    with (
        patch.object(mock_factory, "_get_checkpoint_file", return_value=ckpt_file) as get_mock,
        patch("tensorrt_llm._torch.auto_deploy.models.hf.load_checkpoint_in_model") as load_mock,
    ):
        mock_factory._load_checkpoint(model, "cpu", disable_preload=True)

    get_mock.assert_called_once_with(mock_factory.model)
    load_mock.assert_called_once_with(model, checkpoint=ckpt_file, full_state_dict=False)


def test_recursive_update_config(mock_factory):
    """Test that _recursive_update_config correctly updates a config object recursively."""
    # Get the mocked factory instance
    factory = mock_factory

    # Create a Llama4Config instance
    config = Llama4Config()

    # Create an update dictionary with both simple and nested values
    # NOTE: In transformers 5.x, bos_token_id moved into text_config for
    # Llama4Config, so use boi_token_index (a root-level attribute) instead.
    update_dict = {
        "boi_token_index": 42,  # Simple value at root level
        "text_config": {  # Nested config update
            "hidden_size": 4096,
            "num_attention_heads": 32,
        },
        "vision_config": {  # Another nested config update
            "hidden_size": 1024,
            "image_size": 224,
        },
        "non_existent_key": "this should be ignored",  # This key doesn't exist in the config
    }

    # Apply the recursive update
    updated_config, nested_unused = factory._recursive_update_config(config, update_dict)

    # Check that it returns the same object
    assert updated_config is config

    # Check root level updates
    assert config.boi_token_index == 42

    # Check nested updates in text_config
    assert config.text_config.hidden_size == 4096
    assert config.text_config.num_attention_heads == 32

    # Check nested updates in vision_config
    assert config.vision_config.hidden_size == 1024
    assert config.vision_config.image_size == 224

    # Check that non-existent keys were ignored
    assert not hasattr(config, "non_existent_key")
    # Check that nested_unused contains the non-existent key
    assert nested_unused == {"non_existent_key": "this should be ignored"}

    # Create a more complex update with deeper nesting
    complex_update = {"text_config": {"rope_scaling": {"factor": 2.0, "type": "linear"}}}

    # Apply the recursive update again
    factory._recursive_update_config(config, complex_update)

    # Check that complex nested updates were applied correctly
    assert config.text_config.rope_scaling["factor"] == 2.0
    assert config.text_config.rope_scaling["type"] == "linear"


def test_register_custom_model_cls():
    config_cls_name = "FooConfig"
    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=config_cls_name, custom_model_cls=custom_model_cls
    )

    assert AutoModelForCausalLMFactory._custom_model_mapping[config_cls_name] == custom_model_cls


class MyError(Exception):
    pass


# Needed for `type(config)` calls.
class FooConfig:
    pass


def test_build_model_raises_when_custom_model_cls_does_not_have_from_config(mock_factory):
    custom_model_cls = MagicMock(spec=AutoModelForCausalLM, __name__="FooModel")
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    with (
        patch.object(
            AutoModelForCausalLMFactory,
            "_get_model_config",
            return_value=(FooConfig(), {}),
        ),
        pytest.raises(ValueError, match=r"from_config"),
    ):
        mock_factory.build_model(device="meta")


def test_build_model_uses_custom_model_cls_from_config(mock_factory):
    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    custom_model_cls.configure_mock(_from_config=MagicMock(side_effect=MyError))
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    with (
        patch.object(
            AutoModelForCausalLMFactory,
            "_get_model_config",
            return_value=(FooConfig(), {}),
        ),
        pytest.raises(MyError),
    ):
        mock_factory.build_model(device="meta")


def test_custom_model_mapping_in_parent_does_not_affect_children():
    class Child(AutoModelForCausalLMFactory):
        pass

    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    custom_model_cls.configure_mock(_from_config=MagicMock(side_effect=MyError))
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    assert Child._custom_model_mapping == {}


def test_custom_model_mapping_in_parent_does_not_affect_parent():
    class Child(AutoModelForCausalLMFactory):
        pass

    parent_mapping = copy.copy(AutoModelForCausalLMFactory._custom_model_mapping)

    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    custom_model_cls.configure_mock(_from_config=MagicMock(side_effect=MyError))
    Child.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    assert AutoModelForCausalLMFactory._custom_model_mapping == parent_mapping
