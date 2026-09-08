# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for building scaffolding controllers from a config file.

These tests construct controllers only; no model is loaded and no worker runs,
so they are CPU-only and need no model cache.
"""

import json

import pytest

from tensorrt_llm.scaffolding import (
    BestOfNController,
    ChatWithMCPController,
    Controller,
    MajorityVoteController,
    NativeChatController,
    NativeGenerationController,
    NativeRewardController,
    PRMController,
    build_controller,
    load_controller_config,
    register_controller,
)


class TestBuildFlatController:
    def test_sampling_params_round_trip(self):
        controller = build_controller(
            {
                "type": "NativeGenerationController",
                "args": {"sampling_params": {"max_tokens": 1024, "temperature": 0.9}},
            }
        )

        assert isinstance(controller, NativeGenerationController)
        assert controller.sampling_params == {"max_tokens": 1024, "temperature": 0.9}
        assert controller.streaming is False

    def test_args_key_is_optional(self):
        controller = build_controller({"type": "NativeRewardController"})
        assert isinstance(controller, NativeRewardController)

    def test_plain_dict_arg_is_not_treated_as_a_controller(self):
        """A dict without a "type" key stays a dict."""
        controller = build_controller(
            {
                "type": "NativeGenerationController",
                "args": {"sampling_params": {"max_tokens": 4, "top_k": 50}},
            }
        )
        assert isinstance(controller.sampling_params, dict)

    def test_dict_arg_containing_a_type_key_is_rejected(self):
        """Reserved key: a data dict carrying "type" is read as a controller node.

        Documented in the config module. Failing loudly here is deliberate -- it keeps a
        mistyped controller name from being silently passed through as data.
        """
        with pytest.raises(ValueError, match="unknown controller type"):
            build_controller(
                {
                    "type": "NativeGenerationController",
                    "args": {"sampling_params": {"type": "json_object"}},
                }
            )


class TestBuildNestedController:
    def test_majority_vote_wraps_generation_controller(self):
        controller = build_controller(
            {
                "type": "MajorityVoteController",
                "args": {
                    "default_sample_num": 3,
                    "generation_controller": {
                        "type": "NativeGenerationController",
                        "args": {"sampling_params": {"max_tokens": 8}},
                    },
                },
            }
        )

        assert isinstance(controller, MajorityVoteController)
        assert controller.default_sample_num == 3
        assert isinstance(controller.generation_controller, NativeGenerationController)
        assert controller.generation_controller.sampling_params == {"max_tokens": 8}

    def test_best_of_n_builds_both_sub_controllers(self):
        controller = build_controller(
            {
                "type": "BestOfNController",
                "args": {
                    "default_sample_num": 4,
                    "generation_controller": {"type": "NativeGenerationController"},
                    "reward_controller": {"type": "NativeRewardController"},
                },
            }
        )

        assert isinstance(controller, BestOfNController)
        assert isinstance(controller.generation_controller, NativeGenerationController)
        assert isinstance(controller.reward_controller, NativeRewardController)

    def test_three_levels_of_nesting(self):
        controller = build_controller(
            {
                "type": "MajorityVoteController",
                "args": {
                    "generation_controller": {
                        "type": "BestOfNController",
                        "args": {
                            "generation_controller": {"type": "NativeChatController"},
                            "reward_controller": {"type": "NativeRewardController"},
                        },
                    }
                },
            }
        )

        inner = controller.generation_controller
        assert isinstance(inner, BestOfNController)
        assert isinstance(inner.generation_controller, NativeChatController)


class TestLiveObjectReferences:
    def test_prm_controller_receives_tokenizer(self):
        """Controllers needing live objects get them through {"$ref": ...}."""
        tokenizer = object()

        controller = build_controller(
            {
                "type": "PRMController",
                "args": {"tokenizer": {"$ref": "tokenizer"}, "split_steps": False},
            },
            objects={"tokenizer": tokenizer},
        )

        assert isinstance(controller, PRMController)
        assert controller.tokenizer is tokenizer
        assert controller.split_steps is False

    def test_chat_with_mcp_controller_receives_tools(self):
        tools = [{"name": "search"}]

        controller = build_controller(
            {
                "type": "ChatWithMCPController",
                "args": {
                    "generation_controller": {"type": "NativeGenerationController"},
                    "tools": {"$ref": "tools"},
                    "max_iterations": 5,
                },
            },
            objects={"tools": tools},
        )

        assert isinstance(controller, ChatWithMCPController)
        assert controller.tools is tools
        assert controller.max_iterations == 5

    def test_unresolved_reference_is_reported(self):
        with pytest.raises(KeyError, match="unknown object reference"):
            build_controller(
                {"type": "PRMController", "args": {"tokenizer": {"$ref": "missing"}}},
                objects={},
            )


class TestConfigErrors:
    def test_unknown_type_lists_registered_types(self):
        with pytest.raises(ValueError, match="unknown controller type"):
            build_controller({"type": "NoSuchController"})

    def test_missing_type_key(self):
        with pytest.raises(ValueError, match="missing required"):
            build_controller({"args": {}})

    def test_unexpected_constructor_argument(self):
        with pytest.raises(TypeError, match="cannot construct NativeGenerationController"):
            build_controller({"type": "NativeGenerationController", "args": {"not_a_real_arg": 1}})

    def test_nested_failure_reports_the_path(self):
        with pytest.raises(ValueError, match=r"controller\.args\.generation_controller"):
            build_controller(
                {
                    "type": "MajorityVoteController",
                    "args": {"generation_controller": {"type": "Bogus"}},
                }
            )

    def test_stray_top_level_key_is_rejected(self):
        with pytest.raises(ValueError, match="unexpected key"):
            build_controller({"type": "NativeRewardController", "sampling_params": {}})


class TestRegisterController:
    def test_registered_controller_can_be_built(self):
        class _MyController(Controller):
            def __init__(self, k: int = 1):
                super().__init__()
                self.k = k

        register_controller("_MyController", _MyController)
        controller = build_controller({"type": "_MyController", "args": {"k": 7}})

        assert isinstance(controller, _MyController)
        assert controller.k == 7

    def test_non_controller_is_rejected(self):
        with pytest.raises(TypeError):
            register_controller("_NotAController", dict)

    def test_duplicate_name_with_different_class_is_rejected(self):
        class _First(Controller):
            pass

        class _Second(Controller):
            pass

        register_controller("_Duplicate", _First)
        register_controller("_Duplicate", _First)  # same class is idempotent
        with pytest.raises(ValueError, match="already registered"):
            register_controller("_Duplicate", _Second)


class TestLoadControllerConfig:
    def test_load_from_json(self, tmp_path):
        spec = {
            "controller": {
                "type": "MajorityVoteController",
                "args": {
                    "default_sample_num": 3,
                    "generation_controller": {
                        "type": "NativeGenerationController",
                        "args": {"sampling_params": {"max_tokens": 1024}},
                    },
                },
            }
        }
        path = tmp_path / "controller.json"
        path.write_text(json.dumps(spec, indent=2))

        controller = load_controller_config(str(path))

        assert isinstance(controller, MajorityVoteController)
        assert controller.default_sample_num == 3

    def test_load_from_yaml(self, tmp_path):
        """JSON and YAML share one code path via yaml.safe_load."""
        path = tmp_path / "controller.yaml"
        path.write_text(
            "controller:\n"
            "  type: BestOfNController\n"
            "  args:\n"
            "    default_sample_num: 4\n"
            "    generation_controller:\n"
            "      type: NativeGenerationController\n"
            "      args:\n"
            "        sampling_params: {max_tokens: 512}\n"
            "    reward_controller:\n"
            "      type: NativeRewardController\n"
        )

        controller = load_controller_config(str(path))

        assert isinstance(controller, BestOfNController)
        assert controller.default_sample_num == 4
        assert controller.generation_controller.sampling_params == {"max_tokens": 512}

    def test_missing_top_level_controller_key(self, tmp_path):
        path = tmp_path / "controller.json"
        path.write_text(json.dumps({"nope": {}}))

        with pytest.raises(ValueError, match="missing required top-level"):
            load_controller_config(str(path))
