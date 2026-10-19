# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from tensorrt_llm.inputs.model_config import resolve_top_level_model_type


def test_resolve_top_level_model_type_prefers_composite_class_attribute():
    class CompositeConfig:
        model_type = "top_level"

        def __init__(self):
            self.model_type = "text_model"

    assert resolve_top_level_model_type(CompositeConfig()) == "top_level"


def test_resolve_top_level_model_type_uses_instance_fallback():
    assert resolve_top_level_model_type(SimpleNamespace(model_type="model")) == "model"
    assert resolve_top_level_model_type(SimpleNamespace(model_type=None)) == ""
