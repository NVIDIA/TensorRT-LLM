# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from utils.llm_data import llm_models_root


def get_model_path(model_name):
    engine_dir = os.environ.get("LLM_ENGINE_DIR", None)
    if engine_dir:
        return engine_dir
    return str(llm_models_root() / model_name)


default_model_name = "Qwen3/Qwen3-0.6B"


qwen3_tokenizer_model_name = "Qwen3/Qwen3-0.6B"


llama_model_path = get_model_path(default_model_name)
