# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from tensorrt_llm import SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from ....conftest import llm_models_root
from ...accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ._engine import shortcut_text_engine


@pytest.mark.skip_less_device(4)
class ShortcutTextAccuracy(LlmapiAccuracyTestHarness):
    def evaluate_gsm8k(self):
        with shortcut_text_engine(
            Path(llm_models_root()) / self.MODEL_NAME,
            enable_chunked_prefill=False,
            max_seq_len=4352,
            max_num_tokens=8192,
            max_batch_size=64,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=64),
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True, enable_block_reuse=False),
        ) as llm:
            GSM8K(self.MODEL_NAME).evaluate(
                llm,
                extra_evaluator_kwargs={"num_fewshot": 5, "apply_chat_template": False},
                sampling_params=SamplingParams(
                    temperature=0, max_tokens=256, truncate_prompt_tokens=4096
                ),
            )
