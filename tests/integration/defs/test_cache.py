# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
'''
    Test LLM_MODELS_ROOT has correct data for misc test fixture
'''
import pytest

from .conftest import llm_models_root


@pytest.mark.skipif(llm_models_root() is None,
                    reason="no need to check the cache if its not set")
def test_cache_sanity(
    gpt_next_root,
    llm_gpt2_model_root,
    llm_gpt2_medium_model_root,
    llm_gpt2_next_model_root,
    llm_gpt2_santacoder_model_root,
    llm_gpt2_starcoder_model_root,
    llm_gpt2_starcoder2_model_root,
    llm_gpt2_next_8b_model_root,
    llm_qwen_7b_model_root,
):
    # use this test to be a placeholder to trigger the execution of all the test fixture
    # and this is only executed when the LLM_MODELS_ROOT is set, so it will not trigger any download of the models
    pass
