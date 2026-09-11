# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tensorrt_llm.sampling_params import SamplingParams


def test_bad_token_ids_single_tokens() -> None:
    sp = SamplingParams(bad_token_ids=[5, 7])
    assert sp._get_bad_words() == [[5], [7]]


def test_bad_token_ids_multi_token_sequences() -> None:
    # A user can now block a multi-token bad word explicitly.
    sp = SamplingParams(bad_token_ids=[[10, 11], 5])
    assert sp._get_bad_words() == [[10, 11], [5]]
