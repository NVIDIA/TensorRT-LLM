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
from unittest import mock

import pytest

import tensorrt_llm.models.convert_utils as convert_utils
from tensorrt_llm.models.convert_utils import load_calib_dataset


def test_bare_cnn_dailymail_is_namespaced() -> None:
    # Newer huggingface_hub rejects the bare "cnn_dailymail" repo id; it must
    # be rewritten to the relocated "abisee/cnn_dailymail" (issue #15802/#16124).
    with mock.patch.object(
        convert_utils, "load_dataset", return_value={"article": ["x"]}
    ) as load_dataset:
        load_calib_dataset("cnn_dailymail")

    assert load_dataset.call_args.args[0] == "abisee/cnn_dailymail"


# ccdv/cnn_dailymail is the important boundary: the already-qualified id must
# NOT be rewritten — only the bare "cnn_dailymail" maps to abisee/cnn_dailymail.
@pytest.mark.parametrize("dataset_name", ["lambada", "ccdv/cnn_dailymail"])
def test_other_dataset_ids_are_passed_through(dataset_name: str) -> None:
    with mock.patch.object(
        convert_utils, "load_dataset", return_value={"text": ["x"], "article": ["x"]}
    ) as load_dataset:
        load_calib_dataset(dataset_name)

    assert load_dataset.call_args.args[0] == dataset_name
