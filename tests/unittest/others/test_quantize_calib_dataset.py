# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import tempfile

import torch
from utils.util import skip_no_modelopt


@skip_no_modelopt
def test_is_cnn_dailymail_local_repo():
    from tensorrt_llm.quantization.quantize_by_modelopt import _is_cnn_dailymail_local_repo

    # Non-existent path
    assert not _is_cnn_dailymail_local_repo("/does/not/exist")

    # Empty directory
    with tempfile.TemporaryDirectory() as d:
        assert not _is_cnn_dailymail_local_repo(d)

    # 3.0.0 subdir
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "3.0.0"))
        assert _is_cnn_dailymail_local_repo(d)

    # Other versions not detected
    for version in ("1.0.0", "2.0.0"):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, version))
            assert not _is_cnn_dailymail_local_repo(d), (
                f"version subdir {version} should not be detected"
            )

    # Directory with the cnn_dailymail.py builder script
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "cnn_dailymail.py"), "w").close()
        assert _is_cnn_dailymail_local_repo(d)

    # Directory with unrelated content
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "train"))
        open(os.path.join(d, "data.parquet"), "w").close()
        assert not _is_cnn_dailymail_local_repo(d)


@skip_no_modelopt
def test_get_calib_dataloader_local_cnn_dailymail(monkeypatch):
    from tensorrt_llm.quantization import quantize_by_modelopt

    captured = {}

    def fake_load_dataset(path, **kwargs):
        captured["kwargs"] = kwargs
        return {"article": ["calibration article"] * 2}

    def fake_tokenizer(dataset, **kwargs):
        return {"input_ids": torch.ones(len(dataset), 4, dtype=torch.long)}

    monkeypatch.setattr(quantize_by_modelopt, "load_dataset", fake_load_dataset)

    with tempfile.TemporaryDirectory() as d:
        # Name lacks "cnn_dailymail".
        os.makedirs(os.path.join(d, "3.0.0"))
        dataloader = quantize_by_modelopt.get_calib_dataloader(
            dataset_name_or_dir=d,
            tokenizer=fake_tokenizer,
            calib_size=2,
        )

    assert captured["kwargs"].get("name") == "3.0.0"
    assert len(list(dataloader)) == 2
