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

DATASET = "Magpie-Align/Magpie-Pro-MT-300K-v0.1"


def create_trtllm_magpie_calibration_dataset(output_dir: str,
                                             calib_size: int = 512) -> None:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train", trust_remote_code=True)

    def transform(conversation):
        value = '\n'.join(turn['value']
                          for turn in conversation['conversations'])
        return {"text": value}

    dataset = dataset.select(range(calib_size)).map(
        transform, remove_columns=dataset.column_names)
    # https://github.com/huggingface/datasets/issues/6703#issuecomment-1974766332
    dataset.to_parquet(output_dir + "/data.parquet")


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1]
    create_trtllm_magpie_calibration_dataset(output_dir)
