#!/usr/bin/env python3
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

from pathlib import Path

import numpy as np
import run
import torch


def generate_output(
    model_name: str = "",
    engine_kind: str = "fp16-plugin",
    num_batchs: int = 1,
    num_beams: int = 1,
    max_output_len: int = 512,
    output_logits: bool = False,
):

    examples_chatglm_dir = Path(
        __file__).parent.parent.parent.parent.parent / "examples/chatglm"
    resources_dir = Path(__file__).parent.parent.resolve()

    engine_dir = resources_dir / 'models' / 'rt_engine' / model_name
    '''
    # we do not distinguish TP / PP / engine_kind yet
    tp_size = 1
    pp_size = 1
    tp_pp_dir = 'tp' + str(tp_size) + '-pp' + str(pp_size) + '-gpu/'
    engine_dir = engine_dir / engine_kind / tp_pp_dir
    '''
    data_output_dir = resources_dir / 'data' / model_name
    data_output_dir.mkdir(exist_ok=True, parents=True)
    data_input_file_name = f"inputId-BS{num_batchs}-BM{num_beams}.npy"
    data_output_file_name = f"outputId-BS{num_batchs}-BM{num_beams}.npy"
    input_text = [
        "Born in north-east France, Soyer trained as a",
        "Jen-Hsun Huang was born in Tainan, Taiwan, in 1963. His family",
    ]

    if num_batchs <= 2:
        input_text = input_text[:num_batchs]
    else:
        input_text = input_text + input_text[-1] * (num_batchs - 2)

    args = run.parse_arguments([
        '--engine_dir',
        str(engine_dir),
        '--tokenizer_dir',
        str(examples_chatglm_dir / model_name),
        '--input_text',
        *input_text,
        '--output_npy',
        str(data_output_dir / data_output_file_name),
        '--max_output_len',
        str(max_output_len),
        '--num_beams',
        str(num_beams),
    ])

    # Since main in run.py does not save input_ids, we save it manually
    model_name, model_version = run.read_model_name(args.engine_dir)
    tokenizer, pad_id, end_id = run.load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        model_name=model_name,
        model_version=model_version,
    )
    batch_input_ids = run.parse_input(
        tokenizer,
        input_text=input_text,
        prompt_template=None,
        input_file=None,
        add_special_tokens=True,
        max_input_length=512,
        pad_id=pad_id,
        num_prepend_vtokens=[],
        model_name=model_name,
        model_version=model_version,
    )
    input_len = [x.size(0) for x in batch_input_ids]
    max_input_len = max(input_len)

    batch_input_ids_padding = torch.zeros([num_batchs, max_input_len],
                                          dtype=torch.int32) + end_id

    for i, sample in enumerate(batch_input_ids):
        # padding to left
        batch_input_ids_padding[i, :len(sample)] = sample
        """
        # padding to right
        nPadding = 0
        for token in sample:
            if token == pad_id:
                nPadding += 1
            else:
                break
        batch_input_ids_padding[i, :len(sample[nPadding:])] = sample[nPadding:]
        """
    batch_input_ids = batch_input_ids_padding

    np.save(data_output_dir / data_input_file_name,
            batch_input_ids.detach().cpu().numpy())

    run.main(args)

    output_data = np.load(args.output_npy)
    np.save(args.output_npy, output_data.reshape(num_batchs, num_beams, -1))


if __name__ == '__main__':

    generate_output(model_name='chatglm_6b', num_batchs=1, num_beams=1)

    generate_output(model_name='chatglm2_6b', num_batchs=1, num_beams=1)
    generate_output(model_name='chatglm2_6b', num_batchs=2, num_beams=1)
    generate_output(model_name='chatglm2_6b', num_batchs=1, num_beams=2)

    generate_output(model_name='chatglm3_6b', num_batchs=1, num_beams=1)
    generate_output(model_name='chatglm3_6b', num_batchs=2, num_beams=1)
    generate_output(model_name='chatglm3_6b', num_batchs=1, num_beams=2)

    print("Done")
