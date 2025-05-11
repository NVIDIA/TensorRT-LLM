# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from run import QWenInfer, parse_arguments

import tensorrt_llm

if __name__ == '__main__':
    args = parse_arguments()
    stream = torch.cuda.current_stream()
    tensorrt_llm.logger.set_level(args.log_level)
    qinfer = QWenInfer(
        args.audio_engine_path,
        args.tokenizer_dir,
        args.engine_dir,
        args.log_level,
        args.output_csv,
        args.output_npy,
        args.num_beams,
    )
    qinfer.qwen_model_init(args)

    run_i = 0
    history = []
    audios = None
    global_audio_id = 1
    audio_ids = []

    while True:
        input_text = None
        try:
            input_text = input(
                "Text (type 'q' to quit, or 'audio_url:[url]' to input audio): "
            )
        except:
            continue

        if input_text == "clear history":
            history = []
            audios = None
            continue

        if input_text.lower() == 'q':
            break
        print('\n')

        if input_text.startswith('audio_url:'):
            audio_url = input_text[len('audio_url:'):].strip()
            if isinstance(audios, list):
                audios.extend(qinfer.get_raw_audios([audio_url]))
            else:
                audios = qinfer.get_raw_audios([audio_url])
            user_input = qinfer.build_user_input(audio=audio_url)
            audio_ids.append(global_audio_id)
            global_audio_id += 1
        else:
            user_input = qinfer.build_user_input(text=input_text)

        qinfer.qwen_infer(
            user_input,
            audios,
            audio_ids,
            args,
            stream,
            history,
        )
