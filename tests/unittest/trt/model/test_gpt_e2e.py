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
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.multiprocessing as multiprocessing

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

END_ID = 50256
PAD_ID = 50256

work_dir = Path(__file__).parent.resolve() / 'check_gpt'

from utils.llm_data import llm_models_root

gpt_example_root = os.path.join(os.path.dirname(__file__),
                                '../../../../examples/models/core/gpt')


def run_command(command: Sequence[str], *, cwd=None, **kwargs) -> None:
    print(f"Running: cd %s && %s" % (str(cwd or Path.cwd()), " ".join(command)),
          flush=True)
    subprocess.check_call(command, cwd=cwd, **kwargs)


def convert_ckpt(model_dir: str, output_dir: str, *args):
    convert_cmd = [
        sys.executable, f"{gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={model_dir}", f"--output_dir={output_dir}"
    ] + list(args)
    run_command(convert_cmd)


def build_engine(checkpoint_dir: str, engine_dir: str, *args):
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={checkpoint_dir}",
        f"--output_dir={engine_dir}",
        '--log_level=verbose',
        '--max_batch_size=256',
        '--max_input_len=40',
        '--max_seq_len=60',
        '--max_beam_width=2',
    ]
    legacy_args = [
        "--gpt_attention_plugin=disable",
        "--context_fmha=disable",
        "--paged_kv_cache=disable",
        "--remove_input_padding=disable",
    ]
    build_cmd = build_cmd + legacy_args + list(args)
    run_command(build_cmd)


def build_engines():
    models_root = llm_models_root()
    gpt2_dir = str(models_root / "gpt2")
    # clone if not exists in the cache or the cache is not set
    if models_root is None or not os.path.exists(gpt2_dir):
        gpt2_dir = work_dir / 'gpt2'
        print("Pulling gpt2 from huggingface")
        subprocess.check_call(["rm", "-rf", str(gpt2_dir)])
        subprocess.check_call(
            ["git", "clone", "https://huggingface.co/gpt2",
             str(gpt2_dir)])
        pytorch_model = str(gpt2_dir / "model.safetensors")
        subprocess.check_call(["rm", pytorch_model])
        subprocess.check_call([
            "wget", "-q",
            "https://huggingface.co/gpt2/resolve/main/model.safetensors", "-O",
            pytorch_model
        ])

    ckpt_dir = work_dir / 'c-model/gpt2'
    engine_dir = work_dir / 'rt_engine/gpt2'

    print("\nConverting to fp32")
    fp32_ckpt_dir = ckpt_dir / 'fp32/1-gpu'
    convert_ckpt(str(gpt2_dir), str(fp32_ckpt_dir), "--dtype=float32")

    print("\nBuilding fp32 engines")

    build_engine(str(fp32_ckpt_dir), str(engine_dir / 'fp32-default/1-gpu'))
    build_engine(str(fp32_ckpt_dir), str(engine_dir / 'fp32-plugin/1-gpu'),
                 '--gpt_attention_plugin=float32')

    print("\nConverting to fp16")
    fp16_ckpt_dir = ckpt_dir / 'fp16/1-gpu'
    convert_ckpt(str(gpt2_dir), str(fp16_ckpt_dir), "--dtype=float16")

    print("\nBuilding fp16 engines")
    build_engine(str(fp16_ckpt_dir), str(engine_dir / 'fp16-default/1-gpu'))
    build_engine(str(fp16_ckpt_dir), str(engine_dir / 'fp16-plugin/1-gpu'),
                 '--gpt_attention_plugin=float16')

    build_engine(str(fp16_ckpt_dir), str(engine_dir / 'fp16-plugin-fmha/1-gpu'),
                 '--gpt_attention_plugin=float16', '--context_fmha=enable')

    build_engine(str(fp16_ckpt_dir),
                 str(engine_dir / 'fp16-plugin-packed/1-gpu'),
                 '--gpt_attention_plugin=float16',
                 '--remove_input_padding=enable')

    build_engine(fp16_ckpt_dir,
                 str(engine_dir / 'fp16-plugin-packed-fmha/1-gpu'),
                 '--gpt_attention_plugin=float16',
                 '--remove_input_padding=enable', '--context_fmha=enable')

    print("Done.")


def check_accuracy(engine_dir, input_tokens, max_output_len):
    runtime_rank = tensorrt_llm.mpi_rank()
    runner = ModelRunner.from_dir(engine_dir, rank=runtime_rank)
    sampling_config = SamplingConfig(end_id=END_ID, pad_id=END_ID)

    all_input_ids = [torch.tensor(x, dtype=torch.int32) for x in input_tokens]
    all_input_lengths = [len(x) for x in input_tokens]
    num_samples = len(input_tokens)

    expect_output = None

    for j, batch_size in enumerate([1, 2, 4, 8, 4, 2, 1]):
        output = []
        output_with_fake_dim = []
        print(f"Running batch size: {batch_size}")
        for i in range(num_samples // batch_size):
            batch_input_ids = all_input_ids[i * batch_size:(i + 1) * batch_size]
            batch_input_lengths = all_input_lengths[i * batch_size:(i + 1) *
                                                    batch_size]
            max_input_length = max(batch_input_lengths)
            output_ids = runner.generate(batch_input_ids,
                                         sampling_config=sampling_config,
                                         max_new_tokens=max_output_len)
            torch.cuda.synchronize()

            if runner.remove_input_padding:
                runner.session.setup(batch_size, max_input_length,
                                     max_output_len)
                batch_input_ids_with_fake_dim = torch.concat(
                    batch_input_ids).unsqueeze(0)

                output_ids_with_fake_dim = runner.session.decode(
                    batch_input_ids_with_fake_dim.cuda(),
                    torch.tensor(batch_input_lengths, dtype=torch.int32).cuda(),
                    sampling_config)
                outputs_with_fake_dim_list = [
                    output_ids_with_fake_dim[batch_idx, :,
                                             batch_input_lengths[batch_idx]:
                                             batch_input_lengths[batch_idx] +
                                             max_output_len].cpu()
                    for batch_idx in range(output_ids_with_fake_dim.shape[0])
                ]
                outputs_with_fake_dim = torch.cat(outputs_with_fake_dim_list)
                output_with_fake_dim.append(outputs_with_fake_dim)

            outputs_list = [
                output_ids[batch_idx, :, batch_input_lengths[batch_idx]:
                           batch_input_lengths[batch_idx] +
                           max_output_len].cpu()
                for batch_idx in range(output_ids.shape[0])
            ]
            outputs = torch.cat(outputs_list)
            output.append(outputs)
        output = torch.stack(output, dim=0)
        if runner.remove_input_padding:
            output_with_fake_dim = torch.stack(output_with_fake_dim, dim=0)
            error = np.mean(output.cpu().numpy().flatten() !=
                            output_with_fake_dim.cpu().numpy().flatten())
            assert error < 2.0 / 8, f"diff at batch_size={batch_size}, output_with_fake_dim={output_with_fake_dim}, output={output}"

        if j == 0:
            expect_output = output

        if expect_output is not None:
            error = np.mean(output.cpu().numpy().flatten() !=
                            expect_output.cpu().numpy().flatten())
            assert error < 0.3, f"diff at batch_size={batch_size}, expect_output={expect_output}, output={output}"


def check_output(engine: str, max_output_len: int = 8):
    engine_dir = work_dir / 'rt_engine/gpt2' / engine / '1-gpu/'
    print(f"== Checking output for engine: {engine_dir}")

    input_tokens = [
        [28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257],
        [28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355],
        [28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776],
        [28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263],
        [28524, 287, 5093, 12, 23316, 4881, 11, 30022],
        [28524, 287, 5093, 12, 23316, 4881, 11],
        [28524, 287, 5093, 12, 23316, 4881],
        [28524, 287, 5093, 12, 23316],
    ]

    check_accuracy(engine_dir, input_tokens, max_output_len)


def check_outputs():
    check_output(engine='fp32-default')
    check_output(engine='fp32-plugin')
    check_output(engine='fp16-default')
    check_output(engine='fp16-plugin')
    check_output(engine='fp16-plugin-fmha')
    check_output(engine='fp16-plugin-packed')
    check_output(engine='fp16-plugin-packed-fmha')


class TestGPTE2E(unittest.TestCase):

    def test_check_gpt_e2e(self):
        multiprocessing.set_start_method("spawn")
        build_engines()
        check_outputs()


if __name__ == '__main__':
    unittest.main()
