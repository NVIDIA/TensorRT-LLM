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
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as multiprocessing

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "examples/gpt"))
from build import get_engine_name, run_build  # isort:skip
from hf_gpt_convert import ProgArgs, run_conversion

END_ID = 50256
PAD_ID = 50256

work_dir = Path(__file__).parent.resolve() / 'check_gpt'

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from util import getSMVersion


def build_engine(weigth_dir: Path, engine_dir: Path, *args):
    print(
        f"== Build engine from {weigth_dir} to {engine_dir}, with args {args}")
    run_build([
        '--model_dir',
        str(weigth_dir),
        '--output_dir',
        str(engine_dir),
        '--log_level=verbose',
        '--max_batch_size=256',
        '--max_input_len=40',
        '--max_output_len=20',
        '--max_beam_width=2',
        '--builder_opt=0',
    ] + list(args))


def build_engines():
    llm_models_root = os.environ.get("LLM_MODELS_ROOT", None)
    gpt2_dir = f"{llm_models_root}/gpt2"
    # clone if not exists in the cache or the cache is not set
    if llm_models_root is None or not os.path.exists(gpt2_dir):
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

    weight_dir = work_dir / 'c-model/gpt2'
    engine_dir = work_dir / 'rt_engine/gpt2'

    print("\nConverting to fp32")
    fp32_weight_dir = weight_dir / 'fp32/1-gpu'
    run_conversion(
        ProgArgs(in_file=str(gpt2_dir),
                 out_dir=str(fp32_weight_dir),
                 storage_type='float32'))

    print("\nBuilding fp32 engines")
    fp32_weight_dir_1_gpu = fp32_weight_dir / '1-gpu'
    build_engine(fp32_weight_dir_1_gpu, engine_dir / 'fp32-default/1-gpu',
                 '--dtype=float32')
    build_engine(fp32_weight_dir_1_gpu, engine_dir / 'fp32-plugin/1-gpu',
                 '--dtype=float32', '--use_gpt_attention_plugin=float32')

    print("\nConverting to fp16")
    fp16_weight_dir = weight_dir / 'fp16/1-gpu'
    run_conversion(
        ProgArgs(in_file=str(gpt2_dir),
                 out_dir=str(fp16_weight_dir),
                 storage_type='float16'))

    print("\nBuilding fp16 engines")
    fp16_weight_dir_1_gpu = fp16_weight_dir / '1-gpu'
    build_engine(fp16_weight_dir_1_gpu, engine_dir / 'fp16-default/1-gpu',
                 '--dtype=float16', '--strongly_typed')
    build_engine(fp16_weight_dir_1_gpu, engine_dir / 'fp16-plugin/1-gpu',
                 '--dtype=float16', '--use_gpt_attention_plugin=float16',
                 '--strongly_typed')

    # Skip tests that are not supported in pre-ampere architecture
    if getSMVersion() >= 80:
        build_engine(fp16_weight_dir_1_gpu,
                     engine_dir / 'fp16-plugin-fmha/1-gpu', '--dtype=float16',
                     '--use_gpt_attention_plugin=float16',
                     '--enable_context_fmha', '--strongly_typed')

    build_engine(fp16_weight_dir_1_gpu, engine_dir / 'fp16-plugin-packed/1-gpu',
                 '--dtype=float16', '--use_gpt_attention_plugin=float16',
                 '--remove_input_padding', '--strongly_typed')

    # Skip tests that are not supported in pre-ampere architecture
    if getSMVersion() >= 80:
        build_engine(fp16_weight_dir_1_gpu,
                     engine_dir / 'fp16-plugin-packed-fmha/1-gpu',
                     '--dtype=float16', '--use_gpt_attention_plugin=float16',
                     '--remove_input_padding', '--enable_context_fmha',
                     '--strongly_typed')

    print("Done.")


def check_accuracy(engine_dir, input_tokens, max_output_len):
    config_path = os.path.join(engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    dtype = config['builder_config']['precision']
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    num_kv_heads = config['builder_config']['num_kv_heads']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding,
                               dtype=dtype)
    sampling_config = SamplingConfig(end_id=END_ID, pad_id=END_ID)

    engine_name = get_engine_name('gpt', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(engine_dir, engine_name)
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    input_lengths = torch.tensor([len(x) for x in input_tokens],
                                 dtype=torch.int,
                                 device='cuda')
    num_samples = len(input_tokens)

    expect_output = None

    for j, batch_size in enumerate([1, 2, 4, 8, 4, 2, 1]):
        output = []
        output_with_fake_dim = []
        print(f"Running batch size: {batch_size}")
        for i in range(num_samples // batch_size):
            samples = input_tokens[i * batch_size:(i + 1) * batch_size]
            sample_lengths = input_lengths[i * batch_size:(i + 1) * batch_size]
            if remove_input_padding:
                input_ids = np.concatenate(samples)
                input_ids = torch.tensor(input_ids,
                                         dtype=torch.int,
                                         device='cuda')
                input_ids_with_fake_dim = input_ids.unsqueeze(0)
                max_input_length = torch.max(sample_lengths).item()
            else:
                input_ids = torch.nested.to_padded_tensor(
                    torch.nested.nested_tensor(samples, dtype=torch.int32),
                    PAD_ID).cuda()
                max_input_length = input_ids.size(1)

            decoder.setup(batch_size, max_input_length, max_output_len)
            output_ids = decoder.decode(input_ids, sample_lengths,
                                        sampling_config)
            torch.cuda.synchronize()

            if remove_input_padding:
                decoder.setup(batch_size, max_input_length, max_output_len)
                output_ids_with_fake_dim = decoder.decode(
                    input_ids_with_fake_dim, sample_lengths, sampling_config)
                outputs_with_fake_dim_list = [
                    output_ids_with_fake_dim[
                        batch_idx, :,
                        sample_lengths[batch_idx]:sample_lengths[batch_idx] +
                        max_output_len].cpu()
                    for batch_idx in range(output_ids_with_fake_dim.shape[0])
                ]
                outputs_with_fake_dim = torch.cat(outputs_with_fake_dim_list)
                output_with_fake_dim.append(outputs_with_fake_dim)

            outputs_list = [
                output_ids[batch_idx, :,
                           sample_lengths[batch_idx]:sample_lengths[batch_idx] +
                           max_output_len].cpu()
                for batch_idx in range(output_ids.shape[0])
            ]
            outputs = torch.cat(outputs_list)
            output.append(outputs)
        output = torch.stack(output, dim=0)
        if remove_input_padding:
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

    # Skip tests that are not supported in pre-ampere architecture
    if getSMVersion() >= 80:
        check_output(engine='fp16-plugin-fmha')

    check_output(engine='fp16-plugin-packed')

    # Skip tests that are not supported in pre-ampere architecture
    if getSMVersion() >= 80:
        check_output(engine='fp16-plugin-packed-fmha')


class TestGPTE2E(unittest.TestCase):

    def test_check_gpt_e2e(self):
        multiprocessing.set_start_method("spawn")
        build_engines()
        check_outputs()


if __name__ == '__main__':
    unittest.main()
