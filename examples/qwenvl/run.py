# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import json
import os
from typing import List, Tuple

import tensorrt as trt
import torch
from transformers import AutoConfig, AutoTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (ModelConfig, SamplingConfig, Session,
                                  TensorInfo)


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size,
                                                  pp_size, rank)


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


class QWenInfer(object):

    def __init__(self, tokenizer_dir, qwen_engine_dir, log_level, output_csv,
                 output_npy, num_beams):
        self.tokenizer_dir = tokenizer_dir
        self.qwen_engine_dir = qwen_engine_dir
        self.log_level = log_level
        self.global_max_input_len = 2048
        self.decoder = None
        self.tokenizer = None
        self.config = None
        self.sampling_config = None
        self.output_csv = output_csv
        self.output_npy = output_npy
        self.num_beams = num_beams
        self.model_config = None

    def get_model(self):
        # --load the tokenizer and engine #
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            legacy=False,
            trust_remote_code=True,
        )
        config_path = os.path.join(self.qwen_engine_dir, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        gen_config_path = os.path.join(self.tokenizer_dir,
                                       'generation_config.json')
        with open(gen_config_path, 'r') as f:
            gen_config = json.load(f)
        top_k = gen_config['top_k']
        top_p = gen_config['top_p']
        chat_format = gen_config['chat_format']
        if chat_format == "raw":
            eos_token_id = gen_config['eos_token_id']
            pad_token_id = gen_config['pad_token_id']
        elif chat_format == "chatml":
            pad_token_id = eos_token_id = tokenizer.im_end_id
        else:
            raise Exception("unknown chat format ", chat_format)

        use_gpt_attention_plugin = config['plugin_config'][
            'gpt_attention_plugin']
        remove_input_padding = config['plugin_config']['remove_input_padding']
        dtype = config['builder_config']['precision']
        tp_size = config['builder_config']['tensor_parallel']
        pp_size = config['builder_config']['pipeline_parallel']
        world_size = tp_size * pp_size
        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
        num_heads = config['builder_config']['num_heads'] // world_size
        hidden_size = config['builder_config']['hidden_size'] // world_size
        vocab_size = config['builder_config']['vocab_size']
        num_layers = config['builder_config']['num_layers']
        num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
        paged_kv_cache = config['plugin_config']['paged_kv_cache']
        tokens_per_block = config['plugin_config']['tokens_per_block']
        max_prompt_embedding_table_size = config['builder_config'].get(
            'max_prompt_embedding_table_size', 0)
        quant_mode = QuantMode(config['builder_config']['quant_mode'])
        if config['builder_config'].get('multi_query_mode', False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine."
            )
            num_kv_heads = 1
        # num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
        use_custom_all_reduce = config['plugin_config'].get(
            'use_custom_all_reduce', False)

        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size=world_size,
                                               rank=runtime_rank,
                                               tp_size=tp_size,
                                               pp_size=pp_size)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        model_config = ModelConfig(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            remove_input_padding=remove_input_padding,
            dtype=dtype,
            quant_mode=quant_mode,
            use_custom_all_reduce=use_custom_all_reduce,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )
        sampling_config = SamplingConfig(
            end_id=eos_token_id,
            pad_id=pad_token_id,
            num_beams=self.num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=1.0,
        )

        engine_name = get_engine_name('qwen', dtype, tp_size, pp_size,
                                      runtime_rank)
        serialize_path = os.path.join(self.qwen_engine_dir, engine_name)
        print(f'Loading engine from {serialize_path}')
        return (model_config, sampling_config, runtime_mapping, runtime_rank,
                serialize_path, tokenizer, eos_token_id, pad_token_id)

    def qwen_model_init(self):
        (model_config, sampling_config, runtime_mapping, runtime_rank,
         serialize_path, tokenizer, eos_token_id,
         pad_token_id) = self.get_model()
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()
        self.decoder = tensorrt_llm.runtime.GenerationSession(
            model_config,
            engine_buffer,
            runtime_mapping,
        )
        self.tokenizer = tokenizer
        self.sampling_config = sampling_config
        self.model_config = model_config
        self.config, _ = AutoConfig.from_pretrained(
            self.tokenizer_dir,
            return_unused_kwargs=True,
            trust_remote_code=True,
        )

    def ptuning_setup(self, prompt_table, dtype, hidden_size, tasks, input_ids):
        if prompt_table is not None:
            task_vocab_size = torch.tensor([prompt_table.shape[1]],
                                           dtype=torch.int32,
                                           device="cuda")
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))
            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if tasks is not None:
            tasks = torch.tensor([int(t) for t in tasks.split(',')],
                                 dtype=torch.int32,
                                 device="cuda")
            assert tasks.shape[0] == input_ids.shape[
                0], "Number of supplied tasks must match input batch size"
        else:
            tasks = torch.zeros([input_ids.size(0)], dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def make_context(
        self,
        query: str,
        history: List[Tuple[str, str]] = None,
        system: str = "You are a helpful assistant.",
        max_window_size: int = 6144,
    ):
        if history is None:
            history = []

        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [self.tokenizer.im_start_id]  # 151644
        im_end_tokens = [self.tokenizer.im_end_id]  # [151645]
        nl_tokens = self.tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", self.tokenizer.encode(
                role, allowed_special=set(self.tokenizer.IMAGE_ST)
            ) + nl_tokens + self.tokenizer.encode(
                content, allowed_special=set(self.tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (len(system_tokens) +
                                    len(next_context_tokens) +
                                    len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (nl_tokens + im_start_tokens +
                           _tokenize_str("user", query)[1] + im_end_tokens +
                           nl_tokens + im_start_tokens +
                           self.tokenizer.encode("assistant") + nl_tokens)
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text, context_tokens

    def generate_for_qwenvl(
        self,
        input_tokens,
        max_new_tokens: int,
        prompt_table=None,
        tasks=None,
        task_vocab_size=None,
    ):
        input_ids = None
        input_lengths = None
        input_ids = torch.as_tensor(input_tokens,
                                    device="cuda",
                                    dtype=torch.int32)
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device="cuda",
                                     dtype=torch.int32)
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(max_new_tokens,
                             self.global_max_input_len - max_input_length)
        self.decoder.setup(batch_size=input_lengths.size(0),
                           max_context_length=max_input_length,
                           max_new_tokens=max_new_tokens)
        profiler.start("QWen")
        run_time = 1
        for _ in range(run_time):
            output_ids = self.decoder.decode(input_ids, input_lengths,
                                             self.sampling_config, prompt_table,
                                             tasks, task_vocab_size)
            torch.cuda.synchronize()
        profiler.stop("QWen")
        Qwen_time = profiler.elapsed_time_in_sec("QWen") / run_time

        return output_ids, Qwen_time

    def qwen_infer(self,
                   input_vit,
                   images_path,
                   input_text,
                   max_new_tokens,
                   history=None):
        if images_path is None:
            content_list = []
        else:
            content_list = images_path
        if history is None:
            history = []
        content_list.append({'text': input_text})
        query = self.tokenizer.from_list_format(content_list)
        raw_text, context_tokens = self.make_context(query, history=history)
        # context_tokens = self.tokenizer.encode(query)
        input_ids = torch.tensor([context_tokens]).to('cuda')
        bos_pos = torch.where(input_ids == self.config.visual['image_start_id'])
        eos_pos = torch.where(
            input_ids == self.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        vocab_size = self.config.vocab_size
        fake_prompt_id = torch.arange(vocab_size,
                                      vocab_size +
                                      input_vit.shape[0] * input_vit.shape[1],
                                      device='cuda')
        fake_prompt_id = fake_prompt_id.reshape(input_vit.shape[0],
                                                input_vit.shape[1])
        for idx, (i, a, b) in enumerate(img_pos):
            input_ids[i][a + 1:b] = fake_prompt_id[idx]
        input_ids = input_ids.contiguous().to(torch.int32).cuda()
        input_lengths = torch.tensor(input_ids.size(1),
                                     dtype=torch.int32).cuda()
        dtype = self.model_config.dtype
        prompt_table, tasks, task_vocab_size = self.ptuning_setup(
            input_vit, dtype, self.config.hidden_size, None, input_ids)

        output_ids, Qwen_time = self.generate_for_qwenvl(
            input_ids, max_new_tokens, prompt_table, tasks, task_vocab_size)

        runtime_rank = tensorrt_llm.mpi_rank()
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device="cuda",
                                     dtype=torch.int32)
        effective_output_token = 0
        if runtime_rank == 0:
            if self.output_csv is None and self.output_npy is None:
                for b in range(input_lengths.size(0)):
                    inputs = input_ids[b]
                    if content_list is not None:
                        print(f'Input: \"{content_list}\"')
                        print("\n")
                    if self.num_beams <= 1:
                        outputs = output_ids[b][0, len(inputs):].tolist()
                        try:
                            effective_output_token = effective_output_token + \
                                outputs.index(151643)
                        except:
                            effective_output_token = 1
                        output_text = self.tokenizer.decode(
                            outputs, skip_special_tokens=True)
                        print(f'Output: \"{output_text}\"')
                        print("\n")
                    else:
                        for beam in range(self.num_beams):
                            outputs = output_ids[b][beam, len(inputs):].tolist()
                            output_text = self.tokenizer.decode(
                                outputs, skip_special_tokens=True)
                            print(f'Output(beam: {beam}): \"{output_text}\"')

        logger.info(f'TensorRT-LLM QWen time: {Qwen_time} sec ')
        history.append((query, output_text))
        return output_text


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--vit_engine_dir',
        type=str,
        default='qwen_outputs',
    )
    parser.add_argument(
        '--qwen_engine_dir',
        type=str,
        default='qwen_outputs',
    )
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument('--input_text',
                        type=str,
                        default="Describe the picture")
    parser.add_argument('--images_path',
                        type=list,
                        default=[{
                            'image': './pics/demo.jpeg'
                        }])
    parser.add_argument('--input_dir',
                        type=list,
                        default=[{
                            'image': 'image.pt'
                        }])

    parser.add_argument(
        '--input_tokens',
        dest='input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument("--display", default=False, action='store_true')
    parser.add_argument('--port', type=str, default='8006')
    parser.add_argument("--local_machine", default=False, action='store_true')
    return parser.parse_args()


def vit_process(image_path, engine_dir, stream):
    vit_path = os.path.join(engine_dir,
                            'visual_encoder/visual_encoder_fp16.plan')
    logger.info(f'Loading engine from {vit_path}')
    with open(vit_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine {vit_path}')
    session_vit = Session.from_serialized_engine(engine_buffer)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    images_list = []
    for img in image_path:
        for v in img.values():
            image = torch.load(v)
            if image.device.type == 'cpu':
                image = image.to(device)
            images_list.append(image)
    images = torch.cat(images_list)
    batch_size = images.size(0)
    images = images.expand(batch_size, -1, -1, -1).contiguous()
    visual_inputs = {'input': images.float()}
    visual_output_info = session_vit.infer_shapes(
        [TensorInfo('input', trt.DataType.FLOAT, images.shape)])
    visual_outputs = {
        t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
        for t in visual_output_info
    }
    profiler.start("ViT")

    run_time = 1
    for _ in range(run_time):
        ok = session_vit.run(visual_inputs, visual_outputs, stream)
    profiler.stop("ViT")
    Vit_time = profiler.elapsed_time_in_sec("ViT") / run_time
    logger.info(f'TensorRT-LLM ViT latency: {Vit_time} sec ')

    assert ok, "Runtime execution failed for vit session"

    image_embeds = visual_outputs['output']
    return image_embeds


if __name__ == '__main__':
    args = parse_arguments()
    stream = torch.cuda.current_stream().cuda_stream
    tensorrt_llm.logger.set_level(args.log_level)
    image_embeds = vit_process(args.input_dir, args.vit_engine_dir, stream)
    qinfer = QWenInfer(args.tokenizer_dir, args.qwen_engine_dir, args.log_level,
                       args.output_csv, args.output_npy, args.num_beams)
    qinfer.qwen_model_init()
    qinfer.qwen_infer(image_embeds,
                      args.images_path,
                      args.input_text,
                      args.max_new_tokens,
                      history=[])
