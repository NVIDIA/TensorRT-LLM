import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests
import tensorrt as trt
import torch
from PIL import Image
from transformers import (AutoConfig, AutoTokenizer,
                          Blip2ForConditionalGeneration, Blip2Processor)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo

sys.path.append(str(Path(__file__).parent.parent))
from enc_dec.run import TRTLLMEncDecModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--visual_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing visual TRT engines')
    parser.add_argument('--llm_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing TRT-LLM engines')
    parser.add_argument('--hf_model_dir',
                        type=str,
                        default=None,
                        help="Directory containing tokenizer")
    parser.add_argument(
        '--decoder_llm',
        action='store_true',
        help='Whether LLM is decoder-only or an encoder-decoder variant?')
    parser.add_argument('--blip_encoder',
                        action='store_true',
                        help='Whether visual encoder is a BLIP model')
    parser.add_argument('--input_text',
                        type=str,
                        default='Question: which city is this? Answer:',
                        help='Text prompt to LLM')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--top_k', type=int, default=1)

    return parser.parse_args()


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


class MultiModalModel:

    def __init__(self, args):
        self.args = args

        runtime_rank = tensorrt_llm.mpi_rank()
        device_id = runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.stream = torch.cuda.current_stream().cuda_stream

        self.init_image_encoder()
        self.init_tokenizer()
        self.init_llm()

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_dir,
                                                       use_fast=False,
                                                       use_legacy=False)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_image_encoder(self):
        vit_path = os.path.join(self.args.visual_engine_dir,
                                'visual_encoder_fp16.engine')
        logger.info(f'Loading engine from {vit_path}')
        with open(vit_path, 'rb') as f:
            engine_buffer = f.read()
        logger.info(f'Creating session from engine {vit_path}')
        self.vit_session = Session.from_serialized_engine(engine_buffer)

        if self.args.blip_encoder:
            qformer_path = os.path.join(self.args.visual_engine_dir,
                                        'Qformer_fp16.engine')
            logger.info(f'Loading engine from {qformer_path}')
            with open(qformer_path, 'rb') as f:
                engine_buffer_qformer = f.read()
            logger.info(f'Creating session from engine {qformer_path}')
            self.vit_qformer = Session.from_serialized_engine(
                engine_buffer_qformer)

    def init_llm(self):
        if self.args.decoder_llm:
            self.model = ModelRunner.from_dir(self.args.llm_engine_dir,
                                              rank=tensorrt_llm.mpi_rank(),
                                              debug_mode=False)
            self.model_config = self.model.session._model_config
        else:
            self.model = TRTLLMEncDecModel.from_engine(
                self.args.hf_model_dir.split('/')[-1],
                self.args.llm_engine_dir,
                debug_mode=False)
            self.model_config = self.model.encoder_model_config

            hf_config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            self.decoder_input_ids = torch.IntTensor(
                [[hf_config.decoder_start_token_id]]).repeat(
                    (self.args.batch_size, 1)).to("cuda")

    def generate(self, pre_prompt, post_prompt, image, max_new_tokens):
        visual_features, visual_atts = self.get_visual_features(image)

        pre_input_ids = self.tokenizer(pre_prompt,
                                       return_tensors="pt",
                                       padding=True).input_ids.to("cuda")
        if post_prompt is not None:
            post_input_ids = self.tokenizer(post_prompt,
                                            return_tensors="pt",
                                            padding=True).input_ids.to("cuda")
            length = pre_input_ids.shape[1] + post_input_ids.shape[
                1] + visual_atts.shape[1]
        else:
            post_input_ids = None
            length = pre_input_ids.shape[1] + visual_atts.shape[1]

        input_atts = torch.ones((1, length)).to(torch.int32).to("cuda")
        input_lengths = torch.sum(input_atts, dim=1)

        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths)

        if self.args.decoder_llm:
            prompt_table = ptuning_args[0]
            prompt_table = torch.stack([prompt_table])
            np.save('prompt_table.npy', torch_to_numpy(prompt_table))

        profiler.start("LLM")
        if self.args.decoder_llm:
            output_ids = self.model.generate(
                input_ids.to("cpu"),
                sampling_config=None,
                prompt_table_path='prompt_table.npy',
                max_new_tokens=max_new_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                top_k=self.args.top_k,
                num_beams=self.args.num_beams,
                output_sequence_lengths=False,
                return_dict=False)
        else:
            output_ids = self.model.generate(
                input_ids,
                self.decoder_input_ids,
                max_new_tokens,
                num_beams=self.args.num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                debug_mode=False,
                prompt_embedding_table=ptuning_args[0],
                prompt_tasks=ptuning_args[1],
                prompt_vocab_size=ptuning_args[2])
            # Clear before batch decode in next step
            input_lengths = torch.zeros(input_lengths.shape,
                                        dtype=input_lengths.dtype)
        profiler.stop("LLM")

        if tensorrt_llm.mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True)
                for batch_idx in range(self.args.batch_size)
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(self.args.num_beams)
            ] for batch_idx in range(self.args.batch_size)]
            return stripped_text
        else:
            return None

    def get_visual_features(self, image):
        features, atts = self.vit_pass(image)
        if self.args.blip_encoder:
            features, atts = self.qformer_pass(features, atts)
        return features, atts

    def vit_pass(self, image):
        visual_features = {'input': image.half()}
        visual_output_info = self.vit_session.infer_shapes(
            [TensorInfo('input', trt.DataType.HALF, image.shape)])
        visual_outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device="cuda")
            for t in visual_output_info
        }

        ok = self.vit_session.run(visual_features, visual_outputs, self.stream)
        assert ok, "Runtime execution failed for vit session"

        image_embeds = visual_outputs['output']
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to("cuda")

        return image_embeds, image_atts

    def qformer_pass(self, image_embeds, image_atts):
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1,
                                                -1).contiguous().to("cuda")
        qformer_inputs = {
            'query_tokens': query_tokens.half(),
            'image_embeds': image_embeds.half(),
            'image_atts': image_atts
        }
        qformer_output_info = self.vit_qformer.infer_shapes([
            TensorInfo('query_tokens', trt.DataType.HALF, query_tokens.shape),
            TensorInfo('image_embeds', trt.DataType.HALF, image_embeds.shape),
            TensorInfo('image_atts', trt.DataType.INT64, image_atts.shape)
        ])
        qformer_outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device="cuda")
            for t in qformer_output_info
        }
        ok = self.vit_qformer.run(qformer_inputs, qformer_outputs, self.stream)
        assert ok, "Runtime execution failed for Qformer session"

        visual_features = qformer_outputs["query_output"]
        visual_atts = torch.ones(visual_features.size()[:-1],
                                 dtype=torch.long).to("cuda")

        return visual_features, visual_atts

    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids,
                           input_lengths):
        # Assemble fake prompts which points to image embedding actually
        fake_prompt_id = torch.arange(
            self.model_config.vocab_size,
            self.model_config.vocab_size +
            visual_features.shape[0] * visual_features.shape[1],
            device="cuda")
        fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0],
                                                visual_features.shape[1])

        if post_input_ids is not None:
            input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
        input_ids = torch.cat(input_ids,
                              dim=1).contiguous().to(torch.int32).cuda()

        if self.args.decoder_llm or self.model.encoder_runtime_mapping.is_first_pp_rank(
        ):
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        if prompt_table is not None:
            task_vocab_size = torch.tensor([prompt_table.shape[1]],
                                           dtype=torch.int32,
                                           device="cuda")
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))

            hidden_size = self.model_config.hidden_size
            if not self.args.decoder_llm:
                hidden_size *= self.model.encoder_runtime_mapping.tp_size
            assert prompt_table.shape[
                1] == hidden_size, "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(
                    self.model_config.dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)],
                                dtype=torch.int32).cuda()
            if args.decoder_llm: tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]


def setup_llava_prompt(query):
    # Import these here to avoid installing llava when running blip models only
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    query = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv_mode = 'llava_v1'
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_token_index = prompt.find(DEFAULT_IMAGE_TOKEN)
    pre_prompt = prompt[:image_token_index]
    post_prompt = prompt[image_token_index + len(DEFAULT_IMAGE_TOKEN):]

    return pre_prompt, post_prompt


def load_test_image():
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
    return Image.open(requests.get(img_url, stream=True).raw).convert('RGB')


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    runtime_rank = tensorrt_llm.mpi_rank()

    image = load_test_image()
    if args.blip_encoder:
        if 'opt-2.7b' in args.hf_model_dir:
            model_type = 'Salesforce/blip2-opt-2.7b'
        else:
            model_type = 'Salesforce/blip2-flan-t5-xl'

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        processor = Blip2Processor.from_pretrained(model_type)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_type, torch_dtype=torch.float16)
        model.to(device)

        prompt = "Question: which city is this in? Answer:"
        inputs = processor(image, prompt, return_tensors="pt").to(device)
        image = inputs['pixel_values']
        image = image.expand(args.batch_size, -1, -1,
                             -1).contiguous().to("cuda")

        query_tokens = model.query_tokens

        pre_prompt = [args.input_text] * args.batch_size
        post_prompt = None
    else:
        pre_prompt, post_prompt = setup_llava_prompt(args.input_text)

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        model_path = 'liuhaotian/llava-v1.5-7b'
        model_name = get_model_name_from_path(model_path)
        _, _, image_processor, _ = load_pretrained_model(
            model_path, None, model_name)

        image = image_processor(image, return_tensors='pt')['pixel_values']
        image = image.half().to("cuda")

        query_tokens = None

    model = MultiModalModel(args)
    model.query_tokens = query_tokens

    num_iters = 100
    for _ in range(num_iters):
        stripped_text = model.generate(pre_prompt, post_prompt, image,
                                       args.max_new_tokens)

    if runtime_rank == 0:
        logger.info("---------------------------------------------------------")
        logger.info(f"\n[Q] {args.input_text}")
        logger.info(f"\n[A] {stripped_text}")
        logger.info(
            f'TensorRT-LLM LLM latency: {profiler.elapsed_time_in_sec("LLM") / num_iters} sec'
        )
        logger.info("---------------------------------------------------------")
