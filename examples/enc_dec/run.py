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
import argparse
import json
import time
from pathlib import Path

# isort: off
import torch
import tensorrt as trt
# isort: on
import numpy as np
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          BartForConditionalGeneration,
                          MBartForConditionalGeneration,
                          T5ForConditionalGeneration)

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import torch_to_numpy, trt_dtype_to_torch
from tensorrt_llm.lora_manager import LoraManager
from tensorrt_llm.runtime import ModelConfig, SamplingConfig


def get_engine_name(rank):
    return 'rank{}.engine'.format(rank)


def print_tensor(tensor_name, tensor, num_elements=10):
    if tensor.dtype in (torch.int32, torch.int64):
        tensor = tensor.to(dtype=float)
    print(
        f'{tensor_name}: mean={tensor.abs().mean().item():.3f}, sum={tensor.abs().sum().item():.3f}, max={tensor.abs().max().item():.3f}'
    )
    # Pass num_elements=-1 will print the whole tensor
    if num_elements < 0:
        num_elements = torch.numel(tensor)
    print(f'{tensor.flatten()[:num_elements]}')
    print("Tensor Shape: ", tensor.size())
    print("")


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)

    builder_config = config['build_config']
    plugin_config = builder_config['plugin_config']
    pretrained_config = config['pretrained_config']
    lora_config = builder_config['lora_config']
    auto_parallel_config = builder_config['auto_parallel_config']
    use_gpt_attention_plugin = plugin_config["gpt_attention_plugin"]
    remove_input_padding = plugin_config["remove_input_padding"]
    use_lora_plugin = plugin_config["lora_plugin"]
    tp_size = pretrained_config['mapping']['tp_size']
    pp_size = pretrained_config['mapping']['pp_size']
    gpus_per_node = auto_parallel_config['gpus_per_node']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = pretrained_config["num_attention_heads"]
    hidden_size = pretrained_config["hidden_size"]
    head_size = pretrained_config["head_size"]
    vocab_size = pretrained_config["vocab_size"]
    max_batch_size = builder_config["max_batch_size"]
    max_beam_width = builder_config["max_beam_width"]
    num_layers = pretrained_config["num_hidden_layers"]
    num_kv_heads = pretrained_config.get('num_kv_heads', num_heads)

    assert (num_heads % tp_size) == 0
    num_heads = num_heads // tp_size
    hidden_size = hidden_size // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    cross_attention = pretrained_config["architecture"] == "DecoderModel"
    skip_cross_qkv = pretrained_config.get('skip_cross_qkv', False)
    has_position_embedding = pretrained_config["has_position_embedding"]
    has_token_type_embedding = hasattr(pretrained_config, "type_vocab_size")
    use_custom_all_reduce = plugin_config.get('use_custom_all_reduce', False)
    dtype = pretrained_config["dtype"]

    paged_kv_cache = plugin_config['paged_kv_cache']
    tokens_per_block = plugin_config['tokens_per_block']

    gather_context_logits = builder_config.get('gather_context_logits', False)
    gather_generation_logits = builder_config.get('gather_generation_logits',
                                                  False)
    max_prompt_embedding_table_size = builder_config.get(
        'max_prompt_embedding_table_size', 0)

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        lora_plugin=use_lora_plugin,
        lora_target_modules=lora_config.get('lora_target_modules'),
        trtllm_modules_to_hf_modules=lora_config.get(
            'trtllm_modules_to_hf_modules'),
        skip_cross_qkv=skip_cross_qkv,
    )

    return model_config, tp_size, pp_size, gpus_per_node, dtype


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--engine_dir", "-i", type=str, default="trt_engines")
    parser.add_argument("--engine_name", type=str, default="enc_dec")
    parser.add_argument("--model_name",
                        type=str,
                        help="HuggingFace model name or FairSeq model path",
                        default="t5-small")
    parser.add_argument("--num_beams",
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument("--debug_mode",
                        help="Whether or not to turn on the debug mode",
                        action='store_true')
    parser.add_argument("--compare_hf_fp32",
                        help="Compare results with HuggingFace FP32",
                        action='store_true')
    parser.add_argument('--lora_dir', type=str, default=None, nargs="+")
    parser.add_argument('--lora_task_uids', type=str, default=None, nargs="+")
    parser.add_argument("--output_npy",
                        type=str,
                        default=None,
                        help="Store input/output tensors C++ runtime testing")
    return parser.parse_args()


class TRTLLMEncDecModel:

    def __init__(
        self,
        engine_name,
        engine_dir,
        lora_dir=None,
        lora_task_uids=None,
        debug_mode=False,
        skip_encoder=False,
        stream: torch.cuda.Stream = None,
    ):
        # in multi-node setup, it's important to set_device at the very beginning so .to('cuda') refers to current device
        # accordingly, all input & output tensors should be moved to current device
        # otherwise, it's default to 'cuda:0'
        self.runtime_rank = tensorrt_llm.mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = torch.cuda.current_device()
        self.skip_encoder = skip_encoder
        self.lora_task_uids = lora_task_uids

        # when enc-dec runs by itself, stream can be None and we create new stream here
        # when enc-dec has to run as a component in a bigger workflow (e.g., multimodal), earlier components in the workflow may have results in its stream, which we should pass that stream in to avoid unnecessary stream sync
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_dir = Path(engine_dir)

        def engine_setup(component):
            # model config
            config_path = engine_dir / component / "config.json"
            logger.info(f"Using config path {config_path}")
            model_config, tp_size, pp_size, gpus_per_node, dtype = read_config(
                config_path)

            # MGMN config
            world_size = tp_size * pp_size
            runtime_rank = tensorrt_llm.mpi_rank()
            assert runtime_rank < world_size, "Runtime GPU rank exceeds MPI world size. Did you launch more MPI processes than required?"
            runtime_mapping = tensorrt_llm.Mapping(world_size,
                                                   runtime_rank,
                                                   tp_size=tp_size,
                                                   pp_size=pp_size,
                                                   gpus_per_node=gpus_per_node)

            # load engine
            engine_fname = get_engine_name(runtime_rank)
            with open(engine_dir / component / engine_fname, "rb") as f:
                engine_buffer = f.read()

            return model_config, runtime_mapping, engine_buffer

        # Note: encoder and decoder doesn't necessarily have the same TP & PP config

        if not skip_encoder:
            self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = engine_setup(
                component='encoder')

            self.nccl_comm = None
            if self.encoder_runtime_mapping.has_pp():
                # for Pipeline Parallelism in encoder
                self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
                    self.encoder_runtime_mapping.tp_size,
                    self.encoder_runtime_mapping.pp_size,
                    self.encoder_runtime_mapping.rank)

            # session setup
            self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
                encoder_engine_buffer)

            # encoder lora manager setup
            if self.encoder_model_config.lora_plugin:
                self.encoder_lora_manager = LoraManager()
                # TODO: this is only for bart
                self.encoder_lora_manager.load_from_hf(
                    model_dirs=lora_dir,
                    model_config=self.encoder_model_config,
                    runtime_mapping=self.encoder_runtime_mapping,
                    component='encoder',
                )
            else:
                self.encoder_lora_manager = None
        else:
            self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = None, None, None
            self.nccl_comm, self.encoder_session = None, None

        self.decoder_model_config, self.decoder_runtime_mapping, decoder_engine_buffer = engine_setup(
            component='decoder')
        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            self.decoder_model_config,
            decoder_engine_buffer,
            self.decoder_runtime_mapping,
            debug_mode=debug_mode)

        # decoder lora manager setup
        if self.decoder_model_config.lora_plugin:
            self.decoder_lora_manager = LoraManager()
            # TODO: this is only for bart
            self.decoder_lora_manager.load_from_hf(
                model_dirs=lora_dir,
                model_config=self.decoder_model_config,
                runtime_mapping=self.decoder_runtime_mapping,
                component='decoder',
            )
        else:
            self.decoder_lora_manager = None

    @classmethod
    def from_engine(cls,
                    engine_name,
                    engine_dir,
                    lora_dir=None,
                    lora_task_uids=None,
                    debug_mode=False,
                    skip_encoder=False,
                    stream=None):
        return cls(engine_name,
                   engine_dir,
                   lora_dir,
                   lora_task_uids,
                   debug_mode=debug_mode,
                   skip_encoder=skip_encoder,
                   stream=stream)

    def process_input(self,
                      input_ids,
                      remove_input_padding=False,
                      pad_token_id=0,
                      prompt_tasks=None):
        if remove_input_padding:
            # in remove padding mode --> flatten input, calculate actual length and max length
            # Note: 1st token should never be removed, even if it is pad_token_id
            first_ids = input_ids[:, 0]
            input_ids = input_ids[:, 1:]
            input_lengths = 1 + (input_ids != pad_token_id).sum(dim=1).type(
                torch.IntTensor).to(self.device)  # [batch_size]
            new_ids = []
            for i in range(len(input_ids)):
                row = input_ids[i, :]
                row = row[row != pad_token_id]
                new_ids.append(
                    torch.cat(
                        (torch.IntTensor([first_ids[i]]).to(self.device), row)))
            input_ids = torch.cat(new_ids)  # [num_tokens]
            if prompt_tasks is not None:
                prompt_tasks = prompt_tasks[:input_ids.shape[0]]
        else:
            # in padding mode --> keep input, just calculate actual length and max length
            # Note: 1st token should always count, even if it is pad_token_id. e.g., decoder start id in enc-dec models could be a single pad_token_id, we should count
            input_lengths = torch.tensor(
                1 + (input_ids[:, 1:] != pad_token_id).sum(dim=1).type(
                    torch.IntTensor).to(self.device),
                dtype=torch.int32,
                device=self.device)
        max_input_length = torch.max(input_lengths).item()
        return input_ids, input_lengths, max_input_length, prompt_tasks

    def encoder_run(self,
                    input_ids,
                    input_lengths,
                    max_input_length,
                    position_ids=None,
                    token_type_ids=None,
                    debug_mode=False,
                    prompt_embedding_table=None,
                    prompt_tasks=None,
                    prompt_vocab_size=None,
                    attention_mask=None):

        # each engine has hidden_dim/TP, don't forget to multiply TP
        hidden_size = self.encoder_model_config.hidden_size * self.encoder_runtime_mapping.tp_size
        if input_ids.dim() == 1:
            hidden_states_shape = (input_ids.shape[0], hidden_size
                                   )  # [num_tokens,D]
        else:
            hidden_states_shape = (input_ids.shape[0], input_ids.shape[1],
                                   hidden_size)  # [BS,seqlen,D]
        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name))

        # input tensors. only first PP rank has id input, others are hidden_states input
        inputs = {}
        if self.encoder_runtime_mapping.is_first_pp_rank():
            inputs['input_ids'] = input_ids.contiguous()
            if self.encoder_model_config.has_position_embedding:
                if position_ids is None:
                    if self.encoder_model_config.remove_input_padding:
                        position_ids = [
                            torch.arange(sample_length,
                                         dtype=torch.int32,
                                         device=input_ids.device)
                            for sample_length in torch_to_numpy(input_lengths)
                        ]
                        position_ids = torch.cat(position_ids)
                    else:
                        bsz, seq_len = input_ids.shape[:2]
                        position_ids = torch.arange(
                            seq_len, dtype=torch.int32,
                            device=input_ids.device).expand(bsz, -1)
                inputs['position_ids'] = position_ids.contiguous()
            if self.encoder_model_config.has_token_type_embedding:
                inputs['token_type_ids'] = token_type_ids.contiguous()

            if self.encoder_model_config.max_prompt_embedding_table_size > 0:
                inputs[
                    'prompt_embedding_table'] = prompt_embedding_table.contiguous(
                    )
                inputs['tasks'] = prompt_tasks.contiguous()
                inputs['prompt_vocab_size'] = prompt_vocab_size.contiguous()
        else:
            # just need a placeholder, engine will call NCCL to recv and fill data from previous rank
            inputs['hidden_states_input'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_input'),
                device=self.device).contiguous()
        if attention_mask is not None and not self.encoder_model_config.gpt_attention_plugin:
            inputs['attention_mask'] = attention_mask.contiguous()

        inputs['input_lengths'] = input_lengths
        # use shape info to pass max length info in remove padding mode
        inputs['max_input_length'] = torch.empty(
            (max_input_length, ),
            dtype=hidden_states_dtype('max_input_length'),
            device=self.device).contiguous()
        batch_size = input_lengths.size(0)
        inputs['host_request_types'] = torch.IntTensor([0] *
                                                       batch_size).to('cpu')
        if self.encoder_model_config.remove_input_padding:
            inputs['host_context_lengths'] = input_lengths.to('cpu')

        if self.encoder_model_config.lora_plugin and self.encoder_lora_manager is not None:
            inputs.update(
                self.encoder_lora_manager.input_buffers(
                    self.lora_task_uids,
                    self.encoder_runtime_mapping,
                    self.encoder_model_config.num_layers,
                ))

        # Note: runtime.Session's run() method will set input/output tensor address, here we only need to provide tensor shape
        self.encoder_session.set_shapes(inputs)

        # output tensors. only last PP rank final encoder output, others are intermediate hidden_states output. Need broadcast later
        outputs = {}
        if self.encoder_runtime_mapping.is_last_pp_rank():
            outputs['encoder_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('encoder_output'),
                device=self.device).contiguous()
        else:
            outputs['hidden_states_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_output'),
                device=self.device).contiguous()

        # -------------------------------------------
        if debug_mode:
            engine = self.encoder_session.engine
            context = self.encoder_session.context
            # setup debugging buffer for the encoder
            for i in range(self.encoder_session.engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(
                        name
                ) == trt.TensorIOMode.OUTPUT and name not in outputs.keys():
                    dtype = engine.get_tensor_dtype(name)
                    shape = context.get_tensor_shape(name)
                    outputs[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device=self.device)
                    context.set_tensor_address(name, outputs[name].data_ptr())
        # -------------------------------------------

        # TRT session run
        # Note: need cuda stream ID, not a torch Stream
        ok = self.encoder_session.run(inputs, outputs, self.stream.cuda_stream)
        assert ok, "Runtime execution failed"
        self.stream.synchronize()

        # Tensor Parallelism is handled by model/engine definition
        # But we need to broadcast among PP group at the end of encoder's Pipeline Parallelism
        # After this, all ranks should recv the encoder output, and world might be re-configured using decoder's TP-PP config
        def pp_communicate_encoder_output(encoder_output):
            if self.encoder_runtime_mapping.is_last_pp_rank():
                for pp_rank in self.encoder_runtime_mapping.pp_group:
                    if pp_rank != self.encoder_runtime_mapping.rank:
                        self.nccl_comm.send(encoder_output, pp_rank)
                return encoder_output
            else:
                self.nccl_comm.recv(encoder_output,
                                    self.encoder_runtime_mapping.pp_group[-1])
                return encoder_output

        if self.encoder_runtime_mapping.has_pp():
            # use hidden_states output buffer to receive output as the shapes are same
            encoder_output_buf = outputs[
                'encoder_output'] if self.encoder_runtime_mapping.is_last_pp_rank(
                ) else outputs['hidden_states_output']
            encoder_output = pp_communicate_encoder_output(encoder_output_buf)
        else:
            encoder_output = outputs['encoder_output']

        # -------------------------------------------
        if debug_mode and self.encoder_runtime_mapping.tp_rank == 0:  # only tp_rank 0 print encoder output
            torch.cuda.synchronize()
            # use print_tensor() to print the tensors registered in the encoder network
            print("--------------------------------------")
            print("Debug output for Encoder")
            print("--------------------------------------")
            print("Registered output tensors are: ", outputs.keys())
            for k, v in outputs.items():
                print_tensor(k, v, num_elements=30)
            print_tensor('encoder_output', encoder_output)
            print("--------------------------------------")
        # -------------------------------------------

        return encoder_output

    def generate(self,
                 encoder_input_ids,
                 decoder_input_ids,
                 max_new_tokens,
                 num_beams=1,
                 pad_token_id=None,
                 eos_token_id=None,
                 bos_token_id=None,
                 debug_mode=False,
                 return_dict=False,
                 prompt_embedding_table=None,
                 prompt_tasks=None,
                 prompt_vocab_size=None,
                 attention_mask=None,
                 time_encoder=False,
                 return_encoder_output=False):
        ## ensure all externally provided tensors are on the correct device.
        encoder_input_ids = encoder_input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = torch.tensor(attention_mask,
                                          dtype=torch.int32,
                                          device=self.device)

        ## encoder run
        encoder_remove_input_padding = self.encoder_model_config.remove_input_padding if self.encoder_model_config else self.decoder_model_config.remove_input_padding

        encoder_input_ids, encoder_input_lengths, encoder_max_input_length, prompt_tasks = self.process_input(
            encoder_input_ids, encoder_remove_input_padding, pad_token_id,
            prompt_tasks)

        if not self.skip_encoder:
            logger.info(f"Rank {self.runtime_rank} Running encoder engine ...")
            if time_encoder:
                tik = time.time()
            encoder_output = self.encoder_run(
                encoder_input_ids,
                encoder_input_lengths,
                encoder_max_input_length,
                debug_mode=debug_mode,
                prompt_embedding_table=prompt_embedding_table,
                prompt_tasks=prompt_tasks,
                prompt_vocab_size=prompt_vocab_size,
                attention_mask=attention_mask)
            if time_encoder:
                tok = time.time()
                print(f"TRT-LLM Encoder time {(tok-tik)*1000}ms")
        else:
            encoder_output = prompt_embedding_table
            if encoder_input_ids.dim() > 1:
                encoder_output = encoder_output.unsqueeze(0)

        ## decoder run
        logger.info(f"Rank {self.runtime_rank} Running decoder engine ...")
        decoder_input_ids, decoder_input_lengths, decoder_max_input_length, _ = self.process_input(
            decoder_input_ids, self.decoder_model_config.remove_input_padding,
            pad_token_id)

        # `cross_attention_mask` in context phase [batch_size, query_len, encoder_input_len]
        # where query_len happens to be 1 in current cases, but not necessarily always, and
        # `cross_attention_mask` in generation phase [batch_size, 1, encoder_input_len] where
        # the query_len is always 1 since we have kv cache.
        cross_attention_mask = None
        if attention_mask is not None:
            cross_attention_mask = torch.tensor(attention_mask,
                                                dtype=torch.int32,
                                                device=self.device).reshape(
                                                    attention_mask.shape[0], 1,
                                                    attention_mask.shape[1])

        # generation config
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         min_length=1,
                                         return_dict=return_dict)
        sampling_config.update(output_cum_log_probs=return_dict,
                               output_log_probs=return_dict)

        # decoder autoregressive generation
        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            num_beams,
            max_attention_window_size=None,
            encoder_max_input_length=encoder_max_input_length,
            lora_manager=self.decoder_lora_manager,
            lora_uids=self.lora_task_uids,
        )

        output = self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=encoder_input_lengths,
            return_dict=return_dict,
            cross_attention_mask=cross_attention_mask)

        if return_dict and return_encoder_output:
            output['encoder_output'] = encoder_output

        return output


def test_fairseq_models(args):
    ## Note: NMT is the only FairSeq model. Adding FairSeq dependency is too heavy for the CI workflow, hence we used fixed input/output ids for correctness check and leave FairSeq code in comments. Users can follow Encoder-Decoder's README to install FairSeq and test locally.
    '''
        from fairseq.models.transformer import TransformerModel

        fairseq_model = TransformerModel.from_pretrained(model_name_or_path=args.model_name, data_name_or_path=args.model_name, bpe='subword_nmt', tokenizer='moses').cuda()

        input_text = "Good Morning! How are you doing today?"
        input_ids = fairseq_model.encode(input_text)

        tik = time.time()
        # Note: FairSeq sampling=True results are not deterministic, disable during accuracy check
        fairseq_output_ids = fairseq_model.generate(input_ids, beam=1, sampling=False) #
        tik = time.time()

        fairseq_output_ids = fairseq_output_ids[0]['tokens']
        fairseq_output_text = fairseq_model.decode(fairseq_output_ids)

        print("--------------------------------------")
        print("input text: ", input_text)
        print("input ids: ", input_ids) # [9938, 5384, 9328, 812, 3619, 53, 181, 3829, 1735, 171, 2]
        print("fairseq_output ids: ", fairseq_output_ids) # [9804, 391, 4, 4625, 167, 25, 1003, 5123, 17, 167, 1466, 1234, 171, 2]
        print("fairseq_output text: ", fairseq_output_text) # "Bonjour, Comment vous en tirez-vous aujourd'hui ?"
        print(f"FairSeq E2E time {(tok-tik)*1000}ms")
        print("--------------------------------------")
        '''

    max_new_tokens = args.max_new_tokens
    bos_token_id = 2
    pad_token_id = 0
    eos_token_id = 2
    decoder_start_token_id = bos_token_id

    input_ids = torch.tensor(
        [9938, 5384, 9328, 812, 3619, 53, 181, 3829, 1735, 171, 2])
    fairseq_output_ids = torch.tensor(
        [9804, 391, 4, 4625, 167, 25, 1003, 5123, 17, 167, 1466, 1234, 171, 2])
    input_ids = torch.tensor([input_ids.tolist()]).type(torch.IntTensor).cuda()
    decoder_input_ids = torch.IntTensor([[decoder_start_token_id]]).cuda()
    decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

    tllm_model = TRTLLMEncDecModel.from_engine(args.engine_name,
                                               args.engine_dir,
                                               debug_mode=args.debug_mode)

    inference_dtype = tllm_model.encoder_model_config.dtype

    return_dict = False  # when set return_dict=True, get outputs by key
    tik = time.time()
    tllm_output = tllm_model.generate(
        encoder_input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=args.num_beams,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        debug_mode=args.debug_mode,
    )
    torch.cuda.synchronize()
    tok = time.time()

    if return_dict:
        tllm_output_ids = tllm_output['output_ids']
    else:
        tllm_output_ids = tllm_output

    if tensorrt_llm.mpi_rank() == 0:
        output_ids = tllm_output_ids[:, 0, :]
        output_ids = output_ids[output_ids != eos_token_id]
        fairseq_output_ids = fairseq_output_ids[
            fairseq_output_ids != eos_token_id]

        print("--------------------------------------")
        print("TRT-LLM output_ids: ", output_ids)
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("Precision:", inference_dtype)
        print("--------------------------------------")

        assert output_ids.tolist() == fairseq_output_ids.tolist(
        ), f"TRT-LLM output ids {output_ids} does not match Fairseq ids {fairseq_output_ids}"


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    logger.set_level(args.log_level)

    # FairSeq NMT test logic is different from HuggingFace models
    if 'wmt' in args.model_name:
        test_fairseq_models(args)
        exit()

    test_remove_padding = True
    if not test_remove_padding:
        if 't5' in args.model_name:
            input_text = "translate English to German: The house is wonderful, radiating timeless charm and offering a warm, inviting interior with beautiful details and a serene backyard."
        elif 'bart' in args.model_name:
            input_text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
        else:
            raise RuntimeError('Unsupported model type!')

    else:
        input_text = [
            "translate English to German: The house is wonderful.",
            "summarize: I am a high-performance inference optimizer and runtime.",
            "During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world",
        ]

    # TRT-LLM runtime
    tllm_model = TRTLLMEncDecModel.from_engine(args.engine_name,
                                               args.engine_dir,
                                               args.lora_dir,
                                               args.lora_task_uids,
                                               debug_mode=args.debug_mode)

    inference_dtype = tllm_model.encoder_model_config.dtype
    if inference_dtype == 'float32':
        if "byt5" in args.model_name:
            print(
                "ByT5 models tokenize input by bytes instead of words, causing the input text in this example to be longer than the default value during build stage. Please adjust --max_input_len during trtllm-build to select the right length limit for ByT5 models."
            )
        else:
            input_text.append(
                "Summarize this article in one sentence.\n\nKristine Watts (Molie Weeks) is broken apart, missing her lover; she is not able to overcome her love for him that is lost in the past. She hires a stranger (Douglas Davis) and gives a list of her mistakes to him with things to fix. But time is irreversible and sometimes the cure for the pain is a tragic end.\n\nThe first point that impresses in \"The Cure\" is the stylish cinematography that alternates black and white with color. The concise and sharp screenplay is capable to develop a tragic and bleak tale of love with an unexpected plot point in the very end in less than eight minutes. The soundtrack is beautiful but the volume is a little loud and associated to the fact that English is not my native language, in some moments I needed to repeat some words whispered by the narrator. The unknown lead actress has magnificent performance and is extremely gorgeous. I hope to have a chance to see her again on the screen. Last but not the least, the debut of the director and writer Ryan Jafri could not be better. My vote is nine.\n\nTitle (Brazil): Not Available",
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name)  # TODO: use model path instead
    tokenized_inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    max_new_tokens = args.max_new_tokens
    input_ids = tokenized_inputs.input_ids.type(torch.IntTensor).to(
        'cuda')  # [batch_size, padded_length]
    # by default int64, must cast to int32! otherwise C++ kernel will interpret as [a, 0, b, 0, c, 0, ...]

    if tensorrt_llm.mpi_rank() == 0:
        print("--------------------------------------")
        print(
            f"BOS={tokenizer.bos_token_id}, PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}"
        )
        print("input text: ", input_text)
        print("input ids: ", input_ids)
        print("input lengths: ", tokenized_inputs.attention_mask.sum(dim=1))
        print("--------------------------------------")

    model_config = AutoConfig.from_pretrained(args.model_name)

    # start_id for decoder (could add more input_ids as forced_decoder_ids)
    decoder_input_ids = torch.IntTensor([[model_config.decoder_start_token_id]
                                         ]).to('cuda')
    decoder_input_ids = decoder_input_ids.repeat((input_ids.shape[0], 1))

    # simple comparison with HF on FP32
    if args.compare_hf_fp32:
        if tensorrt_llm.mpi_rank() == 0:
            hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name,  # TODO: use model path instead
                # torch_dtype=torch.float16 if '16' in dtype else torch.float32,  # TODO: use matched torch dtype
            ).to('cuda').eval()  # TODO: create config model path instead
            assert type(hf_model) in (
                T5ForConditionalGeneration, BartForConditionalGeneration,
                MBartForConditionalGeneration), 'Unsupported model!'

            if args.lora_dir is not None:
                assert len(args.lora_dir
                           ) >= 1, "At least one lora model dir is required"
                # we can only test single lora with HF
                from peft import PeftModel
                hf_model = PeftModel.from_pretrained(
                    hf_model, args.lora_dir[0]).to('cuda').eval()

            tik = time.time()
            hf_gen_output = hf_model.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=args.num_beams,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                # control logits processors
                no_repeat_ngram_size=0,  # disable no repeat post-processor
                forced_bos_token_id=None,  # disable forced first/last token
                forced_eos_token_id=None,
                min_length=0,
                # for debug
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True)
            # get hf output scores
            hf_output_ids = hf_gen_output.sequences
            # convert to logits
            torch.cuda.synchronize()
            tok = time.time()

            output_ids = hf_output_ids.squeeze(dim=1)
            hf_output_text = tokenizer.batch_decode(output_ids,
                                                    skip_special_tokens=True)
            decoder_input_lengths = (decoder_input_ids !=
                                     tokenizer.pad_token_id).sum(dim=1)
            output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
                dim=1) - decoder_input_lengths
            print("--------------------------------------")
            print("HF output_ids: ", output_ids)
            print("HF output text: ", hf_output_text)
            print("HF output generated lengths: ", output_gen_lengths)
            print(f"HF E2E time {(tok-tik)*1000}ms")
            print("--------------------------------------")

    return_dict = True  # when set return_dict=True, get outputs by key
    tik = time.time()
    tllm_output = tllm_model.generate(
        encoder_input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=args.num_beams,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        debug_mode=args.debug_mode,
        return_dict=return_dict,
        attention_mask=tokenized_inputs.attention_mask,
        time_encoder=True,
        return_encoder_output=args.output_npy and tensorrt_llm.mpi_rank() == 0)
    torch.cuda.synchronize()
    tok = time.time()

    if tensorrt_llm.mpi_rank() == 0:
        if return_dict:
            tllm_output_ids = tllm_output['output_ids']
        else:
            tllm_output_ids = tllm_output

        output_ids = tllm_output_ids[:, 0, :]
        output_text = tokenizer.batch_decode(output_ids,
                                             skip_special_tokens=True)
        decoder_input_lengths = (decoder_input_ids !=
                                 tokenizer.pad_token_id).sum(dim=1)
        output_gen_lengths = (output_ids != tokenizer.eos_token_id).sum(
            dim=1) - decoder_input_lengths

        print("--------------------------------------")
        print("TRT-LLM output_ids: ", output_ids)
        print("TRT-LLM output text: ", output_text)
        print("TRT-LLM output generated lengths: ", output_gen_lengths)
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("Precision:", inference_dtype)
        print("--------------------------------------")

        # save input/output tensors for C++ runtime testing
        if args.output_npy:
            os.makedirs(args.output_npy, exist_ok=True)

            input_lengths = tokenized_inputs.attention_mask.sum(dim=1).type(
                torch.IntTensor)
            input_ids = tokenized_inputs.input_ids.type(torch.IntTensor)
            input_ids_flatten = torch.cat([
                input_ids[i][:input_lengths[i]]
                for i in range(len(input_lengths))
            ])
            encoder_output = tllm_output['encoder_output'].type(torch.float16)

            def save_npy(tensor, name):
                np.save(os.path.join(args.output_npy, f'{name}.npy'),
                        tensor.cpu().numpy())

            print(
                f"Saving input/output tensors to {args.output_npy} for C++ runtime testing"
            )
            save_npy(input_ids_flatten, 'input_ids')  # [num_tokens]
            save_npy(input_lengths, 'input_lengths')  # [batch_size]
            save_npy(encoder_output,
                     'encoder_output')  # [num_tokens, hidden_size]
            save_npy(
                output_ids, 'output_ids'
            )  # [batch_size, max_output_tokens], max_output_tokens = decoder_input_tokens + max_new_tokens

        # simple accuracy check
        if args.compare_hf_fp32:
            from difflib import SequenceMatcher
            match_rate = SequenceMatcher(None, "\n".join(output_text),
                                         "\n".join(hf_output_text)).ratio()
            print(output_text)
            print(hf_output_text)
            if inference_dtype != "float32":
                print("")
                print(
                    f"[CAVEAT] Comparing TRT-LLM {inference_dtype} results with HF float32 results. Close match are not expected!"
                )
                assert match_rate > 0.8, f"Incorrect results! Match rate {match_rate}"
            else:
                assert match_rate > 0.95, f"Incorrect results! Match rate {match_rate}"
            print(
                f"TRT-LLM results match HF FP32 results with literal match rate {match_rate}"
            )
