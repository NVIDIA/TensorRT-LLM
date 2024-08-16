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

# isort: off
import torch
#isort: on
from base_benchmark import BaseBenchmark

import tensorrt_llm
from tensorrt_llm._utils import (trt_dtype_to_torch, str_dtype_to_trt)
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime.session import TensorInfo
from tensorrt_llm.runtime import ModelConfig
from tensorrt_llm.models.modeling_utils import get_kv_cache_type_from_legacy


class EncDecBenchmark(BaseBenchmark):

    def __init__(self, args, batch_sizes, in_out_lens, gpu_weights_percents,
                 rank, world_size):
        self.engine_dir = args.engine_dir
        self.model_name = args.model
        self.enable_fp8 = False  # hardcode for enc-dec models
        self.dtype = args.dtype
        self.runtime_rank = rank
        self.world_size = world_size
        self.csv_filename = ""  # lazy init
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.num_beams = args.num_beams
        self.build_time = 0
        self.quant_mode = QuantMode(0)
        # In current implementation, encoder and decoder have the same name,
        # builder config, and plugin config. But they can be different in the future.
        # So we use separate variables for encoder and decoder here.
        self.encoder_engine_model_name = args.model
        self.decoder_engine_model_name = args.model
        self.gpu_weights_percents = gpu_weights_percents

        # only for whisper parameter
        self.n_mels = 0

        if self.engine_dir is not None:

            def read_config(component):
                # almost same as enc_dec_model_runner.py::read_config()
                config_path = os.path.join(self.engine_dir, component,
                                           "config.json")
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
                auto_parallel_config['gpus_per_node']
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

                cross_attention = pretrained_config[
                    "architecture"] == "DecoderModel"
                skip_cross_qkv = pretrained_config.get('skip_cross_qkv', False)
                has_position_embedding = pretrained_config[
                    "has_position_embedding"]
                has_token_type_embedding = hasattr(pretrained_config,
                                                   "type_vocab_size")
                dtype = pretrained_config["dtype"]

                paged_kv_cache = plugin_config['paged_kv_cache']
                kv_cache_type = get_kv_cache_type_from_legacy(
                    True, paged_kv_cache)

                tokens_per_block = plugin_config['tokens_per_block']

                gather_context_logits = builder_config.get(
                    'gather_context_logits', False)
                gather_generation_logits = builder_config.get(
                    'gather_generation_logits', False)
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
                    kv_cache_type=kv_cache_type,
                    tokens_per_block=tokens_per_block,
                    cross_attention=cross_attention,
                    has_position_embedding=has_position_embedding,
                    has_token_type_embedding=has_token_type_embedding,
                    dtype=dtype,
                    gather_context_logits=gather_context_logits,
                    gather_generation_logits=gather_generation_logits,
                    max_prompt_embedding_table_size=
                    max_prompt_embedding_table_size,
                    lora_plugin=use_lora_plugin,
                    lora_target_modules=lora_config.get('lora_target_modules'),
                    trtllm_modules_to_hf_modules=lora_config.get(
                        'trtllm_modules_to_hf_modules'),
                    skip_cross_qkv=skip_cross_qkv,
                )

                # additional info for benchmark
                self.max_batch_size = config["build_config"]["max_batch_size"]
                self.max_input_len = config["build_config"][
                    "max_encoder_input_len"]
                self.max_seq_len = config["build_config"]["max_seq_len"]
                if component == "decoder":
                    self.decoder_start_token_id = pretrained_config[
                        'decoder_start_token_id']

                return model_config

            self.encoder_model_config = read_config("encoder")
            self.decoder_model_config = read_config("decoder")

        self.encoder_engine_name = 'rank{}.engine'.format(self.runtime_rank)
        self.decoder_engine_name = 'rank{}.engine'.format(self.runtime_rank)
        self.encoder_runtime_mapping = tensorrt_llm.Mapping(
            world_size=self.world_size,
            rank=self.runtime_rank,
            tp_size=self.world_size,
        )
        self.decoder_runtime_mapping = tensorrt_llm.Mapping(
            world_size=self.world_size,
            rank=self.runtime_rank,
            tp_size=self.world_size,
        )

        torch.cuda.set_device(self.runtime_rank %
                              self.encoder_runtime_mapping.gpus_per_node)
        self.device = torch.cuda.current_device()

        # Deserialize engine from engine directory
        self.encoder_serialize_path = os.path.join(self.engine_dir, "encoder",
                                                   self.encoder_engine_name)
        with open(self.encoder_serialize_path, "rb") as f:
            encoder_engine_buffer = f.read()
            assert encoder_engine_buffer is not None
        self.decoder_serialize_path = os.path.join(self.engine_dir, "decoder",
                                                   self.decoder_engine_name)
        with open(self.decoder_serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()
            assert decoder_engine_buffer is not None

        # session setup
        self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
            encoder_engine_buffer)
        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            self.decoder_model_config, decoder_engine_buffer,
            self.decoder_runtime_mapping)

        # Print context memory size for CI/CD to track.
        context_mem_size = self.encoder_session.context_mem_size + self.decoder_session.context_mem_size
        print(
            f"Allocated {context_mem_size / 1048576.0:.2f} MiB for execution context memory."
        )

    def get_config(self):
        if 'whisper' in self.model_name:
            print(
                f"[WARNING] whisper benchmark is input_len=1500, no text prompt, output_len=arbitrary"
            )
        for inlen, outlen in self.in_out_lens:
            if (inlen > self.max_input_len or outlen > self.max_seq_len):
                print(
                    f"[WARNING] check inlen({inlen}) <= max_inlen({self.max_input_len}) and "
                    f"outlen({outlen}) <= max_seqlen({self.max_seq_len}) failed, skipping."
                )
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    print(
                        f"[WARNING] check batch_size({batch_size}) "
                        f"<= max_batch_size({self.max_batch_size}) failed, skipping."
                    )
                    continue
                for gpu_weights_percent in self.gpu_weights_percents:
                    yield (batch_size, inlen, outlen, gpu_weights_percent)

    def set_weight_streaming(self, config):
        gpu_weights_percent = config[3]
        self.encoder_session._set_weight_streaming(gpu_weights_percent)
        self.decoder_session.runtime._set_weight_streaming(gpu_weights_percent)

    def prepare_inputs(self, config):
        batch_size, encoder_input_len = config[0], config[1]
        attention_mask = None
        whisper_decoder_encoder_input_lengths = None
        outputs = {}
        if 'whisper' in self.model_name:
            # feature_len always fixed 3000 now
            feature_len = 3000
            encoder_input_ids = (torch.randint(
                1, 100, (batch_size, self.n_mels, feature_len)).int().cuda())
            encoder_input_lengths = torch.tensor([
                encoder_input_ids.shape[2] // 2
                for _ in range(encoder_input_ids.shape[0])
            ],
                                                 dtype=torch.int32,
                                                 device=self.device)
            decoder_input_ids = (torch.randint(1, 100, (1, )).int().cuda())
            decoder_input_ids = decoder_input_ids.repeat(
                (encoder_input_ids.shape[0], 1))
            output_list = [
                TensorInfo('input_features', str_dtype_to_trt(self.dtype),
                           encoder_input_ids.shape),
                TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                           encoder_input_lengths.shape)
            ]
            output_info = (self.encoder_session).infer_shapes(output_list)
            outputs = {
                t.name: torch.empty(tuple(t.shape),
                                    dtype=trt_dtype_to_torch(t.dtype),
                                    device='cuda')
                for t in output_info
            }
            whisper_decoder_encoder_input_lengths = torch.tensor(
                [
                    outputs['encoder_output'].shape[1]
                    for x in range(outputs['encoder_output'].shape[0])
                ],
                dtype=torch.int32,
                device='cuda')

            decoder_input_lengths = torch.tensor([
                decoder_input_ids.shape[-1]
                for _ in range(decoder_input_ids.shape[0])
            ],
                                                 dtype=torch.int32,
                                                 device='cuda')
            cross_attention_mask = torch.ones([
                outputs['encoder_output'].shape[0], 1,
                outputs['encoder_output'].shape[1]
            ]).int().cuda()
        else:
            encoder_input_ids = (torch.randint(
                100, (batch_size, encoder_input_len)).int().cuda())
            decoder_input_ids = torch.IntTensor([[self.decoder_start_token_id]
                                                 ]).to(self.device)
            decoder_input_ids = decoder_input_ids.repeat((batch_size, 1))
            encoder_input_lengths = torch.tensor([encoder_input_len] *
                                                 batch_size,
                                                 dtype=torch.int32,
                                                 device=self.device)
            decoder_input_lengths = torch.tensor([1] * batch_size,
                                                 dtype=torch.int32,
                                                 device=self.device)

            if self.encoder_model_config.remove_input_padding:
                encoder_input_ids = torch.flatten(encoder_input_ids)
                decoder_input_ids = torch.flatten(decoder_input_ids)

            # attention mask, always set 1 as if all are valid tokens
            attention_mask = torch.ones(
                (batch_size, encoder_input_len)).int().cuda()
            # cross attention mask, always set 1 as if all are valid tokens
            # [batch_size, query_len, encoder_input_len] currently, use query_len=1
            cross_attention_mask = torch.ones(
                (batch_size, 1, encoder_input_len)).int().cuda()

            hidden_size = (self.encoder_model_config.hidden_size *
                           self.world_size)  # tp_size
            hidden_states_shape = (
                encoder_input_ids.shape[0],
                hidden_size,
            ) if self.encoder_model_config.remove_input_padding else (
                encoder_input_ids.shape[0],
                encoder_input_ids.shape[1],
                hidden_size,
            )
            hidden_states_dtype = lambda name: trt_dtype_to_torch(
                self.encoder_session.engine.get_tensor_dtype(name))

            outputs["encoder_output"] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype("encoder_output"),
                device=self.device,
            ).contiguous()

        stream = torch.cuda.current_stream().cuda_stream
        return (
            encoder_input_ids,
            encoder_input_lengths,
            attention_mask,
            decoder_input_ids,
            decoder_input_lengths,
            cross_attention_mask,
            whisper_decoder_encoder_input_lengths,
            outputs,
            stream,
        )

    def run(self, inputs, config, benchmark_profiler=None):
        output_len = config[2]
        (
            encoder_input_ids,
            encoder_input_lengths,
            attention_mask,
            decoder_input_ids,
            decoder_input_lengths,
            cross_attention_mask,
            whisper_decoder_encoder_input_lengths,
            outputs,
            stream,
        ) = inputs

        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name))

        # input tensors
        inputs = {}
        if 'whisper' in self.model_name:
            inputs['input_features'] = encoder_input_ids.contiguous()
            inputs["input_lengths"] = encoder_input_lengths
        else:
            inputs["input_ids"] = encoder_input_ids.contiguous()
            inputs["input_lengths"] = encoder_input_lengths
            inputs["max_input_length"] = torch.empty(
                (self.max_input_len, ),
                dtype=hidden_states_dtype("max_input_length"),
                device=self.device,
            ).contiguous()

            if not self.encoder_model_config.gpt_attention_plugin:
                inputs["attention_mask"] = attention_mask.contiguous()

            if self.encoder_model_config.has_position_embedding:
                bsz, seq_len = encoder_input_ids.shape[:2]
                position_ids = torch.arange(
                    seq_len, dtype=torch.int32,
                    device=encoder_input_ids.device).expand(bsz, -1)
                inputs['position_ids'] = position_ids.contiguous()

        # run encoder
        self.encoder_session.set_shapes(inputs)
        ok = self.encoder_session.run(inputs, outputs, stream)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

        # run decoder
        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=1, pad_id=0, num_beams=self.num_beams, min_length=output_len)
        encoder_output = outputs["encoder_output"]
        encoder_max_input_length = encoder_output.shape[
            1] if 'whisper' in self.model_name else torch.max(
                encoder_input_lengths).item()

        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            torch.max(decoder_input_lengths).item(),
            output_len,
            beam_width=self.num_beams,
            max_attention_window_size=None,
            encoder_max_input_length=encoder_max_input_length,
        )

        cross_attention_mask = None if self.decoder_model_config.gpt_attention_plugin else cross_attention_mask

        self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=whisper_decoder_encoder_input_lengths
            if 'whisper' in self.model_name else encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )

    def report(self,
               config,
               latency,
               percentile95,
               percentile99,
               peak_gpu_used,
               csv,
               benchmark_profiler=None):
        # Note: Theoretically, the encoder and decoder can have different configs.
        # But for current implementation, we assume they are the same. In the future,
        # we can have a special structure of report_dict for enc-dec models.
        report_dict = super().get_report_dict()
        batch_size, encoder_input_len, output_len = config[0], config[
            1], config[2]
        tokens_per_sec = round(batch_size * output_len / (latency / 1000), 2)
        report_dict["num_heads"] = self.encoder_model_config.num_heads
        report_dict["num_kv_heads"] = self.encoder_model_config.num_kv_heads
        report_dict["num_layers"] = self.encoder_model_config.num_layers
        report_dict["hidden_size"] = self.encoder_model_config.hidden_size
        report_dict["vocab_size"] = self.encoder_model_config.vocab_size
        report_dict["batch_size"] = batch_size
        report_dict["input_length"] = encoder_input_len
        report_dict["output_length"] = output_len
        report_dict["gpu_weights_percent"] = config[3]
        report_dict["latency(ms)"] = latency
        report_dict["build_time(s)"] = self.build_time
        report_dict["tokens_per_sec"] = tokens_per_sec
        report_dict["percentile95(ms)"] = percentile95
        report_dict["percentile99(ms)"] = percentile99
        report_dict["gpu_peak_mem(gb)"] = peak_gpu_used
        if self.runtime_rank == 0:
            if csv:
                line = ",".join([str(v) for v in report_dict.values()])
                print(line)
                with open(self.get_csv_filename(), "a") as file:
                    file.write(line + "\n")
            else:
                kv_pairs = [f"{k} {v}" for k, v in report_dict.items()]
                line = "[BENCHMARK] " + " ".join(kv_pairs)
                print(line)
