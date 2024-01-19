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
from allowed_configs import get_build_config
from base_benchmark import BaseBenchmark, get_engine_name
from build import build_enc_dec

import tensorrt_llm
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.quantization import QuantMode


class EncDecBenchmark(BaseBenchmark):

    def __init__(self, args, batch_sizes, in_out_lens, rank, world_size):
        self.engine_dir = args.engine_dir
        self.model_name = args.model
        self.mode = args.mode
        self.enable_fp8 = False  # hardcode for enc-dec models
        self.dtype = args.dtype
        self.output_dir = args.output_dir
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

        if self.engine_dir is not None:

            def read_config(component):
                config_path = os.path.join(self.engine_dir, component,
                                           "config.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                # Sanity checks
                config_dtype = config["builder_config"]["precision"]
                assert (
                    self.dtype == config_dtype
                ), f"Engine dtype ({config_dtype}) != Runtime dtype ({self.dtype})"
                world_size = config["builder_config"]["tensor_parallel"]
                assert (
                    world_size == self.world_size
                ), f"Engine world size ({world_size}) != Runtime world size ({self.world_size})"
                tp_size = config["builder_config"]["tensor_parallel"]
                # TP only for benchmarking
                assert (
                    tp_size == self.world_size
                ), f"Engine tensor parallel size ({tp_size}) should be equal to world size ({self.world_size})"
                assert (
                    config["plugin_config"]["remove_input_padding"] == False
                ), "remove_input_padding should be False for enc-dec benchmarks"
                num_heads = config["builder_config"]["num_heads"]
                assert (num_heads % tp_size) == 0
                # Get model config
                num_heads = num_heads // tp_size
                hidden_size = config["builder_config"]["hidden_size"] // tp_size
                num_kv_heads = config["builder_config"].get(
                    "num_kv_heads", config["builder_config"]["num_heads"])
                num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

                model_config = tensorrt_llm.runtime.ModelConfig(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    hidden_size=hidden_size,
                    head_size=config["builder_config"]["head_size"],
                    vocab_size=config["builder_config"]["vocab_size"],
                    num_layers=config["builder_config"]["num_layers"],
                    gpt_attention_plugin=config["plugin_config"]
                    ["gpt_attention_plugin"],
                    remove_input_padding=config["plugin_config"]
                    ["remove_input_padding"],
                    cross_attention=config["builder_config"]["cross_attention"],
                    has_position_embedding=config["builder_config"]
                    ["has_position_embedding"],
                    has_token_type_embedding=config["builder_config"]
                    ["has_token_type_embedding"],
                    use_custom_all_reduce=config["plugin_config"].get(
                        "use_custom_all_reduce", False),
                    dtype=config_dtype,
                )
                self.max_batch_size = config["builder_config"]["max_batch_size"]
                self.max_input_len = config["builder_config"][
                    "max_encoder_input_len"]
                self.max_output_len = config["builder_config"]["max_output_len"]

                for key, value in config["builder_config"].items():
                    if key == "name":
                        engine_model_name = value
                        break
                return engine_model_name, model_config

            (
                self.encoder_engine_model_name,
                self.encoder_model_config,
            ) = read_config("encoder")
            (
                self.decoder_engine_model_name,
                self.decoder_model_config,
            ) = read_config("decoder")

        self.encoder_engine_name = get_engine_name(
            self.encoder_engine_model_name,
            self.dtype,
            self.world_size,
            self.runtime_rank,
        )
        self.decoder_engine_name = get_engine_name(
            self.decoder_engine_model_name,
            self.dtype,
            self.world_size,
            self.runtime_rank,
        )
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

        if not args.serial_build:
            torch.cuda.set_device(self.runtime_rank %
                                  self.encoder_runtime_mapping.gpus_per_node)
        self.device = torch.cuda.current_device()

        if self.engine_dir is not None:
            # Deserialize engine from engine directory
            self.encoder_serialize_path = os.path.join(self.engine_dir,
                                                       "encoder",
                                                       self.encoder_engine_name)
            with open(self.encoder_serialize_path, "rb") as f:
                encoder_engine_buffer = f.read()
            self.decoder_serialize_path = os.path.join(self.engine_dir,
                                                       "decoder",
                                                       self.decoder_engine_name)
            with open(self.decoder_serialize_path, "rb") as f:
                decoder_engine_buffer = f.read()
        else:
            build_config = get_build_config(self.model_name)
            self.max_batch_size = build_config['max_batch_size'] \
                if args.max_batch_size is None else args.max_batch_size
            self.max_input_len = build_config['max_encoder_input_len'] \
                if args.max_input_len is None else args.max_input_len
            self.max_output_len = build_config['max_output_len'] \
                if args.max_output_len is None else args.max_output_len
            # Build engine
            (
                encoder_engine_buffer,
                decoder_engine_buffer,
                self.encoder_model_config,
                self.decoder_model_config,
                encoder_build_time,
                decoder_build_time,
            ) = build_enc_dec(args)

            self.build_time = encoder_build_time + decoder_build_time

        assert encoder_engine_buffer is not None
        assert decoder_engine_buffer is not None

        # session setup
        self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
            encoder_engine_buffer)
        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            self.decoder_model_config,
            decoder_engine_buffer,
            self.decoder_runtime_mapping,
        )

    def get_config(self):
        for inlen, outlen in self.in_out_lens:
            if (inlen > self.max_input_len or outlen > self.max_output_len):
                print(
                    f"[WARNING] check inlen({inlen}) <= max_inlen({self.max_input_len}) and "
                    f"outlen({outlen}) <= max_outlen({self.max_output_len}) failed, skipping."
                )
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    print(
                        f"[WARNING] check batch_size({batch_size}) "
                        f"<= max_batch_size({self.max_batch_size}) failed, skipping."
                    )
                    continue
                yield (batch_size, inlen, outlen)

    def prepare_inputs(self, config):
        batch_size, encoder_input_len = config[0], config[1]
        encoder_input_ids = (torch.randint(
            100, (batch_size, encoder_input_len)).int().cuda())
        # For now, just hardcode the decoder_start_token_id to 0 for t5 models.
        decoder_start_token_id = 0
        decoder_input_ids = torch.IntTensor([[decoder_start_token_id]
                                             ]).to(self.device)
        decoder_input_ids = decoder_input_ids.repeat(
            (encoder_input_ids.shape[0], 1))
        # in padding mode --> keep input, just calculate actual length and max length
        # Note: 1st token should always count, even if it is pad_token_id (0). e.g., decoder start id in enc-dec models could be a single pad_token_id, we should count
        encoder_input_lengths = ((1 + (encoder_input_ids[:, 1:] != 0).sum(
            dim=1).type(torch.IntTensor).to(self.device)).clone().detach().to(
                dtype=torch.int32, device=self.device))
        decoder_input_lengths = ((1 + (decoder_input_ids[:, 1:] != 0).sum(
            dim=1).type(torch.IntTensor).to(self.device)).clone().detach().to(
                dtype=torch.int32, device=self.device))

        stream = torch.cuda.current_stream().cuda_stream
        return (
            encoder_input_ids,
            encoder_input_lengths,
            decoder_input_ids,
            decoder_input_lengths,
            stream,
        )

    def run(self, inputs, config, benchmark_profiler=None):
        output_len = config[2]
        (
            encoder_input_ids,
            encoder_input_lengths,
            decoder_input_ids,
            decoder_input_lengths,
            stream,
        ) = inputs

        hidden_size = (self.encoder_model_config.hidden_size * self.world_size
                       )  # tp_size
        hidden_states_shape = (
            encoder_input_ids.shape[0],
            encoder_input_ids.shape[1],
            hidden_size,
        )
        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name))

        # input tensors
        inputs = {}
        inputs["input_ids"] = encoder_input_ids.contiguous()
        inputs["input_lengths"] = encoder_input_lengths
        inputs["max_input_length"] = torch.empty(
            (self.max_input_len, ),
            dtype=hidden_states_dtype("max_input_length"),
            device=self.device,
        ).contiguous()

        if self.encoder_model_config.has_position_embedding:
            bsz, seq_len = encoder_input_ids.shape[:2]
            position_ids = torch.arange(seq_len,
                                        dtype=torch.int32,
                                        device=encoder_input_ids.device).expand(
                                            bsz, -1)
            inputs['position_ids'] = position_ids.contiguous()

        # output tensors
        outputs = {}
        outputs["encoder_output"] = torch.empty(
            hidden_states_shape,
            dtype=hidden_states_dtype("encoder_output"),
            device=self.device,
        ).contiguous()

        # run encoder
        self.encoder_session.set_shapes(inputs)
        ok = self.encoder_session.run(inputs, outputs, stream)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

        # run decoder
        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=1, pad_id=0, num_beams=self.num_beams, min_length=output_len)

        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            torch.max(decoder_input_lengths).item(),
            output_len,
            beam_width=self.num_beams,
            max_attention_window_size=None,
            encoder_max_input_length=torch.max(encoder_input_lengths).item(),
        )
        torch.cuda.synchronize()

        self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=outputs["encoder_output"],
            encoder_input_lengths=encoder_input_lengths,
        )
        torch.cuda.synchronize()

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
