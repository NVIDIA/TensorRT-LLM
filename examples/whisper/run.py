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
import math
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tokenizer import get_tokenizer
from torch.utils.data import DataLoader
from whisper.normalizers import EnglishTextNormalizer
from whisper_utils import (log_mel_spectrogram, store_transcripts,
                           write_error_stats)

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.bindings import GptJsonConfig, KVCacheType
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='warning')
    parser.add_argument('--engine_dir', type=str, default='whisper_large_v3')
    parser.add_argument('--results_dir', type=str, default='tmp')
    parser.add_argument('--assets_dir', type=str, default='./assets')
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--dataset',
                        type=str,
                        default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="clean",
        help=
        "dataset configuration name in the dataset, see https://huggingface.co/docs/datasets/v3.0.0/en/package_reference/loading_methods#datasets.load_dataset"
    )
    parser.add_argument('--dataset_split', type=str, default="validation")
    parser.add_argument('--name',
                        type=str,
                        default="librispeech_dummy_benchmark")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable_warmup', action='store_true')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16'])
    parser.add_argument('--accuracy_check',
                        action='store_true',
                        help="only for CI test")
    parser.add_argument('--use_py_session',
                        action='store_true',
                        help="use python session or cpp session")
    parser.add_argument(
        "--compute_cer",
        action="store_true",
        default=False,
        help="""True to compute character error rate (CER), e.g., for Chinese.
        False to compute word error rate (WER), e.g., for English words.
        """,
    )
    parser.add_argument(
        "--text_prefix",
        default="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        help="""Text prefix to be used for decoding. Default is for English ASR.
        """,
    )
    parser.add_argument(
        "--padding_strategy",
        default="max",
        help=
        """1. max: pad to the 30s, using the option if the model is trained with max padding e.g. openai official models,
           2. longest: pad to the longest sequence in the batch,
           3. nopad: no padding, only works with cpp session,
        """,
    )
    return parser.parse_args()


def remove_tensor_padding(input_tensor,
                          input_tensor_lengths=None,
                          pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None, "input_tensor_lengths should be None when pad_value is provided"
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] !=
            pad_value), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class WhisperEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config('encoder', engine_dir)
        self.n_mels = config['n_mels']
        self.dtype = config['dtype']
        self.num_languages = config['num_languages']
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = engine_dir / 'encoder' / 'rank0.engine'
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self,
                           mel,
                           mel_input_lengths,
                           encoder_downsampling_factor=2):
        if isinstance(mel, list):
            longest_mel = max([f.shape[-1] for f in mel])
            mel = [
                torch.nn.functional.pad(f, (0, longest_mel - f.shape[-1]),
                                        mode='constant') for f in mel
            ]
            mel = torch.cat(mel, dim=0).type(
                str_dtype_to_torch("float16")).contiguous()
        bsz, seq_len = mel.shape[0], mel.shape[2]
        position_ids = torch.arange(
            math.ceil(seq_len / encoder_downsampling_factor),
            dtype=torch.int32,
            device=mel.device).expand(bsz, -1).contiguous()
        if self.encoder_config['plugin_config']['remove_input_padding']:
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)
            position_ids = remove_tensor_padding(
                position_ids, mel_input_lengths // encoder_downsampling_factor)
        inputs = OrderedDict()
        inputs['input_features'] = mel
        inputs['input_lengths'] = mel_input_lengths
        inputs['position_ids'] = position_ids

        output_list = [
            TensorInfo('input_features', str_dtype_to_trt(self.dtype),
                       mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       mel_input_lengths.shape),
            TensorInfo('position_ids', str_dtype_to_trt('int32'),
                       inputs['position_ids'].shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        encoder_output = outputs['encoder_output']
        encoder_output_lengths = mel_input_lengths // encoder_downsampling_factor
        return encoder_output, encoder_output_lengths


class WhisperDecoding:

    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = read_config('decoder', engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            kv_cache_type=KVCacheType.PAGED
            if self.decoder_config['plugin_config']['paged_kv_cache'] == True
            else KVCacheType.CONTINUOUS,
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_max_input_length,
                 encoder_input_lengths,
                 eot_id,
                 max_new_tokens=40,
                 num_beams=1):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones([
            batch_size, decoder_max_input_length + max_new_tokens,
            encoder_max_input_length
        ]).int().cuda()
        # generation config
        sampling_config = SamplingConfig(end_id=eot_id,
                                         pad_id=eot_id,
                                         num_beams=num_beams)
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_max_input_length)

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID)
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full((encoder_outputs.shape[0], ),
                                                 encoder_outputs.shape[1],
                                                 dtype=torch.int32,
                                                 device='cuda')

                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_output_lens)
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRTLLM(object):

    def __init__(self,
                 engine_dir,
                 debug_mode=False,
                 assets_dir=None,
                 batch_size=64,
                 use_py_session=False,
                 num_beams=1):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        encoder_config = read_config('encoder', engine_dir)
        decoder_config = read_config('decoder', engine_dir)
        self.n_mels = encoder_config['n_mels']
        self.num_languages = encoder_config['num_languages']
        is_multilingual = (decoder_config['vocab_size'] >= 51865)
        if is_multilingual:
            tokenizer_name = "multilingual"
            assert (Path(assets_dir) / "multilingual.tiktoken").exists(
            ), "multilingual.tiktoken file is not existed in assets_dir"
        else:
            tokenizer_name = "gpt2"
            assert (Path(assets_dir) / "gpt2.tiktoken").exists(
            ), "gpt2.tiktoken file is not existed in assets_dir"
        self.tokenizer = get_tokenizer(name=tokenizer_name,
                                       num_languages=self.num_languages,
                                       tokenizer_dir=assets_dir)
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set)[0]
        if use_py_session:
            self.encoder = WhisperEncoding(engine_dir)
            self.decoder = WhisperDecoding(engine_dir,
                                           runtime_mapping,
                                           debug_mode=debug_mode)
        else:
            json_config = GptJsonConfig.parse_file(engine_dir / 'decoder' /
                                                   'config.json')
            assert json_config.model_config.supports_inflight_batching
            runner_kwargs = dict(engine_dir=engine_dir,
                                 is_enc_dec=True,
                                 max_batch_size=batch_size,
                                 max_input_len=3000,
                                 max_output_len=96,
                                 max_beam_width=num_beams,
                                 debug_mode=debug_mode,
                                 kv_cache_free_gpu_memory_fraction=0.9,
                                 cross_kv_cache_fraction=0.5)
            self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.use_py_session = use_py_session

    def process_batch(
            self,
            mel,
            mel_input_lengths,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            num_beams=1,
            max_new_tokens=96):
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id)
        batch_size = len(mel)
        decoder_input_ids = prompt_id.repeat(batch_size, 1)
        if self.use_py_session:
            encoder_output, encoder_output_lengths = self.encoder.get_audio_features(
                mel, mel_input_lengths)
            encoder_max_input_length = torch.max(encoder_output_lengths).item()
            output_ids = self.decoder.generate(decoder_input_ids,
                                               encoder_output,
                                               encoder_max_input_length,
                                               encoder_output_lengths,
                                               self.eot_id,
                                               max_new_tokens=max_new_tokens,
                                               num_beams=num_beams)
        else:
            with torch.no_grad():
                if isinstance(mel, list):
                    mel = [
                        m.transpose(1, 2).type(
                            str_dtype_to_torch("float16")).squeeze(0)
                        for m in mel
                    ]
                else:
                    mel = mel.transpose(1, 2)
                outputs = self.model_runner_cpp.generate(
                    batch_input_ids=decoder_input_ids,
                    encoder_input_features=mel,
                    encoder_output_lengths=mel_input_lengths // 2,
                    max_new_tokens=max_new_tokens,
                    end_id=self.eot_id,
                    pad_id=self.eot_id,
                    num_beams=num_beams,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()
                output_ids = outputs['output_ids'].cpu().numpy().tolist()
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts


def decode_wav_file(
        input_file_path,
        model,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        dtype='float16',
        batch_size=1,
        num_beams=1,
        normalizer=None,
        mel_filters_dir=None,
        padding_strategy="longest"):
    mel, total_duration = log_mel_spectrogram(input_file_path,
                                              model.n_mels,
                                              device='cuda',
                                              return_duration=True,
                                              mel_filters_dir=mel_filters_dir)
    mel = mel.type(str_dtype_to_torch(dtype))
    mel = mel.unsqueeze(0)
    # repeat the mel spectrogram to match the batch size
    mel = mel.repeat(batch_size, 1, 1)
    if padding_strategy == "longest":
        pass
    else:
        mel = torch.nn.functional.pad(mel, (0, 3000 - mel.shape[2]))
    features_input_lengths = torch.full((mel.shape[0], ),
                                        mel.shape[2],
                                        dtype=torch.int32,
                                        device=mel.device)
    predictions = model.process_batch(mel, features_input_lengths, text_prefix,
                                      num_beams)
    prediction = predictions[0]

    # remove all special tokens in the prediction
    prediction = re.sub(r'<\|.*?\|>', '', prediction)
    if normalizer:
        prediction = normalizer(prediction)
    print(f"prediction: {prediction}")
    results = [(0, [""], prediction.split())]
    return results, total_duration


def collate_wrapper(batch):
    speeches, durations, labels, ids = [], [], [], []
    for item in batch:
        speech = item["audio"]["array"]
        duration = speech.shape[-1]
        speech = speech.astype(np.float32)
        speech = torch.from_numpy(speech)
        speeches.append(speech)
        durations.append(duration)
        labels.append(item["text"])
        if 'id' in item:
            ids.append(item["id"])
        else:
            ids.append(item["segment_id"])
    return speeches, durations, labels, ids


def decode_dataset(
        model,
        dataset,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        dtype='float16',
        batch_size=1,
        num_beams=1,
        normalizer=None,
        sample_rate=16000,
        mel_filters_dir=None,
        compute_cer=False,
        padding_strategy="longest"):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             pin_memory=True,
                             collate_fn=collate_wrapper)
    results = []
    total_duration = 0
    for batch in data_loader:
        waveforms, durations, texts, ids = batch
        total_duration += sum(durations) / sample_rate

        for wave in waveforms:
            assert wave.is_pinned()

        if padding_strategy == "longest":
            longest_duration = max(durations)
        elif padding_strategy == "nopad":
            longest_duration = 0
        else:
            longest_duration = int(16000 * 30)

        features = [
            log_mel_spectrogram(wave,
                                model.n_mels,
                                padding=longest_duration - wave.shape[-1],
                                device='cuda',
                                mel_filters_dir=mel_filters_dir).unsqueeze(0)
            for wave in waveforms
        ]

        # pad to the even number of features, for remove_padding option, conv layer padding corner case
        for i, feature in enumerate(features):
            if feature.shape[2] % 2:
                features[i] = torch.nn.functional.pad(feature, (0, 1))

        features_input_lengths = torch.tensor([f.shape[2] for f in features],
                                              dtype=torch.int32,
                                              device='cuda')

        predictions = model.process_batch(features, features_input_lengths,
                                          text_prefix, num_beams)
        for wav_id, label, prediction in zip(ids, texts, predictions):
            # remove all special tokens in the prediction
            prediction = re.sub(r'<\|.*?\|>', '', prediction)
            if normalizer:
                prediction, label = normalizer(prediction), normalizer(label)
            label = label.split()
            prediction = prediction.split()
            if compute_cer:
                label = list("".join(label))
                prediction = list("".join(prediction))
            print(f"wav_id: {wav_id}, label: {label}, prediction: {prediction}")
            results.append((wav_id, label, prediction))
    return results, total_duration


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    model = WhisperTRTLLM(args.engine_dir, args.debug, args.assets_dir,
                          args.batch_size, args.use_py_session, args.num_beams)
    normalizer = EnglishTextNormalizer()
    dataset = load_dataset(args.dataset,
                           args.dataset_name,
                           split=args.dataset_split,
                           trust_remote_code=True)
    if args.enable_warmup:
        results, total_duration = decode_dataset(
            model,
            dataset,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            normalizer=normalizer,
            mel_filters_dir=args.assets_dir,
            padding_strategy=args.padding_strategy)

    start_time = time.time()
    if args.input_file:
        results, total_duration = decode_wav_file(
            args.input_file,
            model,
            text_prefix=args.text_prefix,
            dtype=args.dtype,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            mel_filters_dir=args.assets_dir,
            padding_strategy=args.padding_strategy)
    else:
        results, total_duration = decode_dataset(
            model,
            dataset,
            text_prefix=args.text_prefix,
            dtype=args.dtype,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            normalizer=normalizer,
            mel_filters_dir=args.assets_dir,
            compute_cer=args.compute_cer,
            padding_strategy=args.padding_strategy)
    elapsed = time.time() - start_time
    results = sorted(results)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    store_transcripts(filename=f"{args.results_dir}/recogs-{args.name}.txt",
                      texts=results)

    with open(f"{args.results_dir}/errs-{args.name}.txt", "w") as f:
        total_error_rate = write_error_stats(f,
                                             "test-set",
                                             results,
                                             enable_log=True)
        if args.accuracy_check and args.dataset == "hf-internal-testing/librispeech_asr_dummy" and not args.input_file:
            assert total_error_rate <= 2.8, f"Word Error rate using whisper large-v3 model should be 2.40%, but got {total_error_rate}"

    rtf = elapsed / total_duration
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    s += f"num_beams: {args.num_beams}\n"
    s += f"total error rate: {total_error_rate:.2f}%\n"
    print(s)

    with open(f"{args.results_dir}/rtf-{args.name}.txt", "w") as f:
        f.write(s)

    del model
