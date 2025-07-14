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
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from string import punctuation

import librosa
import numpy
import numpy as np
import tensorrt as trt
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import (N_SAMPLES, MelFilterBankFeats, pad_or_trim,
                   store_transcripts, write_error_stats)

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.runtime import (PYTHON_BINDINGS, ModelConfig, ModelRunnerCpp,
                                  SamplingConfig)
from tensorrt_llm.runtime.session import Session, TensorInfo

if PYTHON_BINDINGS:
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='warning')
    parser.add_argument('--engine_dir', type=str, default='canary')
    parser.add_argument('--results_dir', type=str, default='tmp')
    parser.add_argument('--assets_dir', type=str, default='./assets')
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--prompt_text', type=str, default=None)
    parser.add_argument('--manifest_file', type=str, default=None)
    parser.add_argument('--results_manifest', type=str, default=None)

    parser.add_argument('--dataset',
                        type=str,
                        default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument('--name',
                        type=str,
                        default="librispeech_dummy_benchmark")
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable_warmup', action='store_true')
    parser.add_argument('--accuracy_check',
                        action='store_true',
                        help="only for CI test")
    parser.add_argument('--use_py_session',
                        action='store_true',
                        help="use python session or cpp session")
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


def unpack_tensors(input_tensors, input_tensor_lengths):
    output_tensors = []
    for i in range(len(input_tensors)):
        output_tensors.append(input_tensors[i, :input_tensor_lengths[i]])
    return output_tensors


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class CanaryTokenizer:

    def __init__(self, engine_dir, prompt_format='canary1'):
        vocab_file = os.path.join(engine_dir, 'decoder/vocab.json')
        decoder_config = read_config('decoder', engine_dir)
        self.prompt_format = decoder_config.get('prompt_format', prompt_format)
        self.blank = '▁'
        self.has_country_code = False

        with open(vocab_file, 'r') as jfp:
            vocab = json.load(jfp)
        self.token_id_offset = vocab['offsets']
        self.langs = [k for k in vocab['tokens']]
        self.__id_to_token__ = {l: {} for l in vocab['tokens']}

        self.spl_tokens = self.__id_to_token__['spl_tokens']

        for lang in vocab['tokens']:
            self.__id_to_token__[lang] = {
                int(k): v
                for k, v in vocab['tokens'][lang].items()
            }
        self.__token_to_id__ = {}
        for lang in self.__id_to_token__:
            self.__token_to_id__[lang] = {
                v: k
                for k, v in self.__id_to_token__[lang].items()
            }

        for lang in self.langs:
            if lang == 'spl_tokens':
                continue
            lang_token = f"<|{lang.split('-')[0]}|>"
            if "-" in lang:
                lang_country_token = f"<|{lang}|>"
                self.has_country_code = True
            else:
                lang_country_token = lang_token
            if lang_country_token in self.__token_to_id__['spl_tokens']:
                continue

            if lang_country_token not in self.__token_to_id__[
                    'spl_tokens'] and lang_token in self.__token_to_id__[
                        'spl_tokens']:
                self.__token_to_id__['spl_tokens'][
                    lang_country_token] = self.__token_to_id__['spl_tokens'][
                        lang_token]
                if lang_country_token != lang_token:
                    self.langs.append(lang.split('-')[0])

        if self.has_country_code:
            dpl = 'en-US'
        else:
            dpl = 'en'
        if self.prompt_format == 'canary2':
            self.default_prompt = f"<|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|{dpl}|> <|{dpl}|> <|nopnc|> <|noitn|> <|notimestamp|> <|nodiarize|>"
        else:
            self.default_prompt = f"<|startoftranscript|> <|{dpl}|> <|transcribe|> <|{dpl}|> <|pnc|>"

        self.id_to_token = {}
        for lang in self.__id_to_token__:
            self.id_to_token.update(self.__id_to_token__[lang])
        self.task = {
            'transcribe': '<|transcribe|>',
            'translate': '<|translate|>',
            'asr': '<|transcribe|>',
            'ast': '<|translate|>'
        }

        try:
            self.bos_id = vocab['bos_id']
        except Exception:
            self.bos_id = self.spl_tokens['<|startoftranscript|>']
        try:
            self.eos_id = vocab['eos_id']
        except Exception:
            self.eos_id = self.spl_tokens['<|endoftext|>']
        try:
            self.nospeech_id = vocab['nospeech_id']
        except Exception:
            pass
        try:
            self.pad_id = vocab['pad_id']
        except Exception:
            self.pad_id = vocab['pad_id']

        self.blank_id = self.__token_to_id__['spl_tokens'][self.blank]

    def set_prompt_format(self, prompt_format='canary1'):
        self.prompt_format = prompt_format

    @staticmethod
    def word_separator(lang):
        if lang in [
                'ja-JP', 'ko-KR', 'zh-CN', 'th-TH', 'km-KH', 'my-MM', 'lo-LA'
        ]:
            return ''
        else:
            return " "

    def ids_to_lang(self, token_ids: list):
        langs_count = {k: 0 for k in self.langs}
        max_lang = self.langs[0]

        for token_id in token_ids:
            for l in langs_count:
                if token_id in self.__id_to_token__[l]:
                    langs_count[l] += 1
                    if langs_count[l] > langs_count[max_lang]:
                        max_lang = l
        return max_lang

    def ids_to_tokens(self, token_ids: list):
        return [self.id_to_token[k] for k in token_ids]

    def token_to_id(self, token: str, lang='spl_tokens'):
        return self.__token_to_id__[lang].get(token, self.token_id_offset[lang])

    def tokens_to_ids(self, tokens: list | str, lang='spl_tokens'):
        if isinstance(tokens, str):
            tokens = tokens.split(' ')
        return [self.token_to_id(k, lang) for k in tokens]

    def ids_to_text(self, ids: list, lang=None):
        MAX_REPEAT = 10
        clean_ids = []
        prev_id = 0
        id_count = 0
        for i in ids:
            if prev_id == i:
                id_count += 1
                if id_count >= MAX_REPEAT:
                    continue
            else:
                id_count = 0
                prev_id = i

            if i == self.eos_id:
                break

            if i not in self.__id_to_token__['spl_tokens']:

                clean_ids.append(i)

        if lang is None:
            return ''.join(self.ids_to_tokens(clean_ids)).replace('▁',
                                                                  ' ').strip()
        else:
            ws = self.word_separator(lang)
            tokens = [
                self.__id_to_token__[lang].get(k, f" <unk> ").replace('▁', ws)
                for k in clean_ids
            ]

            if ws == "":
                return re.sub(r'(?<=[.,;:])(?=[^\s])', r' ',
                              ''.join(tokens).strip())
            return ''.join(tokens).strip()

    def get_prompt_v2(
        self,
        pnc=True,
        src_lang='en',
        tgt_lang=None,
        itn=False,
        timestamp=False,
        diarize=False,
    ):
        # prompt_format:
        # <|startofcontext|><|startoftranscript|><|emo:undefined|><|{src_lang}|><|{tgt_lang}|><|[no]pnc|><|[no]itn|><|[no]timestamp><|[no]diarize|>
        prompt = "<|startofcontext|> <|startoftranscript|> <|emo:undefined|>"

        if not self.has_country_code and '-' in src_lang:
            src_lang = src_lang.split('-')[0]

        if src_lang not in self.langs:

            raise ValueError(f"Invalid language {src_lang=} specified")

        prompt += f" <|{src_lang}|>"
        if pnc:
            pnc = "<|pnc|>"
        else:
            pnc = "<|nopnc|>"
        if itn:
            itn = "<|itn|>"
        else:
            itn = "<|noitn|>"
        if diarize:
            diarize = "<|diarize|>"
        else:
            diarize = "<|nodiarize|>"
        if timestamp:
            timestamp = "<|timestamp|>"
        else:
            timestamp = "<|notimestamp|>"

        if tgt_lang is None:
            tgt_lang = src_lang
        if not self.has_country_code and '-' in tgt_lang:
            tgt_lang = tgt_lang.split('-')[0]
        if tgt_lang not in self.langs:
            raise ValueError(f"Invalid language {tgt_lang=} specified")

        prompt += f" <|{src_lang}|> <|{tgt_lang}|> {pnc} {itn} {timestamp} {diarize}"

        return prompt

    def get_prompt_legacy(self,
                          task_type='transcribe',
                          pnc=True,
                          src_lang='en',
                          tgt_lang=None):
        prompt = "<|startoftranscript|>"

        if src_lang not in self.langs and "-" in src_lang and src_lang.split(
                '-')[0] in self.langs:
            src_lang = src_lang.split('-')[0]

        if src_lang not in self.langs:
            raise ValueError(f"Invalid language {src_lang=} specified")

        prompt += f" <|{src_lang}|>"
        if pnc:
            pnc = "<|pnc|>"
        else:
            pnc = "<|nopnc|>"

        if task_type == 'translate' or task_type == 'ast':
            if tgt_lang is None:
                tgt_lang = src_lang
            if tgt_lang not in self.langs and "-" in tgt_lang and tgt_lang.split(
                    '-')[0] in self.langs:
                tgt_lang = tgt_lang.split('-')[0]

            if tgt_lang not in self.langs:
                raise ValueError(f"Invalid language {tgt_lang=} specified")

            prompt += f" {self.task[task_type]} <|{tgt_lang}|> {pnc}"

        elif task_type == "transcribe" or task_type == 'asr':
            prompt += f" {self.task[task_type]} <|{src_lang}|> {pnc}"
        else:
            raise ValueError(f"Invalid task {task_type=} specified")
        return prompt

    def get_prompt_ids(self,
                       task='transcribe',
                       pnc=False,
                       src_lang='en',
                       tgt_lang=None):
        return self.tokens_to_ids(
            self.get_prompt_legacy(task,
                                   pnc=pnc,
                                   src_lang=src_lang,
                                   tgt_lang=tgt_lang))

    def get_prompt_ids_from_cfg(self, cfg):

        if self.prompt_format == 'canary2':
            return self.tokens_to_ids(
                self.get_prompt_v2(cfg['pnc'], cfg['source_language'],
                                   cfg['target_language'], cfg['itn'],
                                   cfg['timestamp'], cfg['diarize']))

        else:
            return self.tokens_to_ids(
                self.get_prompt_legacy(cfg['task'], cfg['pnc'],
                                       cfg['source_language'],
                                       cfg['target_language']))

    def encode(self, prompt):
        return self.tokens_to_ids(prompt.split())

    def decode(self, ids: list, lang=None):
        text = self.ids_to_text(ids, lang)
        return re.sub(r'<\|.*?\|>', '', text)


class CanaryEncoder:

    def __init__(self, engine_dir):
        engine_path = os.path.join(engine_dir, 'encoder/encoder.plan')
        self.encoder_config = json.load(
            open(os.path.join(engine_dir, 'encoder/config.json'), 'r'))
        logger.info(f"Loading engine from {engine_path}")
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()
        logger.info(f"Creating session from engine {engine_path}")
        self.session_conformer = Session.from_serialized_engine(engine_buffer)
        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else "cpu"


    @staticmethod
    def get_masked_emb(enc_outputs):
        enc_emb = enc_outputs.get('encoded_outputs', enc_outputs.get('outputs'))
        enc_len = enc_outputs['encoded_lengths']
        batch_size = enc_len.shape[0]
        max_length = enc_emb.shape[1]

        mask = torch.arange(max_length, device='cuda').unsqueeze(0).expand(
            batch_size, max_length) < enc_len.unsqueeze(1)
        enc_mask = torch.where(mask.unsqueeze(2), enc_emb, 0.0)

        return enc_mask, enc_len

    def infer(self, audio_signal, lengths, stream, audio_file=""):

        audio_inputs = {'audio_signal': audio_signal, 'length': lengths}

        outputs_info = self.session_conformer.infer_shapes([
            TensorInfo("audio_signal", trt.DataType.FLOAT, audio_signal.shape),
            TensorInfo("length", trt.DataType.INT64, lengths.shape)
        ])
        enc_outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device="cuda:0")
            for t in outputs_info
        }

        is_ok = self.session_conformer.run(audio_inputs, enc_outputs,
                                           stream.cuda_stream)

        assert is_ok, "Runtime execution failed for Conformer Encoder session"
        stream.synchronize()
        enc_mask, emb_len = self.get_masked_emb(enc_outputs)

        emb_len = torch.clip(emb_len, max=enc_mask.shape[1])
        return enc_mask, emb_len


class CanaryDecoding:

    def __init__(self,
                 engine_dir,
                 runtime_mapping,
                 tokenizer,
                 debug_mode=False,
                 device="cuda:0",
                 use_py_session=False):
        self.tokenizer = tokenizer
        self.decoder_config = read_config('decoder', engine_dir)
        self.prompt_format = self.decoder_config['prompt_format']
        self.tokenizer.set_prompt_format(self.prompt_format)
        self.dtype = str_dtype_to_torch(self.decoder_config['dtype'])
        self.max_seq_len = self.decoder_config['max_seq_len']
        self.max_input_len = self.decoder_config['max_input_len']
        self.device = device
        self.use_py_session = use_py_session

        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode, use_py_session)

    @staticmethod
    def get_x_attention_mask(lens, max_length):
        batch_size = lens.shape[0]
        mask = torch.arange(max_length).repeat(batch_size,
                                               1).to(lens.device) < lens[:,
                                                                         None]
        return mask

    def get_py_session(self, engine_dir, runtime_mapping, debug_mode=False):
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

    def get_cpp_session(self, engine_dir, runtime_mapping, debug_mode=False):
        runner_kwargs = dict(
            engine_dir=os.path.join(engine_dir, 'decoder'),
            is_enc_dec=False,
            max_batch_size=self.decoder_config['max_batch_size'],
            max_input_len=self.max_input_len,
            max_output_len=self.max_seq_len - self.max_input_len,
            max_beam_width=self.decoder_config['max_beam_width'],
            debug_mode=debug_mode,
            kv_cache_free_gpu_memory_fraction=0.9,
            cross_kv_cache_fraction=0.5)
        model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)
        return model_runner_cpp

    def get_session(self,
                    engine_dir,
                    runtime_mapping,
                    debug_mode=False,
                    use_py_session=False):
        if use_py_session:
            return self.get_py_session(engine_dir, runtime_mapping, debug_mode)
        else:
            return self.get_cpp_session(engine_dir, runtime_mapping, debug_mode)

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_input_lengths,
                 max_new_tokens,
                 num_beams=1):

        encoder_outputs = encoder_outputs.to(dtype=self.dtype)
        batch_size = decoder_input_ids.shape[0]

        encoder_max_input_length = encoder_outputs.shape[1]

        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        assert decoder_max_input_length <= self.max_input_len, f"Decoder input length {decoder_max_input_length} exceeds max input length {self.max_input_len}"

        if max_new_tokens > self.max_seq_len:
            print(
                f"max_new_tokens {max_new_tokens} is greater than max_seq_len {self.max_seq_len}, setting max_new_tokens to max_seq_len"
            )
            max_new_tokens = self.max_seq_len
        max_new_tokens -= self.max_input_len

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()

        if self.use_py_session:
            cross_attention_mask = torch.ones([
                batch_size, decoder_max_input_length + max_new_tokens,
                encoder_max_input_length
            ]).int().cuda()
            if self.decoder_config['plugin_config']['remove_input_padding']:
                decoder_input_ids = remove_tensor_padding(
                    decoder_input_ids, pad_value=self.tokenizer.pad_id)
                if encoder_outputs.dim() == 3:
                    encoder_input_lengths = torch.full(
                        (encoder_outputs.shape[0], ),
                        encoder_max_input_length,
                        dtype=torch.int32,
                        device='cuda')
                    encoder_outputs = remove_tensor_padding(
                        encoder_outputs, encoder_input_lengths)

            # generation config
            sampling_config = SamplingConfig(end_id=self.tokenizer.eos_id,
                                             pad_id=self.tokenizer.pad_id,
                                             num_beams=num_beams)
            self.decoder_generation_session.setup(
                decoder_input_lengths.size(0),
                decoder_max_input_length,
                max_new_tokens=max_new_tokens,
                beam_width=num_beams,
                encoder_max_input_length=encoder_max_input_length,
            )
            torch.cuda.synchronize()
            output_ids = self.decoder_generation_session.decode(
                decoder_input_ids,
                decoder_input_lengths,
                sampling_config,
                encoder_output=encoder_outputs,
                encoder_input_lengths=encoder_input_lengths,
                cross_attention_mask=cross_attention_mask,
            )
            torch.cuda.synchronize()
        else:
            cross_attention_masks = [
                torch.ones([
                    decoder_input_lengths[i] + max_new_tokens,
                    encoder_input_lengths[i]
                ],
                           dtype=torch.bool,
                           device='cuda') for i in range(batch_size)
            ]
            decoder_input_ids = unpack_tensors(decoder_input_ids,
                                               decoder_input_lengths)
            encoder_outputs = unpack_tensors(encoder_outputs,
                                             encoder_input_lengths)
            out = self.decoder_generation_session.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=encoder_outputs,
                encoder_output_lengths=encoder_input_lengths,
                cross_attention_masks=cross_attention_masks,
                max_new_tokens=max_new_tokens,
                end_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                num_beams=num_beams,
                output_sequence_lengths=True,
                return_dict=True)
            output_ids = out['output_ids']
            out['sequence_lengths']

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class CanaryTRTLLM(object):

    def __init__(self,
                 engine_dir,
                 debug_mode=False,
                 device="cuda:0",
                 batch_size=8,
                 num_beams=1,
                 use_py_session=False):

        self.device = device
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        read_config('decoder', engine_dir)
        with open(os.path.join(engine_dir, 'preprocessor/config.json')) as f:
            preprocessor_config = json.load(f)

        self.decoder_config = read_config('decoder', engine_dir)
        self.max_seq_len = self.decoder_config['max_seq_len']
        self.max_input_len = self.decoder_config['max_input_len']
        self.max_batch_size = self.decoder_config['max_batch_size']

        self.num_feats = preprocessor_config['features']
        self.n_fft = preprocessor_config['n_fft']
        mel_basis_file = engine_dir / "preprocessor/mel_basis.pt"
        self.mel_basis = torch.load(mel_basis_file,
                                    weights_only=True,
                                    map_location=torch.device(self.device))

        window_size = preprocessor_config.get('window_size', 0.025)
        window_stride = preprocessor_config.get('window_stride', 0.010)
        window_type = preprocessor_config.get('window', 'hann')
        preemp = preprocessor_config.get('preemp', False)
        sample_rate = preprocessor_config.get('sample_rate', 16000)

        self.preprocessor = MelFilterBankFeats(self.mel_basis,
                                               window_size=window_size,
                                               window_stride=window_stride,
                                               window_type=window_type,
                                               fs=sample_rate,
                                               preemp=preemp)
        self.tokenizer = CanaryTokenizer(engine_dir)
        self.encoder = CanaryEncoder(engine_dir)
        self.decoder = CanaryDecoding(engine_dir,
                                      runtime_mapping,
                                      tokenizer=self.tokenizer,
                                      debug_mode=debug_mode,
                                      device=self.device,
                                      use_py_session=use_py_session)

        self.use_py_session = use_py_session

    def process_batch(
        self,
        audio,
        audio_input_lengths,
        text_prefix=None,
        num_beams=1,
        max_new_tokens=None,
        prompts_cfg=None,
    ):
        batch_size = len(audio_input_lengths)
        if prompts_cfg is None:
            if text_prefix is None:
                text_prefix = self.tokenizer.default_prompt

            prompt_id = self.tokenizer.encode(text_prefix)
            prompt_id = torch.tensor(prompt_id)
            decoder_input_ids = prompt_id.repeat(batch_size, 1)
        else:
            prompt_ids = []
            for cfg in prompts_cfg:
                prompt_id = self.tokenizer.get_prompt_ids_from_cfg(cfg)
                prompt_ids.append(torch.tensor(prompt_id))
            decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
                prompt_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_id).to(self.device)

        stream = torch.cuda.current_stream('cuda')

        if max_new_tokens is None:
            max_new_tokens = self.max_seq_len

        mel, mel_input_lengths = self.preprocessor.get_feats(
            audio, audio_input_lengths)

        encoder_output, encoder_output_lengths = self.encoder.infer(
            mel, mel_input_lengths, stream)
        output_ids = self.decoder.generate(decoder_input_ids,
                                           encoder_output,
                                           encoder_output_lengths,
                                           max_new_tokens=max_new_tokens,
                                           num_beams=num_beams)

        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            text = text.lstrip(punctuation)
            texts.append(text)
        return texts


def decode_wav_file(input_file_path,
                    model,
                    max_new_tokens=None,
                    batch_size=1,
                    text_prefix=None,
                    num_beams=1):

    waveform, sample_rate = librosa.load(input_file_path, sr=16000)

    total_duration = waveform.shape[0] / sample_rate
    waveform = torch.from_numpy(waveform).to(dtype=torch.float32,
                                             device="cuda:0")

    predictions = model.process_batch([waveform] * batch_size,
                                      [len(waveform)] * batch_size, text_prefix,
                                      num_beams, max_new_tokens)

    prediction = predictions[0]

    # remove all special tokens in the prediction
    prediction = re.sub(r'<\|.*?\|>', '', prediction)

    print(f"prediction: {prediction}")
    results = [(0, [""], prediction.split())]
    return results, total_duration * batch_size


def batch_manifest(manifest_file, batch_size):
    waveforms, durations, labels, ids, prompts_cfg = [], [], [], [], []
    count = 0
    max_batch_len = 0
    with open(manifest_file, 'r') as manifest:
        for line in manifest:
            data = json.loads(line)
            waveform, sample_rate = librosa.load(data['audio_filepath'],
                                                 sr=16000)

            duration = len(waveform)
            if max_batch_len < duration:
                max_batch_len = duration

            prompt = {
                'task': data.get('taskname', 'transcribe'),
                'pnc': data.get('pnc', 'no') == 'yes',
                'source_language': data.get('source_language', 'en-US'),
                'target_language': data.get('source_language', 'en-US'),
                'itn': data.get('itn', "no") == 'yes',
                'timestamp': data.get('timestamp', "no") == 'yes',
                'diarize': data.get('diarize', "no") == 'yes',
            }
            if 'text' in data:
                prompt['text'] = data['text']
            else:
                data['text'] = "na"
            if 'answer' in data:
                prompt['answer'] = data['answer']
            prompts_cfg.append(prompt)
            labels.append(data['text'])
            ids.append(data['audio_filepath'])
            durations.append(duration)
            waveforms.append(waveform)
            count += 1

            if count == batch_size:
                yield waveforms, durations, labels, ids, prompts_cfg, max_batch_len
                waveforms, durations, labels, ids, prompts_cfg = [], [], [], [], []
                count = 0
                max_batch_len = 0

        if count > 0:
            yield waveforms, durations, labels, ids, prompts_cfg, max_batch_len
        else:
            return


def decode_manifest(manifest_file,
                    model,
                    max_new_tokens=None,
                    batch_size=1,
                    num_beams=1,
                    sample_rate=16000,
                    output_manifest=None,
                    warmstart_batches=None):

    results = []
    total_duration = 0
    total_batches = 0

    for waveforms, durations, texts, ids, prompt_cfg, max_batch_len in batch_manifest(
            manifest_file, batch_size):
        for idx in range(len(waveforms)):
            max_batch_len = max(max_batch_len, sample_rate * 3)
            waveform = pad_or_trim(waveforms[idx], max_batch_len)

            waveform = waveform.astype(np.float32)
            waveform = torch.from_numpy(waveform)
            waveforms[idx] = waveform

        total_duration += sum(durations) / sample_rate
        predictions = model.process_batch(waveforms,
                                          durations,
                                          num_beams=num_beams,
                                          max_new_tokens=max_new_tokens,
                                          prompts_cfg=prompt_cfg)
        for wav_id, label, prediction, cfg in zip(ids, texts, predictions,
                                                  prompt_cfg):
            # remove all special tokens in the prediction
            prediction = re.sub(r'<\|.*?\|>', '', prediction)

            data = {
                'audio_filepath': wav_id,
                'source_lang': cfg['source_language'],
                'target_lang': cfg['target_language'],
                'pnc': 'no' if cfg['pnc'] == 'yes' else 'yes',
                'task': cfg['task'],
                'prediction': prediction,
                'answer': cfg.get('answer', 'na'),
                'text': cfg.get('text', 'na')
            }

            results.append((ids, texts, prompt_cfg))
            if output_manifest is not None:
                output_manifest.append(data)

            total_batches += 1

            if warmstart_batches is not None and total_batches >= warmstart_batches:
                return results, total_duration

    return results, total_duration


def collate_wrapper(batch):
    speeches, durations, labels, ids = [], [], [], []
    for item in batch:
        speech = item["audio"]["array"]
        duration = speech.shape[-1]
        speech = pad_or_trim(speech, N_SAMPLES)
        speech = speech.astype(np.float32)
        speech = torch.from_numpy(speech)
        speeches.append(speech)
        durations.append(duration)
        labels.append(item["text"])
        ids.append(item["id"])
    return speeches, durations, labels, ids


def decode_dataset(model,
                   dataset,
                   max_new_tokens=None,
                   text_prefix=None,
                   batch_size=1,
                   num_beams=1,
                   sample_rate=16000):
    librispeech_dummy = load_dataset(dataset, "clean", split="validation")

    data_loader = DataLoader(librispeech_dummy,
                             batch_size=batch_size,
                             num_workers=4,
                             pin_memory=False,
                             collate_fn=collate_wrapper)
    results = []
    total_duration = 0
    start_time = time.time()
    for batch in data_loader:
        waveforms, durations, texts, ids = batch
        total_duration += sum(durations) / sample_rate
        max_durations = max(durations)
        max_batch_len = max(max_durations, sample_rate * 3)
        waveforms_list = []

        for idx in range(len(waveforms)):
            waveform = pad_or_trim(waveforms[idx], max_batch_len)
            waveforms_list.append(waveform)

        predictions = model.process_batch(waveforms_list, durations,
                                          text_prefix, num_beams,
                                          max_new_tokens)
        for wav_id, label, prediction in zip(ids, texts, predictions):
            # remove all special tokens in the prediction
            prediction = re.sub(r'<\|.*?\|>', '', prediction)

            results.append(
                (wav_id, label.lower().split(), prediction.lower().split()))
    return results, total_duration, start_time


if __name__ == '__main__':
    args = parse_arguments()

    tensorrt_llm.logger.set_level(args.log_level)
    model = CanaryTRTLLM(args.engine_dir,
                         debug_mode=args.debug,
                         device="cuda:0",
                         use_py_session=args.use_py_session,
                         batch_size=args.batch_size,
                         num_beams=args.num_beams)

    if args.batch_size is None:
        if args.input_file:
            args.batch_size = 1
        else:
            args.batch_size = model.max_batch_size
    else:
        if args.batch_size > model.max_batch_size:
            print(
                f"batch_size {args.batch_size} is greater than max_batch_size {model.max_batch_size}, setting batch_size to max_batch_size"
            )
            args.batch_size = model.max_batch_size

    log_file = None
    if args.manifest_file is not None:
        args.results_dir = Path(args.manifest_file).parent.absolute()
        mf_name = Path(args.manifest_file).stem
        args.results_manifest = os.path.join(
            args.results_dir, f"{args.name}_manifest_{mf_name}.json")
        log_file = os.path.join(args.results_dir,
                                f"{args.name}_log_{mf_name}.log")
    if args.results_manifest is not None:
        output_manifest = []
    else:
        output_manifest = None
    if args.enable_warmup:
        if args.manifest_file:
            decode_manifest(args.manifest_file,
                            model,
                            batch_size=args.batch_size,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            output_manifest=None,
                            warmstart_batches=10)
        elif args.input_file:
            decode_wav_file(
                args.input_file,
                model,
                text_prefix=args.prompt_text,
                batch_size=args.batch_size,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            decode_dataset(
                model,
                "hf-internal-testing/librispeech_asr_dummy",
                batch_size=args.batch_size,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
    if args.input_file:
        start_time = time.time()

        results, total_duration = decode_wav_file(
            args.input_file,
            model,
            text_prefix=args.prompt_text,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )

    elif args.manifest_file:

        start_time = time.time()

        results, total_duration = decode_manifest(
            args.manifest_file,
            model,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            output_manifest=output_manifest,
            max_new_tokens=args.max_new_tokens,
        )

    else:

        results, total_duration, start_time = decode_dataset(
            model,
            args.dataset,
            batch_size=args.batch_size,
            text_prefix=args.prompt_text,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )

    elapsed = time.time() - start_time
    results = sorted(results)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    store_transcripts(filename=f"{args.results_dir}/recogs-{args.name}.txt",
                      texts=results)

    s = ""

    if output_manifest is None and args.input_file is None:
        with open(f"{args.results_dir}/errs-{args.name}.txt", "w") as f:
            total_error_rate = write_error_stats(f,
                                                 "test-set",
                                                 results,
                                                 enable_log=True)
            if args.accuracy_check and args.dataset == "hf-internal-testing/librispeech_asr_dummy" and not args.input_file:
                assert total_error_rate <= 2.0, f"Word Error rate using canary model should be 1.22%, but got {total_error_rate}"
            s = f"total error rate: {total_error_rate:.2f}%\n"

    rtf = total_duration / elapsed
    s += f"RTFx: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    s += f"num_beams: {args.num_beams}\n"

    print(s)

    if output_manifest is not None:
        with open(args.results_manifest, 'w') as ofp:
            for data in output_manifest:
                ofp.write(f"{json.dumps(data)}\n")
        with open(log_file, 'a') as f:
            f.write(s)

    with open(f"{args.results_dir}/rtf-{args.name}.txt", "w") as f:
        f.write(s)

    del model
