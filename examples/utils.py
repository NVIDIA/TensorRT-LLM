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
import ast
import json
import os
import subprocess
import sys
from argparse import BooleanOptionalAction
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer

from tensorrt_llm._utils import supports_inflight_batching  # noqa
from tensorrt_llm._utils import (mpi_barrier, mpi_rank, mpi_world_size,
                                 str_dtype_to_torch)
from tensorrt_llm.builder import get_engine_version

DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BaiChuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'GLMModel': 'THUDM/glm-10b',
    'ChatGLMModel': 'THUDM/chatglm3-6b',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'RWForCausalLM': 'tiiuae/falcon-rw-1b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'GPT2LMHeadModel': 'gpt2',
    'GPT2LMHeadCustomModel': 'gpt2',
    'Starcoder2ForCausalLM': 'bigcode/starcoder2-3b',
    'GPTForCausalLM': 'gpt2',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'InternLM2ForCausalLM': 'internlm/internlm2-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'QWenLMHeadModel': 'Qwen/Qwen-7B',
    'QWenForCausalLM': 'Qwen/Qwen-7B',
    'Qwen2ForCausalLM': 'Qwen/Qwen1.5-7B',
    'Qwen2MoeForCausalLM': 'Qwen/Qwen1.5-MoE-A2.7B',
    'RecurrentGemmaForCausalLM': 'google/recurrentgemma-2b',
}

INTERNLM_META_INSTRUCTION = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

QWEN_PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM': "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'InternLM2ForCausalLM': "<|im_start|>system\n" + INTERNLM_META_INSTRUCTION +
    "<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
    'QWenLMHeadModel': QWEN_PROMPT_TEMPLATE,
    'QWenForCausalLM': QWEN_PROMPT_TEMPLATE,
    'Qwen2ForCausalLM': QWEN_PROMPT_TEMPLATE,
    'Qwen2MoeForCausalLM': QWEN_PROMPT_TEMPLATE,
}


def read_decoder_start_token_id(engine_dir):
    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)
    return config['pretrained_config']['decoder_start_token_id']


def read_is_enc_dec(engine_dir: str, is_hf: bool = False):
    if is_hf:
        with open(Path(engine_dir) / "config.json", 'r') as f:
            config = json.load(f)
        is_enc_dec = config.get('is_encoder_decoder', False)
    else:
        is_enc_dec = {'encoder', 'decoder'}.issubset({
            name
            for name in os.listdir(engine_dir)
            if os.path.isdir(os.path.join(engine_dir, name))
        })
    return is_enc_dec


def read_model_name(engine_dir: str, is_hf: bool = False):
    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if is_hf:
        model_arch = config['architectures'][0]
        model_version = config.get('model_type', None)
        return model_arch, model_version

    engine_version = get_engine_version(engine_dir)
    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if 'GLM' in model_arch:
        model_version = config['pretrained_config']['chatglm_version']
    if 'qwen' in model_arch.lower():
        model_version = config['pretrained_config']['qwen_type']
    return model_arch, model_version


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


# Load tokenizer impl, it will be called in external wrapper to avoid loading tokenizer bug under MPI env.
def _load_tokenizer(tokenizer_dir: Optional[str] = None,
                    vocab_file: Optional[str] = None,
                    model_name: str = 'GPTForCausalLM',
                    model_version: Optional[str] = None,
                    tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        if 'whisper' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained('openai/whisper-large-v3',
                                                      language='english',
                                                      task='transcribe',
                                                      predict_timestamps=False)
        elif tokenizer_type == 'language_adapter':
            tokenizer = None
        else:
            use_fast = True
            if tokenizer_type is not None and tokenizer_type == "llama":
                use_fast = False
            # Should set both padding_side and truncation_side to be 'left'
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                legacy=False,
                padding_side='left',
                truncation_side='left',
                trust_remote_code=True,
                tokenizer_type=tokenizer_type,
                use_fast=use_fast)
    elif model_name == 'GemmaForCausalLM' or model_name == 'RecurrentGemmaForCausalLM':
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file,
                                   padding_side='left',
                                   truncation_side='left',
                                   legacy=False)
    elif model_name == 'Grok1ModelForCausalLM':
        tokenizer = LlamaTokenizer(vocab_file=vocab_file,
                                   padding_side='left',
                                   truncation_side='left',
                                   legacy=False,
                                   use_fast=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left',
                                legacy=False)
    if 'qwen' in model_name.lower() and model_version == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config['pad_token_id']
        end_id = gen_config['eos_token_id']
    elif 'GLM' in model_name and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    elif tokenizer_type == 'language_adapter':
        pad_id = 0
        end_id = 2
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    func = partial(_load_tokenizer, tokenizer_dir, vocab_file, model_name,
                   model_version, tokenizer_type)
    if mpi_world_size() > 1:
        # Under MPI env, load tokenizer will result in multiple processes to download the same file to the same folder.
        # This will result some random bug. Force loading on rank0 to warmup the tokenizer to avoid this issue.
        if mpi_rank() == 0:
            func()
        mpi_barrier()
    return func()


def prepare_enc_dec_inputs(batch_input_ids: List[torch.Tensor], model_name: str,
                           engine_dir: str,
                           multimodal_input_file: Optional[str]):
    encoder_input_features = None
    encoder_input_ids = None
    if 'whisper' in model_name.lower():
        # cannot directly import whisper due to name collision
        sys.path.append(f"{os.path.dirname(__file__)}/models/core/whisper")
        from whisper_utils import log_mel_spectrogram

        config_path = os.path.join(engine_dir, 'encoder', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        n_mels = config['pretrained_config']['n_mels']
        dtype = config['pretrained_config']['dtype']

        # download mel filters file
        subprocess.run([
            "wget", "-nc", f"--directory-prefix={engine_dir}",
            "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"
        ],
                       check=True)

        mel, total_duration = log_mel_spectrogram(multimodal_input_file,
                                                  n_mels,
                                                  return_duration=True,
                                                  mel_filters_dir=engine_dir)
        mel = mel.type(str_dtype_to_torch(dtype))  # [featureDim, seqLen]
        decoder_input_ids = batch_input_ids
        encoder_input_features = [torch.einsum('DL->LD', mel)]
        encoder_output_lengths = [encoder_input_features[0].shape[0] // 2]
    else:
        encoder_input_ids = batch_input_ids
        decoder_start_token_id = read_decoder_start_token_id(
            os.path.join(engine_dir, "decoder"))
        decoder_input_ids = [
            torch.tensor([decoder_start_token_id], dtype=torch.int32)
            for _ in batch_input_ids
        ]
        encoder_output_lengths = None
    return encoder_input_ids, encoder_input_features, encoder_output_lengths, decoder_input_ids


def get_beam_width_array(bwa: str = None):
    bwa = ast.literal_eval(bwa)  # Short for "beam_width_array"
    if isinstance(bwa, str):
        bwa = ast.literal_eval(bwa)  # parse again for string

    def parse_one_bwa(row):
        assert isinstance(row, list), f"Beam width array must be a list."
        assert len(
            row
        ) <= 8, "Length of beam width array must not be greater than 8 now."
        assert all([isinstance(beam, int) for beam in row
                    ]), "Numbers in beam width array must be integer."
        bwa_tensor = torch.zeros([8], dtype=torch.int32)
        for j in range(len(row)):
            bwa_tensor[j] = row[j]
        bwa_tensor[len(row):] = row[-1]
        return bwa_tensor, max(row)

    if isinstance(bwa, list):  # Only one BWA
        bwa_tensor, max_beam_width = parse_one_bwa(bwa)
    elif isinstance(bwa, tuple):  # BWA for respective requests
        bwa_tensor_list = []
        max_beam_width = 0
        for row in bwa:
            bwa_tensor, beam_width = parse_one_bwa(row)
            bwa_tensor_list.append(bwa_tensor)
            max_beam_width = max(max_beam_width, beam_width)
        bwa_tensor = torch.stack(bwa_tensor_list, dim=0)
    else:
        raise ValueError(f"Invalid beam width array: {bwa}")

    return bwa_tensor.tolist(), max_beam_width


def add_common_args(parser):
    # sampling arguments
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams > 1",
                        default=1)
    parser.add_argument('--num_return_sequences',
                        type=int,
                        help="Number of sequences to generate for each input.",
                        default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--early_stopping',
                        type=int,
                        help='Use early stopping if num_beams > 1, '
                        '1 for early-stopping, 0 for non-early-stopping'
                        'other values for stopping by length',
                        default=1)
    parser.add_argument(
        '--beam_width_array',
        type=str,
        default=None,
        help=
        'Beam width array for each step. E.g.: --beam_width_array="[2,4,6,8]"',
    )
    parser.add_argument(
        '--end_id',
        default=None,
        type=int,
        help="Override tokenizer end_id to stop on given end_id token.")
    parser.add_argument(
        '--stop_words',
        default=None,
        type=str,
        nargs="+",
        action='append',
        help=
        'Set stop words for a batch. Successive invocations of --stop_words set stop words for other batches.'
        '    E.g.: --stop_words " London" " chef" --stop_words "eventually became" "was not"',
    )
    parser.add_argument(
        '--bad_words',
        default=None,
        type=str,
        nargs="+",
        action='append',
        help=
        'Set bad words for a batch. Successive invocations of --bad_words set bad words for other batches.'
        '    E.g.: --bad_words " London" " chef" --bad_words "eventually became" "was not"',
    )
    parser.add_argument('--no_repeat_ngram_size', type=int, default=None)

    # common runtime arguments
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        nargs="+",
        help=
        'The attention window size that controls the sliding window attention kv cache behavior'
    )
    parser.add_argument(
        '--multi_block_mode',
        type=lambda s: s.lower() in
        ("yes", "true", "t", "1"
         ),  # custom boolean function to convert input string to boolean
        default=True,
        help=
        "Distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel."
    )
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        action='store_true',
                        help="Enable FMHA runner FP32 accumulation.")
    parser.add_argument('--cuda_graph_mode',
                        action='store_true',
                        help="Enable cuda graphs in the inference.")
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['verbose', 'info', 'warning', 'error', 'internal_error'],
        default='info')
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        nargs="+",
                        help="The directory of LoRA weights")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
    parser.add_argument(
        '--draft_target_model_config',
        type=str,
        default=None,
        help=
        "Configuration of Draft-Target-Model decoding, see `examples/draft_target_model/README.md` for more information."
        "   E.g.: [4, [0], [1], False] for [draft_len, draft_model_device_list, target_model_device_list, use_logits]."
    )
    parser.add_argument(
        '--ngram_config',
        type=str,
        default=None,
        help=
        "Configuration of NGram decoding, see `examples/ngram/README.md` for more information."
        "   E.g.: [10,2,[0]] for [max_draft_len, max_matching_ngram_size, device_list].",
    )
    parser.add_argument(
        '--medusa_choices',
        type=str,
        default=None,
        help="Configuration of Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    )
    parser.add_argument(
        '--eagle_choices',
        type=str,
        default=None,
        help="Configuration of Eagle-1 decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 draft tokens."
    )
    parser.add_argument(
        '--eagle_posterior_threshold',
        type=float,
        default=None,
        help="Minimum token probability threshold for typical acceptance. "
        "Enables typical acceptance in Eagle. "
        "Corresponds to epsilon in https://arxiv.org/pdf/2401.10774.")
    parser.add_argument('--eagle_use_dynamic_tree',
                        action='store_true',
                        help="Whether to use Ealge-2")
    parser.add_argument(
        '--eagle_dynamic_tree_max_top_k',
        default=None,
        type=int,
        help=
        "The maximum number of draft tokens to expand for each node in Eagle-2."
    )
    parser.add_argument(
        '--lookahead_config',
        type=str,
        default=None,
        help="Configuration of executor and request lookahead decoding."
        "   E.g.: [5, 6, 7] for [max_window_size, max_ngram_size, max_verification_set_size]."
    )
    # model arguments
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--hf_model_dir', '--model_dir', type=str, default=None)
    parser.add_argument(
        '--tokenizer_dir',
        default=None,
        help='tokenizer path; defaults to hf_model_dir if left unspecified')

    # memory argument
    parser.add_argument(
        '--gpu_weights_percent',
        default=1,
        type=float,
        help=
        'Specify the percentage of weights that reside on GPU instead of CPU and streaming load during runtime.',
    )
    parser.add_argument(
        '--max_tokens_in_paged_kv_cache',
        default=None,
        type=int,
        help=
        'Specify the maximum number of tokens in a kv cache page (only available with cpp session).',
    )
    parser.add_argument(
        '--kv_cache_enable_block_reuse',
        default=True,
        action=BooleanOptionalAction,
        help=
        'Enables block reuse in kv cache (only available with cpp session).',
    )
    parser.add_argument(
        '--kv_cache_free_gpu_memory_fraction',
        default=0.9,
        type=float,
        help='Specify the free gpu memory fraction.',
    )
    parser.add_argument(
        '--cross_kv_cache_fraction',
        default=0.5,
        type=float,
        help=
        'Specify the kv cache fraction reserved for cross attention. Only applicable for encoder-decoder models. By default 0.5 for self and 0.5 for cross.',
    )
    parser.add_argument(
        '--enable_chunked_context',
        action='store_true',
        help='Enables chunked context (only available with cpp session).',
    )

    # hf model argument (if use hf model)
    parser.add_argument(
        '--hf_data_type',
        '--data_type',
        type=str,
        choices=['fp32', 'fp16', 'bf16', 'float32', 'float16', 'bfloat16'],
        default='fp16',
        help="The data type for hf model.")
    parser.add_argument(
        '--hf_device_map_auto',
        action='store_true',
        help="Use device map 'auto' to load a pretrained HF model. This may "
        "help to test a large model that cannot fit into a singlue GPU.")

    parser.add_argument(
        "--return_all_generated_tokens",
        default=False,
        action="store_true",
        help="This option changes the token output only for streaming. "
        "If not specified, return only generated tokens at each step. "
        "If specified, return the full beams/outputs at each step. "
        "It is automatically enabled for num_beams>1 (only available with cpp session). "
        "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."
    )

    parser.add_argument(
        '--language_task_uids',
        type=int,
        nargs='+',
        default=None,
        help=
        "language task id indicating which adapter to use in language adapter. Please include 1 locale per input text"
    )
    parser.add_argument('--backend', type=str, default=None)

    return parser
