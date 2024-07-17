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
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer

from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.builder import get_engine_version

DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'GPTForCausalLM': 'gpt2-medium',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'InternLM2ForCausalLM': 'internlm/internlm2-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'QWenForCausalLM': 'Qwen/Qwen-7B',
    'RecurrentGemmaForCausalLM': 'google/recurrentgemma-2b',
}

INTERNLM_META_INSTRUCTION = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM':
    "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'InternLM2ForCausalLM':
    "<|im_start|>system\n" + INTERNLM_META_INSTRUCTION +
    "<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
    'QWenForCausalLM':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
}


def supports_inflight_batching(engine_dir):
    config_path = Path(engine_dir) / "config.json"
    json_config = GptJsonConfig.parse_file(config_path)
    model_config = json_config.model_config
    return model_config.supports_inflight_batching


def read_decoder_start_token_id(engine_dir):
    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)
    return config['pretrained_config']['decoder_start_token_id']


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = config['pretrained_config']['chatglm_version']
    if model_arch == 'QWenForCausalLM':
        model_version = config['pretrained_config']['qwen_type']
    return model_arch, model_version


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
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
    if model_name == 'QWenForCausalLM' and model_version == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        pad_id = gen_config['pad_token_id']
        end_id = gen_config['eos_token_id']
    elif model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def add_common_args(parser):
    # sampling arguments
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams > 1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--early_stopping',
                        type=int,
                        help='Use early stopping if num_beams > 1'
                        '1 for early-stopping, 0 for non-early-stopping'
                        'other values for stopping by length',
                        default=1)
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
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behavior'
    )
    parser.add_argument('--log_level', type=str, default='info')
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
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
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
        action='store_true',
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
    return parser
