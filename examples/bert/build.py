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
import os
from collections import OrderedDict

# isort: off
import torch
import tensorrt as trt
# isort: on

from transformers import BertConfig, BertForQuestionAnswering, BertForSequenceClassification, BertModel  # isort:skip
from transformers import RobertaConfig, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaModel  # isort:skip

from weight import (load_from_hf_cls_model, load_from_hf_model,
                    load_from_hf_qa_model)

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='Tensor parallelism size')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--timing_cache', type=str, default='model.cache')
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=51200)
    parser.add_argument('--n_labels', type=int, default=2)
    parser.add_argument('--n_layer', type=int, default=24)
    parser.add_argument('--n_positions', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='bert_outputs')

    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument('--use_bert_attention_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--model',
                        default='BertModel',
                        choices=[
                            'BertModel',
                            'BertForQuestionAnswering',
                            'BertForSequenceClassification',
                            'RobertaModel',
                            'RobertaForQuestionAnswering',
                            'RobertaForSequenceClassification',
                        ])
    parser.add_argument('--model_dir', type=str, required=False)
    return parser.parse_args()


def prepare_inputs():
    # opt_shape is set to half of max batch_size and seq_len by default
    # tune this according to real data distribution
    bs_range = [1, (args.max_batch_size + 1) // 2, args.max_batch_size]
    inlen_range = [1, (args.max_input_len + 1) // 2, args.max_input_len]
    num_tokens_range = [
        1,
        (args.max_input_len * args.max_batch_size + 1) // 2,
        args.max_input_len * args.max_batch_size,
    ]
    if not args.remove_input_padding:
        input_ids = tensorrt_llm.Tensor(
            name='input_ids',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('input_len', [inlen_range])]),
        )
        # also called segment_ids
        token_type_ids = tensorrt_llm.Tensor(
            name='token_type_ids',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('input_len', [inlen_range])]),
        )
    else:
        input_ids = tensorrt_llm.Tensor(
            name="input_ids",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("num_tokens", [num_tokens_range])]),
        )
        token_type_ids = tensorrt_llm.Tensor(
            name='token_type_ids',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('num_tokens', [num_tokens_range])]),
        )
        position_ids = tensorrt_llm.Tensor(
            name='position_ids',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('num_tokens', [num_tokens_range])]),
        )
        max_input_length = tensorrt_llm.Tensor(
            name="max_input_length",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("max_input_length", [inlen_range])]),
        )
    input_lengths = tensorrt_llm.Tensor(name='input_lengths',
                                        dtype=trt.int32,
                                        shape=[-1],
                                        dim_range=OrderedDict([('batch_size',
                                                                [bs_range])]))

    inputs = {
        'input_ids': input_ids,
        'input_lengths': input_lengths,
        'token_type_ids': token_type_ids,
    }

    if args.remove_input_padding:
        inputs['position_ids'] = position_ids
        inputs['max_input_length'] = max_input_length

    return inputs


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    torch_dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    trt_dtype = trt.float16 if args.dtype == 'float16' else trt.float32

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=args.model,
        precision=args.dtype,
        timing_cache=args.timing_cache,
        profiling_verbosity=args.profiling_verbosity,
        tensor_parallel=args.world_size,  # TP only
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
    )
    # Initialize model
    if 'Roberta' in args.model:
        model_type = 'Roberta'
    else:
        model_type = 'Bert'

    # initialize config with input arguments and update from json
    config_cls = globals()[f'{model_type}Config']
    config = dict(
        vocab_size=args.vocab_size,
        num_labels=args.n_labels,
        num_hidden_layers=args.n_layer,
        max_position_embeddings=args.n_positions,
        hidden_size=args.n_embd,
        num_attention_heads=args.n_head,
        intermediate_size=4 * args.n_embd if args.n_embd else None,
        hidden_act=args.hidden_act,
        torch_dtype=torch_dtype,
    )
    if args.model_dir is not None:
        json_config = config_cls.get_config_dict(args.model_dir)[0]
        config.update((k, v) for k, v in json_config.items() if v is not None)
    bert_config = config_cls.from_dict(config)

    output_name = 'hidden_states'
    if args.model == 'BertModel' or args.model == 'RobertaModel':
        hf_bert = globals()[f'{model_type}Model'](bert_config,
                                                  add_pooling_layer=False)
        tensorrt_llm_bert = tensorrt_llm.models.BertModel(
            num_layers=bert_config.num_hidden_layers,
            num_heads=bert_config.num_attention_heads,
            hidden_size=bert_config.hidden_size,
            vocab_size=bert_config.vocab_size,
            hidden_act=bert_config.hidden_act,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            pad_token_id=bert_config.pad_token_id,
            is_roberta=(model_type == 'Roberta'),
            mapping=Mapping(world_size=args.world_size,
                            rank=args.rank,
                            tp_size=args.world_size),  # TP only
            dtype=trt_dtype)
        load_from_hf_model(
            tensorrt_llm_bert,
            hf_bert,
            bert_config,
            rank=args.rank,
            tensor_parallel=args.world_size,
            fp16=(args.dtype == 'float16'),
        )

    elif args.model == 'BertForQuestionAnswering' or args.model == 'RobertaForQuestionAnswering':
        hf_bert = globals()[f'{model_type}ForQuestionAnswering'](bert_config)
        tensorrt_llm_bert = tensorrt_llm.models.BertForQuestionAnswering(
            num_layers=bert_config.num_hidden_layers,
            num_heads=bert_config.num_attention_heads,
            hidden_size=bert_config.hidden_size,
            vocab_size=bert_config.vocab_size,
            hidden_act=bert_config.hidden_act,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            pad_token_id=bert_config.pad_token_id,
            is_roberta=(model_type == 'Roberta'),
            num_labels=args.
            n_labels,  # TODO: this might just need to be a constant
            mapping=Mapping(world_size=args.world_size,
                            rank=args.rank,
                            tp_size=args.world_size),  # TP only
            dtype=trt_dtype)
        load_from_hf_qa_model(
            tensorrt_llm_bert,
            hf_bert,
            bert_config,
            rank=args.rank,
            tensor_parallel=args.world_size,
            fp16=(args.dtype == 'float16'),
        )
        output_name = 'logits'
    elif args.model == 'BertForSequenceClassification' or args.model == 'RobertaForSequenceClassification':
        hf_bert = globals()[f'{model_type}ForSequenceClassification'](
            config=bert_config)
        if args.model_dir is not None and os.path.exist(
                os.path.join(args.model_dir, "pytorch_model.bin")):
            state_dict = torch.load(
                os.path.join(args.model_dir, "pytorch_model.bin"))
            hf_bert.load_state_dict(state_dict, strict=False)

        tensorrt_llm_bert = tensorrt_llm.models.BertForSequenceClassification(
            num_layers=bert_config.num_hidden_layers,
            num_heads=bert_config.num_attention_heads,
            hidden_size=bert_config.hidden_size,
            vocab_size=bert_config.vocab_size,
            hidden_act=bert_config.hidden_act,
            max_position_embeddings=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            pad_token_id=bert_config.pad_token_id,
            is_roberta=(model_type == 'Roberta'),
            num_labels=bert_config.num_labels,
            mapping=Mapping(world_size=args.world_size,
                            rank=args.rank,
                            tp_size=args.world_size),  # TP only
            dtype=trt_dtype)
        load_from_hf_cls_model(
            tensorrt_llm_bert,
            hf_bert,
            bert_config,
            rank=args.rank,
            tensor_parallel=args.world_size,
            fp16=(args.dtype == 'float16'),
        )
        output_name = 'logits'
    else:
        assert False, f"Unknown BERT model {args.model}"

    # Module -> Network
    network = builder.create_network()
    network.plugin_config.to_legacy_setting()
    if args.remove_input_padding:
        assert args.model == "BertForSequenceClassification", \
            "remove_input_padding is only supported for BertForSequenceClassification models"
        network.plugin_config.remove_input_padding = True
    if args.use_bert_attention_plugin:
        network.plugin_config.bert_attention_plugin = args.use_bert_attention_plugin
    if args.use_gemm_plugin:
        network.plugin_config.gemm_plugin = args.use_gemm_plugin
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_bert.named_parameters())

        # Forward
        inputs = prepare_inputs()

        # logits for QA BERT, or hidden_state for vanilla BERT
        output = tensorrt_llm_bert(**inputs)

        # Mark outputs
        output_dtype = trt.float16 if args.dtype == 'float16' else trt.float32
        output.mark_output(output_name, output_dtype)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, 'Failed to build engine.'
    engine_file = os.path.join(
        args.output_dir,
        get_engine_name(args.model, args.dtype, args.world_size, args.rank))
    with open(engine_file, 'wb') as f:
        f.write(engine)
    builder.save_config(builder_config,
                        os.path.join(args.output_dir, 'config.json'))
