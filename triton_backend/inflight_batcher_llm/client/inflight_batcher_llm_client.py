#!/usr/bin/env python
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import csv
import os
import queue
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

#
# Simple streaming client for TRT-LLM inflight bacthing backend
#
# In order for this code to work properly, config.pbtxt must contain these values:
#
# model_transaction_policy {
#   decoupled: True
# }
#
# parameters: {
#   key: "gpt_model_type"
#   value: {
#     string_value: "inflight_batching"
#   }
# }
#
# In order for gpt_model_type 'inflight_batching' to work, you must copy engine from
#
# tensorrt_llm/cpp/tests/resources/models/rt_engine/gpt2/fp16-inflight-batching-plugin/1-gpu/
#

np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})

_str_to_np_dict = dict(
    float16=np.float16,
    float32=np.float32,
    int32=np.int32,
    bfloat16=np_bfloat16,
)


def curate_log_output(token_sequence,
                      identifier="Input",
                      log_max_sequence_len=256):
    if len(token_sequence) > log_max_sequence_len:
        print(f"{identifier} sequence starts with: ",
              token_sequence[:log_max_sequence_len])
    else:
        print(f"{identifier} sequence: ", token_sequence)


def str_dtype_to_np(dtype):
    ret = _str_to_np_dict.get(dtype)
    assert ret is not None, f'Unsupported dtype: {dtype}'
    return ret


def check_output_names(expected_outputs, infer_result):
    if expected_outputs:
        output_names = set([o.name for o in infer_result._result.outputs])
        if set(expected_outputs) != output_names:
            raise Exception(
                f"expected outputs do not match actual outputs {expected_outputs} != {output_names}"
            )


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape,
                              np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_outputs(output_names):

    outputs = []
    for output_name in output_names:
        outputs.append(grpcclient.InferRequestedOutput(output_name))
    return outputs


def prepare_inputs(input_ids_data, input_lengths_data, request_output_len_data,
                   beam_width_data, temperature_data, repetition_penalty_data,
                   presence_penalty_data, frequency_penalty_data,
                   streaming_data, end_id, pad_id, prompt_embedding_table_data,
                   prompt_vocab_size_data, lora_task_id_data, lora_weights_data,
                   lora_config_data, return_log_probs_data, top_k_data,
                   top_p_data, draft_ids_data, return_context_logits_data,
                   return_generation_logits_data, decoder_input_ids_data,
                   prompt_table_extra_id_data, exclude_input_in_output,
                   num_return_sequences_data, return_perf_metrics_data,
                   lookahead_config_data, return_num_input_tokens_data,
                   return_num_output_tokens_data):
    inputs = [
        prepare_tensor("input_ids", input_ids_data),
        prepare_tensor("input_lengths", input_lengths_data),
        prepare_tensor("request_output_len", request_output_len_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
        prepare_tensor("streaming", streaming_data),
        prepare_tensor("end_id", end_id),
        prepare_tensor("pad_id", pad_id),
        prepare_tensor("runtime_top_k", top_k_data),
        prepare_tensor("runtime_top_p", top_p_data),
    ]
    if return_num_input_tokens_data is not None:
        inputs += [
            prepare_tensor("return_num_input_tokens",
                           return_num_input_tokens_data),
        ]
    if return_num_output_tokens_data is not None:
        inputs += [
            prepare_tensor("return_num_output_tokens",
                           return_num_output_tokens_data),
        ]
    if num_return_sequences_data is not None:
        inputs += [
            prepare_tensor("num_return_sequences", num_return_sequences_data)
        ]
    if prompt_embedding_table_data is not None:
        inputs += [
            prepare_tensor("prompt_embedding_table",
                           prompt_embedding_table_data),
            prepare_tensor("prompt_vocab_size", prompt_vocab_size_data)
        ]
    if lora_task_id_data is not None:
        inputs += [prepare_tensor("lora_task_id", lora_task_id_data)]
    if lora_weights_data is not None:
        inputs += [
            prepare_tensor("lora_weights", lora_weights_data),
            prepare_tensor("lora_config", lora_config_data),
        ]
    if repetition_penalty_data is not None:
        inputs += [
            prepare_tensor("repetition_penalty", repetition_penalty_data),
        ]
    if presence_penalty_data is not None:
        inputs += [
            prepare_tensor("presence_penalty", presence_penalty_data),
        ]
    if frequency_penalty_data is not None:
        inputs += [
            prepare_tensor("frequency_penalty", frequency_penalty_data),
        ]
    if draft_ids_data is not None:
        inputs += [
            prepare_tensor("draft_input_ids", draft_ids_data),
        ]
    if return_log_probs_data is not None:
        inputs += [
            prepare_tensor("return_log_probs", return_log_probs_data),
        ]
    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits", return_context_logits_data),
        ]
    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits",
                           return_generation_logits_data),
        ]
    if decoder_input_ids_data is not None:
        inputs += [
            prepare_tensor("decoder_input_ids", decoder_input_ids_data),
        ]
    if prompt_table_extra_id_data is not None:
        inputs += [
            prepare_tensor("prompt_table_extra_ids",
                           prompt_table_extra_id_data),
        ]
    if exclude_input_in_output is not None:
        inputs += [
            prepare_tensor("exclude_input_in_output", exclude_input_in_output),
        ]
    if return_perf_metrics_data is not None:
        inputs += [
            prepare_tensor("return_perf_metrics", return_perf_metrics_data),
        ]
    if lookahead_config_data is not None:
        inputs += [
            prepare_tensor(
                "lookahead_window_size",
                np.array([[lookahead_config_data[0]]], dtype=np.int32)),
            prepare_tensor(
                "lookahead_ngram_size",
                np.array([[lookahead_config_data[1]]], dtype=np.int32)),
            prepare_tensor(
                "lookahead_verification_set_size",
                np.array([[lookahead_config_data[2]]], dtype=np.int32)),
        ]
    return inputs


def prepare_stop_signals():

    inputs = [
        grpcclient.InferInput('input_ids', [1, 1], "INT32"),
        grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
        grpcclient.InferInput('request_output_len', [1, 1], "INT32"),
        grpcclient.InferInput('stop', [1, 1], "BOOL"),
    ]

    inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
    inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.int32))
    inputs[3].set_data_from_numpy(np.array([[True]], dtype='bool'))

    return inputs


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        if FLAGS.streaming:
            # Print the first sequence only in streaming.
            seq_idx = result.as_numpy('sequence_index')
            seq_idx = seq_idx[0][0] if seq_idx is not None else 0

            if seq_idx == 0 and result.get_output('output_ids') is not None:
                output_ids = result.as_numpy('output_ids')
                seq_lens = result.as_numpy('sequence_length')
                if seq_lens is None or seq_lens[0][0] > 0:
                    tokens = list(output_ids[0][0])
                    print(tokens, flush=True)


def expand_and_vstack(results: list, axis=0):
    if len(results) == 1:
        return np.vstack(results)
    return np.vstack([np.expand_dims(r, axis=axis) for r in results])


def parse_list(value):
    try:
        # Remove brackets and split by comma
        value = value.strip('[]')
        return [int(x.strip()) for x in value.split(',')]
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f'Invalid list format. Expected [x,y,z], got {value}. Error: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument('--text',
                        type=str,
                        required=False,
                        default='Born in north-east France, Soyer trained as a',
                        help='Input text')

    parser.add_argument('--input-tokens-csv',
                        type=str,
                        required=False,
                        default='',
                        help='Path to csv file containing the input tokens')

    parser.add_argument('--draft-tokens-csv',
                        type=str,
                        required=False,
                        default='',
                        help='Path to csv file containing the draft tokens')

    parser.add_argument(
        '--output-tokens-csv',
        type=str,
        required=False,
        default='',
        help='Path to csv file containing the expected output tokens')

    parser.add_argument(
        '--end-id',
        type=int,
        required=False,
        default=-1,
        help='The token id for end token. Only needed if tokenizer is not used.'
    )

    parser.add_argument(
        '--pad-id',
        type=int,
        required=False,
        default=50256,
        help='The token id for pad token. Only needed if tokenizer is not used.'
    )

    parser.add_argument(
        "-s",
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL encrypted channel to the server",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "-r",
        "--root-certificates",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded root certificates. Default is None.",
    )
    parser.add_argument(
        "-p",
        "--private-key",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded private key. Default is None.",
    )
    parser.add_argument(
        "-x",
        "--certificate-chain",
        type=str,
        required=False,
        default=None,
        help="File holding PEM-encoded certificate chain. Default is None.",
    )
    parser.add_argument(
        "-C",
        "--grpc-compression-algorithm",
        type=str,
        required=False,
        default=None,
        help=
        "The compression algorithm to be used when sending request to server. Default is None.",
    )
    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )
    parser.add_argument(
        "-c",
        "--check-output",
        action="store_true",
        required=False,
        default=False,
        help="Enable check of output ids for CI",
    )
    parser.add_argument(
        "--correctness-threshold",
        type=float,
        required=False,
        default=1.0,
        help="Error tolerance when checking output for CI",
    )

    parser.add_argument(
        "-n",
        "--num-return-sequences",
        type=int,
        required=False,
        default=None,
        help="Number of sequences to generate.",
    )
    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        required=False,
        default=None,
        help="The repetition penalty value",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        required=False,
        default=None,
        help="The frequency penalty value",
    )

    parser.add_argument(
        "--request-output-len",
        type=int,
        required=False,
        default=16,
        help="Request output length",
    )
    parser.add_argument(
        '--stop-after-ms',
        type=int,
        required=False,
        default=0,
        help='Early stop the generation after a few milliseconds')
    parser.add_argument(
        "--stop-via-request-cancel",
        action="store_true",
        required=False,
        default=False,
        help="Early stop use request cancellation instead of stop request")
    parser.add_argument('--tokenizer-dir',
                        type=str,
                        required=False,
                        default='',
                        help='Specify tokenizer directory')
    parser.add_argument('--tokenizer-type',
                        type=str,
                        default='auto',
                        required=False,
                        choices=['auto', 't5', 'llama'],
                        help='Specify tokenizer type')
    parser.add_argument('--request-id',
                        type=str,
                        default='',
                        required=False,
                        help='The request_id for the stop request')

    parser.add_argument('--prompt-embedding-table-path',
                        type=str,
                        default='',
                        required=False,
                        help='The prompt embedding table to use for ptuning')
    parser.add_argument("--lora-path",
                        type=str,
                        default='',
                        required=False,
                        help="LoRA weights")
    parser.add_argument("--lora-task-id",
                        type=int,
                        default=None,
                        required=False,
                        help="LoRA task id")
    parser.add_argument(
        "--exclude-input-in-output",
        action="store_true",
        required=False,
        default=False,
        help="Expect that output IDs do not contain input IDs",
    )

    parser.add_argument('--prompt-task-id',
                        type=int,
                        default=0,
                        required=False,
                        help='The prompt task id in the prompt embedding table')

    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])

    parser.add_argument(
        "--return-log-probs",
        action="store_true",
        required=False,
        default=False,
        help="Enable computation of log probs",
    )

    parser.add_argument(
        "--return-context-logits",
        action="store_true",
        required=False,
        default=False,
        help=
        "Return context logits, the engine must be built with gather_context_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--return-generation-logits",
        action="store_true",
        required=False,
        default=False,
        help=
        "Return generation logits, the engine must be built with gather_ generation_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--return-perf-metrics",
        action="store_true",
        required=False,
        default=False,
        help="Return per-request perf metrics",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        required=False,
        default=1,
        help="top k value",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        required=False,
        default=0.,
        help="top p value",
    )

    parser.add_argument('--requested-outputs',
                        nargs='+',
                        default=[],
                        help='The requested output tensors')

    parser.add_argument('--model-name',
                        type=str,
                        required=False,
                        default='tensorrt_llm',
                        help='Specify model name')
    parser.add_argument(
        "--prompt_table_extra_id",
        type=int,
        required=False,
        default=None,
        help=
        "When enable kv cache reuse, we need a unique id to determine whether the prompt tables are the same. The type of extra id is uint64, and its range is from 1 to the maximum value of uint64.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        '--lookahead_config',
        type=parse_list,
        help=
        'Lookahead parameters in format [window_size,ngram_size,verification_set_size]. Example: [7,7,7]',
        default=None,
        required=False)
    parser.add_argument(
        "--return-num-input-tokens",
        action="store_true",
        required=False,
        default=False,
        help="Return the number of input tokens",
    )
    parser.add_argument(
        "--return-num-output-tokens",
        action="store_true",
        required=False,
        default=False,
        help="Return the number of output tokens",
    )

    FLAGS = parser.parse_args()

    tokenizer = None
    draft_ids = None
    decoder_input_ids = None
    if FLAGS.input_tokens_csv != "":
        with open(FLAGS.input_tokens_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                input_ids = [[int(val) for val in row]]
                break

            curate_log_output(input_ids[0], "Input")

        if FLAGS.draft_tokens_csv != "":
            with open(FLAGS.draft_tokens_csv) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                for row in csv_reader:
                    draft_ids = [[int(val) for val in row]]
                    break

        end_id = FLAGS.end_id
        pad_id = FLAGS.pad_id

    else:
        print('=========')
        if (os.path.isdir(FLAGS.tokenizer_dir)
                and not os.path.exists(FLAGS.tokenizer_dir)):
            raise FileNotFoundError(
                "Input tokens are not provided and tokenizer directory does"
                f" not exist: {FLAGS.tokenizer_dir}", )

        tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  trust_remote_code=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        pad_id = tokenizer.encode(tokenizer.pad_token,
                                  add_special_tokens=False)[0]
        end_id = tokenizer.encode(tokenizer.eos_token,
                                  add_special_tokens=False)[0]
        print("Using pad_id: ", pad_id)
        print("Using end_id: ", end_id)

        input_ids = [tokenizer.encode(FLAGS.text)]
        curate_log_output(input_ids[0], "Input")

    end_id_data = np.array([[end_id]], dtype=np.int32)
    pad_id_data = np.array([[pad_id]], dtype=np.int32)

    #Get the prompt embedding table for the task id
    prompt_embedding_table_data = None
    prompt_vocab_size_data = None
    if (FLAGS.prompt_embedding_table_path != ""):
        prompt_table = np.load(FLAGS.prompt_embedding_table_path)
        prompt_table = prompt_table.astype(str_dtype_to_np(FLAGS.dtype))
        task_vocab_size = prompt_table.shape[1]

        # squeeze the first 2 dimensions
        prompt_embedding_table_data = prompt_table[FLAGS.prompt_task_id]
        prompt_embedding_table_data = np.expand_dims(
            prompt_table[FLAGS.prompt_task_id], axis=0)

        prompt_vocab_size = [[task_vocab_size]]
        prompt_vocab_size_data = np.array(prompt_vocab_size, dtype=np.int32)

    lora_weights_data = None
    lora_config_data = None
    if (FLAGS.lora_path != ""):
        lora_weights_data = np.load(
            os.path.join(FLAGS.lora_path, "model.lora_weights.npy"))
        try:
            lora_config_data = np.load(
                os.path.join(FLAGS.lora_path, "model.lora_config.npy"))
        except Exception:
            lora_config_data = np.load(
                os.path.join(FLAGS.lora_path, "model.lora_keys.npy"))
    lora_task_id_data = None
    if FLAGS.lora_task_id is not None and FLAGS.lora_task_id != 0:
        lora_task_id_data = np.array([[FLAGS.lora_task_id]], dtype=np.uint64)

    input_ids_data = np.array(input_ids, dtype=np.int32)
    input_lengths = [[len(ii)] for ii in input_ids]
    input_lengths_data = np.array(input_lengths, dtype=np.int32)
    request_output_len = [[FLAGS.request_output_len]]
    request_output_len_data = np.array(request_output_len, dtype=np.int32)

    num_return_sequences_data = None
    if FLAGS.num_return_sequences:
        num_return_sequences_data = np.array([[FLAGS.num_return_sequences]],
                                             dtype=np.int32)

    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.int32)
    top_k = [[FLAGS.top_k]]
    top_k_data = np.array(top_k, dtype=np.int32)
    top_p = [[FLAGS.top_p]]
    top_p_data = np.array(top_p, dtype=np.float32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    return_log_probs_data = None
    if FLAGS.return_log_probs:
        return_log_probs_data = np.array([[FLAGS.return_log_probs]], dtype=bool)

    return_context_logits_data = None
    if FLAGS.return_context_logits:
        return_context_logits_data = np.array([[FLAGS.return_context_logits]],
                                              dtype=bool)

    return_generation_logits_data = None
    if FLAGS.return_generation_logits:
        return_generation_logits_data = np.array(
            [[FLAGS.return_generation_logits]], dtype=bool)

    return_perf_metrics_data = None
    if FLAGS.return_perf_metrics:
        return_perf_metrics_data = np.array([[FLAGS.return_perf_metrics]],
                                            dtype=bool)

    repetition_penalty_data = None
    if FLAGS.repetition_penalty is not None:
        repetition_penalty = [[FLAGS.repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty, dtype=np.float32)
    presence_penalty_data = None
    if FLAGS.presence_penalty is not None:
        presence_penalty = [[FLAGS.presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)
    frequency_penalty_data = None
    if FLAGS.frequency_penalty is not None:
        frequency_penalty = [[FLAGS.frequency_penalty]]
        frequency_penalty_data = np.array(frequency_penalty, dtype=np.float32)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)

    draft_ids_data = None
    if draft_ids is not None:
        draft_ids_data = np.array(draft_ids, dtype=np.int32)

    decoder_input_ids_data = None
    if decoder_input_ids is not None:
        decoder_input_ids_data = np.array(decoder_input_ids, dtype=np.int32)

    exclude_input_in_output = None
    if FLAGS.exclude_input_in_output:
        exclude_input_in_output = np.array([[FLAGS.exclude_input_in_output]],
                                           dtype=bool)

    if not FLAGS.vocab_size and tokenizer:
        FLAGS.vocab_size = tokenizer.vocab_size
    prompt_table_extra_id_data = None
    if FLAGS.prompt_table_extra_id is not None:
        if not FLAGS.vocab_size:
            raise Exception(
                "To use the prompt table extra id, you need to specify the vocab size."
            )

        prompt_table_extra_id_data = np.zeros_like(input_ids_data,
                                                   dtype=np.uint64)
        for i in range(input_ids_data.shape[0]):
            prompt_table_extra_id_data[i] = np.where(
                input_ids_data[i] >= FLAGS.vocab_size,
                FLAGS.prompt_table_extra_id, 0)

    lookahead_config_data = None
    if FLAGS.lookahead_config is not None:
        lookahead_config_data = np.array(FLAGS.lookahead_config, dtype=np.int32)

    return_num_input_tokens_data = None
    if FLAGS.return_num_input_tokens:
        return_num_input_tokens_data = np.array(
            [[FLAGS.return_num_input_tokens]], dtype=bool)

    return_num_output_tokens_data = None
    if FLAGS.return_num_output_tokens:
        return_num_output_tokens_data = np.array(
            [[FLAGS.return_num_output_tokens]], dtype=bool)

    inputs = prepare_inputs(
        input_ids_data, input_lengths_data, request_output_len_data,
        beam_width_data, temperature_data, repetition_penalty_data,
        presence_penalty_data, frequency_penalty_data, streaming_data,
        end_id_data, pad_id_data, prompt_embedding_table_data,
        prompt_vocab_size_data, lora_task_id_data, lora_weights_data,
        lora_config_data, return_log_probs_data, top_k_data, top_p_data,
        draft_ids_data, return_context_logits_data,
        return_generation_logits_data, decoder_input_ids_data,
        prompt_table_extra_id_data, exclude_input_in_output,
        num_return_sequences_data, return_perf_metrics_data,
        lookahead_config_data, return_num_input_tokens_data,
        return_num_output_tokens_data)

    if FLAGS.requested_outputs:
        # Must have at least output_ids in requested outputs
        if "output_ids" not in FLAGS.requested_outputs:
            raise Exception(
                "requested outputs must at least have \"output_ids\"")
        outputs = prepare_outputs(FLAGS.requested_outputs)
    else:
        outputs = None

    stop_inputs = None
    if FLAGS.stop_after_ms > 0 and not FLAGS.stop_via_request_cancel:
        stop_inputs = prepare_stop_signals()

    request_id = FLAGS.request_id

    if FLAGS.output_tokens_csv != "":
        with open(FLAGS.output_tokens_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                expected_output_ids = [[int(val) for val in row]]
                break
    else:
        # expected_output_ids holds a list of lists, each list is a version of "expected" output ids
        # The expected output could vary on different GPUs
        expected_output_ids = []
        expected_output_ids.append(
            ([] if FLAGS.exclude_input_in_output else input_ids[0]) + [
                21221,
                290,
                373,
                257,
                2888,
                286,
                262,
                4141,
                2351,
                10006,
                13,
                679,
                373,
                7018,
                284,
                262,
            ])
        # Adding a second expected output ids for testing on A100 GPUs
        expected_output_ids.append(
            ([] if FLAGS.exclude_input_in_output else input_ids[0]) + [
                21221, 290, 257, 4255, 379, 262, 1957, 7072, 11, 4689, 347,
                2852, 2564, 494, 13, 679
            ])

    if FLAGS.num_return_sequences is None:
        num_generations = FLAGS.beam_width
    else:
        num_generations = FLAGS.num_return_sequences
        assert FLAGS.beam_width == 1

    if FLAGS.streaming:
        actual_output_ids = [
            [] if FLAGS.exclude_input_in_output else input_ids[0]
            for _ in range(num_generations)
        ]
    else:
        actual_output_ids = [[] for _ in range(num_generations)]

    # Expected result shapes: [num_sequences, ...]
    sequence_lengths = [None] * num_generations
    cum_log_probs = [None] * num_generations
    output_log_probs = [None] * num_generations
    context_logits = None
    generation_logits = [None] * num_generations
    returned_perf_metrics = {}
    input_token_count = [None] * num_generations
    output_token_count = [None] * num_generations

    def set_output(outputs: list, data, seq_idx=None):
        if FLAGS.beam_width > 1:
            # data = beams
            for seq_idx in range(FLAGS.beam_width):
                outputs[seq_idx] = data[seq_idx]
        else:
            assert seq_idx is not None
            outputs[seq_idx] = data

    user_data = UserData()
    with grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain,
    ) as triton_client:
        try:
            if FLAGS.streaming or FLAGS.num_return_sequences is not None:

                # Establish stream
                triton_client.start_stream(
                    callback=partial(callback, user_data),
                    stream_timeout=FLAGS.stream_timeout,
                )
                # Send request
                triton_client.async_stream_infer(
                    FLAGS.model_name,
                    inputs,
                    outputs=outputs,
                    request_id=request_id,
                )

                if FLAGS.stop_after_ms > 0:
                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    if not FLAGS.stop_via_request_cancel:
                        triton_client.async_stream_infer(
                            FLAGS.model_name,
                            stop_inputs,
                            request_id=request_id,
                            parameters={'Streaming': FLAGS.streaming})

                # Close the grpc stream
                cancel_requests = FLAGS.stop_after_ms > 0 and FLAGS.stop_via_request_cancel
                triton_client.stop_stream(cancel_requests=cancel_requests)

                # Parse the responses
                while True:
                    try:
                        result = user_data._completed_requests.get(block=False)
                    except Exception as e:
                        print(str(e))
                        break

                    if type(result) == InferenceServerException:
                        if result.status() == "StatusCode.CANCELLED":
                            print("Request is cancelled")
                        else:
                            print("Received an error from server:")
                            print(result)
                            raise result
                    else:
                        check_output_names(FLAGS.requested_outputs, result)
                        output_ids = result.as_numpy('output_ids')
                        if output_ids is not None:
                            seq_idx = result.as_numpy('sequence_index')
                            seq_idx = seq_idx[0][0] if seq_idx is not None else 0

                            if FLAGS.return_log_probs:
                                set_output(cum_log_probs,
                                           result.as_numpy('cum_log_probs')[0],
                                           seq_idx)
                                set_output(
                                    output_log_probs,
                                    result.as_numpy('output_log_probs')[0],
                                    seq_idx)
                            if FLAGS.return_context_logits:
                                context_logits = result.as_numpy(
                                    'context_logits')
                            if FLAGS.return_generation_logits:
                                set_output(
                                    generation_logits,
                                    result.as_numpy('generation_logits')[0],
                                    seq_idx)
                            if FLAGS.return_num_input_tokens:
                                input_token_count[seq_idx] = result.as_numpy(
                                    'num_input_tokens')
                            if FLAGS.return_num_output_tokens:
                                output_token_count[seq_idx] = result.as_numpy(
                                    'num_output_tokens')
                            if FLAGS.return_perf_metrics:
                                returned_perf_metrics[
                                    'kv_cache_alloc_new_blocks'] = result.as_numpy(
                                        'kv_cache_alloc_new_blocks')
                                returned_perf_metrics[
                                    'kv_cache_reused_blocks'] = result.as_numpy(
                                        'kv_cache_reused_blocks')
                                returned_perf_metrics[
                                    'kv_cache_alloc_total_blocks'] = result.as_numpy(
                                        'kv_cache_alloc_total_blocks')
                                returned_perf_metrics[
                                    'arrival_time_ns'] = result.as_numpy(
                                        'arrival_time_ns')
                                returned_perf_metrics[
                                    'first_scheduled_time_ns'] = result.as_numpy(
                                        'first_scheduled_time_ns')
                                returned_perf_metrics[
                                    'first_token_time_ns'] = result.as_numpy(
                                        'first_token_time_ns')
                                returned_perf_metrics[
                                    'last_token_time_ns'] = result.as_numpy(
                                        'last_token_time_ns')
                                returned_perf_metrics[
                                    'acceptance_rate'] = result.as_numpy(
                                        'acceptance_rate')
                                returned_perf_metrics[
                                    'total_accepted_draft_tokens'] = result.as_numpy(
                                        'total_accepted_draft_tokens')
                                returned_perf_metrics[
                                    'total_draft_tokens'] = result.as_numpy(
                                        'total_draft_tokens')

                            sequence_lengths[seq_idx] = result.as_numpy(
                                'sequence_length')[0][0]
                            if FLAGS.streaming:
                                # Only one beam is supported
                                if (sequence_lengths[seq_idx] is None
                                        or sequence_lengths[seq_idx] > 0):
                                    tokens = list(output_ids[0][0])
                                    actual_output_ids[seq_idx] = (
                                        actual_output_ids[seq_idx] + tokens)
                            else:
                                # num_return_sequences > 1 under offline mode.
                                tokens = list(output_ids[0][0])
                                actual_output_ids[seq_idx].extend(tokens)
                        else:
                            print("Got cancellation response from server")
            else:
                # Send request
                infer_future = triton_client.async_infer(
                    FLAGS.model_name,
                    inputs,
                    outputs=outputs,
                    request_id=request_id,
                    callback=partial(callback, user_data),
                    parameters={'Streaming': FLAGS.streaming})

                expected_responses = 1
                if FLAGS.num_return_sequences is not None:
                    expected_responses = FLAGS.num_return_sequences

                if FLAGS.stop_after_ms > 0:

                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    if FLAGS.stop_via_request_cancel:
                        infer_future.cancel()
                    else:
                        triton_client.async_infer(
                            FLAGS.model_name,
                            stop_inputs,
                            request_id=request_id,
                            callback=partial(callback, user_data),
                            parameters={'Streaming': FLAGS.streaming})
                        expected_responses += 1

                processed_count = 0
                while processed_count < expected_responses:
                    try:
                        result = user_data._completed_requests.get()
                        print("Got completed request", flush=True)
                    except Exception:
                        break

                    if type(result) == InferenceServerException:
                        if result.status() == "StatusCode.CANCELLED":
                            print("Request is cancelled")
                        else:
                            print("Received an error from server:")
                            print(result)
                            raise result
                    else:
                        check_output_names(FLAGS.requested_outputs, result)
                        output_ids = result.as_numpy('output_ids')
                        seq_idx = result.as_numpy('sequence_index')
                        seq_idx = seq_idx[0][0] if seq_idx is not None else 0

                        if FLAGS.return_log_probs:
                            set_output(cum_log_probs,
                                       result.as_numpy('cum_log_probs')[0],
                                       seq_idx)
                            set_output(output_log_probs,
                                       result.as_numpy('output_log_probs')[0],
                                       seq_idx)
                        if FLAGS.return_context_logits:
                            context_logits = result.as_numpy('context_logits')
                        if FLAGS.return_generation_logits:
                            set_output(generation_logits,
                                       result.as_numpy('generation_logits')[0],
                                       seq_idx)
                        if FLAGS.return_num_input_tokens:
                            input_token_count[seq_idx] = result.as_numpy(
                                'num_input_tokens')
                        if FLAGS.return_num_output_tokens:
                            output_token_count[seq_idx] = result.as_numpy(
                                'num_output_tokens')
                        if FLAGS.return_perf_metrics:
                            returned_perf_metrics[
                                'kv_cache_alloc_new_blocks'] = result.as_numpy(
                                    'kv_cache_alloc_new_blocks')
                            returned_perf_metrics[
                                'kv_cache_reused_blocks'] = result.as_numpy(
                                    'kv_cache_reused_blocks')
                            returned_perf_metrics[
                                'kv_cache_alloc_total_blocks'] = result.as_numpy(
                                    'kv_cache_alloc_total_blocks')
                            returned_perf_metrics[
                                'arrival_time_ns'] = result.as_numpy(
                                    'arrival_time_ns')
                            returned_perf_metrics[
                                'first_scheduled_time_ns'] = result.as_numpy(
                                    'first_scheduled_time_ns')
                            returned_perf_metrics[
                                'first_token_time_ns'] = result.as_numpy(
                                    'first_token_time_ns')
                            returned_perf_metrics[
                                'last_token_time_ns'] = result.as_numpy(
                                    'last_token_time_ns')
                            returned_perf_metrics[
                                'acceptance_rate'] = result.as_numpy(
                                    'acceptance_rate')
                            returned_perf_metrics[
                                'total_accepted_draft_tokens'] = result.as_numpy(
                                    'total_accepted_draft_tokens')
                            returned_perf_metrics[
                                'total_draft_tokens'] = result.as_numpy(
                                    'total_draft_tokens')
                        if output_ids is not None:
                            print(result.as_numpy('sequence_length'))
                            if FLAGS.beam_width > 1:
                                set_output(
                                    sequence_lengths,
                                    result.as_numpy('sequence_length')[0])
                                for beam_idx in range(FLAGS.beam_width):
                                    beam_output_ids = output_ids[0][beam_idx]
                                    tokens = list(beam_output_ids)
                                    actual_output_ids[beam_idx].extend(tokens)
                            else:
                                assert seq_idx is not None
                                sequence_lengths[seq_idx] = result.as_numpy(
                                    'sequence_length')[0][0]
                                tokens = list(output_ids[0][0])
                                actual_output_ids[seq_idx].extend(tokens)
                        else:
                            print("Got cancellation response from server")

                    processed_count = processed_count + 1
        except Exception as e:
            err = "Encountered error: " + str(e)
            print(err)
            sys.exit(err)

        passed = True

        # Keep the output ids of seq_idx=0 for testing randomness
        # across generated sequences.
        output_ids_wo_prompt_0 = None

        for seq_idx in range(num_generations):
            seq_len = (sequence_lengths[seq_idx]
                       if not FLAGS.streaming and len(sequence_lengths) > 0 else
                       len(actual_output_ids[seq_idx]))
            # These should be equal when input IDs are excluded from output
            output_ids_w_prompt = actual_output_ids[seq_idx][:seq_len]
            output_ids_wo_prompt = (
                output_ids_w_prompt if FLAGS.exclude_input_in_output else
                output_ids_w_prompt[input_ids_data.shape[1]:])
            if seq_idx == 0:
                output_ids_wo_prompt_0 = output_ids_wo_prompt
            if tokenizer is not None:
                output_text = tokenizer.decode(output_ids_wo_prompt)
                print(f'Input: {FLAGS.text}')
                print(f'Output beam {seq_idx}: {output_text}')
            if FLAGS.return_num_input_tokens:
                print(f'Input token count: {input_token_count[seq_idx]}')
            if FLAGS.return_num_output_tokens:
                print(f'Output token count: {output_token_count[seq_idx]}')
            # If cancelled, the number of output tokens should be less than request output length.
            if FLAGS.stop_after_ms > 0 and len(
                    output_ids_wo_prompt) >= FLAGS.request_output_len:
                raise AssertionError("expect less than " +
                                     str(FLAGS.request_output_len) +
                                     " output tokens, got " +
                                     str(len(output_ids_wo_prompt)))
            curate_log_output(output_ids_w_prompt, "Output")

            if FLAGS.check_output and seq_idx == 0:
                passed = False
                if FLAGS.correctness_threshold == 1.0:
                    passed = (output_ids_w_prompt in expected_output_ids)
                else:
                    # Compare the output tokens one by one
                    num_same_output_id = [0] * len(expected_output_ids)
                    for i, expect_output in enumerate(expected_output_ids):
                        for output, expected in zip(output_ids_w_prompt,
                                                    expect_output):
                            if output == expected:
                                num_same_output_id[i] += 1

                    # Calculate the match rate
                    match_rate = max(num_same_output_id) / len(
                        output_ids_w_prompt)
                    print(f"Output token matching rate: {match_rate}")
                    passed = (match_rate > FLAGS.correctness_threshold)
                    print("expected_output_ids = ", expected_output_ids)
                print("\n=====")
                print("PASS!" if passed else "FAIL!")
                print("=====")

            non_deterministic_sampling = FLAGS.beam_width == 1 and (
                top_k[0][0] > 1 or top_p[0][0] > 0)
            if FLAGS.check_output and non_deterministic_sampling and seq_idx > 0:
                # Skip the correctness check under non-deterministic sampling.
                # Generated sequences should not be identical.
                passed = output_ids_w_prompt[seq_idx] not in expected_output_ids
                if not passed:
                    print(f"Output tokens of sequence {seq_idx} is identical "
                          f"to the expected sequence.")

        if FLAGS.return_log_probs:
            print('cum_log_probs:', expand_and_vstack(cum_log_probs))
            print('output_log_probs:', expand_and_vstack(output_log_probs))

        if FLAGS.return_context_logits:
            print(f"context_logits.shape: {context_logits.shape}")
            print(f"context_logits: {context_logits}")

        if FLAGS.return_generation_logits:
            generation_logits = expand_and_vstack(generation_logits)
            print(f"generation_logits.shape: {generation_logits.shape}")
            print(f"generation_logits: {generation_logits}")

        if FLAGS.return_num_input_tokens:
            print(f'Input token count: {input_token_count[0]}')
        if FLAGS.return_num_output_tokens:
            print(f'Output token count: {output_token_count[0]}')

        if FLAGS.return_perf_metrics:
            for key, value in returned_perf_metrics.items():
                print(f"{key}: {value[0][0]}")

        sys.exit(not passed)
