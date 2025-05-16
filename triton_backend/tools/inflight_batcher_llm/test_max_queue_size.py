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
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

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


def prepare_tensor(name, input, protocol):
    client = httpclient if protocol == 'http' else grpcclient
    t = client.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_outputs(output_names, protocol):
    client = httpclient if protocol == 'http' else grpcclient
    outputs = []
    for output_name in output_names:
        outputs.append(client.InferRequestedOutput(output_name))
    return outputs


def prepare_inputs(input_ids_data, input_lengths_data, request_output_len_data,
                   beam_width_data, temperature_data, repetition_penalty_data,
                   presence_penalty_data, frequency_penalty_data,
                   streaming_data, end_id, pad_id, prompt_embedding_table_data,
                   prompt_vocab_size_data, lora_task_id_data, lora_weights_data,
                   lora_config_data, return_log_probs_data, top_k_data,
                   top_p_data, draft_ids_data, return_context_logits_data,
                   return_generation_logits_data, decoder_input_ids_data,
                   protocol):
    inputs = [
        prepare_tensor("input_ids", input_ids_data, protocol),
        prepare_tensor("input_lengths", input_lengths_data, protocol),
        prepare_tensor("request_output_len", request_output_len_data, protocol),
        prepare_tensor("beam_width", beam_width_data, protocol),
        prepare_tensor("temperature", temperature_data, protocol),
        prepare_tensor("streaming", streaming_data, protocol),
        prepare_tensor("end_id", end_id, protocol),
        prepare_tensor("pad_id", pad_id, protocol),
        prepare_tensor("return_log_probs", return_log_probs_data, protocol),
        prepare_tensor("runtime_top_k", top_k_data, protocol),
        prepare_tensor("runtime_top_p", top_p_data, protocol),
    ]
    if prompt_embedding_table_data is not None:
        inputs += [
            prepare_tensor("prompt_embedding_table",
                           prompt_embedding_table_data, protocol),
            prepare_tensor("prompt_vocab_size", prompt_vocab_size_data,
                           protocol)
        ]
    if lora_task_id_data is not None:
        inputs += [prepare_tensor("lora_task_id", lora_task_id_data, protocol)]
    if lora_weights_data is not None:
        inputs += [
            prepare_tensor("lora_weights", lora_weights_data, protocol),
            prepare_tensor("lora_config", lora_config_data, protocol),
        ]
    if repetition_penalty_data is not None:
        inputs += [
            prepare_tensor("repetition_penalty", repetition_penalty_data,
                           protocol),
        ]
    if presence_penalty_data is not None:
        inputs += [
            prepare_tensor("presence_penalty", presence_penalty_data, protocol),
        ]
    if frequency_penalty_data is not None:
        inputs += [
            prepare_tensor("frequency_penalty", frequency_penalty_data,
                           protocol),
        ]
    if draft_ids_data is not None:
        inputs += [
            prepare_tensor("draft_input_ids", draft_ids_data, protocol),
        ]
    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits", return_context_logits_data,
                           protocol),
        ]
    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits",
                           return_generation_logits_data, protocol),
        ]
    if decoder_input_ids_data is not None:
        inputs += [
            prepare_tensor("decoder_input_ids", decoder_input_ids_data,
                           protocol),
        ]
    return inputs


def test_http_client(args: argparse.Namespace, inputs, outputs, request_id):
    with httpclient.InferenceServerClient(
            url="localhost:8000",
            verbose=False,
            ssl=False,
            concurrency=args.num_requests,
    ) as triton_client:
        try:
            futures = []
            # Send requests
            for i in range(args.num_requests):
                infer_future = triton_client.async_infer(args.model_name,
                                                         inputs,
                                                         outputs=outputs,
                                                         request_id=request_id)
                futures.append(infer_future)
        except Exception as e:
            err = "Encountered error: " + str(e)
            print(err)
            sys.exit(err)

        passed = False

        for i in range(len(futures)):
            try:
                futures[i].get_result()
            except Exception as e:
                if "Maximum queue size of" in f"{e}":
                    passed = True

        return passed


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would provide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def test_grpc_client(args: argparse.Namespace, inputs, outputs, request_id):
    user_data = UserData()
    with grpcclient.InferenceServerClient(
            url=args.url,
            verbose=False,
            ssl=False,
    ) as triton_client:
        try:
            # Establish stream
            triton_client.start_stream(callback=partial(callback, user_data), )
            # Send requests
            for i in range(args.num_requests):
                triton_client.async_stream_infer(
                    args.model_name,
                    inputs,
                    outputs=outputs,
                    request_id=request_id,
                )
            triton_client.stop_stream(cancel_requests=False)
        except Exception as e:
            err = "Encountered error: " + str(e)
            print(err)
            sys.exit(err)

        passed = False

        while True:
            try:
                result = user_data._completed_requests.get(block=False)
            except Exception:
                break

            if type(result) == InferenceServerException:
                if "Exceeds maximum queue size" in result.message():
                    passed = True

        return passed


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
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        '-p',
        '--protocol',
        type=str,
        required=False,
        default='http',
        choices=['http', 'grpc'],
        help=
        'Protocol to use when communicating with the inference server. Default is HTTP.'
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
        '--num-requests',
        type=int,
        required=True,
        help='Number of requests to send to try and fill up the queue')

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
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.int32)
    top_k = [[FLAGS.top_k]]
    top_k_data = np.array(top_k, dtype=np.int32)
    top_p = [[FLAGS.top_p]]
    top_p_data = np.array(top_p, dtype=np.float32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    return_log_probs = [[FLAGS.return_log_probs]]
    return_log_probs_data = np.array(return_log_probs, dtype=bool)

    return_context_logits_data = None
    if FLAGS.return_context_logits:
        return_context_logits_data = np.array([[FLAGS.return_context_logits]],
                                              dtype=bool)

    return_generation_logits_data = None
    if FLAGS.return_generation_logits:
        return_generation_logits_data = np.array(
            [[FLAGS.return_generation_logits]], dtype=bool)

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
    streaming = [[FLAGS.protocol == 'grpc']]
    streaming_data = np.array(streaming, dtype=bool)

    draft_ids_data = None
    if draft_ids is not None:
        draft_ids_data = np.array(draft_ids, dtype=np.int32)

    decoder_input_ids_data = None
    if decoder_input_ids is not None:
        decoder_input_ids_data = np.array(decoder_input_ids, dtype=np.int32)

    inputs = prepare_inputs(
        input_ids_data, input_lengths_data, request_output_len_data,
        beam_width_data, temperature_data, repetition_penalty_data,
        presence_penalty_data, frequency_penalty_data, streaming_data,
        end_id_data, pad_id_data, prompt_embedding_table_data,
        prompt_vocab_size_data, lora_task_id_data, lora_weights_data,
        lora_config_data, return_log_probs_data, top_k_data, top_p_data,
        draft_ids_data, return_context_logits_data,
        return_generation_logits_data, decoder_input_ids_data, FLAGS.protocol)

    if FLAGS.requested_outputs:
        # Must have at least output_ids in requested outputs
        if "output_ids" not in FLAGS.requested_outputs:
            raise Exception(
                "requested outputs must at least have \"output_ids\"")
        outputs = prepare_outputs(FLAGS.requested_outputs, FLAGS.protocol)
    else:
        outputs = None

    request_id = FLAGS.request_id

    if FLAGS.protocol == 'http':
        res = test_http_client(FLAGS, inputs, outputs, request_id)
    else:
        res = test_grpc_client(FLAGS, inputs, outputs, request_id)

    if not res:
        print(
            "Expected requests to be rejected due to a full queue, but that didn't happen!"
        )
    sys.exit(not res)
