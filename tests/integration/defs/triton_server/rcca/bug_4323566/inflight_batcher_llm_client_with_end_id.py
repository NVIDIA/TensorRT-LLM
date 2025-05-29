#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import queue
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

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


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def prepare_inputs(input_ids_data, input_lengths_data, request_output_len_data,
                   beam_width_data, temperature_data, streaming_data, end_id):

    inputs = [
        grpcclient.InferInput('input_ids', [1, 12], "INT32"),
        grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
        grpcclient.InferInput('request_output_len', [1, 1], "UINT32"),
        grpcclient.InferInput('beam_width', [1, 1], "UINT32"),
        grpcclient.InferInput('temperature', [1, 1], "FP32"),
        grpcclient.InferInput('streaming', [1, 1], "BOOL"),
        grpcclient.InferInput('end_id', [1, 1], "UINT32"),
    ]

    inputs[0].set_data_from_numpy(input_ids_data)
    inputs[1].set_data_from_numpy(input_lengths_data)
    inputs[2].set_data_from_numpy(request_output_len_data)
    inputs[3].set_data_from_numpy(beam_width_data)
    inputs[4].set_data_from_numpy(temperature_data)
    inputs[5].set_data_from_numpy(streaming_data)
    inputs[6].set_data_from_numpy(end_id)

    return inputs


def prepare_stop_signals():

    inputs = [
        grpcclient.InferInput('input_ids', [1, 1], "INT32"),
        grpcclient.InferInput('input_lengths', [1, 1], "INT32"),
        grpcclient.InferInput('request_output_len', [1, 1], "UINT32"),
        grpcclient.InferInput('stop', [1, 1], "BOOL"),
    ]

    inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
    inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
    inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.uint32))
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
        "--request-output-len",
        type=int,
        required=False,
        default=16,
        help="temperature value",
    )
    parser.add_argument(
        '--stop-after-ms',
        type=int,
        required=False,
        default=0,
        help='Early stop the generation after a few milliseconds')

    FLAGS = parser.parse_args()

    print('=========')
    input_ids = [[
        28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257
    ]]
    input_ids_data = np.array(input_ids, dtype=np.int32)
    input_lengths = [[len(ii)] for ii in input_ids]
    input_lengths_data = np.array(input_lengths, dtype=np.int32)
    request_output_len = [[FLAGS.request_output_len]]
    request_output_len_data = np.array(request_output_len, dtype=np.uint32)
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.uint32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)
    end_id = np.array([[6303]], dtype=np.uint32)

    inputs = prepare_inputs(input_ids_data, input_lengths_data,
                            request_output_len_data, beam_width_data,
                            temperature_data, streaming_data, end_id)

    if FLAGS.stop_after_ms > 0:
        stop_inputs = prepare_stop_signals()
    else:
        stop_inputs = None

    request_id = "12345"
    import random
    request_id = str(random.randint(3, 9000))

    expected_output_ids = [
        input_ids[0] + [
            21221, 290, 257, 4255, 379, 262, 1957, 7072, 11, 4689, 347, 2852,
            2564, 494, 13, 679
        ]
    ]
    if FLAGS.streaming:
        actual_output_ids = [input_ids[0]]
    else:
        actual_output_ids = []

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

            if FLAGS.streaming:

                # Establish stream
                triton_client.start_stream(
                    callback=partial(callback, user_data),
                    stream_timeout=FLAGS.stream_timeout,
                )
                # Send request
                triton_client.async_stream_infer(
                    'tensorrt_llm',
                    inputs,
                    request_id=request_id,
                )

                if stop_inputs is not None:

                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    triton_client.async_stream_infer(
                        'tensorrt_llm',
                        stop_inputs,
                        request_id=request_id,
                        parameters={'Streaming': FLAGS.streaming})

                #Wait for server to close the stream
                triton_client.stop_stream()

                # Parse the responses
                while True:
                    try:
                        result = user_data._completed_requests.get(block=False)
                    except Exception:
                        break

                    if type(result) == InferenceServerException:
                        print("Received an error from server:")
                        print(result)
                    else:
                        output_ids = result.as_numpy('output_ids')

                        if output_ids is not None:
                            if (FLAGS.streaming):
                                # Only one beam is supported
                                tokens = list(output_ids[0][0])
                                actual_output_ids[
                                    0] = actual_output_ids[0] + tokens
                            else:
                                for beam_output_ids in output_ids[0]:
                                    tokens = list(beam_output_ids)
                                    actual_output_ids.append(tokens)
                        else:
                            print("Got cancellation response from server")
            else:
                # Send request
                triton_client.async_infer(
                    'tensorrt_llm',
                    inputs,
                    request_id=request_id,
                    callback=partial(callback, user_data),
                    parameters={'Streaming': FLAGS.streaming})

                if stop_inputs is not None:

                    time.sleep(FLAGS.stop_after_ms / 1000.0)

                    triton_client.async_infer(
                        'tensorrt_llm',
                        stop_inputs,
                        request_id=request_id,
                        callback=partial(callback, user_data),
                        parameters={'Streaming': FLAGS.streaming})

                processed_count = 0
                expected_responses = 1 + (1 if stop_inputs is not None else 0)
                while processed_count < expected_responses:
                    try:
                        result = user_data._completed_requests.get()
                        print("Got completed request", flush=True)
                    except Exception:
                        break

                    if type(result) == InferenceServerException:
                        print("Received an error from server:")
                        print(result)
                    else:
                        output_ids = result.as_numpy('output_ids')
                        if output_ids is not None:
                            for beam_output_ids in output_ids[0]:
                                tokens = list(beam_output_ids)
                                actual_output_ids.append(tokens)
                        else:
                            print("Got response for cancellation request")

                    processed_count = processed_count + 1
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()

        passed = True

        print("output_ids = ", actual_output_ids)
        if (FLAGS.check_output):
            passed = (actual_output_ids == expected_output_ids)
            print("expected_output_ids = ", expected_output_ids)
            print("\n=====")
            print("PASS!" if passed else "FAIL!")
            print("=====")

        sys.exit(not passed)
