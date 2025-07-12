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
import queue
import sys
import time
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape,
                              np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def _prepare_inputs(prompt, output_len):
    inputs = [
        prepare_tensor("text_input", prompt),
        prepare_tensor("sampling_param_max_tokens",
                       np.array([output_len], dtype=np.int32)),
    ]
    return inputs


def prepare_stop_signals():

    inputs = [
        grpcclient.InferInput('text_input', [1], "BYTES"),
        grpcclient.InferInput('stop', [1], "BOOL"),
    ]

    inputs[0].set_data_from_numpy(np.empty([1], dtype=np.bytes_))
    inputs[1].set_data_from_numpy(np.array([True], dtype='bool'))

    return inputs


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


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
    parser.add_argument('--text',
                        type=str,
                        required=False,
                        default='Born in north-east France, Soyer trained as a',
                        help='Input text')

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

    parser.add_argument('--request-id',
                        type=str,
                        default='1',
                        required=False,
                        help='The request_id for the request')
    parser.add_argument(
        "--return-perf-metrics",
        action="store_true",
        required=False,
        default=False,
        help="Return per-request perf metrics",
    )
    parser.add_argument('--model-name',
                        type=str,
                        required=False,
                        default='tensorrt_llm',
                        help='Specify model name')

    FLAGS = parser.parse_args()

    input_data = np.array([FLAGS.text], dtype=object)

    output_len = FLAGS.request_output_len

    inputs = _prepare_inputs(input_data, output_len)

    stop_inputs = None
    if FLAGS.stop_after_ms > 0 and not FLAGS.stop_via_request_cancel:
        stop_inputs = prepare_stop_signals()

    request_id = FLAGS.request_id
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
            # Send request
            infer_future = triton_client.async_infer(
                FLAGS.model_name,
                inputs,
                outputs=None,
                request_id=request_id,
                callback=partial(callback, user_data),
                parameters={'Streaming': FLAGS.streaming})

            expected_responses = 1

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
                    print(
                        f'Output text: {result.as_numpy("text_output")[0].decode("utf-8")}'
                    )

                processed_count += 1

        except Exception as e:
            err = "Encountered error: " + str(e)
            print(err)
            sys.exit(err)

        sys.exit(0)
