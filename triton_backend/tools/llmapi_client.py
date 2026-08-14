#!/usr/bin/env python
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def _prepare_inputs(prompt,
                    output_len,
                    lora_id=None,
                    lora_name=None,
                    lora_path=None,
                    lora_ckpt_source=None):
    inputs = [
        prepare_tensor("text_input", prompt),
        prepare_tensor("sampling_param_max_tokens",
                       np.array([output_len], dtype=np.int32)),
    ]
    if lora_id is not None:
        inputs.append(
            prepare_tensor("lora_id", np.array([lora_id], dtype=np.uint64)))
    if lora_name is not None:
        inputs.append(
            prepare_tensor("lora_name",
                           np.array([lora_name.encode("utf-8")], dtype=object)))
    if lora_path is not None:
        inputs.append(
            prepare_tensor("lora_path",
                           np.array([lora_path.encode("utf-8")], dtype=object)))
    if lora_ckpt_source is not None:
        inputs.append(
            prepare_tensor(
                "lora_ckpt_source",
                np.array([lora_ckpt_source.encode("utf-8")], dtype=object)))
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
    parser.add_argument(
        '--lora-id',
        type=int,
        required=False,
        default=None,
        help='LoRA adapter integer id (matches `lora_id` input on the model).',
    )
    parser.add_argument(
        '--lora-name',
        type=str,
        required=False,
        default=None,
        help='LoRA adapter name (matches `lora_name` input on the model).',
    )
    parser.add_argument(
        '--lora-path',
        type=str,
        required=False,
        default=None,
        help=
        'Filesystem path to the LoRA adapter checkpoint readable by the Triton server.',
    )
    parser.add_argument(
        '--lora-ckpt-source',
        type=str,
        required=False,
        default=None,
        choices=("hf", "nemo"),
        help='LoRA checkpoint format. Defaults to "hf" on the server side.',
    )

    FLAGS = parser.parse_args()

    # The llmapi triton backend requires lora_id, lora_name, and lora_path
    # to be sent together (lora_ckpt_source is optional with default "hf").
    # Fail fast on partial input instead of letting the server return a
    # cryptic error response.
    lora_triplet = (FLAGS.lora_id, FLAGS.lora_name, FLAGS.lora_path)
    if any(v is not None
           for v in lora_triplet) and not all(v is not None
                                              for v in lora_triplet):
        parser.error(
            "--lora-id, --lora-name, and --lora-path must be provided together."
        )

    input_data = np.array([FLAGS.text], dtype=object)

    output_len = FLAGS.request_output_len

    inputs = _prepare_inputs(input_data,
                             output_len,
                             lora_id=FLAGS.lora_id,
                             lora_name=FLAGS.lora_name,
                             lora_path=FLAGS.lora_path,
                             lora_ckpt_source=FLAGS.lora_ckpt_source)

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
            # Triton rejects ModelInfer RPC (used by async_infer) on
            # decoupled models. Streaming requests must use the
            # bidirectional stream RPC. Cancellation (via either stop
            # tensor or request-cancel) also needs the stream path when
            # the server is decoupled.
            use_stream_rpc = bool(FLAGS.streaming) or FLAGS.stop_after_ms > 0
            infer_future = None
            if use_stream_rpc:
                triton_client.start_stream(
                    callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    FLAGS.model_name,
                    inputs,
                    request_id=request_id,
                    parameters={'Streaming': FLAGS.streaming})
            else:
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
                    if use_stream_rpc:
                        # cancel_requests=True closes the bidi stream
                        # AND cancels in-flight infers, which the
                        # cancellation_loop in model.py picks up via
                        # request.is_cancelled() and reports back as
                        # StatusCode.CANCELLED. Without the flag,
                        # stop_stream waits for completion instead.
                        triton_client.stop_stream(cancel_requests=True)
                    else:
                        infer_future.cancel()
                else:
                    if use_stream_rpc:
                        triton_client.async_stream_infer(
                            FLAGS.model_name,
                            stop_inputs,
                            request_id=request_id,
                            parameters={'Streaming': FLAGS.streaming})
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
                    # async_infer surfaces cancellation as a real gRPC
                    # status `StatusCode.CANCELLED`, but the bidi stream
                    # RPC delivers cancellation as an
                    # InferenceServerException whose status is None and
                    # whose message is the one model.py's
                    # cancellation_loop/handle_stop_request set on the
                    # TritonError (`Request cancelled by client`).
                    status = result.status()
                    msg = str(result).lower()
                    is_cancelled = (status == "StatusCode.CANCELLED"
                                    or "cancelled by client" in msg
                                    or "request cancelled" in msg)
                    if is_cancelled:
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
