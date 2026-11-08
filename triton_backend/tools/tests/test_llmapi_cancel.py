#!/usr/bin/env python
# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
from functools import partial

import numpy as np
from tritonclient import grpc as grpcclient
from tritonclient.utils import InferenceServerException

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from llmapi_client import (UserData, _prepare_inputs, callback,
                           prepare_stop_signals)

if __name__ == "__main__":
    input_data = np.array([
        "The current time is",
    ], dtype=object)
    output_len = 100
    inputs = _prepare_inputs(input_data, output_len)

    stop_inputs = prepare_stop_signals()
    request_id = 1
    user_data = UserData()
    with grpcclient.InferenceServerClient(
            url="localhost:8001",
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None,
    ) as triton_client:

        # Send stop request for non-existing request
        triton_client.async_infer(
            "tensorrt_llm",
            stop_inputs,
            request_id=str(request_id),  # Request does not exist yet
            callback=partial(callback, user_data),
            parameters={'Streaming': False})

        result = user_data._completed_requests.get()
        assert isinstance(result, InferenceServerException)
        assert result.status() == "StatusCode.CANCELLED"

        # Send actual request
        infer_response = triton_client.async_infer(
            "tensorrt_llm",
            inputs,
            request_id=str(request_id),
            callback=partial(callback, user_data),
            parameters={'Streaming': False})

        result = user_data._completed_requests.get()
        print(
            f'Output text: {result.as_numpy("text_output")[0].decode("utf-8")}')

        # Cancel request after it is completed
        infer_response.cancel()

        # Send stop request for completed request
        triton_client.async_infer("tensorrt_llm",
                                  stop_inputs,
                                  request_id=str(request_id),
                                  callback=partial(callback, user_data),
                                  parameters={'Streaming': False})

        cancel_result = user_data._completed_requests.get()
        assert isinstance(cancel_result, InferenceServerException)
        assert cancel_result.status() == "StatusCode.CANCELLED"

        # Send a second request to check if server is still healthy
        infer_response_2 = triton_client.async_infer(
            "tensorrt_llm",
            inputs,
            request_id=str(request_id + 1),
            callback=partial(callback, user_data),
            parameters={'Streaming': False})

        # Get result of second request
        result_2 = user_data._completed_requests.get()
        print('Got completed request')

        print(
            f'Output text: {result_2.as_numpy("text_output")[0].decode("utf-8")}'
        )

        # Check that both results match
        assert np.array_equal(result.as_numpy("text_output"),
                              result_2.as_numpy("text_output"))
