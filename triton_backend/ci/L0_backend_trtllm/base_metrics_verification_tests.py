#!/usr/bin/python
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
import json
import os
import sys
from collections import defaultdict

import numpy as np
import requests

BACKEND_ROOT = os.environ.get('BACKEND_ROOT',
                              "/opt/tritonserver/tensorrtllm_backend")
sys.path.append(os.path.join(BACKEND_ROOT, "tools/utils"))
import re
import unittest

import utils

# This unit test was generated because the Triton team needed a
# static test in which an equal number of inferences were distributed
# across the 3 models orchestrated by the ensemble. This is so we could
# compare inference request counts and latencies in an equal environment.
# Many of the tests provided by the TRT team unevenly distribute requests
# so when we poll metrics the tensorrt_llm model, for example, will have
# performed 72 inferences whereas the pre/post models will have only
# performed 49. Further, because of this unequal distribution of requests
# we cannot check whether the latency across the 3 models is <= to the
# latency of the ensemble.

# Consider removing this unit test when the TRT tests have stabilized.


class TRTLLMBaseMetricsTest(unittest.TestCase):

    def setUp(self):
        self.expected_input_token_len = []
        self.expected_output_token_len = []

    def _get_metrics(self):
        metrics_url = "http://localhost:8002/metrics"
        r = requests.get(metrics_url)
        r.raise_for_status()
        return r.text

    def _run_infer(self, client, prompts, output_lens, beam_width_value=1):
        model_name = "ensemble"
        async_requests = []
        for i, prompt in enumerate(prompts):
            input0 = [[prompt]]
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
            bad_words_list = np.array([[""]], dtype=object)
            stop_words_list = np.array([[""]], dtype=object)

            inputs = [
                utils.prepare_tensor("text_input", input0_data, "http"),
                utils.prepare_tensor("max_tokens", output0_len, "http"),
                utils.prepare_tensor("bad_words", bad_words_list, "http"),
                utils.prepare_tensor("stop_words", stop_words_list, "http"),
            ]
            if beam_width_value > 1:
                beam_width = np.ones_like(input0).astype(
                    np.int32) * beam_width_value
                inputs.append(
                    utils.prepare_tensor("beam_width", beam_width, "http"))
            self.expected_output_token_len.append(output_lens[i] *
                                                  beam_width_value)
            # Request minimal outputs
            outputs = utils.prepare_outputs("http")
            async_requests.append(
                client.async_infer(model_name,
                                   inputs,
                                   outputs=outputs,
                                   request_id=str(i)))

        try:
            utils.get_http_results(async_requests)
        except Exception as e:
            print("Failed receiving responses: " + str(e))
            sys.exit(1)

    def _all_equal(self, iterable):
        return all(item == iterable[0] for item in iterable)

    def _calculate_bucket_counts(self, token_lengths, le_values):
        """
        Calculate histogram bucket counts based on le_values boundaries.
        Each bucket counts all values less than or equal to its boundary.

        Args:
            token_lengths (list): List of token lengths
            le_values (list): List of bucket boundaries (strings)

        Returns:
            list: Cumulative count of values less than or equal to each boundary
        """
        # Convert le_values to float (except "+Inf")
        boundaries = [
            float(x) if x != "+Inf" else float('inf') for x in le_values
        ]

        # Initialize bucket counts
        bucket_counts = [0] * len(boundaries)

        # Count tokens for each bucket (cumulative)
        for length in token_lengths:
            for i, boundary in enumerate(boundaries):
                if float(length) <= boundary:
                    # Increment this bucket and all higher buckets
                    for j in range(i, len(boundaries)):
                        bucket_counts[j] += 1
                    break

        return bucket_counts

    def _find_metric_values(self, filename, le_values):
        """
        Find metric values in file for given le_values.

        Args:
            filename (str): Path to the metrics file
            le_values (list): List of le values to search for

        Returns:
            tuple: Lists of input and output token values, or (None, None) if error
        """
        input_token_values = []
        output_token_values = []

        try:
            with open(filename, 'r') as file:
                content = file.read()

                for le_value in le_values:
                    # Patterns with dynamic le value
                    # Escape +Inf properly by replacing it with \+Inf in the regex
                    if le_value == "+Inf":
                        le_value = r"\+Inf"

                    input_pattern = rf'nv_llm_input_token_len_bucket{{model="tensorrt_llm",response_metric_type="total_input_tokens",version="1",le="{le_value}"}}\s+(\d+)'
                    output_pattern = rf'nv_llm_output_token_len_bucket{{model="tensorrt_llm",response_metric_type="total_output_tokens",version="1",le="{le_value}"}}\s+(\d+)'

                    input_match = re.search(input_pattern, content)
                    output_match = re.search(output_pattern, content)

                    if input_match:
                        # Extract the actual numeric value from the match
                        input_token_values.append(int(input_match.group(1)))
                    if output_match:
                        # Extract the actual numeric value from the match
                        output_token_values.append(int(output_match.group(1)))

            return input_token_values, output_token_values

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return None, None
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return None, None

    def _verify_per_request_custom_metrics(self, filename,
                                           expected_input_tokens,
                                           expected_output_tokens):
        """
        Helper to Verify request and response metrics.

        Args:
            filename (str): Path to the metrics file
            expected_input_tokens (list): Expected input token counts
            expected_output_tokens (list): Expected output token counts
        """
        # Multiple values lookup
        le_values = ["10", "50", "100", "500", "1000", "+Inf"]

        # Get actual values from file
        input_results, output_results = self._find_metric_values(
            filename, le_values)

        if input_results is None or output_results is None:
            self.fail("Failed to read metric values from file")
            return

        # Calculate expected histogram bucket counts
        input_bucket_counts = self._calculate_bucket_counts(
            expected_input_tokens, le_values)
        output_bucket_counts = self._calculate_bucket_counts(
            expected_output_tokens, le_values)
        # Verify input tokens
        self.assertTrue(
            len(input_bucket_counts) == len(input_results)
            and all(a == b for a, b in zip(input_bucket_counts, input_results)),
            f"Input token arrays don't match:\nExpected: {input_bucket_counts}\nActual: {input_results}"
        )

        # Verify output tokens
        self.assertTrue(
            len(output_bucket_counts) == len(output_results)
            and all(a == b
                    for a, b in zip(output_bucket_counts, output_results)),
            f"Output token arrays don't match:\nExpected: {output_bucket_counts}\nActual: {output_results}"
        )

    def _verify_end_to_end_metrics(self, filename):
        # Used to verify end to end test metrics output with STREAMING ON
        # Read the OUTPUT_SIZE environment variable
        stream_output_size = os.getenv('STREAM_OUTPUT_SIZE')
        stream_input_size = os.getenv('STREAM_INPUT_SIZE')
        if stream_input_size and stream_output_size:
            stream_input_size = int(stream_input_size)
            stream_output_size = int(stream_output_size)
            expected_input_token_len = []
            expected_output_token_len = []
            expected_input_token_len.append(stream_input_size)
            expected_output_token_len.append(stream_output_size)
            self._verify_per_request_custom_metrics(filename,
                                                    expected_input_token_len,
                                                    expected_output_token_len)
        else:
            self.assertTrue(
                False,
                "Unable to read stream_output_size and stream_input_size from env variables"
            )

    def _verify_base_metrics(self, filename):
        # FIXME: Custom parsing is messy. As part of the Triton
        # CLI work, we should add a metrics client API that will
        # return the metrics in a neatly formatted JSON.
        model_metrics = defaultdict(dict)
        with open(filename) as metrics_file:
            for line in metrics_file:
                if line[0] != "#" and "nv_inference" in line:
                    # Splits metric line into:
                    # ex. 'nv_inference_request_success', '{model="ensemble",version="1"}', '104'
                    model_data = line.replace("{", " {").split()
                    key = model_data[0].replace("nv_inference_", "")
                    model = model_data[1].split('"')[1]
                    value = model_data[2]
                    model_metrics[model][key] = value

        print(json.dumps(model_metrics, indent=4))

        # Assert the expected models are in the metrics output
        expected_models = [
            "ensemble", "preprocessing", "postprocessing", "tensorrt_llm"
        ]
        self.assertTrue(all(model in model_metrics
                            for model in expected_models))

        # Assert each model records the same number of metrics
        self.assertTrue(
            self._all_equal(
                [len(model_metrics[model].keys()) for model in model_metrics]))

        # Assert models have the same counts
        count_keys = [
            "request_success", "request_failure", "count", "exec_count",
            "pending_request_count"
        ]
        for stat in count_keys:
            if stat == "exec_count":
                # Dynamic batching is enabled for the post-processing model and
                # pre-processing, so the 'exec_count' will not be the same
                # between the postprocessing model and other models.
                self.assertTrue(
                    self._all_equal([
                        model_metrics[model][stat] for model in model_metrics if
                        model != "postprocessing" and model != "preprocessing"
                    ]))
            else:
                self.assertTrue(
                    self._all_equal([
                        model_metrics[model][stat] for model in model_metrics
                    ]))

        duration_keys = [
            "request_duration_us", "compute_input_duration_us",
            "compute_infer_duration_us", "compute_output_duration_us"
        ]
        for stat in duration_keys:
            composing_stat_duration = sum([
                int(model_metrics[model][stat]) for model in model_metrics
                if model != "ensemble"
            ])
            ensemble_stat_duration = int(model_metrics["ensemble"][stat])
            self.assertTrue(composing_stat_duration > 0)
            self.assertTrue(ensemble_stat_duration > 0)

    def test_end_to_end(self):
        try:
            client = utils.create_inference_server_client("http",
                                                          "localhost:8000",
                                                          concurrency=128,
                                                          verbose=True)
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        max_input_len = 500
        op_tokens_per_word = 1.3
        dataset = "./simple_data.json"

        prompts = []
        output_lens = []
        with open(dataset, "r") as f:
            data_dict = json.load(f)
            for req in data_dict:
                prompt = req["input"] + " " + req["instruction"]
                output = req["output"]
                # 1.3 is a magic number that converts number of words to number of tokens
                if int(len(prompt.split(" ")) /
                       op_tokens_per_word) > max_input_len:
                    continue
                prompts.append(prompt)
                self.expected_input_token_len.append(
                    len(prompt.split(" ")) * op_tokens_per_word)
                output_lens.append(
                    int(len(output.split(" ")) * op_tokens_per_word))

        self._run_infer(client, prompts, output_lens)
        metrics = self._get_metrics()
        filename = "./base_metrics.out"
        with open(filename, "w+") as metrics_file:
            metrics_file.write(metrics)
        self._verify_base_metrics(filename)
        self._verify_per_request_custom_metrics(filename,
                                                self.expected_input_token_len,
                                                self.expected_output_token_len)
        filename = "./end_to_end_token_metrics.out"
        self._verify_end_to_end_metrics(filename)

    def test_end_to_end_beam_width(self):
        # End to end test for beam > 1
        try:
            client = utils.create_inference_server_client("http",
                                                          "localhost:8000",
                                                          concurrency=128,
                                                          verbose=True)
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        max_input_len = 500
        op_tokens_per_word = 1.3
        dataset = "./simple_data.json"
        prompts = []
        output_lens = []
        self.expected_input_token_len = []
        self.expected_output_token_len = []
        with open(dataset, "r") as f:
            data_dict = json.load(f)
            for req in data_dict:
                prompt = req["input"] + " " + req["instruction"]
                output = req["output"]
                # 1.3 is a magic number that converts number of words to number of tokens
                if int(len(prompt.split(" ")) /
                       op_tokens_per_word) > max_input_len:
                    continue
                prompts.append(prompt)
                self.expected_input_token_len.append(
                    len(prompt.split(" ")) * op_tokens_per_word)
                output_lens.append(
                    int(len(output.split(" ")) * op_tokens_per_word))
        beam_width = 2
        self._run_infer(client, prompts, output_lens, beam_width)
        metrics = self._get_metrics()
        filename = "./base_metrics_beam_width.out"
        with open(filename, "w+") as metrics_file:
            metrics_file.write(metrics)
        self._verify_per_request_custom_metrics(filename,
                                                self.expected_input_token_len,
                                                self.expected_output_token_len)


if __name__ == "__main__":
    unittest.main()
