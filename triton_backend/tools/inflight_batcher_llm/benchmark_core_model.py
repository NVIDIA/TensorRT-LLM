#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import json
import sys
import time
from datetime import datetime
from functools import partial

import numpy as np
from transformers import AutoTokenizer
from utils import utils


def callback(user_data, result, error):
    user_data._completed_requests.put((result, error))
    if result is None:
        # There was an error.
        return
    try:
        # GRPC
        req_id = result.get_response().id
    except:
        # HTTP
        req_id = result.get_response()["id"]
    start_time = user_data._start_time_dict[req_id]
    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    user_data._latencies.append(latency)
    user_data._latency_dict[req_id] = latency
    user_data._stop_time_dict[req_id] = stop_time


def append_pad_id_to_tensors(pad_id, inputs):
    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
    else:
        pad_id_data = np.ones_like([[1]]).astype(np.int32) * 0

    inputs += [utils.prepare_tensor("pad_id", pad_id_data, FLAGS.protocol)]


def append_end_id_to_tensors(end_id, inputs):
    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.int32)
    else:
        end_id_data = np.ones_like([[1]]).astype(np.int32) * 1

    inputs += [utils.prepare_tensor("end_id", end_id_data, FLAGS.protocol)]


def test_performance(client,
                     input_start_ids,
                     input_lens,
                     output_lens,
                     delays,
                     FLAGS,
                     pad_id=None,
                     end_id=None):
    model_name = "tensorrt_llm"

    print(f"[INFO] Warm up for benchmarking.")
    if FLAGS.decoupled:
        client.start_stream(callback=lambda result, error: None,
                            stream_timeout=FLAGS.stream_timeout)
    for i in range(10):
        model_name = FLAGS.tensorrt_llm_model_name[i % len(
            FLAGS.tensorrt_llm_model_name)]
        output0_len = np.ones_like([[1]]).astype(np.int32) * 100
        if FLAGS.test_llmapi:
            input_data = np.array(
                [input_start_ids[0]], dtype=object
            )  ## TODO: [JIRA-4496] support batching in llmapi backend and add tests here.
            inputs = [
                utils.prepare_tensor("text_input", input_data, FLAGS.protocol),
                utils.prepare_tensor("sampling_param_max_tokens",
                                     np.array([output_lens[0]], dtype=np.int32),
                                     FLAGS.protocol),
            ]
        else:
            inputs = [
                utils.prepare_tensor("input_ids", input_start_ids[0],
                                     FLAGS.protocol),
                utils.prepare_tensor("input_lengths", input_lens[0],
                                     FLAGS.protocol),
                utils.prepare_tensor("request_output_len", output0_len,
                                     FLAGS.protocol),
            ]
            append_pad_id_to_tensors(pad_id, inputs)
            append_end_id_to_tensors(end_id, inputs)

        if FLAGS.decoupled:
            client.async_stream_infer(model_name, inputs, request_id=str(i))
        else:
            client.infer(model_name, inputs, request_id=str(i))
    if FLAGS.decoupled:
        client.stop_stream()

    print(f"[INFO] Start benchmarking on {len(input_start_ids)} prompts.")
    latency = 0
    async_requests = []
    start_time = datetime.now()
    user_data = utils.UserData()

    if FLAGS.decoupled:
        client.start_stream(callback=partial(callback, user_data),
                            stream_timeout=FLAGS.stream_timeout)
    for i, ids in enumerate(input_start_ids):
        model_name = FLAGS.tensorrt_llm_model_name[i % len(
            FLAGS.tensorrt_llm_model_name)]
        output0_len = np.ones_like([[1]]).astype(np.int32) * output_lens[i]
        if FLAGS.test_llmapi:
            input_data = np.array(
                [ids], dtype=object
            )  ## TODO: [JIRA-4496] support batching in llmapi backend and add tests here.
            inputs = [
                utils.prepare_tensor("text_input", input_data, FLAGS.protocol),
                utils.prepare_tensor("sampling_param_max_tokens",
                                     np.array([output_lens[i]], dtype=np.int32),
                                     FLAGS.protocol),
            ]
        else:
            inputs = [
                utils.prepare_tensor("input_ids", ids, FLAGS.protocol),
                utils.prepare_tensor("input_lengths", input_lens[i],
                                     FLAGS.protocol),
                utils.prepare_tensor("request_output_len", output0_len,
                                     FLAGS.protocol),
            ]

            append_pad_id_to_tensors(pad_id, inputs)
            append_end_id_to_tensors(end_id, inputs)

        time.sleep(delays[i])

        user_data._start_time_dict[str(i)] = datetime.now()
        if FLAGS.protocol == "http":
            async_requests.append(
                client.async_infer(model_name, inputs, request_id=str(i)))
        elif FLAGS.protocol == "grpc":
            if FLAGS.decoupled:
                client.async_stream_infer(model_name, inputs, request_id=str(i))
            else:
                async_requests.append(
                    client.async_infer(model_name,
                                       inputs,
                                       callback=partial(callback, user_data),
                                       request_id=str(i)))
    if FLAGS.decoupled:
        client.stop_stream()
    try:
        if FLAGS.protocol == "http":
            utils.get_http_results(async_requests)
        elif FLAGS.protocol == "grpc":
            responses = utils.get_grpc_results(user_data, len(input_start_ids))
        else:
            raise RuntimeError("Invalid protocol")

        stop_time = datetime.now()
        latency = (stop_time - start_time).total_seconds() * 1000.0
        latency = round(latency, 3)
        print(f"[INFO] Total Latency: {latency} ms")

        # TODO(kaiyu): support `extract_print_stats` for http
        # TODO(achartier): support `extract_print_stats` for LLMAPI
        data_dict = None
        if FLAGS.protocol == "grpc" and not FLAGS.test_llmapi:
            request_latencies = 0.0
            for latency in user_data._latencies:
                request_latencies += latency
            print(f"[INFO] Total request latencies: {request_latencies} ms")

            ip_token_len_list = []
            for ip in input_lens:
                ip_token_len_list.append(
                    ip[0][0])  #for some reason, two level nesting

            data_dict = utils.extract_print_stats(ip_token_len_list, responses,
                                                  user_data, FLAGS)

        if FLAGS.check_perf_json:
            check_performance(data_dict, FLAGS)

    except Exception as e:
        print("Failed receiving responses: " + str(e))
        sys.exit(1)


def check_performance(data_dict, FLAGS):
    if not data_dict:
        print(
            "[ERROR] --check-perf-json was used, but no data was collected. Please use grpc protocol."
        )
    ref = json.load(open(FLAGS.check_perf_json, "r"))
    if FLAGS.check_perf_key not in ref or len(ref[FLAGS.check_perf_key]) == 0:
        print(
            f"[ERROR] There are no reference numbers for {FLAGS.check_perf_key}, so the performance is not checked. Please add an entry to {FLAGS.check_perf_json}."
        )
        sys.exit(1)
    for metric in ref[FLAGS.check_perf_key]:
        if metric not in data_dict:
            print(f"[ERROR] Data for '{metric}' was not found.")
        np.testing.assert_allclose(
            data_dict[metric],
            ref[FLAGS.check_perf_key][metric],
            rtol=FLAGS.check_perf_rtol,
            atol=FLAGS.check_perf_atol,
            err_msg=
            f"'{metric}' check failed - did not match reference in '{FLAGS.check_perf_json}' for '{FLAGS.check_perf_key}'"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='workload')

    parser_dataset = subparsers.add_parser('dataset')
    parser_dataset.add_argument('--dataset',
                                type=str,
                                required=True,
                                help='Dataset path used for the test.')
    parser_dataset.add_argument('--tokenizer-dir',
                                type=str,
                                required=True,
                                help='Specify tokenizer directory')
    parser_dataset.add_argument('--tokenizer-type',
                                type=str,
                                default='auto',
                                required=False,
                                choices=['auto', 't5', 'llama'],
                                help='Specify tokenizer type')
    parser_dataset.add_argument(
        '--op-tokens-per-word',
        type=float,
        default=1.3,
        required=False,
        help=
        'Specify op tokens/word ratio. Useful to have model generate exactly as many tokens as needed by the dataset'
    )

    parser_token_norm_dist = subparsers.add_parser('token-norm-dist')
    parser_token_norm_dist.add_argument(
        '--input-mean',
        type=int,
        required=True,
        help='normal dist mean for input tokens')
    parser_token_norm_dist.add_argument(
        '--input-stdev',
        type=int,
        required=True,
        help='normal dist stdev for input tokens')
    parser_token_norm_dist.add_argument(
        '--output-mean',
        type=int,
        required=True,
        help='normal dist mean for output tokens')
    parser_token_norm_dist.add_argument(
        '--output-stdev',
        type=int,
        required=True,
        help='normal dist stdev for output tokens')

    parser_token_from_hist = subparsers.add_parser('token-from-histogram')
    parser_token_from_hist.add_argument(
        '--histogram-key',
        type=str,
        required=True,
        help='key to retrieve histogram buckets,freqs defined in utils')

    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        choices=['http', 'grpc'],
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument(
        '--decoupled',
        action="store_true",
        required=False,
        default=False,
        help=
        'Uses async_stream_infer which allows decoupled backends (must use grpc protocol)'
    ),
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "--tensorrt-llm-model-name",
        type=str,
        required=False,
        default=["tensorrt_llm"],
        action="append",
        help=
        "Specify the name of the TensorRT LLM model. Can be specified multiple times to use multiple models."
    )
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        default=128,
                        required=False,
                        help='Specify concurrency')
    parser.add_argument('--max-input-len',
                        type=int,
                        required=True,
                        help='Specify max input length')
    parser.add_argument('--request-rate',
                        type=float,
                        required=False,
                        help="# of reqs/sec. -1 indicates SOL/Offline",
                        default=-1.0)
    parser.add_argument('--time-delay-dist',
                        type=str,
                        required=False,
                        choices=["constant", "exponential_dist"],
                        default="exponential_dist",
                        help="# of reqs/sec. -1 indicates SOL/Offline")
    parser.add_argument(
        '--dump-perfetto-trace',
        action="store_true",
        required=False,
        default=False,
        help=
        'Dumps trace of requests in a json (perfetto.json) to be visualized in perfetto'
    ),
    parser.add_argument('--op-stats-csv',
                        type=str,
                        default=None,
                        help='csv filename to dump stats'),
    parser.add_argument(
        "--exclude-input-in-output",
        action="store_true",
        required=False,
        default=False,
        help="Expect that output IDs do not contain input IDs",
    )
    parser.add_argument(
        '--num-requests',
        type=int,
        required=False,
        default=30000,
        help=
        'For dataset, requests = min(dataset, num_requests). number of requests to be generated by the client'
    )
    parser.add_argument(
        '--check-perf-json',
        type=str,
        required=False,
        help=
        'If set, this will compare the latency to the value in this file under the key from --check-perf-key'
    )
    parser.add_argument(
        '--check-perf-key',
        type=str,
        required=False,
        help=
        'Used with --check-perf-json to specify which entry in the file to compare with'
    )
    parser.add_argument('--check-perf-atol',
                        type=float,
                        required=False,
                        help="Absolute tolerance for performance check",
                        default=50)
    parser.add_argument('--check-perf-rtol',
                        type=float,
                        required=False,
                        help="Relative tolerance for performance check",
                        default=0.05)
    parser.add_argument('--test-llmapi',
                        action="store_true",
                        required=False,
                        default=False,
                        help="Use LLMAPI for inference")

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"
    if FLAGS.decoupled and FLAGS.protocol != 'grpc':
        print("Protocol must be set to 'grpc' when using '--decoupled'.")
        sys.exit(1)

    try:
        client = utils.create_inference_server_client(
            FLAGS.protocol,
            FLAGS.url,
            concurrency=FLAGS.concurrency,
            verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if FLAGS.request_rate == -1:
        mean_time_bet_reqs = 0
    else:
        mean_time_bet_reqs = 1.0 / FLAGS.request_rate

    input_start_ids = []
    input_lens = []
    output_lens = []
    ratio = []

    print(FLAGS.workload)
    if FLAGS.workload == "dataset":
        tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left')
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        pad_id = tokenizer.encode(tokenizer.pad_token,
                                  add_special_tokens=False)[0]
        end_id = tokenizer.encode(tokenizer.eos_token,
                                  add_special_tokens=False)[0]

        prompt_cnt = 0

        with open(FLAGS.dataset, 'r') as f:
            data_dict = json.load(f)
            for req in data_dict:
                prompt = req['input'] + ' ' + req['instruction']
                output = req['output']
                line = tokenizer.encode(prompt)
                if len(line) > FLAGS.max_input_len:
                    continue

                prompt_cnt += 1
                if prompt_cnt > FLAGS.num_requests:
                    break

                if FLAGS.test_llmapi:
                    input_start_ids.append(prompt)
                else:
                    input_start_ids.append(np.array([line], np.int32))
                input_lens.append(np.array([[len(line)]], np.int32))
                output_lens.append(
                    int(len(output.split(' ')) * FLAGS.op_tokens_per_word))
                prompt_tokens = len(line)
                prompt_words = len(prompt.split())
                ratio.append(prompt_tokens / prompt_words)

        print("Tokenizer: Tokens per word = ", round(np.mean(ratio), 3))
        num_reqs = len(input_lens)
        delays = utils.get_list_of_delays(FLAGS.time_delay_dist,
                                          mean_time_bet_reqs, num_reqs)
        test_performance(client, input_start_ids, input_lens, output_lens,
                         delays, FLAGS, pad_id, end_id)

    elif FLAGS.workload == "token-norm-dist":
        assert not FLAGS.test_llmapi, "LLMAPI does not support token-norm-dist workload yet"
        input_lens = utils.get_norm_dist_tokens(FLAGS.input_mean,
                                                FLAGS.input_stdev,
                                                FLAGS.num_requests)
        pruned_ip_list = [
            ip_len for ip_len in input_lens if ip_len <= FLAGS.max_input_len
        ]
        num_reqs = len(pruned_ip_list)
        ip_lens_2d_array = [
            np.array([[ip_len]], np.int32) for ip_len in pruned_ip_list
        ]
        output_lens = utils.get_norm_dist_tokens(FLAGS.output_mean,
                                                 FLAGS.output_stdev, num_reqs)
        delays = utils.get_list_of_delays(FLAGS.time_delay_dist,
                                          mean_time_bet_reqs, num_reqs)

        input_start_ids = utils.gen_random_start_ids(pruned_ip_list)
        test_performance(client, input_start_ids, ip_lens_2d_array, output_lens,
                         delays, FLAGS)

    elif FLAGS.workload == "token-from-histogram":
        assert not FLAGS.test_llmapi, "LLMAPI does not support token-from-histogram workload yet"
        input_lens_orig = utils.get_token_list_from_histogram(
            FLAGS.histogram_key + "_ip")
        output_lens_orig = utils.get_token_list_from_histogram(
            FLAGS.histogram_key + "_op")

        final_lens = min(len(input_lens_orig), len(output_lens_orig))
        input_lens = input_lens_orig[:final_lens]
        output_lens = output_lens_orig[:final_lens]

        num_reqs = len(input_lens)
        ip_lens_2d_array = [
            np.array([[ip_len]], np.int32) for ip_len in input_lens
        ]
        output_lens = utils.get_token_list_from_histogram(FLAGS.histogram_key +
                                                          "_op")
        print(len(input_lens), len(output_lens))
        assert (len(input_lens) == len(output_lens))

        delays = utils.get_list_of_delays(FLAGS.time_delay_dist,
                                          mean_time_bet_reqs, num_reqs)

        input_start_ids = utils.gen_random_start_ids(input_lens)
        test_performance(client, input_start_ids, ip_lens_2d_array, output_lens,
                         delays, FLAGS)
