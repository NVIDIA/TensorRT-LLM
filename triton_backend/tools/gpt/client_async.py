#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
from datetime import datetime

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from utils import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('-t',
                        '--text',
                        type=str,
                        required=False,
                        default='Born in north-east France, Soyer trained as a',
                        help='Input text')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        default=1,
                        required=False,
                        help='Specify concurrency')
    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        required=False,
                        help='Specify beam width')
    parser.add_argument('-topk',
                        '--topk',
                        type=int,
                        default=1,
                        required=False,
                        help='topk for sampling')
    parser.add_argument('-topp',
                        '--topp',
                        type=float,
                        default=0.0,
                        required=False,
                        help='topp for sampling')
    parser.add_argument('-o',
                        '--output_len',
                        type=int,
                        default=10,
                        required=False,
                        help='Specify output length')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        required=True,
                        help='Specify tokenizer directory')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                              legacy=False,
                                              padding_side='left')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
    end_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

    line = tokenizer.encode(FLAGS.text)
    input_start_ids = np.array([line], np.int32)
    input_len = np.array([[len(line)]], np.int32)
    inputs = utils.prepare_inputs(input_start_ids, input_len, pad_id, end_id,
                                  FLAGS)

    start_time = datetime.now()

    with utils.create_inference_server_client(FLAGS.protocol,
                                              FLAGS.url,
                                              concurrency=FLAGS.concurrency,
                                              verbose=FLAGS.verbose) as client:
        if FLAGS.protocol == "http":
            async_requests = utils.send_requests_async('tensorrt_llm',
                                                       inputs,
                                                       client,
                                                       FLAGS,
                                                       request_parallelism=1)
            results = utils.get_http_results(async_requests)
        else:
            user_data = utils.send_requests_async('tensorrt_llm',
                                                  inputs,
                                                  client,
                                                  FLAGS,
                                                  request_parallelism=1)
            results = utils.get_grpc_results(user_data, request_parallelism=1)
    output_ids = results[0].as_numpy("output_ids")

    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    print(f"[INFO] Latency: {latency} ms")

    output_ids = output_ids.reshape(
        (output_ids.size, )).tolist()[input_start_ids.shape[1]:]
    output_text = tokenizer.decode(output_ids)
    print(f'Input: {FLAGS.text}')
    print(f'Output: {output_text}')
