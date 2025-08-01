#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import statistics as s
from builtins import range
from datetime import datetime

import numpy as np
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
    parser.add_argument('-w',
                        '--warm_up',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable warm_up before benchmark')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        default=1,
                        required=False,
                        help='Specify concurrency')
    parser.add_argument('-p',
                        '--request_parallelism',
                        type=int,
                        default=10,
                        required=False,
                        help='Specify request parallelism')
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        required=False,
                        default='sync',
                        help='Mode ("sync"/"async").')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=8,
                        required=False,
                        help='Specify batch size')
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
    parser.add_argument('-s',
                        '--start_len',
                        type=int,
                        default=8,
                        required=False,
                        help='Specify input length')
    parser.add_argument('-o',
                        '--output_len',
                        type=int,
                        default=10,
                        required=False,
                        help='Specify output length')
    parser.add_argument(
        '-n',
        '--num_runs',
        type=int,
        default=1,
        required=False,
        help="Spedifty number of runs to get the average latency")

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"
    input_start_ids = np.random.randint(0,
                                        50255,
                                        size=(FLAGS.batch_size,
                                              FLAGS.start_len),
                                        dtype=np.int32)
    input_len = np.array([[input_start_ids.shape[1]]
                          for _ in range(input_start_ids.shape[0])], np.int32)
    inputs = utils.prepare_inputs(input_start_ids,
                                  input_len,
                                  pad_id=0,
                                  end_id=2,
                                  flags=FLAGS)

    # warm up
    if FLAGS.warm_up:
        print("[INFO] sending requests to warm up")
        with utils.create_inference_server_client(
                FLAGS.protocol,
                FLAGS.url,
                concurrency=FLAGS.concurrency,
                verbose=FLAGS.verbose) as client:
            utils.send_requests('tensorrt_llm',
                                inputs,
                                client,
                                request_parallelism=2)

    latencies = []
    for i in range(FLAGS.num_runs):
        start_time = datetime.now()

        with utils.create_inference_server_client(
                FLAGS.protocol,
                FLAGS.url,
                concurrency=FLAGS.concurrency,
                verbose=FLAGS.verbose) as client:
            if FLAGS.mode == 'sync':
                utils.send_requests('tensorrt_llm', inputs, client,
                                    FLAGS.request_parallelism)
            else:
                if FLAGS.protocol == "http":
                    async_requests = utils.send_requests_async(
                        'tensorrt_llm', inputs, client, FLAGS,
                        FLAGS.request_parallelism)
                    results = utils.get_http_results(async_requests)
                else:
                    user_data = utils.send_requests_async(
                        'tensorrt_llm', inputs, client, FLAGS,
                        FLAGS.request_parallelism)
                    results = utils.get_grpc_results(user_data,
                                                     FLAGS.request_parallelism)

        stop_time = datetime.now()
        latencies.append((stop_time - start_time).total_seconds() * 1000.0 /
                         FLAGS.request_parallelism)

    if FLAGS.num_runs > 1:
        latency = s.mean(latencies)
    else:
        latency = latencies[0]
    latency = round(latency, 3)
    throughput = round(1000 / latency * FLAGS.batch_size, 3)
    print(
        f"[INFO] Batch size: {FLAGS.batch_size}, Start len: {FLAGS.start_len}, Output len: {FLAGS.output_len}"
    )
    print(f"[INFO] Latency: {latency} ms")
    print(f"[INFO] Throughput: {throughput} sentences / sec")
