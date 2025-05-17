import csv
import json
import math
import queue
import random
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tabulate import tabulate
from tritonclient.utils import np_to_triton_dtype


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()
        self._latencies = []
        self._latency_dict = {}
        self._start_time_dict = {}
        self._stop_time_dict = {}


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(name, input.shape,
                               np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def prepare_outputs(protocol,
                    return_log_probs=False,
                    return_context_logits=False,
                    return_generation_logits=False,
                    return_finish_reason=False,
                    return_stop_reason=False,
                    return_cumulative_logprob=False):

    client_util = httpclient if protocol == "http" else grpcclient

    outputs = []
    outputs.append(client_util.InferRequestedOutput("text_output"))

    if return_log_probs:
        outputs.append(client_util.InferRequestedOutput("cum_log_probs"))
        outputs.append(client_util.InferRequestedOutput("output_log_probs"))

    if return_context_logits:
        outputs.append(client_util.InferRequestedOutput("context_logits"))

    if return_generation_logits:
        outputs.append(client_util.InferRequestedOutput("generation_logits"))

    if return_finish_reason:
        outputs.append(client_util.InferRequestedOutput("finish_reason"))

    if return_stop_reason:
        outputs.append(client_util.InferRequestedOutput("stop_reason"))

    if return_cumulative_logprob:
        outputs.append(client_util.InferRequestedOutput("cumulative_logprob"))

    return outputs


def prepare_inputs(input_start_ids, input_len, pad_id, end_id, flags):
    output_len = np.ones([input_start_ids.shape[0], 1]).astype(
        np.int32) * flags.output_len
    runtime_top_k = (flags.topk *
                     np.ones([input_start_ids.shape[0], 1])).astype(np.int32)
    runtime_top_p = flags.topp * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    temperature = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    seed = 0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.uint64)
    output_log_probs = True * \
        np.ones([input_start_ids.shape[0], 1]).astype(bool)
    beam_width = (flags.beam_width *
                  np.ones([input_start_ids.shape[0], 1])).astype(np.int32)
    pad_ids = pad_id * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.int32)
    end_ids = end_id * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.int32)
    min_tokens = 1 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.int32)
    presence_penalty = 0.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    frequency_penalty = 0.0 * \
        np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
    bad_words_list = np.concatenate([
        np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
        (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)
    ],
                                    axis=1)
    stop_word_list = np.concatenate([
        np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32),
        (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)
    ],
                                    axis=1)
    inputs = [
        prepare_tensor("input_ids", input_start_ids, flags.protocol),
        prepare_tensor("input_lengths", input_len, flags.protocol),
        prepare_tensor("request_output_len", output_len, flags.protocol),
        prepare_tensor("pad_id", pad_ids, flags.protocol),
        prepare_tensor("end_id", end_ids, flags.protocol),
        prepare_tensor("beam_width", beam_width, flags.protocol),
        prepare_tensor("temperature", temperature, flags.protocol),
        prepare_tensor("runtime_top_k", runtime_top_k, flags.protocol),
        prepare_tensor("runtime_top_p", runtime_top_p, flags.protocol),
        prepare_tensor("len_penalty", len_penalty, flags.protocol),
        prepare_tensor("repetition_penalty", repetition_penalty,
                       flags.protocol),
        prepare_tensor("min_tokens", min_tokens, flags.protocol),
        prepare_tensor("presence_penalty", presence_penalty, flags.protocol),
        prepare_tensor("frequency_penalty", frequency_penalty, flags.protocol),
        prepare_tensor("seed", seed, flags.protocol),
        prepare_tensor("output_log_probs", output_log_probs, flags.protocol),
        # prepare_tensor("bad_words_list", bad_words_list, flags.protocol),
        # prepare_tensor("stop_words_list", stop_word_list, flags.protocol),
    ]
    return inputs


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                 concurrency=concurrency,
                                                 verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url, verbose=verbose)


def send_requests(model_name, inputs, client, request_parallelism):
    results = []
    for _ in range(request_parallelism):
        result = client.infer(model_name, inputs)
        results.append(result)
    return results


def send_requests_async(model_name, inputs, client, flags, request_parallelism):
    if flags.protocol == "http":
        async_requests = []
        for _ in range(request_parallelism):
            async_requests.append(client.async_infer(model_name, inputs))
        return async_requests
    else:
        user_data = UserData()
        for _ in range(request_parallelism):
            client.async_infer(model_name, inputs,
                               partial(completion_callback, user_data))
        return user_data


def get_http_results(async_requests):
    results = []
    for async_request in async_requests:
        results.append(async_request.get_result())
    return results


def get_grpc_results(user_data, request_parallelism):
    results = []
    processed_count = 0
    while processed_count < request_parallelism:
        (result, error) = user_data._completed_requests.get()
        processed_count += 1
        if error is not None:
            raise RuntimeError(error)
        results.append(result)
    return results


def append_start_and_end_ids(inputs,
                             batch_size,
                             flags,
                             start_id=None,
                             end_id=None):
    if start_id is not None:
        start_ids = start_id * np.ones([batch_size, 1]).astype(np.int32)
        inputs.append(prepare_tensor("start_id", start_ids, flags.protocol))
    if end_id is not None:
        end_ids = end_id * np.ones([batch_size, 1]).astype(np.int32)
        inputs.append(prepare_tensor("end_id", end_ids, flags.protocol))


def generate_histogram(range_buckets, frequencies):
    histogram = []

    for i in range(len(range_buckets)):
        bucket = range_buckets[i]
        frequency = frequencies[i]

        # Split the bucket range into min and max values
        min_range, max_range = bucket

        # Generate 'frequency' random values within the specified range
        random.seed(420)
        random_values = [
            random.randint(min_range, max_range) for _ in range(frequency)
        ]

        # Extend the histogram with the random values
        histogram.extend(random_values)

    # Randomize the order of values in the histogram
    random.shuffle(histogram)

    return histogram


def get_token_list_from_histogram(histogram_key):

    histogram_buckets = {
        "example_ip": [(151, 175), (176, 200), (201, 225), (226, 250),
                       (251, 275)],
        "example_op": [(6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    }
    histogram_freq = {
        "example_ip": [220, 225, 150, 150, 140],
        "example_op": [76, 210, 174, 130, 152]
    }

    range_buckets = histogram_buckets[histogram_key]
    freqs = histogram_freq[histogram_key]
    assert (len(range_buckets) == len(freqs))

    return generate_histogram(range_buckets, freqs)


def get_list_of_delays(delay_dist, mean_time_bet_reqs, num_reqs):
    if delay_dist == "constant":
        delays = [mean_time_bet_reqs] * num_reqs
    elif delay_dist == "exponential_dist":
        delays = get_exponential_dist_delays(mean_time_bet_reqs, num_reqs)

    return delays


def get_exponential_dist_delays(mean_time_bet_reqs, num_reqs):
    # set seed for determinism
    np.random.seed(420)
    return np.random.exponential(mean_time_bet_reqs, num_reqs).tolist()


def get_norm_dist_tokens(mean, stdev, num_reqs):
    # set seed for determinism
    np.random.seed(420)
    numbers_list = np.random.normal(loc=mean, scale=stdev,
                                    size=num_reqs).tolist()
    return [max(1, math.ceil(x)) for x in numbers_list]


def gen_random_start_ids(ip_lens):
    input_start_ids = []
    for ip_len in ip_lens:
        start_ids = list(
            np.random.randint(low=0,
                              high=np.iinfo(np.int32).max,
                              size=ip_len,
                              dtype=np.int32))
        input_start_ids.append(np.array([start_ids]))

    return input_start_ids


def get_list_of_delays(delay_dist, mean_time_bet_reqs, num_reqs):
    if delay_dist == "constant":
        delays = [mean_time_bet_reqs] * num_reqs
    elif delay_dist == "exponential_dist":
        delays = get_exponential_dist_delays(mean_time_bet_reqs, num_reqs)

    return delays


def get_exponential_dist_delays(mean_time_bet_reqs, num_reqs):
    return np.random.exponential(mean_time_bet_reqs, num_reqs).tolist()


def get_norm_dist_tokens(mean, stdev, num_reqs):
    numbers_list = np.random.normal(loc=mean, scale=stdev,
                                    size=num_reqs).tolist()
    return [max(1, math.ceil(x)) for x in numbers_list]


def get_inflight_reqs_profile(start_times, end_times, requests_per_sec):
    """
    Receives start and end times of all requests,
    divides total E2E time into equal intervals and assigns how many requests are in flight
    in each interval.
    """
    # Calculate min of start time and max of end time
    min_start_time = min(start_times)
    max_end_time = max(end_times)

    # need to have enough resolution intervals depending on avg. latency per request. 10 times smaller than request processing time
    sec_per_request = 1.0 / requests_per_sec
    NUM_INTERVALS = int((max_end_time - min_start_time) /
                        timedelta(seconds=(sec_per_request / 10)))
    print(NUM_INTERVALS)
    # Calculate interval length
    interval_length = (max_end_time - min_start_time) / NUM_INTERVALS

    # Initialize a list to store the count of requests in each interval
    interval_counts = [0] * NUM_INTERVALS

    # Iterate through the requests and update interval counts
    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]

        # Calculate which interval the request falls into
        interval_index = int((start - min_start_time) / interval_length)

        # Increment the count for that interval and subsequent intervals until end
        while start < end and interval_index < NUM_INTERVALS:
            interval_counts[interval_index] += 1
            interval_index += 1
            start += interval_length

    return interval_counts


def extract_print_stats(ip_token_len_list, responses, user_data, FLAGS):

    #### Gather info about requests
    op_token_len_list = []
    op_token_len_ooo = {}

    for response in responses:
        #JG: long sequence to extract output length from response json dict. Responses are out of order
        op_token_len_ooo[response.get_response(as_json=True)['id']] = \
            int(response.get_response(as_json=True)['outputs'][0]['shape'][2])

    op_token_len_list = [
        value for key, value in sorted(op_token_len_ooo.items())
    ]

    assert (len(op_token_len_list) == len(ip_token_len_list))
    if not FLAGS.exclude_input_in_output:
        for i in range(len(op_token_len_list)):
            op_token_len_list[i] = op_token_len_list[i] - ip_token_len_list[i]

    # Get latencies per request
    # Order latencies based on issue order.
    latency_list_in_order = [
        value for key, value in sorted(user_data._latency_dict.items())
    ]
    start_time_list_in_order = [
        value for key, value in sorted(user_data._start_time_dict.items())
    ]
    stop_time_list_in_order = [
        value for key, value in sorted(user_data._stop_time_dict.items())
    ]

    latency_sorted = np.sort(latency_list_in_order)
    index_99 = math.ceil(len(latency_sorted) * 0.99)
    index_90 = math.ceil(len(latency_sorted) * 0.90)

    data = {
        'latency': latency_list_in_order,
        'start_time': start_time_list_in_order,
        'stop_time': stop_time_list_in_order,
        'num_ip_tokens': ip_token_len_list,
        'num_op_tokens': op_token_len_list
    }

    # Bundle everything in a single DF
    df = pd.DataFrame(data)

    #stats
    df['num_ip_tokens'].sum()
    avg_ip_tokens = df['num_ip_tokens'].mean()
    df['num_ip_tokens'].median()
    df['num_ip_tokens'].std()
    total_op_tokens = df['num_op_tokens'].sum()
    avg_op_tokens = df['num_op_tokens'].mean()
    df['num_op_tokens'].median()
    df['num_op_tokens'].std()

    tend = max(df['stop_time'].tolist())
    t0 = min(df['start_time'].tolist())
    total_latency = (tend - t0).total_seconds()
    requests_per_sec = len(responses) / total_latency
    tokens_generated_per_sec = total_op_tokens / total_latency

    avg_in_flight_requests = 0

    print_data_dict = {}
    print_data_dict["Requests/Sec"] = requests_per_sec
    print_data_dict["OP tokens/sec"] = tokens_generated_per_sec
    print_data_dict["Avg. latency (ms)"] = np.mean(latency_list_in_order)
    print_data_dict["P99 latency (ms)"] = latency_sorted[index_99 - 1]
    print_data_dict["P90 latency (ms)"] = latency_sorted[index_90 - 1]
    print_data_dict["Avg. Input tokens per request"] = avg_ip_tokens
    print_data_dict["Avg. Output tokens per request"] = avg_op_tokens
    print_data_dict["Avg. InFlight requests"] = avg_in_flight_requests
    print_data_dict["Total latency (ms)"] = total_latency * 1000
    print_data_dict["Total requests"] = len(responses)

    print_data = [["Requests/Sec", requests_per_sec],
                  ["OP tokens/sec", tokens_generated_per_sec],
                  ["Avg. latency (ms)",
                   np.mean(latency_list_in_order)],
                  ["P99 latency (ms)", latency_sorted[index_99 - 1]],
                  ["P90 latency (ms)", latency_sorted[index_90 - 1]],
                  ["Avg. IP tokens per request", avg_ip_tokens],
                  ["Avg. OP tokens per request", avg_op_tokens],
                  ["Avg. InFlight requests", avg_in_flight_requests],
                  ["Total latency (ms)", total_latency * 1000],
                  ["Total requests", len(responses)]]

    # Format numerical values to 2 decimal places
    formatted_data = [[item, f"{value:.2f}"] for item, value in print_data]
    headers = ["Stat", "Value"]
    table = tabulate(formatted_data, headers=headers, tablefmt="pretty")

    if FLAGS.op_stats_csv is not None:
        with open(FLAGS.op_stats_csv, "a", newline="") as file:
            filednames = print_data_dict.keys()
            writer = csv.DictWriter(file, fieldnames=filednames)

            # Check if the file is empty, and write the header if needed
            if file.tell() == 0:
                writer.writeheader()

            # Write the dictionaries as new rows
            writer.writerow(print_data_dict)

    print(table)

    if FLAGS.dump_perfetto_trace:
        json_dict = []
        for i in range(len(op_token_len_list)):
            req_dict = {}
            req_dict['name'] = 'req_{}'.format(i)
            req_dict["cat"] = "batch"
            req_dict["ph"] = "X"
            req_dict["ts"] = (start_time_list_in_order[i].timestamp() -
                              t0.timestamp()) * 1000000  #perfetto expects us
            req_dict["dur"] = (
                stop_time_list_in_order[i] -
                start_time_list_in_order[i]).total_seconds() * 1000000
            req_dict["pid"] = "1"
            req_dict["args"] = {
                "isl": int(ip_token_len_list[i]),
                "osl": int(op_token_len_list[i])
            }
            json_dict.append(req_dict)

        with open("prfetto_dump.json", "w") as file:
            json.dump(json_dict, file, indent=4)

    return print_data_dict


def extract_string_from_nested_list(nested_list):
    if isinstance(nested_list, str):
        return nested_list
    elif isinstance(nested_list, list):
        for item in nested_list:
            extracted_string = extract_string_from_nested_list(item)
            if extracted_string:
                return extracted_string
    return ""
