#!/usr/bin/env python3
import json
import re
import sys


def extract_benchmark(log_text):
    pattern = r'(\w+)\s*(\(.*?\))?\s*:\s*([\d.]+)'
    start = log_text.find('num_samples')
    end = log_text.find('_________________________________')
    first = log_text[start:end]

    matches = re.findall(pattern, first)

    kv_pairs = {}
    benchmark_kv_pairs = {}

    for key, _, value in matches:
        if key in ['num_samples', 'total_latency', 'token_throughput']:
            if key in kv_pairs:
                benchmark_kv_pairs[key] = value
            else:
                kv_pairs[key] = value

    benchmark_matches = re.findall(r'\[BENCHMARK\] (\w+)(\(.*?\))?\s*([\d.]+)',
                                   log_text)
    for key, _, value in benchmark_matches:
        if key in ['num_samples', 'total_latency', 'token_throughput']:
            benchmark_kv_pairs[key] = value

    data = {"llmapi": kv_pairs, "cpp": benchmark_kv_pairs}

    print(json.dumps(data, indent=4))


input = sys.stdin.read()
extract_benchmark(input)
