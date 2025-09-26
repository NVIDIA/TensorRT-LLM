# Adopted from
# https://github.com/vllm-project/vllm/blob/200bbf92e8861e2458a6f90bca73f40cc3b1ad1f/benchmarks/benchmark_utils.py
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
from typing import Any, Optional

import requests
from tqdm.asyncio import tqdm


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = extra_info["tensor_parallel_size"]

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)


def download_and_cache_file(url: str, path: Optional[str], name: str,
                            timeout: int) -> str:
    # Adapted from
    # https://github.com/sgl-project/sglang/blob/58f10679e1850fdc86046057c23bac5193156de9/python/sglang/bench_serving.py#L586
    """Read and cache a file from a url."""

    # Check if the path is valid and if the file exists
    if path is None or not os.path.exists(path):
        raise ValueError("download_path is not provided or does not exist")
    filename = os.path.join(path, name)

    if is_file_valid_json(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def is_file_valid_json(path) -> bool:
    # Adapted from
    # https://github.com/sgl-project/sglang/blob/58f10679e1850fdc86046057c23bac5193156de9/python/sglang/bench_serving.py#L620
    if not os.path.isfile(path):
        return False

    # TODO can fuse into the real file open later
    try:
        with open(path) as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(
            f"{path} exists but json loading fails ({e=}), thus treat as invalid file"
        )
        return False
