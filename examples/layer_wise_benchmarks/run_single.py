import argparse

import numpy as np
import nvtx
import torch
import yaml

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._utils import local_mpi_rank, mpi_rank, mpi_world_size
from tensorrt_llm.tools.layer_wise_benchmarks.deepseekv3_runner import (
    BalanceMethod, DeepSeekV3Runner)


def comma_separated_ints(s):
    return [int(x) for x in s.split(",")]


# Parse cmdline
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str)
parser.add_argument("--test-case",
                    type=str,
                    choices=["CTX", "GEN"],
                    default="GEN")
parser.add_argument(
    "--layer-indices",
    type=comma_separated_ints,
    default=[5],
    help="Indices of layers to profile, should be a contiguous range.")
args = parser.parse_args()
with open(args.config_path) as f:
    config = yaml.safe_load(f)[args.test_case]

# MPI args
rank = mpi_rank()
world_size = mpi_world_size()
local_rank = local_mpi_rank()
torch.cuda.set_device(local_rank)

# Create KV cache manager
pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
max_batch_size = config["max_batch_size"]
max_seq_len = config["max_seq_len"]
enable_attention_dp = config["enable_attention_dp"]
mapping = DeepSeekV3Runner.create_mapping(
    enable_attention_dp=enable_attention_dp)
kv_cache_manager = DeepSeekV3Runner.create_kv_cache_manager(
    pretrained_model_name_or_path,
    mapping,
    max_batch_size=max_batch_size,
    max_seq_len=max_seq_len,
    layer_indices=args.layer_indices)
attn_workspace = torch.empty((0, ), device="cuda", dtype=torch.int8)

# Create other global objects
AutoTuner.get().clear_cache()
capture_stream = torch.cuda.Stream()

# Create Runner
max_num_tokens = config["max_num_tokens"]
moe_backend = config["moe_backend"]
use_cuda_graph = config["use_cuda_graph"]
runner = DeepSeekV3Runner(pretrained_model_name_or_path,
                          mapping,
                          moe_backend=moe_backend,
                          layer_indices=args.layer_indices,
                          max_seq_len=max_seq_len,
                          max_num_tokens=max_num_tokens,
                          use_cuda_graph=use_cuda_graph)

# Warm up
batch_size = config["batch_size"]
seq_len_q = config["seq_len_q"]
seq_len_kv_cache = config["seq_len_kv_cache"]
balance_method = BalanceMethod[config["balance_method"]]
balance_ratio = config["balance_ratio"]
run_pack = runner.create_run_pack(args.test_case,
                                  batch_size=batch_size,
                                  seq_len_q=seq_len_q,
                                  seq_len_kv_cache=seq_len_kv_cache,
                                  kv_cache_manager=kv_cache_manager,
                                  attn_workspace=attn_workspace)
runner.replace_routing_method(balance_method=balance_method,
                              balance_ratio=balance_ratio)
capture_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(capture_stream):
    run_pack()
    with autotune():
        run_pack()
torch.cuda.current_stream().wait_stream(capture_stream)
torch.cuda.synchronize()

# Profile: capture graph and replay it
torch.cuda.cudart().cudaProfilerStart()
if use_cuda_graph:
    with with_multi_stream(True):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,
                              stream=capture_stream,
                              capture_error_mode="global"):
            run_pack()

warmup_times = 20
run_times = 100
events = [
    torch.cuda.Event(enable_timing=True)
    for _ in range(warmup_times + run_times + 1)
]
for i in range(warmup_times + run_times):
    events[i].record()
    with nvtx.annotate(f"b={batch_size} s={seq_len_q} EP{world_size}"):
        if use_cuda_graph:
            g.replay()
        else:
            run_pack()
events[-1].record()
torch.cuda.synchronize()

# Print statistics
#   Print before `cudaProfilerStop` to ensure messages are included in the profile
time_list = [
    start.elapsed_time(stop) for start, stop in zip(events, events[1:])
]
time_list = time_list[warmup_times:]
print(f"[RANK {rank}]"
      f"  min {np.min(time_list) * 1000:.1f}"
      f"  max {np.max(time_list) * 1000:.1f}"
      f"  mean {np.mean(time_list) * 1000:.1f}"
      f"  median {np.median(time_list) * 1000:.1f}"
      f"  P90 {np.percentile(time_list, 90) * 1000:.1f}"
      f"  (us)")

torch.cuda.cudart().cudaProfilerStop()
