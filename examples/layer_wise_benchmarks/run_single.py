import argparse
import pathlib

import numpy as np
import nvtx
import torch

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._utils import local_mpi_rank, mpi_rank, mpi_world_size
from tensorrt_llm.tools.layer_wise_benchmarks.deepseekv3_runner import (
    BalanceMethod, DeepSeekV3Runner)

parser = argparse.ArgumentParser()
parser.add_argument("--test-case",
                    type=str,
                    choices=["CTX", "GEN"],
                    default="GEN")
args = parser.parse_args()

# MPI args
rank = mpi_rank()
world_size = mpi_world_size()
local_rank = local_mpi_rank()
torch.cuda.set_device(local_rank)

# Model definition
model_dir = pathlib.Path(__file__).parent / "DeepSeek-R1-0528-FP4-v2"
layer_indices = [5, 6]

# KV cache related args
if args.test_case == "CTX":
    MAX_BATCH_SIZE = 1024
    MAX_SEQ_LEN = 8192 + 1024 + 4
    MAX_NUM_TOKENS = 40960
    enable_attention_dp = True
    moe_backend = "CUTLASS"
elif args.test_case == "GEN":
    MAX_BATCH_SIZE = 1024
    MAX_SEQ_LEN = 8192 + 1024 + 4
    MAX_NUM_TOKENS = 4 * MAX_BATCH_SIZE  # MTP3 as max
    enable_attention_dp = True
    moe_backend = "WIDEEP"
else:
    raise NotImplementedError(f"Not support test case \"{args.test_case}\"")

# Create KV cache manager
mapping = DeepSeekV3Runner.create_mapping(
    enable_attention_dp=enable_attention_dp)
kv_cache_manager = DeepSeekV3Runner.create_kv_cache_manager(
    model_dir,
    mapping,
    max_batch_size=MAX_BATCH_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    layer_indices=layer_indices)
attn_workspace = torch.empty((0, ), device="cuda", dtype=torch.int8)

# Create other global objects
use_cuda_graph = args.test_case == "GEN"
AutoTuner.get().clear_cache()
capture_stream = torch.cuda.Stream()

# Create Runner
runner = DeepSeekV3Runner(model_dir,
                          mapping,
                          moe_backend=moe_backend,
                          layer_indices=layer_indices,
                          max_seq_len=MAX_SEQ_LEN,
                          max_num_tokens=MAX_NUM_TOKENS,
                          use_cuda_graph=use_cuda_graph)

# Warm up
if args.test_case == "CTX":
    batch_size = 3
    seq_len_q = 8193
    seq_len_kv_cache = 0
elif args.test_case == "GEN":
    batch_size = 128
    seq_len_q = 1  # Set to (1 + MTP)
    seq_len_kv_cache = 8193
else:
    raise NotImplementedError(f"Not support test case \"{args.test_case}\"")
balance_method = BalanceMethod.Balanced
balance_ratio = 1.
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
