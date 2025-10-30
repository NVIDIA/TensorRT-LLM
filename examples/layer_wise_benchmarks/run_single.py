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
parser.add_argument("--model", type=str, help="Pretrained model name or path")
parser.add_argument(
    "--layer-indices",
    type=comma_separated_ints,
    help="Comma separated indices of layers, should be a contiguous range")
parser.add_argument("--run-type", type=str, choices=["CTX", "GEN"])
parser.add_argument("--scaled-from", type=int)
# KV cache related args
parser.add_argument("--tokens-per-block", type=int)
parser.add_argument("--max-seq-len", type=int)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--enable-attention-dp",
                   action="store_true",
                   dest="enable_attention_dp")
group.add_argument("--no-enable-attention-dp",
                   action="store_false",
                   dest="enable_attention_dp")
parser.set_defaults(enable_attention_dp=None)
# Model init args
parser.add_argument("--max-num-tokens", type=int)
parser.add_argument("--moe-backend", type=str)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--use-cuda-graph",
                   action="store_true",
                   dest="use_cuda_graph")
group.add_argument("--no-use-cuda-graph",
                   action="store_false",
                   dest="use_cuda_graph")
parser.set_defaults(use_cuda_graph=None)
# Per iteration args
parser.add_argument("--batch-size", type=int)
parser.add_argument("--seq-len-q", type=int)
parser.add_argument("--seq-len-kv-cache", type=int)
parser.add_argument("--balance-method", type=str)
parser.add_argument("--balance-ratio", type=float)
args = parser.parse_args()
with open(args.config_path) as f:
    config = yaml.safe_load(f)
del args.config_path
for k, v in vars(args).items():
    if v is None:
        setattr(args, k, config[k])
print(args)

# MPI args
rank = mpi_rank()
world_size = mpi_world_size()
local_rank = local_mpi_rank()
torch.cuda.set_device(local_rank)

# Create KV cache manager
mapping = DeepSeekV3Runner.create_mapping(
    enable_attention_dp=args.enable_attention_dp)
max_batch_size = 2048
kv_cache_manager = DeepSeekV3Runner.create_kv_cache_manager(
    args.model,
    mapping,
    tokens_per_block=args.tokens_per_block,
    max_batch_size=max_batch_size,
    max_seq_len=args.max_seq_len,
    layer_indices=args.layer_indices)
attn_workspace = torch.empty((0, ), device="cuda", dtype=torch.int8)

# Create other global objects
AutoTuner.get().clear_cache()
capture_stream = torch.cuda.Stream()

# Create Runner
runner = DeepSeekV3Runner(args.model,
                          mapping,
                          moe_backend=args.moe_backend,
                          layer_indices=args.layer_indices,
                          scaled_from=args.scaled_from,
                          max_seq_len=args.max_seq_len,
                          max_num_tokens=args.max_num_tokens,
                          use_cuda_graph=args.use_cuda_graph)

# Warm up
assert args.batch_size <= max_batch_size
assert args.seq_len_q + args.seq_len_kv_cache <= args.max_seq_len
run_pack = runner.create_run_pack(args.run_type,
                                  batch_size=args.batch_size,
                                  seq_len_q=args.seq_len_q,
                                  seq_len_kv_cache=args.seq_len_kv_cache,
                                  kv_cache_manager=kv_cache_manager,
                                  attn_workspace=attn_workspace)
runner.replace_routing_method(balance_method=BalanceMethod[args.balance_method],
                              balance_ratio=args.balance_ratio)
capture_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(capture_stream):
    run_pack()
    with autotune():
        run_pack()
torch.cuda.current_stream().wait_stream(capture_stream)
torch.cuda.synchronize()

# Profile: capture graph and replay it
torch.cuda.cudart().cudaProfilerStart()
if args.use_cuda_graph:
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
    with nvtx.annotate(
            f"b={args.batch_size} s={args.seq_len_q} EP{world_size}"):
        if args.use_cuda_graph:
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
