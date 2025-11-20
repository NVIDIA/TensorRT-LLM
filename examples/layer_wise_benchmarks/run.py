import argparse
import itertools

import numpy as np
import nvtx
import torch
import yaml

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._utils import local_mpi_rank, mpi_rank, mpi_world_size
from tensorrt_llm.tools.layer_wise_benchmarks import BalanceMethod, get_runner_cls


def comma_separated_ints(s):
    return [int(x) for x in s.split(",")]


def comma_separated_floats(s):
    return [float(x) for x in s.split(",")]


# Parse cmdline
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str)
parser.add_argument("--model", type=str, help="Pretrained model name or path")
parser.add_argument(
    "--layer-indices",
    type=comma_separated_ints,
    help="Comma separated indices of layers, should be a contiguous range",
)
parser.add_argument("--run-type", type=str, choices=["CTX", "GEN"])
parser.add_argument("--scaled-from", type=int)
# KV cache related args
parser.add_argument("--max-batch-size", type=int)
parser.add_argument("--tokens-per-block", type=int)
parser.add_argument("--max-seq-len", type=int)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--enable-attention-dp", action="store_true", dest="enable_attention_dp")
group.add_argument("--no-enable-attention-dp", action="store_false", dest="enable_attention_dp")
parser.set_defaults(enable_attention_dp=None)
# Model init args
parser.add_argument("--max-num-tokens", type=int)
parser.add_argument("--moe-backend", type=str)
parser.add_argument("--moe-max-num-tokens", type=int)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--use-cuda-graph", action="store_true", dest="use_cuda_graph")
group.add_argument("--no-use-cuda-graph", action="store_false", dest="use_cuda_graph")
parser.set_defaults(use_cuda_graph=None)
# Per iteration args
parser.add_argument("--batch-size", type=int)
parser.add_argument("--seq-len-q", type=int)
parser.add_argument("--seq-len-kv-cache", type=int)
parser.add_argument("--balance-method", type=str)
parser.add_argument("--balance-ratio", type=float)
# Batched run args
parser.add_argument("--batch-size-list", type=comma_separated_ints)
parser.add_argument("--seq-len-q-list", type=comma_separated_ints)
parser.add_argument("--seq-len-kv-cache-list", type=comma_separated_ints)
parser.add_argument("--balance-ratio-list", type=comma_separated_floats)
# Schedule
parser.add_argument("--warmup-times", type=int, default=20)
parser.add_argument("--run-times", type=int, default=100)
args = parser.parse_args()
# Load YAML file
with open(args.config_path) as f:
    config = yaml.safe_load(f)
del args.config_path
for k, v in vars(args).items():
    if v is None and k in config:
        setattr(args, k, config[k])
# Set list arguments
if args.batch_size_list is None:
    args.batch_size_list = [args.batch_size]
del args.batch_size
if args.seq_len_q_list is None:
    args.seq_len_q_list = [args.seq_len_q]
del args.seq_len_q
if args.seq_len_kv_cache_list is None:
    args.seq_len_kv_cache_list = [args.seq_len_kv_cache]
del args.seq_len_kv_cache
if args.balance_ratio_list is None:
    args.balance_ratio_list = [args.balance_ratio]
del args.balance_ratio
# Set default values
if args.max_batch_size is None:
    args.max_batch_size = max(args.batch_size_list)
if args.max_num_tokens is None:
    args.max_num_tokens = args.max_batch_size * max(args.seq_len_q_list)
print(args)

# MPI args
rank = mpi_rank()
world_size = mpi_world_size()
local_rank = local_mpi_rank()
torch.cuda.set_device(local_rank)

# Create KV cache manager
Runner = get_runner_cls(args.model)
mapping = Runner.create_mapping(enable_attention_dp=args.enable_attention_dp)
kv_cache_manager = Runner.create_kv_cache_manager(
    args.model,
    mapping,
    tokens_per_block=args.tokens_per_block,
    max_batch_size=args.max_batch_size,
    max_seq_len=args.max_seq_len,
    layer_indices=args.layer_indices,
)
attn_workspace = torch.empty((0,), device="cuda", dtype=torch.int8)

# Create other global objects
AutoTuner.get().clear_cache()
capture_stream = torch.cuda.Stream()

# Create Runner
runner = Runner(
    args.model,
    mapping,
    moe_backend=args.moe_backend,
    layer_indices=args.layer_indices,
    scaled_from=args.scaled_from,
    max_seq_len=args.max_seq_len,
    max_num_tokens=args.max_num_tokens,
    moe_max_num_tokens=args.moe_max_num_tokens,
    use_cuda_graph=args.use_cuda_graph,
)

# Warm up
for autotune_flag, batch_size, seq_len_q, seq_len_kv_cache, balance_ratio in [
    [
        True,
        max(args.batch_size_list),
        max(args.seq_len_q_list),
        args.seq_len_kv_cache_list[0],
        args.balance_ratio_list[0],
    ],
    *itertools.product(
        [False],
        args.batch_size_list,
        args.seq_len_q_list,
        args.seq_len_kv_cache_list,
        args.balance_ratio_list,
    ),
]:
    assert batch_size <= args.max_batch_size
    assert seq_len_q + seq_len_kv_cache <= args.max_seq_len
    run_pack = runner.create_run_pack(
        args.run_type,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv_cache=seq_len_kv_cache,
        kv_cache_manager=kv_cache_manager,
        attn_workspace=attn_workspace,
    )
    with runner.replace_routing_method_ctx(
        balance_method=BalanceMethod[args.balance_method], balance_ratio=balance_ratio
    ):
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            if autotune_flag:
                with autotune():
                    run_pack()
            run_pack()
        torch.cuda.current_stream().wait_stream(capture_stream)
torch.cuda.synchronize()

events = [
    torch.cuda.Event(enable_timing=True) for _ in range(args.warmup_times + args.run_times + 1)
]
[e.record() for e in events]  # Explicitly warmup events because torch is lazy

torch.cuda.cudart().cudaProfilerStart()
for batch_size, seq_len_q, seq_len_kv_cache, balance_ratio in itertools.product(
    args.batch_size_list, args.seq_len_q_list, args.seq_len_kv_cache_list, args.balance_ratio_list
):
    # Profile: capture graph and replay it
    run_pack = runner.create_run_pack(
        args.run_type,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv_cache=seq_len_kv_cache,
        kv_cache_manager=kv_cache_manager,
        attn_workspace=attn_workspace,
    )
    with runner.replace_routing_method_ctx(
        balance_method=BalanceMethod[args.balance_method], balance_ratio=balance_ratio
    ):
        if args.use_cuda_graph:
            with with_multi_stream(True):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=capture_stream, capture_error_mode="global"):
                    run_pack()

        balance_ratio_str = "" if balance_ratio is None else f"  balance={balance_ratio:.2g}"
        nvtx_message = f"b={batch_size} s={seq_len_q} past={seq_len_kv_cache}{balance_ratio_str} EP{world_size}"
        for i in range(args.warmup_times + args.run_times):
            events[i].record()
            with nvtx.annotate(nvtx_message):
                if args.use_cuda_graph:
                    g.replay()
                else:
                    run_pack()
        events[-1].record()
    torch.cuda.synchronize()

    # Print statistics
    #   Print before `cudaProfilerStop` to ensure messages are included in the profile
    time_list = [start.elapsed_time(stop) for start, stop in zip(events, events[1:])]
    time_list = time_list[args.warmup_times :]
    print(
        f"[RANK {rank}]"
        f"  batch_size {batch_size}"
        f"  seq_len_q {seq_len_q}"
        f"  seq_len_kv_cache {seq_len_kv_cache}"
        + ("" if balance_ratio is None else f"  balance_ratio {balance_ratio:.2g}")
        + f"  mean {np.mean(time_list) * 1000:.1f}"
        f"  median {np.median(time_list) * 1000:.1f}"
        f"  min {np.min(time_list) * 1000:.1f}"
        f"  max {np.max(time_list) * 1000:.1f}"
        f"  P90 {np.percentile(time_list, 90) * 1000:.1f}"
        f"  (us)"
    )
torch.cuda.cudart().cudaProfilerStop()
