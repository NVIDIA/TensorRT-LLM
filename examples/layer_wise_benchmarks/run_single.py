import pathlib

import numpy as np
import nvtx
import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import \
    DeepseekV3Config

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._utils import local_mpi_rank, mpi_rank, mpi_world_size
from tensorrt_llm.tools.layer_wise_benchmarks.deepseekv3_runner import (
    BalanceMethod, DeepSeekV3Runner)

# MPI args
rank = mpi_rank()
world_size = mpi_world_size()
local_rank = local_mpi_rank()
torch.cuda.set_device(local_rank)

# Model definition
pretrained_config = DeepseekV3Config.from_json_file(
    pathlib.Path(__file__).parent / "config_DeepSeek-R1-FP4.json")
experts_per_rank = 8
pretrained_config.n_routed_experts = experts_per_rank * world_size
layer_indices = [5, 6]

# KV cache related args
MAX_BATCH_SIZE = 1024
MAX_SEQ_LEN = 8192 + 512
SEQ_LEN_Q = 2
MAX_NUM_TOKENS = SEQ_LEN_Q * MAX_BATCH_SIZE
enable_attention_dp = True
moe_backend = "WIDEEP"

# Create KV cache manager
mapping = DeepSeekV3Runner.create_mapping(
    enable_attention_dp=enable_attention_dp)
kv_cache_manager = DeepSeekV3Runner.create_kv_cache_manager(
    pretrained_config,
    mapping,
    kv_cache_dtype=torch.float8_e4m3fn,
    max_batch_size=MAX_BATCH_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    layer_indices=layer_indices)
attn_workspace = torch.empty((0, ), device="cuda", dtype=torch.int8)

# Create other global objects
AutoTuner.get().clear_cache()
capture_stream = torch.cuda.Stream()

# Create Runner
runner = DeepSeekV3Runner(
    pretrained_config,
    mapping,
    moe_backend=moe_backend,
    layer_indices=layer_indices,
    kv_cache_dtype=torch.float8_e4m3fn,
    max_num_tokens=MAX_NUM_TOKENS,
)

# Warm up
batch_size = 128
seq_len_kv = 2000
balance_method = BalanceMethod.Balanced
balance_ratio = 1.
run_pack = runner.create_run_pack(batch_size=batch_size,
                                  seq_len_q=SEQ_LEN_Q,
                                  seq_len_kv=seq_len_kv,
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
if runner.is_cuda_capturable():
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
            f"b={batch_size} s={SEQ_LEN_Q} #E={experts_per_rank}xEP{world_size}"
    ):
        if runner.is_cuda_capturable():
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
