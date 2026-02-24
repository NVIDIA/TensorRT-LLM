import argparse
import itertools
import json
import os
from unittest import mock

import numpy as np
import nvtx
import torch
import yaml

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._utils import local_mpi_rank, mpi_rank, mpi_world_size
from tensorrt_llm.logger import logger
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator
from tensorrt_llm.tools.layer_wise_benchmarks.mark_utils import mark_ranges
from tensorrt_llm.tools.layer_wise_benchmarks.runner import BalanceMethod, Runner


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
group = parser.add_mutually_exclusive_group()
group.add_argument("--enable-attention-dp", action="store_true", dest="enable_attention_dp")
group.add_argument("--no-enable-attention-dp", action="store_false", dest="enable_attention_dp")
parser.set_defaults(enable_attention_dp=None)
parser.add_argument("--kv-cache-dtype", type=str, choices=["fp8", "nvfp4", "auto"])
parser.add_argument(
    "--mamba-ssm-cache-dtype", type=str, choices=["auto", "float16", "bfloat16", "float32"]
)
# Model init args
parser.add_argument("--load-format", type=str, choices=["AUTO", "DUMMY"])
parser.add_argument("--max-num-tokens", type=int)
parser.add_argument("--moe-backend", type=str)
parser.add_argument(
    "--moe-backend-for-prefill", type=str, choices=["CUTLASS", "DEEPGEMM", "WIDEEP"]
)
parser.add_argument("--moe-max-num-tokens", type=int)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--use-low-precision-moe-combine", action="store_true", dest="use_low_precision_moe_combine"
)
group.add_argument(
    "--no-use-low-precision-moe-combine",
    action="store_false",
    dest="use_low_precision_moe_combine",
)
parser.set_defaults(use_low_precision_moe_combine=None)
group = parser.add_mutually_exclusive_group()
group.add_argument("--enable-autotuner", action="store_true", dest="enable_autotuner")
group.add_argument("--no-enable-autotuner", action="store_false", dest="enable_autotuner")
parser.set_defaults(enable_autotuner=None)
group = parser.add_mutually_exclusive_group()
group.add_argument("--use-cuda-graph", action="store_true", dest="use_cuda_graph")
group.add_argument("--no-use-cuda-graph", action="store_false", dest="use_cuda_graph")
parser.set_defaults(use_cuda_graph=None)
# Per iteration args
parser.add_argument("--batch-size", type=comma_separated_ints, dest="batch_size_list")
parser.add_argument("--seq-len-q", type=comma_separated_ints, dest="seq_len_q_list")
parser.add_argument("--seq-len-kv-cache", type=comma_separated_ints, dest="seq_len_kv_cache_list")
parser.add_argument("--balance-method", type=str)
parser.add_argument("--balance-ratio", type=comma_separated_floats, dest="balance_ratio_list")
# Calibration
parser.add_argument("--replay-file-path", type=str)
parser.add_argument("--replay-start-iter", type=int)
parser.add_argument("--replay-stop-iter", type=int)
group = parser.add_mutually_exclusive_group()
group.add_argument("--replay-verify-metadata", action="store_true", dest="replay_verify_metadata")
group.add_argument(
    "--no-replay-verify-metadata", action="store_false", dest="replay_verify_metadata"
)
# Schedule
parser.add_argument("--warmup-times", type=int, default=20)
parser.add_argument("--run-times", type=int, default=100)
args = parser.parse_args()
# Load YAML file
with open(args.config_path) as f:
    config = yaml.safe_load(f)
del args.config_path
for k, v in vars(args).items():
    if k.endswith("_list"):
        config_key = k[: -len("_list")]
        if v is None and config_key in config:
            v = config[config_key]
            if isinstance(v, list):
                pass
            elif v is None or isinstance(v, (int, float)):
                v = [v]
            else:
                raise ValueError(f'Config "{config_key}" in YAML should be a value or a list')
            setattr(args, k, v)
    else:
        config_key = k
        if v is None and config_key in config:
            v = config[config_key]
            setattr(args, k, v)
    if config_key in config:
        del config[config_key]
if config:
    raise ValueError(f"Config {','.join(config.keys())} from file are not options")
# Set default values
if args.max_batch_size is None:
    args.max_batch_size = max(args.batch_size_list)
if args.max_seq_len is None:
    args.max_seq_len = max(args.seq_len_q_list) + max(args.seq_len_kv_cache_list)
if args.enable_attention_dp is None:
    args.enable_attention_dp = False
if args.kv_cache_dtype is None:
    args.kv_cache_dtype = "auto"
if args.mamba_ssm_cache_dtype is None:
    args.mamba_ssm_cache_dtype = "auto"
if args.load_format is None:
    args.load_format = "DUMMY"
if args.max_num_tokens is None:
    args.max_num_tokens = args.max_batch_size * max(args.seq_len_q_list)
if args.moe_backend_for_prefill is None:
    args.moe_backend_for_prefill = "CUTLASS"
if args.use_low_precision_moe_combine is None:
    args.use_low_precision_moe_combine = False
if args.enable_autotuner is None:
    args.enable_autotuner = True
if args.use_cuda_graph is None:
    args.use_cuda_graph = False
if (args.replay_start_iter is None) != (args.replay_stop_iter is None):
    parser.error("Both --replay-start-iter and --replay-stop-iter must be provided or none")
if args.replay_verify_metadata is None:
    args.replay_verify_metadata = True
print(args)

# MPI args
rank = mpi_rank()
world_size = mpi_world_size()
local_rank = local_mpi_rank()
torch.cuda.set_device(local_rank)

# Create KV cache manager
logger.info("Layer-wise benchmarks: Create KV cache manager")
mapping = Runner.create_mapping(enable_attention_dp=args.enable_attention_dp)
kv_cache_manager = Runner.create_kv_cache_manager(
    args.model,
    mapping,
    tokens_per_block=args.tokens_per_block,
    max_batch_size=args.max_batch_size,
    max_seq_len=args.max_seq_len,
    kv_cache_dtype=args.kv_cache_dtype,
    mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
    layer_indices=args.layer_indices,
)
attn_workspace = torch.empty((0,), device="cuda", dtype=torch.int8)
logger.info("Layer-wise benchmarks: Create KV cache manager  ... Done")

# Create other global objects
AutoTuner.get().clear_cache()
capture_stream = torch.cuda.Stream()
mark_ranges()

# Create runner
logger.info("Layer-wise benchmarks: Create runner")
runner = Runner(
    args.model,
    mapping,
    load_format=args.load_format,
    moe_backend=args.moe_backend,
    layer_indices=args.layer_indices,
    scaled_from=args.scaled_from,
    max_seq_len=args.max_seq_len,
    max_num_tokens=args.max_num_tokens,
    moe_max_num_tokens=args.moe_max_num_tokens,
    kv_cache_dtype=args.kv_cache_dtype,
    mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
    use_low_precision_moe_combine=args.use_low_precision_moe_combine,
    use_cuda_graph=args.use_cuda_graph,
)
logger.info("Layer-wise benchmarks: Create runner  ... Done")

calibrator = get_calibrator()
if args.replay_file_path:
    calibrator.init(
        "REPLAY",
        args.replay_file_path,
        args.layer_indices,
        replay_verify_metadata=args.replay_verify_metadata,
        mapping=mapping,
    )
    if args.replay_start_iter is None:
        replay_start_iter, replay_stop_iter = calibrator.get_replay_iteration_range()
    else:
        replay_start_iter, replay_stop_iter = args.replay_start_iter, args.replay_stop_iter
    logger.info(
        f"Layer-wise benchmarks: Replay iteration range [{replay_start_iter}, {replay_stop_iter}]"
    )
else:
    calibrator.init("NONE", None, None)
    replay_start_iter, replay_stop_iter = 1, 1  # To avoid None in mathematics
calibrator.maybe_wrap_model(runner.model)

# Autotune
run_pack = runner.create_run_pack(
    args.run_type,
    batch_size=max(args.batch_size_list),
    request_id_begin=0,
    seq_len_q=max(args.seq_len_q_list),
    seq_len_kv_cache=args.seq_len_kv_cache_list[0],
    kv_cache_manager=kv_cache_manager,
    attn_workspace=attn_workspace,
)
if args.enable_autotuner:
    cache_path = os.getenv("TLLM_AUTOTUNER_CACHE_PATH") or None
    AutoTuner.get().setup_distributed_state(mapping)
    with autotune(cache_path=cache_path):
        run_pack()
else:
    run_pack()

# Prefill KV cache
if args.run_type == "GEN":
    logger.info("Layer-wise benchmarks: Create runner for prefill")
    ctx_seq_len_q = max(args.seq_len_kv_cache_list)
    ctx_batch_size = min(
        args.max_batch_size,
        max(1, 20480 // ctx_seq_len_q),
    )
    ctx_attn_workspace = torch.empty((0,), device="cuda", dtype=torch.int8)
    with mock.patch.dict(
        os.environ,
        {"TRTLLM_FORCE_ALLTOALL_METHOD": "NotEnabled", "TRTLLM_FORCE_COMM_METHOD": "ALLGATHER"},
        clear=False,
    ):
        ctx_runner = Runner(
            args.model,
            mapping,
            load_format=args.load_format,
            moe_backend=args.moe_backend_for_prefill,
            layer_indices=args.layer_indices,
            scaled_from=args.scaled_from,
            max_seq_len=args.max_seq_len,
            max_num_tokens=ctx_batch_size * ctx_seq_len_q,
            moe_max_num_tokens=16384,
            kv_cache_dtype=args.kv_cache_dtype,
            mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
            use_low_precision_moe_combine=args.use_low_precision_moe_combine,
            use_cuda_graph=False,
        )
    logger.info("Layer-wise benchmarks: Create runner for prefill  ... Done")

    logger.info("Layer-wise benchmarks: Prefill KV cache")
    assert ctx_batch_size <= args.max_batch_size
    assert ctx_seq_len_q + 0 <= args.max_seq_len
    num_requests = max(args.batch_size_list)
    for request_id_begin in range(0, num_requests, ctx_batch_size):
        run_pack = ctx_runner.create_run_pack(
            "CTX",
            batch_size=min(ctx_batch_size, num_requests - request_id_begin),
            request_id_begin=request_id_begin,
            seq_len_q=ctx_seq_len_q,
            seq_len_kv_cache=0,
            kv_cache_manager=kv_cache_manager,
            attn_workspace=ctx_attn_workspace,
        )
        run_pack(check=True)
    del ctx_runner
    del ctx_attn_workspace
    logger.info("Layer-wise benchmarks: Prefill KV cache  ... Done")

# Warm up
logger.info("Layer-wise benchmarks: Warmup")
for batch_size, seq_len_q, seq_len_kv_cache, balance_ratio in [
    *itertools.product(
        args.batch_size_list,
        args.seq_len_q_list,
        args.seq_len_kv_cache_list,
        args.balance_ratio_list,
    ),
]:
    assert batch_size <= args.max_batch_size
    assert seq_len_q + seq_len_kv_cache <= args.max_seq_len
    assert batch_size * seq_len_q <= args.max_num_tokens
    run_pack = runner.create_run_pack(
        args.run_type,
        batch_size=batch_size,
        request_id_begin=0,
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
            run_pack(check=True)
        torch.cuda.current_stream().wait_stream(capture_stream)
torch.cuda.synchronize()
logger.info("Layer-wise benchmarks: Warmup  ... Done")

events = [
    torch.cuda.Event(enable_timing=True) for _ in range(args.warmup_times + args.run_times + 1)
]
[e.record() for e in events]  # Explicitly warmup events because torch is lazy

torch.cuda.cudart().cudaProfilerStart()
with nvtx.annotate(f"layer_wise_benchmarks args {json.dumps(args.__dict__)}"):
    pass  # Use `annotate` instead of `mark` to avoid addition lines on the Nsight Systems UI
for batch_size, seq_len_q, seq_len_kv_cache, balance_ratio in itertools.product(
    args.batch_size_list, args.seq_len_q_list, args.seq_len_kv_cache_list, args.balance_ratio_list
):
    # Profile: capture graph and replay it
    problem_spec = {
        "batch_size": batch_size,
        "seq_len_q": seq_len_q,
        "seq_len_kv_cache": seq_len_kv_cache,
        "balance_ratio": balance_ratio,
    }
    with nvtx.annotate(f"layer_wise_benchmarks problem_spec {json.dumps(problem_spec)}"):
        pass
    run_pack = runner.create_run_pack(
        args.run_type,
        batch_size=batch_size,
        request_id_begin=0,
        seq_len_q=seq_len_q,
        seq_len_kv_cache=seq_len_kv_cache,
        kv_cache_manager=kv_cache_manager,
        attn_workspace=attn_workspace,
    )
    with runner.replace_routing_method_ctx(
        balance_method=BalanceMethod[args.balance_method], balance_ratio=balance_ratio
    ):
        run_pack()
        if args.use_cuda_graph:
            with with_multi_stream(True):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=capture_stream, capture_error_mode="global"):
                    run_pack()

        balance_ratio_str = "" if balance_ratio is None else f" balance={balance_ratio:.2g}"
        nvtx_message = f"b={batch_size} s={seq_len_q} past={seq_len_kv_cache}{balance_ratio_str} NP{world_size}"
        calibrator.start()
        for i in range(args.warmup_times + args.run_times):
            events[i].record()
            replay_iter = replay_start_iter + i % (replay_stop_iter - replay_start_iter + 1)
            calibrator.pre_step(replay_iter)
            with nvtx.annotate(nvtx_message):
                if args.use_cuda_graph:
                    g.replay()
                else:
                    run_pack()
            calibrator.post_step(replay_iter)
        events[-1].record()
        calibrator.stop()
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
