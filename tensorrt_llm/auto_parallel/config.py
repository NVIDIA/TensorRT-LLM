from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm._utils import DictConversion
from tensorrt_llm.logger import logger

from .utils import BaseEnum


@dataclass
class MathThroughput(DictConversion):
    int4: int = 0  # Tflops
    int8: int = 0  # Tflops
    fp8: int = 0  # Tflops
    float16: int = 0  # Tflops
    bfloat16: int = 0  # Tflops
    float32: int = 0  # Tflops


@dataclass
class ClusterInfo(DictConversion):
    inter_node_bw_per_device: int = 25  # GBps
    intra_node_bw_per_device: int = 0  # GBps
    inter_node_latency: int = 10  # us
    intra_node_latency: int = 10  # us
    intra_node_sharp: bool = False
    inter_node_sharp: bool = True

    memory_bw: int = 0  # GBps
    memory_budget_per_device: int = 0  # GB

    math_throughput: MathThroughput = field(default_factory=MathThroughput)

    memory_efficiency: float = 1.0
    math_efficiency: float = 1.0
    communication_efficiency: float = 1.0


_math_throughputs = {
    "A100": MathThroughput(
        int8=624,
        float16=312,
        bfloat16=312,
        float32=156,
    ),
}

_bandwidths = {
    "PCIe-3": 16,
    "PCIe-4": 32,
    "PCIe-5": 64,
}

_cluster_infos = {
    # from https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "A100-SXM-80GB":
    ClusterInfo(
        intra_node_bw_per_device=300,
        memory_bw=2039,
        memory_budget_per_device=80,
        math_throughput=_math_throughputs["A100"],
    ),
    "A100-SXM-40GB":
    ClusterInfo(
        intra_node_bw_per_device=300,
        memory_bw=1555,
        memory_budget_per_device=40,
        math_throughput=_math_throughputs["A100"],
    ),
    "A100-PCIe-80GB":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=1935,
        memory_budget_per_device=80,
        math_throughput=_math_throughputs["A100"],
    ),
    "A100-PCIe-40GB":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=1555,
        memory_budget_per_device=40,
        math_throughput=_math_throughputs["A100"],
    ),
    # from https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    "H100-SXM":
    ClusterInfo(
        inter_node_bw_per_device=50,
        intra_node_bw_per_device=450,
        intra_node_sharp=True,
        memory_bw=3350,
        memory_budget_per_device=80,
        math_throughput=MathThroughput(
            int8=1979,
            fp8=1979,
            float16=989,
            bfloat16=989,
            float32=495,
        ),
    ),
    "H100-PCIe":
    ClusterInfo(
        inter_node_bw_per_device=50,
        intra_node_bw_per_device=_bandwidths["PCIe-5"],
        memory_bw=2000,
        memory_budget_per_device=80,
        math_throughput=MathThroughput(
            int8=1513,
            fp8=1513,
            float16=756,
            bfloat16=756,
            float32=378,
        ),
    ),
    # from https://images.nvidia.cn/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "V100-PCIe-16GB":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-3"],
        memory_bw=900,
        memory_budget_per_device=16,
        math_throughput=MathThroughput(float32=112),
    ),
    "V100-PCIe-32GB":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-3"],
        memory_bw=900,
        memory_budget_per_device=32,
        math_throughput=MathThroughput(float32=112),
    ),
    "V100-SMX-16GB":
    ClusterInfo(
        intra_node_bw_per_device=150,
        memory_bw=900,
        memory_budget_per_device=16,
        math_throughput=MathThroughput(float32=125),
    ),
    "V100-SMX-32GB":
    ClusterInfo(
        intra_node_bw_per_device=150,
        memory_bw=900,
        memory_budget_per_device=32,
        math_throughput=MathThroughput(float32=125),
    ),
    "V100S-PCIe":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-3"],
        memory_bw=1134,
        memory_budget_per_device=32,
        math_throughput=MathThroughput(float32=130),
    ),
    # from https://images.nvidia.cn/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
    "A40":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=696,
        memory_budget_per_device=48,
        math_throughput=MathThroughput(
            int4=600,
            int8=300,
            float16=150,
            bfloat16=150,
            float32=75,
        ),
    ),
    # from https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf
    "A30":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=933,
        memory_budget_per_device=24,
        math_throughput=MathThroughput(
            int4=661,
            int8=330,
            float16=165,
            bfloat16=165,
            float32=82,
        ),
    ),
    # from https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/datasheet-new/nvidia-a10-datasheet.pdf
    "A10":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=600,
        memory_budget_per_device=24,
        math_throughput=MathThroughput(
            int4=500,
            int8=250,
            float16=125,
            bfloat16=125,
            float32=62.5,
        ),
    ),
    "A10G":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=600,
        memory_budget_per_device=24,
        math_throughput=MathThroughput(
            int4=280,
            int8=140,
            float16=70,
            bfloat16=70,
            float32=35,
        ),
    ),
    # from https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413
    "L40S":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=864,
        memory_budget_per_device=48,
        math_throughput=MathThroughput(
            int4=733,
            int8=733,
            fp8=733,
            float16=362,
            bfloat16=362,
            float32=183,
        ),
    ),
    # from https://images.nvidia.cn/content/Solutions/data-center/vgpu-L40-datasheet.pdf
    "L40":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=864,
        memory_budget_per_device=48,
        math_throughput=MathThroughput(
            int4=724,
            int8=362,
            fp8=362,
            float16=181,
            bfloat16=181,
            float32=90,
        ),
    ),
    # from https://nvdam.widen.net/s/rvq98gbwsw/l4-datasheet-2595652
    "L4":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=300,
        memory_budget_per_device=24,
        math_throughput=MathThroughput(
            int8=242,
            fp8=242,
            float16=120,
            bfloat16=120,
            float32=60,
        ),
    ),
}


def infer_cluster_key() -> str:

    def is_sxm():
        return "SXM" in device_name

    def is_80gb():
        return "80GB" in device_name

    def is_32gb():
        return "32GB" in device_name

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    if "A100" in device_name:
        if is_sxm():
            if is_80gb():
                return "A100-SXM-80GB"
            else:
                return "A100-SXM-40GB"
        else:
            if is_80gb():
                return "A100-PCIe-80GB"
            else:
                return "A100-PCIe-40GB"
    elif "A10G" in device_name:
        return "A10G"
    elif "A10" in device_name:
        return "A10"
    elif "A30" in device_name:
        return "A30"
    elif "A40" in device_name:
        return "A40"
    elif "H100" in device_name:
        if is_sxm():
            return "H100-SXM"
        else:
            return "H100-PCIe"
    elif "L40S" in device_name:
        return "L40S"
    elif "L40" in device_name:
        return "L40"
    elif "L4" in device_name:
        return "L4"
    elif "V100S" in device_name:
        return "V100S-PCIe"
    elif "V100" in device_name:
        if is_sxm():
            if is_32gb():
                return "V100-SXM-32GB"
            else:
                return "V100-SXM-16GB"
        else:
            if is_32gb():
                return "V100-PCIe-32GB"
            else:
                return "V100-PCIe-16GB"

    fallback_key = "A100-SXM-80GB"
    logger.warning(
        f"Fail to infer cluster key, use {fallback_key} as fallback.")
    return fallback_key


class CostModel(str, BaseEnum):
    ALPHA_BETA = "alpha_beta"
    PROFILE = "profile"
    S_CURVE = "s_curve"
    # Zero cost model is for test purpose.
    # Use zero cost model for communication will make solver prefer sharding
    # Use zero cost model for computation will make solver prefer replication
    ZERO = "zero"


@dataclass
class AutoParallelConfig(DictConversion):
    # cluster configuration
    world_size: int = 1
    gpus_per_node: int = 8
    cluster_key: str = None
    cluster_info: Optional[ClusterInfo] = None

    # cost model configuration
    sharding_cost_model: str = CostModel.ALPHA_BETA
    comm_cost_model: str = CostModel.ALPHA_BETA

    # strategy configuration
    enable_pipeline_parallelism: bool = False
    enable_shard_unbalanced_shape: bool = False
    enable_shard_dynamic_shape: bool = False
    enable_reduce_scatter: bool = True

    # parallelization configuration
    builder_flags: Optional[int] = None
    debug_mode: bool = False
    infer_shape: bool = True
    validation_mode: bool = False
    same_buffer_io: Dict[str, str] = field(default_factory=dict)
    same_spec_io: Dict[str, str] = field(default_factory=dict)
    sharded_io_allowlist: List[str] = field(default_factory=list)
    fast_reduce: bool = True
    fill_weights: bool = False

    # debug configuration
    parallel_config_cache: Optional[str] = None
    profile_cache: Optional[str] = None
    dump_path: Optional[str] = None
    debug_outputs: Union[List[str], str] = field(default_factory=list)

    def get_cluster_info(self) -> ClusterInfo:
        return self.cluster_info or _cluster_infos[self.cluster_key]

    @property
    def enabled(self) -> bool:
        return self.world_size > 1
