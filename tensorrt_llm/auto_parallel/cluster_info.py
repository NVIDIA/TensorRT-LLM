import copy
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import pynvml
import torch
from cuda import cudart

from tensorrt_llm._utils import DictConversion
from tensorrt_llm.logger import logger
from tensorrt_llm.profiler import PyNVMLContext, _device_get_memory_info_fn


@dataclass
class MathThroughput(DictConversion):
    int4: int = 0  # Tflops
    int8: int = 0  # Tflops
    fp8: int = 0  # Tflops
    float16: int = 0  # Tflops
    bfloat16: int = 0  # Tflops
    float32: int = 0  # Tflops

    @staticmethod
    def to_tflops(
        ipc_per_sm: "MathThroughput",
        sm_count: int,
        clock_mhz: int,
    ) -> "MathThroughput":
        tflops = MathThroughput()
        for name in ipc_per_sm.__dataclass_fields__:
            setattr(
                tflops, name,
                getattr(ipc_per_sm, name) * sm_count * clock_mhz // int(1e6))
        return tflops


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

cluster_infos = {
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
    "H20":
    ClusterInfo(
        inter_node_bw_per_device=50,
        intra_node_bw_per_device=450,
        memory_bw=4000,
        memory_budget_per_device=96,
        math_throughput=MathThroughput(
            int8=293,
            fp8=293,
            float16=147,
            bfloat16=147,
            float32=74,
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
    "V100-SXM-16GB":
    ClusterInfo(
        intra_node_bw_per_device=150,
        memory_bw=900,
        memory_budget_per_device=16,
        math_throughput=MathThroughput(float32=125),
    ),
    "V100-SXM-32GB":
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
    "L20":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=864,
        memory_budget_per_device=48,
        math_throughput=MathThroughput(
            int8=238,
            fp8=238,
            float16=119,
            bfloat16=119,
            float32=60,
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
    "L2":
    ClusterInfo(
        intra_node_bw_per_device=_bandwidths["PCIe-4"],
        memory_bw=300,
        memory_budget_per_device=24,
        math_throughput=MathThroughput(
            int8=193,
            fp8=193,
            float16=97,
            bfloat16=97,
            float32=48,
        ),
    ),
}


def infer_cluster_key() -> str:

    def match(product, name):
        # Use A100 as example, the regex pattern matches for:
        # - NVIDIA A100 80GB
        # - NVIDIA A100-PCIE
        # - NVIDIA A100
        # And does not match A1000 etc.
        return re.match(f".*{product}([ -]|$).*", name) is not None

    def is_sxm():
        return "SXM" in device_name

    def is_80gb():
        return "80GB" in device_name

    def is_32gb():
        return "32GB" in device_name

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    if match("A100", device_name):
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
    elif match("A10G", device_name):
        return "A10G"
    elif match("A10", device_name):
        return "A10"
    elif match("A30", device_name):
        return "A30"
    elif match("A40", device_name):
        return "A40"
    elif match("H100", device_name):
        if is_sxm():
            return "H100-SXM"
        else:
            return "H100-PCIe"
    elif match("L40S", device_name):
        return "L40S"
    elif match("L40", device_name):
        return "L40"
    elif match("L4", device_name):
        return "L4"
    elif match("V100S", device_name):
        return "V100S-PCIe"
    elif match("V100", device_name):
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
    return None


def ipc_per_sm(compute_cap: Tuple[int, int]) -> MathThroughput:
    ipc_table = {
        (9, 0):
        MathThroughput(
            int8=16384,
            fp8=16384,
            float16=8192,
            bfloat16=8192,
            float32=4096,
        ),
        (8, 0):
        MathThroughput(
            int4=8192,
            int8=4096,
            float16=2048,
            bfloat16=2048,
            float32=1024,
        ),
        (8, 6):
        MathThroughput(
            int4=4096,
            int8=2048,
            float16=1024,
            bfloat16=1024,
            float32=512,
        ),
        (8, 9):
        MathThroughput(
            int4=2048,
            int8=1024,
            fp8=1024,
            float16=512,
            bfloat16=512,
            float32=256,
        ),
        (7, 0):
        MathThroughput(
            float16=1024,
            float32=128,
        ),
        (7, 5):
        MathThroughput(
            int4=4096,
            int8=2048,
            float16=1024,
            float32=128,
        ),
    }
    return ipc_table.get(compute_cap, MathThroughput())


def nvlink_version(version_enum: int) -> int:
    nvl_version_table = {
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
    }
    return nvl_version_table[version_enum]


def nvlink_bandwidth(nvlink_version: int) -> int:
    nvl_bw_table = {
        1: 80,
        2: 150,
        3: 300,
        4: 450,
    }
    return nvl_bw_table[nvlink_version]


def infer_cluster_info() -> ClusterInfo:
    device = torch.cuda.current_device()
    index = device.index if isinstance(device, torch.device) else device
    with PyNVMLContext():
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        logger.info(f"Compute capability: {compute_cap}")
        err, properties = cudart.cudaGetDeviceProperties(index)
        sm_count = properties.multiProcessorCount
        logger.info(f"SM count: {sm_count}")
        sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(
            handle,
            pynvml.NVML_CLOCK_SM,
        )
        logger.info(f"SM clock: {sm_clock} MHz")
        math_throughput = MathThroughput.to_tflops(
            ipc_per_sm(compute_cap),
            sm_count,
            sm_clock,
        )
        for name in math_throughput.__dataclass_fields__:
            tflops = getattr(math_throughput, name)
            logger.info(f"{name} TFLOPS: {tflops}")

        mem_info = _device_get_memory_info_fn(handle)
        memory_budget = mem_info.total // (1024**3)
        logger.info(f"Total Memory: {memory_budget} GiB")

        mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(
            handle,
            pynvml.NVML_CLOCK_MEM,
        )
        logger.info(f"Memory clock: {mem_clock} MHz")
        if pynvml.__version__ < '11.5.0':
            mem_bus_width = properties.memoryBusWidth
        else:
            mem_bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        logger.info(f"Memory bus width: {mem_bus_width}")
        memory_bw = mem_bus_width * mem_clock * 2 // int(8e3)
        logger.info(f"Memory bandwidth: {memory_bw} GB/s")

        try:
            is_nvl_active = bool(pynvml.nvmlDeviceGetNvLinkState(handle, 0))
            logger.info(f"NVLink is active: {is_nvl_active}")
        except pynvml.NVMLError:
            is_nvl_active = False

        intra_node_sharp = False
        if is_nvl_active:
            nvl_version_enum = pynvml.nvmlDeviceGetNvLinkVersion(handle, 0)
            nvl_version = nvlink_version(nvl_version_enum)
            logger.info(f"NVLink version: {nvl_version}")
            nvl_bw = nvlink_bandwidth(nvl_version)
            logger.info(f"NVLink bandwidth: {nvl_bw} GB/s")
            intra_node_bw = nvl_bw
            if nvl_version >= 4:
                intra_node_sharp = True
        else:
            if pynvml.__version__ < '11.5.0':
                pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
                pcie_speed = (2**pcie_gen) * 1000
            else:
                pcie_speed = pynvml.nvmlDeviceGetPcieSpeed(handle)
            logger.info(f"PCIe speed: {pcie_speed} Mbps")
            pcie_link_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            logger.info(f"PCIe link width: {pcie_link_width}")
            pcie_bw = pcie_speed * pcie_link_width // int(8e3)
            logger.info(f"PCIe bandwidth: {pcie_bw} GB/s")
            intra_node_bw = pcie_bw

        cluster_info = ClusterInfo(
            math_throughput=math_throughput,
            memory_bw=memory_bw,
            memory_budget_per_device=memory_budget,
            intra_node_bw_per_device=intra_node_bw,
            intra_node_sharp=intra_node_sharp,
        )
    return cluster_info


def infer_cluster_config() -> Dict[str, Union[str, ClusterInfo]]:
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    cluster_key = infer_cluster_key()
    if cluster_key is not None:
        return dict(cluster_key=cluster_key)
    else:
        try:
            cluster_info = infer_cluster_info()
        except pynvml.NVMLError:
            fallback_cluster_key = "L40"
            cluster_info = copy.copy(cluster_infos[fallback_cluster_key])
            memory_budget = torch.cuda.mem_get_info()[1] // (1024**3)
            cluster_info.memory_budget_per_device = memory_budget
            logger.warning(
                f"Failed to infer cluster info for {device_name}, "
                f"treat it as a {fallback_cluster_key} node with {memory_budget} GB memory. "
                "This setting makes no effect if you do not use auto parallel.")
        return dict(
            cluster_key=device_name.replace(" ", "-"),
            cluster_info=cluster_info,
        )


if __name__ == "__main__":
    logger.set_level("info")
    infer_cluster_info()
