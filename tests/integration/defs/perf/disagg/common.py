"""Disaggregated Benchmark Configuration."""

import os

DEBUG_MODE = False
DEBUG_JOB_ID = "908390"

SESSION_COLLECT_CMD_TYPE = "session_collect"

# GPU resource configuration
# Simplified - only fields actually used in the codebase
GPU_RESOURCE_CONFIG = {
    # OCI GB200
    "GB200": {
        "gres_gpu": 4,  # srun --gres parameter (None = not required)
        "lock_freq_graphics_mhz": 2062,  # GPU graphics clock lock frequency (MHz)
        "lock_freq_memory_mhz": 3996,  # GPU memory clock lock frequency (MHz)
    },
    # OCI GB300
    "GB300": {
        "gres_gpu": None,  # GB300 does not require gres
        "lock_freq_graphics_mhz": None,  # TODO: Set GB300 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # H100
    "H100": {
        "gres_gpu": None,  # H100 does not require gres
        "lock_freq_graphics_mhz": None,  # TODO: Set H100 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # B200
    "B200": {
        "gres_gpu": 4,
        "lock_freq_graphics_mhz": None,  # TODO: Set B200 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # B300
    "B300": {
        "gres_gpu": 4,
        "lock_freq_graphics_mhz": None,  # TODO: Set B300 lock frequency
        "lock_freq_memory_mhz": None,
    },
}


class EnvManager:
    """Environment variable manager."""

    @staticmethod
    def get_gpu_type() -> str:
        return os.getenv("GPU_TYPE", "GB200")

    @staticmethod
    def get_slurm_partition() -> str:
        return os.getenv("SLURM_PARTITION", "batch")

    @staticmethod
    def get_slurm_account() -> str:
        return os.getenv("SLURM_ACCOUNT", "coreai_comparch_trtllm")

    @staticmethod
    def get_slurm_job_name() -> str:
        return os.getenv("SLURM_JOB_NAME", "unified-benchmark")

    @staticmethod
    def get_container_image() -> str:
        return os.getenv("CONTAINER_IMAGE", "")

    @staticmethod
    def get_script_dir() -> str:
        return os.getenv("SCRIPT_DIR", "/code/bench-sa/scripts")

    @staticmethod
    def get_work_dir() -> str:
        return os.getenv("WORK_DIR", "/code/bench-sa")

    @staticmethod
    def get_repo_dir() -> str:
        return os.getenv("REPO_DIR", "")

    @staticmethod
    def get_trtllm_wheel_path() -> str:
        return os.getenv("TRTLLM_WHEEL_PATH", "")

    @staticmethod
    def get_model_dir() -> str:
        return os.getenv("MODEL_DIR", "/lustre/fsw/portfolios/coreai/users/xqiao")

    @staticmethod
    def get_output_path() -> str:
        output_path = os.getenv("OUTPUT_PATH", "/home/fredricz/tensorrt-llm-bench")
        os.makedirs(output_path, exist_ok=True)
        return output_path

    @staticmethod
    def get_install_mode() -> str:
        return os.getenv("INSTALL_MODE", "none")

    @staticmethod
    def get_container_mount() -> str:
        work_dir = EnvManager.get_work_dir()
        script_dir = EnvManager.get_script_dir()
        model_dir = EnvManager.get_model_dir()
        output_path = EnvManager.get_output_path()
        repo_dir = EnvManager.get_repo_dir()
        trtllm_wheel_path = EnvManager.get_trtllm_wheel_path()

        mounts = [
            f"{work_dir}:{work_dir}",
            f"{script_dir}:{script_dir}",
            f"{model_dir}:{model_dir}",
            f"{output_path}:{output_path}",
        ]

        # Add repo_dir if available
        if repo_dir:
            mounts.append(f"{repo_dir}:{repo_dir}")
        if trtllm_wheel_path:
            trtllm_wheel_dir = os.path.dirname(trtllm_wheel_path)
            mounts.append(f"{trtllm_wheel_dir}:{trtllm_wheel_dir}")
        return ",".join(mounts)


CONFIG_BASE_DIR = os.path.join(EnvManager.get_work_dir(), "test_configs")


def extract_config_fields(config_data: dict) -> dict:
    """Extract critical fields from configuration data to generate test ID and log directory."""
    # Extract basic fields
    isl = config_data["sequence"]["input_length"]
    osl = config_data["sequence"]["output_length"]
    ctx_num = config_data["hardware"]["num_ctx_servers"]
    gen_num = config_data["hardware"]["num_gen_servers"]
    gen_tp_size = config_data["worker_config"]["gen"]["tensor_parallel_size"]
    gen_batch_size = config_data["worker_config"]["gen"]["max_batch_size"]
    gen_enable_dp = config_data["worker_config"]["gen"]["enable_attention_dp"]
    cache_transceiver_backend = config_data["worker_config"]["gen"]["cache_transceiver_config"][
        "backend"
    ]

    eplb_slots = (config_data['worker_config']['gen'].get('moe_config', {}).get(
        'load_balancer', {}).get('num_slots', 0))

    # Get MTP size
    gen_config = config_data["worker_config"]["gen"]
    mtp_size = 0
    if "speculative_config" in gen_config:
        mtp_size = gen_config["speculative_config"].get("num_nextn_predict_layers", 0)

    # Generate derived fields
    dep_flag = "dep" if gen_enable_dp else "tep"
    log_base = f"{isl}-{osl}"
    context_dir = (
        f"ctx{ctx_num}_gen{gen_num}_{dep_flag}{gen_tp_size}_"
        f"batch{gen_batch_size}_eplb{eplb_slots}_mtp{mtp_size}"
    )

    return {
        "isl": isl,
        "osl": osl,
        "ctx_num": ctx_num,
        "gen_num": gen_num,
        "gen_tp_size": gen_tp_size,
        "gen_batch_size": gen_batch_size,
        "gen_enable_dp": gen_enable_dp,
        "eplb_slots": eplb_slots,
        "mtp_size": mtp_size,
        "dep_flag": dep_flag,
        "cache_transceiver_backend": cache_transceiver_backend,
        "log_base": log_base,
        "context_dir": context_dir,
    }
