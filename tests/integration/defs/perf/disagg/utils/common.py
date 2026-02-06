"""Disaggregated Benchmark Configuration."""

import os

# GPU resource configuration
# Centralized configuration for all GPU-specific parameters
GPU_RESOURCE_CONFIG = {
    # OCI GB200
    "GB200": {
        "slurm_extra_args": "--gres=gpu:4",  # SLURM extra arguments (empty string if not required)
        "set_segment": True,
        "lock_freq_graphics_mhz": 2062,  # GPU graphics clock lock frequency (MHz)
        "lock_freq_memory_mhz": 3996,  # GPU memory clock lock frequency (MHz)
    },
    # Lyris GB200
    "GB200_LYRIS": {
        "slurm_extra_args": "",  # GB200 does not require extra args
        "set_segment": True,
        "lock_freq_graphics_mhz": None,  # TODO: Set GB200 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # Lyris GB300
    "GB300": {
        "slurm_extra_args": "",  # GB300 does not require extra args
        "set_segment": True,
        "lock_freq_graphics_mhz": None,  # TODO: Set GB300 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # H100
    "H100": {
        "slurm_extra_args": "",  # H100 does not require extra args
        "set_segment": False,
        "lock_freq_graphics_mhz": None,  # TODO: Set H100 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # B200
    "B200": {
        "slurm_extra_args": "--gres=gpu:4",
        "set_segment": False,
        "lock_freq_graphics_mhz": None,  # TODO: Set B200 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # B300
    "B300": {
        "slurm_extra_args": "--gres=gpu:4",
        "set_segment": False,
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
        return os.getenv("SLURM_PARTITION", "<You slurm partition>")

    @staticmethod
    def get_slurm_account() -> str:
        return os.getenv("SLURM_ACCOUNT", "<You slurm account>")

    @staticmethod
    def get_slurm_job_name() -> str:
        """Job name for sbatch: {SLURM_ACCOUNT}-{base} or just {base}.
        Base customizable via SLURM_JOB_BASE_NAME (default: unified.benchmark).
        """
        account = EnvManager.get_slurm_account()
        base = os.getenv("SLURM_JOB_BASE_NAME", "unified.benchmark")
        if account and not account.startswith("<"):
            return f"{account}-{base}"
        return base

    @staticmethod
    def get_slurm_set_segment() -> bool:
        """Get whether to use SLURM segment parameter based on GPU type.

        Returns:
            bool: True if GPU type requires --segment parameter, False otherwise
        """
        gpu_type = EnvManager.get_gpu_type()
        gpu_config = GPU_RESOURCE_CONFIG.get(gpu_type, {})
        return gpu_config.get("set_segment", False)

    @staticmethod
    def get_slurm_extra_args() -> str:
        """Get SLURM extra arguments based on GPU configuration.

        Returns extra SLURM arguments from GPU_RESOURCE_CONFIG.
        This allows flexible configuration of GPU-specific SLURM parameters
        like --gres, --constraint, etc.

        Returns:
            str: Extra SLURM arguments (e.g., "--gres=gpu:4" or "")

        Examples:
            GB200: "--gres=gpu:4"
            GB300: ""
            Custom: "--gres=gpu:4 --constraint=v100"
        """
        gpu_type = EnvManager.get_gpu_type()
        gpu_config = GPU_RESOURCE_CONFIG.get(gpu_type, {})
        return gpu_config.get("slurm_extra_args", "")

    @staticmethod
    def get_container_image() -> str:
        return os.getenv("CONTAINER_IMAGE", "")

    @staticmethod
    def get_script_dir() -> str:
        return os.getenv("SCRIPT_DIR", "<Your benchmark script directory>")

    @staticmethod
    def get_work_dir() -> str:
        return os.getenv("WORK_DIR", "<Your working directory>")

    @staticmethod
    def get_repo_dir() -> str:
        return os.getenv("REPO_DIR", "<Your TensorRT-LLM repository directory>")

    @staticmethod
    def get_trtllm_wheel_path() -> str:
        return os.getenv("TRTLLM_WHEEL_PATH", "<Your TensorRT-LLM wheel path>")

    @staticmethod
    def get_model_dir() -> str:
        return os.getenv("MODEL_DIR", "<Your model directory>")

    @staticmethod
    def get_dataset_dir() -> str:
        return os.getenv("DATASET_DIR", "<Your dataset directory>")

    @staticmethod
    def get_hf_home_dir() -> str:
        return os.getenv("HF_HOME_DIR", "<Your HF home directory>")

    @staticmethod
    def get_output_path() -> str:
        output_path = os.getenv(
            "OUTPUT_PATH", "<The csv and disagg comparison HTML output directory>"
        )
        # Only create directory if it's a valid path (not a placeholder)
        if output_path and not output_path.startswith("<"):
            os.makedirs(output_path, exist_ok=True)
        return output_path

    @staticmethod
    def get_install_mode() -> str:
        return os.getenv("INSTALL_MODE", "none")

    @staticmethod
    def get_container_mount(model_name: str = "") -> str:
        work_dir = EnvManager.get_work_dir()
        script_dir = EnvManager.get_script_dir()
        model_dir = EnvManager.get_model_dir()
        dataset_dir = EnvManager.get_dataset_dir()
        output_path = EnvManager.get_output_path()
        repo_dir = EnvManager.get_repo_dir()
        trtllm_wheel_path = EnvManager.get_trtllm_wheel_path()

        mounts = [
            f"{work_dir}:{work_dir}",
            f"{script_dir}:{script_dir}",
            f"{model_dir}:{model_dir}",
            f"{output_path}:{output_path}",
        ]

        # Kimi-K2 needs 640G of shared memory, otherwise will cause host memory OOM.
        if model_name.find("kimi-k2") != -1:
            mounts.append("tmpfs:/dev/shm:size=640G")

        if dataset_dir and not dataset_dir.startswith("<"):
            mounts.append(f"{dataset_dir}:{dataset_dir}")
        # Add repo_dir if available
        if repo_dir and not repo_dir.startswith("<"):
            mounts.append(f"{repo_dir}:{repo_dir}")
        if trtllm_wheel_path and not trtllm_wheel_path.startswith("<"):
            trtllm_wheel_dir = os.path.dirname(trtllm_wheel_path)
            mounts.append(f"{trtllm_wheel_dir}:{trtllm_wheel_dir}")
        return ",".join(mounts)

    @staticmethod
    def get_debug_mode() -> bool:
        return os.getenv("DEBUG_MODE", "0") == "1"

    @staticmethod
    def get_debug_job_id() -> str:
        return os.getenv("DEBUG_JOB_ID", "908390")

    # ========== CI/CD Environment Variables ==========
    @staticmethod
    def get_trtllm_branch() -> str:
        return os.getenv("TRT_LLM_BRANCH", "default")

    @staticmethod
    def get_trtllm_repo() -> str:
        return os.getenv("TRT_LLM_REPO", "NVIDIA/TensorRT-LLM")

    @staticmethod
    def get_trtllm_version() -> str:
        return os.getenv("TRT_LLM_VERSION", "default")

    @staticmethod
    def get_commit_hash() -> str:
        return os.getenv("COMMIT_HASH", "default")

    @staticmethod
    def get_commit_time() -> str:
        return os.getenv("COMMIT_TIME", "default")

    @staticmethod
    def get_docker_image() -> str:
        return os.getenv("DOCKER_IMAGE", "default")

    @staticmethod
    def get_wheel_url() -> str:
        return os.getenv("WHEEL_URL", "")

    @staticmethod
    def get_cluster_llm_data() -> str:
        return os.getenv("CLUSTER_LLM_DATA", "")


class InfoPrinter:
    """Print environment information for CI/CD and debugging."""

    @staticmethod
    def print(config_path: str = None, test_id: str = None):
        """Print environment information and optional reproduce command."""
        from utils.logger import logger

        logger.info(f"TRT_LLM_REPO:      {EnvManager.get_trtllm_repo()}")
        logger.info(f"TRT_LLM_BRANCH:    {EnvManager.get_trtllm_branch()}")   
        logger.info(f"TRT_LLM_VERSION:   {EnvManager.get_trtllm_version()}")
        logger.info(f"COMMIT_HASH:       {EnvManager.get_commit_hash()}")
        logger.info(f"COMMIT_TIME:       {EnvManager.get_commit_time()}")
        logger.info(f"DOCKER_IMAGE:      {EnvManager.get_docker_image()}")
        logger.info(f"INSTALL_MODE:      {EnvManager.get_install_mode()}")
        logger.info(f"WHEEL_URL:         {EnvManager.get_wheel_url()}")
        logger.info(f"GPU_TYPE:          {EnvManager.get_gpu_type()}")
        logger.info(f"SLURM_PARTITION:   {EnvManager.get_slurm_partition()}")
        logger.info(f"SLURM_ACCOUNT:     {EnvManager.get_slurm_account()}")
        logger.info(f"CLUSTER_LLM_DATA:  {EnvManager.get_cluster_llm_data()}")       
        if config_path and test_id:
            log_dir = os.path.join(EnvManager.get_output_path(), "slurm_logs", test_id.replace(":", "-"))
            logger.info(f"# Reproduce: python3 submit.py -c {config_path} --log-dir {log_dir}")


CONFIG_BASE_DIR = os.path.join(EnvManager.get_work_dir(), "test_configs")


def extract_config_fields(config_data: dict) -> dict:
    """Extract critical fields from configuration data to generate test ID and log directory."""
    # Extract basic fields
    isl = config_data["benchmark"]["input_length"]
    osl = config_data["benchmark"]["output_length"]
    ctx_num = config_data["hardware"]["num_ctx_servers"]
    gen_num = config_data["hardware"]["num_gen_servers"]

    ctx_max_seq_len = config_data["worker_config"]["ctx"]["max_seq_len"]
    gen_max_seq_len = config_data["worker_config"]["gen"]["max_seq_len"]
    gen_tp_size = config_data["worker_config"]["gen"]["tensor_parallel_size"]
    gen_batch_size = config_data["worker_config"]["gen"]["max_batch_size"]
    gen_enable_dp = config_data["worker_config"]["gen"]["enable_attention_dp"]
    streaming = config_data["benchmark"]["streaming"]
    concurrency_list = [
        int(x.strip()) for x in config_data["benchmark"]["concurrency_list"].split() if x.strip()
    ]
    cache_transceiver_backend = config_data["worker_config"]["gen"]["cache_transceiver_config"][
        "backend"
    ]

    gen_max_tokens = config_data["worker_config"]["gen"]["max_num_tokens"]
    gen_max_batch_size = config_data["worker_config"]["gen"]["max_batch_size"]

    eplb_slots = (
        config_data["worker_config"]["gen"]
        .get("moe_config", {})
        .get("load_balancer", {})
        .get("num_slots", 0)
    )

    # Get MTP size
    gen_config = config_data["worker_config"]["gen"]
    mtp_size = 0
    if "speculative_config" in gen_config:
        mtp_size = gen_config["speculative_config"].get("num_nextn_predict_layers", 0)

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
        "cache_transceiver_backend": cache_transceiver_backend,
        "gen_max_tokens": gen_max_tokens,
        "gen_max_batch_size": gen_max_batch_size,
        "streaming": streaming,
        "concurrency_list": concurrency_list,
        "ctx_max_seq_len": ctx_max_seq_len,
        "gen_max_seq_len": gen_max_seq_len,
    }
