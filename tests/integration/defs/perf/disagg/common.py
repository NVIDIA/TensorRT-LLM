"""
Disaggregated Benchmark Configuration
"""
import os

SESSION_COLLECT_CMD_TYPE = "session_collect"

# GPU resource configuration
# Simplified - only fields actually used in the codebase
GPU_RESOURCE_CONFIG = {
    # OCI GB200
    "GB200": {
        "gres_gpu": 4,                          # srun --gres parameter (None = not required)
        "lock_freq_graphics_mhz": 2062,         # GPU graphics clock lock frequency (MHz)
        "lock_freq_memory_mhz": 3996,           # GPU memory clock lock frequency (MHz)
    },
    # OCI GB300
    "GB300": {
        "gres_gpu": None,                       # GB300 does not require gres
        "lock_freq_graphics_mhz": None,         # TODO: Set GB300 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # H100
    "H100": {
        "gres_gpu": None,                       # H100 does not require gres
        "lock_freq_graphics_mhz": None,         # TODO: Set H100 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # B200
    "B200": {
        "gres_gpu": 4,
        "lock_freq_graphics_mhz": None,         # TODO: Set B200 lock frequency
        "lock_freq_memory_mhz": None,
    },
    # B300
    "B300": {
        "gres_gpu": 4,
        "lock_freq_graphics_mhz": None,         # TODO: Set B300 lock frequency
        "lock_freq_memory_mhz": None,
    },
}


class EnvManager:
    """Environment variable manager"""
    
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
        model_dir = EnvManager.get_model_dir()
        output_path = EnvManager.get_output_path()
        repo_dir = EnvManager.get_repo_dir()
        
        mounts = [
            f"{work_dir}:{work_dir}",
            f"{model_dir}:{model_dir}",
            f"{output_path}:{output_path}",
        ]
        
        # Add repo_dir if available
        if repo_dir:
            mounts.append(f"{repo_dir}:{repo_dir}")
        return ",".join(mounts)

CONFIG_BASE_DIR = os.path.join(EnvManager.get_work_dir(), "test_configs")