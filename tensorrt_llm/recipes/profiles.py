"""Profile implementations for different model configurations.

Each profile encapsulates the mapping logic from high-level scenario constraints
(ISL, OSL, TP, CONC) to low-level TensorRT-LLM configuration parameters
(EP_SIZE, MOE_BACKEND, DP_ATTENTION, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


def compute_max_num_tokens(conc: int, isl: int) -> int:
    """Compute MAX_NUM_TOKENS using the formula from InferenceMax scripts.

    Formula: ((CONC + ISL + 64 + 63) / 64) * 64
    This rounds up to the nearest multiple of 64.
    """
    return ((conc + isl + 64 + 63) // 64) * 64


class ProfileBase(ABC):
    """Base class for configuration profiles."""

    @abstractmethod
    def compute_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Compute configuration from scenario parameters.

        Args:
            scenario: Dictionary containing:
                - target_isl: Input sequence length
                - target_osl: Output sequence length
                - target_concurrency: Target concurrency
                - tp_size: Tensor parallelism size
                - num_gpus: Number of GPUs (optional, used if tp_size not set)

        Returns:
            Dictionary with 'config' and 'env' keys containing the computed values.
        """

    @abstractmethod
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for this profile."""

    def _get_tp_size(self, scenario: Dict[str, Any]) -> int:
        """Get TP size from scenario, defaulting to num_gpus if not specified."""
        return scenario.get("tp_size", scenario.get("num_gpus", 1))


class DSR1FP4Profile(ProfileBase):
    """DeepSeek-R1 FP4 profile based on dsr1_fp4_b200_trt_slurm.sh logic."""

    def get_defaults(self) -> Dict[str, Any]:
        """Default configuration for DSR1-FP4."""
        return {
            "cuda_graph_config": {
                "enable_padding": True,
                "max_batch_size": 512,
            },
            "kv_cache_config": {
                "dtype": "fp8",
                "free_gpu_memory_fraction": 0.8,
                "enable_block_reuse": False,
            },
            "print_iter_log": True,
            "stream_interval": 10,
        }

    def compute_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Compute configuration based on DSR1-FP4 mapping rules.

        Logic from dsr1_fp4_b200_trt_slurm.sh lines 23-76:
        - Complex EP_SIZE logic depending on TP, ISL, OSL, CONC
        - MOE_BACKEND: TRTLLM or CUTLASS
        - DP_ATTENTION: complex conditional based on all params
        """
        isl = scenario["target_isl"]
        osl = scenario["target_osl"]
        conc = scenario["target_concurrency"]
        tp = self._get_tp_size(scenario)

        # Default values
        ep_size = 1
        moe_backend = "TRTLLM"
        dp_attention = False

        # TP-specific logic
        if tp == 4:
            if isl == 1024 and osl == 1024:
                if conc > 32:
                    ep_size = tp
                if conc >= 256:
                    dp_attention = True
                    moe_backend = "CUTLASS"
            elif isl == 1024 and osl == 8192:
                if conc > 32:
                    ep_size = tp
                if conc >= 256:
                    dp_attention = True
                    moe_backend = "CUTLASS"
            elif isl == 8192 and osl == 1024:
                if conc > 32:
                    ep_size = tp
                    dp_attention = True
                    moe_backend = "CUTLASS"
        elif tp == 8:
            if isl == 1024 and osl == 1024:
                if conc > 8:
                    ep_size = tp
                if conc >= 256:
                    dp_attention = True
                    moe_backend = "CUTLASS"
            elif isl == 1024 and osl == 8192:
                if conc > 16:
                    ep_size = tp
                if conc >= 256:
                    dp_attention = True
                    moe_backend = "CUTLASS"
            elif isl == 8192 and osl == 1024:
                if conc > 32:
                    ep_size = tp
                    dp_attention = True
                    moe_backend = "CUTLASS"

        # Build configuration
        config = self.get_defaults()
        config["enable_attention_dp"] = dp_attention
        config["moe_config"] = {"backend": moe_backend}

        # Add attention_dp_config if DP is enabled
        if dp_attention:
            config["attention_dp_config"] = {
                "batching_wait_iters": 0,
                "enable_balance": True,
                "timeout_iters": 60,
            }

        return {
            "config": config,
            "env": {},
            "cli_args": {
                "ep_size": ep_size,
                "tp_size": tp,
                "max_num_tokens": compute_max_num_tokens(conc, isl),
            },
        }


class DSR1FP8Profile(ProfileBase):
    """DeepSeek-R1 FP8 profile based on dsr1_fp8_b200_trt_slurm.sh logic."""

    def get_defaults(self) -> Dict[str, Any]:
        """Default configuration for DSR1-FP8."""
        return {
            "cuda_graph_config": {
                "enable_padding": True,
                "max_batch_size": 256,
            },
            "kv_cache_config": {
                "dtype": "fp8",
                "free_gpu_memory_fraction": 0.8,
                "enable_block_reuse": False,
            },
            "print_iter_log": True,
            "stream_interval": 10,
        }

    def compute_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Compute configuration based on DSR1-FP8 mapping rules.

        Logic from dsr1_fp8_b200_trt_slurm.sh lines 23-70:
        - EP_SIZE: always equals TP
        - MOE_BACKEND: DEEPGEMM
        - DP_ATTENTION: simpler ISL/OSL/CONC rules
        """
        isl = scenario["target_isl"]
        osl = scenario["target_osl"]
        conc = scenario["target_concurrency"]
        tp = self._get_tp_size(scenario)

        # EP_SIZE always equals TP for FP8
        ep_size = tp
        moe_backend = "DEEPGEMM"
        dp_attention = False

        # Simplified DP_ATTENTION logic
        if isl == 1024 and osl == 1024:
            if conc > 32:
                dp_attention = True
        elif isl == 1024 and osl == 8192:
            if conc > 64:
                dp_attention = True
        elif isl == 8192 and osl == 1024:
            if conc > 64:
                dp_attention = True

        # Build configuration
        config = self.get_defaults()
        config["enable_attention_dp"] = dp_attention
        config["moe_config"] = {"backend": moe_backend}

        # Add attention_dp_config if DP is enabled
        if dp_attention:
            config["attention_dp_config"] = {
                "batching_wait_iters": 0,
                "enable_balance": True,
                "timeout_iters": 60,
            }

        return {
            "config": config,
            "env": {},
            "cli_args": {
                "ep_size": ep_size,
                "tp_size": tp,
                "max_num_tokens": compute_max_num_tokens(conc, isl),
            },
        }


class GPTOSSFP4Profile(ProfileBase):
    """GPT-OSS FP4 profile based on gptoss_fp4_b200_trt_slurm.sh logic."""

    def get_defaults(self) -> Dict[str, Any]:
        """Default configuration for GPT-OSS-FP4."""
        return {
            "cuda_graph_config": {
                "enable_padding": True,
                # max_batch_size is set dynamically to CONC
            },
            "kv_cache_config": {
                "dtype": "fp8",
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.85,
            },
            "print_iter_log": True,
            "stream_interval": 20,
            "num_postprocess_workers": 4,
        }

    def compute_config(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Compute configuration based on GPT-OSS-FP4 mapping rules.

        Logic from gptoss_fp4_b200_trt_slurm.sh lines 28-68:
        - EP_SIZE: 1 or TP based on CONC >= 256
        - MOE_BACKEND: always TRTLLM
        - DP_ATTENTION: true if CONC >= 256
        - Special: max_batch_size = CONC
        """
        conc = scenario["target_concurrency"]
        scenario["target_isl"]
        tp = self._get_tp_size(scenario)

        # Simple concurrency-based logic
        ep_size = 1
        dp_attention = False

        if conc >= 256:
            ep_size = tp
            dp_attention = True

        moe_backend = "TRTLLM"

        # Build configuration
        config = self.get_defaults()
        config["cuda_graph_config"]["max_batch_size"] = conc
        config["enable_attention_dp"] = dp_attention
        config["moe_config"] = {"backend": moe_backend}

        # Add attention_dp_config if DP is enabled
        if dp_attention:
            config["attention_dp_config"] = {
                "enable_balance": True,
            }

        # Environment variables specific to GPT-OSS
        env = {
            "TRTLLM_ENABLE_PDL": "1",
            "NCCL_GRAPH_REGISTER": "0",
        }

        return {
            "config": config,
            "env": env,
            "cli_args": {
                "ep_size": ep_size,
                "tp_size": tp,
                "max_num_tokens": 20000,  # Fixed value from the script
                "max_batch_size": 512,  # Fixed value from the script
            },
        }


# Profile registry for easy lookup
PROFILE_REGISTRY: Dict[str, type[ProfileBase]] = {
    "dsr1-fp4": DSR1FP4Profile,
    "dsr1-fp8": DSR1FP8Profile,
    "gptoss-fp4": GPTOSSFP4Profile,
}


def get_profile(profile_name: str) -> ProfileBase:
    """Get a profile instance by name.

    Args:
        profile_name: Name of the profile (e.g., 'dsr1-fp4')

    Returns:
        Instance of the profile class

    Raises:
        ValueError: If profile name is not found in registry
    """
    if profile_name not in PROFILE_REGISTRY:
        available = ", ".join(PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {available}")
    return PROFILE_REGISTRY[profile_name]()


def register_profile(name: str, profile_class: type[ProfileBase]) -> None:
    """Register a custom profile (for plugin architecture).

    Args:
        name: Name to register the profile under
        profile_class: Profile class (must inherit from ProfileBase)
    """
    if not issubclass(profile_class, ProfileBase):
        raise TypeError("Profile class must inherit from ProfileBase")
    PROFILE_REGISTRY[name] = profile_class
