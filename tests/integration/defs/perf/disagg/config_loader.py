"""YAML Configuration Loader with Default Metrics Support."""

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from common import EnvManager, extract_config_fields


@dataclass
class MetricsConfig:
    """Metrics configuration."""

    log_file: str  # Log file name
    extractor_pattern: str  # Regular expression
    metric_names: List[str]  # Metric names list

    def merge(self, override: Optional[Dict]) -> "MetricsConfig":
        """Merge with override dict.

        Args:
            override: Dict with optional keys: log_file, extractor_pattern, metric_names

        Returns:
            New MetricsConfig with overridden values
        """
        if not override:
            return self

        return MetricsConfig(
            log_file=override.get("log_file", self.log_file),
            extractor_pattern=override.get("extractor_pattern",
                                           self.extractor_pattern),
            metric_names=override.get("metric_names", self.metric_names),
        )


@dataclass
class DatasetThreshold:
    """Accuracy threshold configuration for a single dataset."""

    dataset_name: str  # Dataset name: gsm8k, mmlu, humaneval, etc.
    expected_value: float  # Expected accuracy value
    threshold: float  # Threshold value
    threshold_type: str  # "relative" or "absolute"
    filter_type: str = "flexible-extract"  # lm_eval filter type

    def validate(self, actual_value: float) -> tuple[bool, str]:
        """Validate if accuracy passes the threshold.

        Args:
            actual_value: Actual accuracy value from test

        Returns:
            Tuple of (passed, message): Whether validation passed and detail message
        """
        if self.threshold_type == "relative":
            if self.expected_value == 0:
                error = abs(actual_value)
            else:
                error = abs(actual_value - self.expected_value) / abs(
                    self.expected_value)
            passed = error <= self.threshold
            msg = f"Relative error: {error:.6f} (threshold: {self.threshold})"
        else:  # absolute
            error = self.expected_value - actual_value
            passed = error <= 2.326 * self.threshold
            msg = f"Absolute error: {error:.6f} (threshold: {self.threshold})"

        return passed, msg


@dataclass
class AccuracyConfig:
    """Accuracy test configuration (supports multiple datasets)."""

    datasets: List[DatasetThreshold]  # List of dataset threshold configurations

    def get_dataset_config(self,
                           dataset_name: str) -> Optional[DatasetThreshold]:
        """Get configuration by dataset name.

        Args:
            dataset_name: Name of the dataset to look up

        Returns:
            DatasetThreshold config if found, None otherwise
        """
        for ds in self.datasets:
            if ds.dataset_name.lower() == dataset_name.lower():
                return ds
        return None

    def get_all_dataset_names(self) -> List[str]:
        """Get all configured dataset names.

        Returns:
            List of dataset names
        """
        return [ds.dataset_name for ds in self.datasets]


# ============================================================================
# Default Metrics configuration
# ============================================================================

# Accuracy test uses accuracy_eval.log (markdown table output from lm_eval)
# Note: Only log_file is used by AccuracyParser (accuracy_parser.py)
# The regex pattern is hardcoded in AccuracyParser._extract_accuracy_values()
_COMMON_ACCURACY_METRICS = MetricsConfig(
    log_file="accuracy_eval.log",
    extractor_pattern=
    r'\|([a-zA-Z0-9_-]+)\|.*?\|([\w-]+)\|.*?\|exact_match\|.*?\|([0-9.]+)\|',
    metric_names=["flexible-extract", "strict-match"],
)

DEFAULT_METRICS_CONFIG = {
    # Performance test default configuration
    ("disagg", "perf"):
    MetricsConfig(
        log_file="bench.log",
        extractor_pattern=r"""
            ^.*?Median\ TTFT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Median\ E2EL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Benchmark\ with\ concurrency\ (\d+)\ done
        """,
        metric_names=["DISAGG_SERVER_TTFT", "DISAGG_SERVER_E2EL"],
    ),
    ("wideep", "perf"):
    MetricsConfig(
        log_file="bench.log",
        extractor_pattern=r"""
            ^.*?Mean\ TTFT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?Median\ TTFT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?P99\ TTFT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Mean\ TPOT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?Median\ TPOT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?P99\ TPOT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Mean\ ITL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?Median\ ITL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?P99\ ITL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Mean\ E2EL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?Median\ E2EL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?P99\ E2EL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Benchmark\ with\ concurrency\ (\d+)\ done
        """,
        metric_names=[
            "WIDEEP_SERVER_MEAN_TTFT",
            "WIDEEP_SERVER_TTFT",  # Median TTFT (keep the same name as disagg)
            "WIDEEP_SERVER_P99_TTFT",
            "WIDEEP_SERVER_MEAN_TPOT",
            "WIDEEP_SERVER_MEDIAN_TPOT",
            "WIDEEP_SERVER_P99_TPOT",
            "WIDEEP_SERVER_MEAN_ITL",
            "WIDEEP_SERVER_MEDIAN_ITL",
            "WIDEEP_SERVER_P99_ITL",
            "WIDEEP_SERVER_MEAN_E2EL",
            "WIDEEP_SERVER_E2EL",  # Median E2EL (keep the same name as disagg)
            "WIDEEP_SERVER_P99_E2EL",
        ],
    ),
    # Accuracy test configuration
    ("disagg", "accuracy"):
    _COMMON_ACCURACY_METRICS,
    ("wideep", "accuracy"):
    _COMMON_ACCURACY_METRICS,
}


@dataclass
class TestConfig:
    """Test configuration data class."""

    config_path: str  # YAML file path
    test_id: str  # Auto-generated test ID
    test_type: str  # disagg, widep, etc.
    model_name: str  # Model name (read from metadata)
    test_category: str  # perf or accuracy
    benchmark_type: str  # 1k1k, 8k1k, etc. (generated from sequence)
    config_data: dict  # Full YAML content
    metrics_config: MetricsConfig  # Metrics configuration (default or overridden)
    supported_gpus: List[str]  # Supported GPU types list
    accuracy_config: Optional[
        AccuracyConfig] = None  # Accuracy configuration (for accuracy tests)

    @property
    def display_name(self) -> str:
        """Display name for pytest."""
        return f"{self.test_type}/{self.test_category}/{Path(self.config_path).stem}"


class ConfigLoader:
    """Configuration loader with default metrics support."""

    def __init__(self, base_dir: str = "test_configs"):
        """Initialize ConfigLoader.

        Args:
            base_dir: Base directory for test configs
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Config directory not found: {self.base_dir}")

    def scan_configs(
        self,
        test_type: Optional[str] = None,
        test_category: Optional[str] = None,
        model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
    ) -> List[TestConfig]:
        """Scan configuration files.

        Directory structure: test_type/category/model_bench_config.yaml

        Args:
            test_type: Filter by test type (disagg, widep, etc.)
            test_category: Filter by category (perf, accuracy)
            model_name: Filter by model name
            gpu_type: Filter by GPU type (GB200, H100, etc.). If None, uses EnvManager.get_gpu_type()

        Returns:
            List of TestConfig objects (filtered by GPU support)
        """
        # Get current GPU type from environment if not specified
        if gpu_type is None:
            gpu_type = EnvManager.get_gpu_type()

        configs = []

        if not self.base_dir.exists():
            print(f"Warning: Config directory not found: {self.base_dir}")
            return configs

        # Traverse: test_type/category/config.yaml
        for test_type_dir in self.base_dir.iterdir():
            if not test_type_dir.is_dir() or test_type_dir.name == "templates":
                continue

            current_test_type = test_type_dir.name

            # Filter by test_type
            if test_type and current_test_type != test_type:
                continue

            # Traverse category (perf/accuracy)
            for category_dir in test_type_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                current_category = category_dir.name

                # Filter by test_category
                if test_category and current_category != test_category:
                    continue

                # Load all YAML files in this category
                for yaml_file in category_dir.glob("*.yaml"):
                    try:
                        config = self._load_config_file(yaml_file,
                                                        current_test_type,
                                                        current_category)

                        # Filter by model_name
                        if model_name and config.model_name != model_name:
                            continue

                        # Filter by GPU support
                        if gpu_type and gpu_type not in config.supported_gpus:
                            print(
                                f"   ⏭️  Skipping {yaml_file.name}: not supported on {gpu_type} "
                                f"(supported: {config.supported_gpus})")
                            continue

                        configs.append(config)
                    except Exception as e:
                        print(f"Warning: Failed to load {yaml_file}: {e}")

        # Check if any configuration files are found
        if len(configs) == 0:
            # Build detailed error information
            filter_info = []
            if test_type:
                filter_info.append(f"test_type='{test_type}'")
            if test_category:
                filter_info.append(f"test_category='{test_category}'")
            if model_name:
                filter_info.append(f"model_name='{model_name}'")
            if gpu_type:
                filter_info.append(f"gpu_type='{gpu_type}'")

            filters = ", ".join(filter_info) if filter_info else "no filters"

            raise RuntimeError(
                f"❌ No configuration files found in '{self.base_dir}' with {filters}. "
                f"Please check:\n"
                f"  1. Configuration files exist in the correct directory structure\n"
                f"  2. YAML files contain valid 'metadata' section with required fields\n"
                f"  3. GPU type '{gpu_type}' is in the 'supported_gpus' list\n"
                f"  4. Filter parameters match existing configurations")

        print(
            f"\n✅ Loaded {len(configs)} configurations for GPU type: {gpu_type}"
        )
        return configs

    def _make_test_id(self, test_type: str, test_category: str,
                      test_file_name: str, config_data: dict) -> str:
        """Generate test ID based on test type, test category, test file name and configuration data.

        Format: {test_type}_{test_category}_{model_name}_{isl}k{osl}k_
        {dep_flag}{gen_tp_size}_bs{gen_batch_size}_mtp{mtp_size}
        Example: disagg_perf_deepseek-r1-fp4_1k1k_dep32_bs32_mtp3

        Args:
            test_type: Test type (disagg, widep, etc.)
            test_category: Test category (perf, accuracy)
            test_file_name: Test file name (without extension)
            config_data: YAML configuration data

        Returns:
            Generated test ID string
        """
        # Extract configuration fields
        fields = extract_config_fields(config_data)
        # Generate benchmark type (e.g., 1k1k, 8k1k)
        isl_k = fields["isl"] // 1024
        osl_k = fields["osl"] // 1024
        benchmark_type = f"{isl_k}k{osl_k}k"

        # Generate test ID
        test_id = (
            f"{test_type}_{test_category}_file:{test_file_name}_{benchmark_type}_"
            f"ctx:{fields['ctx_num']}_gen:{fields['gen_num']}_"
            f"{fields['dep_flag']}:{fields['gen_tp_size']}_bs:{fields['gen_batch_size']}_"
            f"eplb:{fields['eplb_slots']}_mtp:{fields['mtp_size']}_"
            f"ccbackend:{fields['cache_transceiver_backend']}")

        return test_id

    def _load_config_file(self, yaml_path: Path, test_type: str,
                          test_category: str) -> TestConfig:
        """Load single YAML config file."""
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Extract metadata from YAML file
        metadata = config_data.get("metadata", {})
        model_name = metadata.get("model_name", "unknown")
        supported_gpus = metadata.get(
            "supported_gpus", ["GB200", "GB300", "H100", "B200", "B300"])

        # Override config with environment variables
        config_data = self._apply_env_overrides(config_data)

        # Write back the updated config to the original file
        self._write_config_file(yaml_path, config_data)

        # Generate benchmark_type from sequence configuration
        benchmark_type = self._generate_benchmark_type(config_data)

        # Get metrics config (default or override)
        metrics_config = self._get_metrics_config(test_type, test_category,
                                                  config_data)

        # Extract test file name (without extension)
        test_file_name = yaml_path.stem  # e.g., "deepseek-r1-fp4-0"

        # Generate test ID using config data
        test_id = self._make_test_id(test_type, test_category, test_file_name,
                                     config_data)

        # Load accuracy configuration (only for accuracy tests)
        accuracy_config = None
        if test_category == "accuracy":
            acc_meta = metadata.get('accuracy', {})
            if acc_meta and 'datasets' in acc_meta:
                datasets = []
                for ds_config in acc_meta['datasets']:
                    datasets.append(
                        DatasetThreshold(
                            dataset_name=ds_config.get('name', 'gsm8k'),
                            expected_value=float(
                                ds_config.get('expected_value', 0.0)),
                            threshold=float(ds_config.get('threshold', 0.02)),
                            threshold_type=ds_config.get(
                                'threshold_type', 'relative'),
                            filter_type=ds_config.get('filter_type',
                                                      'flexible-extract')))
                accuracy_config = AccuracyConfig(datasets=datasets)
                print(
                    f"   📊 Loaded accuracy config with {len(datasets)} dataset(s)"
                )

        return TestConfig(
            config_path=str(yaml_path),
            test_id=test_id,
            test_type=test_type,
            model_name=model_name,
            test_category=test_category,
            benchmark_type=benchmark_type,
            config_data=config_data,
            metrics_config=metrics_config,
            supported_gpus=supported_gpus,
            accuracy_config=accuracy_config,
        )

    def _generate_benchmark_type(self, config_data: dict) -> str:
        """Generate benchmark type from sequence configuration.

        Examples:
            input=1024, output=1024 -> "1k1k"
            input=8192, output=1024 -> "8k1k"
            input=16384, output=2048 -> "16k2k"

        Args:
            config_data: Full YAML config data

        Returns:
            Benchmark type string (e.g., "1k1k", "8k1k")
        """
        sequence = config_data.get("sequence", {})
        input_length = sequence.get("input_length", 0)
        output_length = sequence.get("output_length", 0)

        # Convert to k notation
        input_k = input_length // 1024
        output_k = output_length // 1024

        return f"{input_k}k{output_k}k"

    def _get_metrics_config(self, test_type: str, test_category: str,
                            config_data: dict) -> MetricsConfig:
        """Get metrics config: use default or merge with override.

        Args:
            test_category: 'perf' or 'accuracy'
            config_data: Full YAML config data

        Returns:
            MetricsConfig (default or merged with overrides)
        """
        # Get default configuration
        config_key = (test_type, test_category)
        default_config = DEFAULT_METRICS_CONFIG.get(config_key)
        if not default_config:
            # If no default configuration, trigger exception
            print(
                f"   ⚠️  No default metrics config for config_key: {config_key}"
            )
            raise ValueError(
                f"No default metrics config for config_key: {config_key}")

        # Check if there are metrics overrides in YAML
        # Metrics are defined in metadata section instead of benchmark
        metadata_config = config_data.get("metadata", {})
        metrics_override = metadata_config.get("metrics")

        if metrics_override:
            # There are metrics overrides, merge them
            print("   ⚙️  Using custom metrics config (overriding defaults)")
            return default_config.merge(metrics_override)
        else:
            # No metrics overrides, use default
            print(f"   ⚙️  Using default metrics config for {test_category}")
            return default_config

    def _apply_env_overrides(self, config_data: dict) -> dict:
        """Apply environment variable overrides to configuration.

        Intelligently replaces empty or None values based on field path.
        No placeholders needed - automatically matches by field name.

        Args:
            config_data: Original configuration dict

        Returns:
            Updated configuration dict with environment variables applied
        """
        # Create a deep copy to avoid modifying original
        config = copy.deepcopy(config_data)

        # Field path mapping: (path, key) -> environment value getter
        # Uses lazy evaluation to get values only when needed
        field_mapping = {
            ("slurm", "partition"):
            lambda: EnvManager.get_slurm_partition(),
            ("slurm", "account"):
            lambda: EnvManager.get_slurm_account(),
            ("slurm", "job_name"):
            lambda: EnvManager.get_slurm_job_name(),
            ("environment", "container_mount"):
            lambda: EnvManager.get_container_mount(),
            ("environment", "container_image"):
            lambda: EnvManager.get_container_image(),
            ("environment", "trtllm_repo"):
            lambda: EnvManager.get_repo_dir(),
            ("environment", "trtllm_wheel_path"):
            lambda: EnvManager.get_trtllm_wheel_path(),
            ("environment", "dataset_file"):
            lambda: self._get_dataset_file(config),
            ("environment", "work_dir"):
            lambda: EnvManager.get_script_dir(),
            ("environment", "model_path"):
            lambda: self._get_full_model_path(config),
            ("slurm", "script_file"):
            lambda: self._get_script_file(config),
        }

        # Apply overrides based on field paths
        for (section, key), value_getter in field_mapping.items():
            if section in config:
                config[section][key] = value_getter()
        return config

    def _get_full_model_path(self, config: dict) -> str:
        """Get full model path by combining MODEL_DIR with model directory name.

        Priority:
        1. metadata.model_dir_name (explicit model directory name)
        2. Empty string (fallback)
        """
        metadata = config.get("metadata", {})
        model_dir_name = metadata.get("model_dir_name", "")

        if model_dir_name:
            return os.path.join(EnvManager.get_model_dir(), model_dir_name)
        else:
            return ""

    def _get_dataset_file(self, config: dict) -> str:
        """Get dataset file by combining dataset directory with dataset file name.

        Args:
            config: Full YAML config data
        """
        metadata = config.get("metadata", {})
        dataset_file = metadata.get("dataset_file", "")
        return os.path.join(EnvManager.get_model_dir(), dataset_file)

    def _get_script_file(self, config: dict) -> str:
        """Get script file by combining scripts directory with script file name.

        Args:
            config: Full YAML config data
        """
        metadata = config.get("metadata", {})
        script_file = metadata.get("script_file", "disaggr_torch.slurm")
        return os.path.join(EnvManager.get_script_dir(), script_file)

    def _write_config_file(self, yaml_path: Path, config_data: dict) -> None:
        """Write updated configuration back to YAML file.

        This is necessary because submit.py reads the YAML file to submit jobs.
        The file needs to contain actual values, not empty placeholders.

        Args:
            yaml_path: Path to the YAML file
            config_data: Updated configuration dict with environment values applied
        """
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(
                    config_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    width=1000,  # Prevent line wrapping
                )
            print(f"   ✅ Updated config written to: {yaml_path.name}")
        except Exception as e:
            print(
                f"   ⚠️  Warning: Failed to write config file {yaml_path}: {e}")
            # Don't fail the test if write fails, just log warning

    def get_all_models(self) -> List[str]:
        """Get list of all unique model names."""
        configs = self.scan_configs()
        return sorted(set(config.model_name for config in configs))

    def get_all_test_types(self) -> List[str]:
        """Get list of all test types."""
        if not self.base_dir.exists():
            return []
        return sorted([
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and d.name != "templates"
        ])
