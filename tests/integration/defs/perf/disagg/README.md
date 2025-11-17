# Solution 4: YAML Configuration-Based Testing Framework

## Design Philosophy

**Use directories + YAML files to organize test configurations - simple, intuitive, and easy to maintain**

Core Principles:
1. ✅ **Organize by test type and category**: test_type → perf → configuration files
2. ✅ **YAML configuration files**: Each test has its own independent YAML file
3. ✅ **Filename as metadata**: Parse model and benchmark type from filename, no YAML metadata needed
4. ✅ **Default + Override mode**: Provide default metrics configuration, override as needed
5. ✅ **Reuse existing tools**: Use `disagg/slurm/benchmark/submit.py` to submit jobs
6. ✅ **Minimal changes**: Keep pytest framework, only change configuration loading method

---

## Directory Structure

```
test_configs/
├── disagg/                                    # Test type (disaggregated)
│   ├── perf/                                  # Performance tests
│   │   ├── deepseek-r1-fp8_1k1k_tep8_bs32_mtp3_nixl.yaml
│   │   ├── deepseek-r1-fp8_1k1k_tep8_bs32_nixl.yaml
│   │   ├── deepseek-r1-fp8_1k1k_dep16_bs128_nixl.yaml
│   │   ├── deepseek-r1-fp8_8k1k_tep8_bs16_nixl.yaml
│   │   ├── llama-70b_1k1k_tep8_bs256_nixl.yaml
│   │   └── special-model_1k1k_custom_metrics.yaml  # Custom metrics
├── widep/                                     # Another test type (optional)
│   └── perf/
└── templates/                                 # Template files (optional)
    └── disagg_perf_template.yaml
```

---

## GPU Hardware Support

### Supported GPU Types

The system supports multiple GPU hardware types. Each configuration can specify which GPUs it supports:

- **GB200**: NVIDIA GB200 GPU
- **GB300**: NVIDIA GB300 GPU  
- **H100**: NVIDIA H100 GPU
- **B200**: NVIDIA B200 GPU
- **B300**: NVIDIA B300 GPU

### Configuration Method

Specify `supported_gpus` field under the `hardware` section in YAML files:

```yaml
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 4
  supported_gpus: ["GB200", "GB300"]  # This config supports GB200 and GB300
```

### GPU Filtering Mechanism

1. **Environment Variable**: The system gets the current GPU type through the `GPU_TYPE` environment variable
2. **Automatic Filtering**: `ConfigLoader` automatically filters out configurations that don't support the current GPU
3. **pytest Parameterization**: Only configurations that support the current GPU are loaded into test cases

### Usage Scenarios

#### Scenario 1: Large Model Configuration (High-end GPUs Only)
```yaml
hardware:
  supported_gpus: ["GB200", "GB300"]  # Run only on GB200/GB300
```

#### Scenario 2: Small Model Configuration (Multiple GPUs Supported)
```yaml
hardware:
  supported_gpus: ["H100", "B200", "B300"]  # Can run on H100/B200/B300
```

#### Scenario 3: Universal Configuration (All GPUs Supported)
```yaml
hardware:
  supported_gpus: ["GB200", "GB300", "H100", "B200", "B300"]  # Supports all GPUs
```

---

## Metrics Configuration

### Default Configuration Mechanism

The system provides **default metrics configuration** for different test categories. Most tests don't need to configure metrics in YAML.

#### Performance Test (perf) Default Configuration
- **Log file**: `benchmark_result.log`
- **Metrics extracted**: TTFT (Time To First Token), E2EL (End-to-End Latency)
- **Regular expression**: Predefined TTFT/E2EL extraction pattern

### Usage Scenarios

#### ✅ Scenario 1: Use Default Configuration (Recommended, 90% of cases)
```yaml
# No need to configure metrics, default configuration is used automatically
benchmark:
  mode: "e2e"
  multi_round: 8
  concurrency_list: "1 2 4 8 16 36"
  # metrics automatically uses perf default configuration
```

#### ✅ Scenario 2: Partial Override (Modify only specific fields)
```yaml
benchmark:
  mode: "e2e"
  metrics:
    # Only override log_file, pattern and metric_names inherit defaults
    log_file: "custom_benchmark.log"
```

#### ✅ Scenario 3: Fully Custom (Special Requirements)
```yaml
benchmark:
  mode: "e2e"
  metrics:
    log_file: "custom_result.log"
    extractor_pattern: "Custom Pattern:\s+([0-9.]+)"
    metric_names: ["CUSTOM_METRIC"]
```

---

## YAML Configuration File Format

### Performance Test Configuration Examples

#### Example 1: Standard Configuration (Using Default Metrics)

`test_configs/disagg/perf/deepseek-r1-fp8_1k1k_tep8_bs32_mtp3_nixl.yaml`

```yaml
# Metadata - Test metadata (for identification and filtering)
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]  # List of supported GPU types

# SLURM Configuration
slurm:
  script_file: "disaggr_torch.slurm"
  partition: "batch"
  account: "coreai_comparch_trtllm"
  job_time: "02:00:00"
  job_name: "deepseek-r1-fp4-1k1k-tep8-mtp3"
  numa_bind: true

# Benchmark Mode
benchmark:
  mode: "e2e"
  use_nv_sa_benchmark: false
  multi_round: 8
  benchmark_ratio: 0.8
  streaming: true
  concurrency_list: "1 2 4 8 16 36"
  # ⚠️ Note: No metrics configuration, will automatically use perf default metrics
  #   - log_file: benchmark_result.log
  #   - metric_names: [DISAGG_SERVER_TTFT, DISAGG_SERVER_E2EL]
  #   - extractor_pattern: Predefined TTFT/E2EL extraction pattern

# Hardware Configuration
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 4
  supported_gpus: ["GB200", "GB300"]  # List of supported GPU types

# Sequence Configuration
sequence:
  input_length: 1024
  output_length: 1024

# Environment Configuration
environment:
  container_mount: "/lustre:/lustre"
  container_image: "/lustre/fsw/portfolios/coreai/users/deemon/trtllm.sqsh"
  model_path: "/lustre/fsw/portfolios/coreai/users/xqiao/DeepSeek-R1-0528-FP4-V2"
  trtllm_repo: "/lustre/fs1/portfolios/coreai/projects/trtllm"
  build_wheel: false
  dataset_file: "/lustre/fs1/portfolios/coreai/datasets/prompts.json"
  work_dir: "/lustre/fs1/portfolios/coreai/perf_test"

# Profiling Configuration
profiling:
  nsys_on: false

# Worker Configuration
worker_config:
  gen:
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8
    enable_attention_dp: false
    enable_lm_head_tp_in_adp: true
    pipeline_parallel_size: 1
    max_batch_size: 32
    max_num_tokens: 128
    max_seq_len: 2251
    cuda_graph_config:
      enable_padding: true
      batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    print_iter_log: true
    kv_cache_config:
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.9
      dtype: fp8
    moe_config:
      backend: CUTLASS
      use_low_precision_moe_combine: true
    cache_transceiver_config:
      max_tokens_in_buffer: 4608
      backend: NIXL
    stream_interval: 20
    num_postprocess_workers: 4
    speculative_config:
      decoding_type: MTP
      num_nextn_predict_layers: 3
  
  ctx:
    max_batch_size: 4
    max_num_tokens: 4608
    max_seq_len: 1227
    tensor_parallel_size: 4
    moe_expert_parallel_size: 4
    enable_attention_dp: true
    pipeline_parallel_size: 1
    print_iter_log: true
    cuda_graph_config: null
    disable_overlap_scheduler: true
    kv_cache_config:
      enable_block_reuse: false
      free_gpu_memory_fraction: 0.85
      dtype: fp8
    cache_transceiver_config:
      max_tokens_in_buffer: 4608
      backend: NIXL
```

#### Example 2: Custom Log File (Partial Override)

`test_configs/disagg/perf/special-model_1k1k_custom_log.yaml`

```yaml
# Most configuration same as Example 1...

benchmark:
  mode: "e2e"
  multi_round: 8
  concurrency_list: "1 2 4 8 16 36"
  
  # Only override log_file, others use defaults
  metrics:
    log_file: "custom_benchmark_result.log"
    # extractor_pattern and metric_names inherit default values
```

#### Example 3: Fully Custom Metrics

`test_configs/disagg/perf/special-model_1k1k_full_custom.yaml`

```yaml
# Most configuration same as Example 1...

benchmark:
  mode: "e2e"
  multi_round: 8
  
  # Fully custom metrics configuration
  metrics:
    log_file: "throughput_log.txt"
    extractor_pattern: |
      Throughput:\s+([0-9.]+)\s+tokens/s
      Latency:\s+([0-9.]+)\s+ms
    metric_names:
      - "THROUGHPUT_TOKENS_PER_SEC"
      - "AVERAGE_LATENCY_MS"
```

---

## Core Implementation Code

### File 1: `config_loader.py` - Configuration Loader (with Default Metrics)

```python
"""
YAML Configuration Loader with Default Metrics Support
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    log_file: str                          # Log file name
    extractor_pattern: str                 # Regular expression pattern
    metric_names: List[str]                # List of metric names
    
    def merge(self, override: Optional[Dict]) -> 'MetricsConfig':
        """
        Merge with override dict
        
        Args:
            override: Dict with optional keys: log_file, extractor_pattern, metric_names
        
        Returns:
            New MetricsConfig with overridden values
        """
        if not override:
            return self
        
        return MetricsConfig(
            log_file=override.get('log_file', self.log_file),
            extractor_pattern=override.get('extractor_pattern', self.extractor_pattern),
            metric_names=override.get('metric_names', self.metric_names)
        )


# ============================================================================
# Default Metrics Configuration
# ============================================================================

DEFAULT_METRICS_CONFIG = {
    # Performance test default configuration
    "perf": MetricsConfig(
        log_file="benchmark_result.log",
        extractor_pattern=r"""
            ^.*?Median\ TTFT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Median\ E2EL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Benchmark\ with\ concurrency\ (\d+)\ done
        """,
        metric_names=["DISAGG_SERVER_TTFT", "DISAGG_SERVER_E2EL"]
    )
}


@dataclass
class TestConfig:
    """Test configuration data class"""
    config_path: str        # YAML file path
    test_id: str            # Auto-generated test ID
    test_type: str          # disagg, widep, etc.
    model_name: str         # Model name (parsed from filename)
    test_category: str      # perf category
    benchmark_type: str     # 1k1k, 8k1k, etc. (parsed from filename)
    config_data: dict       # Full YAML content
    metrics_config: MetricsConfig  # Metrics configuration (default or overridden)
    supported_gpus: List[str]  # List of supported GPU types
    
    @property
    def display_name(self) -> str:
        """Display name for pytest"""
        return f"{self.test_type}/{self.test_category}/{Path(self.config_path).stem}"


class ConfigLoader:
    """Configuration loader with default metrics support"""
    
    def __init__(self, base_dir: str = "test_configs"):
        """
        Args:
            base_dir: Base directory for test configs
        """
        self.base_dir = Path(base_dir)
    
    def scan_configs(self, test_type: Optional[str] = None, 
                    test_category: Optional[str] = None, 
                    model_name: Optional[str] = None,
                    gpu_type: Optional[str] = None) -> List[TestConfig]:
        """
        Scan configuration files
        
        Directory structure: test_type/category/model_bench_config.yaml
        
        Args:
            test_type: Filter by test type (disagg, widep, etc.)
            test_category: Filter by category (perf)
            model_name: Filter by model name
            gpu_type: Filter by GPU type (GB200, H100, etc.). If None, uses EnvManager.get_gpu_type()
        
        Returns:
            List of TestConfig objects (filtered by GPU support)
        """
        # Get current GPU type from environment if not specified
        if gpu_type is None:
            from disagg_config import EnvManager
            gpu_type = EnvManager.get_gpu_type()
        
        configs = []
        
        if not self.base_dir.exists():
            print(f"Warning: Config directory not found: {self.base_dir}")
            return configs
        
        # Traverse: test_type/category/config.yaml
        for test_type_dir in self.base_dir.iterdir():
            if not test_type_dir.is_dir() or test_type_dir.name == 'templates':
                continue
            
            current_test_type = test_type_dir.name
            
            # Filter by test_type
            if test_type and current_test_type != test_type:
                continue
            
            # Traverse category (perf)
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
                        config = self._load_config_file(
                            yaml_file,
                            current_test_type,
                            current_category
                        )
                        
                        # Filter by model_name
                        if model_name and config.model_name != model_name:
                            continue
                        
                        # Filter by GPU support
                        if gpu_type and gpu_type not in config.supported_gpus:
                            print(f"   ⏭️  Skipping {yaml_file.name}: not supported on {gpu_type} (supported: {config.supported_gpus})")
                            continue
                        
                        configs.append(config)
                    except Exception as e:
                        print(f"Warning: Failed to load {yaml_file}: {e}")
        
        print(f"\n✅ Loaded {len(configs)} configurations for GPU type: {gpu_type}")
        return configs
    
    def _load_config_file(self, yaml_path: Path, test_type: str,
                         test_category: str) -> TestConfig:
        """Load single YAML config file"""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract metadata from YAML file
        metadata = config_data.get('metadata', {})
        model_name = metadata.get('model_name', 'unknown')
        precision = metadata.get('precision', 'unknown')
        supported_gpus = metadata.get('supported_gpus', ["GB200", "GB300", "H100", "B200", "B300"])
        
        # Generate benchmark_type from sequence configuration
        benchmark_type = self._generate_benchmark_type(config_data)
        
        # Get metrics config (default or override)
        metrics_config = self._get_metrics_config(test_category, config_data)
        
        # Generate test ID
        test_id = f"{test_type}_{test_category}_{model_name}_{benchmark_type}"
        
        return TestConfig(
            config_path=str(yaml_path),
            test_id=test_id,
            test_type=test_type,
            model_name=model_name,
            test_category=test_category,
            benchmark_type=benchmark_type,
            config_data=config_data,
            metrics_config=metrics_config,
            supported_gpus=supported_gpus
        )
    
    def _generate_benchmark_type(self, config_data: dict) -> str:
        """
        Generate benchmark type from sequence configuration
        
        Examples:
            input=1024, output=1024 -> "1k1k"
            input=8192, output=1024 -> "8k1k"
            input=16384, output=2048 -> "16k2k"
        
        Args:
            config_data: Full YAML config data
        
        Returns:
            Benchmark type string (e.g., "1k1k", "8k1k")
        """
        sequence = config_data.get('sequence', {})
        input_length = sequence.get('input_length', 0)
        output_length = sequence.get('output_length', 0)
        
        # Convert to k notation
        input_k = input_length // 1024
        output_k = output_length // 1024
        
        return f"{input_k}k{output_k}k"
    
    def _get_metrics_config(self, test_category: str, config_data: dict) -> MetricsConfig:
        """
        Get metrics config: use default or merge with override
        
        Args:
            test_category: 'perf'
            config_data: Full YAML config data
        
        Returns:
            MetricsConfig (default or merged with overrides)
        """
        # Get default configuration
        default_config = DEFAULT_METRICS_CONFIG.get(test_category)
        if not default_config:
            # If no default config, use empty config
            print(f"   ⚠️  No default metrics config for category: {test_category}")
            default_config = MetricsConfig(
                log_file="",
                extractor_pattern="",
                metric_names=[]
            )
        
        # Check if YAML has metrics override
        benchmark_config = config_data.get('benchmark', {})
        metrics_override = benchmark_config.get('metrics')
        
        if metrics_override:
            # Has override config, merge
            print(f"   ⚙️  Using custom metrics config (overriding defaults)")
            return default_config.merge(metrics_override)
        else:
            # No override config, use default
            print(f"   ⚙️  Using default metrics config for {test_category}")
            return default_config
    
    def load_config_by_path(self, config_path: str) -> TestConfig:
        """Load configuration by file path"""
        yaml_path = Path(config_path)
        
        # Parse path to extract metadata
        # Expected: test_configs/{test_type}/{category}/{config}.yaml
        parts = yaml_path.relative_to(self.base_dir).parts
        
        if len(parts) < 3:
            raise ValueError(f"Invalid config path structure: {config_path}")
        
        test_type = parts[0]
        test_category = parts[1]
        
        return self._load_config_file(yaml_path, test_type, test_category)
    
    def get_all_models(self) -> List[str]:
        """Get list of all unique model names"""
        configs = self.scan_configs()
        return sorted(set(config.model_name for config in configs))
    
    def get_all_test_types(self) -> List[str]:
        """Get list of all test types"""
        if not self.base_dir.exists():
            return []
        return sorted([d.name for d in self.base_dir.iterdir() 
                      if d.is_dir() and d.name != 'templates'])
```

### File 2: `test_disagg_yaml.py` - pytest Test File

```python
"""
Disaggregated Benchmark Test - YAML Configuration Based
"""

import pytest
import os
import subprocess
import atexit
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config_loader import ConfigLoader, TestConfig
from utility import session_tracker, TestCaseTracker
from disagg_config import EnvManager
from disagg_executor import JobManager


# Load all test configurations
config_loader = ConfigLoader(base_dir="test_configs")
ALL_TEST_CONFIGS = config_loader.scan_configs()

# Convert to pytest parameters
ALL_TEST_CASES = [
    pytest.param(config, id=config.test_id)
    for config in ALL_TEST_CONFIGS
]


# Flag to track if session end has been called
_session_ended = False

def _ensure_session_end():
    """Ensure session end is called even on abnormal exit"""
    global _session_ended
    if not _session_ended:
        _session_ended = True
        print("\n⚠️  Ensuring session cleanup...")
        session_tracker.end_and_collect()

# Register atexit handler
atexit.register(_ensure_session_end)

@pytest.fixture(scope="session", autouse=True)
def session_lifecycle():
    """Session lifecycle management"""
    session_tracker.start()
    try:
        yield
    finally:
        _ensure_session_end()


class TestDisaggBenchmark:
    """Disaggregated benchmark test class - YAML based"""
    
    @pytest.mark.parametrize("test_config", ALL_TEST_CASES)
    def test_benchmark(self, request, test_config: TestConfig):
        """Benchmark test for YAML configurations"""
        full_test_name = request.node.name
        
        # Create test case tracker
        test_tracker = TestCaseTracker()
        test_case_name = f"{test_config.model_name}-{test_config.benchmark_type}"
        
        # Start tracking test case
        test_tracker.start_test_case(test_case_name)
        
        try:
            print(f"\n{'='*60}")
            print(f"Test: {test_config.display_name}")
            print(f"Config file: {test_config.config_path}")
            print(f"Test type: {test_config.test_type}")
            print(f"Category: {test_config.test_category}")
            print(f"Model: {test_config.model_name}")
            print(f"Benchmark: {test_config.benchmark_type}")
            print(f"Metrics log: {test_config.metrics_config.log_file}")
            print(f"{'='*60}")
            
            # Submit job using submit.py
            success, job_id = self._submit_yaml_job(test_config)
            
            # Validate submission result
            assert success, f"Job submission failed: {test_config.test_id}"
            assert job_id, "Unable to get job ID"
            
            # Wait for completion
            completed = JobManager.wait_for_completion(job_id, 7200)
            if not completed:
                JobManager.cancel_job(job_id)
                assert False, f"Job execution timeout: {job_id}"
            
            # End tracking test case
            test_tracker.end_test_case()
            
            # Get timestamps information
            timestamps = test_tracker.get_timestamps()
            
            # Check results using JobManager.check_job_result
            result = self._check_job_result(
                job_id, test_config, timestamps, full_test_name
            )
            assert result["success"], f"Job execution failed: {job_id}"
            
        except Exception as e:
            test_tracker.end_test_case()
            raise e
    
    def _submit_yaml_job(self, test_config: TestConfig) -> tuple[bool, str]:
        """Submit job using submit.py with YAML config"""
        print(f"🚀 Submitting job using submit.py...")
        
        try:
            # Call submit.py with the config file
            submit_script = os.path.join(
                EnvManager.get_work_dir(),
                "disagg/slurm/benchmark/submit.py"
            )
            
            cmd = ["python3", submit_script, "-c", test_config.config_path]
            
            print(f"   Command: {' '.join(cmd)}")
            
            # Execute submission
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"   ❌ Submission failed: {result.stderr}")
                return False, ""
            
            # Parse job ID from output
            output = result.stdout
            print(f"   Output: {output}")
            
            if "Submitted batch job" in output:
                import re
                match = re.search(r"Submitted batch job (\d+)", output)
                if match:
                    job_id = match.group(1)
                    print(f"   ✅ Job submitted successfully: {job_id}")
                    return True, job_id
            
            print(f"   ❌ Unable to extract job ID from output")
            return False, ""
            
        except Exception as e:
            print(f"   ❌ Job submission exception: {e}")
            return False, str(e)
    
    def _check_job_result(self, job_id: str, test_config: TestConfig,
                         timestamps: Dict[str, str], 
                         test_name: str) -> Dict[str, Any]:
        """
        Check job result using JobManager.check_job_result
        
        This method calls JobManager.check_job_result which:
        1. Parses log files using metrics_config
        2. Generates performance report
        3. Saves results to CSV
        """
        # Extract parameters from YAML config
        config_data = test_config.config_data
        
        isl = config_data['sequence']['input_length']
        osl = config_data['sequence']['output_length']
        ctx_num = config_data['hardware']['num_ctx_servers']
        gen_num = config_data['hardware']['num_gen_servers']
        gen_tp_size = config_data['worker_config']['gen']['tensor_parallel_size']
        gen_batch_size = config_data['worker_config']['gen']['max_batch_size']
        gen_enable_dp = config_data['worker_config']['gen']['enable_attention_dp']
        eplb_slots = config_data['worker_config'].get('eplb_num_slots', 0)
        
        # Get MTP size if exists
        gen_config = config_data['worker_config']['gen']
        mtp_size = 0
        if 'speculative_config' in gen_config:
            mtp_size = gen_config['speculative_config'].get('num_nextn_predict_layers', 0)
        
        # Generate log directory names (matching submit.py logic)
        dep_flag = "dep" if gen_enable_dp else "tep"
        log_base = f"{isl}-{osl}"
        context_dir = (
            f"ctx{ctx_num}_gen{gen_num}_{dep_flag}{gen_tp_size}_"
            f"batch{gen_batch_size}_eplb{eplb_slots}_mtp{mtp_size}"
        )
        
        log_dir_name = log_base
        
        print(f"   📁 Log directory: {log_dir_name}")
        print(f"   📁 Context directory: {context_dir}")
        
        # Call JobManager.check_job_result with metrics_config
        result = JobManager.check_job_result(
            job_id=job_id,
            benchmark_type=test_config.benchmark_type,
            config=config_data,              # Pass dict directly
            metrics_config=test_config.metrics_config,  # Pass metrics config
            model_name=test_config.model_name,
            log_dir_name=log_dir_name,
            context_dir=context_dir,
            timestamps=timestamps,
            test_name=test_name
        )
        
        return result


if __name__ == "__main__":
    """Run benchmark tests"""
    pytest.main([__file__, "-v"])
```

### File 3: Modify `disagg_executor.py` - `check_job_result` Method

Need to modify the signature to accept `metrics_config` parameter:

```python
# Modify in disagg_executor.py

from config_loader import MetricsConfig  # Add import

@staticmethod
def check_job_result(job_id: str, benchmark_type: str, config: dict,
                    metrics_config: MetricsConfig,  # New parameter
                    model_name: str, log_dir_name: str, context_dir: str, 
                    timestamps: Optional[Dict[str, str]] = None, 
                    test_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Check job result with metrics config
    
    Args:
        job_id: SLURM job ID
        benchmark_type: Benchmark type (1k1k, 8k1k, etc.)
        config: Configuration dict (YAML data)
        metrics_config: Metrics configuration (default or custom)
        model_name: Model name
        log_dir_name: Log directory name
        context_dir: Context directory name
        timestamps: Optional timestamps dict
        test_name: Optional test name
    """
    result = {"job_id": job_id, "status": "UNKNOWN", "success": False}

    result_dir = os.path.join(EnvManager.get_work_dir(), log_dir_name, context_dir)
    print(f"   📁 Checking result directory: {result_dir}")
    
    # Print the slurm log to console
    slurm_log_writer = LogWriter(EnvManager.get_work_dir())
    slurm_log_writer.print_to_console(f"slurm-{job_id}.out")
    
    # Print the metrics log file specified in metrics_config
    log_writer = LogWriter(result_dir)
    if os.path.exists(os.path.join(result_dir, metrics_config.log_file)):
        log_writer.print_to_console(metrics_config.log_file)
    else:
        print(f"   ⚠️  Metrics log file not found: {metrics_config.log_file}")
    
    # Parse using metrics config
    log_parser = LogParser(benchmark_type, config, metrics_config, 
                          log_dir_name, context_dir)
    parse_result = log_parser.parse(model_name, timestamps=timestamps, test_name=test_name)
    
    if parse_result["status"] == False:
        return result

    output_path = EnvManager.get_output_path()
    os.makedirs(output_path, exist_ok=True)

    output_csv = os.path.join(output_path, "perf_script_test_results.csv")
    result_saver = ResultSaver(output_csv)
    result_df = parse_result["df"]
    result_saver.append_a_df(result_df)
    result["success"] = True
    result["status"] = "SUCCESS"
    return result
```

### File 4: Modify `disagg_report.py` - `LogParser`

```python
# Modify in disagg_report.py

from config_loader import MetricsConfig  # Add import

class LogParser:
    """Log parser with metrics config support"""
    
    def __init__(self, benchmark_type: str, config: dict,
                 metrics_config: MetricsConfig,  # New parameter
                 log_dir_name: str, context_dir: str):
        self.benchmark_type = benchmark_type
        self.config = config
        self.metrics_config = metrics_config  # Save metrics config
        self.log_dir_name = log_dir_name
        self.context_dir = context_dir
    
    def parse(self, model_name: str, timestamps: Optional[Dict] = None, 
             test_name: Optional[str] = None) -> Dict[str, Any]:
        """Parse logs using configured metrics"""
        
        # Build log file path
        log_file_path = os.path.join(
            EnvManager.get_work_dir(),
            self.log_dir_name,
            self.context_dir,
            self.metrics_config.log_file
        )
        
        if not os.path.exists(log_file_path):
            print(f"   ❌ Log file not found: {log_file_path}")
            return {"status": False, "df": None}
        
        # Read log file
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        # Extract metrics using configured pattern
        import re
        results = {}
        
        matches = re.finditer(
            self.metrics_config.extractor_pattern, 
            log_content, 
            re.MULTILINE | re.VERBOSE
        )
        
        for match in matches:
            groups = match.groups()
            for i, metric_name in enumerate(self.metrics_config.metric_names):
                if i < len(groups):
                    results[metric_name] = groups[i]
        
        # Build DataFrame
        df = self._build_dataframe(results, model_name, timestamps, test_name)
        
        return {"status": True, "df": df}
    
    def _build_dataframe(self, results: Dict, model_name: str, 
                        timestamps: Optional[Dict], test_name: Optional[str]):
        """Build DataFrame from parsed results"""
        # ... existing DataFrame building logic ...
        pass
```

### File 5: `list_configs.py` - Configuration Viewing Tool

```python
"""
List and inspect test configurations
"""

import argparse
from config_loader import ConfigLoader


def main():
    parser = argparse.ArgumentParser(description="List test configurations")
    parser.add_argument("--base-dir", default="test_configs", help="Base config directory")
    parser.add_argument("--test-type", help="Filter by test type (disagg, widep, etc.)")
    parser.add_argument("--category", help="Filter by category (perf)")
    parser.add_argument("--model", help="Filter by model name")
    parser.add_argument("--gpu-type", help="Filter by GPU type (GB200, H100, etc.). Default: from GPU_TYPE env var")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    parser.add_argument("--show-metrics", action="store_true", help="Show metrics config")
    parser.add_argument("--show-all-gpus", action="store_true", help="Show all configs regardless of GPU support")
    
    args = parser.parse_args()
    
    loader = ConfigLoader(base_dir=args.base_dir)
    
    # If --show-all-gpus is specified, pass empty string to disable GPU filtering
    gpu_filter = "" if args.show_all_gpus else args.gpu_type
    
    configs = loader.scan_configs(
        test_type=args.test_type,
        test_category=args.category,
        model_name=args.model,
        gpu_type=gpu_filter
    )
    
    print(f"\nFound {len(configs)} test configurations\n")
    print("=" * 80)
    
    # Group by test_type and category
    grouped = {}
    for config in configs:
        key = (config.test_type, config.test_category)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(config)
    
    for (test_type, category), group_configs in sorted(grouped.items()):
        print(f"\n{test_type} / {category}")
        print("-" * 40)
        print(f"  Total: {len(group_configs)} configurations")
        
        # Group by model
        by_model = {}
        for config in group_configs:
            if config.model_name not in by_model:
                by_model[config.model_name] = []
            by_model[config.model_name].append(config)
        
        for model, model_configs in sorted(by_model.items()):
            print(f"\n  {model}: {len(model_configs)} configs")
            for config in model_configs:
                filename = config.config_path.split('/')[-1]
                print(f"    - {filename}")
                
                if args.verbose:
                    gen_config = config.config_data['worker_config']['gen']
                    print(f"      TP: {gen_config['tensor_parallel_size']}, "
                          f"Batch: {gen_config['max_batch_size']}, "
                          f"DP: {gen_config['enable_attention_dp']}")
                
                if args.show_metrics:
                    metrics = config.metrics_config
                    print(f"      Metrics log: {metrics.log_file}")
                    print(f"      Metric names: {', '.join(metrics.metric_names)}")
                
                if args.verbose or args.show_all_gpus:
                    print(f"      Supported GPUs: {', '.join(config.supported_gpus)}")
    
    print("\n" + "=" * 80)
    print(f"\nTotal: {len(configs)} configurations")
    
    # Show GPU type information
    if not args.show_all_gpus:
        from disagg_config import EnvManager
        current_gpu = args.gpu_type or EnvManager.get_gpu_type()
        print(f"Filtered for GPU type: {current_gpu}")
    
    # Show summary
    print("\nSummary:")
    print(f"  Models: {len(loader.get_all_models())}")
    print(f"  Test types: {', '.join(loader.get_all_test_types())}")


if __name__ == "__main__":
    main()
```

---

## Configuration File Naming Convention

### Filename Format

**Recommended format: `{model}_{benchmark_type}_{config_details}.yaml`**

- Use underscores `_` to separate parts for human readability
- Part 1: Model name (can use hyphens `-` internally)
- Part 2: Benchmark type (like 1k1k, 8k1k)
- After that: Configuration details (like tep8_bs32_mtp3_nixl)

**⚠️ Note**: Filenames are only for human readability. The actual `model_name`, `benchmark_type`, `precision`, `supported_gpus`, etc. are all read from the `metadata` and `sequence` fields inside the YAML file.

### benchmark_type Auto-Generation

`benchmark_type` is automatically generated from the `sequence` configuration in the YAML file:
- `input_length: 1024, output_length: 1024` → `1k1k`
- `input_length: 8192, output_length: 1024` → `8k1k`
- `input_length: 16384, output_length: 2048` → `16k2k`

### Performance Test Naming Examples

- `deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_nixl.yaml`
  - Filename helps identify: deepseek-r1-fp4, 1k1k config, TEP8 architecture
  - Actual data read from YAML's `metadata` and `sequence`

- `llama-70b_1k1k_dep16_bs128_nixl.yaml`
  - Filename helps identify: llama-70b, 1k1k config, DEP16 architecture

---

## How to Use

### 1. Create Test Configurations

```bash
# Create directory structure
mkdir -p test_configs/disagg/perf

# Create performance test config (uses default metrics)
vim test_configs/disagg/perf/deepseek-r1-fp8_1k1k_tep8_bs32.yaml
# No need to configure metrics, defaults are used automatically
```

### 2. View All Configurations

```bash
# List all configurations (automatically filtered by current GPU type)
python list_configs.py

# View all configurations, including those not supported by current GPU
python list_configs.py --show-all-gpus -v

# View configurations for a specific GPU type
python list_configs.py --gpu-type GB200

# View configurations and display metrics info
python list_configs.py --show-metrics

# View specific category
python list_configs.py --category perf -v

# View specific model
python list_configs.py --model deepseek-r1-fp4 --show-metrics

# View specific model on H100
python list_configs.py --model deepseek-v3-lite-fp8 --gpu-type H100 -v
```

### 3. Run Tests

```bash
# Run all tests
pytest test_disagg_yaml.py -v

# Run only performance tests
pytest test_disagg_yaml.py -k "perf" -v

# Run specific model
pytest test_disagg_yaml.py -k "deepseek-r1-fp8" -v

# View detailed output
pytest test_disagg_yaml.py -s -vv
```

---

## Key Improvements

### 1. Default Metrics Configuration

**Before**: Every YAML file needed to configure metrics, lots of duplication

**After**: Define default configuration, 90% of files don't need to configure metrics

```python
DEFAULT_METRICS_CONFIG = {
    "perf": MetricsConfig(
        log_file="benchmark_result.log",
        extractor_pattern=r"...",  # Predefined TTFT/E2EL pattern
        metric_names=["DISAGG_SERVER_TTFT", "DISAGG_SERVER_E2EL"]
    )
}
```

### 2. Smart Merge Mechanism

```python
def _get_metrics_config(self, test_category: str, config_data: dict):
    default_config = DEFAULT_METRICS_CONFIG.get(test_category)
    metrics_override = config_data.get('benchmark', {}).get('metrics')
    
    if metrics_override:
        # Partial override: only override specified fields
        return default_config.merge(metrics_override)
    else:
        # Use default
        return default_config
```

### 3. Flexible Override Options

```yaml
# Use defaults completely
benchmark:
  mode: "e2e"
  # Don't configure metrics

# Partial override
benchmark:
  metrics:
    log_file: "custom.log"  # Only change this

# Fully custom
benchmark:
  metrics:
    log_file: "custom.log"
    extractor_pattern: "..."
    metric_names: [...]
```

---

## Summary of Code Changes Needed

### 1. `disagg_executor.py`

```python
# Modify method signature
def check_job_result(..., metrics_config: MetricsConfig, ...):
    # Use metrics_config.log_file
    # Pass metrics_config to LogParser
```

### 2. `disagg_report.py`

```python
# Modify LogParser constructor
class LogParser:
    def __init__(..., metrics_config: MetricsConfig, ...):
        self.metrics_config = metrics_config
    
    def parse(...):
        # Use self.metrics_config.log_file
        # Use self.metrics_config.extractor_pattern
        # Use self.metrics_config.metric_names
```

---

## Summary

### Core Improvements

1. ✅ **Default config + Optional override**: Reduce 90% of duplicate configuration
2. ✅ **Simplified config files**: Most YAMLs don't need metrics section
3. ✅ **Flexible override**: Support partial override and full customization
4. ✅ **Centralized code management**: Manage default configs in ConfigLoader
5. ✅ **Easy to extend**: Add new test categories by defining them in DEFAULT_METRICS_CONFIG
6. ✅ **Metadata fields**: Centrally manage `model_name`, `precision`, `supported_gpus`, etc.
7. ✅ **Dynamic benchmark_type**: Auto-generated from `sequence` config, avoids filename/content mismatch
8. ✅ **GPU type filtering**: Automatically filter configs by current GPU type, supports multi-GPU environments

### Metrics Configuration Decision Tree

```
Do you need custom metrics?
├─ No (90% of cases)
│  └─ Don't configure metrics section, use defaults
│
├─ Yes (rare cases)
│  ├─ Only need to change log file?
│  │  └─ Only configure log_file
│  │
│  └─ Need full customization?
│     └─ Configure complete metrics section
```

### Benefits Summary

- **Simple**: Most config files are simpler
- **Flexible**: Support override as needed
- **Maintainable**: Default configs centrally managed
- **Extensible**: Easy to add new metrics types
- **Reliable**: Config file is the single source of truth
- **Smart**: Automatically filter configs by GPU type

### Design Philosophy

**Configuration as Data**

All key information (`model_name`, `precision`, `benchmark_type`, `supported_gpus`) is read from YAML file content, not parsed from filenames. This ensures:

1. **Single source of truth**: YAML file content is the authoritative data source
2. **Flexible refactoring**: Can modify config content without renaming files
3. **Program friendly**: Easy to generate and modify configs programmatically
4. **Human readable**: Filenames still maintain readability for browsing and identification

**Metadata Extensibility**

Through the `metadata` field, you can easily add new metadata:

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]
  # Future extensibility
  author: "team-name"
  created_date: "2025-01-15"
  tags: ["production", "high-priority"]
```

That's it! Simple and powerful! 🎉
