# Solution 4: 基于 YAML 配置的测试方案

## 设计理念

**使用目录+YAML文件组织测试配置，简单直观，易于维护**

核心原则：
1. ✅ **按测试类型和类别分目录**：test_type → perf/accuracy → 配置文件
2. ✅ **YAML 配置文件**：每个测试一个独立的 YAML 文件
3. ✅ **文件名即元数据**：从文件名解析模型和benchmark类型，无需YAML metadata
4. ✅ **默认 + 覆盖模式**：提供默认 metrics 配置，按需覆盖
5. ✅ **复用现有工具**：使用 `disagg/slurm/benchmark/submit.py` 提交作业
6. ✅ **最小改动**：保留 pytest 框架，只改配置读取方式

---

## 目录结构

```
test_configs/
├── disagg/                                    # 测试类型（disaggregated）
│   ├── perf/                                  # 性能测试
│   │   ├── deepseek-r1-fp8_1k1k_tep8_bs32_mtp3_nixl.yaml
│   │   ├── deepseek-r1-fp8_1k1k_tep8_bs32_nixl.yaml
│   │   ├── deepseek-r1-fp8_1k1k_dep16_bs128_nixl.yaml
│   │   ├── deepseek-r1-fp8_8k1k_tep8_bs16_nixl.yaml
│   │   ├── llama-70b_1k1k_tep8_bs256_nixl.yaml
│   │   └── special-model_1k1k_custom_metrics.yaml  # 自定义 metrics
│   └── accuracy/                              # 精度测试
│       ├── deepseek-r1-fp8_1k1k_gsm8k.yaml
│       └── llama-70b_1k1k_mmlu.yaml
├── widep/                                     # 另一种测试类型（可选）
│   ├── perf/
│   └── accuracy/
└── templates/                                 # 模板文件（可选）
    ├── disagg_perf_template.yaml
    └── disagg_accuracy_template.yaml
```

---

## GPU 硬件支持机制

### 支持的 GPU 类型

系统支持多种 GPU 硬件类型，每个配置可以指定其支持的 GPU 列表：

- **GB200**: NVIDIA GB200 GPU
- **GB300**: NVIDIA GB300 GPU  
- **H100**: NVIDIA H100 GPU
- **B200**: NVIDIA B200 GPU
- **B300**: NVIDIA B300 GPU

### 配置方式

在 YAML 文件的 `hardware` 节点下指定 `supported_gpus` 字段：

```yaml
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 4
  supported_gpus: ["GB200", "GB300"]  # 此配置支持 GB200 和 GB300
```

### GPU 过滤机制

1. **环境变量**: 系统通过 `GPU_TYPE` 环境变量获取当前 GPU 类型
2. **自动过滤**: `ConfigLoader` 会自动过滤掉不支持当前 GPU 的配置
3. **pytest 参数化**: 只有支持当前 GPU 的配置会被加载到测试用例中

### 使用场景

#### 场景 1: 大模型配置（仅支持高端 GPU）
```yaml
hardware:
  supported_gpus: ["GB200", "GB300"]  # 仅在 GB200/GB300 上运行
```

#### 场景 2: 小模型配置（支持多种 GPU）
```yaml
hardware:
  supported_gpus: ["H100", "B200", "B300"]  # 可在 H100/B200/B300 上运行
```

#### 场景 3: 通用配置（支持所有 GPU）
```yaml
hardware:
  supported_gpus: ["GB200", "GB300", "H100", "B200", "B300"]  # 支持所有 GPU
```

---

## Metrics 配置说明

### 默认配置机制

系统为不同测试类别提供了**默认的 metrics 配置**，大多数测试无需在 YAML 中配置 metrics。

#### 性能测试 (perf) 默认配置
- **日志文件**: `benchmark_result.log`
- **提取指标**: TTFT (Time To First Token), E2EL (End-to-End Latency)
- **正则表达式**: 预定义的 TTFT/E2EL 提取模式

#### 精度测试 (accuracy) 默认配置
- **日志文件**: `accuracy_result.json`
- **提取指标**: Accuracy
- **正则表达式**: 预定义的准确率提取模式

### 使用场景

#### ✅ 场景 1：使用默认配置（推荐，90% 的情况）
```yaml
# 不需要配置 metrics，自动使用默认配置
benchmark:
  mode: "e2e"
  multi_round: 8
  concurrency_list: "1 2 4 8 16 36"
  # metrics 自动使用 perf 默认配置
```

#### ✅ 场景 2：部分覆盖（只修改个别字段）
```yaml
benchmark:
  mode: "e2e"
  metrics:
    # 只覆盖 log_file，pattern 和 metric_names 继承默认
    log_file: "custom_benchmark.log"
```

#### ✅ 场景 3：完全自定义（特殊需求）
```yaml
benchmark:
  mode: "e2e"
  metrics:
    log_file: "custom_result.log"
    extractor_pattern: "Custom Pattern:\s+([0-9.]+)"
    metric_names: ["CUSTOM_METRIC"]
```

---

## YAML 配置文件格式

### 性能测试配置示例

#### 示例 1：标准配置（使用默认 metrics）

`test_configs/disagg/perf/deepseek-r1-fp8_1k1k_tep8_bs32_mtp3_nixl.yaml`

```yaml
# Metadata - 测试元数据（用于识别和过滤）
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]  # 支持的 GPU 类型列表

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
  # ⚠️ 注意：没有 metrics 配置，将自动使用 perf 默认 metrics
  #   - log_file: benchmark_result.log
  #   - metric_names: [DISAGG_SERVER_TTFT, DISAGG_SERVER_E2EL]
  #   - extractor_pattern: 预定义的 TTFT/E2EL 提取模式

# Hardware Configuration
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 4
  supported_gpus: ["GB200", "GB300"]  # 支持的 GPU 类型列表

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
  eplb_num_slots: 0
  
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

#### 示例 2：自定义日志文件（部分覆盖）

`test_configs/disagg/perf/special-model_1k1k_custom_log.yaml`

```yaml
# 大部分配置与示例1相同...

benchmark:
  mode: "e2e"
  multi_round: 8
  concurrency_list: "1 2 4 8 16 36"
  
  # 只覆盖 log_file，其他使用默认
  metrics:
    log_file: "custom_benchmark_result.log"
    # extractor_pattern 和 metric_names 继承默认值
```

#### 示例 3：完全自定义 metrics

`test_configs/disagg/perf/special-model_1k1k_full_custom.yaml`

```yaml
# 大部分配置与示例1相同...

benchmark:
  mode: "e2e"
  multi_round: 8
  
  # 完全自定义 metrics 配置
  metrics:
    log_file: "throughput_log.txt"
    extractor_pattern: |
      Throughput:\s+([0-9.]+)\s+tokens/s
      Latency:\s+([0-9.]+)\s+ms
    metric_names:
      - "THROUGHPUT_TOKENS_PER_SEC"
      - "AVERAGE_LATENCY_MS"
```

### 精度测试配置示例

#### 示例 1：标准精度测试（使用默认 metrics）

`test_configs/disagg/accuracy/deepseek-r1-fp8_1k1k_gsm8k.yaml`

```yaml
# SLURM Configuration
slurm:
  script_file: "disaggr_torch.slurm"
  partition: "batch"
  account: "coreai_comparch_trtllm"
  job_time: "02:00:00"
  job_name: "deepseek-r1-fp8-1k1k-accuracy"
  numa_bind: true

# Benchmark Mode - Accuracy specific
benchmark:
  mode: "accuracy"
  use_nv_sa_benchmark: false
  multi_round: 1
  benchmark_ratio: 1.0
  streaming: false
  concurrency_list: "1"
  
  # 精度验证参数
  expected_accuracy: 85.5
  relative_error_threshold: 1.0  # 相对误差阈值 (%)
  absolute_error_threshold: 0.5  # 绝对误差阈值 (%)
  
  # ⚠️ 注意：没有 metrics 配置，将自动使用 accuracy 默认 metrics
  #   - log_file: accuracy_result.json
  #   - metric_names: [ACCURACY]
  #   - extractor_pattern: 预定义的准确率提取模式

# Hardware Configuration
hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 4

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
  dataset_file: "/lustre/fs1/portfolios/coreai/datasets/gsm8k.json"
  work_dir: "/lustre/fs1/portfolios/coreai/perf_test"

profiling:
  nsys_on: false

worker_config:
  eplb_num_slots: 0
  gen:
    tensor_parallel_size: 8
    moe_expert_parallel_size: 8
    enable_attention_dp: false
    max_batch_size: 1
    max_num_tokens: 128
    max_seq_len: 2251
    kv_cache_config:
      free_gpu_memory_fraction: 0.9
      dtype: fp8
    cache_transceiver_config:
      max_tokens_in_buffer: 4608
      backend: NIXL
  ctx:
    max_batch_size: 1
    max_num_tokens: 4608
    max_seq_len: 1227
    tensor_parallel_size: 4
    moe_expert_parallel_size: 4
    enable_attention_dp: true
    kv_cache_config:
      free_gpu_memory_fraction: 0.85
      dtype: fp8
    cache_transceiver_config:
      max_tokens_in_buffer: 4608
      backend: NIXL
```

#### 示例 2：自定义 accuracy metrics（MMLU 数据集）

`test_configs/disagg/accuracy/deepseek-r1-fp8_1k1k_mmlu.yaml`

```yaml
# 大部分配置与示例1相同...

benchmark:
  mode: "accuracy"
  expected_accuracy: 90.0
  
  # 自定义 metrics（MMLU 有不同的输出格式）
  metrics:
    log_file: "mmlu_results.json"
    extractor_pattern: "MMLU Score:\s+([0-9.]+)"
    metric_names: ["MMLU_SCORE"]
```

---

## 核心实现代码

### 文件 1: `config_loader.py` - 配置加载器（含默认 metrics）

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
    log_file: str                          # 日志文件名
    extractor_pattern: str                 # 正则表达式
    metric_names: List[str]                # 指标名称列表
    
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
# 默认 Metrics 配置
# ============================================================================

DEFAULT_METRICS_CONFIG = {
    # 性能测试默认配置
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
    ),
    
    # 精度测试默认配置
    "accuracy": MetricsConfig(
        log_file="accuracy_result.json",
        extractor_pattern=r"Accuracy:\s+([0-9.]+)%",
        metric_names=["ACCURACY"]
    )
}


@dataclass
class TestConfig:
    """Test configuration data class"""
    config_path: str        # YAML file path
    test_id: str            # Auto-generated test ID
    test_type: str          # disagg, widep, etc.
    model_name: str         # Model name (从文件名解析)
    test_category: str      # perf or accuracy
    benchmark_type: str     # 1k1k, 8k1k, etc. (从文件名解析)
    config_data: dict       # Full YAML content
    metrics_config: MetricsConfig  # Metrics 配置（默认或覆盖后的）
    supported_gpus: List[str]  # 支持的 GPU 类型列表
    
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
            test_category: Filter by category (perf, accuracy)
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
            test_category: 'perf' or 'accuracy'
            config_data: Full YAML config data
        
        Returns:
            MetricsConfig (default or merged with overrides)
        """
        # 获取默认配置
        default_config = DEFAULT_METRICS_CONFIG.get(test_category)
        if not default_config:
            # 如果没有默认配置，使用空配置
            print(f"   ⚠️  No default metrics config for category: {test_category}")
            default_config = MetricsConfig(
                log_file="",
                extractor_pattern="",
                metric_names=[]
            )
        
        # 检查 YAML 中是否有 metrics 覆盖
        benchmark_config = config_data.get('benchmark', {})
        metrics_override = benchmark_config.get('metrics')
        
        if metrics_override:
            # 有覆盖配置，合并
            print(f"   ⚙️  Using custom metrics config (overriding defaults)")
            return default_config.merge(metrics_override)
        else:
            # 没有覆盖配置，使用默认
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

### 文件 2: `test_disagg_yaml.py` - pytest 测试文件

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

### 文件 3: 修改 `disagg_executor.py` 的 `check_job_result` 方法

需要修改签名，接受 `metrics_config` 参数：

```python
# 在 disagg_executor.py 中修改

from config_loader import MetricsConfig  # 新增 import

@staticmethod
def check_job_result(job_id: str, benchmark_type: str, config: dict,
                    metrics_config: MetricsConfig,  # 新增参数
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
    slurm_log_writer = LogWritter(EnvManager.get_work_dir())
    slurm_log_writer.print_to_console(f"slurm-{job_id}.out")
    
    # Print the metrics log file specified in metrics_config
    log_writer = LogWritter(result_dir)
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

### 文件 4: 修改 `disagg_report.py` 的 `LogParser`

```python
# 在 disagg_report.py 中修改

from config_loader import MetricsConfig  # 新增 import

class LogParser:
    """Log parser with metrics config support"""
    
    def __init__(self, benchmark_type: str, config: dict,
                 metrics_config: MetricsConfig,  # 新增参数
                 log_dir_name: str, context_dir: str):
        self.benchmark_type = benchmark_type
        self.config = config
        self.metrics_config = metrics_config  # 保存 metrics 配置
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

### 文件 5: `list_configs.py` - 配置查看工具

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
    parser.add_argument("--category", help="Filter by category (perf, accuracy)")
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

## 配置文件命名规范

### 文件名格式

**推荐格式：`{model}_{benchmark_type}_{config_details}.yaml`**

- 使用下划线 `_` 分隔各部分，便于人类阅读
- 第1部分：模型名（内部可用连字符 `-`）
- 第2部分：benchmark类型（如 1k1k, 8k1k）
- 之后：配置细节（如 tep8_bs32_mtp3_nixl）

**⚠️ 注意**：文件名仅用于人类可读性，实际的 `model_name`、`benchmark_type`、`precision`、`supported_gpus` 等信息均从 YAML 文件内的 `metadata` 和 `sequence` 字段读取。

### benchmark_type 自动生成

`benchmark_type` 会根据 YAML 文件中的 `sequence` 配置自动生成：
- `input_length: 1024, output_length: 1024` → `1k1k`
- `input_length: 8192, output_length: 1024` → `8k1k`
- `input_length: 16384, output_length: 2048` → `16k2k`

### 性能测试命名示例

- `deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_nixl.yaml`
  - 文件名辅助识别：deepseek-r1-fp4, 1k1k配置, TEP8架构
  - 实际数据从 YAML 的 `metadata` 和 `sequence` 读取

- `llama-70b_1k1k_dep16_bs128_nixl.yaml`
  - 文件名辅助识别：llama-70b, 1k1k配置, DEP16架构

### 精度测试命名示例

- `deepseek-r1-fp4_1k1k_gsm8k.yaml`
  - 文件名辅助识别：deepseek-r1-fp4, 1k1k配置, GSM8K数据集

---

## 使用方式

### 1. 创建测试配置

```bash
# 创建目录结构
mkdir -p test_configs/disagg/perf
mkdir -p test_configs/disagg/accuracy

# 创建性能测试配置（使用默认 metrics）
vim test_configs/disagg/perf/deepseek-r1-fp8_1k1k_tep8_bs32.yaml
# 不需要配置 metrics，自动使用默认

# 创建精度测试配置（使用默认 metrics）
vim test_configs/disagg/accuracy/deepseek-r1-fp8_1k1k_gsm8k.yaml
# 不需要配置 metrics，自动使用默认
```

### 2. 查看所有配置

```bash
# 列出所有配置（自动过滤当前 GPU 类型）
python list_configs.py

# 查看所有配置，包括不支持当前 GPU 的
python list_configs.py --show-all-gpus -v

# 查看特定 GPU 类型的配置
python list_configs.py --gpu-type GB200

# 查看配置并显示 metrics 信息
python list_configs.py --show-metrics

# 查看特定类别
python list_configs.py --category perf -v

# 查看特定模型
python list_configs.py --model deepseek-r1-fp4 --show-metrics

# 查看特定模型在 H100 上的配置
python list_configs.py --model deepseek-v3-lite-fp8 --gpu-type H100 -v
```

### 3. 运行测试

```bash
# 运行所有测试
pytest test_disagg_yaml.py -v

# 只运行性能测试
pytest test_disagg_yaml.py -k "perf" -v

# 只运行精度测试
pytest test_disagg_yaml.py -k "accuracy" -v

# 运行特定模型
pytest test_disagg_yaml.py -k "deepseek-r1-fp8" -v

# 查看详细输出
pytest test_disagg_yaml.py -s -vv
```

---

## 关键改进说明

### 1. 默认 Metrics 配置

**改进前**：每个 YAML 文件都要配置 metrics，大量重复

**改进后**：定义默认配置，90% 的文件不需要配置

```python
DEFAULT_METRICS_CONFIG = {
    "perf": MetricsConfig(
        log_file="benchmark_result.log",
        extractor_pattern=r"...",  # 预定义的 TTFT/E2EL 模式
        metric_names=["DISAGG_SERVER_TTFT", "DISAGG_SERVER_E2EL"]
    ),
    "accuracy": MetricsConfig(...)
}
```

### 2. 智能合并机制

```python
def _get_metrics_config(self, test_category: str, config_data: dict):
    default_config = DEFAULT_METRICS_CONFIG.get(test_category)
    metrics_override = config_data.get('benchmark', {}).get('metrics')
    
    if metrics_override:
        # 部分覆盖：只覆盖指定的字段
        return default_config.merge(metrics_override)
    else:
        # 使用默认
        return default_config
```

### 3. 灵活的覆盖方式

```yaml
# 完全使用默认
benchmark:
  mode: "e2e"
  # 不配置 metrics

# 部分覆盖
benchmark:
  metrics:
    log_file: "custom.log"  # 只改这个

# 完全自定义
benchmark:
  metrics:
    log_file: "custom.log"
    extractor_pattern: "..."
    metric_names: [...]
```

---

## 需要修改的现有代码总结

### 1. `disagg_executor.py`

```python
# 修改方法签名
def check_job_result(..., metrics_config: MetricsConfig, ...):
    # 使用 metrics_config.log_file
    # 传递 metrics_config 给 LogParser
```

### 2. `disagg_report.py`

```python
# 修改 LogParser 构造函数
class LogParser:
    def __init__(..., metrics_config: MetricsConfig, ...):
        self.metrics_config = metrics_config
    
    def parse(...):
        # 使用 self.metrics_config.log_file
        # 使用 self.metrics_config.extractor_pattern
        # 使用 self.metrics_config.metric_names
```

---

## 总结

### 核心改进

1. ✅ **默认配置 + 可选覆盖**：减少 90% 的重复配置
2. ✅ **简化配置文件**：大多数 YAML 不需要 metrics 节点
3. ✅ **灵活覆盖**：支持部分覆盖和完全自定义
4. ✅ **代码集中管理**：在 ConfigLoader 中统一管理默认配置
5. ✅ **易于扩展**：添加新的测试类别只需在 DEFAULT_METRICS_CONFIG 中定义
6. ✅ **Metadata 字段**：集中管理 `model_name`、`precision`、`supported_gpus` 等元数据
7. ✅ **动态 benchmark_type**：从 `sequence` 配置自动生成，避免文件名与内容不一致
8. ✅ **GPU 类型过滤**：自动根据当前 GPU 类型过滤配置，支持多 GPU 环境

### Metrics 配置决策树

```
是否需要自定义 metrics？
├─ 否（90% 情况）
│  └─ 不配置 metrics 节点，使用默认
│
├─ 是（少数情况）
│  ├─ 只需修改日志文件？
│  │  └─ 只配置 log_file
│  │
│  └─ 需要完全自定义？
│     └─ 配置完整的 metrics 节点
```

### 优势总结

- **简洁**：大多数配置文件更简单
- **灵活**：支持按需覆盖
- **可维护**：默认配置集中管理
- **可扩展**：易于添加新的 metrics 类型
- **可靠**：配置文件是唯一真实来源（Single Source of Truth）
- **智能**：自动根据 GPU 类型过滤配置

### 设计理念

**配置即数据（Configuration as Data）**

所有关键信息（`model_name`、`precision`、`benchmark_type`、`supported_gpus`）都从 YAML 文件内容读取，而不是从文件名解析。这确保了：

1. **唯一真实来源**：YAML 文件内容是权威数据源
2. **灵活重构**：可以修改配置内容而无需重命名文件
3. **程序友好**：便于程序化生成和修改配置
4. **人类可读**：文件名仍然保留可读性，便于浏览和识别

**元数据扩展性（Metadata Extensibility）**

通过 `metadata` 字段，可以轻松添加新的元数据：

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]
  # 未来可扩展
  author: "team-name"
  created_date: "2025-01-15"
  tags: ["production", "high-priority"]
```

就是这么简单！🎉
