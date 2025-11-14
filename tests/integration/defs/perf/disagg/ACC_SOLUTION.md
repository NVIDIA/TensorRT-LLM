# Accuracy Test Support Design (Simplified)

## Core Design Principles

- Accuracy tests don't save to CSV, only pass/fail judgment
- Performance tests continue to save CSV (existing logic unchanged)
- Use test_category to distinguish between two test types
- Accuracy configuration in YAML metadata section
- **All outputs unified in benchmark.log**
- **Extract accuracy values for different datasets via regex and keyword matching**
- Support multiple datasets (gsm8k, mmlu, humaneval, etc.)

## 1. Data Structure Extension (config_loader.py)

### 1.1 Add DatasetThreshold Class

```python
@dataclass
class DatasetThreshold:
    """Accuracy threshold configuration for a single dataset"""
    dataset_name: str              # Dataset name: gsm8k, mmlu, humaneval, etc.
    expected_value: float          # Expected value
    threshold: float               # Threshold
    threshold_type: str            # "relative" or "absolute"
    
    def validate(self, actual_value: float) -> tuple[bool, str]:
        """Validate if accuracy passes"""
        if self.threshold_type == "relative":
            error = abs(actual_value - self.expected_value) / self.expected_value
            passed = error < self.threshold
            msg = f"Relative error: {error:.6f} (threshold: {self.threshold})"
        else:  # absolute
            error = abs(actual_value - self.expected_value)
            passed = error < self.threshold
            msg = f"Absolute error: {error:.6f} (threshold: {self.threshold})"
        
        return passed, msg
```

### 1.2 Add AccuracyConfig Class

Store threshold configurations for multiple datasets:

```python
@dataclass
class AccuracyConfig:
    """Accuracy test configuration (supports multiple datasets)"""
    datasets: List[DatasetThreshold]  # List of dataset thresholds
    
    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetThreshold]:
        """Get configuration by dataset name"""
        for ds in self.datasets:
            if ds.dataset_name == dataset_name:
                return ds
        return None
```

### 1.3 Modify _load_config_file Method

Read accuracy configuration from YAML metadata:

```python
# Read accuracy config from metadata
accuracy_config = None
if test_category == "accuracy":
    acc_meta = metadata.get('accuracy', {})
    if acc_meta:
        datasets = []
        # Support datasets list configuration
        for ds_config in acc_meta.get('datasets', []):
            datasets.append(DatasetThreshold(
                dataset_name=ds_config.get('name', 'gsm8k'),
                expected_value=ds_config.get('expected_value', 0.0),
                threshold=ds_config.get('threshold', 0.01),
                threshold_type=ds_config.get('threshold_type', 'relative')
            ))
        
        accuracy_config = AccuracyConfig(datasets=datasets)
```

### 1.4 Update DEFAULT_METRICS_CONFIG

Use tuple key approach, accuracy also uses benchmark.log:

```python
_COMMON_ACCURACY_CONFIG = MetricsConfig(
    log_file="benchmark.log",  # Unified use of benchmark.log
    # Regex extraction format: "gsm8k: acc=0.85" or "|gsm8k|acc|0.85|"
    extractor_pattern=r'(\w+)[\s|:]+acc[\s|=:]+([0-9.]+)',
    metric_names=["ACCURACY"]
)

DEFAULT_METRICS_CONFIG = {
    # Disagg performance test
    ("disagg", "perf"): MetricsConfig(
        log_file="benchmark.log",
        extractor_pattern=r"""
            ^.*?Median\ TTFT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Median\ E2EL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Benchmark\ with\ concurrency\ (\d+)\ done
        """,
        metric_names=["DISAGG_SERVER_TTFT", "DISAGG_SERVER_E2EL"]
    ),
    
    # Widep performance test
    ("widep", "perf"): MetricsConfig(
        log_file="benchmark.log",
        extractor_pattern=r"""
            ^.*?Median\ TTFT\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Median\ E2EL\ \(ms\):\s+([0-9.]+).*?$\n
            ^.*?(?:\n|.)*?$\n
            ^.*?Benchmark\ with\ concurrency\ (\d+)\ done
        """,
        metric_names=["WIDEP_SERVER_TTFT", "WIDEP_SERVER_E2EL"]
    ),
    
    # Accuracy test: reuse common config
    ("disagg", "accuracy"): _COMMON_ACCURACY_CONFIG,
    ("widep", "accuracy"): _COMMON_ACCURACY_CONFIG,
}
```

## 2. Log Parsing Extension (report.py)

### 2.1 Add AccuracyParser Class

Parse accuracy from benchmark.log for multiple datasets:

```python
class AccuracyParser:
    """Accuracy test parser (extract from benchmark.log)"""
    
    def __init__(self, metrics_config: MetricsConfig, accuracy_config: AccuracyConfig, result_dir: str):
        self.metrics_config = metrics_config
        self.accuracy_config = accuracy_config
        self.result_dir = result_dir
    
    def parse_and_validate(self) -> Dict[str, Any]:
        """Parse benchmark.log and validate accuracy for all datasets"""
        log_file = os.path.join(self.result_dir, self.metrics_config.log_file)
        
        if not os.path.exists(log_file):
            return {"success": False, "error": f"Log file not found: {log_file}"}
        
        # Read log file
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            log_content = f.read()
        
        # Extract accuracy for all datasets using regex
        # Format examples:
        #   gsm8k: acc=0.85
        #   mmlu: acc=0.75
        #   Or: |gsm8k|acc|0.85|
        pattern = re.compile(self.metrics_config.extractor_pattern, re.IGNORECASE)
        matches = pattern.findall(log_content)
        
        if not matches:
            return {"success": False, "error": "No accuracy values found in log"}
        
        # Parse to dictionary: {dataset_name: accuracy_value}
        parsed_results = {}
        for match in matches:
            dataset_name = match[0].lower()  # Dataset name (lowercase)
            acc_value = float(match[1])      # Accuracy value
            parsed_results[dataset_name] = acc_value
        
        print(f"   📊 Parsed accuracy results: {parsed_results}")
        
        # Validate each configured dataset
        validation_results = []
        all_passed = True
        
        for dataset_config in self.accuracy_config.datasets:
            dataset_name = dataset_config.dataset_name.lower()
            
            if dataset_name not in parsed_results:
                validation_results.append({
                    "dataset": dataset_config.dataset_name,
                    "passed": False,
                    "error": f"Dataset {dataset_config.dataset_name} not found in log"
                })
                all_passed = False
                continue
            
            actual_value = parsed_results[dataset_name]
            passed, msg = dataset_config.validate(actual_value)
            
            validation_results.append({
                "dataset": dataset_config.dataset_name,
                "passed": passed,
                "actual": actual_value,
                "expected": dataset_config.expected_value,
                "threshold": dataset_config.threshold,
                "threshold_type": dataset_config.threshold_type,
                "message": msg
            })
            
            if not passed:
                all_passed = False
        
        return {
            "success": True,
            "all_passed": all_passed,
            "results": validation_results
        }
```

## 3. Executor Modifications (executor.py)

### 3.1 Modify JobManager.check_result Method

Add test_category and accuracy_config parameters:

```python
@staticmethod
def check_result(job_id: str, test_config, timestamps, test_name) -> Dict[str, Any]:
    # ... existing code ...
    
    return JobManager._check_job_result(
        job_id=job_id,
        test_category=test_config.test_category,  # New
        benchmark_type=test_config.benchmark_type,
        config=config_data,
        metrics_config=test_config.metrics_config,
        accuracy_config=test_config.accuracy_config,  # New
        model_name=test_config.model_name,
        result_dir=result_dir,
        timestamps=timestamps,
        test_name=test_name
    )
```

### 3.2 Modify _check_job_result Method Signature

Add test_category and accuracy_config parameters:

```python
@staticmethod
def _check_job_result(job_id: str, test_category: str, benchmark_type: str, 
                     config: dict, metrics_config, accuracy_config, 
                     model_name: str, result_dir: str, 
                     timestamps: Optional[Dict[str, str]] = None, 
                     test_name: Optional[str] = None) -> Dict[str, Any]:
```

### 3.3 Add Routing Logic in _check_job_result

```python
# ... Common logging logic ...

# Route based on test_category
if test_category == "accuracy":
    # Accuracy test: don't save to CSV, only validate pass/fail
    if not accuracy_config:
        return {"success": False, "error": "Accuracy config not found"}
    
    # Parse and validate
    accuracy_parser = AccuracyParser(metrics_config, accuracy_config, result_dir)
    validation_result = accuracy_parser.parse_and_validate()
    
    if not validation_result["success"]:
        result["error"] = validation_result.get("error", "Validation failed")
        return result
    
    # Print validation results
    print(f"   📊 Accuracy Validation Results:")
    all_passed = validation_result["all_passed"]
    
    for ds_result in validation_result["results"]:
        status_icon = "✅" if ds_result["passed"] else "❌"
        print(f"      {status_icon} {ds_result['dataset']}:")
        if "error" in ds_result:
            print(f"         Error: {ds_result['error']}")
        else:
            print(f"         Expected: {ds_result['expected']}")
            print(f"         Actual: {ds_result['actual']}")
            print(f"         Threshold: {ds_result['threshold']} ({ds_result['threshold_type']})")
            print(f"         {ds_result['message']}")
    
    if all_passed:
        print(f"   ✅ All accuracy tests PASSED")
        result["success"] = True
        result["status"] = "PASSED"
    else:
        print(f"   ❌ Some accuracy tests FAILED")
        result["success"] = False
        result["status"] = "FAILED"
    
    result.update(validation_result)
    return result

else:  # perf
    # Performance test: parse and save to CSV (existing logic unchanged)
    # ... existing perf handling logic ...
```

## 4. TestConfig Extension (config_loader.py)

Add accuracy_config field to TestConfig:

```python
@dataclass
class TestConfig:
    config_path: str
    test_id: str
    test_type: str
    model_name: str
    test_category: str
    benchmark_type: str
    config_data: dict
    metrics_config: MetricsConfig
    accuracy_config: Optional[AccuracyConfig] = None  # New
    supported_gpus: List[str]
```

## 5. YAML Configuration Examples

### 5.1 Single Dataset Accuracy Test

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]
  
  # Accuracy test configuration
  accuracy:
    datasets:
      - name: "gsm8k"
        expected_value: 0.85
        threshold: 0.02
        threshold_type: "relative"
```

### 5.2 Multi-Dataset Accuracy Test

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]
  
  # Accuracy test configuration (multiple datasets)
  accuracy:
    datasets:
      - name: "gsm8k"
        expected_value: 0.85
        threshold: 0.02
        threshold_type: "relative"
      
      - name: "mmlu"
        expected_value: 0.75
        threshold: 0.03
        threshold_type: "relative"
      
      - name: "humaneval"
        expected_value: 0.70
        threshold: 0.05
        threshold_type: "absolute"
```

### 5.3 Custom Regex Pattern (Optional)

If lm_eval output format is special, you can override the default regex:

```yaml
benchmark:
  mode: "accuracy"
  # Optional: override default regex pattern
  metrics:
    extractor_pattern: r'\|(\w+)\|acc\|([0-9.]+)\|'  # Match table format
```

## File Modification Checklist

1. `config_loader.py` - Add DatasetThreshold and AccuracyConfig classes, extend TestConfig, modify _load_config_file and _get_metrics_config methods
2. `report.py` - Add AccuracyParser class (simplified version, only parses benchmark.log)
3. `executor.py` - Modify check_result and _check_job_result methods, add routing logic

## Design Key Points

1. **Unified log file**: All outputs in benchmark.log, simplifying file management
2. **Flexible regex**: Supports multiple formats (`gsm8k: acc=0.85` or `|gsm8k|acc|0.85|`)
3. **Multi-dataset support**: Parse and extract all dataset accuracy in one pass, validate by configuration
4. **Keyword matching**: Match corresponding threshold config by dataset name (gsm8k, mmlu, etc.)
5. **Clear output**: Validation results for each dataset displayed independently for easy debugging

## Implementation TODO

- [ ] Add DatasetThreshold dataclass in config_loader.py
- [ ] Add AccuracyConfig dataclass in config_loader.py (supporting multiple datasets)
- [ ] Update DEFAULT_METRICS_CONFIG to use tuple key approach, accuracy uses benchmark.log
- [ ] Extend TestConfig to add accuracy_config field
- [ ] Modify _load_config_file and _get_metrics_config methods to support reading multi-dataset accuracy config from YAML
- [ ] Add AccuracyParser class in report.py (parse and validate from benchmark.log)
- [ ] Modify check_result method in executor.py to pass test_category and accuracy_config
- [ ] Modify _check_job_result method to add test_category routing logic and integrate AccuracyParser
