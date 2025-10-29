# 精度测试支持设计方案（简化版）

## 核心设计原则

- 精度测试不存CSV，只判断pass/fail
- 性能测试继续存CSV（现有逻辑不变）
- 使用test_category区分两种测试类型
- 精度配置放在YAML的metadata部分
- **所有输出统一在 benchmark.log 中**
- **通过正则表达式和关键字匹配提取不同数据集的accuracy值**
- 支持多数据集（gsm8k、mmlu、humaneval等）

## 1. 数据结构扩展 (config_loader.py)

### 1.1 添加DatasetThreshold类

```python
@dataclass
class DatasetThreshold:
    """单个数据集的精度阈值配置"""
    dataset_name: str              # 数据集名称：gsm8k, mmlu, humaneval等
    expected_value: float          # 期望值
    threshold: float               # 阈值
    threshold_type: str            # "relative" 或 "absolute"
    
    def validate(self, actual_value: float) -> tuple[bool, str]:
        """验证精度是否通过"""
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

### 1.2 添加AccuracyConfig类

存储多个数据集的阈值配置：

```python
@dataclass
class AccuracyConfig:
    """精度测试配置（支持多数据集）"""
    datasets: List[DatasetThreshold]  # 数据集阈值列表
    
    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetThreshold]:
        """根据数据集名称获取配置"""
        for ds in self.datasets:
            if ds.dataset_name == dataset_name:
                return ds
        return None
```

### 1.3 修改_load_config_file方法

从YAML的metadata读取精度配置：

```python
# 在metadata中读取accuracy配置
accuracy_config = None
if test_category == "accuracy":
    acc_meta = metadata.get('accuracy', {})
    if acc_meta:
        datasets = []
        # 支持datasets列表配置
        for ds_config in acc_meta.get('datasets', []):
            datasets.append(DatasetThreshold(
                dataset_name=ds_config.get('name', 'gsm8k'),
                expected_value=ds_config.get('expected_value', 0.0),
                threshold=ds_config.get('threshold', 0.01),
                threshold_type=ds_config.get('threshold_type', 'relative')
            ))
        
        accuracy_config = AccuracyConfig(datasets=datasets)
```

### 1.4 更新DEFAULT_METRICS_CONFIG

使用元组key方案，accuracy也使用benchmark.log：

```python
_COMMON_ACCURACY_CONFIG = MetricsConfig(
    log_file="benchmark.log",  # 统一使用benchmark.log
    # 正则提取格式: "gsm8k: acc=0.85" 或 "|gsm8k|acc|0.85|"
    extractor_pattern=r'(\w+)[\s|:]+acc[\s|=:]+([0-9.]+)',
    metric_names=["ACCURACY"]
)

DEFAULT_METRICS_CONFIG = {
    # Disagg 性能测试
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
    
    # Widep 性能测试
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
    
    # 精度测试：复用通用配置
    ("disagg", "accuracy"): _COMMON_ACCURACY_CONFIG,
    ("widep", "accuracy"): _COMMON_ACCURACY_CONFIG,
}
```

## 2. 日志解析扩展 (report.py)

### 2.1 添加AccuracyParser类

从benchmark.log解析多个数据集的accuracy：

```python
class AccuracyParser:
    """精度测试解析器（从benchmark.log提取）"""
    
    def __init__(self, metrics_config: MetricsConfig, accuracy_config: AccuracyConfig, result_dir: str):
        self.metrics_config = metrics_config
        self.accuracy_config = accuracy_config
        self.result_dir = result_dir
    
    def parse_and_validate(self) -> Dict[str, Any]:
        """解析benchmark.log并验证所有数据集的精度"""
        log_file = os.path.join(self.result_dir, self.metrics_config.log_file)
        
        if not os.path.exists(log_file):
            return {"success": False, "error": f"Log file not found: {log_file}"}
        
        # 读取日志
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            log_content = f.read()
        
        # 使用正则提取所有数据集的accuracy
        # 格式示例：
        #   gsm8k: acc=0.85
        #   mmlu: acc=0.75
        #   或：|gsm8k|acc|0.85|
        pattern = re.compile(self.metrics_config.extractor_pattern, re.IGNORECASE)
        matches = pattern.findall(log_content)
        
        if not matches:
            return {"success": False, "error": "No accuracy values found in log"}
        
        # 解析为字典：{dataset_name: accuracy_value}
        parsed_results = {}
        for match in matches:
            dataset_name = match[0].lower()  # 数据集名称（小写）
            acc_value = float(match[1])      # accuracy值
            parsed_results[dataset_name] = acc_value
        
        print(f"   📊 Parsed accuracy results: {parsed_results}")
        
        # 验证每个配置的数据集
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

## 3. 执行器修改 (executor.py)

### 3.1 修改JobManager.check_result方法

添加test_category和accuracy_config参数：

```python
@staticmethod
def check_result(job_id: str, test_config, timestamps, test_name) -> Dict[str, Any]:
    # ... 现有代码 ...
    
    return JobManager._check_job_result(
        job_id=job_id,
        test_category=test_config.test_category,  # 新增
        benchmark_type=test_config.benchmark_type,
        config=config_data,
        metrics_config=test_config.metrics_config,
        accuracy_config=test_config.accuracy_config,  # 新增
        model_name=test_config.model_name,
        result_dir=result_dir,
        timestamps=timestamps,
        test_name=test_name
    )
```

### 3.2 修改_check_job_result方法签名

添加test_category和accuracy_config参数：

```python
@staticmethod
def _check_job_result(job_id: str, test_category: str, benchmark_type: str, 
                     config: dict, metrics_config, accuracy_config, 
                     model_name: str, result_dir: str, 
                     timestamps: Optional[Dict[str, str]] = None, 
                     test_name: Optional[str] = None) -> Dict[str, Any]:
```

### 3.3 在_check_job_result中添加分流逻辑

```python
# ... 打印日志的共通逻辑 ...

# 根据test_category分流
if test_category == "accuracy":
    # 精度测试：不存CSV，只验证pass/fail
    if not accuracy_config:
        return {"success": False, "error": "Accuracy config not found"}
    
    # 解析并验证
    accuracy_parser = AccuracyParser(metrics_config, accuracy_config, result_dir)
    validation_result = accuracy_parser.parse_and_validate()
    
    if not validation_result["success"]:
        result["error"] = validation_result.get("error", "Validation failed")
        return result
    
    # 打印验证结果
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
    # 性能测试：解析并存CSV（现有逻辑不变）
    # ... 现有的perf处理逻辑 ...
```

## 4. TestConfig扩展 (config_loader.py)

在TestConfig添加accuracy_config字段：

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
    accuracy_config: Optional[AccuracyConfig] = None  # 新增
    supported_gpus: List[str]
```

## 5. YAML配置示例

### 5.1 单数据集精度测试

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]
  
  # 精度测试配置
  accuracy:
    datasets:
      - name: "gsm8k"
        expected_value: 0.85
        threshold: 0.02
        threshold_type: "relative"
```

### 5.2 多数据集精度测试

```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  supported_gpus: ["GB200", "GB300"]
  
  # 精度测试配置（多数据集）
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

### 5.3 自定义正则表达式（可选）

如果lm_eval输出格式特殊，可以覆盖默认的正则：

```yaml
benchmark:
  mode: "accuracy"
  # 可选：覆盖默认的正则表达式
  metrics:
    extractor_pattern: r'\|(\w+)\|acc\|([0-9.]+)\|'  # 匹配表格格式
```

## 修改文件清单

1. `config_loader.py` - 添加DatasetThreshold和AccuracyConfig类，扩展TestConfig，修改_load_config_file和_get_metrics_config方法
2. `report.py` - 添加AccuracyParser类（简化版，只解析benchmark.log）
3. `executor.py` - 修改check_result和_check_job_result方法，添加分流逻辑

## 设计要点

1. **统一日志文件**：所有输出都在benchmark.log中，简化文件管理
2. **正则表达式灵活**：支持多种格式（`gsm8k: acc=0.85` 或 `|gsm8k|acc|0.85|`）
3. **多数据集支持**：一次解析提取所有数据集的accuracy，按配置验证
4. **关键字匹配**：通过数据集名称（gsm8k、mmlu等）匹配对应的阈值配置
5. **清晰的输出**：每个数据集的验证结果独立显示，便于调试

## 实施待办

- [ ] 在config_loader.py添加DatasetThreshold数据类
- [ ] 在config_loader.py添加AccuracyConfig数据类（支持多数据集）
- [ ] 更新DEFAULT_METRICS_CONFIG使用元组key方案，accuracy使用benchmark.log
- [ ] 扩展TestConfig添加accuracy_config字段
- [ ] 修改_load_config_file和_get_metrics_config方法，支持从YAML读取多数据集accuracy配置
- [ ] 在report.py添加AccuracyParser类（从benchmark.log解析和验证）
- [ ] 修改executor.py的check_result方法传递test_category和accuracy_config
- [ ] 修改_check_job_result方法添加test_category分流逻辑，集成AccuracyParser

