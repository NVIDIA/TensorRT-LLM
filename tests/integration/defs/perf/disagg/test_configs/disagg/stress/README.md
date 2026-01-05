# Disaggregated Stress Tests

## Purpose

Stress tests combine **performance benchmarking** and **accuracy validation** in a single test run. They are designed to:

- Validate performance under high load/stress conditions
- Ensure accuracy is maintained while pushing system limits
- Write performance metrics to CSV (same as `perf` tests)
- Validate accuracy against expected thresholds (e.g., GSM8K, MMLU)

Test name prefix: `disagg_stress_*`

---

## Quick Start

```bash
# 1. Copy the example template
cp EXAMPLE_deepseek-r1-fp4_1k1k_stress_gsm8k.yaml \
   your_model_1k1k_stress_gsm8k.yaml

# 2. Edit the configuration (see Field Reference below)

# 3. Run the test
cd /path/to/tests/integration/defs/perf/disagg/
poetry run pytest --disagg test_disagg.py -s -vv -m stress
```

---

## Configuration Template

### Minimal Template

```yaml
metadata:
  model_name: your-model-name
  precision: fp8
  model_dir_name: YourModelDir
  supported_gpus: [GB200, GB300]
  script_file: disaggr_torch.slurm
  benchmark_type: 1k1k
  config_index: 0
  
  # Accuracy configuration (required for stress tests)
  accuracy:
    datasets:
    - name: gsm8k
      expected_value: 0.85
      threshold_type: hypothesis_test
      filter_type: flexible-extract

slurm:
  script_file: disaggr_torch.slurm
  partition: <partition>
  account: <account>
  job_time: 04:00:00
  job_name: stress-benchmark
  extra_args: "--gres=gpu:4"
  numa_bind: true

benchmark:
  mode: e2e
  use_nv_sa_benchmark: true
  multi_round: 8
  benchmark_ratio: 0.8
  streaming: true
  concurrency_list: 1 2 4 8 16 32
  input_length: 1024
  output_length: 1024
  dataset_file: <dataset_file>

hardware:
  gpus_per_node: 4
  num_ctx_servers: 1
  num_gen_servers: 4

environment:
  container_mount: <container_mount>
  container_image: <container_image>
  model_path: <model_path>
  trtllm_repo: ''
  build_wheel: false
  work_dir: <full_path_to_work_dir>
  worker_env_var: "TLLM_LOG_LEVEL=INFO ..."
  server_env_var: "TRTLLM_SERVER_DISABLE_GC=1"

profiling:
  nsys_on: false

# Enable accuracy evaluation (required for stress tests)
accuracy:
  enable_accuracy_test: true
  model: local-completions
  tasks: gsm8k
  model_args_extra: num_concurrent=512,max_retries=3,timeout=1200

worker_config:
  gen:
    tensor_parallel_size: 8
    max_batch_size: 32
    max_num_tokens: 128
    # ... other gen worker configs
  ctx:
    tensor_parallel_size: 4
    max_batch_size: 4
    max_num_tokens: 4608
    # ... other ctx worker configs
```

---

## Field Reference

### 1. `metadata` Section

#### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `model_name` | string | Model identifier | `deepseek-r1-fp4` |
| `precision` | string | Model precision | `fp8`, `fp4`, `int8` |
| `model_dir_name` | string | Model directory name | `DeepSeek-R1-0528-FP4-v2` |
| `supported_gpus` | list | GPU types supported | `[GB200, GB300]` |
| `script_file` | string | SLURM script to use | `disaggr_torch.slurm` |
| `benchmark_type` | string | Benchmark configuration | `1k1k`, `8k1k`, etc. |
| `config_index` | int | Configuration index | `0`, `1`, etc. |

#### Accuracy Configuration (Required for Stress Tests)

```yaml
metadata:
  accuracy:
    datasets:
    - name: gsm8k                          # Dataset name
      expected_value: 0.85                 # Expected accuracy (0.0-1.0)
      threshold_type: hypothesis_test      # "hypothesis_test" or "absolute"
      filter_type: flexible-extract        # "flexible-extract" or "strict-match"
      
      # Optional: Hypothesis testing parameters
      alpha: 0.05                          # Type I error rate (default: 0.05)
      beta: 0.20                           # Type II error rate (default: 0.20)
      sigma: 0.05                          # Standard deviation (default: 0.05)
      num_samples: 100                     # Number of samples (default: 100)
      higher_is_better: true               # Direction (default: true)
    
    # Optional: Custom accuracy metrics parsing
    # metrics:
    #   log_file: "7_accuracy_eval.log"
    #   extractor_pattern: '\|...\|'
    #   metric_names: [flexible-extract, strict-match]
```

**Threshold Types:**
- `hypothesis_test`: Statistical hypothesis testing (recommended)
- `absolute`: Simple threshold comparison

**Filter Types:**
- `flexible-extract`: More lenient matching
- `strict-match`: Exact matching required

---

### 2. `slurm` Section

| Field | Type | Description | Recommended |
|-------|------|-------------|-------------|
| `partition` | string | SLURM partition | Your cluster partition |
| `account` | string | SLURM account | Your cluster account |
| `job_time` | string | Maximum job time | `04:00:00` (4 hours) |
| `job_name` | string | Job name | `stress-benchmark` |
| `extra_args` | string | Extra SLURM args | `"--gres=gpu:4"` |
| `numa_bind` | bool | Enable NUMA binding | `true` |

---

### 3. `benchmark` Section

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `mode` | string | Benchmark mode | `e2e` |
| `use_nv_sa_benchmark` | bool | Use NV benchmark | `true` |
| `multi_round` | int | Rounds per concurrency | `8` |
| `benchmark_ratio` | float | Benchmark ratio | `0.8` |
| `streaming` | bool | Enable streaming | `true` |
| `concurrency_list` | string | Concurrency levels | `1 2 4 8 16 32` |
| `input_length` | int | Input token length | `1024` |
| `output_length` | int | Output token length | `1024` |
| `dataset_file` | string | Dataset file path | `<dataset_file>` |

**Tip:** Increase `concurrency_list` for more stress (e.g., `1 2 4 8 16 32 64 128`)

---

### 4. `hardware` Section

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `gpus_per_node` | int | GPUs per node | `4` |
| `num_ctx_servers` | int | Context servers | `1` |
| `num_gen_servers` | int | Generation servers | `4` |

---

### 5. `accuracy` Section (SLURM Script Config)

**Note:** This is different from `metadata.accuracy`. This section is used by the SLURM script to run `lm-evaluation-harness`.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `enable_accuracy_test` | bool | Enable accuracy eval | `true` (required) |
| `model` | string | Model type | `local-completions` |
| `tasks` | string | Eval tasks | `gsm8k`, `mmlu`, `humaneval` |
| `model_args_extra` | string | Extra arguments | See below |

**Common `model_args_extra` parameters:**
```
num_concurrent=512,max_retries=3,tokenized_requests=false,timeout=1200,max_gen_toks=256,max_length=4096
```

---

### 6. `worker_config` Section

Configure generation and context workers. See [main README](../../README.md) for detailed worker configuration options.

**Key parameters:**
- `tensor_parallel_size`: TP parallelism
- `max_batch_size`: Maximum batch size
- `max_num_tokens`: Maximum tokens per batch
- `max_seq_len`: Maximum sequence length

---

## Test Execution Flow

```
1. Configuration Validation
   ↓
2. SLURM Job Submission
   ↓
3. Performance Benchmark
   - Runs benchmark with specified concurrency levels
   - Generates: 6_bench.log
   ↓
4. Accuracy Evaluation
   - Runs lm-evaluation-harness
   - Generates: 7_accuracy_eval.log
   ↓
5. Result Validation
   - Parse performance metrics → Write to CSV
   - Parse accuracy results → Validate against thresholds
   ↓
6. Pass/Fail Decision
   - PASS: Both performance and accuracy checks pass
   - FAIL: Either performance or accuracy fails
```

---

## Output Files

### Log Directory
```
{OUTPUT_PATH}/slurm_logs/disagg_stress_{test_id}/
├── config.yaml                 # Test configuration copy
├── 6_bench.log                 # Performance benchmark log
├── 7_accuracy_eval.log         # Accuracy evaluation log
├── output_gen_*.log            # Generation worker logs
├── output_ctx_*.log            # Context worker logs
└── slurm-{job_id}.out          # SLURM output
```

### CSV Output
```
{OUTPUT_PATH}/perf_script_test_results.csv
```

Performance metrics are written to the same CSV as `perf` tests, with `test_name` prefix `disagg_stress_*`.

### Failed Test Directories
Failed tests are automatically renamed with `_ERROR` suffix:
```
disagg_stress_{test_id}_ERROR/
```

---

## Supported Accuracy Datasets

| Dataset | Task | Description |
|---------|------|-------------|
| `gsm8k` | Math reasoning | Grade school math problems |
| `mmlu` | Knowledge | Multi-domain multiple choice |
| `humaneval` | Coding | Python code generation |
| `hellaswag` | Reasoning | Commonsense reasoning |

See [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for full list.

---

## Common Pitfalls

### 1. Missing Accuracy Config

**Error:** `Stress test has no accuracy_config`

**Solution:** Ensure both accuracy sections are present:
```yaml
metadata:
  accuracy:
    datasets: [...]  # For validation framework

accuracy:
  enable_accuracy_test: true  # For SLURM script
```

### 2. Timeout Issues

**Error:** Job times out before completion

**Solution:** Increase `job_time`:
```yaml
slurm:
  job_time: 06:00:00  # 6 hours for larger models
```

### 3. Accuracy Threshold Too High

**Error:** Accuracy test fails but model performance is reasonable

**Solution:** Adjust `expected_value` or use `hypothesis_test`:
```yaml
metadata:
  accuracy:
    datasets:
    - expected_value: 0.80  # Lower threshold
      threshold_type: hypothesis_test  # More statistical
```

---

## Advanced Usage

### Custom Accuracy Metrics Parsing

Override default accuracy log parsing:

```yaml
metadata:
  accuracy:
    datasets:
    - name: gsm8k
      expected_value: 0.85
    
    metrics:
      log_file: "custom_accuracy.log"
      extractor_pattern: '\|custom_pattern\|'
      metric_names: [custom_metric_1, custom_metric_2]
```

### Multiple Datasets

Test multiple accuracy benchmarks:

```yaml
metadata:
  accuracy:
    datasets:
    - name: gsm8k
      expected_value: 0.85
      threshold_type: hypothesis_test
    - name: mmlu
      expected_value: 0.75
      threshold_type: absolute
```

---

## Running Tests

### Run All Stress Tests
```bash
poetry run pytest --disagg test_disagg.py -s -vv -m stress
```

### Run Specific Test
```bash
poetry run pytest --disagg test_disagg.py -s -vv -k "your_model_1k1k_stress"
```

### Run from Test List
```bash
echo "disagg_stress_your_model_1k1k_stress_gsm8k" > testlist/stress.txt
poetry run pytest --disagg test_disagg.py -s -vv --disagg-test-list=./testlist/stress.txt
```

---

## Naming Convention

Format: `{model}_{benchmark_type}_{config_details}_stress_{dataset}.yaml`

Examples:
- `deepseek-r1-fp4_1k1k_ctx1_gen4_stress_gsm8k.yaml`
- `llama3-8b_8k1k_ctx2_gen2_stress_mmlu.yaml`
- `qwen3-235b_1k1k_ctx1_gen1_stress_humaneval.yaml`

---

## Comparison with Other Test Types

| Feature | perf | accuracy | stress |
|---------|------|----------|--------|
| Performance Metrics | ✅ | ❌ | ✅ |
| CSV Output | ✅ | ❌ | ✅ |
| Accuracy Validation | ❌ | ✅ | ✅ |
| Default Timeout | 2h | 3h | 4h |
| Use Case | Performance only | Accuracy only | Both |

---

## Troubleshooting

### Check Test Status
```bash
# View SLURM jobs
squeue -u $USER

# Check logs
tail -f {OUTPUT_PATH}/slurm_logs/disagg_stress_{test_id}/slurm-*.out
```

### Debug Mode
```bash
export DEBUG_MODE=1
export DEBUG_JOB_ID=12345

poetry run pytest --disagg test_disagg.py -s -vv -k "your_test"
```

### View Results
```bash
# Performance CSV
cat {OUTPUT_PATH}/perf_script_test_results.csv

# Accuracy log
cat {OUTPUT_PATH}/slurm_logs/disagg_stress_{test_id}/7_accuracy_eval.log
```

---

## Best Practices

1. **Start Conservative:** Begin with lower concurrency and shorter job times
2. **Monitor Resources:** Check GPU memory and CPU usage during stress tests
3. **Baseline First:** Run `perf` and `accuracy` tests separately before `stress`
4. **Document Results:** Keep records of thresholds and performance baselines
5. **Iterate:** Gradually increase stress (concurrency, sequence length) until failure

---

## Related Documentation

- [Main README](../../README.md) - General test framework documentation
- [Example Config](EXAMPLE_deepseek-r1-fp4_1k1k_stress_gsm8k.yaml) - Full example configuration
- [Config Loader](../../utils/config_loader.py) - Configuration loading logic
- [Executor](../../execution/executor.py) - Test execution logic

---

## Support

For issues or questions:
1. Check logs in `{OUTPUT_PATH}/slurm_logs/disagg_stress_{test_id}/`
2. Review configuration against this README
3. Compare with `EXAMPLE_deepseek-r1-fp4_1k1k_stress_gsm8k.yaml`
4. Contact your team's test infrastructure maintainer
