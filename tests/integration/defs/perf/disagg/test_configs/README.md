# Test Configurations

本目录包含基于 YAML 的测试配置文件，用于 disaggregated benchmark 测试。

## 目录结构

```
test_configs/
└── disagg/                    # Disaggregated 测试类型
    ├── perf/                  # 性能测试配置
    └── accuracy/              # 精度测试配置（待添加）
```

## 配置文件命名规范

文件名格式：`{model}_{benchmark_type}_{config_details}.yaml`

- **model**: 模型名称（如 deepseek-r1-fp4, Qwen3-8B-FP8）
- **benchmark_type**: 基准类型（如 1k1k, 8k1k）
- **config_details**: 配置详情（如 tep8_bs32_mtp3_nixl）

### 配置详情说明

- **tep/dep**: Tensor Parallel (TEP) 或 Data Parallel with attention_dp (DEP)
- **数字**: TP size (如 tep8 表示 TP=8, dep16 表示 TP=16+DP)
- **bs**: Batch Size (如 bs32 表示 batch_size=32)
- **mtp**: MTP layers (如 mtp3 表示 3 层 MTP，无 mtp 表示不使用）
- **nixl/ucx**: Cache transceiver backend (NIXL 或 UCX)

## 生成的配置文件列表

### DeepSeek R1 FP4 (1k1k)

1. `deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_nixl.yaml` - TEP8, BS32, MTP3, NIXL
2. `deepseek-r1-fp4_1k1k_tep8_bs32_nixl.yaml` - TEP8, BS32, 无MTP, NIXL
3. `deepseek-r1-fp4_1k1k_dep16_bs128_mtp3_nixl.yaml` - DEP16, BS128, MTP3, NIXL (高并发)
4. `deepseek-r1-fp4_1k1k_dep32_bs32_nixl.yaml` - DEP32, BS32, 无MTP, NIXL (超高TP)
5. `deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_ucx.yaml` - TEP8, BS32, MTP3, UCX backend

### DeepSeek R1 FP4 (8k1k)

6. `deepseek-r1-fp4_8k1k_tep8_bs16_mtp3_nixl.yaml` - TEP8, BS16, MTP3, NIXL (长序列)

### Qwen3-235B-A22B FP4

7. `Qwen3-235B-A22B-FP4_1k1k_tep8_bs32_mtp3_nixl.yaml` - TEP8, BS32, MTP3, NIXL

### DeepSeek V3 Lite FP8

8. `deepseek-v3-lite-fp8_1k1k_tep4_bs4_ucx.yaml` - TEP4, BS4, UCX (小模型)

### Qwen3-8B FP8

9. `Qwen3-8B-FP8_1k1k_tep4_bs4_nixl.yaml` - TEP4, BS4, NIXL (小模型)

## 配置来源

所有配置均从 `disagg_config.py` 中的 `DisaggConfig.MODEL_CONFIGS` 转换而来。

## Metadata 字段说明

每个 YAML 配置文件都包含 `metadata` 节点，用于标识测试元数据：

```yaml
metadata:
  model_name: "deepseek-r1-fp4"     # 模型名称
  precision: "fp4"                   # 精度类型（fp4, fp8等）
  supported_gpus: ["GB200", "GB300"] # 支持的 GPU 类型列表
```

### benchmark_type 动态生成

`benchmark_type`（如 1k1k, 8k1k）不再从文件名解析，而是从 YAML 的 `sequence` 配置动态生成：

- `input_length: 1024, output_length: 1024` → `1k1k`
- `input_length: 8192, output_length: 1024` → `8k1k`
- `input_length: 16384, output_length: 2048` → `16k2k`

**优势**：
- ✅ 配置文件是唯一真实来源（Single Source of Truth）
- ✅ 避免文件名与实际配置不一致
- ✅ 便于程序化修改配置而无需重命名文件
- ✅ `metadata` 字段便于未来扩展其他元数据

### GPU 类型过滤

系统会根据当前 GPU 类型（通过 `GPU_TYPE` 环境变量）自动过滤配置：

```bash
export GPU_TYPE=GB200
python list_configs.py  # 只显示支持 GB200 的配置

export GPU_TYPE=H100
python list_configs.py  # 只显示支持 H100 的配置
```

## 使用方法

### 1. 使用 `list_configs.py` 查看配置

```bash
# 列出所有配置
python list_configs.py

# 查看性能测试配置
python list_configs.py --category perf -v

# 查看特定模型配置
python list_configs.py --model deepseek-r1-fp4 --show-metrics
```

### 2. 使用 `submit.py` 提交单个作业

```bash
python disagg/slurm/benchmark/submit.py -c test_configs/disagg/perf/deepseek-r1-fp4_1k1k_tep8_bs32_mtp3_nixl.yaml
```

### 3. 使用 `submit.py` 批量提交目录中的作业

```bash
python disagg/slurm/benchmark/submit.py -d test_configs/disagg/perf/
```

### 4. 使用 pytest 运行测试

```bash
# 运行所有测试
pytest test_disagg_yaml.py -v

# 只运行性能测试
pytest test_disagg_yaml.py -k "perf" -v

# 运行特定模型
pytest test_disagg_yaml.py -k "deepseek-r1-fp4" -v

# 查看详细输出
pytest test_disagg_yaml.py -s -vv
```

## Metrics 配置

所有性能测试配置默认使用以下 metrics：

- **log_file**: `benchmark_result.log`
- **metric_names**: `["DISAGG_SERVER_TTFT", "DISAGG_SERVER_E2EL"]`
- **extractor_pattern**: 预定义的 TTFT/E2EL 提取模式

如需自定义 metrics，可在 YAML 文件的 `benchmark` 节点下添加 `metrics` 配置。详见 `solution4.md`。

## 注意事项

1. **环境路径**: 所有路径（container_image, model_path, work_dir 等）均基于 Lustre 文件系统
2. **GPU 类型**: 大部分配置针对 GB200 GPU，部分小模型配置支持 H100/B200/B300
3. **并发列表**: `concurrency_list` 根据配置的吞吐量能力进行调整
4. **默认配置**: 所有配置均使用默认的 perf metrics，无需在 YAML 中显式配置

## 扩展配置

要添加新的测试配置：

1. 在对应的目录下创建新的 YAML 文件
2. 遵循文件命名规范
3. 参考现有配置文件的结构
4. 根据需要覆盖默认 metrics 配置

## 配置映射

| 原 BenchmarkConfig 参数 | YAML 配置路径 |
|-------------------------|---------------|
| `isl`, `osl` | `sequence.input_length`, `sequence.output_length` |
| `cache_transceiver_max_num_tokens` | `worker_config.gen.cache_transceiver_config.max_tokens_in_buffer` |
| `cache_transceiver_backend` | `worker_config.gen.cache_transceiver_config.backend` |
| `ctx_num`, `gen_num` | `hardware.num_ctx_servers`, `hardware.num_gen_servers` |
| `gen_tp_size` | `worker_config.gen.tensor_parallel_size` |
| `gen_batch_size` | `worker_config.gen.max_batch_size` |
| `gen_enable_attention_dp` | `worker_config.gen.enable_attention_dp` |
| `gen_mtp_size` | `worker_config.gen.speculative_config.num_nextn_predict_layers` |
| `concurrency_list` | `benchmark.concurrency_list` |
| `extractor_pattern`, `metric_names` | `benchmark.metrics` (可选，默认使用 perf 配置) |
