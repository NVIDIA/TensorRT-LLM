# Disagg 提交脚本对比分析

## 概述

本文档详细对比了两个 disagg 提交系统的差异：
- **disagg_acc/submit.sh** - 老版本 Shell 脚本方式
- **disagg/slurm/benchmark/submit.py** - 新版本 Python + YAML 配置方式

---

## 一、核心差异总结

| 维度 | disagg_acc (旧版) | disagg (新版) |
|------|------------------|---------------|
| **实现语言** | Pure Shell | Python + Shell |
| **配置方式** | 硬编码循环 | YAML 配置文件 |
| **参数传递** | 位置参数 (14个) | YAML驱动 (28个) |
| **节点计算** | 简单公式 | 基于 TP 和 GPU/节点 |
| **配置管理** | 单个 config.yaml | ctx_config.yaml + gen_config.yaml 分离 |
| **扩展性** | 低（需修改脚本） | 高（只需修改YAML） |
| **可维护性** | 差 | 好 |

---

## 二、ctx_num 计算逻辑对比

### 2.1 disagg_acc/submit.sh 的计算逻辑

```bash
# 文件：disagg_acc/submit.sh
for b in 1024; do
    concurrency=$((b * 8))                    # concurrency = 1024 * 8 = 8192
    ctx_num=$(((concurrency + 5499)/5500))    # ctx_num = (8192 + 5499) / 5500 = 2
    total_gpu_num=$((ctx_num + 2))            # total_gpu_num = 2 + 2 = 4 (ctx + gen)
    total_tasks=$((total_gpu_num * 4))        # total_tasks = 4 * 4 = 16
done
```

**计算公式：**
```
ctx_num = ceil(concurrency / 5500)
total_nodes = ctx_num + gen_nodes (硬编码为2)
```

**特点：**
- 假设每个 ctx server 处理 5500 并发
- gen_nodes 硬编码（dep8=2, dep16=4, dep32=8）
- 简单但不灵活

### 2.2 disagg/submit.py 的计算逻辑

```python
# 文件：disagg/slurm/benchmark/submit.py

def calculate_nodes(tp_size, num_servers, gpus_per_node):
    """Calculate required nodes based on tensor parallel size and server count."""
    return (tp_size + gpus_per_node - 1) // gpus_per_node * num_servers

# 实际计算
ctx_nodes = calculate_nodes(ctx_tp_size, ctx_num, gpus_per_node)
gen_nodes = calculate_nodes(gen_tp_size, gen_num, gpus_per_node)
total_nodes = ctx_nodes + gen_nodes
```

**计算公式：**
```
nodes_per_server = ceil(tp_size / gpus_per_node)
total_nodes = nodes_per_server * num_servers
```

**特点：**
- 基于 TP size 和 GPU/节点动态计算
- 支持任意 TP 配置
- ctx_num 和 gen_num 由 YAML 配置决定
- 更通用、更灵活

**示例对比：**
```
场景：ctx_tp=4, ctx_num=2, gen_tp=32, gen_num=1, gpus_per_node=4

旧版（硬编码）:
  ctx_num = 2 (配置固定)
  gen_nodes = 8 (dep32 硬编码)
  total = 10 nodes

新版（动态计算）:
  ctx_nodes = ceil(4/4) * 2 = 1 * 2 = 2
  gen_nodes = ceil(32/4) * 1 = 8 * 1 = 8
  total = 10 nodes
```

---

## 三、参数传递对比

### 3.1 disagg_acc 参数传递（14个位置参数）

```bash
# submit.sh -> disaggr_torch.slurm
sbatch ... disaggr_torch.slurm \
    ${ctx_num}              # $1
    4                       # $2 - ctx_tp_size
    4                       # $3 - ctx_batch_size
    4480                    # $4 - ctx_max_num_tokens
    true                    # $5 - ctx_enable_attention_dp
    1                       # $6 - num_gen_servers
    8                       # $7 - gen_tp_size
    1024                    # $8 - gen_batch_size
    1024                    # $9 - gen_max_num_tokens
    true                    # $10 - gen_enable_attention_dp
    "0.8"                   # $11 - gen_gpu_memory_fraction
    0                       # $12 - eplb_num_slots
    "$mtp_size"             # $13 - mtp_size
    "$concurrency"          # $14 - concurrency
```

**问题：**
- 参数顺序固定，易出错
- 难以扩展（增加参数需要修改所有调用）
- 可读性差
- 没有参数验证

### 3.2 disagg 参数传递（28个位置参数 + YAML配置）

```python
# submit.py -> disaggr_torch.slurm
cmd = [
    'sbatch',
    # SLURM 配置通过命令行参数
    f'--partition={slurm_config["partition"]}',
    f'--account={slurm_config["account"]}',
    # ...
    slurm_config['script_file'],
    
    # 硬件配置（6个）
    str(hw_config['gpus_per_node']),        # $1
    str(slurm_config['numa_bind']),         # $2
    str(ctx_nodes),                         # $3
    str(gen_nodes),                         # $4
    str(ctx_tp_size),                       # $5
    str(gen_tp_size),                       # $6
    
    # Worker 配置（5个）
    str(ctx_num),                           # $7
    ctx_config_path,                        # $8 - YAML路径
    str(gen_num),                           # $9
    gen_config_path,                        # $10 - YAML路径
    config['benchmark']['concurrency_list'], # $11
    
    # Benchmark 配置（7个）
    str(config['sequence']['input_length']), # $12
    str(config['sequence']['output_length']),# $13
    str(config['benchmark']['multi_round']), # $14
    str(config['benchmark']['benchmark_ratio']), # $15
    str(config['benchmark']['streaming']),   # $16
    str(config['benchmark']['use_nv_sa_benchmark']), # $17
    config['benchmark']['mode'],             # $18
    str(config['worker_config']['gen']['cache_transceiver_config']['max_tokens_in_buffer']), # $19
    
    # 环境配置（8个）
    env_config['dataset_file'],              # $20
    env_config['model_path'],                # $21
    env_config['trtllm_repo'],               # $22
    env_config['work_dir'],                  # $23
    log_dir,                                 # $24
    env_config['container_mount'],           # $25
    env_config['container_image'],           # $26
    str(env_config['build_wheel']),          # $27
    
    # Profiling（1个）
    str(config['profiling']['nsys_on'])      # $28
]
```

**优势：**
- Worker 配置通过 YAML 文件传递（更清晰）
- Python 代码有类型转换和验证
- 易于添加新参数
- 配置和代码分离

---

## 四、配置管理对比

### 4.1 disagg_acc 配置管理

```bash
# 配置生成：disaggr_torch.slurm 中调用 gen_yaml.py
srun ... python3 ${workdir}/${gen_yaml_file} --config ${full_logdir}/config.yaml \
    --model ${model_dir} \
    --num_ctx_servers ${num_ctx_servers} \
    --ctx_tp_size ${ctx_tp_size} \
    # ... 14个命令行参数
```

**生成文件：**
- `config.yaml` - 单一配置文件，包含所有配置

**特点：**
- 配置在运行时动态生成
- 所有配置在一个文件中
- 需要等待配置文件生成（轮询检查）

### 4.2 disagg 配置管理

```python
# 配置在 submit.py 中预先生成
def save_worker_config(config, output_path, worker_type):
    """Save worker config to a separate YAML file."""
    worker_config = config['worker_config'][worker_type]
    with open(output_path, 'w') as f:
        yaml.dump(worker_config, f, default_flow_style=False)

# 分别保存
save_worker_config(config, ctx_config_path, 'ctx')
save_worker_config(config, gen_config_path, 'gen')
```

**生成文件：**
- `ctx_config.yaml` - ctx worker 配置
- `gen_config.yaml` - gen worker 配置
- `server_config.yaml` - server 配置（运行时生成）

**特点：**
- 配置预先生成，提交前就准备好
- ctx/gen 配置分离，便于独立管理
- 减少运行时依赖

---

## 五、SLURM 脚本对比

### 5.1 disagg_acc/disaggr_torch.slurm 特点

```bash
# 硬编码的 SLURM 配置
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --partition=36x2-a01r
#SBATCH --account=coreai_comparch_trtllm

# 运行时生成配置
srun ... python3 ${workdir}/${gen_yaml_file} --config ${full_logdir}/config.yaml ...

# 串行启动
srun ... bash ${workdir}/start_worker.sh ... &
srun ... bash ${workdir}/start_server.sh ... &
srun ... bash ${workdir}/run_benchmark.sh ...
```

**特点：**
- SLURM 参数部分硬编码在脚本头部
- 需要在脚本内生成配置文件
- 简单的串行启动
- 错误处理较弱

### 5.2 disagg/slurm/benchmark/disaggr_torch.slurm 特点

```bash
# 无硬编码，所有参数由 submit.py 传递
# 文件头部没有 #SBATCH 指令

# 配置已预先生成，直接读取
enable_pdl=$(python3 -c "import yaml; ...")

# 节点分配逻辑
all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
gen_node_list=(${all_nodes[@]:0:${gen_nodes}})
ctx_node_list=(${all_nodes[@]:${gen_nodes}:${total_nodes_num}})

# 循环启动多个 worker
for i in $(seq 0 $((num_gen_servers - 1))); do
    srun -N ${gen_nodes_num_in_single_server} ... \
        bash ${work_dir}/start_worker.sh "GEN" ${i} ... &
done

for i in $(seq 0 $((num_ctx_servers - 1))); do
    srun -N ${ctx_nodes_num_in_single_server} ... \
        bash ${work_dir}/start_worker.sh "CTX" ${i} ... &
done
```

**特点：**
- 完全动态配置，无硬编码
- 支持多 server 实例（循环启动）
- 节点智能分配（gen nodes 在前，ctx nodes 在后）
- 完善的错误处理和日志
- 支持 wheel 构建和安装

---

## 六、日志目录结构对比

### 6.1 disagg_acc 日志结构

```
bm_1028_deepseek-r1-1024-1024/
└── dep8_concurrency8192_eplb0_mtp0/
    ├── config.yaml              # 统一配置
    ├── output_workers.log       # worker 日志
    ├── output_server.log        # server 日志
    └── benchmark.log            # benchmark 日志
```

### 6.2 disagg 日志结构

```
1024-1024/
└── ctx2_gen1_dep32_batch32_eplb0_mtp0/
    ├── ctx_config.yaml          # ctx worker 配置
    ├── gen_config.yaml          # gen worker 配置
    ├── server_config.yaml       # server 配置
    ├── job_info.txt             # SLURM job 信息
    ├── environment.txt          # 环境变量
    ├── container_launch.log     # 容器启动日志
    ├── build.log                # TRT-LLM 构建日志
    ├── install.log              # 安装日志
    ├── output_gen_0.log         # gen worker 0 日志
    ├── output_gen_1.log         # gen worker 1 日志
    ├── output_ctx_0.log         # ctx worker 0 日志
    ├── output_server.log        # server 日志
    └── bench.log                # benchmark 日志
```

**新版优势：**
- 日志更细粒度（每个 worker 独立日志）
- 包含构建和安装日志
- 记录环境信息便于调试
- 目录名包含更多配置信息

---

## 七、融合可行性分析

### 7.1 关键差异点

| 差异点 | 影响 | 融合难度 |
|--------|------|---------|
| ctx_num 计算逻辑 | 中 | **低** - 可选两种模式 |
| 配置生成时机 | 高 | **中** - 需统一为预生成 |
| 参数传递方式 | 高 | **高** - 需大量重构 |
| 节点分配逻辑 | 高 | **中** - disagg_acc 较简单 |
| 多 server 支持 | 中 | **低** - disagg_acc 只用单server |

### 7.2 融合方案建议

#### **方案 A：最小侵入式融合** ⭐ 推荐

在 `submit.py` 中添加 **legacy 模式**支持：

```python
def calculate_ctx_num_legacy(concurrency, capacity_per_ctx=5500):
    """
    Legacy ctx_num calculation (disagg_acc compatible)
    
    Args:
        concurrency: Total concurrency
        capacity_per_ctx: Capacity per ctx server (default 5500)
    
    Returns:
        ctx_num
    """
    return (concurrency + capacity_per_ctx - 1) // capacity_per_ctx

def submit_job(config):
    # 检测是否启用 legacy 模式
    use_legacy_ctx_calc = config.get('metadata', {}).get('use_legacy_ctx_calculation', False)
    
    if use_legacy_ctx_calc:
        # Legacy 模式：根据 concurrency 计算 ctx_num
        concurrency = int(config['benchmark']['concurrency_list'].split(',')[0])
        ctx_num = calculate_ctx_num_legacy(concurrency)
        config['hardware']['num_ctx_servers'] = ctx_num
        print(f"   🔧 Legacy mode: Calculated ctx_num={ctx_num} for concurrency={concurrency}")
    
    # 后续逻辑保持不变
    ctx_tp_size = config['worker_config']['ctx']['tensor_parallel_size']
    gen_tp_size = config['worker_config']['gen']['tensor_parallel_size']
    # ...
```

**YAML 配置示例：**
```yaml
metadata:
  model_name: "deepseek-r1-fp4"
  use_legacy_ctx_calculation: true  # 启用 legacy 模式

benchmark:
  concurrency_list: "8192"  # 用于计算 ctx_num

hardware:
  gpus_per_node: 4
  num_gen_servers: 1  # 仍需指定
  # num_ctx_servers 将自动计算
```

**优势：**
- ✅ 向后兼容 disagg_acc 的计算逻辑
- ✅ 不影响现有功能
- ✅ 通过配置选项控制
- ✅ 实现简单，风险低

#### **方案 B：完全统一（推荐长期）**

废弃 disagg_acc，所有测试迁移到新版：

1. **创建迁移工具**：
```python
# disagg_acc_to_yaml.py
def convert_submit_sh_to_yaml(submit_sh_path):
    """将 submit.sh 中的配置转换为 YAML"""
    # 解析 submit.sh 中的循环和参数
    # 生成对应的 YAML 配置文件
    pass
```

2. **统一配置格式**：
   - 所有配置使用 YAML
   - 使用新版的节点计算逻辑
   - 使用分离的 worker 配置

3. **逐步迁移**：
   - 第一阶段：两套系统并行
   - 第二阶段：新功能只在新版实现
   - 第三阶段：废弃旧版

**优势：**
- ✅ 长期维护成本低
- ✅ 功能统一，避免分裂
- ✅ 更好的扩展性

**劣势：**
- ⚠️ 需要迁移现有配置
- ⚠️ 可能影响现有脚本

---

## 八、具体实施步骤（方案 A）

### Step 1: 在 submit.py 中添加 legacy 支持

```python
# disagg/slurm/benchmark/submit.py

def calculate_ctx_num_legacy(concurrency, capacity_per_ctx=5500):
    """Legacy ctx_num calculation for disagg_acc compatibility"""
    return (concurrency + capacity_per_ctx - 1) // capacity_per_ctx

def submit_job(config):
    # ... 现有代码 ...
    
    # Check for legacy mode
    metadata = config.get('metadata', {})
    use_legacy = metadata.get('use_legacy_ctx_calculation', False)
    
    if use_legacy:
        # Parse first concurrency value
        concurrency_str = config['benchmark']['concurrency_list']
        first_concurrency = int(concurrency_str.split(',')[0])
        
        # Calculate ctx_num using legacy formula
        ctx_num = calculate_ctx_num_legacy(
            first_concurrency,
            capacity_per_ctx=metadata.get('ctx_capacity', 5500)
        )
        
        # Override hardware config
        config['hardware']['num_ctx_servers'] = ctx_num
        
        print(f"   🔧 Legacy mode enabled:")
        print(f"      Concurrency: {first_concurrency}")
        print(f"      Calculated ctx_num: {ctx_num}")
        print(f"      Capacity per ctx: {metadata.get('ctx_capacity', 5500)}")
    
    # ... 继续现有逻辑 ...
```

### Step 2: 更新 config_loader.py

在 `TestConfig` 中添加 legacy 模式识别：

```python
# config_loader.py

def _load_config_file(self, yaml_path: Path, test_type: str,
                     test_category: str) -> TestConfig:
    """Load single YAML config file"""
    # ... 现有代码 ...
    
    # 检测 legacy 模式
    metadata = config_data.get('metadata', {})
    if metadata.get('use_legacy_ctx_calculation'):
        print(f"   🔧 Legacy ctx_num calculation enabled")
    
    # ... 继续 ...
```

### Step 3: 创建 legacy 配置模板

```yaml
# test_configs/disagg/perf/deepseek-r1-fp4_1k1k_dep8_legacy.yaml

metadata:
  model_name: "deepseek-r1-fp4"
  precision: "fp4"
  use_legacy_ctx_calculation: true  # 启用 legacy 模式
  ctx_capacity: 5500                # 每个 ctx server 的容量
  supported_gpus: ["GB200"]

# 其他配置保持不变
benchmark:
  concurrency_list: "8192"  # 将用于计算 ctx_num
  
hardware:
  gpus_per_node: 4
  num_gen_servers: 1
  # num_ctx_servers 将自动计算为 2
```

### Step 4: 测试验证

```bash
# 测试 legacy 模式
python3 disagg/slurm/benchmark/submit.py \
    -c test_configs/disagg/perf/deepseek-r1-fp4_1k1k_dep8_legacy.yaml

# 应该看到：
#   🔧 Legacy mode enabled:
#      Concurrency: 8192
#      Calculated ctx_num: 2
#      Capacity per ctx: 5500
```

---

## 九、总结

### 主要区别

1. **架构层面**：
   - 旧版：Shell 脚本 + 硬编码配置
   - 新版：Python + YAML + 动态配置

2. **节点计算**：
   - 旧版：基于并发数的简单公式
   - 新版：基于 TP size 的精确计算

3. **配置管理**：
   - 旧版：运行时生成单一配置
   - 新版：预生成分离配置

4. **扩展性**：
   - 旧版：需修改脚本代码
   - 新版：只需修改 YAML 配置

### 融合建议

**短期（推荐）：方案 A - 最小侵入式融合**
- 在新版中添加 legacy 模式支持
- 通过配置开关控制
- 保持向后兼容

**长期：方案 B - 完全统一**
- 迁移所有配置到新版
- 废弃 disagg_acc
- 统一维护

### 优先级

1. ✅ **高优先级**：实现方案 A（1-2天工作量）
2. ⭐ **中优先级**：创建配置迁移工具（1周）
3. 📝 **低优先级**：完全废弃旧版（长期计划）

---

**建议：先实施方案 A，验证稳定后再考虑完全迁移。**

