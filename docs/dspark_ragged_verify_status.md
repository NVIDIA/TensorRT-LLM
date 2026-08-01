# DSpark ragged verify — 目标与现状

配套文档：`docs/dspark_confidence_schedule_goal.md`（设计与方案对比）。
本文档记录**工作状态**：做到哪、验到哪、哪些还不能信。

最后更新：2026-07-31。分支 `pr-17056`，基线 PR [NVIDIA/TensorRT-LLM#17056]。
**所有改动均未提交**（工作区 15 个修改 + 4 个新增文件）。

---

## 0. 观测范围 —— 所有实测数字只适用于 x86 DGX B300

本文档里每一个耗时、吞吐、显存、page-cache 数字都是在**一台机器**上量的：

| | |
|---|---|
| 节点 | `umb-b300-001` |
| 分区 | `b300@ts6/dgx-b300@ts1/8gpu-256cpu-2048gb` |
| 架构 | **x86_64** + NVIDIA B300 SXM6，compute_cap **10.3（SM103）**，275 GB/卡 |
| 主机内存 | 2015 GB |
| 权重存储 | `/home/scratch.trt_llm_data`，共享 **NFS** |

**换集群这些数字全部作废，尤其是 GB300 / `b300-nvl8` / `gb300`**：那是
Grace-Blackwell，**aarch64** 架构、CPU 与 GPU 之间是 NVLink-C2C 统一内存、
主机内存容量与存储后端都不同。具体会变的：

- **编译目标**：`103-real` 是为 x86 B300 编的。GB300 需要 aarch64 工具链和对应镜像
  （`LLM_SBSA_DOCKER_IMAGE`，见 `jenkins/current_image_tags.properties`），
  不能直接复用本仓库现有的 `cpp/build`。
- **加载耗时结构**（§6.4b 那张表）：预取 2 h / 装配 40 min 是这台机器的 NFS 和
  x86 内存带宽决定的。Grace 的内存带宽和统一内存寻址会改变整个比例。
- **page cache 复用策略**：依赖 2 TB 主机内存能装下 835 GB 权重。GB300 的内存
  配置不同，这个前提要重新验证。
- **显存余量**：275 GB/卡 是 B300 SXM6 的数字。
- **Slurm 账号与分区**：`trt-llm_b300` 对应 x86；GB300 走 `trt-llm_gb300`。

**结论性的东西（六条 P0 的定位、kernel 的 next_n 假设、可观测性设计、
"空输出不是通过"那几条教训）与平台无关，可以直接迁移。
所有带具体数字的运行特征都不行。**

---

## 1. 目标

主线：让 **ragged verify**（每个 request 各自的 verify 长度）在 TensorRT-LLM 里真正跑通。
这是 confidence-head 调度收益的唯一来源，**必须实现**，不接受退化成 batch-uniform K 的方案。

验收条件，缺一不可：

| # | 条件 |
|---|------|
| 1 | 目标模型 DeepSeek-V4-Pro，真实 checkpoint |
| 2 | GSM8K 端到端精度不低于 baseline |
| 3 | **必须验 ADP**（目标平台已定为 **B300 / SM103 / 8 卡**）：`tensor_parallel_size=8` + `moe_expert_parallel_size=8` + `enable_attention_dp=True` |
| 4 | **确认 ragged 真的生效**，而非「配置打开了但代码走了 uniform fallback」 |
| 5 | kernel 侧兼容（indexer top-k / DSv4 compressor 不能静默算错） |
| 6 | CUDA graph 侧兼容（真在 replay，不是被迫 eager） |
| 7 | **overlap scheduler 开启也要正确**——这是 default 配置，不能只验关闭的路径 |

关于条件 2 的性质：调度只决定「多少 drafted 位置送去 target」，不改接受规则，
所以输出分布必须不变。**精度掉点 = 有 bug，不是吞吐 tradeoff。**

关于条件 4 的必要性：该功能的每一种失效都是**静默**的——planner 拒绝 trim、
cost model 退化、batch 错过捕获形状掉出 graph、部分加窗被拒——每一种都仍然产出
正确输出和基线精度。**GSM8K 通过本身不能证明功能跑过。**

---

## 2. 现状

### 2.1 已验证（有运行数据）

| 项 | 证据 |
|---|---|
| ragged top-k kernel 正确 | `tests/unittest/_torch/thop/parallel/test_indexer_topk_ragged.py` **10 passed** on B200，含 CUDA graph capture/replay，且与 `torch.topk` 参照比对 |
| 无回归 | ragged + `hw_agnostic` DSpark 共 **227 passed** |
| C++ 编译通过 | `indexerTopK.cu` / `IndexerTopKOp.cpp` / `compressorKernels.cu` / `compressorOp.cpp` 已编入 `.so` |
| checkpoint 可用 | `mtp.2.confidence_head.proj.weight`、`mtp.2.markov_head.markov_w1/w2.weight` 存在 |

### 2.2 未验证（这是最重要的一节）

**整条 e2e 从未跑通过。** 下面六条 P0 全部只经过读码与审计定位，
**没有一条经过运行验证**。逻辑上都成立、都能指到具体行号，但那不等于对。

e2e 冒烟第一次跑就撞上 `PYTHONPATH` 未设——说明这类问题还会有更多。

### 2.3 已知阻塞（非本工作引入，但会挡住验证）

1. **NVRTC 编译 MLA FMHA kernel 失败**
   `Failed to preprocess kernel fmhaSm100aKernel_QkvBfloat16OBfloat16HQk576HV512...:
   NVRTC_ERROR_COMPILATION`。影响所有 MLA 模型，**很可能包括 DSv4-Pro**。
   若冒烟撞上它，这是先于一切的阻塞。
2. `test_context_sparse_attention_mqa` 6 个失败 —— 已用有效对照（stash 三个 sparse
   文件后跑同一子集，基线同样 6 failed）证明是既有问题。
   注意：另外 53 个 sparse 失败**未逐一对照**。
3. `.venv-3.12` 的 cutlass 曾完全坏掉（`import tensorrt_llm` 起不来），已修复，
   详见 §5。

---

## 3. 已完成的改动

### 3.1 kernel 层

**`indexerTopK.cu`** —— `topKPerRowDecode` 原来在 kernel 内部用标量 `next_n`
同时推导 row→request 映射和 causal 上界：

```cpp
int seq_len = seqLens[rowIdx / next_n];
int actual_kv_len = seq_len - next_n + (rowIdx % next_n) + 1;
```

新增可空 `rowKvLens`：非空时 `actual_kv_len = rowKvLens[rowIdx]`，空时保持原式。
数学上是严格推广（请求 verify `v` 个位置时第 `o` 行可见 `kv_len - v + o + 1`，
代入 `v = next_n` 即还原），所以 uniform 路径逐位不变。

`IndexerTopKOp.cpp` 新增 `Tensor? row_kv_lens`，并在 ragged 时**替换**而非放宽
`seq_lens.size(0) * next_n == numRows` 检查——放宽的话，恰好整除的 ragged batch
（如 `verify_lens=[6,4,1,1]`，total=12，bs=4 → next_n=3）会通过检查然后静默算错。

**`compressorKernels.cu`** —— `NEXT_N` 是模板参数，但 body 里只用于 `sp`、
`last_token_idx` 和两个循环上界，且两个循环**本来就有守卫**
（Phase 1 的 `token_idx < kv_len`、Phase 3 的 `c >= num_compressions`）。
所以把 `NEXT_N` 从「确切长度」降级为「batch 内最大值（编译期上界）」即可，
真实 per-request 计数走新增的可空 `new_tokens_per_seq`。
单次 launch、固定 grid、**CUDA graph 拓扑不变**。

### 3.2 六条 P0

| | 问题 | 修法 | 触发条件 |
|---|---|---|---|
| P0-1 | `attn_metadata.prepare()` 是 `ragged_verify_lens` 的唯一消费者，却在它发布前 86 行就跑了 | 拆出 `_publish_ragged_verify_lens` 提前调用 | 全部 |
| P0-2 | `capture()` 用 `bs*(key[1]+1)` 定宽；graph `seq_lens` 冻结 | 改用 `key[5]`（ragged bucket）；ragged 且行数不变时放行 `seq_lens` 刷新 | 第一个 trim 步 |
| P0-3 | `expand_per_gen_token` 在 capture body 内做 pinned 分配 + H2D | 新增常驻 `row_req_idx_cuda`，refresh 改成 `index_select(out=) + add_` | replay |
| P0-4 | `on_update_kv_lens` 无条件整除覆盖 compressor 的 `next_n` | 抽出 `_sync_gen_tokens_per_seq` 单一写入点 | 每次 forward |
| P0-5 | drafter 用定长 `Kp1` 跨步读 flat-indexed 的 captured hidden states | 改用 `qo_indptr`；`base + gidx` 补 clamp | **overlap 开启** |
| P0-6 | overlap rewind 读到已被下一步覆盖的 `py_verify_len` | `SampleStateSpec.verify_lens_snapshot` | **overlap 开启** |

P0-2 的两个半边必须**一起**修：`num_tokens` 来自 `seq_lens.sum()`，
而 `capture()` 从 `key[1]` 定宽，只修一边会让 `input_ids.shape[0] != num_tokens`。
最大 bucket 恰好等于那个乘积，而 warmup 捕的正是它——所以捕获阶段完全干净，
**第一个真正 trim 的 step 才崩**。

P0-3/5/6 属于「跑出来的数字看着正常但其实是错的」：replay 读已释放内存、
跨请求串读、KV 回退量错。**这类问题不会报错。**

### 3.3 可观测性（`tensorrt_llm/_torch/speculative/dspark_observability.py`）

对齐 SGLang 的 `SGLANG_RAGGED_VERIFY_MODE` 语义：

| 开关 | 作用 |
|---|---|
| `TLLM_DSPARK_RAGGED_VERIFY_MODE=static\|compact` | 选路径。`cap-accept` **未实现，显式抛 `NotImplementedError`** |
| `TLLM_DSPARK_FORCE_VERIFY_LENS=1` | 按 tier 阶梯轮转出确定性非均匀切分 |
| `DSparkRaggedStats.assert_ragged_active()` | 断言 ragged 真的生效 |

`assert_ragged_active()` 检查四条：`steps_ragged > 0`、`distinct_verify_lens >= 2`、
`trim_ratio > 0`、`graph_eager == 0`，失败时带完整计数器摘要。

**`cap-accept` 为什么不能 alias 成 `compact`**：它唯一的价值是「输出必须与 `static`
逐 token 一致」，别名化会让这个对照失去意义。

**`FORCE_VERIFY_LENS` 解决的循环依赖**：没有 profiled SPS cost table 时
planner 在构造上不会 trim（`llm_args.py` 自述「the budget degenerates to
verify-all」），于是验证 ragged 打包正确性需要先有 table，而产出 table 又需要
能跑的 run。这个开关替换的是 **planner 的决策**，不绕开任何下游代码路径——
所以 forced-ragged 与 static 在 `temperature=0` 下必须逐 token 一致。

---

## 4. 剩余工作

| 优先级 | 项 | 说明 |
|---|---|---|
| P0 | e2e 冒烟通过 | 两个配置各一遍：`disable_overlap_scheduler=True` 与 **default（overlap 开启）** |
| P0 | 排除 NVRTC 阻塞 | 若 DSv4-Pro 起不来，其余都无从谈起 |
| P1 | **SPS cost table profiler** | 没有它 planner 必然不 trim，**任何性能结论都是空的**。参考 SGLang `python/sglang/benchmark/dspark_sps_profiler.py`：服务端 `RAGGED_VERIFY_MODE=static` + record flag，扫 batch size 记 step time |
| P1 | GSM8K + ADP | 冒烟通过后才有意义 |
| P1 | graph 按 token 数分桶 + ADP tier 协商 | SGLang `compute_target_verify_graph_key` 对 ragged 返回 `(graph_num_tokens, graph_num_tokens)`——两个轴都变 token 数；DP 下各 rank 共用同一 tier 并一起降档 |
| P2 | 实现 `cap-accept` | 隔离「kernel 算错」与「planner 没触发」的最强工具 |
| P2 | 对照剩余 53 个 sparse 失败 | 目前只对照了 mqa 那 6 个 |

---

## 5. 任务清单（截至 2026-07-31）

| # | 任务 | 状态 |
|---|------|------|
| 1 | 目标文档 | 完成 |
| 2 | indexer top-k kernel ragged 化 | 完成（B200 上 10 用例通过，**但在错误镜像下跑的，需重验**） |
| 3 | DSv4 compressor ragged 化 | 编译通过，**未运行验证** |
| 4 | 审计 DSv4 gen 路径剩余 ragged 缺口 | 未完成 |
| 5 | CUDA graph 兼容 | 见 P0-2/P0-3，**未运行验证** |
| 6 | GSM8K + ADP 验证 | 未开始 |
| 7 | mode 开关 + 可观测性 | 完成 |
| 8 | graph 按 token 分桶 + ADP tier 协商 | 未完成 |
| 9 | SPS cost table profiler | 代码完成（1503 行 + 42 测试），**从未在真实 engine 上跑过** |
| 10 | `on_update_kv_lens` 刷新 row_kv_lens | 完成 |
| 11 | overlap 开启的正确性验证 | 未完成（**default 配置，必须验**） |
| 12-16 | P0-2 … P0-6 | 代码完成，**六条全部零运行验证** |

**当前阻塞**：环境重建中（见 §6）。六条 P0 一条都没经过运行验证，这是最大的未知。

---

## 6. 环境（复现必读）

### 6.1 容器镜像 —— 先看这里

**镜像由仓库指定，不要自己挑。**

```
jenkins/current_image_tags.properties:16
LLM_DOCKER_IMAGE=urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:\
  pytorch-26.05-py3-x86_64-ubuntu24.04-skip-tritondevel-202607211045-16608
```

本工作前期用错了镜像（`...trt10.16.1.11-...202607151440-16194`，从本地 `docker images`
里挑的），代价是**一整轮关于 cutlass 版本冲突的错误分析**：错误镜像的系统 cutlass 与
`requirements.txt` 的 pin 不配套，我先归因给自己的 pip 操作，又归因给「repo 与镜像的
结构性冲突」，两次都错。正确镜像的 cutlass 是自洽的。

### 6.2 cutlass 版本

正确镜像自带 **4.6.1 且四个子包全部一致**。`requirements.txt` pin `==4.5.0`，但
`constraints.txt` 只要求 `>=4.4.2`，所以**安装依赖时排除 cutlass、沿用镜像自带的**：

```bash
grep -v "nvidia-cutlass-dsl" requirements.txt > requirements.nocutlass.txt
pip install -r requirements.nocutlass.txt
```

过滤后的文件**必须写在 `requirements.txt` 旁边**——里面有 `-c constraints.txt` 相对路径，
放 `/tmp` 会报 `Could not open constraint file`。

为什么不能装 4.5.0：那套包没有 `libs-core`，装进 `--system-site-packages` 的 venv 后，
前端是 4.5.0 而 MLIR verifier 是系统的 4.6.1，DSv4-Pro 会死在
`'nvgpu.cvt_fptrunc' op operand #0 must be ... 1-d vector, but got 'f32'`
（`f8E8M0FNU` 就是 DSv4 的 ue8m0 scale）。

**验证 venv 不能只看 `import` 成功**，要看 `pip list | grep cutlass` 各子包版本是否一致。

### 6.3 venv

```bash
/usr/bin/python3 -m pip install --user 'virtualenv<22.0,>=20.29.1'
/usr/bin/python3 -m virtualenv --system-site-packages .venv-3.12
```

`virtualenv` 而非 `venv`：镜像的 `ensurepip` 不可用。
`--system-site-packages` 必需：torch 在镜像的 dist-packages 里。

### 6.4 编译

```bash
python3 scripts/build_wheel.py --cuda_architectures "100-real" -j $(nproc)
```

- **PATH 只能追加不能替换**：`/usr/local/cuda/bin` 在镜像默认 PATH 里，覆盖掉会让
  cmake 报 `No CUDA compiler found`。
- **配置失败后必须 `rm -rf cpp/build`**：CMake 会把失败的编译器探测缓存成
  `CMAKE_CUDA_COMPILER:FILEPATH=NOTFOUND` 并在下次复用，环境修好了也不会重新探测。
- **目标平台是 x86 DGX B300（SM103），编译用 `--cuda_architectures "103-real"`。**
  GB300（`gb300` / `b300-nvl8`）是 aarch64，需要 SBSA 镜像和单独的编译产物。
  B200 是 SM100；`100-real` 纯 SASS 无 PTX，装不进 B300。B300 有 275 GB/卡，
  B200 只有 178 GB —— DSv4-Pro 权重 104 GB/卡，B200 上余量很紧。
- 容器需 `--user $(id -u):$(id -g)`（scratch 是 NFS root_squash）。

### 6.4b DSv4-Pro 的加载时间结构 —— 复用同一节点的 page cache

> **仅适用于 x86 DGX B300 + NFS（见 §0）。GB300 等其他平台需重新实测。**

冷启动实测（`umb-b300-001`，权重在 `/home/scratch.trt_llm_data`，NFS）：

| 阶段 | 耗时 | 可观测性 |
|---|---|---|
| import + 8-rank NCCL 建组 | ~1 min | 有日志 |
| **预取（NFS → 内存）** | **~2 h** | 只在整个 13 GB 分片读完时打一条 `Finished prefetching`，中间静默 90 min 以上 |
| **装配（内存 → 显存）** | **> 40 min** | tqdm 进度条，但 ETA 严重低估（并发流均匀推进、集中收尾） |

**冷启动总计 3 小时以上，`batch-short` 的 4 小时窗口装不下「加载 + 验证」。**

**解法：复用同一节点。** 该分区节点有 **2 TB 内存**，一次加载后 page cache 会保留
约 1.1 TB，足以完整覆盖 835 GB 权重。作业结束后 cache 不会被清掉，所以：

```bash
# 当前作业还没结束时就把后续作业排到同一节点，接力启动、cache 仍热
sbatch -A trt-llm_b300 --qos=batch-short \
  -p "b300@ts6/dgx-b300@ts1/8gpu-256cpu-2048gb" \
  --gres=gpu:8 --nodelist=<同一节点> -t 4:0:0 --wrap='sleep 14400'
```

查 cache 是否还热：`free -g` 看 `buff/cache` 列。

**判活提醒**：预取阶段日志静默、GPU 利用率 0%、显存不变**同时出现是正常的**。
唯一可靠的判据是固定时间窗口内 `/proc/<pid>/io` 的 `read_bytes` 增量
（实测健康值约 112 MB/s）。不要用 tqdm 的 ETA 估算剩余时间。

### 6.5 跑测试 / 跑模型

容器额外需要：`-e USER=<name>`（否则 `getpass.getuser()` 因 UID 无 passwd 条目而
KeyError）、`-e PYTHONPATH=/code/tensorrt_llm`（直接跑脚本时；`sys.path[0]` 是脚本所在目录）、
`-e LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models`、
`--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g`。

**srun 步骤内 docker 的 `--gpus` 只能用 `all`**，写 `device=N` 会被代理拒绝。

**任何 `tensor_parallel_size > 1` 的脚本必须有 `if __name__ == "__main__":`**——
`MpiPoolSession` 靠重新执行模块来 spawn worker，没有守卫会无限递归 spawn，
表现为**静默挂死而非报错**（本工作因此浪费 2 小时窗口）。

**Slurm**：Blackwell 分区 `DenyQos=batch,interactive-isolated,oversubscribe`，必须
`--qos=batch-short`（上限 **4 小时**）。报 `Invalid qos specification` 不表示没有权限。

---

## 7. 教训（给接手的人）

### 7.0 诊断动作本身会破坏运行 —— 两次杀掉自己的冒烟

冒烟从未跑到 ragged 代码，两次都死在加载阶段，**两次都是我杀的**：

1. **B200**：据「GPU 利用率 0% + 显存不变 + 日志静默」判定挂死并杀掉。
   事后 `py-spy` 证明它在 `weight_loader.py:338` 正常从 NFS 预取权重。
   残留进程占着 135 GB/卡不释放，导致下一次启动直接 OOM。
2. **B300**：为避免重犯，改用 `py-spy dump` 做「正确诊断」——附加超时，
   把 worker 留在 `ptrace_stop` 冻住了整个运行；再用
   `pkill -f "py-spy dump"` 去救，那个模式匹配到 srun 自己的命令行，把 step 杀了。
   `/proc/<pid>/io` 显示它冻住前已读 13.2 GB，本来是好的。

**三条铁律：**

- **不要 py-spy 附加正在加载的 worker。** 附加开销大到会超时，且会把进程留在
  ptrace 停止态。要用也只在已经确认卡死之后用。
- **不要 `pkill -f <字符串>`。** 会匹配到 srun / docker 自己的命令行。
  清理只按 PID：`nvidia-smi --query-compute-apps=pid --format=csv,noheader`。
- **判活只用 `/proc/<pid>/io` 的 `read_bytes` 是否增长。** 只读、零副作用，
  是加载阶段唯一安全且可靠的存活信号。

**加载阶段的正常表现**：GPU 利用率 0%、日志静默、显存不变——三者同时出现也是正常的。
DSv4-Pro 权重 835 GB 从共享 NFS 冷读，20-30 分钟属于合理范围。

**杀掉运行之后必须清残留**：孤儿进程会一直占着显存，下一次启动就会 OOM。
`for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done`

### 7.1 空输出不是通过

本工作中反复栽在这一条，四次：

1. **kernel 输出顺序**：`indexer_topk_decode` 不保证输出顺序（radix split-work 的 merge
   顺序不定），只有 top-k **集合**是契约。逐元素比数组得到 ~95% 假阳性 mismatch。
   必须 `sort(dim=-1)` 后比，并用 `torch.topk` 做参照。
2. **对照实验没跑起来**：一次 baseline 对照用了 `--gpus device=7`，docker 直接
   `invalid gpus device: 7` 退出，空输出被当成了有效数据。
3. **冒烟挂死**：连续两次报告「仍在加载 835G」，实际 GPU 1-7 各 4 MiB、利用率全 0。
4. **编译白跑**：CMakeCache 缓存了 NOTFOUND，容器「在运行」但 0 个 object 文件。

统一的教训：**「进程在跑」「没有报错」都不是进展的证据**。要有正向信号——
object 文件数、GPU 显存占用、明确的成功输出。

### 7.2 工具缺陷要立刻修，不要记下来等下次

`run.sh` 里的 `... | tail -80` 会缓冲全部输出，进程退出前看不到任何进度。
我在发现后说了「下次改」，没有立即改——结果冒烟静默挂死 2 小时才被发现，
而改成 `python3 -u ... | tee` 之后**一分钟**就看到了完整的 MPI 错误。同样的信息一直都在。

### 7.3 局部通过不等于整体正确

一度宣告「venv 已修复」，依据只是 `import cutlass` 和 `import tensorrt_llm` 成功。
实际版本组合并不自洽，DSv4-Pro 起不来。**导入成功只证明模块能加载，不证明版本配套。**

### 7.4 环境问题先查仓库怎么规定，不要自己拼

本工作在环境上的弯路（镜像、cutlass、PATH、CMakeCache、constraints 路径、ensurepip）
**没有一个是产品代码的问题**，全部是容器调用姿势，而每个都吃掉一轮迭代。
根源是「见招拆招」——需要 virtualenv 就改 PATH，需要过滤 requirements 就写 /tmp，
每次改动都没检查会不会破坏别的东西。
正确做法是一开始就照 `jenkins/current_image_tags.properties` 和
`.dspark-logs/build.log` 复现官方姿势。

### 7.5 这个功能的所有失效都是静默的

planner 拒绝 trim、cost model 退化、batch 错过捕获形状掉出 graph、部分加窗被拒——
每一种都仍然产出正确输出和基线精度。这是 §3.3 那套可观测性存在的全部理由。
**新增任何决定 verify 长度的代码路径，都要同时接上 `record_step`**，
否则 `assert_ragged_active()` 会给出虚假的通过。
