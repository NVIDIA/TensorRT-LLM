# DSpark confidence-head 调度 P0 阶段：执行报告

> 分支：`dspark-p0`（基于 PR #17056 head `d8897046`，PR 分支未改动）
> 硬件：`bia` 集群，8× B300 SXM6（sm103），x86_64
> 构建：`/lustre/fsw/coreai_comparch_trtllm/laliao/build/dspark-p0/`（`-a '103-real;'`，rc23）
> 日期：2026-08-01
>
> 本文沿用 goal doc 的规范：凡未经读码或实测确认的推断，显式标注「待验证」。

---

## 0. 先说三个与任务书不同的结论

### 0.1 goal doc 的 11 个 host 阻断点里，7 个在 PR head 上已经修好了

任务书 §1(b) 说「H3 和 H4 现在是已经进了 PR head 的 bug」。**这一条不成立。**

证据：

```
$ git log --oneline -S "row_kv_lens.add_(expanded" --all -- .../dsa.py
(空)
```

那个「CUDA tensor 加 CPU tensor」的形式**从未进入过任何 commit**。goal doc §1.3 描述的是提交*之前*的工作区快照；等它变成 `e1c28eeae2` 时，H3/H4 已经按 goal doc 建议的方式修好了：

| 项 | 现状 | 锚点 |
|---|---|---|
| **H0** | 已修 | `model_engine.py:4670-4671` 非 overlap 分支已走 `get_request_tokens_per_gen_step` |
| **H2** | 已修 | `model_engine.py:5410-5412` `_publish_ragged_verify_lens` 在 `prepare()` **之前** |
| **H3** | 已修 | `dsa.py:1697-1757` 全在 host 上组装 + 单次 pinned H2D |
| **H4** | 已修 | `dsa.py:902` → `refresh_ragged_row_kv_lens`（`dsa.py:1759-1785`） |
| **H5** | 已修 | `cuda_graph_runner.py:462-464` `num_tokens_for_capture` 取 `key[5]` |
| **H6** | 已修 | `SampleStateSpec.verify_lens_snapshot`（`spec_sampler_base.py:63,246,340-354`） |
| **H10** | 已修 | `dspark.py:629-634` ragged 时用 `qo_indptr` |

全部由 `e1c28eeae2` 引入（`git log -S` 逐个确认）。

### 0.2 B2 的结论是「token-major 不慢」，ragged 路线不被否决

见 §1。这与 goal doc §7.1 U1 担心的方向相反。

### 0.3 参数化后的 `test_deepseek_v4_sparse_mla` **抓不到** H1

goal doc §6.3 称它是「性价比最高的一个测试」。参数化之后它确实覆盖了宽度 1/2/4/6（此前只有 1），但**实测确认它对 H1 不敏感**：把 stride 修复回退成修复前的行为，12 个 case 依然全绿。

原因：该测试对 `compress_ratio=4` 的层**直接传入 `topk_indices`**（`deepseek_v4.py:1674-1676` 走 `forward_args.topk_indices`），于是 `Indexer.forward` 根本不执行 —— 而 `kv_lens_expanded` / `block_table_expanded` / DeepGEMM schedule 正是在那里被消费的。

这一点已写进测试文件本身，避免下一个人被绿色的测试误导。

---

## 1. 交付物 0：B2 实测数字（最先交）

**问题**（goal doc §7.1 U1）：trtllm-gen sparse-MLA generation 在 `(batch=B, s_q=tier+1)` 与
`(batch=B*(tier+1), s_q=1)` 两种呈现下，相同总 token、相同逐 token 稀疏索引集时的吞吐差多少。
D1 和 D2 都靠把 generation 半边呈现成「长度为 1 的序列」来让
`thop/attentionOp.cpp` 的 `num_tokens % num_seqs == 0` 平凡通过，二者押的是同一个赌注。

**脚本**：`tests/microbenchmarks/dsv4_sparse_mla_presentation.py`（单卡，不需要模型权重，分钟级）

**结果**（B300 sm103，context_len=4096，20 iters，中位数）：

| B | tier | next_n | tokens | batch-major (ms) | token-major (ms) | slowdown |
|---|---|---|---|---|---|---|
| 4 | 1 | 2 | 8 | 0.148 | 0.146 | **0.99×** |
| 4 | 3 | 4 | 16 | 0.142 | 0.137 | **0.97×** |
| 4 | 5 | 6 | 24 | 0.150 | 0.140 | **0.93×** |
| 16 | 1 | 2 | 32 | 0.147 | 0.139 | **0.95×** |
| 16 | 3 | 4 | 64 | 0.154 | 0.146 | **0.95×** |
| 16 | 5 | 6 | 96 | 0.145 | 0.145 | **1.00×** |
| 64 | 1 | 2 | 128 | 0.148 | 0.147 | **0.99×** |

另一次独立测量：B=8, tier=5 → **0.88×**。

**结论：token-major 没有变慢，范围 0.93×–1.00×（甚至略快）。**
按任务书 §3 表三的判据（「若 token-major 显著变慢或不正确：立刻停下来报告，改推 D3 Stage A」），
**不触发兜底条件，ragged 路线在吞吐上是可行的。**

这与 goal doc §7.1 U1 的推断方向一致（DEP8 下每 rank 保有全部 Q head，BMM1 的 M 维从
`128·(tier+1)` 降到 128；且稀疏索引集逐行不同，本来就没有跨窗口共享 KV tile 的机会可损失），
但现在是**测量**而不是推断。

**未覆盖**：B≥64 且 tier≥3 的点。这些点卡在 benchmark 自己的 KV pool sizing 上
（token-major 呈现有 `next_n` 倍的 sequence，整块分配的 rounding 也就付 `next_n` 倍），
不是 kernel 结果。已把 pool 按整块向上取整并加一块 slack，仍未覆盖到 B=128。
考虑到 7 个点跨 B=4..64 × 全部三个 tier 全部落在 0.93–1.00×，方向性结论是稳的；
但**「高并发下是否仍然如此」严格说仍待验证**，而高并发恰恰是这个 feature 唯一有收益的区间。

**范围说明**：该脚本计时的是整个 attention layer，包含 indexer（其呈现也随之改变：
batch-major 走 expanded 一 token 一行的路径，token-major 走 `next_n == 1` 的 strided 路径）。
这是有意的 —— 那正是 ragged step 真实要付的代价，并且它顺带覆盖了 B3。
纯 attention 的差值无法在不上 kernel-level profiler 的前提下分离出来。

---

## 2. 交付物 1：修复清单

每项一个 commit，commit message 里引用 goal doc 编号。

| commit | 项 | 内容 |
|---|---|---|
| `cb5cb0bb92` | **H1 + K8** | DSA 展开 stride 走 runtime tier |
| `44a81930ca` | **§4.3 sync ×2** | `repeat_interleave` 补 `output_size=`；`apply_sts` 缓存 host 副本 |
| `f99c34a498` | **H8 / K5 / K6 / K13** | 显式 config-reject，不静默降级 |
| `b3a5fef92a` | **H9 + §4.4** | 集合通信收敛为恒定一次；ADP 资格门比 bucket |
| `3b52d19fcd` | **H7** | rejection sampling 的 fail-closed guard 用 `total_verify_tokens` |

### 2.1 H1 + K8：这是 tier-3 hang 的自洽解释

`DSAtrtllmAttentionMetadata.max_draft_tokens` 是**静态上限**，不是本步窗口。
对 parallel-draft 模式（DSpark 属于其一），`forward()` 传给 `update_spec_dec_param` 的是
`original_max_total_draft_tokens`（`model_engine.py:6686-6688`），
即 `tokens_per_gen_step - 1`，engine 生命周期内**永不改变**。

于是：

- **producer**（`prepare_for_spec_decode`）按 stride `1 + max_draft_tokens = 6` 铺
  `kv_lens_expanded` / `block_table_expanded`；
- **consumer**（`Indexer.forward`, `dsa.py:2967`）按 `num_gen_tokens // num_generations` 切片。

tier 3 时 consumer 取 4。3 个请求、宽度 4 的例子：

```
正确:  idx 0-3 → kv0,  4-7  → kv1,  8-11 → kv2
修前:  idx 0-5 → kv0,  6-11 → kv1,  (kv2 从不出现)
```

**每个 r>0 的请求都拿到邻居的 kv_len 和 block table。**

K8（`dsa.py:2496-2497`）是同一个错位，但落在一个会 **hang 而不是算错**的地方：
`prepare_scheduler_metadata` 按 `num_generations * (1 + max_draft_tokens)` 行建 DeepGEMM
schedule，而 paged-MQA-logits 只拿到 `num_gen_tokens` 行 —— schedule 承诺了没人会产出的 tile。

这与唯一一次实测失败**逐字吻合**：`draft_len=5` 时 warmup 完成（768 == 768），
`draft_len=3` 时永不返回（768 vs 512）。

**修法**：新增 `runtime_tokens_per_gen_step` 与 `gen_token_stride`，所有 *stride* 走它，
所有 *capacity* 保持静态上限（expanded buffer 必须按最大值分配，否则较短的 tier 会在
capture 期触发重分配 —— 那是第二类 hang）。未发布（0）时回落到 `1 + max_draft_tokens`，
所以任何不缩短 draft 长度的路径 bit-identical。

发布点在 `_prepare_tp_inputs` 而不是 `update_spec_dec_param`：后者作用于 base metadata，
而 `prepare()` 跑在之后取的 per-key CUDA-graph 副本上，对 base 的属性赋值对副本不可见。

`use_expanded_buffers_for_mtp` **有意**保持按静态上限判定，使 layout regime 不随 tier 变化。

### 2.2 H9：这不只是「多一次集合通信」，是时序相关的死锁

`_dspark_confidence_draft_len` 里每个 rank 发出的集合通信次数不同：

| rank 走到哪 | 次数 |
|---|---|
| 完整 ragged | 2（planner 的 `all_rank_max` + `peer_stats`） |
| 在 planner 处放弃 | 1（uniform `decide_draft_len`） |
| 到 bucket fit 才放弃 | 3 |

而**所有放弃的理由都是 rank-local 的**，其中一条 —— confidence 快照的 copy event
是否落地（`_ready_snapshot` → `_copy_event.query()`）—— **纯粹是时序**。
同一步上两个 rank 可以因为谁都观察不到的原因走进不同分支。

**修法**：拆成三段 —— (1) 本地决策，不发任何集合通信；(2) 发**一次**定宽 allgather；
(3) 基于归约结果分支，之后不再发。payload 顺带把 `peer_stats` 带上，所以常见路径从
两次降到**一次**。只有当**所有** rank 都能走 ragged 时才走 ragged；有一个反对就整组退回
uniform —— 那是唯一一种 token layout 仍然一致的结局。

### 2.3 §4.4：ADP 资格门

`fit_ragged_verify_lens` 在达成一致*之后*仍可能在单个 rank 上失败（它的 pad 行拟合用的是
本地请求数），所以 ADP 门必须兜住。它此前只比 `is_all_gen_only` 和 batch size，**从不比 token 数**。
而这个不匹配在 replay 时**无法被察觉**：`all_rank_num_tokens` 是 forward 内读取的 host list，
replay 的 graph 用的是 capture 时那份。现在同时比 ragged token bucket 与 draft length。

### 2.4 H7：让精心写的 ragged 分支不再是死代码

`_rejection_buffers_valid` 要求 `num_contexts + num_gens * (draft_len + 1)` 行，
而 ragged 下 target 只发 `num_contexts + total_verify_tokens` —— 差值恰好是调度 trim 掉的量。
**guard 恰好在 feature 起作用时失败**，然后 fail-closed 静默退回严格接受，改变了
temperature > 0 的采样分布。

rejection 路径的 ragged 分支本身是**完整的**（按 `total_verify_tokens` 打包、scatter 进
padded 矩形、pad 行填 one-hot 使其不可能被接受、最后把 accepted 数 clamp 回 `verify_lens`），
所以该修的是行数而不是 config —— 直接拒绝「ragged + 非贪婪」会丢掉能用的代码。

goal doc 说得对：**贪婪 GSM8K 观察不到这一点**，所以只能从代码论证。

---

## 3. 交付物 2：测试

| 测试 | 状态 |
|---|---|
| `test_deepseek_v4_sparse_mla` 按每请求 q 长度参数化（宽度 1/2/4/6） | **12/12 通过**，但见 §0.3 的覆盖限制 |
| compressor ragged 差分（`test_compressor_ragged.py`，新增） | **4/4 通过** |
| C++ ragged top-k（`test_indexer_topk_ragged.py`，PR 已有） | 见 §5 |
| A3 布局一致性断言（`TLLM_DSPARK_ASSERT_LAYOUT=1`） | 已加；e2e run 全程启用，**57 个 graph 的两轮 warmup/capture 无一触发** |
| A4 时序断言 | 已加（无条件，代价是一次 host 求和） |
| A6 planner 计数并入 ragged stats summary | 已加 |

### 3.1 参数化过程中抓到的两个测试自身的 bug

1. `_allocate_kv_cache_for_generation` 每步只扩一个 token —— 只有在
   `generation_seq_len_q == 1` 时才对。宽度 4 时第二步一旦跨页就越界（illegal access，
   不是干净的失败）。
2. `_build_compressed_topk_indices` 按**每请求**一行构建，而 kernel 要的是**每 query token**
   一行。这正是被测试的那个「均匀窗口」假设本身。

### 3.2 compressor 差分测试的设计

第一个测试（uniform 填充 == 传 `None`，bitwise）是让那个可选参数**可以被接受**的论据。
但它有一个结构性盲点：对于一个**完全忽略** `new_tokens_per_seq` 的 kernel，uniform 向量
也是正确答案。所以第二个测试用真正不同的计数，与**单请求** uniform 调用逐个比对 ——
batch 为 1 时「uniform」无歧义，且与 ragged 调用不共享代码路径。

---

## 4. 交付物 4：hang 的真实根因 —— **H1/K8，已实测确认**

goal doc §7 列了三个独立候选。**本轮 e2e run 把它们分开了。**

### 4.1 实测：hang 消失

07/30 那次失败（goal doc §1.4）：

```
DSpark confidence scheduling: capturing 102 graphs (34 batch sizes x 3 draft-length tiers [1, 3, 5])
Attention workspace size is not enough, increase the size from 0 bytes to 168493568 bytes
Run generation-only CUDA graph warmup (greedy) for batch size=128, draft_len=5   ← 完成
Run generation-only CUDA graph warmup (greedy) for batch size=128, draft_len=3   ← never returns
TEST_EXIT=124
```

本轮（修复 H1/K8 后，overlap scheduler 开启，DEP8）：

```
06:36:23  DSpark confidence scheduling: capturing 57 graphs (19 batch sizes x 3 draft-length tiers [1, 3, 5])
06:36:23  Run generation-only CUDA graph warmup (greedy) for batch size=128, draft_len=5
          Attention workspace size is not enough, increase the size from 0 bytes to 168493568 bytes   ← 同一个数字
06:36:24  Run generation-only CUDA graph warmup (greedy) for batch size=128, draft_len=3   ← ~1 秒完成
06:36:29  Run generation-only CUDA graph warmup (greedy) for batch size=128, draft_len=1
06:36:35  ... batch size=120, draft_len=5
          ... 继续走完全部 19 个 batch size × 3 个 tier
```

（graph 数从 102 变 57 是因为本轮配置的 batch-size 桶只有 19 个，与 hang 无关。）

### 4.2 三个候选的判定

| 候选 | 判定 | 依据 |
|---|---|---|
| **1. H1 stride 不匹配** | **✅ 根因** | 唯一被修的东西；修完之后 `batch=128, draft_len=3` 从「永不返回」变成 ~1 秒完成。且它的机制与症状逐字吻合：tier 5 时 768==768 通过，tier 3 时 schedule 描述 768 行而 logits 只有 512 行 —— K8 那一处**机制上就是 hang 而不是算错**（DeepGEMM schedule 承诺了没人会产出的 tile，kernel 等在那里）。 |
| **2. capture 期 attention workspace 重分配** | **❌ 排除（是共存现象，不是原因）** | 同一条警告、**同一个字节数 168493568**、同一个位置（紧邻 `draft_len=5` warmup）在本轮**照样出现**，而 capture 正常走完。所以它不是 hang 的原因。 |
| **3. torch.compile / max-autotune 在 capture 区内** | **❌ 排除为根因** | tier 5 / 3 / 1 各自在 1–6 秒内完成。若 Inductor 在 capture 区内编译并 hang，这里就走不过去。Inductor 编译大概率确实发生了（每个 tier 一次 guard），但它会完成。**未用 `TORCH_LOGS=recompiles` 单独确认编译次数，这一点仍标「待验证」。** |

**结论：hang 的真实根因是 H1（连带 K8）—— DSA 展开 stride 用静态 max 而非 runtime tier。**
候选 2 是无害的共存现象；候选 3 未被证明有害。

## 5. 交付物 3：e2e 实测 —— **未完成**

- 均匀 tier 路径 + overlap scheduler + DEP8 的 GSM8K：**运行中，未取得分数**。
- ragged 路径：**未跑**。
- planner.stats 的 fallback 计数：**未取得**。

已排除的环境障碍（供下一次复现参考）：

1. 容器里没有 `pytest` / `parameterized` / `mako` / `oyaml` 等；已装到
   `/lustre/fsw/coreai_comparch_trtllm/laliao/pyextra`，通过 `PYTHONPATH` 引入。
2. `srun --ntasks=8 pytest` 会让 LLM API 再去 `MPI_Comm_spawn`，报
   `MPI_ERR_SPAWN`。**必须用 `trtllm-llmapi-launch` 包一层**（仓库里
   `tmp/serve_*.slurm` 就是这么写的）。
3. 便捷封装：`tmp/run_in_alloc.sh <jobid> <ntasks> "<cmd>"`。
   注意它用 `$*` 展开，测试 id 里的 `[...]` 会被 shell 当 glob 吃掉 —— 用 `-k` 选择。

已确认到达的阶段：权重加载完成 → DSpark worker 初始化 →
`DSpark verify planner: tiers=[1, 3, 5], profiled_cost_table=False, ragged_verify_mode=static`。

⚠️ **`profiled_cost_table=False`**：没有 profiled SPS 表时 `_decide_local` 无条件返回
`max_tier`（`dspark_verify.py:268-270`）。所以即便这个 run 跑完，它证明的是
**102 个 graph 的 capture 不再 hang**（那正是 07/30 失败的地方），
**不是**「调度真的在 trim」。后者需要 §7 的 Q5。

---

## 6. 交付物 5：对 goal doc §8 剩余待拍板问题的建议

### Q4：`cap-accept` 要不要实现 —— **建议实现，但优先级低于 Q5**

任务书倾向实现，理由是把 pass/fail 硬币变成可定位诊断。这个理由成立，但要注意
**PR head 已经有了一个更便宜的等价物**：`TLLM_DSPARK_FORCE_VERIFY_LENS=1`
（`dspark_observability.py:131-162`）从 tier ladder 里按批内位置轮转发窗口，
确定性、每个 rank 一致、且不需要 cost table。它已经能把
「ragged packing 对不对」与「planner 要不要 trim」分开 —— 而这正是 A2 想要的那一半。

`cap-accept` 额外提供的是另一半：跑**均匀 kernel 路径**但只提交窗口内 token，
于是 `cap-accept ≠ compact` ⇒ 必然是 ragged kernel bug。这个价值是真的，
但在 forced-lens 已经存在的前提下没有那么紧急。

### Q5：STS 表和 SPS cost table —— **这是当前最高优先级的缺口**

不是「性能结论是空的」那么简单：**没有它 feature 在构造上无法 trim**，
于是任何 e2e run 都只测到「多一次 matmul + 3× graph」。本轮的 run 就正好落在这个状态
（`profiled_cost_table=False`）。

建议：先做 SPS cost table 的 profiling 脚本（比 STS 简单得多 ——
它只是「总 verified token 数 → decode step 耗时」的一维曲线，用现成的
`dspark_sps_profiler.py` 加一个 driver 即可），存到 `examples/configs/`，
按 (模型, 并行度, GPU) 命名。STS 表需要 1-D 网格搜索，成本高，排在其后。

### Q6：tier ladder 按 batch size 门控 —— **建议做，但要一个共享 helper**

102 → ~54 个 graph，换回 ~0.4–0.8 GB/rank 的 KV。风险在于
`_get_graphs_to_capture` 与 planner 的 `allowed_lens` 一旦漂移，
planner 就会选一个没有 captured graph 的 tier，**静默掉出 graph replay** ——
那比多占的显存贵得多。所以必须让两边读同一个函数，而不是各写一遍。

### Q7：非贪婪采样 —— **已按「修 guard」处理**（见 §2.4），不再需要拍板

理由：ragged 分支本来就是完整的，静默降级是三个选项里最差的那个。

### Q9：DeepGEMM 的 `indices` varlen 参数 —— **建议本轮不动，但记录为明确的后续项**

B2 的结果（§1）说明 expanded 路径**没有**吞吐问题，所以采用 `indices` 的收益
主要是省掉 `[num_tokens, max_blocks_per_seq]` 那个 block table 的物化，
而不是速度。代价是把该路径钉死在 SM100（DeepGEMM varlen 分支断言
`arch_major == 10 and next_n == 1`）。在 B2 已经证明 expanded 不慢的前提下，
这笔交易不划算 —— 除非 block table 的 H2D 在 128K 上下文下成为瓶颈（U7，未测）。

---

## 7. 还没做的

1. **e2e 分数**（§5）—— 均匀 tier 与 ragged 两条路径都没有分数。
2. **B1 的对照实验**：`TORCH_LOGS=recompiles`，用来排除 §4 的候选 3。
3. **两个并发点**（goal doc §6.5）—— 一个都没跑。
4. **G2 逐 token 等价**、**A2 `cap-accept` 差分**。
5. **B2 在 B≥64 且 tier≥3 的点**（§1 末尾）。
6. ~~`scripts/generate_llm_args_golden_manifest.py`~~ —— **已确认**：在容器内重跑该脚本后
   `tensorrt_llm/usage/llm_args_golden_manifest.json` 无 diff。本轮只加了 validator 和一个
   私有方法，没有新增用户可见字段，所以不需要 telemetry/privacy CODEOWNER 审批。
