# DSpark Confidence-Head 调度在 DeepSeek-V4-Pro 上的落地：目标文档

> 状态：scoping / 目标定义，**尚未定稿实现方案**
> 关联：PR NVIDIA/TensorRT-LLM#17056，分支 `pr-17056`
> 目标模型：DeepSeek-V4-Pro-DSpark（target = DSv4 Pro + DSA 稀疏注意力；drafter = DSpark block drafter + confidence head）
> `max_draft_len=5` → `block_size K=5` → 均匀 verify 窗口 `next_n = 6`
> 硬件：8×Blackwell（B200/B300，SM100），DEP8 = TP8 + `moe_expert_parallel_size=8` + `enable_attention_dp=True`

本文档中所有断言均标注了 `file:line`。凡是**没有**通过读代码确认的推断，一律显式标注「待验证」。

---

## 1. 背景与现状

### 1.1 PR #17056 已经做了什么

两个 commit：

| commit | 内容 |
|---|---|
| `830afe4e58` | `[feat] Add DSpark confidence-head verification scheduling` |
| `8723506608` | `[fix] Fix host syncs, slot aliasing and cost model` |

核心数据流（已读码确认）：

1. **Confidence head 产出**：`DSparkConfidenceHead.forward` 返回 `[G, K]` fp32 raw logits，按 batch 位置排列（`tensorrt_llm/_torch/models/dspark/heads.py:258-268`），在 draft 的 captured graph 内计算（`tensorrt_llm/_torch/models/dspark/draft.py:108-121`）。
2. **batch → slot 重索引**：`DSparkWorker` 立刻把它写进按 slot 索引的常驻 device buffer `self._confidence_logits[slots] = confidence`（`tensorrt_llm/_torch/speculative/dspark.py:637-638`），buffer 形状 `[max_batch+2, K]`，其中 `max_batch+1` 行是永不写入的 neutral 行（`dspark.py:285-300`）。
3. **滞后一步的 host 读取**：`PyExecutor._dspark_confidence_draft_len` 先读上一步 stage 好的 async D2H 快照，再 stage 新的一份，**全程不 sync**（`tensorrt_llm/_torch/pyexecutor/py_executor.py:3189`；staging 见 `tensorrt_llm/_torch/speculative/dspark_verify.py:99-134`）。event 没落地就退化成 `max_tier`（verify 全量）。
4. **语义**：`py_verify_len` 计的是**每请求的 draft 位置数**，token 窗口是 `1 + py_verify_len`（`tensorrt_llm/_torch/pyexecutor/llm_request.py:1283-1286`）。

两个配置开关（`tensorrt_llm/llmapi/llm_args.py:2805` / `:2849`，默认均为 `False`，`status='prototype'`）：

| 开关 | 行为 |
|---|---|
| `enable_confidence_scheduling` | **均匀路径**：`planner.decide_draft_len` 给整个 batch 选一个 tier，把每个请求的 `py_verify_len` 显式清成 `None`（`py_executor.py:3283`），复用既有的 dynamic-draft-len pad/truncate 机制。CUDA graph 变成 `batch_size × tiers` 的叉积（`model_engine.py:1828-1856`）。 |
| `enable_ragged_verify` | **ragged 路径**：`decide_verify_lens` 给出每请求窗口，`fit_ragged_verify_lens` 把 batch token 总数吸附到某个 captured bucket，`runtime_draft_len` 钉死在**最高 tier**（因为 block 永远整块 draft），graph key 增加一条 token-count 轴。依赖 `enable_confidence_scheduling`（validator 见 `llm_args.py:2863-2871`）。 |

**默认路径（两个都 False）完全不受影响**，这一点已确认：`verify_len_tiers` 退化成 `[max_draft_len]`（`llm_args.py:2899-2905`），`py_verify_len` 从不设置，`expand_per_gen_token` 走标量 `repeat_interleave` 分支（`dsa.py:693-697`），与 PR 之前的定长扩展 bit-identical。

### 1.2 已经做对的部分

PR 的 host 侧管线质量是高的，不要推翻重做：

- `ragged_verify_lens_cuda` / `ragged_qo_indptr_cuda` 一次性预分配、原地写入（`model_engine.py:715-720`、`:3802-3813`），地址稳定，captured graph 能看到新值。
- 每个 `repeat_interleave` 都传了 `output_size=`，避免 device→host sync（`dspark_ragged.py:411-429`、`dsa.py:693-706`）。
- staging 用 pinned memory + `non_blocking=True`。
- `choose_ragged_capture_shape` 对 `peer_stats` 做跨 rank 归约，保证各 rank 落在同一个 `(padded_bs, bucket)`（`dspark_ragged.py:359-382`）。
- raggedness 活在**内容**里、shape 由 graph key 固定 —— 这是对的构造。

### 1.3 工作区当前状态（**与 investigation 描述不同，必须先对齐**）

`git status` 显示 **11 个已跟踪文件被修改**、2 个新增文件，均**未提交**。这部分工作已经超出了 PR HEAD：

```
 M cpp/tensorrt_llm/kernels/IndexerTopK.h
 M cpp/tensorrt_llm/kernels/indexerTopK.cu
 M cpp/tensorrt_llm/thop/IndexerTopKOp.cpp
 M cpp/tensorrt_llm/kernels/compressorKernels/compressorKernels.{cu,h}
 M cpp/tensorrt_llm/thop/compressorOp.cpp
 M tensorrt_llm/_torch/attention_backend/sparse/dsa.py
 M tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/{compressor.py,deepseek_v4.py}
 M tensorrt_llm/_torch/pyexecutor/py_executor.py
 M tensorrt_llm/_torch/speculative/dspark.py
?? tensorrt_llm/_torch/speculative/dspark_observability.py
?? tests/unittest/_torch/thop/parallel/test_indexer_topk_ragged.py
```

已完成的内容（读码确认）：

| 项 | 状态 | 锚点 |
|---|---|---|
| `indexer_topk_decode` 新增 `Tensor? row_kv_lens=None` | **已实现** | `IndexerTopKOp.cpp:258`（schema）、`indexerTopK.cu:665-690`（kernel 分支）、`IndexerTopKOp.cpp:57-66`（ragged 时**替换**而非放宽 TORCH_CHECK）、`indexerTopK.cu:918-922`（与 GVR `preIdx` 互斥的 TLLM_CHECK） |
| Python 侧路由 ragged → C++ top-k，`next_n` 传死值 1 | **已实现** | `dsa.py:3056-3081` |
| `compressor_paged_kv_compress` 新增 `Tensor? new_tokens_per_seq=None` | **已实现** | `compressorOp.cpp:153`（schema）、`compressorKernels.cu:340-348`（`nn` 取代 `NEXT_N`，`NEXT_N` 降级为编译期上界）、`:425`（Phase 2）；Phase 1 由 `token_idx < kv_len` 守卫（`:369`）、Phase 3 由 `num_compressions` 守卫（`:440-442`），二者对 `nn < NEXT_N` 均正确 |
| `num_gen_tokens_per_seq` ragged 时取 **max** 而非 mean | **已实现** | `deepseek_v4.py:748-769` |
| `gen_new_tokens_per_seq` 每请求向量 + graph-stable buffer | **已实现** | `deepseek_v4.py:337-352` |
| ragged verify mode 开关（static / cap-accept / compact）+ 统计 | **部分实现**，`cap-accept` 显式 `NotImplementedError` | `dspark.py:457-496`、`dspark_observability.py` |
| C++ ragged top-k 单测 | **已新增** | `tests/unittest/_torch/thop/parallel/test_indexer_topk_ragged.py` |

**重要副作用**：`num_gen_tokens_per_seq` 改成取 max 之后，compressor 输出 buffer 的 sizing 公式 `num_generations * ceil(w/cr)`（`deepseek_v4.py:753-757`、`:766-768`）重新成为**合法上界**（因为 `w_r ≤ w_max` ⇒ `Σ_r ceil(w_r/cr) ≤ n·ceil(w_max/cr)`）。这消除了「越界写 `kv_comp`」这一类内存破坏风险。取 mean 时该公式**不是**上界，是真实的 OOB 写。

### 1.4 已知的**未验证**部分

**唯一一次端到端尝试是失败的。** `.dspark-logs/gsm8k.log`（07/30 23:08，早于当前工作区改动）：

```
DSpark confidence scheduling: capturing 102 graphs (34 batch sizes x 3 draft-length tiers [1, 3, 5])
Run generation-only CUDA graph warmup (greedy) for batch size=128, draft_len=5   ← 完成
Run generation-only CUDA graph warmup (greedy) for batch size=128, draft_len=3   ← never returns
TEST_EXIT=124
```

`hang-stack-299685.txt` 停在 `thop.attention`（`fmha/fallback.py:181`）→ `trtllm.py:1775` → `deepseek_v4.py:1564` → `mla.py:1132` → `cuda_graph_runner.py:499 capture`。

注意 hang 前紧邻的日志是 `Attention workspace size is not enough, increase the size from 0 bytes to 168493568 bytes` —— **capture 期间的 workspace 重分配**是一个独立的候选根因，必须与 stride 假说区分开（见 §7）。

同时，**这次 run 其实什么都没验证到**：配置里 `enable_ragged_verify` 未开，且没给 `confidence_sps_table_path`，planner 拿到 `SpsCostTable.flat()` 后 `_decide_local` 无条件返回 `max_tier`（`dspark_verify.py:268-270`），日志确认 `profiled_cost_table=False`。也就是说该 run 的行为与 baseline 的唯一差别是：多一次 confidence matmul、多一次 D2H staging、以及 3× 的 captured graph。

---

## 2. 目标 / 非目标

### 2.1 目标（可测）

| # | 目标 | 验收方式 |
|---|---|---|
| **G1** | DSv4-Pro 上 confidence-head 调度可用：`enable_confidence_scheduling=True` 能完成 warmup + capture 且不 hang | 8×B200 上 GSM8K 测试跑到 evaluation 阶段并返回分数 |
| **G2** | 端到端正确性：调度**不改变输出分布** | temperature=0 下，`static` 与开启调度后逐 token 输出**完全一致**（比 GSM8K 分数敏感度高 3 个数量级） |
| **G3** | GSM8K 精度达标 | `score >= 96.0`（`tests/.../references/gsm8k.yaml:158-163` 记录的实测值为 96.475） |
| **G4** | **CUDA graph 兼容** | 稳态 decode 步骤 100% 命中 graph replay；capture 期间无 device→host sync、无新分配；graph key 与实际 token layout 逐字段一致（断言化，见 §6） |
| **G5** | ADP/TP8 下各 rank 形状与集合通信一致 | 各 rank 的 `(padded_bs, bucket)` 与 collectives 调用次数逐步相同；无 hang |
| **G6** | 默认路径零回归 | 两个开关关闭时与 `main` bit-identical（现已成立，需保持） |

### 2.2 隐含目标（工作本身要求，但用户未明说）

- **G7**：为「feature 到底有没有在工作」提供可观测性。当前 flat cost table 下 planner 拒绝 trim，run 看起来完全正常但收益为零 —— 必须能从日志/统计区分「调度生效」与「调度退化」。`dspark_observability.py` 是这个方向，需要落地。
- **G8**：产出两个**目前在 checkpoint 和 repo 里都不存在**的 JSON side-file：STS 校准表、profiled SPS cost table。没有它们，G2/G3 之外的任何**性能**结论都是空的。

### 2.3 非目标（明确排除）

| 排除项 | 理由 |
|---|---|
| 改变 acceptance 规则本身 | 调度只决定「送多少 draft token 给 target」，verify 逻辑不动。这是 G2 可以用逐 token 相等来验证的前提。 |
| Disaggregated serving | 生成侧首步 `py_batch_idx is None` 会命中一条已知有问题的分支（§3.3 K1），单独处理。 |
| 其他模型（DSv4-Flash、非 DSA 模型） | DSv4-Flash-DSpark 权重虽在 `llm_models_root()` 下，但 repo 里零测试；作为后续的小规模 proxy，不在本轮范围。 |
| Pipeline parallelism / LoRA | 与 vLLM #47808 声明的限制一致，本轮不支持。 |
| `enable_heuristic_topk` / `use_cute_dsl_topk` 与 ragged 共存 | GVR 的 `preIdx` 按请求索引且经 `next_n` 展开，CuTe-DSL 把 `next_n` 作为 JIT compile key。二者与 ragged 结构性互斥，**必须显式 config-reject 而非静默降级**。工作区已在 kernel 层加了 TLLM_CHECK（`indexerTopK.cu:918-922`），Python 侧的 config 校验还欠。 |
| 非贪婪采样下的 rejection sampling 等价性 | ragged 下 fail-closed guard 会关掉 rejection sampling（§3.4）。本轮先保证贪婪路径正确，非贪婪单列。 |
| 输出 logprobs | 与 vLLM 同样的限制。 |

---

## 3. 核心问题：kernel 层的 per-request verify len 不兼容

### 3.1 问题的本质

一旦每个请求有自己的 verify 窗口，**所有从标量 `next_n` 反推 `(request, offset)` 的 kernel 都会失效**。典型形式是：

```
request = rowIdx / next_n
offset  = rowIdx % next_n
causal  = seqLens[request] - next_n + offset + 1
```

而 `next_n` 的来源是 `num_gen_tokens // num_generations`（`dsa.py:2800`、`deepseek_v4.py:748-769`）—— ragged 下这是一个**整数平均值**，没有任何语义。

**最危险的一点：失败是静默的，不是报错的。** `ModelEngine.ragged_verify_token_buckets` 返回 `padded_bs * (t+1)`（`model_engine.py:3625`），token 总数**永远**是行数的整数倍，所以形如 `seq_lens.size(0) * next_n == numRows` 的 TORCH_CHECK 会**通过**，然后 kernel 把行归错请求、按错误深度做因果裁剪。这正是 vLLM 在 #47808 之前踩过的坑（他们在 `mla/indexer.py` 里留了明确注释说 `q.view(batch, next_n, ...)` 对非均匀 decode 长度会静默算错）。

DSpark 论文 §5.3 对 DeepSeek-V4 的结论与此一致：变长路由「在物理执行层引入了严峻挑战」，解法是把所有 token 展平、用一个 marker tensor 承载序列内结构，并且**「在 DeepSeek-V4 架构上，只有 index-attention 和 compress kernel 需要修改」**。这句话既是我们工作范围的上界，也是「主 sparse-MLA 路径不用动」的外部佐证。

### 3.2 DSv4-Pro DSpark 生成路径 kernel 逐项审计

按前向顺序。「ragged 容忍」列：✅ 天然支持 / ⚠️ 需 host 侧改动 / ❌ kernel 层阻断 / 🔧 工作区已修。

| # | Kernel / Op | Query 轴形状契约 | Ragged? | 阻断点与锚点 |
|---|---|---|---|---|
| **K1** | `mla_rope_generation` → `applyMLARopeAndAssignQKVKernelGeneration` | `TLLM_CHECK(acc_q_len % batch_size == 0)` 后 `seq_len = acc_q_len / batch_size`，**单一标量** | ❌ | `mlaKernels.cu:1161-1163`。派生出 `batch_idx = tok/seq_len`、RoPE `position_id = kv_len[b] - seq_len + local`、**KV 写入偏移**（两处）、`seqQOffset`。窗口 `[6,3]` → `9%2≠0` 直接 abort；`[6,2]` → 8 tokens、`seq_len=4`、check 通过但**每个请求的 K/V 写进错误的行**。这是持久性缓存污染。 |
| **K2** | trtllm-gen sparse MLA generation（`thop.attention` → `AttentionOp::mlaGeneration`） | `mMaxSeqLenQ = acc_q_len / batch_beam`；`mSumOfSeqLensQ = batch_beam * mMaxSeqLenQ` | ❌ | `attentionOp.cpp:1161,1167`。`cumSeqLensQPtr` 字段存在（`fmhaRunnerParams.h:279`）但**在 generation 分支从不赋值**（全 repo 仅 `fmhaDispatcher.cpp:213` 上下文相、`xqaDispatcher.cpp:475` 赋值）。上游 `thop/attentionOp.cpp:920-922` 还有 `TLLM_CHECK(num_tokens % num_seqs == 0, "seq_len should be same for all generation requests")`。**这是最深的阻断点。** |
| **K3** | `fp8_paged_mqa_logits` / `fp8_fp4_paged_mqa_logits`（DeepGEMM） | expanded 路径下 `q.view(-1,1,...)`、`context_lens = kv_lens_expanded[:n].view(-1,1)`、`block_table_expanded` ⇒ `next_n == 1`，**每 token 一行** | ✅ | `dsa.py:2831-2838`。形状上**已经是 ragged 布局**。关键事实：`max_draft_len=5 → next_n=6`，DeepGEMM 在 SM100 只支持 `next_n ∈ {1,2,4}`，所以 **DSv4-Pro 今天本来就跑在这条展平路径上** —— ragged 不是新 regime，它就是 regime。但注意：该 kernel 输出**未裁剪的全长 logits**（`clean_logits` 默认 false），窗口内因果裁剪**全部委托给 top-k**。 |
| **K4** | `indexer_topk_decode`（C++） | 原：`seqLens[rowIdx/next_n]`、`seq_len - next_n + rowIdx%next_n + 1` | 🔧 **工作区已修** | 新增 `Tensor? row_kv_lens`（`IndexerTopKOp.cpp:258`），kernel 分支见 `indexerTopK.cu:665-690`。ragged 时**替换**而非放宽 TORCH_CHECK（`IndexerTopKOp.cpp:57-66`）—— 这个选择是对的，因为放宽后仍会被 `padded_bs*(t+1)` 的整除性静默骗过。split-work 的 merge pass 也正确转发了指针（`indexerTopK.cu:1073-1076`，merge 阶段 `rowEnd` 被覆盖但仍需转发以避免 uniform 分支越界索引 `seqLens`）。 |
| **K5** | GVR / heuristic top-k（C++ `heuristicTopKDecode.cu` + CuTe-DSL `gvr_topk_decode.py`） | `preIdx + (rowIdx/next_n)*stride`；DSL 里 `next_n` 是 `const_expr` 且是 JIT cache key | ❌ **须 reject** | `heuristicTopKDecode.cu:57-65,103`；`IndexerTopKOp.cpp:83-84` 断言 `preIdx.size(0)*next_n == numRows`；`gvr_topk_decode.py:3733-3734,4954`。`order_row` 是**按请求**的 argsort（`dsa.py:1010-1017`），假定每请求占固定连续行数。默认关闭，但是调优部署会打开的旋钮。工作区已在 kernel 层加互斥 check。 |
| **K6** | CuTe-DSL radix / filtered top-k | `next_n` 是 `cutlass.const_expr` + compile key；每行长度 `seqlen[row//next_n] - next_n + row%next_n + 1` | ❌ **已绕开** | `single_pass_multi_cta_radix_topk.py:1062,1124-1125`。DSv4 默认不可达（`dsa.py:867` 在 `compress_ratio>1 且 next_n>1` 时早退，DSv4 的 indexer cr=4）。工作区把 ragged 无条件路由到 C++ kernel（`dsa.py:3056-3081`），进一步保证不可达。 |
| **K7** | CuTe-DSL paged-MQA atom-split `_pick_dsl_expand` | `expand_factor * atom == next_n`，`atom` 是 kernel constexpr | ❌ **结构性不可表达，已 force-off** | `dsa.py:211-263`、ragged 时 `use_dsl=False`（`dsa.py:1566-1574`）。**DSv4-Pro 上性能代价为零**：`use_cute_dsl_paged_mqa_logits` 默认 False（`llm_args.py:888-892`），默认本来就走 DeepGEMM + expanded。 |
| **K8** | `Indexer.prepare_scheduler_metadata`（DeepGEMM schedule） | 硬编码 `num_tokens = num_generations * (1 + max_draft_tokens)` | ⚠️ | `dsa.py:2329-2330`。另两处展开点已改走 `expand_per_gen_token`，唯独这里没改。**这也是均匀 tier 阶梯的 bug**（见 §3.3 H1）。 |
| **K9** | `compressor_paged_kv_compress` | 原：`NEXT_N` 是 C++ **模板常量**，`sp = kv_len - NEXT_N` | 🔧 **工作区已修** | 新增 `Tensor? new_tokens_per_seq`（`compressorOp.cpp:153`）。`nn` 取代 `NEXT_N` 做起点与 Phase-2 计数（`compressorKernels.cu:348,425`），`NEXT_N` 降级为编译期**上界**以保住 uniform 路径的全展开。Phase 1 由 `token_idx < kv_len` 守卫、Phase 3 由 `num_compressions` 守卫，二者对 `nn < NEXT_N` 均正确。`in_off`/`out_off` 本来就是精确前缀和，无需改。 |
| **K10** | DSv4 compressed-token 计数 / `kv_comp` 分配 | `num_generations * ceil(num_gen_tokens_per_seq / cr)` | 🔧 **工作区已修** | `deepseek_v4.py:748-769`。改取 **max** 后该式重新成为合法上界，消除 OOB 写。`_compute_gen_compressed_position_ids` 用 `searchsorted(cu_new_comp, ...)` 回推请求（`deepseek_v4.py:982-990`），对任何**过分配**都仍然正确。 |
| **K11** | `sparse_mla_topk_lens` / `swa_local_indices` / `compressed_local_indices` | 严格逐 token，从 `token_positions` 构造 | ✅ | `deepseek_v4.py:557-648`。**且是 overlap-safe 的**：`on_update_kv_lens` 会用修正后的 device `kv_lens` **重新推导** `token_positions` 并重跑 `prepare_for_deepseek_v4_indices`（`deepseek_v4.py:885-895`）。这条是「token-major 下丢掉 `mIsCausalSpecDecodingGen` 是安全的」的**唯一承重论据**，它是可验证的，不是假设。 |
| **K12** | DSA indexer K-cache slot mapping、`req_idx_per_token`、`convert_req_index_to_global`、`mla_rope_append_paged_kv_assign_q` | 逐 token / indptr 驱动 | ✅ | `dsa.py:1757-1769`、`:2148-2178`、`:3287-3335` |
| **K13** | `spec_decoding_generation_lengths` / packed mask（dense XQA 路径） | C++ 侧**真支持**每请求向量（`gptKernels.cu:140`），但**所有 Python 写入点都是标量 `fill_`** | ⚠️ 但 DSv4 不可达 | `trtllm.py:1189-1191`。SM100 + linear tree 下 `is_spec_decoding_enabled` 被强制 False（`trtllm.py:999-1001`），整块分配跳过；且 MLA 路径走 `mlaGeneration`，**从不读**该字段。**建议加显式 assert 而不是依赖 SM 判断。** |
| **K14** | MoE / dense MLP / allreduce / ADP `num_tokens` allgather | token-flat | ✅ | `attn_metadata.num_tokens` 就是 `_seq_lens.sum().item()`（`attention_backend/interface.py:206-215`），是真实打包计数。`fused_moe/` 与 `distributed/` 全树 grep 无 `next_n` / draft len 耦合。 |
| **K15** | DSpark drafter 自身的 forward | `[G, block]`，与 target 布局解耦 | ✅ | draft 的 attention 用 worker 自有的 `_kv_windows` 滚动 buffer，不碰 paged KV metadata。 |

### 3.3 Host 侧阻断点（与 kernel 同等重要，且**独立于方案选择**）

这些不修，任何 kernel 方案都不可能产出正确结果。

| # | 问题 | 锚点 | 严重性 |
|---|---|---|---|
| **H0** | **`disable_overlap_scheduler=True` 时，ragged token 布局根本不会被产出。** `_prepare_tp_inputs` 的非 overlap 分支条件是 `next_draft_tokens_device is None or request.is_dummy or request.py_batch_idx is None`（`model_engine.py:4606`），该分支 `sequence_lengths.append(1 + num_draft_tokens)`，**完全不看 `py_verify_len`**（`:4616,4625`）。而 `next_draft_tokens_device` 仅当 `new_tensors_device` 是 `SampleStateTensorsSpec` 时非空（`model_engine.py:4270-4277`），非 overlap 循环调用 `self._forward_step(scheduled_batch)` 不传该参数（`py_executor.py:4276`，默认 `None`，`py_executor.py:6425`）。**精度测试用的就是 `disable_overlap_scheduler=True`**（`test_llm_api_pytorch.py:4036`）。 | 上述 | **P0 / prerequisite #0**。这不是 investigation 描述的「padding 行 / 首步边缘情况」，而是在目标配置下**每个请求每一步**都命中。后果：`spec_metadata.verify_lens` / `qo_indptr` / `total_verify_tokens` 和 graph key 都在描述一个 ragged 布局，而真实 `input_ids`/`seq_lens` 是均匀的 `1+top_tier`；`_accept_draft_tokens` 随后按 `total_verify_tokens` 切片（`interface.py:1436-1441`），**逐请求 token 错位**。 |
| **H1** | **DSA 展开 stride 用的是静态 max，不是运行时 tier —— 均匀 tier 阶梯本身就是坏的。** `expand_per_gen_token` 非 ragged 分支用 `stride = 1 + self.max_draft_tokens`（`dsa.py:696`），`gen_token_repeat_list` 同（`dsa.py:662`）。`self.max_draft_tokens` 只在 `update_spec_dec_param` 赋值（`dsa.py:1401`），而 DSpark 属于 `is_parallel_draft()`（`interface.py:320-324`，`is_dspark()` 在内），故取 `original_max_total_draft_tokens = tokens_per_gen_step - 1 = 5`（`model_engine.py:342`）—— **静态值**。于是 tier 3 时 `kv_lens_expanded` / `block_table_expanded` 仍按 stride 6 铺，而 `Indexer.forward` 按 `q_decode.shape[0] = bs*4` 切片（`dsa.py:2831-2838`），请求 r>0 拿到 r-1 的 kv_len 和 block table。 | 上述 + `dsa.py:2329-2330` | **P0**。这是**所有三个方案共同需要的修复**，约 3 行 Python（`stride = spec_metadata.runtime_tokens_per_gen_step`），且**独立于 ragged 与否**。它对 07/30 那次 tier-3 capture hang 是一个完整自洽的解释：tier 5 时 768==768 通过，tier 3 时 schedule 描述 768 行而 logits 只有 512 行。**注意：`max_draft_tokens` 同时是 buffer 容量（`dsa.py:1310,1321,1342,1355`），修复时容量必须保持静态 max，只让 stride 变，否则会在 capture 期间触发重分配 —— 那是第二类 hang。** |
| **H2** | **`attn_metadata.ragged_verify_lens` 在 `attn_metadata.prepare()` **之后**才发布。** `prepare()` 在 `model_engine.py:5326`，`_attach_ragged_verify_layout`（唯一写入者）在 `:5412`，同一函数内相隔 86 行。 | 上述 | **P0**。后果：`prepare_for_spec_decode` 里的 `use_expanded_buffers_for_mtp` 判定、`kv_lens_expanded_*` 与 `block_table_expanded` 的铺设**全部用上一步的窗口**。因为 `maybe_get_cuda_graph` 返回的是 per-key 的 graph 常驻 metadata 对象（`cuda_graph_runner.py:387-407`），这个「上一步」其实是「**上一次该 key 跑的时候**」。修复：把 attach 移到 `prepare()` 之前（它只需要 `generation_requests`，已在作用域内）。 |
| **H3** | **工作区新增的 `_prepare_ragged_row_kv_lens` 有 device 不匹配，首个 ragged step 必抛异常。** `row_kv_lens.add_(expanded.to(torch.int32))`（`dsa.py:1695`）：`row_kv_lens` 是 `row_kv_lens_cuda` 的切片（CUDA），而 `expanded` 来自 `expand_per_gen_token(kv_lens[...])`，`kv_lens = cached_token_lens + seq_lens_kv` 两项**都在 CPU 上**（`dsa.py:790-801`；`cached_token_lens` 显式 `device='cpu'`）。`.to(torch.int32)` 不改变 device。同函数 docstring 写「kv_len 留在 device 上」，与自身输入不符。 | `dsa.py:1673-1695` | **P0（新引入）**。修法也更省：把逐行修正量直接折进 `row_kv_lens_host`，做**一次** H2D 拷贝 —— 这正是相邻的 `kv_lens_expanded_host` 已经在用的模式（`dsa.py:1590-1592`）。 |
| **H4** | **`row_kv_lens_cuda` 构建时机错 + 从不被 `on_update_kv_lens` 刷新。** (a) 它在 `prepare_for_spec_decode` 内构建（`dsa.py:1608`），受 H2 影响读到陈旧窗口；(b) overlap 场景下 `_preprocess_inputs` 会在 device 上把 `previous_kv_lens_offsets_cuda` 加进 `kv_lens_cuda` 再调 `on_update_kv_lens`（`model_engine.py:2921-2937`），该函数重建了 `kv_lens_expanded_cuda`、`scheduler_metadata_buffer_expanded`、slot mapping —— **但没有 `row_kv_lens_cuda`**。于是同一次 op 调用里，`gen_kv_lens_cuda`（已修正）与 `row_kv_lens`（未修正）对同一个量的说法不一致。 | `dsa.py:1608`、`dsa.py:902-977` | **P0（新引入）** |
| **H5** | **CUDA graph capture 的 token 轴来自 `key[1]`，不看 bucket。** `num_tokens_for_capture = batch_size * max_beam_width * (key[1] + 1)`（`cuda_graph_runner.py:446-453`），而 ragged 下 `key[1]` 被钉死在 top tier（`model_engine.py:1840-1842`）。 | 上述 | **P0**。结合 H0，同一 bs 下的三个 graph 都是按 `bs*(top_tier+1)` 的布局 capture 的 —— **key 里的 bucket 是虚构的**。H0 与 H5 必须**同时**修，否则 graph 只会更糟。 |
| **H6** | **overlap 下 KV rewind 读的是被下一步覆盖过的 `py_verify_len`。** `SpecSamplerBase` 已经把 batch-wide 的 `runtime_draft_len` 快照进 `SampleStateSpec`（`spec_sampler_base.py:218`）正是为了规避这个 hazard，但 `_verified_len` 却**实时**读请求上的 `py_verify_len`（`spec_sampler_base.py:195-196`）。而第 N 步的 `_handle_dynamic_draft_len`（`py_executor.py:4652`）先于第 N-1 步的 `_update_requests`（`py_executor.py:4746`）执行。 | 上述 | **P1**。修法一致：把 `py_verify_len` 同样快照进 `SampleStateSpec`。**方向修正**：investigation 声称的「泄漏 3 个 KV entry」对 KVCacheManager v1 不成立 —— `update_resources` 之后还有补偿项 `extra_rewind = _kv_reserve_draft_tokens - (py_rewind_len + accepted)`（`resource_manager.py:1051-1063`），`py_rewind_len` 在合法区间内代数抵消；真实误差只在 clamp 处出现，方向是**过度 rewind**（丢已接受的 KV）。但在 `kv_cache_manager_v2` 里补偿被 `if self.is_draft:` 门控（`kv_cache_manager_v2.py:3538-3546`），**target manager 无补偿**，那里 investigation 的场景是真的。**须先确认 DSv4-Pro 实际走哪个 manager**「待验证」。 |
| **H7** | **ragged batch 上 rejection sampling 被 fail-closed guard 静默关掉。** `_rejection_buffers_valid` 要求 `logits_rows >= num_contexts + num_gens*(draft_len+1)`（`interface.py:1707-1710`），而 ragged 下 target 只发 `num_contexts + total_verify_tokens` 行，**当调度真的在 trim 时该式必然不成立**。于是永远退回 `_sample_and_accept_draft_tokens_base`（严格 token 相等）。 | 上述 | **P1**。副作用：PR 在 rejection 路径里精心加的 ragged 分支（`interface.py:1793-1806`）成了死代码。GSM8K temperature=0 **检测不到**这一点。 |
| **H8** | **`fit_ragged_verify_lens` 假定 CUDA graph padding 已开。** 它只检查 `runner.enabled`（`model_engine.py:3644-3648`）就按 `padded_bs` 行数规划并预留 `n_pad * pad_len` 个 token，而 `_get_padded_batch` 在 `not self.padding_enabled` 时直接返回 0（`cuda_graph_runner.py:576-579`），且 `BaseCudaGraphConfig.enable_padding` **默认 False**（`llm_args.py:180-184`）。 | 上述 | **P1**。两种后果：(a) padding 关闭时每个 ragged step 静默掉出 graph replay；(b) **混合 ctx+gen batch** 无论如何都无法跑 graph，真实请求却被按「要给 pad 行留位置」缩小了窗口 —— 系统性的吞吐损失，且 `runner.ragged_pad_verify_len` 会作为陈旧状态残留。须在 config 层校验 `enable_padding=True`。 |
| **H9** | **rank-local 早退门控了条件性的 `tp_allgather`，是 hang 风险。** `decide_verify_lens` 的 `all_rank_max` 在早退之后（`dspark_verify.py:234-241,256-259`），`peer_stats` allgather 被 `if ragged_lens is not None` 门控（`py_executor.py:3258-3261`），fallback 分支又调 `decide_draft_len` 再发一次。早退条件是 rank-local 且**时序相关**（`_copy_event.query()`，`dspark_verify.py:132-133`）。 | 上述 | **P1**。走 ragged 的 rank 发 2 次集合通信，fallback 的 rank 发 1 或 3 次 → 死锁。修法：让每个 rank **无条件**发出相同序列的集合通信，再基于归约结果分支。对比之下**均匀路径天然没有这个问题**：`decide_draft_len` 的 `all_rank_max` 无条件执行（`dspark_verify.py:194-201`）。 |
| **H10** | **DSpark drafter 用固定 stride 取 target hidden state。** `Kp1 = runtime_draft_len + 1`；`base = gen_start + arange_g * Kp1`（`dspark.py:600,624`）。ragged 下 `runtime_draft_len` 被钉在 top tier，而真实偏移是 `spec_metadata.qo_indptr[r]`。`dspark.py` 全文无 `qo_indptr` 引用。 | 上述 | **P1**。静默：只是接受率崩塌，不报错（`interim_base` 还被 clamp 到 `captured.shape[0]-1`）。注意 `8723506608` 已把它从 `max_draft_len+1` 修成 `runtime_draft_len+1`，**均匀路径是对的**，只有 ragged 错。 |

### 3.4 一条重要的语义澄清

`indexer_topk_decode` 是**整条 indexer 链路上唯一施加窗口内因果裁剪的地方**。因为 K3 的 paged-MQA-logits GEMM 在 expanded 布局下 `next_n==1`，每行的 `q_pos = ctx_len - 1`，且一个请求的所有行共享同一个 `ctx_len`，所以它输出**全长未裁剪**的 logits。这意味着 `row_kv_lens` 算错**不是噪声，是让一个 verify token 看到它本不该看到的 KV**（这些 KV 确实已经写进去了，因为整个窗口在 attention 之前就已 append）—— 属于信息泄漏级别的正确性错误，不是精度抖动。

---

## 4. CUDA graph 兼容性要求

### 4.1 不变量（replay 期间必须恒定的东西）

| 类别 | 内容 |
|---|---|
| **可以原地改内容** | 所有 capture 时地址固定的预分配 buffer：`shared_static_tensors['input_ids'/'position_ids']`（`cuda_graph_runner.py:172-184`）、per-key 常驻的 `attn_metadata` 及其 `get_memory_buffers` 持有的一切、`ragged_verify_lens_cuda` / `ragged_qo_indptr_cuda`（`model_engine.py:715-720`）、新增的 `row_kv_lens_cuda`（`dsa.py:1321-1345`）、`gen_new_tokens_per_seq_cuda`（`deepseek_v4.py:337-346`）、DSpark worker 的 `captured_hidden_states` / `_kv_windows` / `_confidence_logits`。 |
| **必须恒定（capture 时冻结）** | 每一个 tensor **shape**，以及每一个用于切片或分支的 **host 标量**：`spec_metadata.total_verify_tokens`、`metadata.is_ragged_verify`、`use_expanded_buffers_for_mtp`、`next_n`、`num_gen_tokens_per_seq`、`attn_metadata.num_tokens` / `num_ctx_tokens` / `num_generations`、`all_rank_num_tokens`、`sum(ragged_verify_lens)`。 |
| **合法** | graph 私有内存池内的分配（`scatter_ragged_to_padded`、`count_accepted_ragged` 等，`dspark_ragged.py:432-512`）。 |

这条清单解释了为什么 §3 里的多数 bug 是**静默**的：`num_gen_tokens_per_seq` 之流在 capture 时被烤进 graph，replay 时既不会更新也不会报错。同理，`dsa.py` 里那些 `assert` 只在 capture 时执行过一次，replay 时永远不跑。

**一个确认是安全、不要「顺手修」的点**：`spec_metadata.total_verify_tokens` 虽然是 captured region 内读取的 host int（`interface.py:1439-1441`），但它**等于 bucket，即 `key` 的一个分量**，因此是 graph key 的函数，构造上安全。

### 4.2 Graph 数量与内存代价（DSv4-Pro / dep8 实测）

来自 `.dspark-logs/gsm8k.log` 的**实测**数字（不是估算）：

```
capturing 102 graphs (34 batch sizes x 3 draft-length tiers [1, 3, 5])
```

| 项 | 值 |
|---|---|
| baseline graph 数 | 34 |
| confidence scheduling 后 | **102**（3×） |
| tier ladder | `[1, ceil(K/2), K] = [1,3,5]`（`llm_args.py:2883-2897`） |
| 每 graph metadata 成本 | ~10–23 MB（PR 自述，`model_engine.py:1825-1826`） |
| 额外 68 个 graph | **~0.7–1.6 GB / rank** |
| dep8 聚合 | **~5–13 GB** 的 KV 容量 |

这笔钱**直接从 KV pool 里出**：captured graph 的字节落在 `extra_cost = total_used_bytes - torch_used_bytes` 项里，`_util.py:990-1015` 把它折进 `peak_memory`，而 `peak_memory` 决定 `kv_cache_max_memory`（该处日志行本身就点名了 CUDA graphs）。

**没有**第二次翻倍：`needs_non_greedy_capture` 门控在 `use_one_engine()` 上，DSPARK 不在其中（`interface.py:307-309` vs `:322-323`）；`DeepSeekV4SparseAttentionConfig.needs_separate_short_long_cuda_graphs()` 返回 False（`llm_args.py:1126-1128`）。

Warmup 会把整个 ladder 跑两遍（`model_engine.py:1220-1227`），即 204 次前向。

**这是个真实的张力**：feature 的收益按其自身 docstring 只在高并发出现（`llm_args.py:2812-2816`），而高并发恰恰是 KV 容量最吃紧的时候。缓解手段是把 tier ladder 按 batch size 门控（低于 SPS 拐点时只留 `[max_tier]`），因为 planner 的 argmax 在拐点以下本来也会选 max_tier。

### 4.3 剩余的同步点

| 位置 | 问题 |
|---|---|
| `model_engine.py:3910-3919` | `_update_target_input_tensors` 的 ragged 分支 `torch.repeat_interleave(previous_slots, tokens_per_request_device)` **没传 `output_size`**，两个参数都在 device 上 → torch 把 cumsum 读回 host 定尺寸。这正是 `dspark_ragged.py:414-424` 文档里明说不能干的事。在 capture 之外，所以不会让 capture 失败，但每个 ragged step 都在 `_apply_incremental_update_target` 里插一次 D2H sync，把 overlap scheduler 的收益吃掉。host 侧已有总数，直接传即可（旁边 `model_engine.py:5101-5109` 就做对了）。 |
| `heads.py:280-283` | `apply_sts` 在 CPU 快照上做校准，会把 CUDA 上的 `sts_temperatures` `.to('cpu')` 到**可分页**内存 —— 同步拷贝。每个 decode step 一次，且**只在提供了 profiled cost table 时触发**（否则 `is_flat` 提前返回），也就是恰好在真正要用的配置里出现。加载时缓存一份 host 副本即可消除。 |
| `py_executor.py:3232-3236` | ADP 下每 decode step 多一次阻塞式 host collective（`all_rank_max`），PR 自己的注释已承认。 |

### 4.4 Attention-DP 形状一致性

`cuda_graph_runner.py:362-372` 的 ADP 资格门只比较 **`is_all_gen_only` 与 batch size 相等**，**从不比较 token 数**。token 数走另一条 allgather（`model_engine.py:2987-2998`）进 `all_rank_num_tokens`，且因为是 host list 在 forward 内读取，会被**烤进 captured graph** —— replay 时用的是 capture 时的那份，与本步 allgather 结果无关。

顺利路径下这是自洽的（`choose_ragged_capture_shape` 归约 `peer_stats` 使各 rank 落在同一 `(padded_bs, bucket)`，且 `fill_bucket` 宁可抛异常也不返回短 batch，`dspark_ragged.py:37-48`）。**暴露面在于任一 rank 退回均匀路径**：它产出 5 元组 key 和 `bs*(chosen+1)` 个 token，而同伴产出 6 元组 key 和 bucket 个 token；batch size 仍相等所以 ADP 门通过，一个 rank replay 满是集合通信的 graph、另一个 eager 跑不同 token 数，而冻结的 `all_rank_num_tokens` 使这个不匹配在 replay 时**无法被察觉**。

**要求：ADP 门必须同时比较 bucket。**

### 4.5 一个尚未有人排查的 capture 期风险

`deepseek_v4.py` 里有三个 `@maybe_compile(dynamic=True, options={"max-autotune": True})` 的函数 —— `_prepare_deepseek_v4_indices_compiled`（`:586-589`）、`_compute_gen_compressed_position_ids`（`:955-957`）、`_compute_compressed_mask`（`:996-998`）—— 全部经 `on_update_kv_lens`（`:854-895`）在 `_preprocess_inputs`（`model_engine.py:2891-2895`）内被调用，**而那正是 `CUDAGraphRunner.capture` 录制的区域**。`_compute_gen_compressed_position_ids` 把 `num_gen_tokens_per_seq` 当 Python int 用（`:979-983`），构成 guard → **每个 tier 一次 Inductor 编译，且发生在 graph capture 期间**。

`max-autotune` 需要同步来 benchmark 候选 kernel。confidence ladder 把 capture 期编译次数变成 3 倍。这是 07/30 那次 hang 的**另一个独立候选根因**，必须与 H1 的 stride 假说区分开（见 §7）。「待验证」。

---

## 5. 方案对比

三个候选方案。注意 D1 与 D2 **不是独立选项** —— 二者都把成败押在同一个无法从源码证明的 trtllm-gen 行为上（K2 的 cubin 是预编译的）。真正的选择是 **token-major vs tier-uniform**。

| 维度 | **D1 Token-flat（`next_n ≡ 1`）** | **D2 Token-major + 3 处签名改动** | **D3 TUV 分层均匀** |
|---|---|---|---|
| **机制** | 把 batch 的 generation 半边呈现成 `num_gen_tokens` 个「长度为 1 的序列」，每 token 带自己的 kv_len / block table / cu_seqlens。所有 `rowIdx/next_n` 退化为恒等式。 | 引入每 token 行向量（`row_req_idx` / `row_kv_len` / `row_idx_end`），给 3 个 kernel 加**可选**参数；attention 仍用 token-major 呈现规避 cubin 改动。 | 不教 kernel ragged，而是把 verify 长度量化到 tier ladder，**保证任何一次 launch 只看到一个 `next_n`**。Stage A：整步一个 tier。 |
| **kernel 改动** | **0**（宣称）。compressor 无法展平（Phase 3 归约的是 Phase 1 同 block 写的状态，`compressorKernels.cu:355-400` vs `:440-530`），靠 host 侧按 tier 分组多次 launch + scratch 行绕开。 | 2 个 op 加可选参数（**工作区已完成**：K4、K9）。attention 不改。 | **0**。四个 uniform kernel 在 `num_gen_tokens = n·(K+1)` 时**逐一精确正确**（已逐条读码确认：`indexerTopK.cu:667-668`、`mlaKernels.cu:1161-1163`、`compressorKernels.cu:294,340`、`attentionOp.cpp:1161`）。 |
| **graph 数量** | 102（bs × tiers，token 轴入 key） | 102（同上） | 102 → 按 bs 门控可降到 **~54** |
| **预期收益** | 完整 Θ = τ·SPS(B) | 完整 Θ | 拿满 **SPS(B)** 项；损失 **τ** 项。D3 自估双峰 batch 下 τ 差距 ~14%（**未实测**） |
| **风险** | **高**。核心假设（K2 在 `mMaxSeqLenQ=1` 下正确且不显著变慢）无法读码证明。compressor 的 tier 分组 + nulled 行 + scratch 是全案最难 review 的部分。 | **高**（同一 K2 假设）。但 kernel 改动是加性可选参数，非 ragged 路径可证明不变。 | **中**。无 K2 假设。但 τ 留在桌上，且 Stage B（两 tier 分子 batch）依赖一个未验证的 FMHA semaphore 复用假设（`attentionOp.cpp:1126-1128`）。 |
| **工作量** | XL | L（**已完成约 40%**） | M |
| **被 D1/D2 漏掉的隐性成本** | token-major 要求 `num_seqs = host_context_lengths.size(0)`（`thop/attentionOp.cpp:1325`、`dsv3RopeOp.cpp:139`），故 `prompt_lens_cpu/cuda_runtime`、`kv_lens_runtime`、`host_request_types_runtime`、**主 KV 的 `kv_cache_block_offsets`**（与 indexer 的 `block_table_expanded` 是不同 buffer）都要按 token 膨胀；`max_num_requests` 变成 `max_batch*(1+max_draft_len)`，连带 `reserveSemaphoreArray`（`thop/attentionOp.cpp:407`）、FMHA/FlashMLA workspace（`common/attentionOp.cpp:925-955`）一起放大。4096 上下文下展开后的主 block table 约 400 KB，128K 上下文下约 12 MB，**每个 decode step 重新 H2D**。D3 完全不付这笔钱。 | 同左 | — |

### 5.1 推荐

**分三阶段推进，先把「独立于方案选择」的东西修干净并拿到真实测量，再定 kernel 方向。**

理由：

1. **§3.3 的 H0–H5 全部是 host 侧问题，且三个方案都需要。** 尤其 H1（3 行 Python）就能完整解释唯一一次实测 hang，且**独立于 ragged 与否**。在它修好、`enable_confidence_scheduling` 的均匀路径能跑通之前，讨论 D1/D2/D3 是没有依据的。
2. **决定 D1/D2 vs D3 的那个数字，谁都没测过**：trtllm-gen sparse-MLA generation 在 `(batch=B, s_q=tier+1)` 与 `(batch=B*(tier+1), s_q=1)` 两种呈现下、相同总 token 与相同逐 token 稀疏索引集时的吞吐。这决定 token-major 是否值得它的爆炸半径。它读不出来（`selectMlaGenerationKernel` 在预编译 cubin 里），但**一张 B200 上直接调 `thop.attention` 就能测**，不需要模型权重。
3. **工作区已经沿 D2 走了一半**（K4、K9 已实现且设计正确 —— 可选参数、非 ragged 路径 bit-identical、TORCH_CHECK 替换而非放宽）。这部分不该丢，它们对任何 ragged 方案都是必需的，而且对 D3 无害（不传参数即退化）。
4. **D3 有一个 D1/D2 没有的结构性优势**：`decide_draft_len` 的 `tp_allgather` 是**无条件**的（`dspark_verify.py:194-201`），且各 rank token 数恒为 `bs*(K+1)`，H9 那类 collectives 发散的 hang 在构造上不存在。同理 H7（rejection sampling）在均匀路径下是等式成立，不会被关掉。

**阶段划分：**

| 阶段 | 内容 | 交付 |
|---|---|---|
| **P0：地基（不涉及方案选择）** | H1（stride 用 runtime tier，容量保持静态 max）、H2（attach 移到 prepare 之前）、H3（`row_kv_lens` host 侧折叠 + 单次 H2D）、H4（`on_update_kv_lens` 补刷 `row_kv_lens`）、H0+H5（dummy 分支用 `get_request_tokens_per_gen_step`；`capture()` 吃 bucket —— **必须同时改**）、K8（`prepare_scheduler_metadata` 走 `expand_per_gen_token`）、§4.3 两处 sync、H8 config 校验、K5/K6/K13 显式 config-reject + assert | **均匀 tier 路径**在 8×B200 上跑通 GSM8K；hang 消失 |
| **P1：测量与定向** | 单卡 B200 上的三个微基准（见 §6.4）；产出 STS 校准表与 SPS cost table（G8） | 用数据定 token-major vs tier-uniform |
| **P2：ragged 落地** | 按 P1 结论推进 D2（已完成 K4/K9）或 D3；无论哪条都要修 H6、H7、H9、H10 | ragged 路径 GSM8K + 逐 token 等价 |

**若 P1 显示 token-major 显著变慢，D3 Stage A 就是 ship 目标** —— 它拿满 SPS 项（vLLM 报的 1.71×/2.19× 主要来自跨过吞吐拐点，而非预算分配的精细度），风险和工作量都低得多，且能把 graph 数从 102 压到 ~54。

---

## 6. 验收标准

> **所有测试都必须在 Slurm 分配的 GPU 节点上跑。当前节点不可用于任何 GPU 测试。**
> 用 `computelab-node-allocator` / `gpu-test-runner` skill 分配（B200/B300 分区，≥140 GB/GPU，≥4 h 预约）。

### 6.1 GSM8K 精度门（G3）

```bash
export LLM_MODELS_ROOT=/scratch.trt_llm_data/llm-models   # 容器内路径
cd /code/tensorrt_llm && python -m pytest \
  "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestDeepSeekV4ProDSpark::test_gsm8k_dep8_megamoe_deepgemm_confidence_scheduling" -vs
```

`MODEL_PATH` 在 **class body**（import 时）求值（`test_llm_api_pytorch.py:3964`），所以 `LLM_MODELS_ROOT` 必须在 pytest 启动**前**导出，否则整个 module 收集失败。无需 `mpirun` —— 单 pytest 进程通过 mpi4py.futures 起 8 worker。

**这个门本身不够**：GSM8K 的统计阈值是 `96.0 - 3.203 = 92.797`，允许 **42/1319** 题翻转；额外的 `score >= 96.0` 断言对实测 96.475 只有 **6.3 题**余量（1 题 = 0.0758 分），而 GSM8K 在 p=0.96 处的二项标准差约 0.54 分。它既太松（错几个 token 的 bug 能过）又太紧（干净 run 可能假失败）。

另外必须注意：该测试**没有被任何 CI stage 引用**（全 `tests/integration/` grep `confidence_scheduling` 只命中测试文件自身），且其 baseline 兄弟 `test_gsm8k_dep8_megamoe_deepgemm` 已被 `waives.txt:184` 以 nvbugs/6506920 跳过 —— **当前没有活的 baseline 可比**。

### 6.2 更锐利的不变量测试（G2，这才是真正的门）

| 测试 | 内容 | 为什么锐利 |
|---|---|---|
| **A1 逐 token 等价** | 同一 engine 配置、同一批 prompt，`static`（verify 全量）与开启调度后，temperature=0 下断言每个 prompt 的 `token_ids` **完全相等** | 贪婪投机解码定义上是 lossless，任何分歧都是 bug。把 6 题的容忍度变成 1 token 的容忍度。**注意**：跨不同 graph 形状的严格 bitwise 相等 TRT-LLM 并不保证（MoE/attention 规约顺序会变），所以诚实的形式是允许极少数 prompt 分歧并设硬上限，或用 `cap-accept` 模式（下条）。 |
| **A2 `cap-accept` 差分** | 实现 SGLang 的三模式（`static` / `cap-accept` / `compact`）。`cap-accept` 跑**均匀 kernel 路径**（`next_n=6`）但只提交每请求窗口内的 token，输出应与 `compact` **逐 token 相同** | `cap-accept` vs `compact` 的差异 ⇒ **必然是 ragged kernel bug**；`static` vs `cap-accept` 的差异 ⇒ **必然是调度/接受逻辑 bug**。把 pass/fail 变成可定位。工作区的 `dspark_observability.py` 已有 mode 枚举，`cap-accept` 目前显式 `NotImplementedError`（`dspark.py:471-478`）—— **这是需要补的**。 |
| **A3 布局一致性断言** | host 侧断言：`sum(1 + py_verify_len over padded generation_requests) == attn_metadata.num_tokens - num_ctx_tokens == spec_metadata.total_verify_tokens == key[bucket]` | 这一条就能同时抓住 H0、H5、H8 |
| **A4 时序断言** | 断言 `attn_metadata.ragged_verify_lens` 在 `prepare()` 执行的那一刻是**本步**的（而非上一步） | 抓 H2、H4 |
| **A5 接受长度分布** | 记录并断言每请求接受长度直方图，以及 `sum(accepted)` 与接受 kernel 报告一致 | 抓「token 落到了错误请求的槽位但整批仍生成合法文本」这一类 |
| **A6 调度确实生效** | 断言 `planner.stats` 的 `fallback_flat_cost` / `fallback_no_snapshot` / `fallback_no_confidence` / `fallback_short_snapshot` 计数（`dspark_verify.py:86-93`）显示 trim 路径被走到 | 否则一个静默退化的 run 会以「正确的分数、错误的理由」通过 |
| **A7 capture 期无 sync/无分配** | capture 期间断言无 device→host sync、无新分配 | G4 |

### 6.3 必须存在的单元测试

| 测试 | 现状 |
|---|---|
| C++ ragged top-k 差分：均匀 batch 用 ragged 形式表达（`row_kv_lens` 填成 uniform 算术会产出的值）必须与 `row_kv_lens=None` **bitwise 相同** | **已新增** `tests/unittest/_torch/thop/parallel/test_indexer_topk_ragged.py`（需在 GPU 节点上实跑验证） |
| compressor ragged 差分：`new_tokens_per_seq` 全填 `next_n` 必须与传 `None` bitwise 相同 | **欠缺** |
| `test_deepseek_v4_sparse_mla.py` 按每请求 q 长度参数化 | **欠缺**。该文件已构造真实的 `DeepseekV4TrtllmAttention` + cache manager（单卡 B200，`l0_b200.yml:80`，TIMEOUT 60），但其参考实现在 `:415` 和 `:473` **硬编码** `seq_len_q = fused_q.shape[0] // num_requests` —— 那正是被测假设本身。**这是性价比最高的一个测试**，单卡分钟级就能抓住 tier-3 那类 hang。 |
| `_prepare_tp_inputs` 在混合 `py_verify_len` + pad 行的 batch 上，断言 `sum(attn_metadata.seq_lens) == _ragged_verify_bucket(batch)` | **欠缺**。现有 220 个 hw_agnostic 测试全部停在纯 Python 布局算术；`test_dspark_ragged.py:515-521` 的 `_fit` 是 `fit_ragged_verify_lens` 的**手写镜像**而非真函数，所以 §3.3 的每一个 bug 对 CI 都不可见。 |

### 6.4 定向微基准（P1，单卡 B200，分钟级）

| # | 内容 | 用途 |
|---|---|---|
| **B1** | 修完 H1 后重跑均匀 tier capture，确认 hang 消失 | 区分 H1 stride 假说 vs §4.5 torch.compile 假说。**跑时带 `TORCH_LOGS=recompiles`。** |
| **B2** | 直接调 `thop.attention`，比较 `(batch=B, s_q=tier+1)` 与 `(batch=B*(tier+1), s_q=1)` | **决定 D1/D2 vs D3 的那个数字** |
| **B3** | expanded 布局下 paged-MQA-logits 的吞吐 | 展平路径的 KV 重读代价（每请求 6 次而非 2 次）在关键路径上，必须实测 |

### 6.5 GSM8K 的并发要求

vLLM #47808 在**并发 16 和 64** 两个点上报了 0.945 / 0.951。原因很实际：**调度只在高并发下才真的 trim**，单一低并发的 GSM8K run 是空跑。本项目的验收也必须至少覆盖两个并发点。

---

## 7. 风险与未知

### 7.1 必须在定稿前测量的

| # | 未知 | 为什么阻塞 | 怎么解 |
|---|---|---|---|
| **U1** | trtllm-gen sparse-MLA generation 在 `mMaxSeqLenQ=1` + 6× batch 下 vs `s_q=6` 的吞吐 | **决定 token-major 是否可行**。DSv4-Pro 在 DEP8 下每 rank 保有全部 128 个 Q head，BMM1 的 M 维从 768 降到 128；由于稀疏索引集逐行不同，本来也没有跨 6 行共享 KV tile 的机会 —— 但这是**推断，不是测量**。 | B2 |
| **U2** | 07/30 hang 的真实根因 | H1 的 stride 不匹配是一个完整自洽的解释（tier 5 时 768==768 通过，tier 3 时 768 vs 512），但日志里紧邻 hang 的是 **attention workspace 从 0 扩到 168 MB** 的警告，`model_engine.py:1789-1806` 明确记录 capture 期 workspace 重分配是真实 hazard，而 Case 2b **没有** Case 2 那样的 `max_spec_graph` 规避（`model_engine.py:1810-1813`）。§4.5 的 torch.compile / max-autotune 是第三个候选。 | B1 + `TORCH_LOGS=recompiles`；必要时逐个消除 |
| **U3** | ragged 相对 tier-uniform 在固定 token 预算下的 τ 收益 | 决定 D3 是否「够好」。D3 自估双峰 batch ~14%、同质 batch ~0%，**没有任何真实 confidence head 的数据支撑** | 需要先有 STS 表（U4） |
| **U4** | STS 校准表 | `sts_temperatures` 初始化为全 1（`heads.py:249-258`），checkpoint 的 index 里**没有**任何 sts/temperature/calibration 张量，repo 里也没有该 JSON。论文 §3.2.1 明确是**每个位置一个温度**的 1D 网格搜索。未校准的逐位置偏差在累积乘积下**几何放大**。 | 必须先 profile 产出 |
| **U5** | SPS cost table | 没有它 `SpsCostTable.flat()` ⇒ `is_flat` ⇒ `_decide_local` 无条件返回 `max_tier`（`dspark_verify.py:268-270`）⇒ **feature 在构造上无法 trim**。任何在此之前的 A/B 只测到了「多一次 matmul + 3× graph」。 | 必须先 profile 产出 |
| **U6** | DSv4-Pro 实际走 KVCacheManager v1 还是 v2 | H6 的**症状和严重性按 manager 不同**：v1 有补偿项、误差方向是过度 rewind；v2 的 target 路径无补偿（`kv_cache_manager_v2.py:3538-3551`），investigation 的「泄漏」场景才成立。说错会把 debug 引向反方向。 | 读配置/加日志确认 |
| **U7** | token-major 下 buffer 膨胀的真实代价 | `num_seqs` 从 B 变成 ~6B，牵连 semaphore array、FMHA workspace、**主 KV block table 每步重新 H2D**。128K 上下文下约 12 MB/step。 | 若走 token-major，需容量审计 + 实测 |

### 7.2 已知但可接受 / 需显式处理

- **性能故事必须诚实**：**draft pass 是不可裁剪的** —— block 永远整块 draft（`runtime_draft_len` 在 ragged 下钉在 top tier），只有 verification 被裁。所以收益是纯粹的 target 侧 SPS 效应，**只在高并发出现**，小 batch 下可能持平甚至因为 3× graph 占用 KV 而**净负**。vLLM 报的 1.71×(c=128) / 2.19×(c=256) 的对照组是「固定 7-token verify」，而后者在 c=256 时比不投机还慢 33% —— 那是被修复的病态，不是通用加速比。
- **`derive_verify_len_tiers` 实际是死代码**：`_build_verify_planner` 只在 `verify_len_tiers` 为假值时才调它（`dspark.py:433-450`），而该属性总是返回非空 list（`llm_args.py:2899-2905`）。这**其实是安全的**（planner 的 tier 必须与 `_get_graphs_to_capture` capture 的一致），但 `dspark.py:450` 的日志永不触发，reviewer 会误以为 ladder 是 profile 派生的。要么删分支、要么让派生结果同时喂给 capture 列表。
- **Case 2b 丢了 `draft_len=0` 的 graph**：它提前 return（`model_engine.py:1848,1856`），从不到达 `should_capture_no_spec` 逻辑（`:1867-1873`），而 tier ladder 最小值是 1。一旦 speculation gate 因低接受率关掉投机（`py_executor.py:3317-3322` 设 `runtime_draft_len = 0`），key 变成从未 capture 过的 `(bs, 0, ...)`，**之后每个 decode step 都 eager** —— 大 MoE 模型上是巨大的静默性能悬崖。
- **PR 的外部引用有误**，会削弱 reviewer 信任：算法在论文 **§3.2.2**（Algorithm 1）而非 §5；SGLang **没有** `dspark_schedule.py`（其真实布局是 `speculative/ragged_verify.py` + `speculative/dspark_components/*`）；`align_verify_tokens_to_graph_tier` 全网零命中（SGLang 的实际原语是 `round_up_grid()` / `compute_target_verify_graph_key()`）。
- **论文 §5.2 支持 PR 的一处偏离**：Algorithm 1 的 greedy first-descent 被论文自己否定（「真实硬件容量 SPS(B) 是离散的、阶梯状退化的」），生产中「移除 early-stopping break，启用无约束全局搜索」—— 这正是 PR 采用 global argmax 的理由，可以引用。但论文用的是 **t-2** 快照（为兼容 Zero-Overhead Scheduling 并形成「因果屏障」），PR 用的是 t-1。**t-1 是否足够早取决于 TRT-LLM 的 overlap scheduler，不取决于论文**「待验证」。

### 7.3 上游可复用的东西（尚未采用）

TRT-LLM pin 的 DeepGEMM commit（`245dc5d6`，`cpp/build/_deps/deepgemm-subbuild/CMakeLists.txt:30`）**已经**在 `get_paged_mqa_logits_metadata` / `fp8_paged_mqa_logits` / `fp8_fp4_paged_mqa_logits` 上暴露了可选的 `indices` 参数（varlen 分支由 `indices.has_value()` 门控，断言 `dim()==1 && size(0)==batch_size`、int32、contiguous）—— 这就是论文说的「marker tensor」。vLLM #47808 用的正是它（`indices=decode_metadata.indices`）。

TRT-LLM 的 `dsa.py` **从不传 `indices`**，而是用既有的 "expanded buffers" 展平达到同样形状。二者数值等价，差别在于 expanded 路径要物化一个 `[num_tokens, max_blocks_per_seq]` 的 block table，而 varlen 只需一个 1-D int32 marker。**限制**：DeepGEMM 的 varlen 分支断言 `arch_major == 10 and next_n == 1` —— 仅 SM100，且必须是展平形式。这条路值得作为一个**有意的决定**评估，而不是默认忽略。

---

## 8. 开放问题（需要你拍板）

1. **token-major 还是 tier-uniform？** 这是全文最大的分叉。建议先跑 B2 微基准再定 —— 但如果你出于其他原因（例如已经决定要与 SGLang/vLLM 的实现保持形态一致）倾向某一边，可以省掉这一步。

2. **`enable_ragged_verify` 是否是本轮必须交付的？** 若可以接受先只交付**均匀 tier 阶梯**（D3 Stage A），P0 阶段结束就能 ship 一个正确且 CUDA-graph 兼容的版本，把 ragged 留作 follow-up。这会显著降低风险，代价是 τ 项（未量化）。

3. **工作区那 11 个未提交的改动如何处置？** K4/K9 的实现质量是好的（可选参数、非 ragged bit-identical、TORCH_CHECK 替换而非放宽），但 H3/H4 是新引入的 P0 bug。建议：先修 H3/H4 再提交，**不要**在带着必抛异常的代码的状态下推进其他工作。

4. **`cap-accept` 模式要不要实现？** 它是把 GSM8K 从「pass/fail 硬币」变成「可定位诊断」的关键（§6.2 A2），成本不高（复用 PR 已有的 `torch.minimum(..., verify_lens)` clamp，`interface.py:1473-1480`、`:1889-1896`）。当前是显式 `NotImplementedError`。

5. **STS 表和 SPS 表由谁、按什么流程产出？** 这是 U4/U5，也是 G8。在它们存在之前，任何**性能**结论都是空的，而且 feature 在构造上无法 trim。需要一个 profiling 脚本 + 产物的存放约定（checkpoint 内？repo 内 `examples/configs/`？）。

6. **tier ladder 要不要按 batch size 门控？** 低于 SPS 拐点时 planner 本来也会选 `max_tier`，门控后 graph 数从 102 降到 ~54，换回 ~0.4–0.8 GB/rank 的 KV。需要一个共享 helper 让 `_get_graphs_to_capture`（`model_engine.py:1841-1856`）与 planner 的 `allowed_lens`（`dspark_verify.py:280`）**不可能漂移**。

7. **非贪婪采样（H7）的处理时机？** 现在 ragged batch 上 rejection sampling 被 fail-closed guard 静默关掉，改变了 temperature>0 的采样分布。要么本轮修 guard（放宽成 `num_contexts + total_verify_tokens`），要么显式在 config 层拒绝「ragged + 非贪婪」组合。**静默降级是最差的选项。**

8. **KVCacheManager v1 还是 v2？**（U6）影响 H6 的修复优先级和症状描述。

9. **要不要采用 DeepGEMM 的 `indices` varlen 参数？**（§7.3）能去掉一个大 buffer 并与 vLLM 对齐，但把该路径钉死在 SM100。
