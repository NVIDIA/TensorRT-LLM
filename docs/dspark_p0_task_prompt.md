# 任务：让 DSpark confidence-head 调度在 DeepSeek-V4-Pro 上跑通（P0 阶段）

> 本文是给 agent 的任务书。全部路径、commit、权重张量名均已在 `bia` 集群 login
> 节点上实地核实过（2026-08-01）。

---

## 0. 先读这个（不要跳过，也不要从零设计方案）

本任务的 **唯一 source of truth** 是 PR NVIDIA/TensorRT-LLM#17056 里的
`docs/dspark_confidence_schedule_goal.md`（399 行中文目标文档）。

它已经完成了全部前期分析：H0–H10 十个 host 侧阻断点、K1–K15 kernel 逐项审计、
D1/D2/D3 三个方案对比、G1–G8 验收标准、§7 的 U1–U7 未知项、§8 的 9 个待拍板
问题。**每一条断言都带 `file:line` 锚点。**

你的工作是**执行它的 P0 阶段**，不是重新做分析。任何与该文档冲突的结论，必须先用
`file:line` 证明文档错了，再改。

配套还有 `docs/dspark_ragged_verify_status.md`（388 行）。

### 拉取方式

本地 checkout：`/lustre/fsw/coreai_comparch_trtllm/laliao/repos/trtllm-dsv4/TensorRT-LLM`
（`origin` = `github.com/lancelly/TensorRT-LLM`，`upstream` = `github.com/NVIDIA/TensorRT-LLM`）

```bash
cd /lustre/fsw/coreai_comparch_trtllm/laliao/repos/trtllm-dsv4/TensorRT-LLM
git fetch upstream refs/pull/17056/head:pr-17056
git checkout pr-17056        # head = d8897046653331bca51242bf068b052568fca4ce
git show pr-17056:docs/dspark_confidence_schedule_goal.md
```

PR 的 4 个 commit：

| commit | 内容 |
|---|---|
| `830afe4e58` | feat: Add DSpark confidence-head verification scheduling（26 files, +4683） |
| `8723506608` | fix: Fix host syncs, slot aliasing and cost model（13 files, +641） |
| `e1c28eeae2` | feat: Make DSpark ragged verification work on the DSv4 kernel path（21 files, +4172） |
| `d889704665` | doc: Scope the ragged-verify measurements to x86 DGX B300 |

---

## 1. 三个必须先纠正的认知

### (a) confidence head 已经实现了

不要从零写。代码在：

- `tensorrt_llm/_torch/models/dspark/heads.py`（`DSparkConfidenceHead`）
- `tensorrt_llm/_torch/models/dspark/{draft,attention}.py`、`modeling_dspark.py`
- `tensorrt_llm/_torch/speculative/dspark{,_planner,_verify,_ragged,_schedule,_observability,_sps_profiler}.py`
- 开关：`llm_args.py` 的 `enable_confidence_scheduling` / `enable_ragged_verify`
  （均默认 `False`，`status='prototype'`）

任务是**让它正确、且 CUDA-graph 兼容地跑起来**，不是新增功能。

### (b) goal doc §1.3 描述的「11 个未提交的工作区改动」现在已经是 commit `e1c28eeae2`

文件清单与该 commit 逐项吻合。所以文档里标为「新引入的 P0 bug」的 **H3** 和 **H4**
现在是**已经进了 PR head 的 bug**：

- **H3**：`dsa.py:1673-1695`，`row_kv_lens.add_(expanded.to(torch.int32))` —— CUDA
  tensor 加 CPU tensor（`kv_lens = cached_token_lens + seq_lens_kv` 两项都在 CPU），
  首个 ragged step 必抛异常。修法：把逐行修正量折进 `row_kv_lens_host` 做**一次**
  H2D，复用相邻 `kv_lens_expanded_host` 已有模式（`dsa.py:1590-1592`）。
- **H4**：`row_kv_lens_cuda` 从不被 `on_update_kv_lens` 刷新。

另外注意：本地 checkout 的工作区是干净的，goal doc §1.3 描述的是**另一个**工作区
的状态，不要去找那 11 个「未提交」的文件。

### (c) PR 的外部引用有误（goal doc §7.2 已指出）

找参考实现时用正确的地址：

- **SGLang 没有 `dspark_schedule.py`**。真实布局是
  `python/sglang/srt/speculative/ragged_verify.py` +
  `python/sglang/srt/speculative/dspark_components/*`。
  三模式 `static` / `cap-accept` / `compact` 出自这里。
- **vLLM 是 PR vllm-project/vllm#47808**。关键参考点：
  - `vllm/v1/attention/backends/mla/indexer.py` 里关于 `q.view(batch, next_n, ...)`
    对非均匀 decode 长度**静默算错**的注释
  - 它如何使用 DeepGEMM 的 varlen `indices` 参数（TRT-LLM 目前用 "expanded
    buffers" 展平达到同样形状，从不传 `indices`；见 goal doc §7.3）
- `align_verify_tokens_to_graph_tier` 这个符号全网不存在。SGLang 的实际原语是
  `round_up_grid()` / `compute_target_verify_graph_key()`。

---

## 2. 目标配置

- **模型**：DeepSeek-V4-Pro-DSpark
  （target = DSv4 Pro + DSA 稀疏注意力；drafter = DSpark block drafter + confidence head）
- **并行**：DEP8 = TP8 + `moe_expert_parallel_size=8` + `enable_attention_dp=True`
- **投机**：`max_draft_len=5` → block K=5 → 均匀 verify 窗口 `next_n=6`
- **硬件**：8×Blackwell（B200/B300，SM100）

### 2.1 权重（已实地确认存在且可读）

```
/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-ai_deepseek-v4-pro-dspark/hf/hf-cd80265_orig
```

832 GB，66 shards，149782 个 tensor。路径 world-readable。

**决定性证据 —— confidence head 权重就在里面：**

```
mtp.2.confidence_head.proj.weight    BF16   [1, 7680]   (model-00066-of-00066.safetensors)
```

checkpoint 结构：`mtp.0` / `mtp.1` / `mtp.2` 三个 DSpark block，只有 `mtp.2` 带
confidence head。config 关键字段：

| key | 值 |
|---|---|
| `dspark_block_size` | **5**（与 `max_draft_len=5` 对上） |
| `dspark_target_layer_ids` | `[58, 59, 60]` |
| `dspark_markov_rank` | 512 |
| `dspark_noise_token_id` | 128799 |
| `expert_dtype` | fp4 |
| `index_topk` | **1024**（普通 dsv4-pro 是 512） |
| `num_nextn_predict_layers` | 1 |

**全 index grep `sts|temperature|calib` 零命中** —— 印证 goal doc 的 U4：STS 校准表
既不在 checkpoint 也不在 repo，必须自己 profile 产出。

### 2.2 必须先建软链

测试里是 `MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4-Pro-DSpark"`，而实际目录名
是 `hf-cd80265_orig` 且嵌了两层：

```bash
mkdir -p ~/llm-models
ln -s /lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-ai_deepseek-v4-pro-dspark/hf/hf-cd80265_orig \
      ~/llm-models/DeepSeek-V4-Pro-DSpark
export LLM_MODELS_ROOT=~/llm-models
```

`MODEL_PATH` 在 class body（import 时）求值，`LLM_MODELS_ROOT` 必须在 pytest 启动
**前**导出，否则整个 module 收集失败。

容器里 `~` 可能不是同一个 home，建议软链建在 `/lustre/fsw/coreai_comparch_trtllm/laliao/llm-models/`
下并用绝对路径导出。

### 2.3 用不到但容易认错的几个目录

| 路径 | 是什么 | 为什么不是你要的 |
|---|---|---|
| `/lustre/share/coreai_comparch_trtllm/dsv4-pro` | 806 GB，纯 DSv4-Pro | 只有 `mtp.0`，**无 confidence head** |
| `/lustre/share/coreai_comparch_trtllm/dsv4-flash` | 149 GB | DSv4-Flash，非本轮范围 |
| `/lustre/share/coreai_comparch_aarwlt/hf_repos/deepseek-ai/DeepSeek-V4-{Pro,Flash}` | HF 原始副本 | 同上，无 DSpark |
| `/lustre/share/coreai_comparch_trtllm/llm-models/` | **空目录** | 不要指向这里 |

### 2.4 ⚠️ 与现有测试的冲突（必须先解决）

现有测试 `TestDeepSeekV4ProDSpark::test_gsm8k_dep8_megamoe_deepgemm_confidence_scheduling`
（`tests/integration/defs/accuracy/test_llm_api_pytorch.py`）用的是
**`disable_overlap_scheduler=True`**。

但本任务**要求 overlap scheduler 开启**。这不是配置项调换，而是把 goal doc 的
**H0 从边缘情况变成第一优先级**：

> **H0**：`disable_overlap_scheduler=True` 时，ragged token 布局**根本不会被产出**。
> `_prepare_tp_inputs` 的非 overlap 分支（`model_engine.py:4606`）条件含
> `next_draft_tokens_device is None`，该分支 `sequence_lengths.append(1 + num_draft_tokens)`，
> **完全不看 `py_verify_len`**（`:4616,4625`）。后果：`spec_metadata.verify_lens` /
> `qo_indptr` / `total_verify_tokens` 和 graph key 都在描述一个 ragged 布局，而真实
> `input_ids`/`seq_lens` 是均匀的 `1+top_tier`，`_accept_draft_tokens` 随后按
> `total_verify_tokens` 切片（`interface.py:1436-1441`）→ **逐请求 token 错位**。

goal doc §3.3 明确要求 **H0 与 H5 必须同时修**（H5：CUDA graph capture 的 token 轴
来自 `key[1]` 而不看 bucket，`cuda_graph_runner.py:446-453`），否则 graph 只会更糟。

**默认做法**：新增一个 `disable_overlap_scheduler=False` 的 test case，保留原 case
作对照。若要直接改原 case，先确认。

### 2.5 已经拍板的两个决策（goal doc §8 的 Q2 / Q8 已关闭）

goal doc §8 把这两条列为待拍板，**现在已定，不要再当开放问题讨论**。

#### 决策一：**ragged path 是第一优先级**（关闭 Q2）

Q2 原本问的是：本轮交付「均匀 tier 阶梯」还是「ragged」。两者的区别是——

| | **均匀 tier 阶梯（D3 Stage A）** | **ragged（D1/D2）** |
|---|---|---|
| 语义 | 每个 decode step **整个 batch 用同一个 verify 长度**，从 tier ladder `[1,3,5]` 里选一个 | **每个请求各自的 verify 窗口**，同一步里可以是 `[6,3,5,2,...]` |
| kernel | 0 改动。四个 uniform kernel 在 `num_gen_tokens = n·(K+1)` 时逐一精确正确 | 需要 kernel 支持每 token 行向量；K4/K9 已完成 |
| 收益 | 拿满 SPS(B) 项，**损失 τ 项** | 完整 Θ = τ·SPS(B) |
| 风险 | 中。无 K2 假设 | **高**。押在 K2（trtllm-gen sparse-MLA generation）这个读码无法证明的行为上 |

> **后续（见 §10.9）**：均匀 tier 阶梯不只是「没选它」，而是**已从代码里整个删除**。
> 它在实测中退化成「验证整块」，却仍占着一条独立代码路径。下表保留作决策记录。

**已定：做 ragged。** 因此：

1. goal doc §5.1 里划给 **P2 的 H6 / H7 / H9 / H10 全部上提到 P0**（见 §3 表二）。
2. **K2 从「以后再说」变成本轮最大风险**。它的确切阻断点已核实：
   `cpp/tensorrt_llm/thop/attentionOp.cpp:920-921`
   ```cpp
   TLLM_CHECK_WITH_INFO(num_tokens % num_seqs == 0,
       "seq_len should be same for all generation requests, num_tokens=%d, num_seqs=%d",
       num_tokens, num_seqs);
   int32_t const input_seq_length = num_tokens / num_seqs;
   ```
   加上 `mMaxSeqLenQ = acc_q_len / batch_beam`（`attentionOp.cpp:1161,1167`），以及
   `cumSeqLensQPtr` 字段虽存在（`fmhaRunnerParams.h:279`）但**在 generation 分支从不
   赋值**。D1 和 D2 都靠把 generation 半边**呈现成 `num_gen_tokens` 个「长度为 1 的
   序列」**来让这个检查平凡通过（`seq_len=1`），二者押的是同一个赌注。
3. **因此 B2 微基准从 P1 上提为 P0 的 gating 测量**，而且要**最先跑**：它单卡、
   分钟级、**不需要模型权重**，却决定整条 ragged 路线可不可行。见 §3 表三。

#### 决策二：**DSv4-Pro 走 KVCacheManager v2**（关闭 Q8 / U6）

已确认。后果：

- **H6 上提为 P0**（原 P1）。
- 修法不变且与 manager 版本无关：`SpecSamplerBase._verified_len` **实时**读请求上的
  `py_verify_len`（`spec_sampler_base.py:195-196`），而第 N 步的
  `_handle_dynamic_draft_len`（`py_executor.py:4652`）先于第 N-1 步的
  `_update_requests`（`py_executor.py:4746`）执行 → 读到被下一步覆盖过的值。
  `runtime_draft_len` 已经为此快照进 `SampleStateSpec`（`spec_sampler_base.py:218`），
  **把 `py_verify_len` 一并快照进去即可**。
- ⚠️ **一处需要你自己核实的措辞**：goal doc 说 v2 的补偿项被 `if self.is_draft:`
  门控、「target manager 无补偿」所以会泄漏 KV。但该分支紧邻的注释明确写着
  *"Target managers do not allocate this reserve slack."*
  （`kv_cache_manager_v2.py:3538-3546`）——即 target 本来就没分配这块 slack，不补偿
  是**正确的**。所以「泄漏」这个结论还差一步论证。
  **`py_verify_len` 读到陈旧值这个 bug 本身独立成立**，先按这个修；「是否额外泄漏」
  作为一个独立问题实测确认，标注「待验证」。

---

## 3. 执行范围：P0 阶段

因为 ragged 是第一优先级（§2.5 决策一），P0 的范围比 goal doc §5.1 原本划的要大：
它 = 原 P0（地基）+ 原 P2 的 ragged 必需项 + 原 P1 的 B2 测量。

**执行顺序有讲究，不要并行乱做：**

```
表三 B2 微基准（单卡，无需权重）  ──┐
                                    ├──► 表一 地基（H0–H5, K8, sync, H8, K5/K6/K13）
                                    │         └──► 均匀 tier 路径跑通 = 中间里程碑
                                    └──► 表二 ragged 必需项（H6, H7, H9, H10）
                                              └──► ragged 路径跑通 = 交付
```

- **B2 最先跑**，它可能直接否掉整条 ragged 路线（§2.5 决策一第 3 点）。
- **表一必须先于表二**。表一全部是「无论 ragged 与否都要修」的东西，其中 H1 只有
  约 3 行 Python 却能完整解释唯一一次实测 hang。在均匀 tier 路径跑通之前调 ragged，
  等于在两层未知上 debug。
- **均匀 tier 路径不是可以砍掉的中间产物**：G2 的逐 token 等价、A2 的 `cap-accept`
  差分都要拿它当参照系（§6）。

### 表一：地基（全部 host 侧，独立于 kernel 方案选择）

| 项 | 内容 | 锚点 |
|---|---|---|
| **H1** | DSA 展开 stride 用 runtime tier 而非静态 max。**约 3 行 Python**（`stride = spec_metadata.runtime_tokens_per_gen_step`）。⚠️ **buffer 容量必须保持静态 max**，只让 stride 变，否则 capture 期重分配 → 第二类 hang | `dsa.py:662,696,1401`、`model_engine.py:342` |
| **H2** | `_attach_ragged_verify_layout` 移到 `attn_metadata.prepare()` **之前**（它只需要 `generation_requests`，已在作用域内） | `model_engine.py:5326` vs `:5412` |
| **H3** | `row_kv_lens` device 不匹配。修法：折进 `row_kv_lens_host` + 单次 H2D | `dsa.py:1673-1695` |
| **H4** | `on_update_kv_lens` 补刷 `row_kv_lens_cuda` | `dsa.py:1608`、`:902-977` |
| **H0+H5** | **必须同时改**。dummy 分支用 `get_request_tokens_per_gen_step`；`capture()` 吃 bucket | `model_engine.py:4606`、`cuda_graph_runner.py:446-453` |
| **K8** | `prepare_scheduler_metadata` 改走 `expand_per_gen_token`（现在硬编码 `num_generations * (1 + max_draft_tokens)`） | `dsa.py:2329-2330` |
| **sync** | `_update_target_input_tensors` 的 `repeat_interleave` 补 `output_size=`（旁边 `model_engine.py:5101-5109` 已做对）；`apply_sts` 缓存 host 副本 | `model_engine.py:3910-3919`、`heads.py:280-283` |
| **H8** | config 层校验 `cuda_graph_config.enable_padding=True`（默认 False，`fit_ragged_verify_lens` 却假定已开） | `model_engine.py:3644-3648`、`llm_args.py:180-184` |
| **K5/K6/K13** | 显式 **config-reject** 而非静默降级：`enable_heuristic_topk` / `use_cute_dsl_topk` 与 ragged 结构性互斥（kernel 层已有 TLLM_CHECK，Python config 校验欠缺）；K13 加显式 assert 而非依赖 SM 判断 | `indexerTopK.cu:918-922`、`trtllm.py:999-1001` |

**表一交付（中间里程碑）**：均匀 tier 路径（`enable_confidence_scheduling=True`、
`enable_ragged_verify=False`）在 8×B200/B300 上跑通 GSM8K，hang 消失。

### 表二：ragged 必需项（原 goal doc P2，因决策一上提到 P0）

| 项 | 内容 | 锚点 |
|---|---|---|
| **H6** | `py_verify_len` 快照进 `SampleStateSpec`，不要实时读。**KVCacheManager v2 已确认**，见 §2.5 决策二（含一处需自行核实的措辞） | `spec_sampler_base.py:195-196,218`、`py_executor.py:4652,4746`、`kv_cache_manager_v2.py:3538-3546` |
| **H7** | ragged batch 上 rejection sampling 被 fail-closed guard **静默关掉**：`_rejection_buffers_valid` 要求 `logits_rows >= num_contexts + num_gens*(draft_len+1)`，而 ragged 下 target 只发 `num_contexts + total_verify_tokens` 行 —— **调度真的在 trim 时该式必然不成立**。于是 PR 在 rejection 路径里精心加的 ragged 分支（`interface.py:1793-1806`）成了死代码。**GSM8K temperature=0 检测不到这一点。** 要么放宽 guard 成 `num_contexts + total_verify_tokens`，要么在 config 层显式拒绝「ragged + 非贪婪」。**静默降级是最差的选项** | `interface.py:1707-1710,1793-1806` |
| **H9** | **rank-local 早退门控了条件性的 `tp_allgather`，是 hang 风险。** `decide_verify_lens` 的 `all_rank_max` 在早退之后，`peer_stats` allgather 被 `if ragged_lens is not None` 门控，fallback 分支又发一次 → 走 ragged 的 rank 发 2 次集合通信、fallback 的 rank 发 1 或 3 次 → **死锁**。且早退条件是 rank-local 且**时序相关**（`_copy_event.query()`）。修法：每个 rank **无条件**发出相同序列的集合通信，再基于归约结果分支 | `dspark_verify.py:234-241,256-259,132-133`、`py_executor.py:3258-3261` |
| **H10** | DSpark drafter 用固定 stride 取 target hidden state：`base = gen_start + arange_g * (runtime_draft_len + 1)`，而 ragged 下真实偏移是 `spec_metadata.qo_indptr[r]`（`dspark.py` 全文无 `qo_indptr` 引用）。**静默：只是接受率崩塌，不报错**。注意 `8723506608` 已把它从 `max_draft_len+1` 修成 `runtime_draft_len+1`，**均匀路径是对的**，只有 ragged 错 | `dspark.py:600,624` |
| **§4.4** | **ADP 资格门必须同时比较 bucket**。现在只比 `is_all_gen_only` 和 batch size，从不比 token 数；`all_rank_num_tokens` 是 host list、会被**烤进 captured graph**，所以「任一 rank 退回均匀路径」造成的不匹配在 replay 时**无法被察觉** | `cuda_graph_runner.py:362-372`、`model_engine.py:2987-2998` |

**表二交付**：ragged 路径（两个开关都开）GSM8K + 逐 token 等价（§6）。

### 表三：B2 微基准（最先跑，单卡，无需模型权重）

**这是决定整条 ragged 路线可不可行的那个数字**（goal doc §7.1 U1）：

> 直接调 `thop.attention`，在**相同总 token、相同逐 token 稀疏索引集**下，比较
> `(batch=B, s_q=tier+1)` 与 `(batch=B*(tier+1), s_q=1)` 两种呈现的**正确性与吞吐**。

- 读不出来（`selectMlaGenerationKernel` 在预编译 cubin 里），但一张 B200/B300 上
  直接测就行，分钟级。
- 关注点：DSv4-Pro 在 DEP8 下每 rank 保有全部 128 个 Q head，BMM1 的 M 维从 768 降到
  128；由于稀疏索引集逐行不同，本来也没有跨 6 行共享 KV tile 的机会 —— 但这是
  **推断，不是测量**。
- **若 token-major 显著变慢或不正确**：立刻停下来报告，不要硬推 ragged。
  goal doc §5.1 的兜底是 D3 Stage A（均匀 tier 阶梯），它拿满 SPS(B) 项、无 K2 假设、
  还能把 graph 数从 102 压到 ~54。

顺带把 goal doc §6.4 的另两个也跑了（都是分钟级）：

- **B1**：修完 H1 后重跑均匀 tier capture，确认 hang 消失。**带 `TORCH_LOGS=recompiles`**
  —— 这是区分 §7 三个 hang 候选根因的最短路径。
- **B3**：expanded 布局下 paged-MQA-logits 的吞吐（展平路径每请求重读 6 次 KV 而非
  2 次，在关键路径上，必须实测）。

### 仍然不做的

- D1 vs D2 的最终选型：等 B2 出数再定。工作区已沿 D2 走了一半（K4/K9 已实现且设计
  正确 —— 可选参数、非 ragged 路径 bit-identical、TORCH_CHECK 替换而非放宽），
  **这部分不要丢**，它对任何 ragged 方案都是必需的，对 D3 也无害（不传参数即退化）。
- STS 校准表与 SPS cost table 的产出流程（goal doc G8 / U4 / U5）—— 见 §8 交付物第 5 条。

---

## 4. 编译

**不要手工敲 `build_wheel.py`。**用仓库里现成的脚本：

```bash
/lustre/fsw/coreai_comparch_trtllm/laliao/repos/trtllm-dsv4/TensorRT-LLM/tmp/compile.sh
```

它做的事（`tmp/compile.slurm`）：

- `sbatch --partition=batch --nodes=1`，`--time=4:00:00`，
  `--account=coreai_comparch_trtllm`
- 容器镜像从 `jenkins/current_image_tags.properties` 的 `LLM_DOCKER_IMAGE` 读，
  URL 里的 `urm.nvidia.com/` 被换成 `urm.nvidia.com#` 供 enroot 使用
- mount：`/lustre:/lustre` + repo → `/code/tensorrt_llm`，workdir `/code/tensorrt_llm`
- 构建目录按 **git short hash** 命名：`/lustre/fsw/coreai_comparch_trtllm/laliao/build/<HASH>/`
- 实际命令：`git clean -fdx cpp/` → 删 `tensorrt_llm/flash_mla` →
  `python scripts/build_wheel.py --clean -G Ninja -a '103-real;'` → `pip install`
- 产物：`build/<HASH>/trtllm.sqsh`（`--container-save`）+ `build/<HASH>/tensorrt_llm-*.whl`
- 日志：`tmp/logs/compile_<jobid>.{log,err}`

**注意事项：**

1. **`-a '103-real;'` 只编 SM103（B300）。** 如果你拿到的是 B200（SM100），必须改成
   `'100-real;'` 或 `'100-real;103-real;'`，否则 kernel 加载会失败。
2. **构建目录以 git hash 为键，并且会先 `rm -r` 掉旧的。** 两个同 hash 的 compile
   job 并发跑会互相清目录 —— 见 §5 的竞态提交规则。
3. Python-only 改动（本任务 P0 清单里除 K5/K6/K13 的 C++ assert 外**全部是 Python**）
   不需要重编。直接用已有的 `trtllm.sqsh` + mount repo 到 `/code/tensorrt_llm` 即可，
   因为 `PYTHONPATH=/code/tensorrt_llm` 会让改动立即生效（见 `tmp/serve_*.slurm` 的写法）。
   **先判断要不要编，再编。**
4. 跑测试时用 `CONT="/lustre/fsw/coreai_comparch_trtllm/laliao/build/<HASH>/trtllm.sqsh"`，
   参照 `tmp/serve_dsv4_bs8_relaxed_constraint.slurm` 的 srun 写法
   （`--mpi=pmix -N 1 --ntasks=8 --ntasks-per-node=8`）。

---

## 5. 申请节点：所有账号一起拍，没排到的取消

### 5.1 集群事实（实测）

| 项 | 值 |
|---|---|
| 集群 | `bia`（login-bia02.bia.clusters.nvidia.com），Slurm 25.11.2 |
| 节点 | 128 × B300，`bia0001-0128`，x86_64，2 socket × 64 core（256 线程），~2.0 TiB 内存 |
| **每节点 8 GPU** | 由 `tmp/serve_*.slurm` 的 `--ntasks-per-node=8` 佐证；Slurm **未配 GRES**，不能按卡申请，`OverSubscribe=EXCLUSIVE` 整节点独占 |
| 分区 | `batch`（默认，默认 2h / 最大 **5h**，PriorityTier 2）<br>`backfill`（默认 2h / 最大 **8h**，PriorityTier 1，**`PreemptMode=CANCEL` 会被直接取消**，GraceTime 600s） |
| 你的账号 | `coreai_comparch_trtllm`、`coreai_comparch_aarwlt` |
| 配额 | 单用户 **104 节点**（QOS `MaxTRESPU`），最多 1000 个排队作业 |
| 当前负载 | 112/128 alloc、0 idle、303 个 pending —— **要排队** |

DEP8 需要 8 GPU，**1 个节点即可**。

### 5.2 竞态提交 → 先跑的赢，其余取消

四个组合全拍上：`{coreai_comparch_trtllm, coreai_comparch_aarwlt} × {batch, backfill}`。
`sbatch` 命令行的 `--account` / `--partition` **会覆盖**脚本内的 `#SBATCH` 指令。

**推荐做法：竞态的是「分配」，不是「负载」。** 先抢一个空 allocation，抢到之后再往
里塞真正的活。这样不存在两个 job 同时开跑把对方的 build 目录清掉的问题。

```bash
#!/bin/bash
# race_alloc.sh —— 抢一个 1 节点 allocation，先到先得，其余取消
set -uo pipefail

ACCOUNTS=(coreai_comparch_trtllm coreai_comparch_aarwlt)
PARTITIONS=(batch backfill)
TIME_LIMIT=5:00:00          # batch 上限 5h；backfill 可以到 8h
LOGDIR=/lustre/fsw/coreai_comparch_trtllm/laliao/repos/trtllm-dsv4/TensorRT-LLM/tmp/logs
mkdir -p "$LOGDIR"

JOBS=()
for A in "${ACCOUNTS[@]}"; do
  for P in "${PARTITIONS[@]}"; do
    T="$TIME_LIMIT"
    [[ "$P" == backfill ]] && T=8:00:00
    JID=$(sbatch --parsable \
        --account="$A" --partition="$P" \
        --nodes=1 --exclusive --time="$T" \
        --job-name="race-${A}-${P}" \
        --output="$LOGDIR/race_%j.log" --error="$LOGDIR/race_%j.err" \
        --wrap="sleep infinity")
    echo "submitted $JID  account=$A partition=$P time=$T"
    JOBS+=("$JID")
  done
done

# 轮询，谁先 RUNNING 谁赢
WINNER=""
while [[ -z "$WINNER" ]]; do
  for J in "${JOBS[@]}"; do
    ST=$(squeue -h -j "$J" -o "%T" 2>/dev/null)
    if [[ "$ST" == RUNNING ]]; then WINNER="$J"; break; fi
  done
  [[ -z "$WINNER" ]] && sleep 10
done

NODE=$(squeue -h -j "$WINNER" -o "%N")
echo "WINNER=$WINNER on $NODE"

# 只取消这批里没排到的，不要动别人/别的作业
for J in "${JOBS[@]}"; do
  [[ "$J" == "$WINNER" ]] && continue
  scancel "$J" && echo "cancelled $J"
done

echo "export WINNER_JOBID=$WINNER"
```

拿到 allocation 之后，把活塞进去（`--overlap` 是必须的，因为 `sleep infinity` 那个
step 已经占着资源）：

```bash
srun --jobid=$WINNER_JOBID --overlap \
     --container-image="$CONT" \
     --container-mounts="/lustre:/lustre,$REPO:/code/tensorrt_llm" \
     --container-workdir=/code/tensorrt_llm \
     -N1 --ntasks=1 --pty bash          # 交互式排查

# 或者跑测试（8 卡）
srun --jobid=$WINNER_JOBID --overlap --mpi=pmix \
     --container-image="$CONT" \
     --container-mounts="/lustre:/lustre,$REPO:/code/tensorrt_llm" \
     --container-workdir=/code/tensorrt_llm \
     -N1 --ntasks=8 --ntasks-per-node=8 \
     bash -c "..."
```

活干完后**记得 `scancel $WINNER_JOBID`**，`sleep infinity` 不会自己退。

### 5.3 直接竞态提交真实负载（更简单，但有坑）

如果你就想直接把 `compile.sh` 那种脚本拍四份：

```bash
for A in coreai_comparch_trtllm coreai_comparch_aarwlt; do
  for P in batch backfill; do
    sbatch --parsable --account="$A" --partition="$P" \
           --job-name="compile-${A}-${P}" tmp/compile.slurm
  done
done
```

⚠️ **`compile.slurm` 的 `BUILD_DIR` 只按 git hash 命名，且开头就 `rm -r`。** 四个 job
在同一 commit 上跑，谁先起来谁就把别人的目录删了。所以：

- 轮询间隔要短（≤5s），**一发现有 job 离开 PENDING 就立刻取消其余的**
- 如果不幸有两个同时起来了，取消其一之后**必须重跑赢家**，不能假定它的 build 是完整的
- 更稳的做法就是 §5.2：竞态 allocation，然后只在赢家里跑一次编译

### 5.4 收尾

```bash
squeue -u $USER                       # 看还剩什么
scancel <jobid> [<jobid> ...]         # 精确取消
```

**不要用 `scancel -u $USER`** 无差别清场 —— 可能会杀掉正在跑的别的活。

---

## 6. 验收（goal doc §6）

> **所有测试都必须在 Slurm 分配的 GPU 节点上跑。login node 不可用于任何 GPU 测试。**

**必须做的，不只是 GSM8K：**

- **G3 GSM8K**：`score >= 96.0`。但 goal doc §6.1 明确说这个门**不够** —— 统计阈值
  允许 42/1319 题翻转，而 `>=96.0` 对实测 96.475 只有 6.3 题余量，既太松又太紧。
  **且当前没有活的 baseline 可比**（兄弟测试 `test_gsm8k_dep8_megamoe_deepgemm` 已被
  `waives.txt:184` 以 nvbugs/6506920 跳过）。
- **G2 逐 token 等价（这才是真正的门）**：temperature=0 下，`static` 与开启调度后
  逐 token 输出一致。贪婪投机解码定义上 lossless，任何分歧都是 bug。
  ⚠️ 诚实的形式：跨不同 graph 形状的严格 bitwise 相等 TRT-LLM 并不保证（MoE/attention
  规约顺序会变），所以允许极少数 prompt 分歧并设硬上限。
- **A2 `cap-accept` 差分（ragged 成为第一优先级后，这条从可选变成强烈建议）**：
  实现 SGLang 的三模式 `static` / `cap-accept` / `compact`。`cap-accept` 跑**均匀
  kernel 路径**（`next_n=6`）但只提交每请求窗口内的 token，输出应与 `compact`
  **逐 token 相同**。于是：
  - `cap-accept` ≠ `compact` ⇒ **必然是 ragged kernel bug**
  - `static` ≠ `cap-accept` ⇒ **必然是调度/接受逻辑 bug**

  这是把「GSM8K 掉分了但不知道为什么」变成可定位诊断的唯一手段。
  `dspark_observability.py` 已有 mode 枚举，`cap-accept` 目前显式 `NotImplementedError`
  （`dspark.py:471-478`）。
- **A3 布局一致性断言**（一条同时抓 H0/H5/H8）：
  ```
  sum(1 + py_verify_len over padded generation_requests)
    == attn_metadata.num_tokens - num_ctx_tokens
    == spec_metadata.total_verify_tokens
    == key[bucket]
  ```
- **A4 时序断言**（抓 H2/H4）：断言 `attn_metadata.ragged_verify_lens` 在 `prepare()`
  那一刻是**本步**的（而非上一步）。
- **A6 调度确实生效**（抓静默退化）：断言 `planner.stats` 的 `fallback_flat_cost` /
  `fallback_no_snapshot` / `fallback_no_confidence` / `fallback_short_snapshot`
  （`dspark_verify.py:86-93`）显示 trim 路径被走到。
  **没有 profiled SPS cost table 时 `_decide_local` 无条件返回 `max_tier`
  （`dspark_verify.py:268-270`），feature 在构造上无法 trim** —— 一个看起来完全正常的
  run 收益可能是零。
- **G4 CUDA graph**：稳态 decode 100% 命中 replay；capture 期无 D2H sync、无新分配；
  graph key 与实际 token layout 逐字段一致。
- **G5 ADP 一致性**：goal doc §4.4 要求 **ADP 资格门必须同时比较 bucket**（现在只比
  `is_all_gen_only` 和 batch size，从不比 token 数）。
- **G6 零回归**：两个开关关闭时与 `main` bit-identical。
- **单测**：`tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_sparse_mla.py`
  按每请求 q 长度参数化 —— goal doc §6.3 称这是**性价比最高的一个测试**，单卡分钟级
  就能抓住 tier-3 那类 hang（该文件 `:415`/`:473` 现在硬编码
  `seq_len_q = fused_q.shape[0] // num_requests`，那正是被测假设本身）。

集成测试命令（无需 `mpirun`，单 pytest 进程经 mpi4py.futures 起 8 worker）：

```bash
export LLM_MODELS_ROOT=/lustre/fsw/coreai_comparch_trtllm/laliao/llm-models
python -m pytest "tests/integration/defs/accuracy/test_llm_api_pytorch.py::\
TestDeepSeekV4ProDSpark::test_gsm8k_dep8_megamoe_deepgemm_confidence_scheduling" -vs
```

**验收必须覆盖至少两个并发点**（goal doc §6.5）：调度只在高并发下才真的 trim，单一
低并发 run 是空跑。vLLM #47808 报的是并发 16 和 64。

---

## 7. 已知 hang（必须区分三个候选根因，不要认定其中之一）

唯一一次端到端尝试失败在 tier-3 的 CUDA graph capture：

```
capturing 102 graphs (34 batch sizes x 3 draft-length tiers [1, 3, 5])
warmup batch=128, draft_len=5   ← 完成
warmup batch=128, draft_len=3   ← never returns   TEST_EXIT=124
```

三个**独立**候选（goal doc §7.1 U2）：

1. **H1 stride 不匹配** —— tier 5 时 768==768 通过，tier 3 时 schedule 描述 768 行而
   logits 只有 512 行。完整自洽。
2. **capture 期 attention workspace 重分配** —— hang 前紧邻的日志是
   `Attention workspace ... increase from 0 bytes to 168493568 bytes`；
   `model_engine.py:1789-1806` 明确记录这是真实 hazard，且 Case 2b **没有** Case 2 的
   `max_spec_graph` 规避（`:1810-1813`）。
3. **torch.compile / max-autotune 在 capture 区内**（goal doc §4.5）——
   `deepseek_v4.py` 三个 `@maybe_compile(dynamic=True, max-autotune)` 函数经
   `on_update_kv_lens` 在 `CUDAGraphRunner.capture` 录制区内被调用；
   `_compute_gen_compressed_position_ids` 把 `num_gen_tokens_per_seq` 当 Python int 用
   （`:979-983`）构成 guard → **每个 tier 一次 Inductor 编译，发生在 graph capture
   期间**，而 max-autotune 需要同步 benchmark。

**先修 H1 再重跑，并带 `TORCH_LOGS=recompiles`** —— 这是区分三者的最短路径。

goal doc 引用的 `.dspark-logs/gsm8k.log` 和 `hang-stack-299685.txt` 不在本 repo 里，
是写文档那位的工作区产物。不要去找，重新复现即可。

---

## 8. 交付物

0. **B2 的实测数字**（最先交，可能改变后续全部计划）：`(batch=B, s_q=tier+1)` vs
   `(batch=B*(tier+1), s_q=1)` 的正确性与吞吐对比。若 token-major 不可行，立刻报告
   并改推 D3 Stage A。
1. 一个基于 `pr-17056` 的分支，表一 + 表二逐项修完，每项 commit message 里引用
   goal doc 的编号（H0/H1/.../K8）
2. A3/A4/A6 断言 + 参数化的 `test_deepseek_v4_sparse_mla.py` 单测；compressor ragged
   差分单测（`new_tokens_per_seq` 全填 `next_n` 必须与传 `None` bitwise 相同 —— goal
   doc §6.3 标为「欠缺」）
3. GSM8K 在 **overlap scheduler + DEP8 + ragged** 下的实测分数，**以及 planner.stats
   的 fallback 计数**（证明调度真的生效，而不是静默退化成 max_tier）；均匀 tier 路径
   的同口径分数作为对照
4. 一份结论：hang 的真实根因是 §7 三个候选中的哪个（附证据）
5. **对 goal doc §8 剩余的待拍板问题给出你的推荐**（Q2 和 Q8 已关闭，见 §2.5），尤其：
   - **Q4**：`cap-accept` 模式要不要实现。**建议实现** —— ragged 成为第一优先级之后
     它的价值更高了：`cap-accept` vs `compact` 的差异 ⇒ 必然是 ragged kernel bug；
     `static` vs `cap-accept` 的差异 ⇒ 必然是调度/接受逻辑 bug。把 pass/fail 硬币变成
     可定位诊断。成本不高（复用已有的 `torch.minimum(..., verify_lens)` clamp，
     `interface.py:1473-1480`、`:1889-1896`）。当前是显式 `NotImplementedError`
     （`dspark.py:471-478`）
   - **Q5**：STS 表和 SPS cost table 由谁、按什么流程产出、存哪（checkpoint 内？
     repo `examples/configs/`？）。**在它们存在之前，任何性能结论都是空的，而且
     feature 在构造上无法 trim**
   - **Q6**：tier ladder 要不要按 batch size 门控（102 → ~54 个 graph，换回
     ~0.4–0.8 GB/rank 的 KV）
   - **Q7**：非贪婪采样的处理时机 —— 见表二 H7
   - **Q9**：要不要采用 DeepGEMM 的 `indices` varlen 参数（goal doc §7.3）。**ragged
     成为第一优先级之后这条值得重新评估**：vLLM #47808 用的正是它，能去掉
     `[num_tokens, max_blocks_per_seq]` 那个大 block table，代价是把该路径钉死在 SM100
     （DeepGEMM varlen 分支断言 `arch_major == 10 and next_n == 1`）

---

## 9. 工作方式

- 每个改动都要能指出它对应 goal doc 的哪一条，以及 `file:line`
- 凡是没读码确认的推断，显式标注「待验证」—— 沿用 goal doc 的规范
- **不要静默降级**。goal doc 反复强调：不支持的组合要显式 config-reject，静默降级是
  最差的选项
- 遇到与 goal doc 冲突的发现，先给证据再改结论，不要默默按自己的理解走
- 提交遵循仓库 `AGENTS.md`：`git commit -s`（DCO），PR 标题格式
  `[JIRA/NVBUG/None][type] description`，不要加 AI co-author

---

## 10. P0 执行结果（2026-08-02 收尾）

本节由执行 agent 追加，记录实际跑出来的结论。**与前文冲突处以本节为准**，因为前文
是任务下达时的认知，本节是实测。

> **硬件与配置口径:全部实测都在 x86 + NVIDIA B300 SXM6 AC(275040 MiB / 卡,
> 驱动 580.167.08)、8 卡单节点、DEP8(TP8 + EP8 + attention-DP)上取得。MoE
> backend 为 MEGAMOE_DEEPGEMM 或 TRTLLM,`max_draft_len=5`,tier 阶梯 `{1,3,5}`。
>
> **换到 GB300(或任何 Grace-Blackwell 一体机)结论可能不同**,以下几组明确依赖
> 硬件:
>
> * **显存(10.4)**:「ragged 的 CUDA graph 开销 ≈0」是在 267 GiB HBM、
>   `free_gpu_memory_fraction=0.5` 下测的。GB300 的显存容量、NVLink 拓扑和
>   C2C 一致性内存会改变 graph 池与 KV 的相对占比。
> * **裁剪是否划算(10.3)**:planner 的 argmax 比较「每秒接受 token 数」,分母
>   是实测的 `T(bs, M)`。不同的算力/带宽比会改变 `theta(M)` 的斜率,从而移动
>   「裁剪开始划算」的接受率阈值。本文结论是「p≈0.90 时任何批次档位都不该裁」,
>   **这个阈值不可跨硬件搬运**。
> * **SPS cost table 必须在目标硬件上重采**。`sps_real_final.json` 只对 B300
>   有效,在目标机上重跑 `dspark_sps_profiler` 即可。
> * **每 rank 批次 = 并发 / DP 数**,本文按 DP=8 计算。
>
> 与硬件**无关**的部分:10.5 的机制验证(每请求不同 verify len 能否正确运行)、
> 10.6 的六个 bug 及其修复(都是 host 侧逻辑错误)、10.7 的方法论。这些是正确性
> 问题,不是性能问题。


### 10.1 核心结论：K1/K2 已解除，纯 Python，不需要编译

ragged 原本卡在 `mla_rope_generation`（`mlaKernels.cu:1161`）和 FMHA dispatch
（`thop/attentionOp.cpp:920`）都要求 `acc_q_len % batch_size == 0`——per-request
窗口从构造上违反它。解法是把 generation 半边按 **token-major** 呈现：每个
generation token 作为一行长度为 1 的序列，`seq_len == 1` 整除任何行数。

`git diff pr-17056..HEAD -- cpp/` **为空**。op 的 `mMaxNumRequests` 只是 JIT
warmup 提示，per-step 请求数来自 batch，`getFmhaMultiCtasKvScratchSize` 按 SM 数
而非 batch 算——从 Python 侧传入静态行数上限即可。

### 10.2 精度（G3/G4/G5）

DEP8 + attention-DP + overlap scheduler + CUDA graph，forced ragged 窗口：

| MoE backend | exact_match (flexible) | exact_match (strict) |
|---|---|---|
| TRTLLM | 96.4367 ± 0.5106 | 96.5125 ± 0.5053 |
| MEGAMOE_DEEPGEMM | 断言通过（分数未捕获，该轮早于 `-s` 开关） |

uniform 基线 96.2092。`steps_ragged 81/129`，5 种窗口，`trim_ratio 0.23`。

### 10.3 吞吐（G6）：**这个 checkpoint 上没有收益空间**

planner 在三种工况下都正确地选择不裁剪：

| 工况 | 每 rank 批次 | accept_len | planner |
|---|---|---|---|
| GSM8K | 小 | 3.30 | 不裁 |
| 合成随机 token | 4–8 | 4.09 | 不裁 |
| 合成随机 token | **64** | 3.73（p≈0.90） | **不裁**，零 fallback |

DSpark block drafter 在此模型上每 token 接受率约 **90%**。拟合表在该接受率下
于所有批次档位都判定"验满整块最优"（判据：`bs=64, p≥0.80 → L=5`）。
**这不是功能失效，是没有机会。** 要看到收益需要接受率显著更低的模型或负载。

吞吐实测（128 请求，isl1024/osl512，并发 64，KV 泄漏修复后）：

| 配置 | tok/s |
|---|---|
| baseline（confidence head 关） | 7032.37 |
| ragged 窗口=5（不裁剪） | 7319.47 |
| rotating（confidence 无关地裁 29%） | 5994.39 |

**唯一站得住的吞吐结论是 KV 泄漏修复值 +6.5%**（6869.73 → 7319.47，同配置）。
ragged 相对 baseline 的 +4.1% 不作为结论：单次测量，且从未表征吞吐的
run-to-run 方差。rotating 的 5994.39 **不得**当作本功能的代价——它是与
confidence 无关的盲裁，是最坏情况且方向与真实策略相反。

### 10.4 显存代价:实测约等于零

**本节的前一版是错的,已按实测重写。** 前一版声称「CUDA graph 张数 3×、
attention workspace 9×、ragged 让显存悬崖来得更早」,其中只有张数是对的,
而张数并不等于显存。

实测(同节点、同数据集、同并发,只切换 baseline / ragged 的 yaml,用
`torch.cuda.memory_snapshot()` 求 CUDA graph 私有池的**绝对值**):

| max_batch_size | uniform 图数 / 池 | ragged 图数 / 池 | 差 |
|---|---|---|---|
| 128 | 19 / **8.746 GiB** | 57 / **8.740 GiB** | **−0.006** |
| 256 | 21 / **16.527 GiB** | 63 / **16.588 GiB** | **+0.061** |
| 512 | 25 / **33.14 GiB** | 75 / — | — |

**图数确实是 3×(等于 tier 数),但显存差落在噪声里。** CUDA graph 共享内存池,
多捕的形状复用同样的缓冲区,张数没有转化成显存代价。

真正决定显存的是 `max_batch_size` 本身:128 → 256 → 512 对应
8.7 → 16.5 → 33.1 GiB,与 ragged 无关。

前一版还声称「ragged 让 bs=512 OOM」。补做对照后:同配置下 uniform 跑通,
而 ragged 那次是在 **25.10 GiB** 处挂的,**低于 uniform 自己 33.14 GiB 的峰值**
—— 一个更费显存的配置不会在更低点失败。那次的日志里有别的进程占着显存,最可能
是节点残留(本次实验中已两次遇到:一次是 relaunch 过快导致两个进程各占 134 GiB)。
一次不可复现、且与其余三个档位矛盾的观察,不足以支撑结论。

`attention workspace 68 MiB → 604 MiB` 那条同样**从未做过 uniform 对照**,
在补做之前不应引用。

**方法论**:前一版三条断言里,唯一做过对照的那条(张数)是对的,另外两条都是
单次观察 + 事后编的机制解释。而且第一次测显存用的是「捕获前后取差值」,在多轮
捕获共用分配器的情况下得出 `+13.9 / +5.5 / −0.8 / +0.7 GiB` 这种自相矛盾的
数——增量法本身不成立;换成私有池绝对值后三次独立测量复现到小数点后三位。

### 10.5 每请求不同的 verify len:已验证

这是 P0 的核心能力,也是 K1/K2 那个原始 blocker 要解决的东西。**已由 planner
自主产出,无任何外部强制**(`TLLM_DSPARK_FORCE_VERIFY_LENS` 已从代码中删除):

```
steps_ragged: 211/318          同一步内窗口不相同的步数
distinct_verify_lens: 4        {1: 27375, 2: 5831, 3: 12, 5: 66}
trim_ratio: 0.6064             裁掉 60.6% 的 token 上限
KV 溢出: 0    OOM: 0
```

分配链路是真实的:confidence head → 每请求 survival 概率 → 成本表算出预算 →
top-k 按 survival 高低分配窗口。置信度高的拿长窗口,容易早被拒的拿短窗口。

**这证明了机制,不证明策略有收益。** 该轮用的是刻意造陡的合成成本表
(`tests/microbenchmarks/dspark_make_steep_sps_table.py`,输出带 `SYNTHETIC: true`),
planner 在对着虚构的成本曲线优化,所以那轮的吞吐数字无意义。真实表下 planner 在
所有批次档位都选满窗口(见 10.3)。

**为什么必须造这张表**:此 checkpoint 接受率约 90%,planner 永远不会自己走到
裁剪路径。而下面三个 bug **只在真裁剪时可达**——没有这张表,它们会带着绿色的
测试合入。

### 10.6 修掉的真 bug

1. **KV 泄漏(ragged 独有)**。预留侧每步加 `1 + max(draft_len, reserve)`,
   回退侧只退 `py_rewind_len = verified_len − accepted`;`kv_cache_manager_v2`
   把补偿项 gate 在 `if self.is_draft` 后面,于是 target manager 每步漏
   `draft_len − verified_len`,直到序列需要的 block 超过 `max_seq_len`,报
   `User-provided base page indices is too short`。uniform 下该项恒为 0。
   **定位方式**:窗口=5 vs 窗口轮转、其余全同的一次对照,而非读代码。

2. **token-major block table 布局**。展开表用了 `[rows, max_blocks]`,而 op 读
   `[num_pools, num_seqs, 2, max_blocks]`,且 gather 折掉了 `num_seqs` 轴 →
   8 卡 device-side OOB。

3. **`host_request_types` / `prompt_lens_cuda` 未行化**。op 按 `num_seqs`(=896)
   索引 128 长的数组。**主动审计 12 个 per-sequence 参数找出来的,不是等它崩。**

4. **bucket fit 假设 padding 必然发生**。fit 按 padded_bs 推导 bucket 网格并把
   余量分给尚不存在的 pad 行,但 `_get_padded_batch` 可以拒绝 padding。拒绝时
   网格对应的行数不存在,填出的总数落在网格外:`graph_miss_keys
   {'(193, 5, False, False, True, 449)': 16}`,捕获档位有 192 和 256 却没有 193。
   修法:fit 前先问 `runner.will_pad_to`,不能 pad 就退回 uniform。

5. **跨 rank bucket 不一致**。bucket 被独立推导两次:fit 从 allgather 的
   `peer_stats` 算,graph key 又遍历 `generation_requests`(padding 追加**之后**)
   重新求和。两条独立推导却要求跨 rank 精确相等,一致性是碰巧的。实测
   `peer_shape_mismatch` 不裁剪时 2/265 步、裁剪时 16/318 步,每次让 8 个 rank
   一起掉出 replay。修法:fit 发布商定值,key 直接读——按构造必然一致。

6. **lm-eval 分数选取**(与 DSpark 无关的既有 bug)。把 `samples`(1319×100)
   混进平均,报出 **44030.98 并通过了 96.0 的门槛**。按 `metric,filter` 键结构修正。

### 10.7 踩过的坑(值得写进流程)

**探针失效比功能失效更常见。** 本次至少七次「失败」中,探针本身有问题的占多数:

| 现象 | 真因 |
|---|---|
| `graph_miss_shapes` 为空 | 模块版本偏斜,默认参数把「没记录」伪装成「没发生」 |
| sync 探针零警告 | 探针根本没 arm,我差点报通过 |
| gate 在 ramp 期误杀三轮实验 | floor 数的是总步数,而并发从 0 爬升 |
| 显存增量 `+13.9/+5.5/−0.8/+0.7` | 多轮捕获共用分配器,增量法不成立 |
| 造表后 ragged 仍不亮(两次) | 固定开销压倒可裁剪项;网格覆盖不到工作区间 |
| 预览说会裁、实跑不裁 | 我重推了 planner 的 argmax,与真实实现不符 |

由此得到三条:

* **沉默不等于通过。** 任何「没有告警」的结论,必须先证明探针确实生效。
* **能调用就别重推。** 造表预览改成直接 `import` 真实的
  `budget_argmax_over_uniform_lens` 后一次就对,并能回测历史结果;测试里也用真的
  `DSparkScheduleConfig` 而非 mock。
* **夹具必须能自检。** 造表工具现在会预测 planner 的选择,并在「任何档位都不裁」
  时直接 WARNING——失败在生成的那一秒暴露,而不是 20 分钟后。

**没有对照的归因一律不成立。** 本次有四条结论因补对照而推翻或修正:

| 断言 | 结局 |
|---|---|
| ragged graph 显存 3× | 实测 ≈0(8.746 vs 8.740 GiB) |
| ragged 让 bs=512 OOM | 不开 ragged 同样配置也曾跑通;且 ragged 在更低的 25.10 GiB 就挂,与「更费显存」矛盾 |
| `peer_shape_mismatch` 是 bug | 是那道门在正常拦截(但推导重复确实该修) |
| G2 token 等价性失败 | 两次**相同** static run 之间也 15/16 发散——噪声底 = 信号,该测试在此部署上问不出东西 |

**局部证据不能支撑全局断言。** 我曾报「ragged 未引入新的 host-device sync」,
依据是审计了自己新增的 token-major 代码(那部分确实干净),但动态探针在
`dspark.py:117/516/551` 和 `deepseek_v4.py:891` 抓到 32 次真同步。静态审计只能
覆盖「我写的代码」,动态探针才覆盖「这条路径实际执行的一切」。

**在功能没满负荷运行时做的验证,适用范围比看起来窄。** 上面 4 和 5 两个 bug 都
只在真裁剪时可达;我此前核对过「两条 padded_bs 规则一致」,但那是在不裁剪的前提
下做的,总数恒定完全掩盖了行数这一维。

### 10.8 未完成

* **动态 sync 证明**(唯一未定论项)。ragged 侧已测得 arm 后 32 次真同步,位置见
  上;**对照(confidence head 关、其余逐字相同)尚未跑完**,因此「这些是 ragged
  引入的还是 DSpark 既有的」暂无结论。在此之前不应引用任何 sync 相关结论。
* **`peer_not_gen_only` 类 graph miss**:32/318 步,属正常连续批处理(uniform
  路径同样存在),非缺陷,但可作为后续调度优化的方向。
* **吞吐收益需要更低接受率的模型或负载**。这是 checkpoint 的属性,非本 PR 可解。

### 10.9 删除均匀 tier 阶梯:三状态收敛成两状态

调度原本有三种状态,中间那种不值得保留:

| 配置 | 语义 | 结局 |
|---|---|---|
| `scheduling=False` | 验证整个 drafted block | **保留**,唯一兜底 |
| `scheduling=True, ragged=False` | 整批取同一个 verify 长度(tier `{1,3,5}` 里选) | **已删除** |
| `scheduling=True, ragged=True` | 每请求各自的窗口 | **保留** |

删除理由有两条,都是实测出来的:

1. **它做不到调度的本意**。整批一个长度,就无法把长窗口给高置信请求、短窗口给
   坍塌的请求——而那正是 confidence head 存在的理由。
2. **它在本 checkpoint 上恒等于「验证整块」**。阶梯只有 `{1,3,5}` 三档,接受率
   ~90% 时 argmax 永远落在 5,即整块。也就是说:一条独立代码路径(进 graph key、
   进跨 rank allgather、进 planner 的 fallback 计数),运行效果和 `scheduling=False`
   完全一样。

SGLang 也没有对应物:它的 scheduler 一律经 top-k 产出每请求窗口,"均匀"只作为
verify-all 的退化情形出现。

删掉的东西:`decide_draft_len` 及其 `_decide_local`、`snap_to_tier`(唯一调用者
就是 uniform 分支)、`py_executor` 里的 uniform 分支、跨 rank allgather 里的 tier
字段(5→4 个 int)、以及覆盖该路径的 12 个测试和一个 rank 模拟 helper。共
−350 行。兜底现在直接是 `runtime_draft_len = planner.max_tier`,不需要跨 rank
协商,所以这一步也少了一次归约。

`llm_args` 现在**直接拒绝** `scheduling=True + ragged=False`,而不是把它悄悄当成
别的东西。关掉特性用 `enable_confidence_scheduling=False`。

> **历史数据是否受影响**:§10.2 的 uniform 基线 96.2092 仍然有效,但理由不是
> 「它没用中间态」——我没有回去核对那一轮的 flag(§2 表一把中间里程碑定义成
> `scheduling=True + ragged=False`,所以有可能就是它)。理由是 §10.3 已实测
> planner 在三种工况下**一次都没裁剪**:中间态在本 checkpoint 上必然选到 tier 5,
> 即整块。因此无论那一轮用的是哪套 flag,96.2092 描述的都是「验证整块」,而那条
> 路径原封不动地留着。
>
> §10.8 的 sync 对照写的是「confidence head 关」= `scheduling=False`,配置依然合法,
> 不需要重新设计。

### 10.10 实现 `cap-accept`(A2 / Q4 收口)

三模式现已全部可用,`dspark.py` 里的 `NotImplementedError` 已移除。

`cap-accept` 算出每请求窗口并**按窗口提交**,但**不缩小交给 target 的 token 轴**。
它把 `compact` 混在一起的两件事拆开:

    cap-accept 输出 ≠ static 输出   ⇒ 调度策略有问题
    compact 输出  ≠ cap-accept 输出 ⇒ 布局压缩有问题

**实现是纯增量的**:窗口写在独立属性 `py_verify_cap` 上,**不碰 `py_verify_len`**。
于是布局侧(`model_engine` / `cuda_graph_runner` / `dsa.py`)完全看不见窗口、自动
走整块;`_verified_len` 因为 `py_verify_len is None` 自动返回整块,**KV rewind 随之
正确**(`rewind = 整块 − capped`)。这样就不必在「为 compact 调了三个 bug 才对」的
记账代码里再加一组条件分支——§10.6 那个 KV leak 正是这类记账错误。

**截断必须在设备端 acceptance 内部**(`interface.apply_accept_caps`)。第一版写在主机侧
`update_requests`,是错的——这条值得单独记:

`num_accepted_tokens` 在**同一次 forward 的后续、设备上还要被消费**。`dspark.py:606-652`
里 drafter 用它挑下一块的条件隐状态、回填滚动 KV 窗口、并推进持久解码位置
(`_ctx_len += nacc`)。主机同步后再截断,设备已按未截断的数推进过 drafter 状态,而主机
只提交 capped 前缀——输出因验证权威仍无损,但 drafter 状态永久漂移,这个模式也就不再
是它存在意义所在的可信参照。SGLang 截在同一位置(`dspark_accept.py:138`)。

**且不能借 `is_ragged_verify` 触发已有的 clamp**:那个标志同时意味着「draft 缓冲已按
窗口打包」(`_padded_gen_draft_tokens` 切 `draft_tokens[:total−num_gens]`)。cap-accept
的缓冲是普通矩形,借用会把矩形当 ragged 解包,一个请求的 draft 记到另一个头上,**全程
不报错**。所以新增独立字段 `SpecMetadata.accept_caps`。

**它产出 `compact` 结构上拿不到的数**:`cap_trim_tokens` = 被丢弃的、本可接受的位置
数,即裁剪的真实接受代价。`compact` 下那些位置根本没算,SGLang 为此专门养了一个
~600 行的统计估计器;我们此前只有 `trim_regret_rate` 这个下界。

两个埋在计数器里的坑,都已处理并有测试:

1. **不能给它记上算力节省**。`record_step` 会按 `sum(1+v)` 推算 delivered,那会让
   cap-accept 显示出一个它刻意不去拿的 trim_ratio——而那正是判断功能是否生效的
   头号指标。新增显式 `delivered` 参数,由调用方传整块。
2. **`assert_ragged_active(require_trim=True)` 会误杀它**。它的 `trim_ratio` 恒为 0
   是设计如此。该判据现在对不裁 token 轴的模式豁免,否则这道门在它唯一该跑的模式
   里恒不可满足。

代价是**不省任何算力**,所以它是诊断模式,永远不是服务配置。

损失**逐请求**记录,不是只记总量:总量分不出「每个请求丢一点」和「少数请求被砍烂」,
两者 `cap_trim_tokens` 相同而结论相反。因此另记 `requests_cap_trimmed` /
`cap_trim_max` / `cap_trim_hist` / `cap_trim_concentration`。链路是
`apply_accept_caps` 写持久 `[bs]` 缓冲 → worker 返回 `cap_trim_lens` → 采样器 store →
随**已有的** host 拷贝回来,**不额外引入同步**。

这里的真风险是**陈旧缓冲**:缓冲持久且按 slot 索引,不产出 caps 的步必须往本批次
slot **写零**,否则 slot 回收给新请求后会报上一个占用者的损失——今晚已经踩过两次的
同一类。`_process_outputs` 显式 `zeros_like`,有测试卡住。

覆盖:`tests/unittest/_torch/speculative/hw_agnostic/test_dspark_cap_accept.py`
(63 项,含 clamp 的全枚举不变式、context slot 清零、同一总量的两种分布可区分、
三个计数器坑的回归)。dspark 全套 **335 passed**。**尚未 e2e 跑过。**
