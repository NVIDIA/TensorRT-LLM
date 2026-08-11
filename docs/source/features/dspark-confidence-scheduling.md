# DSpark 置信度调度验证（Confidence-Scheduled Verification）

DSpark 每个 decode step 都会 draft 出一整个 block，并把它全部送给 target 验证。
它的 draft 网络本来就带了一个 **confidence head**，为每个 draft 位置打分，预测该位置
能否活过验证。置信度调度把这个 head 接进了验证调度器：block 依然完整 draft，但**真正
送去验证的位置数**由这些分数决定。

本文讲设计。用户可配置项见[配置](#配置)；DSpark 在各投机算法中的位置见
[Speculative Decoding](./speculative-decoding.md)。

## 目录

- [问题与核心洞察](#问题与核心洞察)
- [三条设计不变量](#三条设计不变量)
- [模块分层](#模块分层)
- [一个 decode step 的数据流](#一个-decode-step-的数据流)
- [relay 滞后](#relay-滞后)
- [算法](#算法)
- [一个完整的数值例子](#一个完整的数值例子)
- [Ragged 验证](#ragged-验证)
- [CUDA graph 处理](#cuda-graph-处理)
- [跨 rank 一致性](#跨-rank-一致性)
- [退化路径](#退化路径)
- [配置](#配置)
- [性能天花板与已知局限](#性能天花板与已知局限)
- [部署要求：DSv4 必须显式设 max_seq_len](#部署要求dsv4-必须显式设-max_seq_len否则并发被钉在-maxbs2)
- [基准配置（可复现）](#基准配置可复现)

## 问题与核心洞察

基线行为：

```text
draft:  一次并行 backbone forward -> [t1 t2 t3 t4 t5 t6 t7]   (K = 7，固定成本)
target: 验证全部 7 个 + 1 个 bonus = 8 个 token 位置
```

如果某个请求的 draft 在位置 3 就已经断了，那么花在位置 4-7 上的验证算力全是浪费。
confidence head 预测的是每个位置的**条件**接受概率 `P(accept_k | accept_1..k-1)`，
因此前缀存活概率就是累积乘积：

```text
survival[r][j] = prod_{i <= j} conf[r][i]     # 位置 j 被"走到 且 被接受"的概率
```

由于 `conf` 是 sigmoid 输出、落在 `(0, 1)` 内，**survival 沿位置严格非增**。
这条单调性是后面整个分配算法的地基。

## 三条设计不变量

下面所有取舍都可以从这三条推出来：

| # | 不变量 | 推论 |
|---|--------|------|
| **I1** | block 永远完整 draft | draft 是一次并行 forward，砍长度省不了钱。只有送给 target 的 token 数可裁 |
| **I2** | 接受规则完全不动 | 调度只决定**验多少**，绝不决定**接不接受**，因此**输出分布不变** |
| **I3** | 决策在 host 上做，且必须落在已捕获的 graph 上 | CUDA graph key 由 host 值构成；未捕获的 draft 长度不会报错，而是静默掉进 eager 执行 |

**I2 是最重要的性质**：这不是精度/吞吐的 tradeoff，而是纯粹的算力回收。所以这个特性的
精度测试断言的是与 baseline **完全相同**的分数 —— 任何偏差都是不变量被破坏，而不是调参结果。

## 模块分层

```text
配置层
  llmapi/llm_args.py :: DSparkDecodingConfig

DEVICE 侧（在 target 的 captured CUDA graph 内）
  _torch/models/dspark/heads.py :: DSparkConfidenceHead
      forward()    [*, K, hidden] -> [*, K] 原始 logits（fp32 matmul）
      apply_sts()  原始 logits -> 标定后的概率（逐位置温度）
  _torch/speculative/dspark.py :: DSparkWorker._draft_gen_block_batched
      _confidence_logits[slots] = conf        # slot 索引，原地写
                       |
                       |  非阻塞 D2H，滞后（见下文）
                       v
HOST 侧（决策，完全在 capture 区之外）
  _torch/speculative/dspark_verify.py :: DSparkVerifyPlanner    # 协调器、退化守门
        |
        +-- dspark_schedule.py    survival（cumprod）+ 全局 top-k 分配
        +-- dspark_planner.py     SpsCostTable + Theta = tau / T 的 argmax + tier 推导
  _torch/pyexecutor/py_executor.py :: _dspark_confidence_draft_len   # 每步 hook

批次成形层
  _torch/pyexecutor/model_engine.py       ragged 输入布局、graph 形状拟合
  _torch/pyexecutor/cuda_graph_runner.py  graph key 上的 token 轴
  _torch/speculative/dspark_ragged.py     qo_indptr 打包、bucket 对齐、scatter
  _torch/attention_backend/sparse/dsa.py  indexer 元数据按 token 展开
```

## 一个 decode step 的数据流

```text
                 HOST                          |        DEVICE（captured graph）
--------------------------------------------------------------------------------
 step t                                        |
   _dspark_confidence_draft_len()              |
     1. 读取滞后的快照                          |
        （cudaEvent.query()，非阻塞）           |
     2. 按 slot 查出每个请求对应的行             |
     3. apply_sts -> cumprod -> survival        |
     4. argmax Theta -> tiers 中的 draft_len     |
     5. 跨 rank allgather(max)                  |
                                               |
   _prepare_tp_inputs()                        |
     铺 input_ids / position_ids                |
     _attach_ragged_verify_layout()            |
                                               |
   graph_key = (bs, draft_len, ..., [bucket])  |
   ------------- replay -------------------->  |  target forward
                                               |  sample_and_accept_draft_tokens
                                               |  draft forward（完整 block K）
                                               |  confidence head
                                               |  _confidence_logits[slots] = conf   (*)
     6. stage_confidence(buffer, generations)  |
        轮转 relay ring，然后发起               |
        非阻塞 D2H              <---------------+
--------------------------------------------------------------------------------
```

## relay 滞后

调度决策跑在一个**滞后**的快照上，而且这个滞后是**钉死的常量**，不是"引擎给多少算多少"。
DSpark 规定了两步回溯（two-step-back）的 relay；SGLang 也把这个值钉死，并在其 overlap
scheduler 会给出不同值时用 ring buffer 补齐。

钉死很重要，因为 STS 标定表和 SPS 成本表是在**某一个特定滞后**下拟合出来的。如果有效滞后
会随一个无关的引擎开关漂移，那么一张 profile 好的表就会静默失效。

TensorRT LLM 这两步是怎么来的 —— 调度 hook 跑在 forward **之前**：

```text
iteration t:  [hook(t)]  ->  [prepare]  ->  [forward(t)]
                 |  读取 hook(t-1) staged 的快照
                 |  发起 D2H，流序在 forward(t-1) 之后，
                 |    所以抓到的是 forward(t-1) 的 confidence
                 v
      hook(t) 读到的是 forward(t-2) 的 confidence   ==>  引擎滞后 = 2
```

无论 overlap scheduler 开不开都成立：overlap 改变的是拷贝**何时落地**，而不是它抓的是
哪一次 forward。没落地的拷贝会退化成"验全块"，而不是退化成一个陈旧的猜测。

实现中滞后不可配置：executor loop 恒定给出两步（上图），planner 直接消费该快照。快照行的新鲜度由 `_confidence_stamp` 守卫 —— 逐 buffer 行的 draft-pass 序号，随 confidence 经同一个 `slots` scatter 在 graph 内写入；陈旧或从未打分的行读到中性 confidence，survival 归 1.0，退化为验证整块。

### slot 索引与 generation 守卫

快照按 **worker slot** 索引，绝不按 batch 位置。请求会在"写入分数的那一步"和"读取分数的
那一步"之间进出，所以位置 `i` 经常已经是另一个请求了。

slot 还会被回收，这带来两种不同的陈旧风险，由两个互补机制处理：

| 风险 | 机制 |
|------|------|
| device 上那一行还留着上一个占用者的分数 | `_assign_slot` 在把行交给新请求时重置为中性值 |
| host 侧快照里还留着回收之前的副本，device 侧的重置够不着 | 每行带一个 `_confidence_stamp`（draft-pass 序号），与 confidence 一起 stage；与当前 pass 对不上就把该行按中性处理，survival 归 1.0 |

这两者是**互补的，不是二选一**。generation 标签是在 stage 时于 host 侧快照的，而
confidence 是在 forward 时于 device 上写的 —— 所以正是那次重置让标签说的是真话：只 bump
不重置，就会把上一个占用者的分数打上新占用者的标签。反过来，只重置也清不掉快照里已经
存在的 host 副本。

两种风险都收敛到同一个安全方向：**survival = 1.0，即验证整个 block**，也就是启用该特性
之前的行为。

## 算法

### 第 1 步 —— survival

```python
survival = torch.cumprod(sigmoid(logits / sts_temperatures), dim=1)
```

标定（STS）之所以重要，是因为调度器消费的是**累积乘积**：逐位置的偏差会沿 block
**几何级放大**，未标定的 head 会系统性地高估前缀存活率。

### 第 2 步 —— 成本模型

本实现遵循已公开的形式：

```text
T(bs, K) = bias + alpha(bs) + theta(M)
           ----   ---------   --------
           固定    批次相关     verify-token 成本
                  （draft pass、  ^ 唯一可裁剪的一项
                    权重搬运）
```

`SpsCostTable` 按总验证 token 数 `M` 做**钳位线性插值**查表（对齐 SGLang 的加法表消费端）。想表达"平台"就必须把平台测出来——两个等值断点；消费端不再替硬件假设断点之间是平的（那个假设曾把 1512 token 的满块按 768 的价格计费，整个特性因此从未裁剪过）：

```text
step_time
    |        +--------      <- riser：新的 kernel wave 起来了
  9 |        |
    |  +-----+
  4 |  |                    <- shelf：这一段上加 token 是真的免费
  2 |--+
    +--+-----+-----+----> M（总 token 数）
       8    16    32
```

**token 口径**：一个请求验证 `L` 个 draft 位置，实际提交 `L + 1` 个 token —— 它手上那个
bonus token 加上它的 draft。所以 `M = bs * (L + 1)`。这个换算只在一个地方计算
（`dspark_planner.total_verify_tokens`），保证 planner 和 tier 推导不会漂移。

由于 `Theta = tau / T` 是**比值**，不可裁剪的那几项**不会约掉**。低估它们会让每个验证
token 显得比实际更贵，planner 就会过度裁剪。因此这些项是**给进来的，不是猜的**：
`fixed_overhead_ms` 承载 `bias`，可选的 `batch_sizes` / `batch_overhead_ms` 阶梯承载
`alpha(bs)`。

### 第 3 步 —— 目标函数与全局 argmax

```text
tau(L) = num_gen_requests + sum_r sum_{j<L} survival[r][j]
         ----------------   ---------------------------
         每请求一个白送的     各验证位置的期望产出
         bonus token

Theta(L) = tau(L) / T(bs, M(L))          # 期望 token / 毫秒

选择      = argmax_{L in tiers} Theta(L)
```

**取全局 argmax，不是首次下降。** 因为成本是阶梯，`Theta` 并不单峰：

```text
Theta
  |    /|      /-----    <- 贪心循环停在这里（第一个 shelf 的末尾），
  |   / |     /             把大部分收益留在了桌上
  |  /  +----/
  +---------------> L
```

### 第 4 步 —— shelf 右边缘性质

> 在一个 cost shelf 内，step time 恒定，而 `tau` 随每个新增 token 严格增，
> 所以 `Theta = tau / cost` 在 shelf 上严格递增。因此最优点**永远落在 shelf 的右边缘**，
> 绝不在其内部。

于是由各 shelf 右边缘构成的 tier 集合，**在推导时所用的那个 batch size 下**精确包含连续最优解：

```text
right_edge(breakpoint T) = (T - 1) // bs - 1        # length 空间
```

这个**零损失性质跨 batch size 不成立** —— cost shelf 活在 token 空间，所以 shelf 右边缘
在 length 空间是 `bs` 的函数。一个部署会捕获很多 batch size 但只能捕获一套 ladder，因此
应当在**稳态众数 batch size** 上推导，并**实测**其他 batch size 上的残差损失，而不是假装它不存在。

### 第 5 步 —— 逐请求分配（ragged 模式）

预算是 `num_gen * (L* - min_verify_len)` —— 和 uniform 决策**花掉的 token 总量相同**，
只是重新分配。

`min_verify_len` 在本实现中**固定为 1**（配置校验强制 >= 1）：每个请求每步至少验证
anchor 位置。SGLang 允许 floor 为 0——draft 已死的请求拿 0 个位置也不是饿死（bonus token
不在 confidence 矩阵里，该请求照样出 1 个 token，退化为普通自回归），省下的预算可以喂给
强请求的深位置，混合批下其可达 Θ 严格更优（test_starving_weak_rows_is_a_known_gap 钉死了
这个差异）。移植 floor=0 需要 executor 支持零窗口请求（layout/采样路径改动），是已知的
待决 gap 而非本实现的默认。

```python
candidates = survival[:, floor:].flatten()   # 每一个 (request, position) 对
# 按 survival 全局排序，取前 budget 个，然后按行计数
```

不需要任何显式的前缀约束：survival 沿位置非增，所以**任何全局 top-k 自动就是每个请求的
前缀**。因此分配逻辑只需要**数**每行中了几个，不需要追踪中的是哪几个。

tie 用 `(position, request)` 打破，**绝不用数值**。两个 TP rank 即使算出 bitwise 不同的
confidence，也必须选出相同的 verify 长度，否则它们的 batch 形状（进而 collective）会发散。

## 一个完整的数值例子

`K=7, bs=4, tiers=[1,3,7], min_verify_len=0`（注意：这是 SGLang 语义的演示例；本实现 floor 固定为 1），成本表
`token_counts=(0, 8, 16, 32)`、`step_time_ms=(2.0, 2.2, 2.6, 9.0)`。

survival 矩阵：

```text
        j=0    j=1    j=2    j=3    j=4    j=5    j=6
 r0 |  0.95   0.87   0.79   0.69   0.59   0.47   0.35   <- draft 很健康
 r1 |  0.60   0.18   0.02   0.00   0.00   0.00   0.00   <- 第 2 位就死了
 r2 |  0.90   0.81   0.73   0.66   0.59   0.53   0.48   <- 健康
 r3 |  0.70   0.49   0.34   0.24   0.17   0.12   0.08   <- 中等
 列和  3.15   2.35   1.88   1.59   1.35   1.12   0.91
```

uniform 决策：

| L | M = 4*(L+1) | T (ms) | tau = 4 + 列和 | Theta |
|---|-------------|--------|----------------|-------|
| 1 | 8 | 2.2 | 7.15 | 3.25 |
| **3** | **16** | **2.6** | **11.38** | **4.38 <- 最大** |
| 7 | 32 | 9.0 | 16.35 | 1.82 |

于是 `runtime_draft_len = 3`，每个请求验证 3 个 draft 位置。

ragged 用**同样的预算**（`4 * 3 = 12` 个位置）重新分配。floor 固定为 1（`min_verify_len >= 1`，每个请求至少验证 anchor 位置；SGLang 允许 floor 为 0、可把绝望请求饿到零窗口，这是两者的已知语义差异），floor 之上的位置参与全局竞争，排序后取预算内的前若干个：

```text
0.95(r0) 0.90(r2) 0.87(r0) 0.81(r2) 0.79(r0) 0.73(r2)
0.70(r3) 0.69(r0) 0.66(r2) 0.60(r1) 0.59(r0) 0.59(r2)
 \--------------- r0 拿 5 个，r2 拿 5 个，r3 和 r1 各 1 个 ---------------/
r3 的下一个候选 0.49、r1 的 0.18 都落在预算之外
```

| | verify_len（draft 位置数） | 期望产出 |
|---|---|---|
| r0 | **5** | 3.89 |
| r1 | **1** | 0.60 |
| r2 | **5** | 3.69 |
| r3 | **1** | 0.70 |
| **合计** | **12**（与 uniform 的 `4 * 3` 完全相同） | **tau = 12.88** |

```text
uniform L=3 :  12 个验证位置  ->  tau = 11.38
ragged      :  12 个验证位置  ->  tau = 12.88     （成本完全相同，+13%）
```

这就是 ragged 模式的全部动机：把 r1 和 r3 注定要浪费掉的位置，转移给 r0 和 r2。

## Ragged 验证

以 `verify_lens`（token 单位，含 bonus）`= [6, 2, 6, 2]` 为例：

```text
qo_indptr = [0, 6, 8, 14, 16]            # 独占前缀和

flat token 轴（attention kernel 想要的布局）:
 idx  0   1   2   3   4   5 | 6   7 | 8   9  10  11  12  13 | 14  15
      b  d0  d1  d2  d3  d4 | b  d0 | b  d0  d1  d2  d3  d4 |  b  d0
      \------- r0 ---------/ \- r1 -/ \-------- r2 ---------/ \- r3 -/
      ^                      ^        ^                       ^
      qo_indptr[0]=0         [1]=6    [2]=8                   [3]=14
```

### bucket 对齐

被捕获的 token bucket 是每个 tier 对应的 `bs * (t + 1)`。一个 batch 把自己的总数向上取整
到其中之一，`RaggedVerifyLayout.fill_bucket` 分两阶段花掉这部分余量：**先给真实请求**
（上限 `max_verify_len`，这样这些 token 去验证的是 step 本来就要付钱的真实 draft 位置），
剩下的才给 pad 行。

token 数必须**精确**命中 bucket。`seq_lens.sum()` 会变成 `attn_metadata.num_tokens`，
它会在 attention-DP 各 rank 间 all-gather，并驱动 MoE 的 chunk 数。如果少报，attention 和
MoE 对"在飞多少 token"的认知就不一致 —— **而且什么都不会报错**。

### 形状恒定，内容可变

这正是 ragged batch 能被捕获的原因：

```text
graph 捕获的是  ->  (padded_bs 行, bucket 个 token)      # 一小组固定形状
每步变化的是    ->  verify_lens / qo_indptr 里的数值      # 持久 buffer
```

## CUDA graph 处理

```text
uniform:   (bs, draft_len, False, short_seq_mode, is_all_greedy)
                 ^ tiers 中的某一个

ragged:    (bs, top_tier, False, short_seq_mode, is_all_greedy, bucket)
                                                                ^ token 轴
```

ragged 下 `draft_len` 钉死在最高 tier（不变量 I1：block 总是完整 draft），变化的是 token
总数。两种模式都捕获 `|batch_sizes| * |tiers|` 张图。

与"按 batch size 调度 draft 长度"不同，这里的 tier **不是** batch size 的函数 —— 在固定
`bs` 下，随着 batch 的 confidence 变化，调度器仍可能选不同的长度 —— 所以捕获集合是**叉积**。
每张捕获的图大约花 10-23 MB 元数据，而这块内存**直接从 KV cache 里出**（KV pool 是按 capture
之后剩下的量来定的）。所以 ladder 必须短。

capture 安全性依赖几个具体选择：

| 选择 | 原因 |
|------|------|
| `return_confidence` 在构造时读一次，绝不逐步判断 | 否则捕获的 draft graph 会发散 |
| STS buffer 原地更新，绝不 rebind | `nn.Module.__setattr__` 换 buffer 会换 `data_ptr`；先前捕获的 graph 会一直读旧存储，标定静默失效 |
| `_confidence_logits` 是持久 buffer，通过 slot 散射写入 | replay 时落到当前 batch 实际占用的槽位 |
| ragged buffer 通过持久分配的切片写入 | 保持 `data_ptr` 稳定，graph 才看得见新值 |
| `repeat_interleave` 一律传 `output_size` | 否则 torch 要把 repeats 的累积和读回 host 来确定输出尺寸 —— 在 capture 期非法，在其他地方则是每步一次同步 |

## 跨 rank 一致性

```text
draft_len 是 CUDA graph key 的一部分
        |
        v
但它（当时）不在 attention-DP 的一致性 allgather 里——这是促成现行 shape gate 的
历史事故：现在的实现已把 draft_len 纳入 allgather 并逐 rank 比对，下图保留为动机说明
        |
        v
两个 rank 选了不同长度 -> 选到不同的 graph
        |
        v
一个 replay、一个掉进 eager -> 它们的 collective 发散
```

归约取的是 **max 而不是 min**：想裁更多的 rank 只是多验几个 token，而 graph key 不一致
是**不可恢复**的。ragged 下捕获形状是二维的，一致性来自"对归约结果取整"，而不是"各 rank
对自己的数取整"：

```text
padded_bs = round_up_bs( max_r num_real_r )
bucket    = round_up_bucket( padded_bs + max_r slack_r )
```

## 退化路径

静默退化是这个特性的主要失效模式，所以每一条放弃裁剪的路径都在 `planner.stats` 里计数：

```text
                     +- planner 未构建 -----------------> 全块（无 checkpoint）
                     |
                     +- 成本表 flat（仅当只给了 sts 表时可达；两表
                     |   皆缺在构造时即被拒）------------> 全块 + 告警
                     |     "每个 token 都免费" => 裁剪等于盲猜
                     |
_dspark_confidence --+- 快照还没落地 -------------------> 全块
_draft_len()         |     （cudaEvent.query() == False）
                     |
                     +- 快照行数少于 batch --------------> 全块
                     |
                     +- 请求从未被打分 ------------------> 中性行 -> 全块
                     |     （刚 prefill 完，或 slot 刚回收）
                     |
                     +- 找不到已捕获的 ragged 形状 ------> 退回 uniform（uniform 一定有图）
```

所有路径都朝同一个方向退化：**验证整个 block**，即启用该特性之前的行为。输入不可信时，
调度器**永远不会**比静态基线验得更少。

部分结果被当作失败而非"部分成功"：如果 planner 无法给出恰好每请求一个 verify 长度，整批
退回 uniform。一个"半窗口"的 batch 比完全不加窗口更糟 —— 输入布局是逐请求构建的（会走成
ragged），而 spec metadata 看到缺失的窗口后会保持 uniform。

## 配置

```yaml
speculative_config:
  decoding_type: DSpark
  max_draft_len: 7

  # 主开关。block 仍然完整 draft，只改变送给 target 验证的 token 数。
  enable_confidence_scheduling: true

  # 逐位置 STS 标定。省略 => 裸 sigmoid（恒等标定）。
  # 调度器消费的是累积乘积，所以逐位置误差会沿 block 几何级放大。
  confidence_sts_path: /path/to/sts.json

  # profile 好的 step-cost 曲线。开启 ragged 时必填（除非提供 confidence_sts_path）：
  # 两者皆缺在构造时即报错 —— flat 成本模型下 planner 永远拒绝裁剪，跑完也是空跑。
  confidence_sps_table_path: /path/to/sps.json

  # 被捕获的长度 ladder；调度器只能从这个集合里选。
  # 默认：[1, ceil(K/2), K]
  confidence_verify_len_tiers: [1, 3, 7]

  # 逐请求 verify 窗口。必须与 enable_confidence_scheduling 同开 —— 两个 flag 描述同一个
  # 特性，只开其中一个会在构造时被拒绝；关闭特性 = 两个都设 false。
  enable_ragged_verify: true
```

`sps.json`：

```json
{
  "token_counts":      [0, 128, 256, 512],
  "step_time_ms":      [2.0, 2.1, 4.0, 9.0],
  "fixed_overhead_ms": 0.0,
  "batch_sizes":       [1, 16, 64, 256],
  "batch_overhead_ms": [3.0, 5.0, 17.0, 40.0]
}
```

`token_counts` 按**总验证 token 数**索引，**包含 bonus token**。`fixed_overhead_ms` 和
`batch_*` 这一对是可选的，但绝不是装饰 —— 见[算法](#算法)中的成本模型部分。

## 性能天花板与已知局限

```text
一个 decode step  =  bias  +  alpha(bs)  +  theta(M)
                     ------------------     --------
                     调度动不了这部分         唯一可回收的一项
```

因此收益是一个**高并发效应**。在小 batch 下，step 成本对验证 token 数几乎是平的，裁剪省
不下时间，却仍要付出接受长度的代价 —— 打平甚至为负。

**在目标硬件上量出可裁剪部分的占比是前提，不是事后工作。** 如果 `alpha(bs)` 压过
`theta(M)`，那么无论 confidence head 多好，天花板都很低。两个测量陷阱：

- 在**固定 batch size** 下 profile，会让 `M = bs * (L + 1)` 与 `L` 完全共线，`alpha` 和
  `theta` 分不开。必须扫 batch size。
- 如果 `d(step)/dK` 在各 batch size 上是平的，说明成本是 per-draft-pass 而非
  per-verify-token —— 调度器几乎没有可回收的东西。

其他局限：

| 局限 | 说明 |
|------|------|
| tier ladder 跨 batch size 有损 | shelf 右边缘随 `bs` 移动，一套 ladder 不可能同时坐在所有 batch size 的边缘上。残差要实测 |
| 快照是 2 步之前的 | 纯吞吐启发式（I2 保证正确性与它无关），但在剧烈抖动的负载下决策质量会下降 |
| ragged 模式的端到端数字仅在一个平台上取得 | DeepSeek-V4-Pro-DSpark，8×B300 DEP8，overlap + CUDA graph（GSM8K 96.44，基线 96.21）；仍默认关闭，跨硬件需重测 |
| 逐 token 采样参数路径只有非全贪心 batch 才会触发 | `temperature=0` 的精度跑覆盖不到它 |

## 部署要求：DSv4 必须显式设 `max_seq_len`（否则并发被钉在 ~maxbs/2）

DSv4 的窗口池组（SWA + compressor KV/score，窗口 = 128 + `max_draft_len`，DL=5 时 133
token）在 v2 KV manager 里是固定槽分配器。不显式设 `max_seq_len` 时，serve 会从模型推断
出 1M 并把它当 `typical_seq_len` 喂给比例定容——全注意力池吃掉全部配额，窗口池被饿到下限
`min_slots ≈ 1.05 × max_batch_size` 块（`kv_cache_manager_v2.py:1775-1779`），而下限按
history=0 建模成每请求 1 块（`:1803-1812`）。运行时窗口 133 > tokens_per_block 128，每个
生成请求实际锁 2（偶 3）块（`_life_cycle_registry.py:38-51`），于是可调度并发
= 1.05·maxbs / 2.07 ≈ **0.51 × max_batch_size**（实测：maxbs=128 → 61-63 行；maxbs=256
→ 131-136 行），伴随 ~2.7× 的步时惩罚。关闭 spec 时同一机制不降级而是**调度器死锁**
（`scheduler_v2.py:439` 处崩溃，报错信息有误导性）。

**修法：serving 配置里显式写 `max_seq_len`（按真实负载，比如 8k1k 场景写 9216）。**
`typical_seq_len` 随之落地，比例定容给窗口池的份额远高于下限，满 maxbs 可调度。判别实验
证明只动这一项即解钉；只调 KV fraction / max_num_tokens 均无效。历史工作绕法
`max_batch_size = 2 × 目标并发` 是同一算术的另一侧（把下限抬高一倍），在显式
`max_seq_len` 之后不再必要，但对 ctx 准入 headroom 仍有独立价值（见下）。

代码侧的根治（`typical_seq_len` 不应默认推断的 max_seq_len、下限按稳态 2 块建模、
no-spec 应降级而非死锁、字节估算器补 draft 窗口放大）待报 upstream，不阻塞部署规约。

## 基准配置（可复现）

完整实验记录与逐轮数据见 `docs/dspark_sts_program.md`。以下为两类负载的精确配置。

### 负载 A：poetry / arena（agg，burst-drain；收益 +5.9~+24.7% 的来源）

- 硬件/形态：DeepSeek-V4-Pro-DSpark，8×B300 单节点，`--tp_size 8 --ep_size 8`（DEP8，
  `enable_attention_dp: true`），`--max_batch_size 256 --max_num_tokens 4096`。
- 负载：poetry = 固定短 prompt（写诗，输出 ~760 tok/req）；arena = arena 真实问题集。
  客户端 burst-drain：一次性下发全部 N（512/1024）个请求，计时到全部完成，n=10 轮取中位。
  **注意：该协议下 99% 的步是纯 gen 步**（prefill 波在头几十步内结束）——这是收益成立的
  regime 前提。
- S 臂 spec 配置（`pro_sched_arm.yaml`）：tiers `[2,3,4,5]`，
  `confidence_sps_table_path: postfix_pro_table.json`（修复 #31/#32 后在 maxbs=256 满批
  重采：bs128 实测 θ(L2/3/4/5)=80.8/90.3/102.7/110.3ms），**未挂 STS**（裸 sigmoid）。
  N 臂同配置去掉两个 confidence 开关。kv 0.5，block reuse 关，graphs [1,8,32,64,128] 带
  padding，overlap on。

### 负载 B：throughput_1k（roofline 对齐负载，agg 持续饱和）

- 负载：throughput_1k parquet 前 1024 条（ISL≈1110），greedy，`max_tokens=64`；客户端
  持续饱和（固定在飞数，完成即补位，ramp 256 后计量 1024 个完成）。
- 关键物理：OSL=64 + 长 prompt + 持续补位 → **~50% 的步是 ctx+gen 混合步**，纯 gen 采的
  成本表在混合步高估节省（成本模型 ctx 盲），裁剪收益不成立（sched −1.2~−6.3%）。收益是
  **负载 regime 的函数**，不是普适常数。
- **把 gen bs 打到目标值的配方**（实测绑定约束是准入带宽：mnt=8192 时有准入的步 token 顶
  死预算,到达率 3.76/步 × 寿命 24.6 步 = 均衡 bs 92）：
  1. `max_num_tokens: 16384` —— 解除准入带宽上限；
  2. `max_batch_size = 2 × 目标` —— ctx 准入槽位 headroom；
  3. 客户端在飞 = 目标并发 —— 让 client 成为干净的绑定约束。
- 成本表与 STS 均须在本负载重采（表按 `--from-iter-log` 纯 gen 段拟合）。

### 负载 B'：disagg（gen-only 吞吐；裁剪的顺风 regime）

ctx/gen 分离后 gen server 每步纯 gen，成本表全程在域内。两台 B300 独立 slurm 作业,
NIXL 传输。**四个必踩的坑与解法**：

1. v2 KV manager 只支持 `transceiver_runtime: PYTHON` + `backend: NIXL`（C++/DEFAULT
   传输在启动时 `CUDA_ERROR_INVALID_CONTEXT`）。
2. **必须 `export UCX_CUDA_IPC_ENABLE_MNNVL=n`（两侧）**：v2 用
   `CU_MEM_HANDLE_TYPE_FABRIC` 分配 KV,无 NVLink 互联的独立作业间 UCX 会误判 cuda_ipc
   可达 → RDMA 卡死 ~336s → NIXL 元数据雪崩（全请求失败）。`ucx_perftest` 用普通
   cudaMalloc,对此 bug 是假阴性。NVL72 机架内（IMEX 就位）不需要。
3. 超时三层都要抬：`cache_transceiver_config.kv_transfer_timeout_ms: 600000`（默认 60s,
   超发排队下成批误杀）;router `trtllm-serve disaggregated --request_timeout 900`（默认
   180s）;（传输层由 2 解决）。
4. **ctx 与 gen 必须共用同一个 `speculative_config`**（YAML anchor）——否则窗口池几何不
   一致（128/8 vs 133/13）且 draft KV 无人 prefill。

```yaml
# ctx 与 gen 共同的骨架（两侧一致的部分）
max_seq_len: 8192            # 见"部署要求"
speculative_config: &spec
  decoding_type: DSpark
  max_draft_len: 5
  speculative_model: <model_path>
kv_cache_config: {enable_block_reuse: false, free_gpu_memory_fraction: 0.5}  # gen 侧 0.7
enable_attention_dp: true
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  max_tokens_in_buffer: 8192
  kv_transfer_timeout_ms: 600000
# ctx 侧另加 disable_overlap_scheduler: true;gen 侧在 *spec 上打开 confidence 开关。
```

## 参考

- DeepSeek DeepSpec：<https://github.com/deepseek-ai/DeepSpec>
- SGLang 的 DSpark 集成：<https://www.lmsys.org/blog/2026-07-06-dspark-sglang>
